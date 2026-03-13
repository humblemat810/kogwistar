from typing import Optional, Callable, Iterable, Tuple
import os
import pathlib
from json import JSONDecodeError


def nullable_concat(a: list | None, b: list | None) -> list | None:
    if a is None and b is None:
        return None
    return (a or []) + (b or [])


class RawFileLoader:
    def __init__(
        self,
        env_flist_path: Optional[str] = None,
        allow_file_list: None | list[str] = None,
        allow_file_list_ref: Optional[str] = None,
        max_num_file=float("inf"),
        oldest_datetime=None,
        newest_datetime=None,
        root_folder_name: str | None = None,
        #  in_folder_name :Optional[str] = None,
        walk_root: Optional[str] = None,
        compare_root: Optional[str] = None,
        include=None,
        bucket_blob_connection_str=None,
        file_walker_callback: Optional[
            Callable[[str, Optional[int]], Iterable[Tuple[str, str, str]]]
        ] = None,
        pattern=None,
        allow_startwith_relative_paths=False,
        filtering_callbacks: Optional[list[Callable]] = None,
    ):
        """file loader to either backward support for local folder loading behaviour, or cloud bucket/ blob stoages

        Args:
            env_var_name (_type_): environment variable that contains the path to the file list
            allow_file_list (None | list[str] | str): file path that contains the file information or a list of filenames directly passed into
            pattern re.compile returned pattern to apply on the path
            filtering_callbacks: that apply to each path, when True, it allows continue, when False, it continue to next file
        """

        assert not (
            (bucket_blob_connection_str is not None) and (env_flist_path is not None)
        )
        self.allow_startwith_relative_paths = allow_startwith_relative_paths
        self.pattern = pattern
        self.include = include or []
        self.oldest_datetime = oldest_datetime
        self.newest_datetime = newest_datetime
        self.bucket_blob_connection_str = bucket_blob_connection_str
        self.file_walker_callback: Optional[Callable] = file_walker_callback
        if root_folder_name is None:
            root_folder_name = os.getcwd()
        if walk_root is None:
            walk_root = os.getcwd()
        self.walk_root = walk_root
        if compare_root is None:
            compare_root = walk_root
        self.compare_root = compare_root
        self.max_num_file = max_num_file
        allowed_relative_paths: Optional[list[str]] = None

        if allow_file_list_ref is not None:
            if allow_file_list is None:
                allow_file_list = []
            if isinstance(allow_file_list_ref, str):
                need_attempt_readlines = False
                try:
                    if allow_file_list_ref.endswith(".json"):
                        import json

                        with open(allow_file_list_ref, "r") as f:
                            flist = json.load(f)
                            if isinstance(flist, list):
                                pass
                            else:
                                raise Exception("only list-like json is accepted")
                        allow_file_list = flist  # type: ignore
                    else:
                        need_attempt_readlines = True
                except JSONDecodeError:
                    need_attempt_readlines = True

                except Exception:
                    raise
                if need_attempt_readlines:
                    with open(allow_file_list_ref, "r") as f:
                        flist = f.readlines()

                    allow_file_list = flist  # type: ignore
            else:
                raise (
                    ValueError(
                        "allow_file_list does not either provide a path to file containing file list "
                        "or directly passing file list"
                    )
                )

        # allow combine flist from argument with env specifed paths list concatenated
        if env_flist_path is not None:
            if allowed_relative_paths is None:
                allowed_relative_paths = []
            if flist := os.environ.get(env_flist_path, None):
                if flist is not None:
                    if os.path.exists(flist):
                        if flist.endswith(".txt"):
                            with open(flist, "r", encoding="utf-8") as f:
                                allowed_relative_paths = [
                                    i.strip() for i in f.readlines()
                                ]

                        if flist.endswith(".csv"):
                            with open(flist, "r") as f:
                                for ln in f.readlines():
                                    allowed_relative_paths.append(
                                        os.path.join(
                                            *(i.strip() for i in ln.split(","))
                                        )
                                    )
                        # elif flist.endswith('.xls') or flist.endswith('.xlsx'):
                        #     from pandas import read_excel
                        #     df = read_excel(flist)
                        #     allowed_relative_paths = []
                        #     for i, row in df.iterrows():
                        #         new_path = os.path.join(*(row[:3]))
                        #         allowed_relative_paths.append(new_path)
        if allowed_relative_paths is not None:
            if allow_file_list is None:
                allow_file_list = []

            allow_file_list = allow_file_list + allowed_relative_paths
        self.allow_file_list = allow_file_list
        self.filtering_callbacks = filtering_callbacks or []
        self.resolve_paths = False
        if self.resolve_paths:
            self.check_allowed_relative_path()
        pass

    def check_allowed_relative_path(
        self,
        paths=None,
    ):
        # ensure allowed
        allow_file_list = nullable_concat(self.allow_file_list, paths)
        if allow_file_list is None:
            return
        out_file_list = []
        try:
            for p in allow_file_list:
                out_path = pathlib.Path(p).relative_to(self.compare_root)
                out_file_list.append(str(out_path))
        except ValueError as e:
            if "is not in the subpath of" in str(e):
                print("allow_file_list contains path outside of compare root")
                raise
        self.allow_file_list = out_file_list

    def __iter__(
        self,
        leaf_only=False,
        file_non_exist_ok=False,
        include=None,
        allowed_files: Optional[list[str]] = None,
        # allowed_prefixes : Optional[list[str | int]] = None,
        allowed_relative_paths: Optional[list[str]] = None,
    ):
        """Iterate through availble files

        Args:
            leaf_only (bool, optional): _description_. Defaults to True.
            file_non_exist_ok (bool, optional): _description_. Defaults to False.
            include (_type_, optional): _description_. Defaults to None.
            allowed_files (Optional[list[str]], optional): _description_. Defaults to None.
            allowed_relative_paths (Optional[list[str]], optional): _description_. Defaults to None.
            allow_startwith_relative_paths also check if the file start with any of the allowed relative paths
        Yields:
            _type_: _description_
        """
        from collections import Counter

        filter_stat = Counter()
        if include is None:
            include = include or self.include or set(["files"])  # , ['files', "dirs"]
        else:
            include = set(include)
            include.update(self.include)
        count = 0
        from datetime import datetime

        if self.oldest_datetime is not None:
            if type(self.oldest_datetime) is str:
                time_threshold_dt = datetime.strptime(
                    self.oldest_datetime, "%Y-%m-%d %H:%M"
                )
            else:
                time_threshold_dt = self.oldest_datetime
        else:
            time_threshold_dt = None
        if self.newest_datetime is not None:
            if type(self.newest_datetime) is str:
                time_threshold_upper_dt = datetime.strptime(
                    self.newest_datetime, "%Y-%m-%d %H:%M"
                )
            else:
                time_threshold_upper_dt = self.newest_datetime
        else:
            time_threshold_upper_dt = None
        nullable_allowed_relative_paths_set = []
        nullable_allowed_relative_paths_set_t = nullable_concat(
            allowed_relative_paths, self.allow_file_list
        )
        if nullable_allowed_relative_paths_set_t is not None:
            nullable_allowed_relative_paths_set_t = set(
                nullable_allowed_relative_paths_set_t
            )
            if len(nullable_allowed_relative_paths_set_t) > 0:
                nullable_allowed_relative_paths_set = list(
                    nullable_allowed_relative_paths_set_t
                )
        else:
            nullable_allowed_relative_paths_set = None
        # debug_list = []
        if self.file_walker_callback is None:
            if self.bucket_blob_connection_str is None:
                file_walker = os.walk(self.walk_root)
            else:
                raise NotImplementedError(
                    "blob or bucket stage file walker not implemented"
                )
        else:
            file_walker = self.file_walker_callback(self.walk_root)
        spec = None
        if nullable_allowed_relative_paths_set is not None:
            if self.allow_startwith_relative_paths:
                import pathspec

                prefixes = [
                    pathlib.Path(i).as_posix()
                    for i in nullable_allowed_relative_paths_set
                ]
                # normalized_prefixes = [p.rstrip("/\\") + "/" for p in raw_prefixes]
                spec = pathspec.PathSpec.from_lines("gitwildmatch", prefixes)
        for root, dirs, files in file_walker:
            # print(root)
            if leaf_only:
                if dirs != []:
                    filter_stat.update(["dir != []"])
                    continue
            if allowed_files:
                if pathlib.Path(root).parts[-1] in allowed_files:
                    pass
                else:
                    filter_stat.update(
                        ["pathlib.Path(root).parts[-1] in allowed_files"]
                    )
                    continue
            to_iter = []
            if "files" in include:
                to_iter += files
            if "dirs" in include:
                to_iter += dirs
            if "root" in include:
                to_iter += ["."]
            for f in to_iter:
                if self.pattern:
                    if self.pattern.match(os.path.join(root, f)):
                        pass
                    else:
                        filter_stat.update(
                            ["self.pattern.match(os.path.join(root, f))"]
                        )
                        continue
                f_rel = str(
                    pathlib.Path(os.path.join(root, f)).relative_to(self.compare_root)
                )
                if spec is not None:
                    matched = True
                    matched = spec.match_file(pathlib.Path(f_rel).as_posix())

                    if not matched:
                        filter_stat.update(
                            ["not spec.match_file(pathlib.Path(f_rel).as_posix())"]
                        )
                        continue
                else:
                    if nullable_allowed_relative_paths_set is None:
                        pass
                    elif f_rel in nullable_allowed_relative_paths_set:
                        pass
                    else:
                        filter_stat.update(["f_rel in allowed_relative_paths"])
                        continue
                filtered = False
                for cb in self.filtering_callbacks:
                    if cb(os.path.join(self.compare_root, f_rel)):
                        pass
                    else:
                        filter_stat.update([f"callback filtered by {cb}"])
                        filtered = True
                        break
                if filtered:
                    continue
                creation_timestamp = os.path.getctime(
                    os.path.join(self.compare_root, f_rel)
                )
                creation_time = datetime.fromtimestamp(creation_timestamp)
                if time_threshold_dt is not None:
                    if not (creation_time > time_threshold_dt):
                        filter_stat.update(["not (creation_time > time_threshold_dt)"])
                        continue
                if time_threshold_upper_dt is not None:
                    if not (creation_time < time_threshold_upper_dt):
                        filter_stat.update(
                            ["not (creation_time < time_threshold_upper_dt)"]
                        )
                        continue
                count += 1
                # debug_list.append(f_rel)
                yield f_rel
                if count >= self.max_num_file:
                    print(filter_stat.most_common())
                    return
        pass
        print(filter_stat.most_common())

    pass


def filter_folder(
    folder_root=os.path.join("..", "doc_data", "split_pages"),
    min_page=45,
    max_page: int | float = 55,
    first=10,
    verbose=True,
):
    folders = []
    for root, dirs, files in os.walk(folder_root):
        if dirs == []:
            n_page = len([i for i in files if i.lower().endswith(".pdf")])

            if n_page < max_page and min_page < n_page:
                rel_path = pathlib.Path(root).relative_to(folder_root)
                folders.append(str(rel_path))

    if verbose:
        print(folders)
    if first is not None:
        return sorted(folders)[:first]
    else:
        return sorted(folders)
