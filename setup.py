import os
from setuptools import setup
from setuptools.command.build_ext import build_ext

class BuildPyGraphviz(build_ext):
    def finalize_options(self):
        import sys
        import pathlib

        candidates = [
            r"C:\Program Files\Graphviz",
            r"C:\Program Files (x86)\Graphviz",
        ]
        for base in candidates:
            inc, lib, bin = (
                pathlib.Path(base, "include"),
                pathlib.Path(base, "lib"),
                pathlib.Path(base, "bin"),
            )
            if inc.exists() and lib.exists():
                os.environ.setdefault("INCLUDE", str(inc))
                os.environ.setdefault("LIB", str(lib))
                os.environ["PATH"] = f"{bin};" + os.environ["PATH"]
                break

        super().finalize_options()

setup(
    cmdclass={"build_ext": BuildPyGraphviz},
)