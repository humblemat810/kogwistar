# tests/conftest.py
from __future__ import annotations
import shutil
import hashlib
import uuid
import json
import os
import re
import asyncio
import pathlib
import tempfile
import sys
from _pytest.monkeypatch import MonkeyPatch
from typing import Any, cast
import dataclasses
from .auth_env import TEST_JWT_ALG, TEST_JWT_SECRET, ensure_test_jwt_env
from .net_helpers import pick_free_port

try:
    import sitecustomize  # type: ignore  # pragma: no cover
except Exception:  # pragma: no cover - local env may not provide it
    sitecustomize = None  # type: ignore
_TEST_ENV = MonkeyPatch()
_TEST_ENV.setenv("ANONYMIZED_TELEMETRY", "FALSE")
# Preserve any caller-provided JWT settings. This avoids clobbering server
# subprocess env when it imports tests.conftest indirectly through test helpers.
_TEST_ENV.setenv("JWT_SECRET", os.environ.get("JWT_SECRET", TEST_JWT_SECRET))
_TEST_ENV.setenv("JWT_ALG", os.environ.get("JWT_ALG", TEST_JWT_ALG))
try:
    import sqlalchemy as sa

    has_sa = True
except Exception:  # pragma: no cover - optional for non-pg test subsets
    sa = None  # type: ignore[assignment]
    has_sa = False


try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional in minimal CI environments
    def load_dotenv(*args, **kwargs):  # type: ignore[no-redef]
        return False

try:
    import _pytest.pathlib as _pytest_pathlib
    import _pytest.tmpdir as _pytest_tmpdir

    _cleanup_dead_symlinks = _pytest_pathlib.cleanup_dead_symlinks

    def _safe_cleanup_dead_symlinks(root):  # type: ignore[override]
        try:
            return _cleanup_dead_symlinks(root)
        except PermissionError:
            return None

    _pytest_pathlib.cleanup_dead_symlinks = _safe_cleanup_dead_symlinks
    _pytest_tmpdir.cleanup_dead_symlinks = _safe_cleanup_dead_symlinks
except Exception:  # pragma: no cover - optional hardening for Windows temp ACLs
    pass

_TEST_ROOT = pathlib.Path(__file__).resolve().parents[1]
for _env_name in (".env", ".env.test"):
    load_dotenv(_TEST_ROOT / _env_name, override=False)
import pytest
from typing import List, Optional, Sequence, Iterator, TYPE_CHECKING

if TYPE_CHECKING:
    import sqlalchemy as sa
    from testcontainers.postgres import PostgresContainer
try:
    from langchain_core.runnables import Runnable
except Exception:  # pragma: no cover - optional for langchain-dependent tests

    class Runnable:  # type: ignore
        pass

try:
    from kogwistar.conversation.models import (
        ConversationEdge,
        ConversationNode,
    )
    from kogwistar.conversation.service import ConversationService
    from kogwistar.engine_core.engine import GraphKnowledgeEngine
    from kogwistar.engine_core.models import (
        Edge,
        LLMGraphExtraction,
        LLMMergeAdjudication,
        AdjudicationVerdict,
        Node,
        Span,
        Grounding,
        MentionVerification,
    )
    from kogwistar.llm_tasks import LLMTaskSet
except Exception:  # pragma: no cover - allow lightweight test subsets on limited envs
    ConversationEdge = Any  # type: ignore[assignment]
    ConversationNode = Any  # type: ignore[assignment]
    ConversationService = Any  # type: ignore[assignment]
    GraphKnowledgeEngine = Any  # type: ignore[assignment]
    Edge = Any  # type: ignore[assignment]
    LLMGraphExtraction = Any  # type: ignore[assignment]
    LLMMergeAdjudication = Any  # type: ignore[assignment]
    AdjudicationVerdict = Any  # type: ignore[assignment]
    Node = Any  # type: ignore[assignment]
    Span = Any  # type: ignore[assignment]
    Grounding = Any  # type: ignore[assignment]
    MentionVerification = Any  # type: ignore[assignment]
    LLMTaskSet = Any  # type: ignore[assignment]

    def install_engine_hooks(*args, **kwargs):  # type: ignore[no-redef]
        return None

try:
    from kogwistar.engine_core.postgres_backend import PgVectorBackend
except Exception:  # pragma: no cover - optional in lightweight envs
    PgVectorBackend = Any  # type: ignore[assignment]

_TEST_NS = uuid.UUID("00000000-0000-0000-0000-000000000000")


@pytest.fixture(scope="session")
def stable_uuid(*parts: object) -> str:
    return str(uuid.uuid5(_TEST_NS, "|".join(str(p) for p in parts)))


class _SimpleTmpPathFactory:
    def __init__(self) -> None:
        base_root = _TEST_ROOT / ".tmp_pytest"
        base_root.mkdir(parents=True, exist_ok=True)
        self._base = pathlib.Path(
            tempfile.mkdtemp(prefix="kogwistar_pytest_", dir=str(base_root))
        )

    def mktemp(self, name: str, numbered: bool = True) -> pathlib.Path:
        prefix = f"{name}_" if numbered else name
        return pathlib.Path(
            tempfile.mkdtemp(prefix=prefix, dir=str(self._base))
        )

    def getbasetemp(self) -> pathlib.Path:
        return self._base


@pytest.fixture(scope="session")
def tmp_path_factory() -> _SimpleTmpPathFactory:
    factory = _SimpleTmpPathFactory()
    yield factory
    shutil.rmtree(factory.getbasetemp(), ignore_errors=True)


@pytest.fixture
def tmp_path(request: pytest.FixtureRequest, tmp_path_factory: _SimpleTmpPathFactory):
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", request.node.name.replace(os.sep, "_"))
    path = tmp_path_factory.mktemp(safe_name)
    request.addfinalizer(lambda: shutil.rmtree(path, ignore_errors=True))
    return path


from pathlib import Path

import logging

logging.captureWarnings(True)

from kogwistar.utils.log import EngineLogManager

logger = logging.getLogger(__name__)


def _run_async_windows_safe(coro):
    """Run an async coroutine in a loop compatible with psycopg on Windows."""
    if sys.platform == "win32":
        runner = asyncio.Runner(loop_factory=asyncio.SelectorEventLoop)
        try:
            return runner.run(coro)
        finally:
            runner.close()
    return asyncio.run(coro)


def _is_missing_pgvector_extension(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return (
        "extension \"vector\" is not available" in msg
        or "could not open extension control file" in msg
        or "vector.control" in msg
    )


@pytest.fixture(scope="session")
def event_loop_policy():
    """Use a selector loop on Windows so psycopg async connections work."""
    if sys.platform == "win32":
        return asyncio.WindowsSelectorEventLoopPolicy()
    return asyncio.DefaultEventLoopPolicy()


@pytest.fixture
def llm_provider_name(request) -> str:
    """Fixture for LLM provider name. Can be overriden via parameterization."""
    if hasattr(request, "param"):
        return request.param
    return request.config.getoption("--llm-provider")


@pytest.fixture
def llm_cache_dir(request) -> str:
    return request.config.getoption("--llm-cache-dir")


@pytest.fixture(scope="session")
def embedding_kind(request) -> str:
    """Default test embedding kind.

    Override per test with:

    ```python
    @pytest.mark.parametrize("embedding_kind", ["lexical_hash"], indirect=True)
    def test_something(conversation_engine):
        ...
    ```
    """

    if hasattr(request, "param"):
        return str(request.param)
    return str(request.config.getoption("--embedding-kind"))


@pytest.fixture(scope="session")
def embedding_dim(request) -> int:
    """Default embedding dimension for test embeddings."""

    if hasattr(request, "param"):
        return int(request.param)
    return int(request.config.getoption("--embedding-dim"))


@pytest.fixture(scope="function")
def embedding_function(embedding_kind: str, embedding_dim: int):
    """Embedding function chosen from the test config.

    Use ``constant`` for the smallest possible test footprint,
    ``lexical_hash`` for deterministic lexical similarity, or
    ``provider`` / ``real`` to let the engine resolve its default provider.
    """

    return build_test_embedding_function(embedding_kind, dim=embedding_dim)


def _to_stable_key(obj):
    """Recursively normalize objects for stable JSON serialization (hashing)."""
    if isinstance(obj, type) and hasattr(obj, "model_dump"):
        return obj.model_json_schema
    if hasattr(obj, "model_dump"):  # Pydantic v2
        return _to_stable_key(obj.model_dump(mode="json"))
    if dataclasses.is_dataclass(obj):
        return {k: _to_stable_key(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, (list, tuple)):
        return [_to_stable_key(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_stable_key(v) for k, v in obj.items()}
    if isinstance(obj, type):
        return f"{obj.__module__}.{obj.__name__}"
    return obj
class LLMCallTracker:
    def __init__(self):
        self.hits = []
        self.misses = []

    def reset(self):
        self.hits.clear()
        self.misses.clear()

    @property
    def total_calls(self):
        return len(self.hits) + len(self.misses)


@pytest.fixture(scope="session")
def llm_cache_tracker():
    return LLMCallTracker()

@pytest.fixture
def llm_tasks(llm_provider_name, llm_cache_dir, llm_cache_tracker) -> Iterator[LLMTaskSet]:
    """
    Function-scoped LLM task set with centralized joblib caching.
    Supports gemini, openai, and ollama.
    """
    from joblib import Memory
    from kogwistar.llm_tasks import (
        build_default_llm_tasks,
        DefaultTaskProviderConfig,
    )

    # Configure the base task set
    config = DefaultTaskProviderConfig(
        extract_graph_provider=llm_provider_name,
        adjudicate_pair_provider=llm_provider_name,
        adjudicate_batch_provider=llm_provider_name,
        filter_candidates_provider=llm_provider_name,
        summarize_context_provider=llm_provider_name,
        answer_with_citations_provider=llm_provider_name,
        repair_citations_provider=llm_provider_name,
    )
    
    # Special defaults for providers
    if llm_provider_name == "ollama":
        # Force qwen3:4b for ollama tests as per plan
        config = dataclasses.replace(config, ollama_model_name="qwen3:4b")
    
    base_tasks = build_default_llm_tasks(config)
    
    # Set up caching
    os.makedirs(llm_cache_dir, exist_ok=True)
    memory = Memory(location=llm_cache_dir, verbose=0)

    # We wrap the task set with caching. 
    # Since LLMTaskSet is a frozen dataclass, we use dataclasses.replace.


    from functools import wraps
    def _make_cached_task(task_fn):
        @memory.cache(ignore=["request"])
        def _actual_cached_call(stable_key: str, request: Any):
            return task_fn(request)
        @wraps(task_fn)
        def _wrapper(request):
            # We normalize and JSON-dump the request to create a stable key.
            # This handles Pydantic models (via model_dump) and types (via stringify).
            norm = _to_stable_key(request)
            key = json.dumps(norm, sort_keys=True, default=str)
            
            # Check if it's already in cache (best effort check to avoid double call)
            # joblib doesn't have a simple is_cached, so we'll just log before/after or use a wrapper.
            # Actually, joblib's call() will either run or use cache.
            # We can use memory.check_call_in_cache
            
            is_hit = False
            try:
                is_hit = _actual_cached_call.check_call_in_cache(key, request)
            except Exception:
                pass
            
            if is_hit:
                llm_cache_tracker.hits.append(task_fn.__name__)
                print(f"\n[LLM CACHE HIT] task={task_fn.__name__}")
            else:
                llm_cache_tracker.misses.append(task_fn.__name__)
                print(f"\n[LLM CACHE MISS] task={task_fn.__name__} key={key[:100]}...")
            
            # Debug logging to file for comparison
            debug_file = os.environ.get("GKE_LLM_CACHE_DEBUG_FILE")
            if debug_file:
                with open(debug_file, "a", encoding="utf-8") as f:
                    # Log as a single line for easy sorting/diffing
                    f.write(f"TASK:{task_fn.__name__} KEY:{key}\n")
                
            return _actual_cached_call(key, request)

        return _wrapper

    cached_tasks = dataclasses.replace(
        base_tasks,
        extract_graph=_make_cached_task(base_tasks.extract_graph),
        adjudicate_pair=_make_cached_task(base_tasks.adjudicate_pair),
        adjudicate_batch=_make_cached_task(base_tasks.adjudicate_batch),
        filter_candidates=_make_cached_task(base_tasks.filter_candidates),
        summarize_context=_make_cached_task(base_tasks.summarize_context),
        answer_with_citations=_make_cached_task(base_tasks.answer_with_citations),
        repair_citations=_make_cached_task(base_tasks.repair_citations),
    )
    
    yield cached_tasks



def _env_truthy(name: str) -> bool:
    value = os.getenv(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _configure_testcontainers_ryuk_env() -> bool:
    """Return whether Ryuk is effectively disabled for pg test containers."""
    if _env_truthy("GKE_TEST_PG_DISABLE_RYUK"):
        os.environ["TESTCONTAINERS_RYUK_DISABLED"] = "true"
        return True
    return _env_truthy("TESTCONTAINERS_RYUK_DISABLED")


def _purge_testcontainers_modules() -> None:
    for name in list(sys.modules):
        if name == "testcontainers" or name.startswith("testcontainers."):
            sys.modules.pop(name, None)


def _load_postgres_container_cls():
    _configure_testcontainers_ryuk_env()
    try:
        from testcontainers.postgres import PostgresContainer
    except Exception:  # pragma: no cover - optional for pg integration tests
        return None
    return PostgresContainer


def _is_ryuk_port_mapping_failure(exc: Exception) -> bool:
    text = str(exc).lower()
    return "port mapping" in text and "8080" in text and "not available" in text


def _start_postgres_container(image: str):
    postgres_container_cls = _load_postgres_container_cls()
    if postgres_container_cls is None:
        return None
    pg = postgres_container_cls(image)
    pg.start()
    return pg


def _stop_postgres_container_best_effort(pg, image: str, timeout_s: int = 10) -> None:
    import threading

    err: list[BaseException] = []

    def _stop() -> None:
        try:
            pg.stop()
        except BaseException as exc:  # pragma: no cover - teardown best effort
            err.append(exc)

    t = threading.Thread(target=_stop, name="pg-container-stop", daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        logger.warning(
            "Timed out stopping pg test container image=%s after %ss; leaving best-effort teardown",
            image,
            timeout_s,
        )
        return
    if err:
        logger.exception("Failed to stop pg test container image=%s", image, exc_info=err[0])


def pytest_addoption(parser):
    parser.addoption(
        "--run-manual",
        action="store_true",
        default=False,
        help="Run tests marked as manual.",
    )
    parser.addoption(
        "--llm-provider",
        action="store",
        default="gemini",
        choices=["gemini", "openai", "ollama"],
        help="LLM provider for real-model tests (default: gemini)",
    )
    parser.addoption(
        "--llm-cache-dir",
        action="store",
        default=".cache/tests/llm_results",
        help="Directory for caching LLM results",
    )
    parser.addoption(
        "--embedding-kind",
        action="store",
        default="constant",
        choices=["constant", "lexical_hash", "provider", "real"],
        help=(
            "Default embedding kind for test engines. "
            "Use lexical_hash to mirror the tutorial ladder embedder, "
            "or provider/real to let the engine resolve its default provider."
        ),
    )
    parser.addoption(
        "--embedding-dim",
        action="store",
        type=int,
        default=384,
        help="Default dimension used by test embedding factories.",
    )
    parser.addoption(
        "--backend-kind",
        action="store",
        default="chroma",
        choices=["fake", "chroma", "pg"],
        help=(
            "Default backend kind for test engines. "
            "Use fake for the in-memory backend, chroma for the current backend, "
            "or pg for pgvector."
        ),
    )


def _normalize_pytest_arg(value: str) -> str:
    return value.replace("\\", "/").lstrip("./")


def _is_specific_test_function_target(arg: str) -> bool:
    """
    True only for explicit function/method nodeids, e.g.:
      - tests/x.py::test_case
      - tests/x.py::TestClass::test_case
      - tests/x.py::test_case[param]
    False for file/class/folder targets.
    """
    if "::" not in arg:
        return False
    leaf = arg.rsplit("::", 1)[-1]
    leaf_base = leaf.split("[", 1)[0]
    return leaf_base.startswith("test_")


def _is_manual_test_explicitly_selected(
    config: pytest.Config, item: pytest.Item
) -> bool:
    nodeid = _normalize_pytest_arg(item.nodeid)
    cli_args = getattr(config.invocation_params, "args", ()) or ()

    for raw_arg in cli_args:
        if not isinstance(raw_arg, str):
            continue
        arg = _normalize_pytest_arg(raw_arg)
        if not arg or arg.startswith("-"):
            continue

        # Explicit function/method nodeid selection only.
        if _is_specific_test_function_target(arg):
            if (
                nodeid == arg
                or nodeid.startswith(arg + "[")
                or nodeid.rsplit("/", 1)[-1] in arg
                or pathlib.Path(arg).parts[-1] == nodeid.rsplit("/", 1)[-1]
            ):
                return True

    return False


_AREA_MARKERS = {"core", "workflow", "conversation", "runtime"}
_LIGHTWEIGHT_CI_MARKEXPR_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CORE_DIRS = {"cdc", "core", "ingestions", "outbox", "pg_sql", "primitives"}
_CONVERSATION_DIRS = {"kg_conversation", "mcp", "wisdom"}
_WORKFLOW_DIRS = {"runtime", "workflow"}
_RUNTIME_DIRS = {"runtime"}
_LIGHTWEIGHT_CI_DIR_BLOCKERS = {
    "cdc",
    "ingestions",
    "mcp",
    "outbox",
    "pg_sql",
    "runtime",
    "server",
    "wisdom",
    "workflow",
}
_LIGHTWEIGHT_CI_FILE_BLOCKERS = {
    "test_adjudication_merge.py",
    "test_adjudication_merge_positive.py",
    "test_aliasing.py",
    "test_batch_adjudication.py",
    "test_custom_embedder_injection.py",
    "test_document_rollback.py",
    "test_edge_endpoints_rollback.py",
    "test_embedder_default_and_verify.py",
    "test_embeddings_optional.py",
    "test_generate_cross_kind_candidates.py",
    "test_metadata_sanitization.py",
    "test_node_docs_rollback.py",
    "test_node_edge_ref.py",
    "test_engine_sqlite_uow_join.py",
    "test_shortids_smoke.py",
    "test_conversation_flow_v2_param_e2e.py",
    "test_conversation_flow.py",
    "test_verify_mentions.py",
}
_LIGHTWEIGHT_CI_IMPORT_BLOCKERS = (
    r"^\s*(?:from|import)\s+sqlalchemy\b",
    r"^\s*(?:from|import)\s+testcontainers\b",
    r"^\s*(?:from|import)\s+websocket\b",
    r"^\s*(?:from|import)\s+fastapi\b",
    r"^\s*(?:from|import)\s+mcp\b",
    "kogwistar.engine_core.postgres_backend",
    "kogwistar.engine_core.engine_postgres",
    "kogwistar.engine_core.engine_postgres_meta",
    "kogwistar.engine_core.chroma_backend",
    "kogwistar.server.auth.",
    "kogwistar.server.resources",
    "kogwistar.server.chat_api",
    "kogwistar.server.runtime_api",
    "kogwistar.server.auth_middleware",
    "kogwistar.server_mcp_with_admin",
    "kogwistar.cdc.change_bridge",
)


def _item_path_parts(item: pytest.Item) -> tuple[str, ...]:
    path = getattr(item, "path", None)
    if path is None:
        return ()
    try:
        return tuple(str(path).replace("\\", "/").split("/"))
    except Exception:
        return ()


def _has_marker(item: pytest.Item, names: set[str]) -> bool:
    return any(name in item.keywords for name in names)


def _is_lightweight_ci_run(config: pytest.Config) -> bool:
    markexpr = getattr(config.option, "markexpr", "") or ""
    if not markexpr:
        return False
    tokens = set(_LIGHTWEIGHT_CI_MARKEXPR_RE.findall(markexpr))
    return "ci" in tokens and "ci_full" not in tokens


def _path_contains_blocked_lightweight_import(collection_path: Path) -> bool:
    try:
        text = collection_path.read_text(encoding="utf-8")
    except Exception:
        return False

    for pattern in _LIGHTWEIGHT_CI_IMPORT_BLOCKERS:
        if pattern.startswith("^"):
            if re.search(pattern, text, re.MULTILINE):
                return True
        elif pattern in text:
            return True
    return False


def pytest_ignore_collect(**kwargs) -> bool:
    collection_path = kwargs.get("collection_path", kwargs.get("path"))
    config = kwargs.get("config")
    if collection_path is None or config is None:
        return False
    collection_path = pathlib.Path(str(collection_path))
    if any(part.startswith(".tmp") for part in collection_path.parts):
        return True
    if not _is_lightweight_ci_run(config):
        return False
    if collection_path.suffix != ".py":
        return False
    if "tests" not in collection_path.parts:
        return False
    if collection_path.name in _LIGHTWEIGHT_CI_FILE_BLOCKERS:
        return True
    if any(part in _LIGHTWEIGHT_CI_DIR_BLOCKERS for part in collection_path.parts):
        return True
    return _path_contains_blocked_lightweight_import(collection_path)


def _primary_test_dir(item: pytest.Item) -> str | None:
    parts = _item_path_parts(item)
    try:
        idx = parts.index("tests")
    except ValueError:
        return None
    if idx + 2 < len(parts):
        return parts[idx + 1]
    return None


def _infer_area_markers(item: pytest.Item) -> list[pytest.MarkDecorator]:
    parts = set(_item_path_parts(item))
    primary_dir = _primary_test_dir(item)
    markers: list[pytest.MarkDecorator] = []
    if parts & _WORKFLOW_DIRS and not _has_marker(item, {"workflow"}):
        markers.append(pytest.mark.workflow)
    if parts & _CONVERSATION_DIRS and not _has_marker(item, {"conversation"}):
        markers.append(pytest.mark.conversation)
    if parts & _RUNTIME_DIRS and not _has_marker(item, {"runtime"}):
        markers.append(pytest.mark.runtime)
    if not markers and (parts & _CORE_DIRS or primary_dir is None) and not _has_marker(
        item, {"core"}
    ):
        markers.append(pytest.mark.core)
    return markers


def _infer_runtime_submarkers(item: pytest.Item) -> list[pytest.MarkDecorator]:
    parts = set(_item_path_parts(item))
    if "runtime" not in parts:
        return []

    name = pathlib.Path(str(item.fspath)).name
    markers: list[pytest.MarkDecorator] = []
    if "async_runtime" in name and not _has_marker(item, {"runtime_async"}):
        markers.append(pytest.mark.runtime_async)
    if "sync_runtime" in name and not _has_marker(item, {"runtime_sync"}):
        markers.append(pytest.mark.runtime_sync)
    if "parity_bridge" in name and not _has_marker(item, {"runtime_bridge_parity"}):
        markers.append(pytest.mark.runtime_bridge_parity)
    return markers


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--run-manual"):
        return

    skip_manual = pytest.mark.skip(
        reason="manual test skipped by default; run with --run-manual or target a specific test function."
    )
    for item in items:
        for marker in _infer_area_markers(item):
            item.add_marker(marker)
        for marker in _infer_runtime_submarkers(item):
            item.add_marker(marker)
        if "manual" in item.keywords and not _is_manual_test_explicitly_selected(
            config, item
        ):
            item.add_marker(skip_manual)

    collected = []

    for item in items:
        entry = {
            "nodeid": item.nodeid,
            "name": item.name,
            "file": str(item.fspath),
            "params": {},
        }

        if hasattr(item, "callspec"):
            entry["params"] = item.callspec.params

        collected.append(entry)

    # store somewhere (file, graph, etc.)
    import json
    with open("collected_tests.json", "w") as f:
        json.dump(collected, f, indent=2, default=str)

def pytest_configure(config):
    temp_root = (
        pathlib.Path(tempfile.gettempdir())
        / f"kogwistar_pytest_tmp_{os.getpid()}_{uuid.uuid4().hex}"
    )
    temp_root.mkdir(parents=True, exist_ok=True)
    config.option.basetemp = str(temp_root)


@pytest.fixture(autouse=True)
def _per_test_engine_logging(request: pytest.FixtureRequest):
    worker_id = os.environ.get("PYTEST_XDIST_WORKER", "gw0")
    test_slug = re.sub(r"[^A-Za-z0-9._-]+", "_", request.node.name)[:80]
    test_hash = hashlib.sha1(request.node.nodeid.encode("utf-8")).hexdigest()[:12]
    base_dir = Path(".logs/test") / worker_id / f"{test_slug}_{test_hash}"

    EngineLogManager.reset()
    EngineLogManager.configure(
        base_dir=base_dir,
        app_name="gke_test",
        level=logging.DEBUG,
        enable_files=True,
        enable_sqlite=False,
        enable_jsonl=True,
    )
    try:
        yield
    finally:
        EngineLogManager.reset()


def pytest_unconfigure(config):
    _TEST_ENV.undo()


@pytest.fixture(scope="function")
def mcp_admin_server(tmp_path: Path) -> Iterator[dict[str, Any]]:
    """Start server_mcp_with_admin with isolated per-test data directories."""
    import subprocess
    import threading
    import time
    from collections import deque

    import requests

    host = "127.0.0.1"
    port = pick_free_port()
    base_http = f"http://{host}:{port}"
    base_mcp = f"{base_http}/mcp"

    data_root = tmp_path / "mcp_data"
    data_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["MCP_CHROMA_DIR"] = str(data_root / "docs")
    env["MCP_CHROMA_DIR_CONVERSATION"] = str(data_root / "conversation")
    env["MCP_CHROMA_DIR_WORKFLOW"] = str(data_root / "workflow")
    env["MCP_CHROMA_DIR_WISDOM"] = str(data_root / "wisdom")
    ensure_test_jwt_env(env)

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "kogwistar.server_mcp_with_admin:app",
        "--host",
        host,
        "--port",
        str(port),
        "--log-level",
        "warning",
    ]

    log_buf: deque[str] = deque(maxlen=400)
    proc = subprocess.Popen(
        cmd,
        cwd=str(Path(__file__).resolve().parents[1]),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    def _reader() -> None:
        if proc.stdout is None:
            return
        for line in proc.stdout:
            log_buf.append(line.rstrip())

    t = threading.Thread(target=_reader, daemon=True)
    t.start()

    deadline = time.time() + 90.0
    last_err: Exception | None = None
    while time.time() < deadline:
        if proc.poll() is not None:
            joined = "\n".join(log_buf)
            raise RuntimeError(
                f"MCP server exited before becoming healthy (exit={proc.returncode}).\n{joined}"
            )
        try:
            health = requests.get(f"{base_http}/health", timeout=1.5)
            if health.ok:
                break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
        time.sleep(0.2)
    else:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            proc.kill()
            proc.wait(timeout=5)
        if proc.stdout is not None:
            proc.stdout.close()
        joined = "\n".join(log_buf)
        raise RuntimeError(
            f"MCP server did not become healthy at {base_http}: {last_err}\n{joined}"
        )

    try:
        yield {
            "base_http": base_http,
            "base_mcp": base_mcp,
            "port": port,
            "data_root": str(data_root),
        }
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:  # noqa: BLE001
                proc.kill()
                proc.wait(timeout=5)
        if proc.stdout is not None:
            proc.stdout.close()


from tests._helpers.embeddings import build_test_embedding_function
from tests._helpers.engine_factories import (
    FakeEmbeddingFunction,
    FakeStructuredRunnable,
    _is_missing_pgvector_extension,
    _make_async_engine,
    _make_engine_pair,
    _make_workflow_engine,
)
try:
    from tests._helpers.fake_backend import build_fake_backend
except Exception:  # pragma: no cover - allow collection without the engine stack
    def build_fake_backend(*args, **kwargs):  # type: ignore[no-redef]
        pytest.importorskip("pydantic")
        raise RuntimeError("fake backend helper requires the engine stack")


class _SubprocessSemanticEmbeddingFunction:
    """Deterministic embedder for server subprocesses.

    It focuses on salient JSON/text fields instead of raw backend JSON noise so
    vector search sees enough signal to separate unrelated nodes/edges.
    """

    def __init__(self, dim: int = 384):
        self._dim = dim

    @staticmethod
    def name() -> str:
        return "default"

    def _field_text(self, payload: Any) -> str:
        parts: list[str] = []
        if isinstance(payload, dict):
            for key in ("label", "summary", "relation", "type", "doc_id"):
                val = payload.get(key)
                if val:
                    parts.append(str(val))
            props = payload.get("properties")
            if isinstance(props, dict):
                for key in ("signature_text", "name", "description"):
                    val = props.get(key)
                    if val:
                        parts.append(str(val))
            mentions = payload.get("mentions") or []
            for mention in mentions:
                spans = mention.get("spans") if isinstance(mention, dict) else None
                for span in spans or []:
                    if not isinstance(span, dict):
                        continue
                    for key in ("excerpt", "context_before", "context_after"):
                        val = span.get(key)
                        if val:
                            parts.append(str(val))
        return " ".join(parts).strip()

    def __call__(self, input):
        import hashlib
        import json as _json
        import math
        import re as _re

        vectors: list[list[float]] = []
        token_re = _re.compile(r"[a-z0-9_]+")
        stopwords = {
            "a",
            "an",
            "and",
            "as",
            "at",
            "by",
            "for",
            "from",
            "in",
            "is",
            "it",
            "of",
            "on",
            "or",
            "that",
            "the",
            "their",
            "these",
            "this",
            "to",
            "with",
        }
        for raw in input:
            text = str(raw or "")
            try:
                payload = _json.loads(text)
                if isinstance(payload, dict):
                    text = self._field_text(payload)
            except Exception:
                pass
            vec = [0.0] * self._dim
            tokens = token_re.findall(text.lower())
            for tok in tokens:
                if len(tok) < 3 or tok in stopwords:
                    continue
                # Two hashes per token: one index, one sign. Helps spread nearby
                # texts without collapsing everything into the same bucket.
                idx = int(hashlib.sha256(tok.encode("utf-8")).hexdigest()[:8], 16) % self._dim
                sign = -1.0 if int(hashlib.md5(tok.encode("utf-8")).hexdigest()[:2], 16) % 2 else 1.0
                vec[idx] += sign
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            vectors.append([x / norm for x in vec])
        return vectors


def build_default_test_embedding_function():
    dim = int(os.getenv("KOGWISTAR_TEST_EMBEDDING_DIM", "384"))
    kind = str(os.getenv("KOGWISTAR_TEST_EMBEDDING_KIND", "lexical_hash")).strip().lower()
    if kind in {"real", "provider", "default"}:
        try:
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

            return DefaultEmbeddingFunction()
        except Exception:
            pass
    if kind in {"lexical", "lexical_hash", "hash", "tutorial"}:
        from tests._helpers.embeddings import build_test_embedding_function

        return build_test_embedding_function("lexical_hash", dim=dim)
    return _SubprocessSemanticEmbeddingFunction(dim=dim)


_TEST_ENV.setenv(
    "KOGWISTAR_TEST_EMBEDDING_FUNCTION_IMPORT",
    "tests.conftest:build_default_test_embedding_function",
)


@pytest.fixture(scope="session")
def backend_kind(request) -> str:
    """Default backend kind for test engines.

    Use ``fake`` for an in-memory backend with no Chroma/PG dependency,
    ``chroma`` for the current vector-store backend, or ``pg`` for pgvector.
    Per-test override example:

    ```python
    @pytest.mark.parametrize("backend_kind", ["fake"], indirect=True)
    def test_smoke(conversation_engine):
        ...
    ```
    """

    if hasattr(request, "param"):
        return str(request.param)
    return str(request.config.getoption("--backend-kind"))


def _mk_span_from_excerpt(
    *,
    doc_id: str,
    content: str,
    excerpt: str,
    insertion_method: str,
    page_number: int = 1,
):
    idx = content.index(
        excerpt
    )  # will raise if excerpt not present -> good early failure
    start = idx
    end = idx + len(excerpt)
    return {
        "collection_page_url": "N/A",
        "document_page_url": "N/A",
        "doc_id": doc_id,
        "insertion_method": insertion_method,
        "page_number": page_number,
        "start_char": start,
        "end_char": end,
        "excerpt": excerpt,
        "context_before": content[max(0, start - 40) : start],
        "context_after": content[end : end + 40],
        # optional fields
        "chunk_id": None,
        "source_cluster_id": None,
        "verification": {
            "method": "heuristic",
            "is_verified": False,
            "score": None,
            "notes": "no explicit verification from LLM",
        },
    }


@pytest.fixture(scope="session")
def pg_container() -> Iterator[Optional["PostgresContainer"]]:
    """
    Spin up a disposable Postgres for the whole test session.

    Requirements:
      - Docker daemon running (Docker Desktop on Windows/macOS)
      - Python deps: testcontainers[postgresql], psycopg[binary], sqlalchemy
    """

    image = os.getenv("GKE_TEST_PG_IMAGE", "postgres:16")
    initial_ryuk_disabled = _configure_testcontainers_ryuk_env()
    logger.info(
        "Starting pg test container image=%s ryuk_disabled=%s",
        image,
        initial_ryuk_disabled,
    )

    pg = None
    try:
        pg = _start_postgres_container(image)
    except (
        Exception
    ) as exc:  # pragma: no cover - environment-dependent (docker availability)
        if (not initial_ryuk_disabled) and _is_ryuk_port_mapping_failure(exc):
            logger.warning(
                "Failed to start pg test container with Ryuk enabled; retrying once without Ryuk. image=%s err=%s",
                image,
                exc,
            )
            os.environ["TESTCONTAINERS_RYUK_DISABLED"] = "true"
            _purge_testcontainers_modules()
            try:
                pg = _start_postgres_container(image)
            except Exception as retry_exc:  # pragma: no cover - environment-dependent
                logger.warning(
                    "Retry without Ryuk also failed for pg test container image=%s: %s",
                    image,
                    retry_exc,
                )
                yield None
                return
        else:
            logger.warning("Failed to start pg test container image=%s: %s", image, exc)
            yield None
            return

    if pg is None:
        yield None
        return

    try:
        yield pg
    finally:
        if pg is None:
            return
        try:
            logger.info("Stopping pg test container image=%s", image)
            _stop_postgres_container_best_effort(pg, image)
        except Exception:
            logger.exception("Failed to stop pg test container image=%s", image)


@pytest.fixture(scope="session")
def pg_dsn(pg_container: Optional[PostgresContainer]) -> Optional[str]:
    """
    SQLAlchemy DSN for the running test container.
    """
    for env_name in ("GKE_PG_DSN", "PG_DSN", "DATABASE_URL"):
        env_value = os.getenv(env_name)
        if env_value:
            return env_value
    if pg_container is None:
        return None
    url = sa.engine.make_url(pg_container.get_connection_url())
    # Normalize to psycopg3 for both sync and async SQLAlchemy engines.
    # Testcontainers may return psycopg2 URLs by default.
    return url.set(drivername="postgresql+psycopg").render_as_string(
        hide_password=False
    )


@pytest.fixture(scope="session")
def sa_engine(pg_dsn: Optional[str]) -> Iterator[Any]:
    if (not has_sa) or pg_dsn is None:
        yield None
        return
    engine = sa.create_engine(pg_dsn, future=True)
    try:
        yield engine
    finally:
        try:
            engine.dispose()
        except Exception:
            logger.exception("Failed to dispose SQLAlchemy engine for pg test fixture")


@pytest.fixture(scope="session")
def async_sa_engine(pg_dsn: Optional[str]) -> Iterator[Any]:
    if (not has_sa) or pg_dsn is None:
        yield None
        return
    try:
        from sqlalchemy.ext.asyncio import create_async_engine
    except Exception:
        yield None
        return
    engine = create_async_engine(pg_dsn, future=True)
    try:
        yield engine
    finally:
        try:
            _run_async_windows_safe(engine.dispose())
        except Exception:
            logger.exception(
                "Failed to dispose async SQLAlchemy engine for pg test fixture"
            )


@pytest.fixture()
def pg_schema(sa_engine) -> Iterator[Optional[str]]:
    """
    Unique schema per test, dropped afterwards.

    Tests should pass this schema into PgVectorBackend(schema=...).
    """
    if sa_engine is None:
        yield None
        return
    schema = f"gke_test_{uuid.uuid4().hex}"
    with sa_engine.begin() as conn:
        conn.execute(sa.text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))
    try:
        yield schema
    finally:
        with sa_engine.begin() as conn:
            conn.execute(sa.text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))


@pytest.fixture()
def async_pg_schema(async_sa_engine) -> Iterator[Optional[str]]:
    """
    Unique schema per async test, dropped afterwards.
    """
    if async_sa_engine is None:
        yield None
        return
    schema = f"gke_async_test_{uuid.uuid4().hex}"

    async def _create() -> None:
        async with async_sa_engine.begin() as conn:
            await conn.execute(sa.text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))

    async def _drop() -> None:
        async with async_sa_engine.begin() as conn:
            await conn.execute(sa.text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))

    _run_async_windows_safe(_create())
    try:
        yield schema
    finally:
        _run_async_windows_safe(_drop())


@pytest.fixture()
def async_pg_backend(async_sa_engine, async_pg_schema):
    if async_sa_engine is None or async_pg_schema is None:
        pytest.skip("async pg backend requested but async pg fixtures are unavailable")
    from kogwistar.engine_core.postgres_backend import PgVectorBackend

    try:
        return PgVectorBackend(
            engine=async_sa_engine, embedding_dim=3, schema=async_pg_schema
        )
    except Exception as exc:
        if _is_missing_pgvector_extension(exc):
            pytest.skip(f"async pg backend unavailable: {exc}")
        raise


@pytest.fixture()
def async_pg_uow(async_sa_engine):
    if async_sa_engine is None:
        pytest.skip("async pg unit-of-work requested but async pg fixtures are unavailable")
    from kogwistar.engine_core.postgres_backend import AsyncPostgresUnitOfWork

    return AsyncPostgresUnitOfWork(engine=async_sa_engine)


class FakeLLMForAdjudication:
    """
    Test double for your LLM. Mimics `.with_structured_output(...)` by returning
    a Runnable that yields a fixed LLMMergeAdjudication.
    """

    def __init__(self, verdict, include_raw: bool = False):
        from kogwistar.engine_core.models import AdjudicationVerdict as AV

        self._verdict: AV = cast(AV, verdict)
        self._include_raw = include_raw

    def with_structured_output(
        self, schema, include_raw: bool = False, many: bool = False
    ):
        # Build a deterministic structured reply; ignore schema/many in this simple fake
        from kogwistar.engine_core.models import (
            LLMMergeAdjudication as LLMMA,
        )

        parsed = LLMMA(verdict=self._verdict)
        return FakeStructuredRunnable(
            parsed, include_raw=include_raw or self._include_raw
        )


class _FakeLLMForExtraction:
    """Mocks .with_structured_output(..., include_raw=True) â†’ .invoke(...) for extraction."""

    def with_structured_output(self, schema, include_raw=False, many=False):
        self._include_raw = include_raw
        self._schema = schema
        self._many = many
        return self

    def invoke(self, variables):
        # Deterministic graph from any document
        from kogwistar.engine_core.models import (
            LLMGraphExtraction as LLMGE,
        )
        from kogwistar.engine_core.models import (
            LLMNode as LLMN,
            LLMEdge as LLME,
        )

        parsed = LLMGE(
            nodes=[
                LLMN(
                    label="Photosynthesis",
                    type="entity",
                    summary="Process converting light to chemical energy",
                ),
                LLMN(
                    label="Chlorophyll",
                    type="entity",
                    summary="Molecule absorbing sunlight",
                ),
            ],
            edges=[
                LLME(
                    label="causes",
                    type="relationship",
                    summary="Chlorophyll absorption enables photosynthesis",
                    source_ids=["Chlorophyll"],  # will be mapped later in your pipeline
                    target_ids=["Photosynthesis"],
                    relation="enables",
                    mentions=Grounding(spans=[Span.from_dummy_for_document()]),
                )
            ],
        )
        if self._include_raw:
            return {"raw": "fake_raw", "parsed": parsed, "parsing_error": None}
        return parsed


class _FakeLLMForAdjudication:
    """Mocks .with_structured_output(LLMMergeAdjudication) for adjudication."""

    def with_structured_output(self, schema, include_raw=False, many=False):
        self._schema = schema
        self._include_raw = include_raw
        self._many = many
        return self

    def invoke(self, variables):
        # Always say "same entity" with high confidence for test simplicity
        ver = AdjudicationVerdict(
            same_entity=True,
            confidence=0.97,
            reason="Labels and summaries strongly match.",
            canonical_entity_id=str(uuid.uuid4()),
        )
        return LLMMergeAdjudication(verdict=ver)


class _CompositeFakeLLM:
    """Single fake that behaves for both extraction and adjudication chains."""

    def with_structured_output(self, schema, include_raw=False, many=False):
        # route by schema class name
        if getattr(schema, "__name__", "") == "LLMGraphExtraction":
            self._impl = _FakeLLMForExtraction()
        else:
            self._impl = _FakeLLMForAdjudication()
        return self._impl


@pytest.fixture(scope="function")
def tmp_chroma_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("chroma_db")
    yield str(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(scope="function")
def engine(request, tmp_chroma_dir, monkeypatch, backend_kind, embedding_function):
    if backend_kind == "fake":
        eng = GraphKnowledgeEngine(
            persist_directory=os.path.join(tmp_chroma_dir, "kg"),
            embedding_cache_path=os.path.join(os.getcwd(), ".embedding_cache"),
            embedding_function=embedding_function,
            backend_factory=build_fake_backend,
        )
    elif backend_kind == "pg":
        try:
            sa_engine = request.getfixturevalue("sa_engine")
            pg_schema = request.getfixturevalue("pg_schema")
        except Exception as exc:  # pragma: no cover - optional backend
            pytest.skip(f"pg backend requested but fixtures are unavailable: {exc}")
        eng = GraphKnowledgeEngine(
            persist_directory=os.path.join(tmp_chroma_dir, "kg"),
            embedding_cache_path=os.path.join(os.getcwd(), ".embedding_cache"),
            embedding_function=embedding_function,
            backend=PgVectorBackend(
                engine=sa_engine, embedding_dim=384, schema=f"{pg_schema}_kg"
            ),
        )
    else:
        try:
            eng = GraphKnowledgeEngine(
                persist_directory=os.path.join(tmp_chroma_dir, "kg"),
                embedding_cache_path=os.path.join(os.getcwd(), ".embedding_cache"),
                embedding_function=embedding_function,
            )
        except Exception as exc:
            if embedding_function is None:
                pytest.skip(f"real embedding provider unavailable: {exc}")
            raise
    # Patch the real LLM with a deterministic fake
    # eng.llm = _CompositeFakeLLM()
    return eng


@pytest.fixture(scope="function")
def tmp_conv_chroma_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("chroma_db")
    yield str(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(scope="function")
def conversation_engine(
    request, tmp_conv_chroma_dir, monkeypatch, backend_kind, embedding_function
):
    if backend_kind == "fake":
        eng = GraphKnowledgeEngine(
            persist_directory=os.path.join(tmp_conv_chroma_dir, "conversation"),
            kg_graph_type="conversation",
            embedding_function=embedding_function,
            backend_factory=build_fake_backend,
        )
    elif backend_kind == "pg":
        try:
            sa_engine = request.getfixturevalue("sa_engine")
            pg_schema = request.getfixturevalue("pg_schema")
        except Exception as exc:  # pragma: no cover - optional backend
            pytest.skip(f"pg backend requested but fixtures are unavailable: {exc}")
        eng = GraphKnowledgeEngine(
            persist_directory=os.path.join(tmp_conv_chroma_dir, "conversation"),
            kg_graph_type="conversation",
            embedding_function=embedding_function,
            backend=PgVectorBackend(
                engine=sa_engine, embedding_dim=384, schema=f"{pg_schema}_conv"
            ),
        )
    else:
        try:
            eng = GraphKnowledgeEngine(
                persist_directory=os.path.join(tmp_conv_chroma_dir, "conversation"),
                kg_graph_type="conversation",
                embedding_function=embedding_function,
            )
        except Exception as exc:
            if embedding_function is None:
                pytest.skip(f"real embedding provider unavailable: {exc}")
            raise
    # Patch the real LLM with a deterministic fake
    # eng.llm = _CompositeFakeLLM()
    return eng


@pytest.fixture(scope="function")
def workflow_engine(
    request, tmp_conv_chroma_dir, monkeypatch, backend_kind, embedding_function
):
    if backend_kind == "fake":
        eng = GraphKnowledgeEngine(
            persist_directory=os.path.join(tmp_conv_chroma_dir, "workflow"),
            kg_graph_type="workflow",
            embedding_function=embedding_function,
            backend_factory=build_fake_backend,
        )
    elif backend_kind == "pg":
        try:
            sa_engine = request.getfixturevalue("sa_engine")
            pg_schema = request.getfixturevalue("pg_schema")
        except Exception as exc:  # pragma: no cover - optional backend
            pytest.skip(f"pg backend requested but fixtures are unavailable: {exc}")
        eng = GraphKnowledgeEngine(
            persist_directory=os.path.join(tmp_conv_chroma_dir, "workflow"),
            kg_graph_type="workflow",
            embedding_function=embedding_function,
            backend=PgVectorBackend(
                engine=sa_engine, embedding_dim=384, schema=f"{pg_schema}_wf"
            ),
        )
    else:
        try:
            eng = GraphKnowledgeEngine(
                persist_directory=os.path.join(tmp_conv_chroma_dir, "workflow"),
                kg_graph_type="workflow",
                embedding_function=embedding_function,
            )
        except Exception as exc:
            if embedding_function is None:
                pytest.skip(f"real embedding provider unavailable: {exc}")
            raise
    # Patch the real LLM with a deterministic fake
    # eng.llm = _CompositeFakeLLM()
    return eng

