# tests/conftest.py
from __future__ import annotations
import shutil
import uuid
import json
import os
import re
import asyncio
import pathlib
import tempfile
from _pytest.monkeypatch import MonkeyPatch
from typing import Any, cast
import dataclasses

try:
    import sitecustomize  # type: ignore  # pragma: no cover
except Exception:  # pragma: no cover - local env may not provide it
    sitecustomize = None  # type: ignore
_TEST_ENV = MonkeyPatch()
_TEST_ENV.setenv("ANONYMIZED_TELEMETRY", "FALSE")
try:
    import sqlalchemy as sa

    has_sa = True
except Exception:  # pragma: no cover - optional for non-pg test subsets
    sa = None  # type: ignore[assignment]
    has_sa = False


import sys
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
    from kogwistar.conversation.policy import install_engine_hooks
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
        self._base = pathlib.Path(
            tempfile.mkdtemp(prefix="kogwistar_pytest_", dir=tempfile.gettempdir())
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
    path = tmp_path_factory.mktemp(request.node.name.replace(os.sep, "_"))
    request.addfinalizer(lambda: shutil.rmtree(path, ignore_errors=True))
    return path


from pathlib import Path

import logging

logging.captureWarnings(True)

import importlib.util
print("DEBUG sys.path =", sys.path)
print("DEBUG find_spec(kogwistar) =", importlib.util.find_spec("kogwistar"))
print("DEBUG find_spec(kogwistar.utils) =", importlib.util.find_spec("kogwistar.utils"))
print("DEBUG existing sys.modules['kogwistar'] =", sys.modules.get("kogwistar"))
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


def _install_conversation_policy(engine: GraphKnowledgeEngine) -> None:
    if getattr(engine, "kg_graph_type", None) == "conversation":
        install_engine_hooks(engine)

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


_AREA_MARKERS = {"core", "workflow", "conversation"}
_LIGHTWEIGHT_CI_MARKEXPR_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CORE_DIRS = {"cdc", "core", "ingestions", "outbox", "pg_sql", "primitives"}
_CONVERSATION_DIRS = {"kg_conversation", "mcp", "wisdom"}
_WORKFLOW_DIRS = {"runtime", "workflow"}
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
    if _has_marker(item, _AREA_MARKERS):
        return []

    parts = set(_item_path_parts(item))
    primary_dir = _primary_test_dir(item)
    markers: list[pytest.MarkDecorator] = []
    if parts & _WORKFLOW_DIRS:
        markers.append(pytest.mark.workflow)
    if parts & _CONVERSATION_DIRS:
        markers.append(pytest.mark.conversation)
    if not markers and (parts & _CORE_DIRS or primary_dir is None):
        markers.append(pytest.mark.core)
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
    EngineLogManager.configure(
        # EngineLogConfig(
        base_dir=Path(".logs/test"),
        app_name="gke_test",
        level=logging.DEBUG,
        enable_files=True,  # <-- ENABLE FILE LOGGING
        enable_sqlite=False,
        # mode="prod",              # <-- NOT pytest
        enable_jsonl=True,
        # )
    )


def pytest_unconfigure(config):
    _TEST_ENV.undo()


def _pick_free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="function")
def mcp_admin_server(tmp_path: Path) -> Iterator[dict[str, Any]]:
    """Start server_mcp_with_admin with isolated per-test data directories."""
    import subprocess
    import threading
    import time
    from collections import deque

    import requests

    host = "127.0.0.1"
    port = _pick_free_port()
    base_http = f"http://{host}:{port}"
    base_mcp = f"{base_http}/mcp"

    data_root = tmp_path / "mcp_data"
    data_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["MCP_CHROMA_DIR"] = str(data_root / "docs")
    env["MCP_CHROMA_DIR_CONVERSATION"] = str(data_root / "conversation")
    env["MCP_CHROMA_DIR_WORKFLOW"] = str(data_root / "workflow")
    env["MCP_CHROMA_DIR_WISDOM"] = str(data_root / "wisdom")
    env.setdefault("JWT_SECRET", "dev-secret")
    env.setdefault("JWT_ALG", "HS256")

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


from tests._helpers.embeddings import (
    ConstantEmbeddingFunction,
    build_test_embedding_function,
)
try:
    from tests._helpers.fake_backend import build_fake_backend
except Exception:  # pragma: no cover - allow collection without the engine stack
    def build_fake_backend(*args, **kwargs):  # type: ignore[no-redef]
        pytest.importorskip("pydantic")
        raise RuntimeError("fake backend helper requires the engine stack")


FakeEmbeddingFunction = ConstantEmbeddingFunction


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


def _make_engine_pair(
    *,
    backend_kind: str,
    tmp_path,
    sa_engine,
    pg_schema,
    dim: int = 3,
    use_fake: bool = False,
    embedding_kind: str | None = None,
    embedding_function: Any | None = None,
):
    """Build `(kg_engine, conv_engine)` for a chosen backend and embedding.

    Backend choices:
    - ``fake``: in-memory backend with Chroma-shaped collections
    - ``chroma``: current vector-store backend
    - ``pg``: PostgreSQL + pgvector backend

    Embedding choices:
    - ``constant``: minimal deterministic test embedding
    - ``lexical_hash``: tutorial-style lexical hash embedding
    - ``provider`` / ``real``: let the engine choose its configured provider

    Tests can also pass a concrete ``embedding_function`` instance directly.
    ``use_fake=True`` remains as a legacy alias for the constant embedder.
    """

    ef = (
        embedding_function
        if embedding_function is not None
        else build_test_embedding_function(
            embedding_kind or "constant", dim=dim
        )
    )
    if backend_kind == "fake":
        kg_engine = GraphKnowledgeEngine(
            persist_directory=str(tmp_path / "kg"),
            kg_graph_type="knowledge",
            embedding_function=ef,
            backend_factory=build_fake_backend,
        )
        conv_engine = GraphKnowledgeEngine(
            persist_directory=str(tmp_path / "conv"),
            kg_graph_type="conversation",
            embedding_function=ef,
            backend_factory=build_fake_backend,
        )
        _install_conversation_policy(conv_engine)
        return kg_engine, conv_engine

    if backend_kind == "chroma":
        try:
            kg_engine = GraphKnowledgeEngine(
                persist_directory=str(tmp_path / "kg"),
                kg_graph_type="knowledge",
                embedding_function=ef,
            )
            conv_engine = GraphKnowledgeEngine(
                persist_directory=str(tmp_path / "conv"),
                kg_graph_type="conversation",
                embedding_function=ef,
            )
        except Exception as exc:
            if embedding_function is None and str(embedding_kind or "").lower() in {
                "provider",
                "real",
            }:
                pytest.skip(f"real embedding provider unavailable: {exc}")
            raise
        _install_conversation_policy(conv_engine)
        return kg_engine, conv_engine

    if backend_kind == "pg":
        if sa_engine is None or pg_schema is None:
            pytest.skip(
                "pg backend requested but sa_engine/pg_schema fixtures not available"
            )
        kg_schema = f"{pg_schema}_kg"
        conv_schema = f"{pg_schema}_conv"
        kg_backend = PgVectorBackend(
            engine=sa_engine, embedding_dim=dim, schema=kg_schema
        )
        conv_backend = PgVectorBackend(
            engine=sa_engine, embedding_dim=dim, schema=conv_schema
        )
        kg_engine = GraphKnowledgeEngine(
            persist_directory=str(tmp_path / "kg_meta"),
            kg_graph_type="knowledge",
            embedding_function=ef,
            backend=kg_backend,
        )
        conv_engine = GraphKnowledgeEngine(
            persist_directory=str(tmp_path / "conv_meta"),
            kg_graph_type="conversation",
            embedding_function=ef,
            backend=conv_backend,
        )
        _install_conversation_policy(conv_engine)
        return kg_engine, conv_engine

    raise ValueError(f"unknown backend_kind: {backend_kind!r}")


def _make_async_engine(
    *,
    backend_kind: str,
    tmp_path,
    request: pytest.FixtureRequest,
    dim: int = 3,
    graph_kind: str = "knowledge",
    embedding_kind: str | None = None,
    embedding_function: Any | None = None,
):
    """Build a single GraphKnowledgeEngine against an async backend.

    The helper mirrors `_make_engine_pair(...)` but targets a single engine and
    uses the async PG / async Chroma fixtures.
    """

    ef = (
        embedding_function
        if embedding_function is not None
        else build_test_embedding_function(
            embedding_kind or "constant", dim=dim
        )
    )
    persist_dir = tmp_path / graph_kind

    if backend_kind == "pg":
        async_sa_engine = request.getfixturevalue("async_sa_engine")
        async_pg_schema = request.getfixturevalue("async_pg_schema")
        if async_sa_engine is None or async_pg_schema is None:
            pytest.skip(
                "async pg backend requested but async_pg fixtures are unavailable"
            )
        try:
            backend = PgVectorBackend(
                engine=async_sa_engine, embedding_dim=dim, schema=async_pg_schema
            )
        except Exception as exc:
            if _is_missing_pgvector_extension(exc):
                pytest.skip(f"async pg backend unavailable: {exc}")
            raise
        return GraphKnowledgeEngine(
            persist_directory=str(persist_dir),
            kg_graph_type=graph_kind,
            embedding_function=ef,
            backend=backend,
        )

    if backend_kind == "chroma":
        try:
            real_chroma_server = request.getfixturevalue("real_chroma_server")
        except Exception as exc:
            pytest.skip(f"real async chroma fixture is unavailable: {exc}")
        from tests.core._async_chroma_real import make_real_async_chroma_backend

        backend_client, backend, _collections = _run_async_windows_safe(
            make_real_async_chroma_backend(
                real_chroma_server,
                collection_prefix=f"{graph_kind}_{uuid.uuid4().hex}",
            )
        )
        _ = backend_client, _collections
        return GraphKnowledgeEngine(
            persist_directory=str(persist_dir),
            kg_graph_type=graph_kind,
            embedding_function=ef,
            backend_factory=lambda _engine, backend=backend: backend,
        )

    raise ValueError(f"unknown backend_kind: {backend_kind!r}")


def _make_workflow_engine(
    *,
    backend_kind: str,
    tmp_path,
    sa_engine,
    pg_schema,
    dim: int = 384,
    use_fake: bool = False,
    embedding_kind: str | None = None,
    embedding_function: Any | None = None,
) -> GraphKnowledgeEngine:
    """Build a workflow engine with an explicit test embedding strategy."""

    ef = (
        embedding_function
        if embedding_function is not None
        else build_test_embedding_function(
            embedding_kind or "constant", dim=dim
        )
    )
    if backend_kind == "fake":
        return GraphKnowledgeEngine(
            persist_directory=str(tmp_path / "wf"),
            kg_graph_type="workflow",
            embedding_function=ef,
            backend_factory=build_fake_backend,
        )
    if backend_kind == "chroma":
        try:
            return GraphKnowledgeEngine(
                persist_directory=str(tmp_path / "wf"),
                kg_graph_type="workflow",
                embedding_function=ef,
            )
        except Exception as exc:
            if embedding_function is None and str(embedding_kind or "").lower() in {
                "provider",
                "real",
            }:
                pytest.skip(f"real embedding provider unavailable: {exc}")
            raise
    if backend_kind == "pg":
        if sa_engine is None or pg_schema is None:
            pytest.skip(
                "pg backend requested but sa_engine/pg_schema fixtures not available"
            )
        wf_schema = f"{pg_schema}_wf"
        wf_backend = PgVectorBackend(
            engine=sa_engine, embedding_dim=dim, schema=wf_schema
        )
        return GraphKnowledgeEngine(
            persist_directory=str(tmp_path / "wf_meta"),
            kg_graph_type="workflow",
            embedding_function=ef,
            backend=wf_backend,
        )
    raise ValueError(f"unknown backend_kind: {backend_kind!r}")


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


class FakeStructuredRunnable(Runnable):
    """A minimal Runnable that returns a fixed structured result."""

    def __init__(self, parsed: Any, include_raw: bool = False):
        self._parsed = parsed
        self._include_raw = include_raw

    # sync single
    def invoke(self, input: Any, config: Optional[dict] = None, **kwargs: Any) -> Any:
        if self._include_raw:
            return {"raw": None, "parsed": self._parsed, "parsing_error": None}
        return self._parsed

    # async single
    async def ainvoke(
        self, input: Any, config: Optional[dict] = None, **kwargs: Any
    ) -> Any:
        return self.invoke(input, config=config, **kwargs)

    # sync batch
    def batch(
        self, inputs: List[Any], config: Optional[dict] = None, **kwargs: Any
    ) -> List[Any]:
        return [self.invoke(i, config=config, **kwargs) for i in inputs]

    # async batch
    async def abatch(
        self, inputs: List[Any], config: Optional[dict] = None, **kwargs: Any
    ) -> List[Any]:
        return [self.invoke(i, config=config, **kwargs) for i in inputs]


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
            pg.stop()
        except Exception:
            logger.exception("Failed to stop pg test container image=%s", image)


@pytest.fixture(scope="session")
def pg_dsn(pg_container: Optional[PostgresContainer]) -> Optional[str]:
    """
    SQLAlchemy DSN for the running test container.
    """
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
    """Mocks .with_structured_output(..., include_raw=True) → .invoke(...) for extraction."""

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
    _install_conversation_policy(eng)
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


@pytest.fixture()
def real_small_graph(tmp_path: Path):
    e = GraphKnowledgeEngine(persist_directory=str(tmp_path / "small_graph"))
    doc_id = "D1"

    # nodes
    def add_node(nid, label):
        n = Node(
            id=nid,
            label=label,
            type="entity",
            summary=label,
            mentions=[
                Grounding(
                    spans=[
                        Span(
                            collection_page_url=f"document_collection/{doc_id}",
                            document_page_url=f"document/{doc_id}",
                            doc_id=doc_id,
                            insertion_method="pytest-conftext-fixture",
                            page_number=1,
                            start_char=0,
                            end_char=1,
                            excerpt="x",
                            context_before="",
                            context_after="",
                            chunk_id=None,
                            source_cluster_id=None,
                            verification=MentionVerification(
                                method="heuristic",
                                is_verified=False,
                                notes="fixture",
                                score=0.9,
                            ),
                        )
                    ]
                )
            ],
            doc_id=doc_id,
        )
        e.node_collection.add(
            ids=[nid],
            documents=[n.model_dump_json(field_mode="backend")],
            metadatas=[{"doc_id": doc_id, "label": n.label, "type": n.type}],
        )
        # node_docs link
        ndid = f"{nid}::{doc_id}"
        row = {"id": ndid, "node_id": nid, "doc_id": doc_id}
        e.node_docs_collection.add(
            ids=[ndid], documents=[json.dumps(row)], metadatas=[row]
        )
        return n

    A = add_node("A", "Smoking")
    B = add_node("B", "Lung Cancer")
    C = add_node("C", "Cough")

    # edge A -[causes]-> B
    e_id = "E1"
    edge = Edge(
        id=e_id,
        label="Smoking causes Lung Cancer",
        type="relationship",
        summary="causal",
        relation="causes",
        source_ids=["A"],
        target_ids=["B"],
        source_edge_ids=[],
        target_edge_ids=[],
        mentions=A.mentions,
        doc_id=doc_id,
    )
    e.edge_collection.add(
        ids=[e_id],
        documents=[edge.model_dump_json(field_mode="backend")],
        metadatas=[{"doc_id": doc_id, "relation": "causes"}],
    )
    # endpoints fan-out
    rows = [
        {
            "id": f"{e_id}::src::node::A",
            "edge_id": e_id,
            "endpoint_id": "A",
            "endpoint_type": "node",
            "role": "src",
            "relation": "causes",
            "doc_id": doc_id,
        },
        {
            "id": f"{e_id}::tgt::node::B",
            "edge_id": e_id,
            "endpoint_id": "B",
            "endpoint_type": "node",
            "role": "tgt",
            "relation": "causes",
            "doc_id": doc_id,
        },
    ]
    e.edge_endpoints_collection.add(
        ids=[r["id"] for r in rows],
        documents=[json.dumps(r) for r in rows],
        metadatas=rows,
    )

    # final summary link S -> docnode:D1
    S = add_node("S", "Final Summary")
    e_id2 = "E2"
    e.edge_collection.add(
        ids=[e_id2],
        documents=[
            Edge(
                id=e_id2,
                label="summarizes_document",
                type="relationship",
                summary="S summarizes document",
                relation="summarizes_document",
                source_ids=["S"],
                target_ids=[f"docnode:{doc_id}"],
                source_edge_ids=[],
                target_edge_ids=[],
                mentions=S.mentions,
                doc_id=doc_id,
            ).model_dump_json(field_mode="backend")
        ],
        metadatas=[{"doc_id": doc_id, "relation": "summarizes_document"}],
    )
    rows2 = [
        {
            "id": f"{e_id2}::src::node::S",
            "edge_id": e_id2,
            "endpoint_id": "S",
            "endpoint_type": "node",
            "role": "src",
            "relation": "summarizes_document",
            "doc_id": doc_id,
        },
        {
            "id": f"{e_id2}::tgt::node::docnode:{doc_id}",
            "edge_id": e_id2,
            "endpoint_id": f"docnode:{doc_id}",
            "endpoint_type": "node",
            "role": "tgt",
            "relation": "summarizes_document",
            "doc_id": doc_id,
        },
    ]
    e.edge_endpoints_collection.add(
        ids=[r["id"] for r in rows2],
        documents=[json.dumps(r) for r in rows2],
        metadatas=rows2,
    )

    return e, doc_id


@pytest.fixture()
def small_test_docs_nodes_edge_adjudcate():
    """
    sample llm extracted tripples emulating data from llm graph extraction with insertion_method = "llm_graph_extraction"
    """
    # ---------- 0) Documents (exact strings you provided) ----------
    docs = {
        "DOC_A": (
            "In plant biology, many processes sustain life. "
            "Photosynthesis is a process used by plants to convert light energy into chemical energy. "
            "Chlorophyll is the molecule that absorbs sunlight. "
            "Within the leaves, specialized cells organize the chloroplasts. "
            "Plants perform photosynthesis in their leaves. "
            "Inside each chloroplast, stacks of thylakoid membranes called grana host the light-dependent reactions, "
            "where photons drive the formation of ATP and NADPH. "
            "These energy carriers then power the Calvin cycle in the stroma, fixing atmospheric CO2 into sugars using the enzyme Rubisco. "
            "Leaf anatomy supports this workflow: palisade mesophyll concentrates chloroplasts for maximal light capture, "
            "while spongy mesophyll and intercellular spaces facilitate gas diffusion. "
            "Stomata on the epidermis balance CO2 uptake with water conservation, opening and closing in response to light, humidity, and internal signals. "
            "Environmental factors—light intensity, temperature, CO2 concentration, and water availability—modulate overall photosynthetic rate. "
            "Different strategies evolved to mitigate photorespiration and arid stress: C3 plants fix carbon directly via Rubisco, "
            "C4 plants use a spatial separation with PEP carboxylase in bundle sheath cells, and CAM plants temporally separate uptake and fixation. "
            "Through these coordinated structures and pathways, photosynthesis underpins plant growth and, ultimately, most food webs."
        ),
        "DOC_B": (
            "Botanists distinguish several pigments in leaves. "
            "In most plants, chlorophyll a and chlorophyll b are present. "
            "The pigment chlorophyll gives leaves their green color. "
            "Both variants aid in harvesting light across different wavelengths. "
            "Chlorophyll a typically absorbs strongly in the blue-violet and red regions, while chlorophyll b extends coverage further into blue and a slightly different red band, "
            "broadening the effective spectrum plants can use. "
            "Accessory pigments such as carotenoids protect photosystems and capture light that chlorophylls miss, funneling energy into reaction centers by resonance transfer. "
            "The relative abundance of these pigments varies with species, developmental stage, and light environment—shade leaves often adjust pigment ratios to maximize efficiency. "
            "Seasonal changes alter pigment visibility: as chlorophyll degrades in autumn, carotenoids and, in some species, anthocyanins become more apparent, shifting leaf color. "
            "At the microscopic level, pigments are organized within protein complexes (photosystems I and II) embedded in thylakoid membranes, "
            "where the arrangement optimizes energy capture, minimizes photodamage, and supports electron transport."
        ),
        "DOC_C": (
            "Across the literature, the pigment chlorophyll is essential for photosynthesis, especially in terrestrial plants and algae. "
            "This role has been confirmed by numerous experiments. "
            "Classic action spectra align the rate of photosynthesis with wavelengths that chlorophyll absorbs, and historic demonstrations—such as Engelmann’s—linked oxygen evolution to those bands. "
            "Mutational studies that reduce chlorophyll content depress photosynthetic performance, while stressors that damage chlorophyll or its binding proteins impair growth. "
            "Comparative work shows parallel solutions in other phototrophs: bacteriochlorophylls in anoxygenic bacteria illustrate how pigment chemistry adapts to distinct ecological niches. "
            "Modern techniques, including chlorophyll fluorescence measurements and satellite indices like NDVI, exploit chlorophyll’s optical signatures to assess plant health and productivity at scales from leaves to landscapes. "
            "Biotechnological approaches aim to tune pigment composition and antenna size to minimize energy losses under high light, increase carbon gain, and enhance crop yields. "
            "Taken together, diverse lines of evidence establish chlorophyll as the core photochemical hub that initiates and regulates the flow of solar energy into biosynthetic pathways."
        ),
        "DOC_D": (
            "In human physiology, hemoglobin plays a central role. "
            "Hemoglobin absorbs oxygen in red blood cells and transports it to tissues. "
            "Binding is mediated by iron within the heme groups. "
            "Structurally, adult hemoglobin (HbA) is a tetramer whose subunits exhibit cooperative binding, creating a sigmoidal oxygen dissociation curve that suits loading in the lungs and unloading in tissues. "
            "Physiological modulators—including pH and CO2 (the Bohr effect), temperature, and 2,3-bisphosphoglycerate—shift hemoglobin’s affinity to match metabolic demand. "
            "Fetal hemoglobin (HbF) binds oxygen more tightly, facilitating transfer across the placenta, while variants such as sickle hemoglobin alter red-cell properties and clinical outcomes. "
            "Heme iron (Fe2+) reversibly binds O2 but can be blocked by carbon monoxide, which competes strongly at the same site; methemoglobin formation (Fe3+) reduces oxygen-carrying capacity. "
            "Beyond oxygen transport, hemoglobin contributes to CO2 carriage and acid–base buffering. "
            "When erythrocytes are recycled, heme is catabolized to bilirubin and iron is conserved—a tightly regulated process reflecting hemoglobin’s systemic importance."
        ),
    }

    # ---- helpers: find spans & build ReferenceSession-like dicts ----
    def _find_span(doc_id: str, excerpt: str) -> tuple[int, int]:
        text = docs[doc_id]
        idx = text.find(excerpt)
        if idx < 0:
            raise AssertionError(f"Exact excerpt not found in {doc_id}: {excerpt!r}")
        return idx, idx + len(excerpt)

    def _ref(doc_id: str, excerpt: str, method: str = "llm"):
        s, e = _find_span(doc_id, excerpt)
        # Span-like dict compatible with kogwistar.models.Span
        # (kept as a dict because this fixture emulates a JSON payload).
        return {
            "doc_id": doc_id,
            "collection_page_url": "N/A",
            "document_page_url": "N/A",
            "page_number": 1,
            "start_char": s,
            "end_char": e,
            "excerpt": docs[doc_id][s:e],
            "context_before": docs[doc_id][max(0, s - 40) : s],
            "context_after": docs[doc_id][e : min(len(docs[doc_id]), e + 40)],
            # Optional verification payload
            "verification": {
                "method": method,
                "is_verified": True,
                "score": 1.0,
                "notes": "fixture",
            },
            "insertion_method": "llm_graph_extraction",
        }

    # ---------- Nodes ----------
    nodes = [
        {
            "id": "N_CHLORO",
            "label": "Chlorophyll",
            "type": "entity",
            "summary": "A green pigment found in plants.",
            "mentions": [
                {
                    "spans": [
                        _ref(
                            "DOC_A",
                            "Chlorophyll is the molecule that absorbs sunlight.",
                        ),
                        _ref("DOC_C", "chlorophyll is essential for photosynthesis"),
                    ]
                }
            ],
        },
        {
            "id": "N_CHLORO_ALIAS",
            "label": "Chlorophyll (pigment)",
            "type": "entity",
            "summary": "Alias name for the same pigment.",
            "mentions": [
                {
                    "spans": [
                        _ref("DOC_C", "the pigment chlorophyll is essential"),
                    ]
                }
            ],
        },
        {
            "id": "N_CHLORO_A",
            "label": "Chlorophyll a",
            "type": "entity",
            "summary": "One variant of chlorophyll.",
            "mentions": [
                {
                    "spans": [
                        _ref("DOC_B", "chlorophyll a and chlorophyll b are present."),
                    ]
                }
            ],
        },
        {
            "id": "N_PHOTOSYN",
            "label": "Photosynthesis",
            "type": "entity",
            "summary": "Converts light energy to chemical energy.",
            "mentions": [
                {
                    "spans": [
                        _ref(
                            "DOC_A",
                            "Photosynthesis is a process used by plants to convert light energy",
                        ),
                        _ref(
                            "DOC_C",
                            "rate of photosynthesis with wavelengths that chlorophyll absorbs",
                        ),
                    ]
                }
            ],
        },
        {
            "id": "N_LEAVES",
            "label": "Leaves",
            "type": "entity",
            "summary": "Plant organs that host photosynthesis.",
            "mentions": [
                {
                    "spans": [
                        _ref("DOC_A", "Plants perform photosynthesis in their leaves."),
                        _ref(
                            "DOC_B",
                            "The pigment chlorophyll gives leaves their green color.",
                        ),
                    ]
                }
            ],
        },
        {
            "id": "N_SUN",
            "label": "Sunlight",
            "type": "entity",
            "summary": "Incoming solar radiation.",
            "mentions": [
                {
                    "spans": [
                        _ref("DOC_A", "light energy into chemical energy"),
                    ]
                }
            ],
        },
        {
            "id": "N_HEMO",
            "label": "Hemoglobin",
            "type": "entity",
            "summary": "Oxygen transport protein in blood.",
            "mentions": [
                {
                    "spans": [
                        _ref(
                            "DOC_D",
                            "Hemoglobin absorbs oxygen in red blood cells and transports it to tissues.",
                        ),
                    ]
                }
            ],
        },
        {
            "id": "N_OXY",
            "label": "Oxygen",
            "type": "entity",
            "summary": "O₂ molecule transported in blood.",
            "mentions": [
                {
                    "spans": [
                        _ref(
                            "DOC_D",
                            "Hemoglobin absorbs oxygen in red blood cells and transports it to tissues.",
                        ),
                    ]
                }
            ],
        },
        # Reified relation as a node (for cross-type positive)
        {
            "id": "N_PHOTO_REIFIED",
            "label": "Photosynthesis occurs in Leaves",
            "type": "entity",
            "summary": "Photosynthesis occurs in Leaves",  # Reified relation concept.
            "properties": {"signature_text": "occurs_in(Photosynthesis, Leaves)"},
            "mentions": [
                {
                    "spans": [
                        _ref("DOC_A", "Plants perform photosynthesis in their leaves."),
                    ]
                }
            ],
        },
    ]

    # ---------- Edges ----------
    edges = [
        {
            "id": "E_CHLORO_ABSORB",
            "label": "Chlorophyll absorbs sunlight",
            "type": "relationship",
            "summary": "Chlorophyll absorbs sunlight.",
            "relation": "absorbs",
            "source_ids": ["N_CHLORO"],
            "target_ids": ["N_SUN"],
            "source_edge_ids": [],
            "target_edge_ids": [],
            "mentions": [
                {
                    "spans": [
                        _ref(
                            "DOC_A",
                            "Chlorophyll is the molecule that absorbs sunlight.",
                        ),
                    ]
                }
            ],
            "properties": {"signature_text": "absorbs(Chlorophyll, Sunlight)"},
        },
        {
            "id": "E_PHOTO_LEAVES",
            "label": "Photosynthesis occurs in leaves",
            "type": "relationship",
            "summary": "Photosynthesis happens in leaves.",
            "relation": "occurs_in",
            "source_ids": ["N_PHOTOSYN"],
            "target_ids": ["N_LEAVES"],
            "source_edge_ids": [],
            "target_edge_ids": [],
            "mentions": [
                {
                    "spans": [
                        _ref("DOC_A", "Plants perform photosynthesis in their leaves."),
                    ]
                }
            ],
            "properties": {"signature_text": "occurs_in(Photosynthesis, Leaves)"},
        },
        {
            "id": "E_PHOTO_LEAVES_DUP",
            "label": "Photosynthesis in leaves (duplicate)",
            "type": "relationship",
            "summary": "Duplicate of Photosynthesis→Leaves relation.",
            "relation": "occurs_in",
            "source_ids": ["N_PHOTOSYN"],
            "target_ids": ["N_LEAVES"],
            "source_edge_ids": [],
            "target_edge_ids": [],
            "mentions": [
                {
                    "spans": [
                        _ref("DOC_A", "Plants perform photosynthesis in their leaves."),
                    ]
                }
            ],
            "properties": {"signature_text": "occurs_in(Photosynthesis, Leaves)"},
        },
        {
            "id": "E_HEMO_TRANSPORT",
            "label": "Hemoglobin transports oxygen",
            "type": "relationship",
            "summary": "Hemoglobin transports oxygen in blood.",
            "relation": "transports",
            "source_ids": ["N_HEMO"],
            "target_ids": ["N_OXY"],
            "source_edge_ids": [],
            "target_edge_ids": [],
            "mentions": [
                {
                    "spans": [
                        _ref(
                            "DOC_D",
                            "Hemoglobin absorbs oxygen in red blood cells and transports it to tissues.",
                        ),
                    ]
                }
            ],
            "properties": {"signature_text": "transports(Hemoglobin, Oxygen)"},
        },
    ]

    # ---------- Ground-truth pairs for tests ----------
    adjudication_pairs = {
        # Node ↔ Node (2 positive, 2 negative typical)
        "positive": [
            ["N_CHLORO", "N_CHLORO_ALIAS"],  # alias
            [
                "N_PHOTO_REIFIED",
                "E_PHOTO_LEAVES",
            ],  # cross-type positive (see below, also listed under cross)
        ],
        "negative": [
            ["N_CHLORO", "N_HEMO"],  # different domains
            ["N_CHLORO_A", "N_CHLORO_ALIAS"],  # related but not same
        ],
        # Edge ↔ Edge
        "edge_positive": [
            ["E_PHOTO_LEAVES", "E_PHOTO_LEAVES_DUP"],  # duplicate semantics
        ],
        "edge_negative": [
            ["E_CHLORO_ABSORB", "E_HEMO_TRANSPORT"],  # different relation/entities
        ],
        # Cross-type (node ↔ edge)
        "cross_positive": [
            ["N_PHOTO_REIFIED", "E_PHOTO_LEAVES"],  # reified node vs relation
        ],
        "cross_negative": [
            ["N_HEMO", "E_CHLORO_ABSORB"],  # unrelated
        ],
    }

    # Pre-flight validate the JSON-ish payload against the current models.
    # This keeps the fixture from silently drifting out-of-date with models.py.
    try:
        _ = LLMGraphExtraction.FromLLMSlice(
            {"nodes": nodes, "edges": edges}, insertion_method="fixture_sample"
        )
    except Exception as e:
        raise AssertionError(
            f"Fixture small_test_docs_nodes_edge_adjudcate is not model-compatible: {e}"
        )

    sample_dataset = {
        "docs": docs,
        "nodes": nodes,
        "edges": edges,
        "adjudication_pairs": adjudication_pairs,
    }

    return sample_dataset


def mk_verification(
    *,
    method: str = "human",
    is_verified: bool = True,
    score: float = 1.0,
    notes: str = "seed",
) -> MentionVerification:
    return MentionVerification(
        method=method,
        is_verified=is_verified,
        score=score,
        notes=notes,
    )


def mk_span(
    *,
    doc_id: str,
    full_text: str,
    start_char: int = 0,
    end_char: Optional[int] = None,
    page_number: int = 1,
    insertion_method: str = "seed",
    collection_page_url: str = "url",
    document_page_url: str = "url",
    context_before: str = "",
    context_after: str = "",
    chunk_id: Optional[str] = None,
    source_cluster_id: Optional[str] = None,
    verification: Optional[MentionVerification] = None,
) -> Span:
    if end_char is None:
        end_char = len(full_text)
    excerpt = full_text[start_char:end_char]
    return Span(
        collection_page_url=collection_page_url,
        document_page_url=document_page_url,
        doc_id=doc_id,
        insertion_method=insertion_method,
        page_number=page_number,
        start_char=start_char,
        end_char=end_char,
        excerpt=excerpt,
        context_before=context_before,
        context_after=context_after,
        chunk_id=chunk_id,
        source_cluster_id=source_cluster_id,
        verification=verification or mk_verification(notes=f"seed:{insertion_method}"),
    )


def mk_grounding(*spans: Span) -> Grounding:
    return Grounding(spans=list(spans))


def add_node_raw(
    engine: GraphKnowledgeEngine,
    node: Node | ConversationNode,
    *,
    embedding_dim: int = 384,
    embedding: Optional[Sequence[float]] = None,
) -> None:
    """
    Adds a node by directly writing to the underlying Chroma collection,
    using the engine's own serialization helper.

    This mirrors what your existing test already does:
      doc, meta = engine._node_doc_and_meta(n)
      engine.node_collection.add(...)
    """
    doc, meta = engine._node_doc_and_meta(node)
    if embedding is None and getattr(node, "embedding", None) is None:
        embedding = [0.1] * embedding_dim
    if getattr(node, "embedding", None) is None:
        node.embedding = embedding  # type: ignore

    engine.node_collection.add(
        ids=[node.id],
        documents=[doc],
        embeddings=[list(node.embedding)],  # type: ignore[arg-type]
        metadatas=[meta],
    )


def add_edge_raw(
    engine: Any,
    edge: Edge | ConversationEdge,
    *,
    embedding_dim: int = 384,
    embedding: Optional[Sequence[float]] = None,
) -> None:
    """
    Same idea for edges.
    """
    doc, meta = engine._edge_doc_and_meta(edge)
    if embedding is None and getattr(edge, "embedding", None) is None:
        embedding = [0.1] * embedding_dim
    if getattr(edge, "embedding", None) is None:
        edge.embedding = embedding  # type: ignore

    engine.edge_collection.add(
        ids=[edge.id],
        documents=[doc],
        embeddings=[list(edge.embedding)],  # type: ignore[arg-type]
        metadatas=[meta],
    )


# ---------------------------------------------------------------------
# Seed KG graph (real Node/Edge objects with proper mentions/spans)
# ---------------------------------------------------------------------


def seed_kg_graph(
    *, kg_engine: GraphKnowledgeEngine, kg_doc_id: str = "D_KG_001"
) -> dict[str, Any]:
    """
    Seeds a minimal KG doc with:
      - N1, N2 nodes
      - E1 edge N1 -> N2
    Returns ids for later linking from conversation graph.
    """
    text1 = "Project KGE stores entities and relations with provenance spans."
    text2 = "Conversation graph nodes can reference KG nodes/edges for grounding."

    n1 = Node(
        id="KG_N1",
        label="KGE provenance",
        type="entity",
        summary="KGE stores entities/relations with spans for provenance.",
        mentions=[
            mk_grounding(
                mk_span(
                    doc_id=kg_doc_id,
                    full_text=text1,
                    insertion_method="seed_kg_node",
                    document_page_url=f"doc/{kg_doc_id}#KG_N1",
                    collection_page_url=f"collection/{kg_doc_id}",
                )
            )
        ],
        metadata={"level_from_root": 0, "entity_type": "kg_entity"},
        domain_id=None,
        canonical_entity_id=None,
        properties={"kind": "kg_node"},
        embedding=None,
        doc_id=kg_doc_id,
        level_from_root=0,
    )

    n2 = Node(
        id="KG_N2",
        label="Conversation grounding",
        type="entity",
        summary="Conversation graph nodes can reference KG items for grounding.",
        mentions=[
            mk_grounding(
                mk_span(
                    doc_id=kg_doc_id,
                    full_text=text2,
                    insertion_method="seed_kg_node",
                    document_page_url=f"doc/{kg_doc_id}#KG_N2",
                    collection_page_url=f"collection/{kg_doc_id}",
                )
            )
        ],
        metadata={"level_from_root": 0, "entity_type": "kg_entity"},
        domain_id=None,
        canonical_entity_id=None,
        properties={"kind": "kg_node"},
        embedding=None,
        doc_id=kg_doc_id,
        level_from_root=0,
    )

    e1 = Edge(
        id="KG_E1",
        label="supports",
        type="relationship",
        summary="Provenance spans support conversation grounding.",
        source_ids=[n1.id],
        target_ids=[n2.id],
        relation="supports",
        source_edge_ids=[],
        target_edge_ids=[],
        mentions=[
            mk_grounding(
                mk_span(
                    doc_id=kg_doc_id,
                    full_text="supports",
                    insertion_method="seed_kg_edge",
                    document_page_url=f"doc/{kg_doc_id}#KG_E1",
                    collection_page_url=f"collection/{kg_doc_id}",
                    start_char=0,
                    end_char=8,
                )
            )
        ],
        metadata={"level_from_root": 0, "entity_type": "kg_edge"},
        domain_id=None,
        canonical_entity_id=None,
        properties={"kind": "kg_edge"},
        embedding=None,
        doc_id=kg_doc_id,
    )

    # Insert (bypass ingestion)
    add_node_raw(kg_engine, n1)
    add_node_raw(kg_engine, n2)
    add_edge_raw(kg_engine, e1)

    return {
        "doc_id": kg_doc_id,
        "node_ids": (n1.id, n2.id),
        "edge_ids": (e1.id,),
        "n1": n1,
        "n2": n2,
        "e1": e1,
    }


# ---------------------------------------------------------------------
# Seed Conversation graph with refs to KG
# ---------------------------------------------------------------------


def seed_conversation_graph(
    *,
    conversation_engine: GraphKnowledgeEngine,
    user_id: str = "U_TEST",
    conversation_id: str = "CONV_TEST_001",
    start_node_id: str = "CONV_START_001",
    kg_seed: dict[str, Any],
) -> dict[str, Any]:
    """
    Seeds:
      - conversation start (via engine.create_conversation)
      - two turns (user + assistant)
      - next_turn edge between turns
      - memory_context node
      - summary node
      - kg_ref node referencing kg_seed[n1/e1]
    """
    # Create conversation start through the service facade.
    conv_svc = ConversationService.from_engine(
        conversation_engine,
        knowledge_engine=conversation_engine,
    )
    conv_id, start_id = conv_svc.create_conversation(
        user_id,
        conversation_id,
        start_node_id,
    )
    assert conv_id == conversation_id
    assert start_id == start_node_id

    # Turn 0 (user)
    t0_text = "Show me what happened in the graph engine."
    t0_id = "TURN_000"
    t0_span = mk_span(
        doc_id=f"conv:{conv_id}",
        full_text=t0_text,
        insertion_method="conversation_turn",
        collection_page_url=f"conversation/{conv_id}",
        document_page_url=f"conversation/{conv_id}#{t0_id}",
        page_number=1,
    )
    turn0 = ConversationNode(
        user_id=user_id,
        id=t0_id,
        label="Turn 0 (user)",
        type="entity",
        doc_id=t0_id,
        summary=t0_text,
        role="user",  # type: ignore
        turn_index=0,
        conversation_id=conv_id,
        mentions=[mk_grounding(t0_span)],
        properties={},
        metadata={
            "entity_type": "conversation_turn",
            "level_from_root": 0,
            "in_conversation_chain": True,
        },
        domain_id=None,
        canonical_entity_id=None,
        embedding=None,
        level_from_root=0,
    )
    conversation_engine.write.add_node(turn0, None)

    # Turn 1 (assistant)
    t1_text = "Here are the relevant KG nodes and the conversation timeline."
    t1_id = "TURN_001"
    t1_span = mk_span(
        doc_id=f"conv:{conv_id}",
        full_text=t1_text,
        insertion_method="conversation_turn",
        collection_page_url=f"conversation/{conv_id}",
        document_page_url=f"conversation/{conv_id}#{t1_id}",
        page_number=1,
    )
    turn1 = ConversationNode(
        user_id=user_id,
        id=t1_id,
        label="Turn 1 (assistant)",
        type="entity",
        doc_id=t1_id,
        summary=t1_text,
        role="assistant",  # type: ignore
        turn_index=1,
        conversation_id=conv_id,
        mentions=[mk_grounding(t1_span)],
        properties={},
        metadata={
            "entity_type": "conversation_turn",
            "level_from_root": 0,
            "in_conversation_chain": True,
        },
        domain_id=None,
        canonical_entity_id=None,
        embedding=None,
        level_from_root=0,
    )
    conversation_engine.write.add_node(turn1, None)

    # next_turn edge (TURN_000 -> TURN_001)
    next_edge = ConversationEdge(
        id="EDGE_NEXT_000_001",
        source_ids=[turn0.safe_get_id()],
        target_ids=[turn1.safe_get_id()],
        relation="next_turn",
        label="next_turn",
        type="relationship",
        summary="Sequential flow",
        doc_id=f"conv:{conv_id}",
        mentions=[mk_grounding(t1_span)],
        metadata={"causal_type": "chain"},
        domain_id=None,
        canonical_entity_id=None,
        properties={"entity_type": "conversation_edge"},
        embedding=None,
        source_edge_ids=[],
        target_edge_ids=[],
    )
    conversation_engine.write.add_edge(next_edge)

    # memory_context node (references memory nodes/edges if you want; keep empty here but schema-valid)
    memctx_id = "MEMCTX_001"
    memctx_text = "Active memory context: user wants graph debugging view."
    memctx_span = mk_span(
        doc_id=f"conv:{conv_id}",
        full_text=memctx_text,
        insertion_method="memory_context",
        collection_page_url=f"conversation/{conv_id}",
        document_page_url=f"conversation/{conv_id}#{memctx_id}",
    )
    memctx = ConversationNode(
        user_id=user_id,
        id=memctx_id,
        label="Memory context (turn 1)",
        type="entity",
        doc_id=memctx_id,
        summary=memctx_text,
        role="system",  # type: ignore
        turn_index=1,
        conversation_id=conv_id,
        mentions=[mk_grounding(memctx_span)],
        properties={
            "user_id": user_id,
            "source_memory_nodes_ids": [],
            "source_memory_edges_ids": [],
        },
        metadata={
            "entity_type": "memory_context",
            "level_from_root": 0,
            "in_conversation_chain": False,
        },
        domain_id=None,
        canonical_entity_id=None,
        embedding=None,
        level_from_root=0,
    )
    conversation_engine.write.add_node(memctx, None)

    # summary node (system)
    summ_id = "SUMM_001"
    summ_text = "Summary: user asks to inspect graph flow; assistant will show KG + conversation links."
    summ_span = mk_span(
        doc_id=f"conv:{conv_id}",
        full_text=summ_text,
        insertion_method="conversation_summary",
        collection_page_url=f"conversation/{conv_id}",
        document_page_url=f"conversation/{conv_id}#{summ_id}",
    )
    summ = ConversationNode(
        user_id=user_id,
        id=summ_id,
        label="Summary 0-1",
        type="entity",
        doc_id=summ_id,
        summary=summ_text,  # type: ignore
        role="system",  # type: ignore
        turn_index=1,
        conversation_id=conv_id,
        mentions=[mk_grounding(summ_span)],
        properties={"content": summ_text},
        metadata={
            "entity_type": "conversation_summary",
            "level_from_root": 1,
            "in_conversation_chain": True,
        },
        domain_id=None,
        canonical_entity_id=None,
        embedding=None,
        level_from_root=1,
    )
    conversation_engine.write.add_node(summ, None)

    # summarizes edge (summary -> turns)
    summ_edge = ConversationEdge(
        id="EDGE_SUMM_001",
        source_ids=[summ.safe_get_id()],
        target_ids=[turn0.safe_get_id(), turn1.safe_get_id()],
        relation="summarizes",
        label="summarizes",
        type="relationship",
        summary="Memory summarization",
        doc_id=f"conv:{conv_id}",
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
        embedding=None,
        mentions=[mk_grounding(summ_span)],
        metadata={"causal_type": "summary"},
        source_edge_ids=[],
        target_edge_ids=[],
    )
    conversation_engine.write.add_edge(summ_edge)

    # kg_ref node: points to KG node/edge (this is what your dump tool should bridge)
    kg_ref_id = "KGREF_001"
    kg_ref_text = f"KG ref: node={kg_seed['node_ids'][0]} edge={kg_seed['edge_ids'][0]}"
    kg_ref_span = mk_span(
        doc_id=f"conv:{conv_id}",
        full_text=kg_ref_text,
        insertion_method="kg_ref",
        collection_page_url=f"conversation/{conv_id}",
        document_page_url=f"conversation/{conv_id}#{kg_ref_id}",
    )
    kg_ref_node = ConversationNode(
        user_id=user_id,
        id=kg_ref_id,
        label="KG reference",
        type="entity",
        doc_id=kg_ref_id,
        summary=kg_ref_text,
        role="system",  # type: ignore
        turn_index=1,
        conversation_id=conv_id,
        mentions=[mk_grounding(kg_ref_span)],
        properties={
            # critical: dump uses these to build hyperlinks / cross-bundle paths
            "ref_kind": "kg",
            "ref_doc_id": kg_seed["doc_id"],
            "ref_node_ids": list(kg_seed["node_ids"]),
            "ref_edge_ids": list(kg_seed["edge_ids"]),
        },
        metadata={
            "entity_type": "kg_ref",
            "level_from_root": 0,
            "in_conversation_chain": False,
        },
        domain_id=None,
        canonical_entity_id=None,
        embedding=None,
        level_from_root=0,
    )
    conversation_engine.write.add_node(kg_ref_node, None)

    # edge: turn1 -> kg_ref (optional but useful for viz)
    ref_edge = ConversationEdge(
        id="EDGE_TURN1_KGREF",
        source_ids=[turn1.safe_get_id()],
        target_ids=[kg_ref_node.safe_get_id()],
        relation="mentions_kg",
        label="mentions_kg",
        type="relationship",
        summary="Assistant mentions KG refs",
        doc_id=f"conv:{conv_id}",
        domain_id=None,
        canonical_entity_id=None,
        properties={"ref_kind": "kg"},
        embedding=None,
        mentions=[mk_grounding(kg_ref_span)],
        metadata={"causal_type": "reference"},
        source_edge_ids=[],
        target_edge_ids=[],
    )
    conversation_engine.write.add_edge(ref_edge)
    conv_id = "conv_test_1"
    user_id = "user_test_1"

    kg_target_id = (
        "KG_N1"  # must exist in the KG bundle if you want openRef to succeed later
    )

    kg_ref_node = ConversationNode(
        id="CONV_REF_KG_N1",
        label="KG ref → KG_N1",
        type="reference_pointer",  # allowed: 'entity' | 'relationship' | 'reference_pointer'
        summary="Conversation-side pointer to a KG node (for testing focus/openRef).",
        doc_id="CONV_REF_KG_N1",
        mentions=[
            Grounding(
                spans=[
                    Span.from_dummy_for_conversation()
                ]  # ensures spans>=1, mentions>=1
            )
        ],
        properties={
            # keep these JSON-primitive only (validator expects primitives / lists / nested mappings)
            "ref_target_kind": "kg_node",
            "ref_target_id": kg_target_id,
        },
        metadata={
            # REQUIRED by ConversationNodeMetadata
            "level_from_root": 0,
            "entity_type": "kg_ref",
            "in_conversation_chain": False,
            # OPTIONAL but useful (ConversationRoleMixin syncs these too)
            "role": "system",
            "turn_index": 1,
            "conversation_id": conv_id,
            "user_id": user_id,
        },
        role="system",
        turn_index=1,
        conversation_id=conv_id,
        user_id=user_id,
        embedding=None,
        domain_id=None,
        canonical_entity_id=None,
        level_from_root=0,
    )

    # then upsert it into the conversation engine alongside your other seeded nodes
    conversation_engine.write.add_node(kg_ref_node)
    return {
        "conversation_id": conv_id,
        "start_node_id": start_id,
        "turn_ids": (turn0.id, turn1.id),
        "edge_ids": (next_edge.id, summ_edge.id, ref_edge.id),
        "memctx_id": memctx.id,
        "summary_id": summ.id,
        "kg_ref_id": kg_ref_node.id,
    }


# def seed_both_graphs(
#     *,
#     kg_engine: Any,
#     conversation_engine: Any,
#     user_id: str = "U_TEST",
# ) -> dict[str, Any]:
#     kg_seed = seed_kg_graph(kg_engine=kg_engine, kg_doc_id="D_KG_001")
#     conv_seed = seed_conversation_graph(
#         conversation_engine=conversation_engine,
#         user_id=user_id,
#         conversation_id="CONV_TEST_001",
#         start_node_id="CONV_START_001",
#         kg_seed=kg_seed,
#     )
#     return {"kg": kg_seed, "conversation": conv_seed}


@pytest.fixture
def seeded_kg_and_conversation(request, tmp_path: Path, backend_kind, embedding_function):
    """
    Returns (kg_engine, conversation_engine, kg_seed, conv_seed, kg_dir, conv_dir)

    - Real persisted engines in tmp_path
    - KG is seeded with real Node/Edge objects (schema-correct Span/Grounding)
    - Conversation is seeded with conversation nodes/edges + memory ctx + summary + kg_ref
    - Conversation kg_ref points to KG ids via properties.ref_node_ids/ref_edge_ids
    """
    kg_dir = tmp_path / f"{backend_kind}_kg"
    conv_dir = tmp_path / f"{backend_kind}_conversation"

    if backend_kind == "fake":
        kg_engine = GraphKnowledgeEngine(
            persist_directory=str(kg_dir),
            kg_graph_type="knowledge",
            embedding_function=embedding_function,
            backend_factory=build_fake_backend,
        )
        conversation_engine = GraphKnowledgeEngine(
            persist_directory=str(conv_dir),
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
        kg_engine = GraphKnowledgeEngine(
            persist_directory=str(kg_dir),
            kg_graph_type="knowledge",
            embedding_function=embedding_function,
            backend=PgVectorBackend(
                engine=sa_engine, embedding_dim=384, schema=f"{pg_schema}_kg"
            ),
        )
        conversation_engine = GraphKnowledgeEngine(
            persist_directory=str(conv_dir),
            kg_graph_type="conversation",
            embedding_function=embedding_function,
            backend=PgVectorBackend(
                engine=sa_engine, embedding_dim=384, schema=f"{pg_schema}_conv"
            ),
        )
    else:
        try:
            kg_engine = GraphKnowledgeEngine(
                persist_directory=str(kg_dir),
                kg_graph_type="knowledge",
                embedding_function=embedding_function,
            )
            conversation_engine = GraphKnowledgeEngine(
                persist_directory=str(conv_dir),
                kg_graph_type="conversation",
                embedding_function=embedding_function,
            )
        except Exception as exc:
            if embedding_function is None:
                pytest.skip(f"real embedding provider unavailable: {exc}")
            raise
    _install_conversation_policy(conversation_engine)

    kg_seed = seed_kg_graph(kg_engine=kg_engine, kg_doc_id="D_KG_001")
    conv_seed = seed_conversation_graph(
        conversation_engine=conversation_engine,
        user_id="U_TEST",
        conversation_id="CONV_TEST_001",
        start_node_id="CONV_START_001",
        kg_seed=kg_seed,
    )

    return kg_engine, conversation_engine, kg_seed, conv_seed, kg_dir, conv_dir
