from __future__ import annotations
#### log utils

import logging
import sqlite3
import threading
import os
import traceback
def safe_format_exception(exc: Exception, base_path: str = None):
    """Format exception with paths relative to project root."""
    if base_path is None:
        base_path = os.getcwd()  # default to current working directory

    lines = []
    for line in traceback.format_exception(type(exc), exc, exc.__traceback__):
        if line.startswith('  File'):
            parts = line.split('"')
            if len(parts) >= 3:
                full_path = parts[1]
                try:
                    relative_path = os.path.relpath(full_path, base_path)
                    line = line.replace(full_path, relative_path)
                except ValueError:
                    # relpath failed (different drive?), keep original
                    pass
        lines.append(line)
    
    return ''.join(lines)

def trace_logger_hierarchy(logger):
    while logger:
        print(f"Logger Name: {logger.name}")
        print(f"  Level: {logging.getLevelName(logger.level)}")
        print(f"  Handlers: {logger.handlers}")
        print(f"  Propagate: {logger.propagate}")
        print("-" * 40)
        logger = logger.parent

class SQLiteHandler(logging.Handler):
    """
    Custom logging handler that writes log records to an SQLite database,
    including filename and line number information.
    """

    def __init__(self, db_path):
        """
        Initializes the handler with the database path.
        Ensures the log table exists.
        """
        super().__init__()
        self.db_path = db_path
        self.lock = threading.Lock()
        self._initialize_database()
        # Set up a formatter to format the log records
        self.formatter = logging.Formatter('%(asctime)s', '%Y-%m-%d %H:%M:%S')

    def _initialize_database(self):
        """
        Creates the logs table if it doesn't already exist.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    level TEXT,
                    module TEXT,
                    filename TEXT,
                    line_number INTEGER,
                    message TEXT
                )
            ''')
            conn.commit()

    def emit(self, record):
        """
        Inserts a new log record into the database.
        """
        try:
            # Ensure the record is formatted to populate all fields
            self.format(record)
            # Format the timestamp using the formatter
            timestamp = self.formatter.formatTime(record)
            #with self.lock:
            with sqlite3.connect(self.db_path, timeout=10) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO logs (timestamp, level, module, filename, line_number, message)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (timestamp, record.levelname, record.module,
                        record.filename, record.lineno, record.getMessage()))
                conn.commit()
        except Exception:
            self.handleError(record)
    def __del__(self):
        """
        Destructor to perform a WAL checkpoint when the handler is destroyed.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('PRAGMA wal_checkpoint;')
                conn.commit()
        except Exception as e:
            print(f"Error during WAL checkpoint: {e}")
            
            
"""_summary_

usage:

import logging

# Configure the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create the SQLite logging handler
sqlite_handler = SQLiteHandler('application_logs.db')
sqlite_handler.setLevel(logging.DEBUG)

# Add the handler to the logger
logger.addHandler(sqlite_handler)

# Register the handler's close method with the logging shutdown
logging.shutdown = sqlite_handler.close

# Log messages
logger.info('This is an info message.')
logger.error('This is an error message.')

# When the application is terminating
logging.shutdown()

"""




# ----------------------------
# v2 Compatibility Logging Layer
# (keeps existing SQLiteHandler intact)
# ----------------------------


from dataclasses import dataclass
from pathlib import Path
import contextlib
import contextvars
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Dict, Optional, Iterator, Literal


EngineType = Literal["conversation", "workflow", "kg"]


# Context fields you’ll want everywhere
_ctx_engine_type: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("engine_type", default=None)
_ctx_engine_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("engine_id", default=None)
_ctx_conversation_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("conversation_id", default=None)
_ctx_workflow_run_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("workflow_run_id", default=None)
_ctx_step_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("step_id", default=None)
_ctx_op: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("op", default=None)

class ContextFilter(logging.Filter):
    """Inject engine/runtime context into all log records."""
    def filter(self, record: logging.LogRecord) -> bool:
        record.engine_type = _ctx_engine_type.get() or "-"
        record.engine_id = _ctx_engine_id.get() or "-"
        record.conversation_id = _ctx_conversation_id.get() or "-"
        record.workflow_run_id = _ctx_workflow_run_id.get() or "-"
        record.step_id = _ctx_step_id.get() or "-"
        record.op = _ctx_op.get() or "-"
        return True


@contextlib.contextmanager
def bind_log_context(
    *,
    engine_type: Optional[str] = None,
    engine_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    workflow_run_id: Optional[str] = None,
    step_id: Optional[str] = None,
    op: Optional[str] = None,
) -> Iterator[None]:
    """
    Context manager to bind IDs to logs without passing them around.
    Works across async tasks too (contextvars).
    """
    tokens = []
    try:
        if engine_type is not None:
            tokens.append((_ctx_engine_type, _ctx_engine_type.set(engine_type)))
        if engine_id is not None:
            tokens.append((_ctx_engine_id, _ctx_engine_id.set(engine_id)))
        if conversation_id is not None:
            tokens.append((_ctx_conversation_id, _ctx_conversation_id.set(conversation_id)))
        if workflow_run_id is not None:
            tokens.append((_ctx_workflow_run_id, _ctx_workflow_run_id.set(workflow_run_id)))
        if step_id is not None:
            tokens.append((_ctx_step_id, _ctx_step_id.set(step_id)))
        if op is not None:
            tokens.append((_ctx_op, _ctx_op.set(op)))

        yield
    finally:
        # reset in reverse order
        for var, tok in reversed(tokens):
            try:
                var.reset(tok)
            except Exception:
                pass


@dataclass(frozen=True)
class EngineLogConfig:
    """
    Logging configuration.
    - base_dir: where to write log files if enabled
    - app_name: prefix for log file names
    - level: default logging level
    - enable_files: whether to write to rotating files
    - enable_sqlite: whether to attach your existing SQLiteHandler
    - sqlite_db_path: path to sqlite db (required if enable_sqlite)
    - mode: "prod" installs handlers; "pytest" avoids handlers (pytest captures)
    """
    base_dir: Path = Path(".logs")
    app_name: str = "graph_knowledge_engine"
    level: int = logging.INFO

    enable_files: bool = True
    max_bytes: int = 10 * 1024 * 1024
    backup_count: int = 5

    enable_sqlite: bool = False
    sqlite_db_path: Optional[Path] = None
    enable_jsonl: bool = False

    mode: Literal["prod", "pytest"] = "prod"


class EngineLogManager:
    """
    One-time logging setup + logger factory for the 3 engine types:
    conversation / workflow / kg.

    Design goals:
    - In prod: write to stderr (optional) + rotating file per engine type (optional)
    - In pytest: DO NOT install extra handlers; let pytest/vscode capture logs
    - Always inject context fields via ContextFilter
    """
    _configured: bool = False
    _config: Optional[EngineLogConfig] = None
    _loggers: Dict[str, logging.Logger] = {}

    def configure(
        *,
        base_dir: Path,
        app_name: str = "gke",
        level: int = logging.INFO,
        enable_console: bool = True,
        enable_files: bool = True,
        enable_jsonl: bool = True,
        enable_sqlite: bool = False,
        sqlite_db_path: Optional[Path] = None,
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5,
    ) -> None:
        """
        Call ONCE at process startup (or pytest_configure).
        Installs handlers on ROOT only.
        """
        root = logging.getLogger()
        root.setLevel(level)

        if root.handlers:
            return  # avoid duplicate install

        base_dir.mkdir(parents=True, exist_ok=True)

        fmt = (
            "%(asctime)s %(levelname)s [%(name)s] "
            "engine=%(engine_type)s engine_id=%(engine_id)s "
            "conv=%(conversation_id)s run=%(workflow_run_id)s step=%(step_id)s op=%(op)s "
            "%(message)s"
        )

        formatter = logging.Formatter(fmt)

        # ---- Console ----
        if enable_console:
            sh = logging.StreamHandler()
            sh.setLevel(level)
            sh.setFormatter(formatter)
            sh.addFilter(ContextFilter())
            root.addHandler(sh)

        if not enable_files:
            return

        # ---- Consolidated file ----
        all_log = RotatingFileHandler(
            base_dir / f"{app_name}.all.log",
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        all_log.setLevel(level)
        all_log.setFormatter(formatter)
        all_log.addFilter(ContextFilter())
        root.addHandler(all_log)

        # ---- Per-engine routing ----
        for engine_type in ("conversation", "workflow", "kg"):

            def engine_filter(record, et=engine_type):
                return getattr(record, "engine_type", None) == et

            h = RotatingFileHandler(
                base_dir / f"{app_name}.{engine_type}.log",
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            h.setLevel(level)
            h.setFormatter(formatter)
            h.addFilter(ContextFilter())
            h.addFilter(engine_filter)
            root.addHandler(h)

            if enable_jsonl:
                j = RotatingFileHandler(
                    base_dir / f"{app_name}.{engine_type}.jsonl",
                    maxBytes=max_bytes,
                    backupCount=backup_count,
                    encoding="utf-8",
                )
                j.setLevel(level)
                j.setFormatter(JsonlFormatter())
                j.addFilter(ContextFilter())
                j.addFilter(engine_filter)
                root.addHandler(j)

        # ---- SQLite (optional) ----
        if enable_sqlite and sqlite_db_path is not None:
            sqlite_handler = SQLiteHandler(str(sqlite_db_path))
            sqlite_handler.setLevel(level)
            sqlite_handler.addFilter(ContextFilter())
            root.addHandler(sqlite_handler)

    @classmethod
    def _ensure_per_engine_handlers(cls, logger: logging.Logger, engine_type: EngineType) -> None:
        cfg = cls._config
        if cfg is None:
            # default safe config: pytest-like behavior (no extra handlers)
            return
        if cfg.mode == "pytest":
            return

        # We let records propagate to root for console, but optionally add file/sqlite on the leaf logger.
        logger.propagate = True

        # Add file handler only once per logger
        if cfg.enable_files and not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
            base_dir = Path(cfg.base_dir)
            base_dir.mkdir(parents=True, exist_ok=True)
            log_path = base_dir / f"{cfg.app_name}.{engine_type}.log"

            fh = RotatingFileHandler(
                log_path,
                maxBytes=cfg.max_bytes,
                backupCount=cfg.backup_count,
                encoding="utf-8",
            )
            fh.setLevel(cfg.level)
            fmt = (
                "%(asctime)s %(levelname)s "
                "[%(name)s] "
                "engine=%(engine_type)s engine_id=%(engine_id)s "
                "conv=%(conversation_id)s run=%(workflow_run_id)s step=%(step_id)s op=%(op)s  "
                "%(message)s"
            )
            fh.setFormatter(logging.Formatter(fmt))
            fh.addFilter(ContextFilter())
            logger.addHandler(fh)
            
            jsonl_path = base_dir / f"{cfg.app_name}.{engine_type}.jsonl"
            jh = RotatingFileHandler(
                jsonl_path,
                maxBytes=cfg.max_bytes,
                backupCount=cfg.backup_count,
                encoding="utf-8",
            )
            jh.setLevel(cfg.level)
            jh.setFormatter(JsonlFormatter())
            jh.addFilter(ContextFilter())
            logger.addHandler(jh)
        # Attach your existing SQLiteHandler if enabled
        if cfg.enable_sqlite and cfg.sqlite_db_path is not None:
            # only add once
            if not any(h.__class__.__name__ == "SQLiteHandler" for h in logger.handlers):
                try:
                    sqlite_handler = SQLiteHandler(str(cfg.sqlite_db_path))  # uses your existing class
                    sqlite_handler.setLevel(cfg.level)
                    sqlite_handler.addFilter(ContextFilter())
                    logger.addHandler(sqlite_handler)
                except Exception:
                    # Never let logging setup crash the app
                    logger.exception("Failed to initialize SQLiteHandler for logs")

    @classmethod
    def get_logger(cls, engine_type: EngineType) -> logging.Logger:
        """
        Returns a logger for engine_type:
          - graph_knowledge_engine.conversation
          - graph_knowledge_engine.workflow
          - graph_knowledge_engine.kg
        """
        name = f"graph_knowledge_engine.{engine_type}"
        if name in cls._loggers:
            return cls._loggers[name]

        logger = logging.getLogger(name)
        cfg = cls._config
        if cfg is not None:
            logger.setLevel(cfg.level)
        cls._ensure_per_engine_handlers(logger, engine_type)
        cls._loggers[name] = logger
        return logger

    @classmethod
    def is_pytest_running(cls) -> bool:
        # Heuristic: VSCode pytest, pytest CLI, etc.
        return ("PYTEST_CURRENT_TEST" in os.environ) or ("pytest" in sys.modules)


# Convenience getters (so callsites stay tiny)
def conversation_logger() -> logging.Logger:
    return EngineLogManager.get_logger("conversation")


def workflow_logger() -> logging.Logger:
    return EngineLogManager.get_logger("workflow")


def kg_logger() -> logging.Logger:
    return EngineLogManager.get_logger("kg")


import json
import datetime as _dt

class JsonlFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": _dt.datetime.utcfromtimestamp(record.created).isoformat(timespec="milliseconds") + "Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "engine_type": getattr(record, "engine_type", "-"),
            "engine_id": getattr(record, "engine_id", "-"),
            "conversation_id": getattr(record, "conversation_id", "-"),
            "workflow_run_id": getattr(record, "workflow_run_id", "-"),
            "step_id": getattr(record, "step_id", "-"),
            "file": record.pathname,
            "line": record.lineno,
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)