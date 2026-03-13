from __future__ import annotations
import os

from sqlalchemy import create_engine
from sqlalchemy.engine import URL, make_url
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from .models import Base

_SessionLocal = None


def _is_test_env() -> bool:
    if os.getenv("PYTEST_CURRENT_TEST"):
        return True
    if os.getenv("TESTING") in {"1", "true", "yes"}:
        return True
    if (os.getenv("ENV") or "").lower() in {"test", "ci"}:
        return True
    return False


def _is_sqlite_in_memory(url: URL) -> bool:
    if url.get_backend_name() != "sqlite":
        return False
    database = url.database or ""
    if database in {"", ":memory:"}:
        return True
    if (url.query or {}).get("mode") == "memory":
        return True
    return False


def create_auth_engine(db_url: str, *, allow_in_memory: bool | None = None):
    url = make_url(db_url)
    in_memory = _is_sqlite_in_memory(url)
    if in_memory:
        if allow_in_memory is None:
            allow_in_memory = _is_test_env()
        if not allow_in_memory:
            raise RuntimeError(
                "In-memory SQLite auth DB is only allowed in tests. "
                "Use a file-based SQLite database or a shared server database."
            )
        return create_engine(
            db_url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
    return create_engine(db_url)


def init_auth_db(engine):
    global _SessionLocal
    Base.metadata.create_all(bind=engine)
    _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_session() -> Session:
    if _SessionLocal is None:
        raise RuntimeError("Auth DB not initialized. Call init_auth_db first.")
    return _SessionLocal()
