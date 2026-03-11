from __future__ import annotations
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from .models import Base

_SessionLocal = None

def init_auth_db(engine):
    global _SessionLocal
    Base.metadata.create_all(bind=engine)
    _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_session() -> Session:
    if _SessionLocal is None:
        raise RuntimeError("Auth DB not initialized. Call init_auth_db first.")
    return _SessionLocal()
