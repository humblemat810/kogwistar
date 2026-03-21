from __future__ import annotations
from datetime import datetime
from typing import Optional
from sqlalchemy import String, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    display_name: Mapped[Optional[str]] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    global_role: Mapped[Optional[str]] = mapped_column(String(32))  # e.g. 'ro', 'rw'
    global_ns: Mapped[Optional[str]] = mapped_column(
        String(255)
    )  # comma-separated or wildcard
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_login_at: Mapped[Optional[datetime]] = mapped_column(DateTime)


class ExternalIdentity(Base):
    __tablename__ = "external_identities"

    issuer: Mapped[str] = mapped_column(String(255), primary_key=True)
    subject: Mapped[str] = mapped_column(String(255), primary_key=True)
    user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.user_id"), index=True
    )
    email: Mapped[Optional[str]] = mapped_column(String(255))

    user: Mapped[User] = relationship("User")


class WorkflowACL(Base):
    __tablename__ = "workflow_acl"

    workflow_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.user_id"), primary_key=True
    )
    role: Mapped[str] = mapped_column(String(32))  # e.g. 'ro', 'rw'
