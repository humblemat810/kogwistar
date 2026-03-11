from __future__ import annotations
from datetime import datetime
from typing import Optional
from sqlalchemy.orm import Session
from .models import User, ExternalIdentity, WorkflowACL

class AuthRepository:
    def __init__(self, session: Session):
        self.session = session

    def get_user(self, user_id: str) -> Optional[User]:
        return self.session.query(User).filter(User.user_id == user_id).first()

    def get_user_by_email(self, email: str) -> Optional[User]:
        return self.session.query(User).filter(User.email == email).first()

    def upsert_user(
        self, 
        user_id: str, 
        email: str, 
        display_name: Optional[str] = None,
        global_role: Optional[str] = None,
        global_ns: Optional[str] = None
    ) -> User:
        user = self.get_user(user_id)
        if user:
            user.email = email
            if display_name:
                user.display_name = display_name
            if global_role:
                user.global_role = global_role
            if global_ns:
                user.global_ns = global_ns
        else:
            user = User(
                user_id=user_id,
                email=email,
                display_name=display_name,
                global_role=global_role,
                global_ns=global_ns,
                created_at=datetime.utcnow(),
            )
            self.session.add(user)
        self.session.commit()
        return user

    def update_last_login(self, user_id: str):
        user = self.get_user(user_id)
        if user:
            user.last_login_at = datetime.utcnow()
            self.session.commit()

    def get_external_identity(self, issuer: str, subject: str) -> Optional[ExternalIdentity]:
        return self.session.query(ExternalIdentity).filter(
            ExternalIdentity.issuer == issuer,
            ExternalIdentity.subject == subject
        ).first()

    def link_external_identity(self, user_id: str, issuer: str, subject: str, email: Optional[str] = None):
        identity = ExternalIdentity(
            user_id=user_id,
            issuer=issuer,
            subject=subject,
            email=email
        )
        self.session.add(identity)
        self.session.commit()

    def get_workflow_acl(self, workflow_id: str, user_id: str) -> Optional[WorkflowACL]:
        return self.session.query(WorkflowACL).filter(
            WorkflowACL.workflow_id == workflow_id,
            WorkflowACL.user_id == user_id
        ).first()

    def set_workflow_acl(self, workflow_id: str, user_id: str, role: str):
        acl = self.get_workflow_acl(workflow_id, user_id)
        if acl:
            acl.role = role
        else:
            acl = WorkflowACL(workflow_id=workflow_id, user_id=user_id, role=role)
            self.session.add(acl)
        self.session.commit()
