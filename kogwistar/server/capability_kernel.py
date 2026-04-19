from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from typing import Any, Iterable

from fastapi import HTTPException


@dataclass(frozen=True)
class CapabilitySpec:
    name: str
    description: str = ""
    action_kind: str = "general"
    parent: str | None = None


@dataclass(frozen=True)
class CapabilityDecision:
    ts_ms: int
    subject: str
    action: str
    resource: str
    required: tuple[str, ...]
    granted: tuple[str, ...]
    outcome: str
    reason: str = ""
    parent_capabilities: tuple[str, ...] = ()


@dataclass
class CapabilityKernel:
    specs: dict[str, CapabilitySpec] = field(default_factory=dict)
    approvals: dict[tuple[str, str], set[str]] = field(default_factory=dict)
    revoked: dict[str, set[str]] = field(default_factory=dict)
    audit_log: list[CapabilityDecision] = field(default_factory=list)

    def register(self, spec: CapabilitySpec) -> CapabilitySpec:
        self.specs[spec.name] = spec
        return spec

    def list_specs(self) -> list[CapabilitySpec]:
        return list(self.specs.values())

    @staticmethod
    def _normalize_caps(value: str | Iterable[str]) -> tuple[str, ...]:
        if isinstance(value, str):
            items = [value]
        else:
            items = list(value)
        out = []
        for item in items:
            text = str(item).strip().lower()
            if text and text != "*":
                out.append(text)
        return tuple(dict.fromkeys(out))

    def grant(self, *, subject: str, action: str, capabilities: str | Iterable[str]) -> None:
        key = (str(subject).strip().lower(), str(action).strip().lower())
        self.approvals.setdefault(key, set()).update(self._normalize_caps(capabilities))

    def revoke(self, *, subject: str, capability: str) -> None:
        subj = str(subject).strip().lower()
        cap = str(capability).strip().lower()
        if not subj or not cap:
            return
        self.revoked.setdefault(subj, set()).add(cap)

    def materialize_capabilities(
        self,
        *,
        subject: str,
        parent_capabilities: str | Iterable[str] | None = None,
    ) -> tuple[str, ...]:
        subj = str(subject).strip().lower()
        parent = self._normalize_caps(parent_capabilities or ())
        granted = set(parent)
        for (row_subject, _action), caps in self.approvals.items():
            if row_subject == subj:
                granted.update(caps)
        revoked = self.revoked.get(subj, set())
        return tuple(sorted({cap for cap in granted if cap not in revoked}))

    def allowed(
        self,
        *,
        subject: str,
        action: str,
        required: str | Iterable[str],
        parent_capabilities: str | Iterable[str] | None = None,
    ) -> tuple[bool, CapabilityDecision]:
        subj = str(subject).strip().lower()
        act = str(action).strip().lower()
        req = self._normalize_caps(required)
        parent = self._normalize_caps(parent_capabilities or ())
        granted = set(parent)
        granted.update(self.approvals.get((subj, act), set()))
        revoked = self.revoked.get(subj, set())
        effective = [cap for cap in req if cap in granted and cap not in revoked]
        outcome = "allow" if effective and len(effective) == len(req) else "deny"
        reason = ""
        if outcome == "deny":
            missing = [cap for cap in req if cap not in granted]
            blocked = [cap for cap in req if cap in revoked]
            reason = "missing=" + ",".join(missing or ["*"])
            if blocked:
                reason += ";revoked=" + ",".join(blocked)
        decision = CapabilityDecision(
            ts_ms=int(time() * 1000),
            subject=subj,
            action=act,
            resource=act,
            required=req,
            granted=tuple(sorted(granted)),
            outcome=outcome,
            reason=reason,
            parent_capabilities=parent,
        )
        return outcome == "allow", decision

    def require(
        self,
        *,
        subject: str,
        action: str,
        required: str | Iterable[str],
        parent_capabilities: str | Iterable[str] | None = None,
        approval_message: str | None = None,
    ) -> CapabilityDecision:
        ok, decision = self.allowed(
            subject=subject,
            action=action,
            required=required,
            parent_capabilities=parent_capabilities,
        )
        self.audit_log.append(decision)
        if ok:
            return decision
        detail = approval_message or (
            f"Forbidden: action '{action}' requires capability {sorted(decision.required)}"
        )
        raise HTTPException(status_code=403, detail=detail)

    def snapshot(self) -> dict[str, Any]:
        return {
            "specs": [
                {
                    "name": spec.name,
                    "description": spec.description,
                    "action_kind": spec.action_kind,
                    "parent": spec.parent,
                }
                for spec in self.list_specs()
            ],
            "approvals": [
                {
                    "subject": subject,
                    "action": action,
                    "capabilities": sorted(caps),
                }
                for (subject, action), caps in sorted(self.approvals.items())
            ],
            "revoked": [
                {"subject": subject, "capabilities": sorted(caps)}
                for subject, caps in sorted(self.revoked.items())
            ],
            "audit_log": [
                {
                    "ts_ms": d.ts_ms,
                    "subject": d.subject,
                    "action": d.action,
                    "required": list(d.required),
                    "granted": list(d.granted),
                    "outcome": d.outcome,
                    "reason": d.reason,
                    "parent_capabilities": list(d.parent_capabilities),
                }
                for d in self.audit_log
            ],
        }


DEFAULT_CAPABILITY_SPECS = [
    CapabilitySpec("read_graph", "Read graph state", action_kind="read"),
    CapabilitySpec("write_graph", "Write graph state", action_kind="write"),
    CapabilitySpec("send_message", "Send conversation message", action_kind="message"),
    CapabilitySpec("spawn_process", "Spawn workflow process", action_kind="process"),
    CapabilitySpec("invoke_tool", "Invoke tool", action_kind="tool"),
    CapabilitySpec("read_security_scope", "Read security scope", action_kind="read"),
    CapabilitySpec("project_view", "Project view", action_kind="read"),
    CapabilitySpec("approve_action", "Approve blocked action", action_kind="approve"),
    CapabilitySpec("workflow.design.inspect", "Inspect workflow design", action_kind="read"),
    CapabilitySpec("workflow.design.write", "Mutate workflow design", action_kind="write"),
    CapabilitySpec("workflow.run.read", "Read workflow run", action_kind="read"),
    CapabilitySpec("workflow.run.write", "Create or mutate workflow run", action_kind="write"),
    CapabilitySpec("service.inspect", "Inspect service state", action_kind="read"),
    CapabilitySpec("service.manage", "Manage service lifecycle", action_kind="write"),
    CapabilitySpec("service.heartbeat", "Record service heartbeat", action_kind="write"),
]
