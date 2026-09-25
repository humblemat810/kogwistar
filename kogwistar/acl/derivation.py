"""ACL taint propagation and opt-in declassification.

The strict join is the security decision. ``LLM_GUARDED`` may only propose a
post-join declassification; it is never an alternative way to classify output.
No model-produced ACL field is trusted by this module.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Callable, Iterable, Literal, Mapping

from .graph import ACLMode, ACLRecord


ACLDerivationPolicy = Literal["STRICT", "LLM_GUARDED"]
DeclassificationResult = Literal[
    "not_requested",
    "not_configured",
    "approved",
    "rejected",
    "failed",
    "malformed",
    "uncertain",
]

LLM_GUARDED_WARNING = (
    "LLM_GUARDED is opt-in declassification. It does not change ACL taint "
    "semantics; STRICT taint is computed first and remains the fallback. "
    "LLM_GUARDED requires explicit acknowledgement."
)


def _rank(mode: ACLMode) -> int:
    return {"public": 0, "group": 1, "shared": 2, "scope": 3, "private": 4}[mode]


def normalize_derivation_policy(policy: str | None) -> ACLDerivationPolicy:
    value = str(policy or "STRICT").strip().upper().replace("-", "_")
    if value not in {"STRICT", "LLM_GUARDED"}:
        raise ValueError("ACL derivation policy must be STRICT or LLM_GUARDED")
    return value  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class ACLInput:
    """Non-content ACL descriptor for one semantic input.

    Content is intentionally absent. Audit stores identity and ACL metadata,
    not prompts, documents, or private payloads.
    """

    object_id: str
    mode: ACLMode
    owner_id: str | None = None
    security_scope: str | None = None
    source_kind: str = "semantic_input"
    provenance: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in {"private", "shared", "scope", "group", "public"}:
            raise ValueError("ACLInput mode is invalid")
        if not str(self.object_id).strip():
            raise ValueError("ACLInput object_id is required")


@dataclass(frozen=True, slots=True)
class ACLJoin:
    mode: ACLMode
    owner_id: str | None
    security_scope: str | None
    source_ids: tuple[str, ...]
    inputs: tuple[dict[str, Any], ...]
    clean_room: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "owner_id": self.owner_id,
            "security_scope": self.security_scope,
            "source_ids": list(self.source_ids),
            "inputs": [dict(item) for item in self.inputs],
            "clean_room": self.clean_room,
        }


@dataclass(frozen=True, slots=True)
class ACLDerivationResult:
    """Auditable output of strict join plus optional declassification."""

    original_acl: ACLJoin
    final_mode: ACLMode
    policy: ACLDerivationPolicy
    declassification_attempted: bool
    declassification_result: DeclassificationResult
    object_id: str | None = None
    generation_id: str | None = None
    proposed_mode: ACLMode | None = None
    reason: str | None = None
    confidence: float | None = None
    evidence: tuple[str, ...] = ()
    classifier_model: str | None = None
    classifier_version: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def clean_room(self) -> bool:
        return self.original_acl.clean_room

    @property
    def source_ids(self) -> tuple[str, ...]:
        return self.original_acl.source_ids

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_acl": self.original_acl.mode,
            "proposed_acl": self.proposed_mode,
            "final_acl": self.final_mode,
            "object_id": self.object_id,
            "generation_id": self.generation_id,
            "policy": self.policy,
            "classifier_model": self.classifier_model,
            "classifier_version": self.classifier_version,
            "reason": self.reason,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
            "timestamp": self.timestamp,
            "inputs": [dict(item) for item in self.original_acl.inputs],
            "sources": list(self.source_ids),
            "visible_context_acl_join": self.original_acl.mode,
            "declassification_attempted": self.declassification_attempted,
            "declassification_result": self.declassification_result,
            "clean_room": self.clean_room,
        }

    def to_metadata(self) -> dict[str, Any]:
        """Return metadata suitable for a derived node/artifact."""
        return {
            "acl_mode": self.final_mode,
            "owner_id": self.original_acl.owner_id,
            "security_scope": self.original_acl.security_scope,
            "source_ids": list(self.source_ids),
            "derivation_policy": self.policy,
            "derivation_audit": self.to_dict(),
        }


def coerce_acl_input(value: ACLInput | ACLRecord | Mapping[str, Any]) -> ACLInput:
    """Parse trusted ACL descriptors; missing ACL fails closed to private."""
    if isinstance(value, ACLInput):
        return value
    if isinstance(value, ACLRecord):
        return ACLInput(
            object_id=value.target.entity_id,
            mode=value.mode,
            owner_id=value.owner_id,
            security_scope=value.security_scope,
            source_kind="acl_record",
            provenance=value.source_ids,
        )
    if not isinstance(value, Mapping):
        raise TypeError("ACL input must be ACLInput, ACLRecord, or mapping")
    object_id = str(value.get("object_id") or value.get("id") or "unknown")
    raw_mode = str(value.get("mode") or value.get("acl_mode") or value.get("visibility") or "private")
    mode = raw_mode.strip().lower()
    if mode == "global":
        mode = "public"
    if mode not in {"private", "shared", "scope", "group", "public"}:
        mode = "private"
    raw_sources = value.get("source_ids") or value.get("provenance") or ()
    if isinstance(raw_sources, str):
        raw_sources = (raw_sources,)
    return ACLInput(
        object_id=object_id,
        mode=mode,  # type: ignore[arg-type]
        owner_id=str(value["owner_id"]) if value.get("owner_id") else None,
        security_scope=(str(value["security_scope"]) if value.get("security_scope") else None),
        source_kind=str(value.get("source_kind") or "semantic_input"),
        provenance=tuple(str(item) for item in raw_sources),
    )


def join_acl_inputs(
    inputs: Iterable[ACLInput | ACLRecord | Mapping[str, Any]],
    *,
    clean_room: bool = False,
) -> ACLJoin:
    """Compute conservative high-water ACL join.

    A mixed-owner private result keeps ``private`` and clears owner privilege;
    this avoids accidentally granting one source owner access to another's
    derived material. Source identities remain in audit metadata.
    """
    parsed = tuple(coerce_acl_input(item) for item in inputs)
    if not parsed:
        mode: ACLMode = "public" if clean_room else "private"
        return ACLJoin(mode, None, None, (), (), clean_room=bool(clean_room))
    mode = max((item.mode for item in parsed), key=_rank)
    owners = {item.owner_id for item in parsed if item.owner_id}
    scopes = {item.security_scope for item in parsed if item.security_scope}
    owner_id = next(iter(owners)) if len(owners) == 1 else None
    security_scope = next(iter(scopes)) if len(scopes) == 1 else None
    source_ids = tuple(dict.fromkeys(item.object_id for item in parsed if item.object_id))
    audit_inputs = tuple(
        {
            "object_id": item.object_id,
            "mode": item.mode,
            "owner_id": item.owner_id,
            "security_scope": item.security_scope,
            "source_kind": item.source_kind,
            "provenance": list(item.provenance),
        }
        for item in parsed
    )
    return ACLJoin(
        mode=mode,
        owner_id=owner_id,
        security_scope=security_scope,
        source_ids=source_ids,
        inputs=audit_inputs,
        clean_room=bool(clean_room and mode == "public"),
    )


Declassifier = Callable[[Mapping[str, Any]], Mapping[str, Any]]


def derive_acl(
    inputs: Iterable[ACLInput | ACLRecord | Mapping[str, Any]],
    *,
    policy: str | None = "STRICT",
    declassifier: Declassifier | None = None,
    acknowledge_llm_guarded: bool = False,
    clean_room: bool = False,
    object_id: str | None = None,
    generation_id: str | None = None,
) -> ACLDerivationResult:
    """Join first; optionally apply explicit, auditable declassification."""
    selected = normalize_derivation_policy(policy)
    original = join_acl_inputs(inputs, clean_room=clean_room)
    if selected == "STRICT":
        return ACLDerivationResult(
            original_acl=original,
            final_mode=original.mode,
            policy=selected,
            declassification_attempted=False,
            declassification_result="not_requested",
            object_id=object_id,
            generation_id=generation_id,
        )
    if not acknowledge_llm_guarded:
        raise PermissionError(LLM_GUARDED_WARNING)
    warnings.warn(LLM_GUARDED_WARNING, RuntimeWarning, stacklevel=2)
    base_audit = original.to_dict()
    if declassifier is None:
        return ACLDerivationResult(
            original_acl=original,
            final_mode=original.mode,
            policy=selected,
            declassification_attempted=False,
            declassification_result="not_configured",
            object_id=object_id,
            generation_id=generation_id,
        )
    try:
        proposal = declassifier(base_audit)
    except Exception as exc:
        return ACLDerivationResult(
            original_acl=original,
            final_mode=original.mode,
            policy=selected,
            declassification_attempted=True,
            declassification_result="failed",
            object_id=object_id,
            generation_id=generation_id,
            reason=f"guard failed: {type(exc).__name__}",
        )
    if not isinstance(proposal, Mapping):
        result = "malformed"
        return ACLDerivationResult(
            original_acl=original,
            final_mode=original.mode,
            policy=selected,
            declassification_attempted=True,
            declassification_result=result,
            object_id=object_id,
            generation_id=generation_id,
            reason="guard response is not an object",
        )
    proposed_raw = str(proposal.get("proposed_acl") or proposal.get("mode") or "")
    proposed = proposed_raw.strip().lower()
    if proposed == "global":
        proposed = "public"
    if proposed not in {"private", "shared", "scope", "group", "public"}:
        return ACLDerivationResult(
            original_acl=original,
            final_mode=original.mode,
            policy=selected,
            declassification_attempted=True,
            declassification_result="malformed",
            object_id=object_id,
            generation_id=generation_id,
            reason="guard proposed unknown ACL mode",
        )
    confidence = proposal.get("confidence")
    try:
        confidence_value = float(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        confidence_value = None
    approved = proposal.get("approved") is True
    evidence = proposal.get("evidence") or ()
    if isinstance(evidence, str):
        evidence = (evidence,)
    evidence_values = tuple(str(item) for item in evidence)
    model = str(proposal.get("classifier_model")) if proposal.get("classifier_model") else None
    version = str(proposal.get("classifier_version")) if proposal.get("classifier_version") else None
    reason = str(proposal.get("reason")) if proposal.get("reason") else None
    if not approved or _rank(proposed) > _rank(original.mode):
        return ACLDerivationResult(
            original_acl=original,
            final_mode=original.mode,
            policy=selected,
            declassification_attempted=True,
            declassification_result="rejected" if approved else "uncertain",
            object_id=object_id,
            generation_id=generation_id,
            proposed_mode=proposed,  # type: ignore[arg-type]
            reason=reason or "guard did not prove safe declassification",
            confidence=confidence_value,
            evidence=evidence_values,
            classifier_model=model,
            classifier_version=version,
        )
    return ACLDerivationResult(
        original_acl=original,
        final_mode=proposed,  # type: ignore[arg-type]
        policy=selected,
        declassification_attempted=True,
        declassification_result="approved",
        object_id=object_id,
        generation_id=generation_id,
        proposed_mode=proposed,  # type: ignore[arg-type]
        reason=reason,
        confidence=confidence_value,
        evidence=evidence_values,
        classifier_model=model,
        classifier_version=version,
    )


__all__ = [
    "ACLDerivationPolicy",
    "ACLInput",
    "ACLJoin",
    "ACLDerivationResult",
    "Declassifier",
    "LLM_GUARDED_WARNING",
    "coerce_acl_input",
    "derive_acl",
    "join_acl_inputs",
    "normalize_derivation_policy",
]
