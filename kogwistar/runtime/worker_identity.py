"""Versioned worker build identity contract shared with Rust.

Identity-specific lane claims require this value in the API's verified JWT
claim ``kogwistar_worker_identity``. The trusted issuer must only issue it for
host-registered artifact and dependency-lock digests; a worker's local copy is
not itself registration or proof.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator


class WorkerIdentityAttestationV1(BaseModel):
    """Self-reported worker identity data; not authorization or trusted proof.

    ``artifact_sha256`` identifies the immutable deployed image or executable
    build; ``dependency_lock_sha256`` identifies the exact dependency lock.
    Admission must compare both with host-controlled deployment registration
    before treating them as evidence of the running artifact.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1]
    environment_id: StrictStr
    artifact_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    dependency_lock_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")

    def to_wire(self) -> dict[str, int | str]:
        """Return the canonical JSON-compatible contract payload."""
        return self.model_dump(mode="json")

    @field_validator("environment_id")
    @classmethod
    def _validate_environment_id(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError("environment_id must be non-empty and canonical")
        return value
