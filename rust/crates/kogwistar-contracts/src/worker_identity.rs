use serde::{Deserialize, Serialize};
use thiserror::Error;

/// `artifact_sha256` identifies the deployed image or executable build;
/// `dependency_lock_sha256` identifies its exact dependency lock. These are
/// declarations only until admission checks them against trusted registration.
/// PYTHON_PARITY_V1: rust=rust/crates/kogwistar-contracts/src/worker_identity.rs#worker_identity_attestation_from_str; python=kogwistar/runtime/worker_identity.py#WorkerIdentityAttestationV1.to_wire; fixture=contracts/golden/worker-identity-attestation-v1.json; rust_test=rust/crates/kogwistar-contracts/src/worker_identity.rs#worker_identity_attestation_matches_shared_python_golden_fixture; python_test=tests/runtime/test_worker_identity_contract.py#test_worker_identity_attestation_matches_shared_fixture
pub const WORKER_IDENTITY_ATTESTATION_VERSION: u32 = 1;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerIdentityAttestationV1 {
    pub schema_version: u32,
    pub environment_id: String,
    pub artifact_sha256: String,
    pub dependency_lock_sha256: String,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum WorkerIdentityAttestationError {
    #[error("worker identity attestation payload is invalid: {0}")]
    InvalidPayload(String),
    #[error("worker identity attestation schema_version must be 1")]
    UnsupportedVersion,
    #[error("worker identity attestation environment_id must be non-empty and canonical")]
    InvalidEnvironmentId,
    #[error("worker identity attestation {field} must be 64 lowercase hexadecimal characters")]
    InvalidDigest { field: &'static str },
}

impl WorkerIdentityAttestationV1 {
    pub fn validate(&self) -> Result<(), WorkerIdentityAttestationError> {
        if self.schema_version != WORKER_IDENTITY_ATTESTATION_VERSION {
            return Err(WorkerIdentityAttestationError::UnsupportedVersion);
        }
        if self.environment_id.trim().is_empty()
            || self.environment_id.trim() != self.environment_id
        {
            return Err(WorkerIdentityAttestationError::InvalidEnvironmentId);
        }
        for (field, digest) in [
            ("artifact_sha256", self.artifact_sha256.as_str()),
            (
                "dependency_lock_sha256",
                self.dependency_lock_sha256.as_str(),
            ),
        ] {
            if digest.len() != 64
                || !digest
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            {
                return Err(WorkerIdentityAttestationError::InvalidDigest { field });
            }
        }
        Ok(())
    }
}

pub fn worker_identity_attestation_from_str(
    payload: &str,
) -> Result<WorkerIdentityAttestationV1, WorkerIdentityAttestationError> {
    let attestation: WorkerIdentityAttestationV1 = serde_json::from_str(payload)
        .map_err(|error| WorkerIdentityAttestationError::InvalidPayload(error.to_string()))?;
    attestation.validate()?;
    Ok(attestation)
}

#[cfg(test)]
mod tests {
    use super::worker_identity_attestation_from_str;
    use serde::Deserialize;
    use serde_json::Value;

    #[derive(Deserialize)]
    struct Fixture {
        valid: Value,
        invalid: Vec<Value>,
    }

    #[test]
    fn worker_identity_attestation_matches_shared_python_golden_fixture() {
        let fixture: Fixture = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/worker-identity-attestation-v1.json"
        )))
        .expect("shared worker identity fixture");

        let parsed = worker_identity_attestation_from_str(&fixture.valid.to_string())
            .expect("valid worker identity attestation");
        assert_eq!(
            serde_json::to_value(parsed).expect("serialize attestation"),
            fixture.valid
        );
        for invalid in fixture.invalid {
            assert!(worker_identity_attestation_from_str(&invalid.to_string()).is_err());
        }
    }
}
