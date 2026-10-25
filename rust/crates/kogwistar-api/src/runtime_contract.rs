use base64::Engine as _;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

pub(crate) fn default_runtime_kind() -> String {
    "sync".to_owned()
}

pub(crate) fn worker_protocol_for_runtime_kind(runtime_kind: &str) -> &'static str {
    if runtime_kind == "async" {
        kogwistar_runtime::WORKER_PROTOCOL_ASYNC_V2
    } else {
        kogwistar_runtime::WORKER_PROTOCOL_SYNC_V1
    }
}

pub(crate) fn runtime_implementation_attestation(runtime_kind: &str) -> Value {
    json!({
        "schema_version": 1,
        "source": "rust_server_admission",
        "scheduler": {
            "language": "rust",
            "package": "kogwistar-api",
            "version": env!("CARGO_PKG_VERSION"),
        },
        "worker_contract": {
            "runtime_kind": runtime_kind,
            "protocol": worker_protocol_for_runtime_kind(runtime_kind),
            "identity_status": "resolved_on_lane_claim",
        },
    })
}

pub(crate) fn runtime_worker_recipient(environment_id: Option<&str>) -> String {
    let Some(environment_id) = environment_id.filter(|value| !value.trim().is_empty()) else {
        return "python-worker".to_owned();
    };
    let digest = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .encode(Sha256::digest(environment_id.as_bytes()));
    format!("python-worker-env-{digest}")
}
