use kogwistar_runtime::canonical_json_value;
use serde_json::Value;
use sha2::{Digest, Sha256};

pub fn canonical_json(value: &Value) -> String {
    canonical_json_value(value)
}

pub fn sha256_hex(payload: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(payload);
    format!("{:x}", hasher.finalize())
}
