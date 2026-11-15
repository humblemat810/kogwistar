use serde_json::Value;

/// Versioned evidence that a response was produced by the native Rust path.
///
/// This is diagnostic metadata, not an authorization grant. Keeping its
/// construction in one module prevents individual Rust services from silently
/// drifting in the fields used to audit language-seam crossings.
pub const NATIVE_SEAM_ATTESTATION_VERSION: u32 = 1;
pub const RUST_NATIVE_IMPLEMENTATION_AUTHORITY: &str = "rust";
pub const RUST_NATIVE_EXECUTOR_AUTHORITY: &str = "rust-native";

/// Authorities that may truthfully claim a native, zero-crossing result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeSeamAuthority {
    Graph,
    RuntimeExecutor,
}

impl NativeSeamAuthority {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Graph => RUST_NATIVE_IMPLEMENTATION_AUTHORITY,
            Self::RuntimeExecutor => RUST_NATIVE_EXECUTOR_AUTHORITY,
        }
    }
}

pub fn attach_typed_native_seam_attestation(value: &mut Value, authority: NativeSeamAuthority) {
    attach_native_seam_attestation(value, authority.as_str());
}

pub fn attach_native_seam_attestation(value: &mut Value, implementation_authority: &str) {
    if !matches!(
        implementation_authority,
        RUST_NATIVE_IMPLEMENTATION_AUTHORITY | RUST_NATIVE_EXECUTOR_AUTHORITY
    ) {
        return;
    }
    let Some(object) = value.as_object_mut() else {
        return;
    };
    object.insert(
        "implementation_authority".to_owned(),
        Value::String(implementation_authority.to_owned()),
    );
    object.insert(
        "seam_attestation_version".to_owned(),
        Value::from(NATIVE_SEAM_ATTESTATION_VERSION),
    );
    object.insert("seam_crossings".to_owned(), Value::Array(Vec::new()));
}

pub fn attach_rust_native_seam_attestation(value: &mut Value) {
    attach_typed_native_seam_attestation(value, NativeSeamAuthority::Graph);
}

pub fn attach_rust_native_executor_attestation(value: &mut Value) {
    attach_typed_native_seam_attestation(value, NativeSeamAuthority::RuntimeExecutor);
}

/// Validate the conservative meaning of a native zero-crossing attestation.
pub fn native_seam_attestation_is_valid(value: &Value) -> bool {
    let Some(object) = value.as_object() else {
        return false;
    };
    let authority = object
        .get("implementation_authority")
        .and_then(Value::as_str);
    let version = object
        .get("seam_attestation_version")
        .and_then(Value::as_u64);
    let crossings_empty = object
        .get("seam_crossings")
        .and_then(Value::as_array)
        .is_some_and(Vec::is_empty);
    matches!(
        authority,
        Some(RUST_NATIVE_IMPLEMENTATION_AUTHORITY) | Some(RUST_NATIVE_EXECUTOR_AUTHORITY)
    ) && version == Some(NATIVE_SEAM_ATTESTATION_VERSION as u64)
        && crossings_empty
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn native_seam_attestation_accepts_only_typed_native_authorities() {
        for authority in [
            NativeSeamAuthority::Graph,
            NativeSeamAuthority::RuntimeExecutor,
        ] {
            let mut value = json!({"result": "ok"});
            attach_typed_native_seam_attestation(&mut value, authority);
            assert!(native_seam_attestation_is_valid(&value));
        }

        let mut forged = json!({"result": "ok"});
        attach_native_seam_attestation(&mut forged, "python");
        assert!(!native_seam_attestation_is_valid(&forged));
    }

    #[test]
    fn native_seam_attestation_rejects_version_or_crossing_drift() {
        let mut value = json!({"result": "ok"});
        attach_typed_native_seam_attestation(&mut value, NativeSeamAuthority::Graph);

        value["seam_attestation_version"] = json!(2);
        assert!(!native_seam_attestation_is_valid(&value));

        attach_typed_native_seam_attestation(&mut value, NativeSeamAuthority::Graph);
        value["seam_crossings"] = json!(["python-json"]);
        assert!(!native_seam_attestation_is_valid(&value));
    }
}
