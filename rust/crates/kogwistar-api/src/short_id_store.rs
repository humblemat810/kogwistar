//! Shared persistence for Python-compatible short-ID maps.
//!
//! Python owns ContextVar selection; this module accepts only the opaque
//! 32-hex-character map key derived at authenticated ingress.

use fs2::FileExt;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, thiserror::Error)]
pub enum ShortIdStoreError {
    #[error("invalid short-ID map key")]
    InvalidMapKey,
    #[error("invalid short-ID state: {0}")]
    InvalidState(String),
    #[error("short-ID persistence failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("short-ID JSON is invalid: {0}")]
    Json(#[from] serde_json::Error),
    #[error("short-ID transform failed: {0}")]
    Transform(String),
}

/// Match Python's `_file()` key derivation without retaining the bearer token.
pub fn short_id_map_key(run_id: &str) -> String {
    let digest = Sha256::digest(run_id.as_bytes());
    digest[..16]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

pub fn short_id_map_key_for_bearer(token: Option<&str>) -> String {
    short_id_map_key(token.unwrap_or("anonymous"))
}

pub fn short_id_for(
    root: impl AsRef<Path>,
    map_key: &str,
    canonical_id: &str,
) -> Result<String, ShortIdStoreError> {
    validate_map_key(map_key)?;
    let root = root.as_ref();
    fs::create_dir_all(root)?;
    let state_path = root.join(format!("{map_key}.json"));
    let lock_path = root.join(format!("{map_key}.lock"));
    let lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(lock_path)?;
    lock.lock_exclusive()?;

    let mut state = read_state(&state_path)?;
    let input = json!({
        "state": state,
        "input": canonical_id,
        "direction": "l2s",
        "depth": 0,
        "scalar_keys": ["id", "doc_id", "node_id", "edge_id", "edge_endpoint_id"],
        "list_keys": ["source_ids", "target_ids", "source_edge_ids", "target_edge_ids"],
        "primitive": true
    });
    let transformed = kogwistar_contracts::short_id_transform_from_str(&input.to_string())
        .map_err(|error| ShortIdStoreError::Transform(error.to_string()))?;
    let result: Value = serde_json::from_str(&transformed)?;
    state = result
        .get("state")
        .cloned()
        .ok_or_else(|| ShortIdStoreError::InvalidState("missing state".to_owned()))?;
    validate_state(&state)?;
    let short_id = result
        .get("value")
        .and_then(Value::as_str)
        .ok_or_else(|| ShortIdStoreError::InvalidState("missing string value".to_owned()))?
        .to_owned();
    write_state_atomic(&state_path, &state)?;
    FileExt::unlock(&lock)?;
    Ok(short_id)
}

fn validate_map_key(map_key: &str) -> Result<(), ShortIdStoreError> {
    if map_key.len() != 32
        || !map_key
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(ShortIdStoreError::InvalidMapKey);
    }
    Ok(())
}

fn read_state(path: &Path) -> Result<Value, ShortIdStoreError> {
    if !path.exists() {
        return Ok(json!({"next": 1, "l2s": {}, "s2l": {}}));
    }
    let mut contents = String::new();
    File::open(path)?.read_to_string(&mut contents)?;
    let value: Value = serde_json::from_str(&contents)?;
    validate_state(&value)?;
    Ok(value)
}

fn validate_state(value: &Value) -> Result<(), ShortIdStoreError> {
    let valid_map = |map: Option<&Value>| {
        map.and_then(Value::as_object)
            .is_some_and(|object| object.values().all(Value::is_string))
    };
    if value.get("next").and_then(Value::as_u64).is_none()
        || !valid_map(value.get("l2s"))
        || !valid_map(value.get("s2l"))
    {
        return Err(ShortIdStoreError::InvalidState(
            "expected next/l2s/s2l mapping".to_owned(),
        ));
    }
    Ok(())
}

fn write_state_atomic(path: &Path, state: &Value) -> Result<(), ShortIdStoreError> {
    let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let mut temporary_name = path
        .file_name()
        .ok_or_else(|| ShortIdStoreError::InvalidState("state path has no filename".to_owned()))?
        .to_os_string();
    temporary_name.push(format!(".{}.{}.tmp", std::process::id(), sequence));
    let temporary = path.with_file_name(temporary_name);
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temporary)?;
    file.write_all(serde_json::to_string(state)?.as_bytes())?;
    file.sync_all()?;
    drop(file);
    replace_file(&temporary, path)?;
    sync_parent(path)?;
    Ok(())
}

#[cfg(not(windows))]
fn sync_parent(path: &Path) -> Result<(), ShortIdStoreError> {
    if let Some(parent) = path.parent() {
        File::open(parent)?.sync_all()?;
    }
    Ok(())
}

#[cfg(windows)]
fn sync_parent(_path: &Path) -> Result<(), ShortIdStoreError> {
    Ok(())
}

#[cfg(not(windows))]
fn replace_file(source: &Path, destination: &Path) -> Result<(), ShortIdStoreError> {
    fs::rename(source, destination)?;
    Ok(())
}

#[cfg(windows)]
fn replace_file(source: &Path, destination: &Path) -> Result<(), ShortIdStoreError> {
    use std::os::windows::ffi::OsStrExt;
    use winapi::um::winbase::{MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH, MoveFileExW};

    let source: Vec<u16> = source.as_os_str().encode_wide().chain(Some(0)).collect();
    let destination: Vec<u16> = destination
        .as_os_str()
        .encode_wide()
        .chain(Some(0))
        .collect();
    let ok = unsafe {
        MoveFileExW(
            source.as_ptr(),
            destination.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    };
    if ok == 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{short_id_for, short_id_map_key};
    use serde_json::Value;
    use std::path::PathBuf;

    fn test_directory() -> PathBuf {
        let path =
            std::env::temp_dir().join(format!("kogwistar-shortids-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&path).expect("create temporary directory");
        path
    }

    #[test]
    fn shared_python_state_fixture_continues_with_the_same_alias_sequence() {
        let directory = test_directory();
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/short-id-state-v1.json"
        )))
        .expect("shared short-ID fixture is valid");
        let token = fixture["run_id"].as_str().expect("run ID");
        let key = short_id_map_key(token);
        assert_eq!(key, fixture["file_key"].as_str().expect("file key"));
        std::fs::write(
            directory.join(format!("{key}.json")),
            fixture["state"].to_string(),
        )
        .expect("seed shared state");
        let next = short_id_for(&directory, &key, "node-c").expect("next short id");
        let repeated = short_id_for(&directory, &key, "node-a").expect("stable id");

        assert_eq!(next, "<sid>3");
        assert_eq!(repeated, "<sid>1");
        let state: Value = serde_json::from_slice(
            &std::fs::read(directory.join(format!("{key}.json"))).expect("persisted state"),
        )
        .expect("valid JSON");
        assert_eq!(state["l2s"]["node-c"], next);
        assert_eq!(state["s2l"]["<sid>2"], "node-b");
        std::fs::remove_dir_all(directory).expect("remove test directory");
    }

    #[test]
    fn malformed_map_key_is_rejected_before_io() {
        let directory = test_directory();
        assert!(short_id_for(&directory, "../escape", "node").is_err());
        assert_eq!(std::fs::read_dir(&directory).unwrap().count(), 0);
        std::fs::remove_dir_all(directory).expect("remove test directory");
    }
}
