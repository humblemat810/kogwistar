//! Canonical latest-entity revision and fingerprint shared with Python.

use blake2::{
    Blake2b,
    digest::{Digest, consts::U16},
};
use kogwistar_contracts::canonical_json;
use kogwistar_store::{EntityEvent, StoreError, StoreResult};
use serde_json::{Value, json};

type Blake2b128 = Blake2b<U16>;

#[derive(Clone, Debug, PartialEq)]
pub struct CanonicalEntityRevision {
    pub namespace: String,
    pub entity_kind: String,
    pub entity_id: String,
    pub revision: i64,
    pub state: String,
    pub payload: Value,
}

impl CanonicalEntityRevision {
    pub fn is_deleted(&self) -> bool {
        self.state == "deleted"
    }
}

/// Fold one entity's event history using the same operation semantics as Python.
/// Malformed payloads and unknown operations fail closed for projection fencing.
pub fn read_canonical_entity_revision(
    events: &[EntityEvent],
    namespace: &str,
    entity_kind: &str,
    entity_id: &str,
) -> StoreResult<Option<CanonicalEntityRevision>> {
    let mut latest = None;
    for event in events.iter().filter(|event| {
        event.namespace == namespace
            && event.entity_kind == entity_kind
            && event.entity_id == entity_id
    }) {
        let Value::Object(payload) = &event.payload else {
            return Err(invalid(format!(
                "canonical event payload must be an object for {entity_kind}:{entity_id} at {}",
                event.seq
            )));
        };
        let state = match event.op.to_ascii_uppercase().as_str() {
            "ADD" | "REPLACE" => {
                let tombstoned = payload
                    .get("lifecycle_patch")
                    .and_then(Value::as_object)
                    .and_then(|patch| patch.get("lifecycle_status"))
                    .and_then(Value::as_str)
                    == Some("tombstoned");
                if tombstoned { "deleted" } else { "active" }
            }
            "TOMBSTONE" | "DELETE" => "deleted",
            _ => {
                return Err(StoreError::UnsupportedEntityEventOperation {
                    op: event.op.clone(),
                });
            }
        };
        latest = Some(CanonicalEntityRevision {
            namespace: namespace.to_owned(),
            entity_kind: entity_kind.to_owned(),
            entity_id: entity_id.to_owned(),
            revision: event.seq,
            state: state.to_owned(),
            payload: event.payload.clone(),
        });
    }
    Ok(latest)
}

/// Return the version-1 queue payload used to fence derived projections.
pub fn canonical_revision_job_payload(revision: &CanonicalEntityRevision) -> String {
    let source = json!({
        "entity_kind": revision.entity_kind,
        "entity_id": revision.entity_id,
        "state": revision.state,
        "revision": revision.revision,
        "payload": revision.payload,
    });
    let source_json = canonical_json(&source, true);
    let fingerprint = hex::encode(Blake2b128::digest(source_json.as_bytes()));
    canonical_json(
        &json!({
            "source_fingerprint": fingerprint,
            "source_state": revision.state,
            "canonical_revision": revision.revision,
        }),
        true,
    )
}

fn invalid(message: impl Into<String>) -> StoreError {
    StoreError::InvalidEntityEventPayload {
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::{canonical_revision_job_payload, read_canonical_entity_revision};
    use kogwistar_store::EntityEvent;
    use serde_json::Value;

    #[test]
    fn canonical_revision_matches_shared_python_golden() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../../contracts/golden/canonical-entity-revision-v1.json"
        ))
        .unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let namespace = case["namespace"].as_str().unwrap();
            let entity_kind = case["entity_kind"].as_str().unwrap();
            let entity_id = case["entity_id"].as_str().unwrap();
            let events = case["events"]
                .as_array()
                .unwrap()
                .iter()
                .map(|row| EntityEvent {
                    namespace: row["namespace"].as_str().unwrap().to_owned(),
                    seq: row["seq"].as_i64().unwrap(),
                    event_id: row["event_id"].as_str().unwrap().to_owned(),
                    entity_kind: row["entity_kind"].as_str().unwrap().to_owned(),
                    entity_id: row["entity_id"].as_str().unwrap().to_owned(),
                    op: row["op"].as_str().unwrap().to_owned(),
                    payload: row["payload"].clone(),
                })
                .collect::<Vec<_>>();
            let revision =
                read_canonical_entity_revision(&events, namespace, entity_kind, entity_id)
                    .unwrap()
                    .unwrap();
            assert_eq!(
                revision.revision,
                case["expected"]["revision"].as_i64().unwrap()
            );
            assert_eq!(revision.state, case["expected"]["state"].as_str().unwrap());
            assert_eq!(revision.payload, case["expected"]["payload"]);
            assert_eq!(
                canonical_revision_job_payload(&revision),
                case["expected"]["job_payload"].as_str().unwrap()
            );
        }
    }

    #[test]
    fn canonical_revision_fails_closed_on_unknown_operation() {
        let event = EntityEvent {
            namespace: "ns".to_owned(),
            seq: 1,
            event_id: "event-1".to_owned(),
            entity_kind: "node".to_owned(),
            entity_id: "n1".to_owned(),
            op: "MERGE".to_owned(),
            payload: serde_json::json!({"id":"n1"}),
        };
        assert!(read_canonical_entity_revision(&[event], "ns", "node", "n1").is_err());
    }
}
