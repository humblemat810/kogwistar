//! Pure deterministic entity-event recovery reducer.
//!
//! Stores own transactionality. This crate owns only validation, folding, and
//! canonical representation so SQLite and PostgreSQL have one replay meaning.

use kogwistar_contracts::canonical_json;
use kogwistar_store::{EntityEvent, StoreError, StoreResult};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

const STATE_VERSION: i64 = 1;

/// Reducer state kept inside a named projection payload. It has a stable
/// public JSON form so Python consumers can inspect unknown entity kinds too.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EntityProjection {
    entities: BTreeMap<String, Value>,
    last_seq: i64,
}

impl EntityProjection {
    pub fn empty() -> Self {
        Self::default()
    }

    /// Parse a projection previously produced by this reducer. Missing payload
    /// means an empty disposable projection. Other shapes are hard failures:
    /// silently accepting them would let cursor advance past lost state.
    pub fn from_payload(payload: Option<&Map<String, Value>>) -> StoreResult<Self> {
        let Some(payload) = payload else {
            return Ok(Self::empty());
        };
        if payload.get("reducer").and_then(Value::as_str) != Some("entity_event_v1")
            || payload.get("version").and_then(Value::as_i64) != Some(STATE_VERSION)
        {
            return Err(invalid("projection payload is not entity_event_v1"));
        }
        let last_seq = payload
            .get("last_seq")
            .and_then(Value::as_i64)
            .ok_or_else(|| invalid("projection payload last_seq must be integer"))?;
        let entity_values = payload
            .get("entities")
            .and_then(Value::as_object)
            .ok_or_else(|| invalid("projection payload entities must be object"))?;
        let mut entities = BTreeMap::new();
        for (identity, entity) in entity_values {
            validate_state_entity(identity, entity)?;
            entities.insert(identity.clone(), entity.clone());
        }
        Ok(Self { entities, last_seq })
    }

    pub fn fold(&mut self, event: &EntityEvent) -> StoreResult<()> {
        if event.seq <= self.last_seq {
            return Err(invalid("event sequence must be strictly increasing"));
        }
        let key = entity_key(&event.entity_kind, &event.entity_id);
        let op = event.op.to_ascii_uppercase();
        match op.as_str() {
            "ADD" | "REPLACE" | "UPSERT" => {
                let entity = authoritative_entity(event)?;
                self.entities.insert(key, entity);
            }
            "TOMBSTONE" | "DELETE" | "REMOVE" => {
                // Preserve an explicit state record. A sparse delete must be
                // consumer-visible; only a later authoritative replace clears it.
                self.entities.insert(
                    key,
                    json!({
                        "deleted": true,
                        "entity_id": event.entity_id,
                        "entity_kind": event.entity_kind,
                        "event_id": event.event_id,
                        "op": op,
                        "seq": event.seq,
                    }),
                );
            }
            _ => {
                return Err(StoreError::UnsupportedEntityEventOperation {
                    op: event.op.clone(),
                });
            }
        }
        self.last_seq = event.seq;
        Ok(())
    }

    pub fn payload(&self) -> Map<String, Value> {
        Map::from_iter([
            (
                "entities".to_owned(),
                Value::Object(Map::from_iter(self.entities.clone())),
            ),
            ("last_seq".to_owned(), Value::from(self.last_seq)),
            (
                "reducer".to_owned(),
                Value::String("entity_event_v1".to_owned()),
            ),
            ("version".to_owned(), Value::from(STATE_VERSION)),
        ])
    }

    pub const fn last_seq(&self) -> i64 {
        self.last_seq
    }

    pub fn canonical_payload(&self) -> String {
        canonical_json(&Value::Object(self.payload()), true)
    }

    pub fn digest(&self) -> String {
        hex::encode(Sha256::digest(self.canonical_payload().as_bytes()))
    }
}

fn invalid(message: impl Into<String>) -> StoreError {
    StoreError::InvalidEntityEventPayload {
        message: message.into(),
    }
}

fn entity_key(kind: &str, id: &str) -> String {
    // Stable, JSON-object-safe identity without an unprintable separator.
    // Canonical JSON makes this collision-free for arbitrary Unicode values.
    canonical_json(
        &Value::Array(vec![
            Value::String(kind.to_owned()),
            Value::String(id.to_owned()),
        ]),
        false,
    )
}

fn authoritative_entity(event: &EntityEvent) -> StoreResult<Value> {
    let entity = match &event.payload {
        Value::Object(payload) => payload
            .get("entity")
            .or_else(|| payload.get("replacement"))
            .unwrap_or(&event.payload),
        _ => return Err(invalid("ADD/REPLACE payload must be JSON object")),
    };
    let Value::Object(entity) = entity else {
        return Err(invalid("ADD/REPLACE payload entity must be JSON object"));
    };
    let id = entity
        .get("id")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid("ADD/REPLACE payload entity id must be string"))?;
    if id != event.entity_id {
        return Err(invalid(format!(
            "ADD/REPLACE payload id {id:?} conflicts with entity_id {:?}",
            event.entity_id
        )));
    }
    Ok(json!({
        "deleted": false,
        "entity": entity,
        "entity_id": event.entity_id,
        "entity_kind": event.entity_kind,
        "event_id": event.event_id,
        "op": event.op.to_ascii_uppercase(),
        "seq": event.seq,
    }))
}

fn validate_state_entity(key: &str, value: &Value) -> StoreResult<()> {
    if serde_json::from_str::<Vec<String>>(key)
        .ok()
        .is_none_or(|parts| parts.len() != 2)
    {
        return Err(invalid("projection entity key invalid"));
    }
    let Value::Object(entity) = value else {
        return Err(invalid("projection entity state must be object"));
    };
    for field in [
        "deleted",
        "entity_id",
        "entity_kind",
        "event_id",
        "op",
        "seq",
    ] {
        if !entity.contains_key(field) {
            return Err(invalid(format!("projection entity state missing {field}")));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use kogwistar_store::EntityEvent;

    fn event(seq: i64, op: &str, id: &str, payload: Value) -> EntityEvent {
        EntityEvent {
            namespace: "ns".to_owned(),
            seq,
            event_id: format!("event-{seq}"),
            entity_kind: "mystery".to_owned(),
            entity_id: id.to_owned(),
            op: op.to_owned(),
            payload,
        }
    }

    #[test]
    fn replacement_delete_and_replacement_are_explicit_and_stable() {
        let mut projection = EntityProjection::empty();
        projection
            .fold(&event(1, "ADD", "x", json!({"id":"x","v":1})))
            .unwrap();
        projection
            .fold(&event(2, "REPLACE", "x", json!({"id":"x","v":2})))
            .unwrap();
        projection
            .fold(&event(3, "TOMBSTONE", "x", json!({"why":"gone"})))
            .unwrap();
        let tombstone = projection.payload()["entities"][r#"["mystery","x"]"#].clone();
        assert_eq!(tombstone["deleted"], true);
        projection
            .fold(&event(4, "REPLACE", "x", json!({"id":"x","v":3})))
            .unwrap();
        assert_eq!(
            projection.payload()["entities"][r#"["mystery","x"]"#]["entity"]["v"],
            3
        );
        assert_eq!(
            projection.canonical_payload(),
            EntityProjection::from_payload(Some(&projection.payload()))
                .unwrap()
                .canonical_payload()
        );
    }

    #[test]
    fn conflicting_payload_id_rejected_before_cursor_can_move() {
        let mut projection = EntityProjection::empty();
        let error = projection
            .fold(&event(1, "ADD", "x", json!({"id":"other"})))
            .unwrap_err();
        assert!(matches!(
            error,
            StoreError::InvalidEntityEventPayload { .. }
        ));
        assert_eq!(projection.payload()["last_seq"], 0);
    }
}
