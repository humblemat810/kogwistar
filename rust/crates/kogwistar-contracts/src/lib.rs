use serde::de::{MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::VecDeque;
use std::fmt::Write as _;
use std::fmt::{self, Formatter};
use thiserror::Error;
use uuid::Uuid;

pub const CONTRACT_VERSION: &str = "1.0.0";

#[derive(Debug, Error)]
pub enum ContractError {
    #[error("invalid JSON contract payload: {0}")]
    InvalidJson(#[from] serde_json::Error),
    #[error("evidence-pack digest must be a JSON object")]
    EvidenceDigestMustBeObject,
    #[error("metadata-filter request must be a JSON object")]
    MetadataFilterRequestMustBeObject,
    #[error("metadata-filter request requires metadata")]
    MetadataFilterMetadataMissing,
    #[error("metadata-filter metadata must be a JSON object")]
    MetadataFilterMetadataMustBeObject,
    #[error("where must be a dict or None, got {actual}")]
    MetadataFilterWhereType { actual: &'static str },
    #[error("logical operator {operator:?} requires an iterable clause list, got {actual}")]
    MetadataFilterLogicalClausesNotIterable {
        operator: String,
        actual: &'static str,
    },
    #[error("Unsupported where operator: {operator:?}")]
    MetadataFilterUnsupportedOperator { operator: String },
    #[error("short-id request must be a JSON object")]
    ShortIdRequestMustBeObject,
    #[error("short-id request has an invalid state")]
    ShortIdInvalidState,
    #[error("Only <sid>… is accepted in id fields.")]
    ShortIdInvalidInput,
    #[error("Unknown short id '{short_id}' for this run.")]
    ShortIdUnknown { short_id: String },
    #[error("state-update request must be a JSON object")]
    StateUpdateRequestMustBeObject,
    #[error("state-update state must be a JSON object")]
    StateUpdateStateMustBeObject,
    #[error("state-update item must be a two-item array")]
    StateUpdateItemMustBePair,
    #[error("state-update payload must be a JSON object")]
    StateUpdatePayloadMustBeObject,
    #[error("state-update append target must be a list")]
    StateUpdateTargetMustBeList,
    #[error("Either update or state_update can be used")]
    StateUpdateConflict,
    #[error("event envelope must be a JSON object")]
    EventEnvelopeMustBeObject,
    #[error("event envelope requires string type")]
    EventTypeMissing,
    #[error("unsupported entity event type: {event_type}")]
    EventTypeUnsupported { event_type: String },
    #[error("event envelope requires a JSON object entity")]
    EventEntityMustBeObject,
    #[error("event envelope entity requires string id")]
    EventEntityIdMissing,
    #[error("event envelope requires positive integer event_seq")]
    EventSeqInvalid,
    #[error("event replay sequence must be strictly increasing")]
    EventSequenceOrder,
    #[error("event replay request requires an events array")]
    EventReplayEventsMissing,
    #[error("workflow topology request must be a JSON object")]
    WorkflowTopologyRequestMustBeObject,
    #[error("workflow topology request requires string array {field}")]
    WorkflowTopologyStringArray { field: &'static str },
    #[error("workflow topology request requires edge pairs")]
    WorkflowTopologyEdgesInvalid,
    #[error("workflow topology request requires string {field}")]
    WorkflowTopologyString { field: &'static str },
}

impl ContractError {
    /// Stable identifier for callers that must not parse human-readable errors.
    pub const fn code(&self) -> &'static str {
        match self {
            Self::InvalidJson(_) => "KOGWISTAR_CONTRACT_INVALID_JSON",
            Self::EvidenceDigestMustBeObject => "KOGWISTAR_CONTRACT_EVIDENCE_DIGEST_OBJECT",
            Self::MetadataFilterRequestMustBeObject => {
                "KOGWISTAR_CONTRACT_METADATA_FILTER_REQUEST_TYPE"
            }
            Self::MetadataFilterMetadataMissing => {
                "KOGWISTAR_CONTRACT_METADATA_FILTER_METADATA_MISSING"
            }
            Self::MetadataFilterMetadataMustBeObject => {
                "KOGWISTAR_CONTRACT_METADATA_FILTER_METADATA_TYPE"
            }
            Self::MetadataFilterWhereType { .. } => "KOGWISTAR_CONTRACT_METADATA_FILTER_WHERE_TYPE",
            Self::MetadataFilterLogicalClausesNotIterable { .. } => {
                "KOGWISTAR_CONTRACT_METADATA_FILTER_LOGICAL_CLAUSES_TYPE"
            }
            Self::MetadataFilterUnsupportedOperator { .. } => {
                "KOGWISTAR_CONTRACT_METADATA_FILTER_UNSUPPORTED_OPERATOR"
            }
            Self::ShortIdRequestMustBeObject => "KOGWISTAR_CONTRACT_SHORT_ID_REQUEST_TYPE",
            Self::ShortIdInvalidState => "KOGWISTAR_CONTRACT_SHORT_ID_STATE_INVALID",
            Self::ShortIdInvalidInput => "KOGWISTAR_CONTRACT_SHORT_ID_INVALID",
            Self::ShortIdUnknown { .. } => "KOGWISTAR_CONTRACT_SHORT_ID_UNKNOWN",
            Self::StateUpdateRequestMustBeObject => "KOGWISTAR_CONTRACT_STATE_UPDATE_REQUEST_TYPE",
            Self::StateUpdateStateMustBeObject => "KOGWISTAR_CONTRACT_STATE_UPDATE_STATE_TYPE",
            Self::StateUpdateItemMustBePair => "KOGWISTAR_CONTRACT_STATE_UPDATE_ITEM_TYPE",
            Self::StateUpdatePayloadMustBeObject => "KOGWISTAR_CONTRACT_STATE_UPDATE_PAYLOAD_TYPE",
            Self::StateUpdateTargetMustBeList => "KOGWISTAR_CONTRACT_STATE_UPDATE_TARGET_TYPE",
            Self::StateUpdateConflict => "KOGWISTAR_CONTRACT_STATE_UPDATE_CONFLICT",
            Self::EventEnvelopeMustBeObject => "KOGWISTAR_CONTRACT_EVENT_ENVELOPE_TYPE",
            Self::EventTypeMissing => "KOGWISTAR_CONTRACT_EVENT_TYPE_MISSING",
            Self::EventTypeUnsupported { .. } => "KOGWISTAR_CONTRACT_EVENT_TYPE_UNSUPPORTED",
            Self::EventEntityMustBeObject => "KOGWISTAR_CONTRACT_EVENT_ENTITY_TYPE",
            Self::EventEntityIdMissing => "KOGWISTAR_CONTRACT_EVENT_ENTITY_ID_MISSING",
            Self::EventSeqInvalid => "KOGWISTAR_CONTRACT_EVENT_SEQ_INVALID",
            Self::EventSequenceOrder => "KOGWISTAR_CONTRACT_EVENT_SEQUENCE_ORDER",
            Self::EventReplayEventsMissing => "KOGWISTAR_CONTRACT_EVENT_REPLAY_EVENTS_MISSING",
            Self::WorkflowTopologyRequestMustBeObject => {
                "KOGWISTAR_CONTRACT_WORKFLOW_TOPOLOGY_REQUEST_TYPE"
            }
            Self::WorkflowTopologyStringArray { .. } => {
                "KOGWISTAR_CONTRACT_WORKFLOW_TOPOLOGY_STRING_ARRAY"
            }
            Self::WorkflowTopologyEdgesInvalid => "KOGWISTAR_CONTRACT_WORKFLOW_TOPOLOGY_EDGES_TYPE",
            Self::WorkflowTopologyString { .. } => {
                "KOGWISTAR_CONTRACT_WORKFLOW_TOPOLOGY_STRING_TYPE"
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EntityEventEnvelope {
    pub namespace: String,
    pub seq: i64,
    pub event_id: String,
    pub entity_kind: String,
    pub entity_id: String,
    pub op: String,
    pub payload: Value,
}

fn json_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(value) => value.as_f64().is_some_and(|value| value != 0.0),
        Value::String(value) => !value.is_empty(),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
    }
}

fn short_id(value: &str) -> bool {
    value.strip_prefix("<sid>").is_some_and(|suffix| {
        !suffix.is_empty() && suffix.bytes().all(|byte| byte.is_ascii_digit())
    })
}

fn short_id_keys(value: Option<&Value>) -> Result<Vec<String>, ContractError> {
    let Some(Value::Array(values)) = value else {
        return Err(ContractError::ShortIdInvalidState);
    };
    values
        .iter()
        .map(|value| match value {
            Value::String(value) => Ok(value.clone()),
            _ => Err(ContractError::ShortIdInvalidState),
        })
        .collect()
}

type ShortIdMaps<'a> = (&'a Map<String, Value>, &'a Map<String, Value>, u64);

fn short_id_state_maps(state: &Value) -> Result<ShortIdMaps<'_>, ContractError> {
    let Value::Object(state) = state else {
        return Err(ContractError::ShortIdInvalidState);
    };
    let (Some(Value::Object(l2s)), Some(Value::Object(s2l)), Some(next)) =
        (state.get("l2s"), state.get("s2l"), state.get("next"))
    else {
        return Err(ContractError::ShortIdInvalidState);
    };
    let Some(next) = next.as_u64() else {
        return Err(ContractError::ShortIdInvalidState);
    };
    Ok((l2s, s2l, next))
}

fn short_id_allocate(state: &mut Value, long_id: &str) -> Result<String, ContractError> {
    let (existing, next) = {
        let (l2s, _, next) = short_id_state_maps(state)?;
        let existing = l2s.get(long_id).and_then(Value::as_str).map(str::to_owned);
        (existing, next)
    };
    if let Some(existing) = existing {
        return Ok(existing);
    }
    let sid = format!("<sid>{next}");
    let Value::Object(state) = state else {
        return Err(ContractError::ShortIdInvalidState);
    };
    let Some(Value::Object(l2s)) = state.get_mut("l2s") else {
        return Err(ContractError::ShortIdInvalidState);
    };
    l2s.insert(long_id.to_owned(), Value::String(sid.clone()));
    let Some(Value::Object(s2l)) = state.get_mut("s2l") else {
        return Err(ContractError::ShortIdInvalidState);
    };
    s2l.insert(sid.clone(), Value::String(long_id.to_owned()));
    state.insert("next".to_owned(), Value::from(next + 1));
    Ok(sid)
}

fn short_id_to_long(state: &Value, value: &str) -> Result<String, ContractError> {
    if !short_id(value) {
        return Err(ContractError::ShortIdInvalidInput);
    }
    let (_, s2l, _) = short_id_state_maps(state)?;
    s2l.get(value)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| ContractError::ShortIdUnknown {
            short_id: value.to_owned(),
        })
}

fn short_id_value(value: Value, state: &mut Value, to_short: bool) -> Result<Value, ContractError> {
    match value {
        Value::String(value) if to_short => Ok(Value::String(if short_id(&value) {
            value
        } else {
            short_id_allocate(state, &value)?
        })),
        Value::String(value) => Ok(Value::String(short_id_to_long(state, &value)?)),
        Value::Array(values) => values
            .into_iter()
            .map(|value| short_id_value(value, state, to_short))
            .collect::<Result<Vec<_>, _>>()
            .map(Value::Array),
        value => Ok(value),
    }
}

fn short_id_walk(
    value: Value,
    state: &mut Value,
    depth: i64,
    scalar_keys: &[String],
    list_keys: &[String],
    to_short: bool,
) -> Result<Value, ContractError> {
    if depth < 0 {
        return Ok(value);
    }
    match value {
        Value::Object(values) => {
            let mut output = Map::new();
            for (key, value) in values {
                let value = if scalar_keys.iter().any(|candidate| candidate == &key) {
                    short_id_value(value, state, to_short)?
                } else if list_keys.iter().any(|candidate| candidate == &key) {
                    match value {
                        Value::Array(values) => Value::Array(
                            values
                                .into_iter()
                                .map(|value| short_id_value(value, state, to_short))
                                .collect::<Result<Vec<_>, _>>()?,
                        ),
                        value => value,
                    }
                } else if depth > 0 {
                    short_id_walk(value, state, depth - 1, scalar_keys, list_keys, to_short)?
                } else {
                    value
                };
                output.insert(key, value);
            }
            Ok(Value::Object(output))
        }
        Value::Array(values) if depth > 0 => values
            .into_iter()
            .map(|value| short_id_walk(value, state, depth, scalar_keys, list_keys, to_short))
            .collect::<Result<Vec<_>, _>>()
            .map(Value::Array),
        value => Ok(value),
    }
}

/// Transform JSON-compatible short IDs. Persistence and ContextVars remain Python-owned.
pub fn short_id_transform_from_str(payload_json: &str) -> Result<String, ContractError> {
    let Value::Object(request) = serde_json::from_str::<Value>(payload_json)? else {
        return Err(ContractError::ShortIdRequestMustBeObject);
    };
    let mut state = request
        .get("state")
        .cloned()
        .ok_or(ContractError::ShortIdInvalidState)?;
    short_id_state_maps(&state)?;
    let input = request
        .get("input")
        .cloned()
        .ok_or(ContractError::ShortIdRequestMustBeObject)?;
    let direction = request
        .get("direction")
        .and_then(Value::as_str)
        .ok_or(ContractError::ShortIdRequestMustBeObject)?;
    let to_short = match direction {
        "l2s" => true,
        "s2l" => false,
        _ => return Err(ContractError::ShortIdRequestMustBeObject),
    };
    let depth = request
        .get("depth")
        .and_then(Value::as_i64)
        .ok_or(ContractError::ShortIdRequestMustBeObject)?;
    let scalar_keys = short_id_keys(request.get("scalar_keys"))?;
    let list_keys = short_id_keys(request.get("list_keys"))?;
    let value = if request
        .get("primitive")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        short_id_value(input, &mut state, to_short)?
    } else {
        short_id_walk(input, &mut state, depth, &scalar_keys, &list_keys, to_short)?
    };
    Ok(canonical_json(
        &Value::Object(Map::from_iter([
            ("state".to_owned(), state),
            ("value".to_owned(), value),
        ])),
        false,
    ))
}

fn state_object_mut(value: &mut Value) -> Result<&mut Map<String, Value>, ContractError> {
    value
        .as_object_mut()
        .ok_or(ContractError::StateUpdateStateMustBeObject)
}

fn extend_json_list(target: &mut Vec<Value>, value: Value) -> Result<(), ContractError> {
    match value {
        Value::Array(values) => target.extend(values),
        Value::String(value) => {
            target.extend(value.chars().map(|value| Value::String(value.to_string())))
        }
        Value::Object(values) => {
            target.extend(values.into_iter().map(|(key, _)| Value::String(key)))
        }
        _ => return Err(ContractError::StateUpdatePayloadMustBeObject),
    }
    Ok(())
}

fn state_list_mut<'a>(
    state: &'a mut Map<String, Value>,
    key: &str,
) -> Result<&'a mut Vec<Value>, ContractError> {
    if !state.contains_key(key) {
        state.insert(key.to_owned(), Value::Array(Vec::new()));
    }
    state
        .get_mut(key)
        .and_then(Value::as_array_mut)
        .ok_or(ContractError::StateUpdateTargetMustBeList)
}

fn state_update_entries(value: &Value) -> Result<&Vec<Value>, ContractError> {
    value
        .as_array()
        .ok_or(ContractError::StateUpdateItemMustBePair)
}

/// Apply the JSON subset of the Python runtime's `u`/`a`/`e` state reducer.
/// The bridge keeps Python ownership for non-JSON values and preserves in-place
/// mutation by updating the caller's existing dict after this fold succeeds.
pub fn apply_state_update_from_str(payload_json: &str) -> Result<String, ContractError> {
    let Value::Object(request) = serde_json::from_str::<Value>(payload_json)? else {
        return Err(ContractError::StateUpdateRequestMustBeObject);
    };
    let mut state = request
        .get("state")
        .cloned()
        .ok_or(ContractError::StateUpdateStateMustBeObject)?;
    state_object_mut(&mut state)?;
    let state_update = request
        .get("state_update")
        .ok_or(ContractError::StateUpdateItemMustBePair)?;
    let update = request.get("update").cloned().unwrap_or(Value::Null);
    if json_truthy(&update) && json_truthy(state_update) {
        return Err(ContractError::StateUpdateConflict);
    }
    for item in state_update_entries(state_update)? {
        let Some(pair) = item.as_array() else {
            return Err(ContractError::StateUpdateItemMustBePair);
        };
        if pair.len() != 2 {
            return Err(ContractError::StateUpdateItemMustBePair);
        }
        let Some(mode) = pair[0].as_str() else {
            return Err(ContractError::StateUpdateItemMustBePair);
        };
        let Some(payload) = pair[1].as_object() else {
            return Err(ContractError::StateUpdatePayloadMustBeObject);
        };
        match mode {
            "a" => {
                for (key, value) in payload {
                    state_list_mut(state_object_mut(&mut state)?, key)?.push(value.clone());
                }
            }
            "u" => {
                for (key, value) in payload {
                    state_object_mut(&mut state)?.insert(key.clone(), value.clone());
                }
            }
            "e" => {
                for (key, value) in payload {
                    extend_json_list(
                        state_list_mut(state_object_mut(&mut state)?, key)?,
                        value.clone(),
                    )?;
                }
            }
            _ => {}
        }
    }
    if json_truthy(&update) {
        let Some(update) = update.as_object() else {
            return Err(ContractError::StateUpdatePayloadMustBeObject);
        };
        let schema = request.get("state_schema").and_then(Value::as_object);
        for (key, value) in update {
            if schema
                .and_then(|schema| schema.get(key))
                .and_then(Value::as_str)
                == Some("a")
            {
                extend_json_list(
                    state_list_mut(state_object_mut(&mut state)?, key)?,
                    value.clone(),
                )?;
            } else {
                state_object_mut(&mut state)?.insert(key.clone(), value.clone());
            }
        }
    }
    Ok(canonical_json(&state, false))
}

fn normalize_entity_event(value: &Value) -> Result<Value, ContractError> {
    let Value::Object(event) = value else {
        return Err(ContractError::EventEnvelopeMustBeObject);
    };
    let event_type = event
        .get("type")
        .and_then(Value::as_str)
        .ok_or(ContractError::EventTypeMissing)?;
    let normalized_type = match event_type {
        "entity.upsert" => "entity.upsert",
        "entity.tombstone" | "entity.delete" | "entity.remove" => "entity.tombstone",
        value => {
            return Err(ContractError::EventTypeUnsupported {
                event_type: value.to_owned(),
            });
        }
    };
    let entity = event
        .get("entity")
        .and_then(Value::as_object)
        .ok_or(ContractError::EventEntityMustBeObject)?;
    if entity.get("id").and_then(Value::as_str).is_none() {
        return Err(ContractError::EventEntityIdMissing);
    }
    if event
        .get("event_seq")
        .and_then(Value::as_i64)
        .is_none_or(|seq| seq <= 0)
    {
        return Err(ContractError::EventSeqInvalid);
    }
    let mut output = event.clone();
    output.insert("type".to_owned(), Value::String(normalized_type.to_owned()));
    Ok(Value::Object(output))
}

/// Validate and canonicalize one consumer-shaped entity event envelope.
pub fn canonical_entity_event_from_str(payload_json: &str) -> Result<String, ContractError> {
    let value: Value = serde_json::from_str(payload_json)?;
    Ok(canonical_json(&normalize_entity_event(&value)?, false))
}

/// Deterministically fold consumer-shaped entity events for contract tests.
pub fn replay_entity_events_from_str(payload_json: &str) -> Result<String, ContractError> {
    let Value::Object(request) = serde_json::from_str::<Value>(payload_json)? else {
        return Err(ContractError::EventEnvelopeMustBeObject);
    };
    let events = request
        .get("events")
        .and_then(Value::as_array)
        .ok_or(ContractError::EventReplayEventsMissing)?;
    let mut previous_seq = 0_i64;
    let mut active = Map::<String, Value>::new();
    let mut tombstoned = Vec::<String>::new();
    for event in events {
        let event = normalize_entity_event(event)?;
        let event_object = event.as_object().expect("normalized event is object");
        let seq = event_object["event_seq"]
            .as_i64()
            .expect("validated sequence");
        if seq <= previous_seq {
            return Err(ContractError::EventSequenceOrder);
        }
        previous_seq = seq;
        let entity = event_object["entity"]
            .as_object()
            .expect("validated entity");
        let entity_id = entity["id"].as_str().expect("validated entity id");
        if event_object["type"] == Value::String("entity.upsert".to_owned()) {
            active.insert(entity_id.to_owned(), event.clone());
            tombstoned.retain(|value| value != entity_id);
        } else {
            active.remove(entity_id);
            if !tombstoned.iter().any(|value| value == entity_id) {
                tombstoned.push(entity_id.to_owned());
            }
        }
    }
    let mut active_entities: Vec<_> = active.into_iter().map(|(key, _)| key).collect();
    active_entities.sort();
    tombstoned.sort();
    Ok(canonical_json(
        &Value::Object(Map::from_iter([
            (
                "active_entities".to_owned(),
                Value::Array(active_entities.into_iter().map(Value::String).collect()),
            ),
            ("cursor".to_owned(), Value::from(previous_seq)),
            (
                "tombstoned_entities".to_owned(),
                Value::Array(tombstoned.into_iter().map(Value::String).collect()),
            ),
        ])),
        false,
    ))
}

fn push_python_json_string(out: &mut String, value: &str, ensure_ascii: bool) {
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{0008}' => out.push_str("\\b"),
            '\u{000c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if (ch as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", ch as u32);
            }
            ch if ensure_ascii && (ch as u32) > 0x7f && (ch as u32) <= 0xffff => {
                let _ = write!(out, "\\u{:04x}", ch as u32);
            }
            ch if ensure_ascii && (ch as u32) > 0xffff => {
                let scalar = ch as u32 - 0x1_0000;
                let high = 0xd800 + (scalar >> 10);
                let low = 0xdc00 + (scalar & 0x3ff);
                let _ = write!(out, "\\u{high:04x}\\u{low:04x}");
            }
            ch => out.push(ch),
        }
    }
    out.push('"');
}

fn push_python_json(out: &mut String, value: &Value, ensure_ascii: bool) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(value) => out.push_str(if *value { "true" } else { "false" }),
        Value::Number(value) => out.push_str(&value.to_string()),
        Value::String(value) => push_python_json_string(out, value, ensure_ascii),
        Value::Array(values) => {
            out.push('[');
            for (index, value) in values.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                push_python_json(out, value, ensure_ascii);
            }
            out.push(']');
        }
        Value::Object(values) => {
            out.push('{');
            let mut entries: Vec<_> = values.iter().collect();
            entries.sort_by(|(left, _), (right, _)| left.cmp(right));
            for (index, (key, value)) in entries.into_iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                push_python_json_string(out, key, ensure_ascii);
                out.push(':');
                push_python_json(out, value, ensure_ascii);
            }
            out.push('}');
        }
    }
}

pub fn canonical_json(value: &Value, ensure_ascii: bool) -> String {
    let mut out = String::new();
    push_python_json(&mut out, value, ensure_ascii);
    out
}

pub fn canonical_json_from_str(payload_json: &str) -> Result<String, ContractError> {
    let value: Value = serde_json::from_str(payload_json)?;
    Ok(canonical_json(&value, false))
}

pub fn stable_id(kind: &str, parts: &[String]) -> Uuid {
    let mut values = Vec::with_capacity(parts.len() + 1);
    values.push(Value::String(kind.to_owned()));
    values.extend(parts.iter().cloned().map(Value::String));
    stable_id_values(&values)
}

pub fn stable_id_values(values: &[Value]) -> Uuid {
    let key = canonical_json(&Value::Array(values.to_vec()), false);
    let project_namespace = Uuid::new_v5(&Uuid::NAMESPACE_URL, b"graph-knowledge-engine");
    Uuid::new_v5(&project_namespace, key.as_bytes())
}

pub fn stable_id_from_json(payload_json: &str) -> Result<Uuid, ContractError> {
    let values: Vec<Value> = serde_json::from_str(payload_json)?;
    Ok(stable_id_values(&values))
}

fn python_string(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        Value::Null => "None".to_owned(),
        Value::Bool(true) => "True".to_owned(),
        Value::Bool(false) => "False".to_owned(),
        Value::Number(value) => value.to_string(),
        other => canonical_json(other, false),
    }
}

fn normalized_id_array(value: Option<&Value>) -> Vec<Value> {
    let mut ids: Vec<String> = match value {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Array(values)) => values
            .iter()
            .map(python_string)
            .filter(|value| !value.is_empty())
            .collect(),
        Some(value) => {
            let value = python_string(value);
            if value.is_empty() {
                Vec::new()
            } else {
                vec![value]
            }
        }
    };
    ids.sort();
    ids.into_iter().map(Value::String).collect()
}

pub fn canonicalize_evidence_pack_digest(value: &Value) -> Result<Value, ContractError> {
    let Value::Object(input) = value else {
        return Err(ContractError::EvidenceDigestMustBeObject);
    };
    let mut payload: Map<String, Value> = input.clone();
    payload.insert(
        "node_ids".to_owned(),
        Value::Array(normalized_id_array(input.get("node_ids"))),
    );
    payload.insert(
        "edge_ids".to_owned(),
        Value::Array(normalized_id_array(input.get("edge_ids"))),
    );
    payload.remove("evidence_pack_hash");
    Ok(Value::Object(payload))
}

pub fn evidence_pack_digest_hash(value: &Value) -> Result<String, ContractError> {
    let canonical = canonicalize_evidence_pack_digest(value)?;
    let payload = canonical_json(&canonical, true);
    Ok(hex::encode(Sha256::digest(payload.as_bytes())))
}

pub fn evidence_pack_digest_hash_from_str(payload_json: &str) -> Result<String, ContractError> {
    let value: Value = serde_json::from_str(payload_json)?;
    evidence_pack_digest_hash(&value)
}

// `serde_json::Value` stores objects in sorted-key maps by default.  Filter
// evaluation must instead retain Python dict insertion order: an early false or
// invalid operator is observable.  This small JSON tree also lets the evaluator
// implement Python's deliberately permissive JSON-value comparisons directly.
#[derive(Clone, Debug, PartialEq)]
enum FilterValue {
    Null,
    Bool(bool),
    Number(serde_json::Number),
    String(String),
    Array(Vec<FilterValue>),
    Object(Vec<(String, FilterValue)>),
}

impl<'de> Deserialize<'de> for FilterValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct FilterValueVisitor;

        impl<'de> Visitor<'de> for FilterValueVisitor {
            type Value = FilterValue;

            fn expecting(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
                formatter.write_str("a JSON value")
            }

            fn visit_unit<E>(self) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Ok(FilterValue::Null)
            }

            fn visit_none<E>(self) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Ok(FilterValue::Null)
            }

            fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Ok(FilterValue::Bool(value))
            }

            fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Ok(FilterValue::Number(value.into()))
            }

            fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Ok(FilterValue::Number(value.into()))
            }

            fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                serde_json::Number::from_f64(value)
                    .map(FilterValue::Number)
                    .ok_or_else(|| E::custom("JSON number must be finite"))
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Ok(FilterValue::String(value.to_owned()))
            }

            fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Ok(FilterValue::String(value))
            }

            fn visit_seq<A>(self, mut values: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut out = Vec::new();
                while let Some(value) = values.next_element()? {
                    out.push(value);
                }
                Ok(FilterValue::Array(out))
            }

            fn visit_map<A>(self, mut values: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut out: Vec<(String, FilterValue)> = Vec::new();
                while let Some((key, value)) = values.next_entry::<String, FilterValue>()? {
                    // json.loads keeps a duplicate key's original order while
                    // replacing its value; match that behavior.
                    if let Some((_, current)) = out.iter_mut().find(|(name, _)| name == &key) {
                        *current = value;
                    } else {
                        out.push((key, value));
                    }
                }
                Ok(FilterValue::Object(out))
            }
        }

        deserializer.deserialize_any(FilterValueVisitor)
    }
}

fn filter_type_name(value: &FilterValue) -> &'static str {
    match value {
        FilterValue::Null => "NoneType",
        FilterValue::Bool(_) => "bool",
        FilterValue::Number(number) if number.is_i64() || number.is_u64() => "int",
        FilterValue::Number(_) => "float",
        FilterValue::String(_) => "str",
        FilterValue::Array(_) => "list",
        FilterValue::Object(_) => "dict",
    }
}

fn python_truthy(value: &FilterValue) -> bool {
    match value {
        FilterValue::Null => false,
        FilterValue::Bool(value) => *value,
        FilterValue::Number(value) => value.as_f64().is_some_and(|number| number != 0.0),
        FilterValue::String(value) => !value.is_empty(),
        FilterValue::Array(value) => !value.is_empty(),
        FilterValue::Object(value) => !value.is_empty(),
    }
}

fn object_get<'a>(entries: &'a [(String, FilterValue)], key: &str) -> Option<&'a FilterValue> {
    entries
        .iter()
        .find(|(name, _)| name == key)
        .map(|(_, value)| value)
}

fn number_as_f64(number: &serde_json::Number) -> f64 {
    number.as_f64().expect("serde_json number is finite")
}

fn number_equals(left: &serde_json::Number, right: &serde_json::Number) -> bool {
    match (left.as_i64(), left.as_u64(), right.as_i64(), right.as_u64()) {
        (Some(left), _, Some(right), _) => left == right,
        (Some(left), _, _, Some(right)) if left >= 0 => left as u64 == right,
        (_, Some(left), Some(right), _) if right >= 0 => left == right as u64,
        (_, Some(left), _, Some(right)) => left == right,
        _ => number_as_f64(left) == number_as_f64(right),
    }
}

fn number_compare(left: &serde_json::Number, right: &serde_json::Number) -> Ordering {
    match (left.as_i64(), left.as_u64(), right.as_i64(), right.as_u64()) {
        (Some(left), _, Some(right), _) => left.cmp(&right),
        (Some(left), _, _, Some(_right)) if left < 0 => Ordering::Less,
        (Some(left), _, _, Some(right)) => (left as u64).cmp(&right),
        (_, Some(_left), Some(right), _) if right < 0 => Ordering::Greater,
        (_, Some(left), Some(right), _) => left.cmp(&(right as u64)),
        (_, Some(left), _, Some(right)) => left.cmp(&right),
        _ => number_as_f64(left)
            .partial_cmp(&number_as_f64(right))
            .expect("JSON numbers are finite"),
    }
}

fn bool_number_equals(value: bool, number: &serde_json::Number) -> bool {
    number_equals(&serde_json::Number::from(u64::from(value)), number)
}

fn python_equals(left: &FilterValue, right: &FilterValue) -> bool {
    match (left, right) {
        (FilterValue::Null, FilterValue::Null) => true,
        (FilterValue::Bool(left), FilterValue::Bool(right)) => left == right,
        (FilterValue::Bool(left), FilterValue::Number(right)) => bool_number_equals(*left, right),
        (FilterValue::Number(left), FilterValue::Bool(right)) => bool_number_equals(*right, left),
        (FilterValue::Number(left), FilterValue::Number(right)) => number_equals(left, right),
        (FilterValue::String(left), FilterValue::String(right)) => left == right,
        (FilterValue::Array(left), FilterValue::Array(right)) => {
            left.len() == right.len()
                && left
                    .iter()
                    .zip(right)
                    .all(|(left, right)| python_equals(left, right))
        }
        (FilterValue::Object(left), FilterValue::Object(right)) => {
            left.len() == right.len()
                && left.iter().all(|(key, value)| {
                    object_get(right, key).is_some_and(|other| python_equals(value, other))
                })
        }
        _ => false,
    }
}

fn python_compare(left: &FilterValue, right: &FilterValue) -> Option<Ordering> {
    match (left, right) {
        (FilterValue::Bool(left), FilterValue::Bool(right)) => Some(left.cmp(right)),
        (FilterValue::Bool(left), FilterValue::Number(right)) => {
            number_as_f64(&serde_json::Number::from(u64::from(*left)))
                .partial_cmp(&number_as_f64(right))
        }
        (FilterValue::Number(left), FilterValue::Bool(right)) => number_as_f64(left)
            .partial_cmp(&number_as_f64(&serde_json::Number::from(u64::from(*right)))),
        (FilterValue::Number(left), FilterValue::Number(right)) => {
            Some(number_compare(left, right))
        }
        (FilterValue::String(left), FilterValue::String(right)) => Some(left.cmp(right)),
        (FilterValue::Array(left), FilterValue::Array(right)) => {
            for (left_item, right_item) in left.iter().zip(right) {
                let comparison = python_compare(left_item, right_item)?;
                if comparison != Ordering::Equal {
                    return Some(comparison);
                }
            }
            Some(left.len().cmp(&right.len()))
        }
        _ => None,
    }
}

fn filter_python_string(value: &FilterValue) -> String {
    match value {
        FilterValue::Null => "None".to_owned(),
        FilterValue::Bool(true) => "True".to_owned(),
        FilterValue::Bool(false) => "False".to_owned(),
        FilterValue::Number(value) => value.to_string(),
        FilterValue::String(value) => value.clone(),
        FilterValue::Array(values) => format!(
            "[{}]",
            values
                .iter()
                .map(python_repr)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        FilterValue::Object(values) => format!(
            "{{{}}}",
            values
                .iter()
                .map(|(key, value)| format!("{}: {}", python_repr_string(key), python_repr(value)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn python_repr_string(value: &str) -> String {
    format!("'{}'", value.replace('\\', "\\\\").replace('\'', "\\'"))
}

fn python_repr(value: &FilterValue) -> String {
    match value {
        FilterValue::String(value) => python_repr_string(value),
        other => filter_python_string(other),
    }
}

fn is_operator_dict(value: &FilterValue) -> bool {
    matches!(value, FilterValue::Object(entries) if entries.iter().any(|(key, _)| key.starts_with('$')))
}

enum FieldValue<'a> {
    Missing,
    Present(&'a FilterValue),
}

fn sequence_contains(values: &[FilterValue], expected: &FilterValue) -> bool {
    values.iter().any(|value| python_equals(value, expected))
}

fn safe_comparison(left: &FilterValue, right: &FilterValue, operator: &str) -> bool {
    if operator == "$eq" {
        return python_equals(left, right);
    }
    if operator == "$ne" {
        return !python_equals(left, right);
    }
    let Some(comparison) = python_compare(left, right) else {
        return false;
    };
    match operator {
        "$gt" => comparison == Ordering::Greater,
        "$gte" => comparison != Ordering::Less,
        "$lt" => comparison == Ordering::Less,
        "$lte" => comparison != Ordering::Greater,
        _ => false,
    }
}

fn matches_field(value: FieldValue<'_>, condition: &FilterValue) -> Result<bool, ContractError> {
    let value = match value {
        FieldValue::Present(value) => value,
        // Python calls `metadata.get(key)` for direct equality conditions, so a
        // missing key is indistinguishable from a present `None` there. Operator
        // dictionaries use the explicit missing sentinel and always fail.
        FieldValue::Missing if is_operator_dict(condition) => return Ok(false),
        FieldValue::Missing => &FilterValue::Null,
    };
    let FilterValue::Object(operators) = condition else {
        return Ok(python_equals(value, condition));
    };
    if !is_operator_dict(condition) {
        return Ok(python_equals(value, condition));
    }

    for (operator, expected) in operators {
        match operator.as_str() {
            "$in" => {
                let matched = match (value, expected) {
                    (FilterValue::Array(values), FilterValue::Array(expected)) => values
                        .iter()
                        .any(|value| sequence_contains(expected, value)),
                    (_, FilterValue::Array(expected)) => sequence_contains(expected, value),
                    (FilterValue::Array(values), expected) => sequence_contains(values, expected),
                    _ => python_equals(value, expected),
                };
                if !matched {
                    return Ok(false);
                }
            }
            "$nin" => {
                let matched = match (value, expected) {
                    (FilterValue::Array(values), FilterValue::Array(expected)) => values
                        .iter()
                        .any(|value| sequence_contains(expected, value)),
                    (_, FilterValue::Array(expected)) => sequence_contains(expected, value),
                    (FilterValue::Array(values), expected) => sequence_contains(values, expected),
                    _ => python_equals(value, expected),
                };
                if matched {
                    return Ok(false);
                }
            }
            "$contains" => {
                let matched = match value {
                    FilterValue::String(value) => value.contains(&filter_python_string(expected)),
                    FilterValue::Array(values) => sequence_contains(values, expected),
                    _ => false,
                };
                if !matched {
                    return Ok(false);
                }
            }
            "$eq" | "$ne" | "$gt" | "$gte" | "$lt" | "$lte" => {
                if !safe_comparison(value, expected, operator) {
                    return Ok(false);
                }
            }
            _ => {
                return Err(ContractError::MetadataFilterUnsupportedOperator {
                    operator: operator.clone(),
                });
            }
        }
    }
    Ok(true)
}

fn matches_where(
    metadata: &[(String, FilterValue)],
    where_value: &FilterValue,
) -> Result<bool, ContractError> {
    if !python_truthy(where_value) {
        return Ok(true);
    }
    let FilterValue::Object(where_entries) = where_value else {
        return Err(ContractError::MetadataFilterWhereType {
            actual: filter_type_name(where_value),
        });
    };

    for (key, condition) in where_entries {
        if key == "$and" || key == "$or" {
            let clauses: &[FilterValue] = if python_truthy(condition) {
                match condition {
                    FilterValue::Array(clauses) => clauses,
                    FilterValue::Object(_) | FilterValue::String(_) => {
                        return Err(ContractError::MetadataFilterWhereType {
                            actual: filter_type_name(condition),
                        });
                    }
                    other => {
                        return Err(ContractError::MetadataFilterLogicalClausesNotIterable {
                            operator: key.clone(),
                            actual: filter_type_name(other),
                        });
                    }
                }
            } else {
                &[]
            };
            if key == "$and" {
                for clause in clauses {
                    if !matches_where(metadata, clause)? {
                        return Ok(false);
                    }
                }
            } else {
                let mut matched = false;
                for clause in clauses {
                    if matches_where(metadata, clause)? {
                        matched = true;
                        break;
                    }
                }
                if !matched {
                    return Ok(false);
                }
            }
            continue;
        }

        let value = object_get(metadata, key).map_or(FieldValue::Missing, FieldValue::Present);
        if !matches_field(value, condition)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn metadata_filter_request(
    payload_json: &str,
) -> Result<(FilterValue, FilterValue), ContractError> {
    let request: FilterValue = serde_json::from_str(payload_json)?;
    let FilterValue::Object(request) = request else {
        return Err(ContractError::MetadataFilterRequestMustBeObject);
    };
    let metadata = object_get(&request, "metadata")
        .ok_or(ContractError::MetadataFilterMetadataMissing)?
        .clone();
    let where_value = object_get(&request, "where")
        .cloned()
        .unwrap_or(FilterValue::Null);
    if !matches!(metadata, FilterValue::Object(_)) {
        return Err(ContractError::MetadataFilterMetadataMustBeObject);
    }
    Ok((metadata, where_value))
}

/// Evaluate the current Python in-memory backend's Chroma-shaped `where` subset.
pub fn metadata_filter_matches_from_str(payload_json: &str) -> Result<bool, ContractError> {
    let (metadata, where_value) = metadata_filter_request(payload_json)?;
    let FilterValue::Object(metadata) = metadata else {
        unreachable!("metadata_filter_request validates metadata")
    };
    matches_where(&metadata, &where_value)
}

/// Canonical request representation for deterministic fixtures and telemetry.
pub fn normalize_metadata_filter_from_str(payload_json: &str) -> Result<String, ContractError> {
    let value: Value = serde_json::from_str(payload_json)?;
    let Value::Object(request) = &value else {
        return Err(ContractError::MetadataFilterRequestMustBeObject);
    };
    let metadata = request
        .get("metadata")
        .ok_or(ContractError::MetadataFilterMetadataMissing)?;
    if !metadata.is_object() {
        return Err(ContractError::MetadataFilterMetadataMustBeObject);
    }
    Ok(canonical_json(&value, false))
}

/// Static workflow graph request crossing the JSON-only Phase-1 boundary.
/// Unknown endpoints and duplicate edges are deliberately ignored, matching the
/// Python runtime's topology preparation.
#[derive(Debug, Deserialize)]
struct WorkflowTopologyRequest {
    node_ids: Vec<String>,
    #[serde(default)]
    edges: Vec<Vec<Value>>,
    #[serde(default)]
    join_ids: Vec<String>,
}

fn workflow_topology_request(payload_json: &str) -> Result<WorkflowTopologyRequest, ContractError> {
    let value: Value = serde_json::from_str(payload_json)?;
    let Value::Object(object) = value else {
        return Err(ContractError::WorkflowTopologyRequestMustBeObject);
    };
    for field in ["node_ids", "join_ids"] {
        let Some(Value::Array(values)) = object.get(field) else {
            return Err(ContractError::WorkflowTopologyStringArray { field });
        };
        if values.iter().any(|value| !value.is_string()) {
            return Err(ContractError::WorkflowTopologyStringArray { field });
        }
    }
    let Some(Value::Array(edges)) = object.get("edges") else {
        return Err(ContractError::WorkflowTopologyEdgesInvalid);
    };
    for edge in edges {
        let Value::Array(pair) = edge else {
            return Err(ContractError::WorkflowTopologyEdgesInvalid);
        };
        if pair.len() != 2 {
            return Err(ContractError::WorkflowTopologyEdgesInvalid);
        }
        if pair.iter().any(|value| !value.is_string()) {
            return Err(ContractError::WorkflowTopologyString { field: "edges" });
        }
    }
    serde_json::from_value(Value::Object(object)).map_err(ContractError::InvalidJson)
}

fn tarjan_scc(succ: &[Vec<usize>]) -> (Vec<usize>, Vec<Vec<usize>>) {
    struct Tarjan<'a> {
        succ: &'a [Vec<usize>],
        index: usize,
        stack: Vec<usize>,
        on_stack: Vec<bool>,
        indices: Vec<Option<usize>>,
        low: Vec<usize>,
        component_of: Vec<usize>,
        components: Vec<Vec<usize>>,
    }

    impl Tarjan<'_> {
        fn visit(&mut self, vertex: usize) {
            let vertex_index = self.index;
            self.indices[vertex] = Some(vertex_index);
            self.low[vertex] = vertex_index;
            self.index += 1;
            self.stack.push(vertex);
            self.on_stack[vertex] = true;

            for &next in &self.succ[vertex] {
                if self.indices[next].is_none() {
                    self.visit(next);
                    self.low[vertex] = self.low[vertex].min(self.low[next]);
                } else if self.on_stack[next] {
                    self.low[vertex] = self.low[vertex]
                        .min(self.indices[next].expect("visited vertices have an index"));
                }
            }

            if self.low[vertex] == vertex_index {
                let component_id = self.components.len();
                let mut component = Vec::new();
                loop {
                    let member = self.stack.pop().expect("current SCC has a member");
                    self.on_stack[member] = false;
                    self.component_of[member] = component_id;
                    component.push(member);
                    if member == vertex {
                        break;
                    }
                }
                self.components.push(component);
            }
        }
    }

    let count = succ.len();
    let mut tarjan = Tarjan {
        succ,
        index: 0,
        stack: Vec::new(),
        on_stack: vec![false; count],
        indices: vec![None; count],
        low: vec![0; count],
        component_of: vec![0; count],
        components: Vec::new(),
    };
    for vertex in 0..count {
        if tarjan.indices[vertex].is_none() {
            tarjan.visit(vertex);
        }
    }
    (tarjan.component_of, tarjan.components)
}

/// Compute join reachability with SCC condensation. Values are bit positions so
/// Python can construct unbounded integer bitsets without a native word-size cap.
pub fn workflow_may_reach_join_from_str(payload_json: &str) -> Result<String, ContractError> {
    let request = workflow_topology_request(payload_json)?;
    let node_count = request.node_ids.len();
    let mut node_index = std::collections::HashMap::new();
    for (index, node_id) in request.node_ids.iter().enumerate() {
        node_index.insert(node_id.as_str(), index);
    }
    let mut successors = vec![Vec::new(); node_count];
    for edge in &request.edges {
        let source = edge[0].as_str().expect("validated edge source");
        let target = edge[1].as_str().expect("validated edge target");
        if let (Some(&source), Some(&target)) = (node_index.get(source), node_index.get(target)) {
            successors[source].push(target);
        }
    }

    let (component_of, components) = tarjan_scc(&successors);
    let component_count = components.len();
    let mut component_join_bits = vec![Vec::new(); component_count];
    let mut join_index = std::collections::HashMap::new();
    for (index, join_id) in request.join_ids.iter().enumerate() {
        join_index.insert(join_id.as_str(), index);
    }
    for (node, node_id) in request.node_ids.iter().enumerate() {
        if let Some(&join) = join_index.get(node_id.as_str()) {
            component_join_bits[component_of[node]].push(join);
        }
    }

    let mut component_successors = vec![Vec::new(); component_count];
    let mut indegree = vec![0usize; component_count];
    for (source, successors) in successors.iter().enumerate() {
        let source_component = component_of[source];
        for &target in successors {
            let target_component = component_of[target];
            if source_component != target_component
                && !component_successors[source_component].contains(&target_component)
            {
                component_successors[source_component].push(target_component);
                indegree[target_component] += 1;
            }
        }
    }
    let mut queue = VecDeque::new();
    for (component, &degree) in indegree.iter().enumerate() {
        if degree == 0 {
            queue.push_back(component);
        }
    }
    let mut topological = Vec::with_capacity(component_count);
    while let Some(component) = queue.pop_back() {
        topological.push(component);
        for &next in &component_successors[component] {
            indegree[next] -= 1;
            if indegree[next] == 0 {
                queue.push_back(next);
            }
        }
    }
    let mut may_reach = component_join_bits;
    for &component in topological.iter().rev() {
        for &next in &component_successors[component] {
            let next_bits = may_reach[next].clone();
            for bit in next_bits {
                if !may_reach[component].contains(&bit) {
                    may_reach[component].push(bit);
                }
            }
        }
        may_reach[component].sort_unstable();
    }
    let result: Map<String, Value> = request
        .node_ids
        .iter()
        .enumerate()
        .map(|(node, node_id)| {
            let bits = may_reach[component_of[node]]
                .iter()
                .map(|bit| Value::from(*bit as u64))
                .collect();
            (node_id.clone(), Value::Array(bits))
        })
        .collect();
    Ok(canonical_json(&Value::Object(result), false))
}

/// Whether any terminal is reachable from start, ignoring predicates.
pub fn workflow_terminal_reachable_from_str(payload_json: &str) -> Result<bool, ContractError> {
    let request = workflow_topology_request(payload_json)?;
    let value: Value = serde_json::from_str(payload_json)?;
    let Value::Object(object) = value else {
        return Err(ContractError::WorkflowTopologyRequestMustBeObject);
    };
    let Some(Value::String(start)) = object.get("start_node_id") else {
        return Err(ContractError::WorkflowTopologyString {
            field: "start_node_id",
        });
    };
    let Some(Value::Array(terminals)) = object.get("terminal_ids") else {
        return Err(ContractError::WorkflowTopologyStringArray {
            field: "terminal_ids",
        });
    };
    if terminals.iter().any(|terminal| !terminal.is_string()) {
        return Err(ContractError::WorkflowTopologyStringArray {
            field: "terminal_ids",
        });
    }
    let node_refs: std::collections::HashMap<&str, &str> = request
        .node_ids
        .iter()
        .map(|node| (node.as_str(), node.as_str()))
        .collect();
    let mut successors: std::collections::HashMap<&str, Vec<&str>> = request
        .node_ids
        .iter()
        .map(|node| (node.as_str(), Vec::new()))
        .collect();
    for edge in &request.edges {
        let source = edge[0].as_str().expect("validated edge source");
        let target = edge[1].as_str().expect("validated edge target");
        if let (Some(&source), Some(&target)) = (node_refs.get(source), node_refs.get(target)) {
            successors
                .get_mut(source)
                .expect("source exists")
                .push(target);
        }
    }
    let terminal_ids: std::collections::HashSet<&str> =
        terminals.iter().filter_map(Value::as_str).collect();
    let mut seen = std::collections::HashSet::new();
    let mut stack = vec![start.as_str()];
    while let Some(node) = stack.pop() {
        if !seen.insert(node) {
            continue;
        }
        if terminal_ids.contains(node) {
            return Ok(true);
        }
        if let Some(next_nodes) = successors.get(node) {
            stack.extend(next_nodes.iter().copied());
        }
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn fixture() -> Value {
        serde_json::from_str(include_str!(
            "../../../../contracts/golden/deterministic-primitives.json"
        ))
        .expect("valid deterministic primitive fixture")
    }

    #[test]
    fn stable_ids_match_python_golden() {
        for case in fixture()["stable_ids"].as_array().unwrap() {
            let mut values = vec![case["kind"].clone()];
            values.extend(case["parts"].as_array().unwrap().iter().cloned());
            assert_eq!(stable_id_values(&values).to_string(), case["expected"]);
        }
    }

    #[test]
    fn canonical_json_matches_python_golden() {
        for case in fixture()["canonical_json"].as_array().unwrap() {
            assert_eq!(canonical_json(&case["value"], false), case["expected"]);
        }
    }

    #[test]
    fn evidence_hash_matches_python_golden() {
        for case in fixture()["evidence_hashes"].as_array().unwrap() {
            let canonical = canonicalize_evidence_pack_digest(&case["value"]).unwrap();
            assert_eq!(canonical, case["canonical"]);
            assert_eq!(
                evidence_pack_digest_hash(&case["value"]).unwrap(),
                case["expected"]
            );
        }
    }

    proptest! {
        #[test]
        fn short_id_json_round_trip_preserves_order(
            values in prop::collection::vec("[A-Za-z][A-Za-z0-9_-]{0,12}", 0..24)
        ) {
            let request = serde_json::json!({
                "state": {"next": 1, "l2s": {}, "s2l": {}},
                "input": values,
                "direction": "l2s",
                "depth": 0,
                "scalar_keys": [],
                "list_keys": [],
                "primitive": true
            });
            let shortened: Value = serde_json::from_str(
                &short_id_transform_from_str(&request.to_string()).unwrap()
            ).unwrap();
            let restore = serde_json::json!({
                "state": shortened["state"],
                "input": shortened["value"],
                "direction": "s2l",
                "depth": 0,
                "scalar_keys": [],
                "list_keys": [],
                "primitive": true
            });
            let restored: Value = serde_json::from_str(
                &short_id_transform_from_str(&restore.to_string()).unwrap()
            ).unwrap();
            prop_assert_eq!(restored["value"].clone(), request["input"].clone());
        }

        #[test]
        fn state_update_fold_preserves_append_order(
            initial in prop::collection::vec(any::<i32>(), 0..20),
            appended in any::<i32>(),
            extended in prop::collection::vec(any::<i32>(), 0..20),
        ) {
            let request = serde_json::json!({
                "state": {"items": initial},
                "state_update": [
                    ["a", {"items": appended}],
                    ["e", {"items": extended}]
                ],
                "update": null,
                "state_schema": {}
            });
            let result: Value = serde_json::from_str(
                &apply_state_update_from_str(&request.to_string()).unwrap()
            ).unwrap();
            let mut expected = request["state"]["items"].as_array().unwrap().clone();
            expected.push(request["state_update"][0][1]["items"].clone());
            expected.extend(
                request["state_update"][1][1]["items"].as_array().unwrap().clone()
            );
            prop_assert_eq!(result["items"].as_array().unwrap(), &expected);
        }

        #[test]
        fn workflow_lineage_matches_independent_bfs_oracle(
            node_count in 0usize..12,
            edge_flags in prop::collection::vec(any::<bool>(), 1..144),
            join_flags in prop::collection::vec(any::<bool>(), 1..12),
        ) {
            let node_ids: Vec<String> = (0..node_count).map(|index| format!("n{index}")).collect();
            let mut edges = Vec::new();
            let mut flag_index = 0usize;
            for source in 0..node_count {
                for target in 0..node_count {
                    if edge_flags[flag_index] {
                        edges.push(serde_json::json!([node_ids[source], node_ids[target]]));
                    }
                    flag_index += 1;
                    if flag_index == edge_flags.len() {
                        flag_index = 0;
                    }
                }
            }
            let join_ids: Vec<String> = node_ids.iter().enumerate()
                .filter(|(index, _)| join_flags[*index % join_flags.len()])
                .map(|(_, node_id)| node_id.clone())
                .collect();
            let request = serde_json::json!({
                "node_ids": node_ids,
                "edges": edges,
                "join_ids": join_ids,
            });
            let actual: Value = serde_json::from_str(
                &workflow_may_reach_join_from_str(&request.to_string()).unwrap()
            ).unwrap();
            for node in request["node_ids"].as_array().unwrap() {
                let node_id = node.as_str().unwrap();
                let mut reachable = std::collections::HashSet::new();
                let mut stack = vec![node_id];
                while let Some(current) = stack.pop() {
                    if !reachable.insert(current) {
                        continue;
                    }
                    for edge in request["edges"].as_array().unwrap() {
                        if edge[0].as_str() == Some(current) {
                            stack.push(edge[1].as_str().unwrap());
                        }
                    }
                }
                let expected: Vec<Value> = request["join_ids"].as_array().unwrap().iter()
                    .enumerate()
                    .filter(|(_, join)| reachable.contains(join.as_str().unwrap()))
                    .map(|(index, _)| Value::from(index as u64))
                    .collect();
                prop_assert_eq!(&actual[node_id], &Value::Array(expected));
            }
        }
    }
}
