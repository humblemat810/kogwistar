use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use thiserror::Error;

pub const CONVERSATION_WIRE_VERSION: i32 = 1;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ConversationContractError {
    #[error("conversation payload is invalid JSON: {message}")]
    InvalidPayload { message: String },
    #[error("conversation payload must have schema_version=1")]
    InvalidVersion,
    #[error("conversation field {field} must not be empty")]
    EmptyField { field: &'static str },
    #[error("conversation turn_index must be non-negative")]
    InvalidTurnIndex,
    #[error("conversation turn_count must be non-negative")]
    InvalidTurnCount,
    #[error("conversation turns must be strictly ordered by turn_index")]
    InvalidTurnOrder,
    #[error("unsupported conversation entity_type {entity_type:?}")]
    InvalidEntityType { entity_type: String },
    #[error("context snapshot selection requires an ordering field")]
    MissingSnapshotOrder,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ContextSnapshotV1 {
    pub schema_version: i32,
    pub snapshot_node_id: String,
    pub conversation_id: String,
    pub metadata: BTreeMap<String, Value>,
    pub properties: BTreeMap<String, Value>,
}

impl ContextSnapshotV1 {
    pub fn validate(&self) -> Result<(), ConversationContractError> {
        if self.schema_version != CONVERSATION_WIRE_VERSION {
            return Err(ConversationContractError::InvalidVersion);
        }
        for (field, value) in [
            ("snapshot_node_id", &self.snapshot_node_id),
            ("conversation_id", &self.conversation_id),
        ] {
            if value.trim().is_empty() {
                return Err(ConversationContractError::EmptyField { field });
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ConversationSummaryV1 {
    pub schema_version: i32,
    pub conversation_id: String,
    pub user_id: String,
    pub status: String,
    pub start_node_id: String,
    pub tail_node_id: String,
    pub turn_count: i64,
}

impl ConversationSummaryV1 {
    pub fn validate(&self) -> Result<(), ConversationContractError> {
        if self.schema_version != CONVERSATION_WIRE_VERSION {
            return Err(ConversationContractError::InvalidVersion);
        }
        for (field, value) in [
            ("conversation_id", &self.conversation_id),
            ("user_id", &self.user_id),
            ("status", &self.status),
            ("start_node_id", &self.start_node_id),
        ] {
            if value.trim().is_empty() {
                return Err(ConversationContractError::EmptyField { field });
            }
        }
        if self.turn_count < 0 {
            return Err(ConversationContractError::InvalidTurnCount);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ConversationTurnV1 {
    pub schema_version: i32,
    pub node_id: String,
    pub turn_index: i64,
    pub role: String,
    pub content: String,
    pub entity_type: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ConversationTranscriptV1 {
    pub schema_version: i32,
    pub conversation_id: String,
    pub turns: Vec<ConversationTurnV1>,
}

impl ConversationTranscriptV1 {
    pub fn validate(&self) -> Result<(), ConversationContractError> {
        if self.schema_version != CONVERSATION_WIRE_VERSION {
            return Err(ConversationContractError::InvalidVersion);
        }
        if self.conversation_id.trim().is_empty() {
            return Err(ConversationContractError::EmptyField {
                field: "conversation_id",
            });
        }
        let mut previous_index = None;
        for turn in &self.turns {
            turn.validate()?;
            if previous_index.is_some_and(|previous| turn.turn_index <= previous) {
                return Err(ConversationContractError::InvalidTurnOrder);
            }
            previous_index = Some(turn.turn_index);
        }
        Ok(())
    }
}

impl ConversationTurnV1 {
    pub fn validate(&self) -> Result<(), ConversationContractError> {
        if self.schema_version != CONVERSATION_WIRE_VERSION {
            return Err(ConversationContractError::InvalidVersion);
        }
        for (field, value) in [
            ("node_id", &self.node_id),
            ("role", &self.role),
            ("entity_type", &self.entity_type),
        ] {
            if value.trim().is_empty() {
                return Err(ConversationContractError::EmptyField { field });
            }
        }
        if self.turn_index < 0 {
            return Err(ConversationContractError::InvalidTurnIndex);
        }
        if !matches!(
            self.entity_type.as_str(),
            "conversation_turn" | "assistant_turn"
        ) {
            return Err(ConversationContractError::InvalidEntityType {
                entity_type: self.entity_type.clone(),
            });
        }
        Ok(())
    }
}

pub fn conversation_summary_from_str(
    payload: &str,
) -> Result<ConversationSummaryV1, ConversationContractError> {
    let value: ConversationSummaryV1 = serde_json::from_str(payload).map_err(|error| {
        ConversationContractError::InvalidPayload {
            message: error.to_string(),
        }
    })?;
    value.validate()?;
    Ok(value)
}

pub fn conversation_turn_from_str(
    payload: &str,
) -> Result<ConversationTurnV1, ConversationContractError> {
    let value: ConversationTurnV1 = serde_json::from_str(payload).map_err(|error| {
        ConversationContractError::InvalidPayload {
            message: error.to_string(),
        }
    })?;
    value.validate()?;
    Ok(value)
}

pub fn conversation_transcript_from_str(
    payload: &str,
) -> Result<ConversationTranscriptV1, ConversationContractError> {
    let value: ConversationTranscriptV1 = serde_json::from_str(payload).map_err(|error| {
        ConversationContractError::InvalidPayload {
            message: error.to_string(),
        }
    })?;
    value.validate()?;
    Ok(value)
}

pub fn context_snapshot_from_str(
    payload: &str,
) -> Result<ContextSnapshotV1, ConversationContractError> {
    let value: ContextSnapshotV1 = serde_json::from_str(payload).map_err(|error| {
        ConversationContractError::InvalidPayload {
            message: error.to_string(),
        }
    })?;
    value.validate()?;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_conversation_fixture_round_trips() {
        let summary: ConversationSummaryV1 = serde_json::from_str(include_str!(
            "../../../../contracts/golden/conversation-wire-v1.json"
        ))
        .unwrap();
        summary.validate().unwrap();
        assert_eq!(summary.schema_version, CONVERSATION_WIRE_VERSION);
        assert_eq!(summary.turn_count, 2);
    }

    #[test]
    fn unknown_fields_fail_closed() {
        let error = conversation_summary_from_str(
            r#"{"schema_version":1,"conversation_id":"c","user_id":"u","status":"active","start_node_id":"n","tail_node_id":"t","turn_count":0,"extra":true}"#,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            ConversationContractError::InvalidPayload { .. }
        ));
    }

    #[test]
    fn invalid_turn_entity_type_fails_closed() {
        let error = conversation_turn_from_str(
            r#"{"schema_version":1,"node_id":"n","turn_index":0,"role":"user","content":"x","entity_type":"system"}"#,
        )
        .unwrap_err();
        assert_eq!(
            error,
            ConversationContractError::InvalidEntityType {
                entity_type: "system".to_owned()
            }
        );
    }

    #[test]
    fn shared_transcript_fixture_round_trips_and_requires_order() {
        let payload =
            include_str!("../../../../contracts/golden/conversation-transcript-wire-v1.json");
        let transcript = conversation_transcript_from_str(payload).unwrap();
        assert_eq!(transcript.turns.len(), 2);
        assert_eq!(transcript.turns[0].turn_index, 0);

        let error = conversation_transcript_from_str(
            r#"{"schema_version":1,"conversation_id":"c","turns":[{"schema_version":1,"node_id":"n2","turn_index":2,"role":"assistant","content":"later","entity_type":"assistant_turn"},{"schema_version":1,"node_id":"n1","turn_index":1,"role":"user","content":"first","entity_type":"conversation_turn"}]}"#,
        )
        .unwrap_err();
        assert_eq!(error, ConversationContractError::InvalidTurnOrder);
    }

    #[test]
    fn context_snapshot_contract_rejects_unknown_fields_and_round_trips() {
        let snapshot = ContextSnapshotV1 {
            schema_version: 1,
            snapshot_node_id: "snapshot-1".to_owned(),
            conversation_id: "conversation-1".to_owned(),
            metadata: BTreeMap::from([(String::from("stage"), Value::from("answer"))]),
            properties: BTreeMap::from([(String::from("count"), Value::from(2))]),
        };
        let payload = serde_json::to_string(&snapshot).unwrap();
        assert_eq!(context_snapshot_from_str(&payload).unwrap(), snapshot);
        assert!(matches!(
            context_snapshot_from_str(
                r#"{"schema_version":1,"snapshot_node_id":"s","conversation_id":"c","metadata":{},"properties":{},"extra":true}"#
            ),
            Err(ConversationContractError::InvalidPayload { .. })
        ));
    }
}
