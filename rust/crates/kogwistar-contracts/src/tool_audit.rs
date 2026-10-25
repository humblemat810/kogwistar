use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;
use thiserror::Error;

pub const TOOL_AUDIT_WIRE_VERSION: i32 = 1;

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ToolAuditItemV1 {
    pub schema_version: i32,
    pub node_id: String,
    pub entity_type: String,
    pub tool_name: String,
    pub conversation_id: Option<String>,
    pub turn_index: Option<i64>,
    pub summary: String,
    pub properties: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ToolAuditResponseV1 {
    pub schema_version: i32,
    pub conversation_id: Option<String>,
    pub items: Vec<ToolAuditItemV1>,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ToolAuditContractError {
    #[error("tool audit payload is invalid JSON: {message}")]
    InvalidPayload { message: String },
    #[error("tool audit payload must have schema_version=1")]
    InvalidVersion,
    #[error("tool audit field {field} must not be empty")]
    EmptyField { field: &'static str },
    #[error("tool audit turn_index must be non-negative")]
    InvalidTurnIndex,
    #[error("unsupported tool audit entity_type {entity_type:?}")]
    InvalidEntityType { entity_type: String },
}

impl ToolAuditItemV1 {
    pub fn validate(&self) -> Result<(), ToolAuditContractError> {
        if self.schema_version != TOOL_AUDIT_WIRE_VERSION {
            return Err(ToolAuditContractError::InvalidVersion);
        }
        for (field, value) in [("node_id", &self.node_id), ("tool_name", &self.tool_name)] {
            if value.trim().is_empty() {
                return Err(ToolAuditContractError::EmptyField { field });
            }
        }
        if let Some(conversation_id) = &self.conversation_id {
            if conversation_id.trim().is_empty() {
                return Err(ToolAuditContractError::EmptyField {
                    field: "conversation_id",
                });
            }
        }
        if self.turn_index.is_some_and(|index| index < 0) {
            return Err(ToolAuditContractError::InvalidTurnIndex);
        }
        if !matches!(self.entity_type.as_str(), "tool_call" | "tool_result") {
            return Err(ToolAuditContractError::InvalidEntityType {
                entity_type: self.entity_type.clone(),
            });
        }
        Ok(())
    }
}

impl ToolAuditResponseV1 {
    pub fn validate(&self) -> Result<(), ToolAuditContractError> {
        if self.schema_version != TOOL_AUDIT_WIRE_VERSION {
            return Err(ToolAuditContractError::InvalidVersion);
        }
        if let Some(conversation_id) = &self.conversation_id {
            if conversation_id.trim().is_empty() {
                return Err(ToolAuditContractError::EmptyField {
                    field: "conversation_id",
                });
            }
        }
        for item in &self.items {
            item.validate()?;
            if let (Some(response_conversation), Some(item_conversation)) =
                (&self.conversation_id, &item.conversation_id)
            {
                if response_conversation != item_conversation {
                    return Err(ToolAuditContractError::EmptyField {
                        field: "conversation_id",
                    });
                }
            }
        }
        Ok(())
    }
}

pub fn tool_audit_response_from_str(
    payload: &str,
) -> Result<ToolAuditResponseV1, ToolAuditContractError> {
    let value: ToolAuditResponseV1 =
        serde_json::from_str(payload).map_err(|error| ToolAuditContractError::InvalidPayload {
            message: error.to_string(),
        })?;
    value.validate()?;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_tool_audit_fixture_round_trips() {
        let payload = include_str!("../../../../contracts/golden/tool-audit-v1.json");
        let audit = tool_audit_response_from_str(payload).unwrap();
        assert_eq!(audit.items.len(), 2);
        assert_eq!(audit.items[0].entity_type, "tool_call");
    }

    #[test]
    fn unknown_fields_and_invalid_items_fail_closed() {
        let unknown = tool_audit_response_from_str(
            r#"{"schema_version":1,"conversation_id":null,"items":[],"extra":true}"#,
        )
        .unwrap_err();
        assert!(matches!(
            unknown,
            ToolAuditContractError::InvalidPayload { .. }
        ));

        let invalid = tool_audit_response_from_str(
            r#"{"schema_version":1,"conversation_id":"c","items":[{"schema_version":1,"node_id":"n","entity_type":"tool_call","tool_name":"x","conversation_id":"other","turn_index":0,"summary":"","properties":{}}]}"#,
        )
        .unwrap_err();
        assert_eq!(
            invalid,
            ToolAuditContractError::EmptyField {
                field: "conversation_id"
            }
        );
    }
}
