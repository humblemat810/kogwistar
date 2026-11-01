use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use thiserror::Error;

pub const PROVIDER_EVENT_CONTRACT_VERSION: &str = "v1";

fn default_contract_version() -> String {
    PROVIDER_EVENT_CONTRACT_VERSION.to_owned()
}

fn default_empty_text() -> String {
    String::new()
}

fn default_false() -> bool {
    false
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", deny_unknown_fields)]
pub enum ProviderEventV1 {
    #[serde(rename = "response_started")]
    ResponseStarted {
        #[serde(default = "default_contract_version")]
        contract_version: String,
        #[serde(default)]
        provider_event_id: Option<String>,
        #[serde(default)]
        provider_sequence: Option<i64>,
        #[serde(default)]
        response_id: Option<String>,
    },
    #[serde(rename = "text_delta")]
    TextDelta {
        #[serde(default = "default_contract_version")]
        contract_version: String,
        #[serde(default)]
        provider_event_id: Option<String>,
        #[serde(default)]
        provider_sequence: Option<i64>,
        #[serde(default)]
        response_id: Option<String>,
        delta: String,
    },
    #[serde(rename = "field_completed")]
    FieldCompleted {
        #[serde(default = "default_contract_version")]
        contract_version: String,
        #[serde(default)]
        provider_event_id: Option<String>,
        #[serde(default)]
        provider_sequence: Option<i64>,
        #[serde(default)]
        response_id: Option<String>,
        field_name: String,
        value: Value,
    },
    #[serde(rename = "reasoning_summary")]
    ReasoningSummary {
        #[serde(default = "default_contract_version")]
        contract_version: String,
        #[serde(default)]
        provider_event_id: Option<String>,
        #[serde(default)]
        provider_sequence: Option<i64>,
        #[serde(default)]
        response_id: Option<String>,
        text: String,
    },
    #[serde(rename = "tool_call_completed")]
    ToolCallCompleted {
        #[serde(default = "default_contract_version")]
        contract_version: String,
        #[serde(default)]
        provider_event_id: Option<String>,
        #[serde(default)]
        provider_sequence: Option<i64>,
        #[serde(default)]
        response_id: Option<String>,
        call_id: String,
        name: String,
        arguments: BTreeMap<String, Value>,
    },
    #[serde(rename = "response_completed")]
    ResponseCompleted {
        #[serde(default = "default_contract_version")]
        contract_version: String,
        #[serde(default)]
        provider_event_id: Option<String>,
        #[serde(default)]
        provider_sequence: Option<i64>,
        #[serde(default)]
        response_id: Option<String>,
        #[serde(default = "default_empty_text")]
        text: String,
        #[serde(default)]
        structured_output: Option<BTreeMap<String, Value>>,
        #[serde(default = "default_false")]
        has_tool_calls: bool,
        #[serde(default)]
        input_tokens: Option<i64>,
        #[serde(default)]
        output_tokens: Option<i64>,
    },
    #[serde(rename = "response_failed")]
    ResponseFailed {
        #[serde(default = "default_contract_version")]
        contract_version: String,
        #[serde(default)]
        provider_event_id: Option<String>,
        #[serde(default)]
        provider_sequence: Option<i64>,
        #[serde(default)]
        response_id: Option<String>,
        error_code: String,
        message: String,
        #[serde(default = "default_false")]
        retryable: bool,
    },
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ProviderEventContractError {
    #[error("provider event payload is invalid JSON or shape: {message}")]
    InvalidPayload { message: String },
    #[error("provider event contract_version must be v1")]
    InvalidVersion,
    #[error("provider event field {field} must not be empty")]
    EmptyField { field: &'static str },
}

impl ProviderEventV1 {
    pub fn validate(&self) -> Result<(), ProviderEventContractError> {
        let (contract_version, required_nonempty): (&str, &[(&str, &str)]) = match self {
            Self::ResponseStarted {
                contract_version, ..
            } => (contract_version, &[]),
            Self::TextDelta {
                contract_version, ..
            } => (contract_version, &[]),
            Self::FieldCompleted {
                contract_version, ..
            } => (contract_version, &[]),
            Self::ReasoningSummary {
                contract_version, ..
            } => (contract_version, &[]),
            Self::ToolCallCompleted {
                contract_version,
                call_id,
                name,
                ..
            } => (contract_version, &[("call_id", call_id), ("name", name)]),
            Self::ResponseCompleted {
                contract_version, ..
            } => (contract_version, &[]),
            Self::ResponseFailed {
                contract_version, ..
            } => (contract_version, &[]),
        };
        if contract_version != PROVIDER_EVENT_CONTRACT_VERSION {
            return Err(ProviderEventContractError::InvalidVersion);
        }
        for (field, value) in required_nonempty {
            if value.is_empty() {
                return Err(ProviderEventContractError::EmptyField { field });
            }
        }
        Ok(())
    }
}

// PYTHON_PARITY_V1: rust=rust/crates/kogwistar-contracts/src/provider_event.rs#provider_event_from_str; python=kogwistar/agent/streaming.py#ProviderEvent; fixture=contracts/golden/provider-events-v1.json; rust_test=rust/crates/kogwistar-contracts/src/provider_event.rs#provider_events_match_shared_python_golden_fixture; python_test=tests/agent/test_provider_event_cross_language_contract.py#test_python_provider_events_match_shared_rust_golden_fixture
pub fn provider_event_from_str(
    payload: &str,
) -> Result<ProviderEventV1, ProviderEventContractError> {
    let event: ProviderEventV1 = serde_json::from_str(payload).map_err(|error| {
        ProviderEventContractError::InvalidPayload {
            message: error.to_string(),
        }
    })?;
    event.validate()?;
    Ok(event)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Deserialize)]
    struct GoldenEvents {
        events: Vec<Value>,
        invalid_events: Vec<Value>,
    }

    #[test]
    fn provider_events_match_shared_python_golden_fixture() {
        let fixture: GoldenEvents = serde_json::from_str(include_str!(
            "../../../../contracts/golden/provider-events-v1.json"
        ))
        .unwrap();
        assert_eq!(fixture.events.len(), 8);
        for expected in fixture.events {
            let event = provider_event_from_str(&expected.to_string()).unwrap();
            assert_eq!(serde_json::to_value(event).unwrap(), expected);
        }
        for invalid in fixture.invalid_events {
            assert!(provider_event_from_str(&invalid.to_string()).is_err());
        }
    }

    #[test]
    fn provider_event_parser_rejects_unknown_and_truncated_events() {
        for payload in [
            r#"{"kind":"response_started","extra":true}"#,
            r#"{"kind":"tool_call_completed","call_id":"","name":"x","arguments":{}}"#,
            r#"{"kind":"response_completed","contract_version":"v2"}"#,
            r#"{"kind":"response_completed""#,
        ] {
            assert!(
                provider_event_from_str(payload).is_err(),
                "accepted {payload}"
            );
        }
    }
}
