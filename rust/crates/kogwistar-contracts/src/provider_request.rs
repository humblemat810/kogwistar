use serde::{Deserialize, Serialize};
use serde_json::{Number, Value};
use std::collections::BTreeMap;
use thiserror::Error;

pub const PROVIDER_REQUEST_CONTRACT_VERSION: &str = "v1";

fn default_contract_version() -> String {
    PROVIDER_REQUEST_CONTRACT_VERSION.to_owned()
}

fn default_instructions() -> String {
    String::new()
}

fn default_tools() -> Vec<BTreeMap<String, Value>> {
    Vec::new()
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderRequestV1 {
    #[serde(default = "default_contract_version")]
    pub contract_version: String,
    pub model: String,
    #[serde(default = "default_instructions")]
    pub instructions: String,
    pub input: String,
    #[serde(default)]
    pub response_schema: Option<BTreeMap<String, Value>>,
    #[serde(default = "default_tools")]
    pub tools: Vec<BTreeMap<String, Value>>,
    #[serde(default)]
    pub previous_response_id: Option<String>,
    #[serde(default)]
    pub max_output_tokens: Option<Number>,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ProviderRequestContractError {
    #[error("provider request payload is invalid: {message}")]
    InvalidPayload { message: String },
    #[error("provider request contract_version must be v1")]
    InvalidVersion,
    #[error("provider request field {field} must be an integer")]
    InvalidInteger { field: &'static str },
}

impl ProviderRequestV1 {
    pub fn validate(&self) -> Result<(), ProviderRequestContractError> {
        if self.contract_version != PROVIDER_REQUEST_CONTRACT_VERSION {
            return Err(ProviderRequestContractError::InvalidVersion);
        }
        if self.max_output_tokens.as_ref().is_some_and(|value| {
            let decimal = value.to_string();
            let digits = decimal.strip_prefix('-').unwrap_or(&decimal);
            digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit())
        }) {
            return Err(ProviderRequestContractError::InvalidInteger {
                field: "max_output_tokens",
            });
        }
        Ok(())
    }
}

pub fn provider_request_from_str(
    payload: &str,
) -> Result<ProviderRequestV1, ProviderRequestContractError> {
    let request: ProviderRequestV1 = serde_json::from_str(payload).map_err(|error| {
        ProviderRequestContractError::InvalidPayload {
            message: error.to_string(),
        }
    })?;
    request.validate()?;
    Ok(request)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Deserialize)]
    struct GoldenRequests {
        requests: Vec<Value>,
        legacy_requests: Vec<LegacyRequest>,
        invalid_requests: Vec<Value>,
    }

    #[derive(Deserialize)]
    struct LegacyRequest {
        input: Value,
        normalized: Value,
    }

    #[test]
    fn provider_requests_match_shared_python_golden_fixture() {
        let fixture: GoldenRequests = serde_json::from_str(include_str!(
            "../../../../contracts/golden/provider-requests-v1.json"
        ))
        .unwrap();
        assert_eq!(fixture.requests.len(), 3);
        for expected in fixture.requests {
            let request = provider_request_from_str(&expected.to_string()).unwrap();
            assert_eq!(serde_json::to_value(request).unwrap(), expected);
        }
        assert_eq!(fixture.legacy_requests.len(), 1);
        for case in fixture.legacy_requests {
            let request = provider_request_from_str(&case.input.to_string()).unwrap();
            assert_eq!(serde_json::to_value(request).unwrap(), case.normalized);
        }
        let expected_errors = [
            "invalid_version",
            "invalid_payload",
            "invalid_payload",
            "invalid_integer",
        ];
        assert_eq!(fixture.invalid_requests.len(), expected_errors.len());
        for (invalid, expected) in fixture.invalid_requests.into_iter().zip(expected_errors) {
            let error = provider_request_from_str(&invalid.to_string()).unwrap_err();
            match expected {
                "invalid_version" => {
                    assert_eq!(error, ProviderRequestContractError::InvalidVersion)
                }
                "invalid_payload" => assert!(matches!(
                    error,
                    ProviderRequestContractError::InvalidPayload { .. }
                )),
                "invalid_integer" => assert_eq!(
                    error,
                    ProviderRequestContractError::InvalidInteger {
                        field: "max_output_tokens"
                    }
                ),
                other => panic!("unknown provider-request fixture error {other}"),
            }
        }
    }
}
