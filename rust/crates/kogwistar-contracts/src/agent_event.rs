use crate::ProviderEventV1;
use serde::{Deserialize, Serialize};

fn default_attempt() -> i64 {
    1
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EventTraceMetadataV1 {
    pub run_id: String,
    pub turn_id: String,
    pub workflow_node_id: String,
    pub step_id: String,
    #[serde(default = "default_attempt")]
    pub attempt: i64,
    #[serde(default)]
    pub producer_step_seq: Option<i64>,
    #[serde(default)]
    pub producer_attempt: Option<i64>,
    #[serde(default)]
    pub trace_id: Option<String>,
    #[serde(default)]
    pub span_id: Option<String>,
    #[serde(default)]
    pub parent_span_id: Option<String>,
    pub provider_id: String,
    #[serde(default)]
    pub provider_response_id: Option<String>,
    #[serde(default)]
    pub provider_sequence: Option<i64>,
    #[serde(default)]
    pub client_sse_sequence: Option<i64>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DurableProviderEventV1 {
    pub event_id: String,
    pub trace: EventTraceMetadataV1,
    pub event: ProviderEventV1,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DurableProviderEventPublicationV1 {
    pub namespace: String,
    pub conversation_id: String,
    pub event: DurableProviderEventV1,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    #[derive(Deserialize)]
    struct GoldenEnvelope {
        envelope: Value,
        publication: Value,
        invalid_envelopes: Vec<Value>,
    }

    #[test]
    fn durable_provider_event_matches_python_golden_fixture() {
        let fixture: GoldenEnvelope = serde_json::from_str(include_str!(
            "../../../../contracts/golden/agent-durable-provider-event-v1.json"
        ))
        .unwrap();
        let envelope: DurableProviderEventV1 =
            serde_json::from_value(fixture.envelope.clone()).unwrap();
        assert_eq!(serde_json::to_value(envelope).unwrap(), fixture.envelope);
        let publication: DurableProviderEventPublicationV1 =
            serde_json::from_value(fixture.publication.clone()).unwrap();
        assert_eq!(
            serde_json::to_value(publication).unwrap(),
            fixture.publication
        );
        for invalid in fixture.invalid_envelopes {
            assert!(
                serde_json::from_value::<DurableProviderEventV1>(invalid.clone()).is_err(),
                "accepted invalid envelope {invalid}"
            );
        }
    }
}
