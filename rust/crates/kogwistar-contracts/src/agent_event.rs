use crate::ProviderEventV1;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

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

/// Build the replay-stable envelope used by the Python provider-event journal.
/// The identity is computed from the original attempt coordinates first; the
/// stored trace then uses the provider response as its stable step identity.
pub fn durable_provider_event(
    trace: &EventTraceMetadataV1,
    event: ProviderEventV1,
) -> DurableProviderEventV1 {
    let event_id = durable_provider_event_id(trace, &event);
    let event_value = serde_json::to_value(&event).expect("provider events are serializable");
    let response_id = event_value
        .get("response_id")
        .and_then(Value::as_str)
        .or(trace.provider_response_id.as_deref())
        .unwrap_or("unknown");
    let mut stable_trace = trace.clone();
    stable_trace.step_id = format!("provider-response:{response_id}");
    stable_trace.attempt = 1;
    DurableProviderEventV1 {
        event_id,
        trace: stable_trace,
        event,
    }
}

/// Return the same replay-stable provider event ID used by the Python journal.
pub fn durable_provider_event_id(trace: &EventTraceMetadataV1, event: &ProviderEventV1) -> String {
    let event_value = serde_json::to_value(event).expect("provider events are serializable");
    let mut identity = json!({
        "provider_id": &trace.provider_id,
        "response_id": event_value
            .get("response_id")
            .and_then(Value::as_str)
            .or(trace.provider_response_id.as_deref()),
        "workflow_node_id": &trace.workflow_node_id,
        "event": event_value,
    });
    let provider_coordinate = identity["event"]
        .get("provider_event_id")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .or_else(|| {
            identity["event"]
                .get("provider_sequence")
                .and_then(Value::as_i64)
                .map(|sequence| sequence.to_string())
        });
    if let Some(coordinate) = provider_coordinate {
        identity["provider_coordinate"] = Value::String(coordinate);
    } else {
        identity["fallback_attempt"] = json!({
            "run_id": &trace.run_id,
            "turn_id": &trace.turn_id,
            "workflow_node_id": &trace.workflow_node_id,
            "step_id": &trace.step_id,
            "attempt": trace.attempt,
        });
    }
    let digest = Sha256::digest(crate::canonical_json(&identity, true).as_bytes());
    format!("agent-event:{}", hex::encode(digest))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    #[derive(Deserialize)]
    struct GoldenEnvelope {
        envelope: Value,
        publication: Value,
        computed_event_id: String,
        computed_sequence_event_id: String,
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
        assert_eq!(
            durable_provider_event_id(&envelope.trace, &envelope.event),
            fixture.computed_event_id
        );
        let mut sequence_event_value = fixture.envelope["event"].clone();
        sequence_event_value["provider_event_id"] = Value::Null;
        let sequence_event: ProviderEventV1 = serde_json::from_value(sequence_event_value).unwrap();
        assert_eq!(
            durable_provider_event_id(&envelope.trace, &sequence_event),
            fixture.computed_sequence_event_id
        );
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

    #[test]
    fn durable_provider_event_stabilizes_trace_without_losing_producer_attempt() {
        let trace = EventTraceMetadataV1 {
            run_id: "run-1".to_owned(),
            turn_id: "turn-1".to_owned(),
            workflow_node_id: "wf:agent.event.v1:request".to_owned(),
            step_id: "17".to_owned(),
            attempt: 3,
            producer_step_seq: Some(17),
            producer_attempt: Some(3),
            trace_id: None,
            span_id: None,
            parent_span_id: None,
            provider_id: "fixture".to_owned(),
            provider_response_id: Some("resp-1".to_owned()),
            provider_sequence: Some(2),
            client_sse_sequence: Some(4),
        };
        let event: ProviderEventV1 = serde_json::from_value(json!({
            "contract_version": "v1",
            "provider_event_id": "resp-1:2",
            "provider_sequence": 2,
            "response_id": "resp-1",
            "kind": "response_completed",
            "has_tool_calls": false
        }))
        .expect("valid provider event");
        let expected_id = durable_provider_event_id(&trace, &event);
        let durable = durable_provider_event(&trace, event);

        assert_eq!(durable.event_id, expected_id);
        assert_eq!(durable.trace.step_id, "provider-response:resp-1");
        assert_eq!(durable.trace.attempt, 1);
        assert_eq!(durable.trace.producer_step_seq, Some(17));
        assert_eq!(durable.trace.producer_attempt, Some(3));
    }
}
