use std::hint::black_box;
use std::time::{Duration, Instant};

use kogwistar_contracts::{GraphEntityKind, GraphModelClass, GraphModelEnvelope};
use kogwistar_store::{
    GraphMutation, GraphRecord, GraphScope, StoreError, validate_graph_mutation,
    validate_typed_graph_mutation,
};
use serde_json::{Map, Value};

const WARMUP_ITERATIONS: usize = 200;
const ITERATIONS_PER_SAMPLE: usize = 2_000;
const SAMPLES: usize = 9;

fn fixture_payload(index: usize) -> Result<Value, Box<dyn std::error::Error>> {
    let path = format!(
        "{}/../../../contracts/golden/python-graph-model-payloads-v1.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let fixture: Value = serde_json::from_slice(&std::fs::read(path)?)?;
    Ok(fixture["cases"][index]["payload"].clone())
}

fn mutation(
    entity_kind: &str,
    payload: &Value,
) -> Result<GraphMutation, Box<dyn std::error::Error>> {
    let id = payload["id"].as_str().ok_or("fixture lacks entity ID")?;
    let metadata: Map<String, Value> = payload["metadata"].as_object().cloned().unwrap_or_default();
    Ok(GraphMutation {
        scope: GraphScope {
            namespace: "bench".to_owned(),
            workspace_id: Some("bench-workspace".to_owned()),
            graph_space: Some("facts".to_owned()),
        },
        table: if entity_kind == "node" {
            "gke_nodes"
        } else {
            "gke_edges"
        }
        .to_owned(),
        entity_kind: entity_kind.to_owned(),
        event_id: "bench-event".to_owned(),
        op: "ADD".to_owned(),
        payload: payload.clone(),
        record: GraphRecord {
            id: id.to_owned(),
            document: None,
            metadata,
            embedding: None,
        },
        embedding_dim: 0,
    })
}

// Reproduces the validation sequence before the single-decode optimization.
fn previous_validation(
    mutation: &GraphMutation,
    envelope: &GraphModelEnvelope,
) -> Result<(), StoreError> {
    validate_graph_mutation(mutation)?;
    envelope
        .validate_family_v1()
        .map_err(|error| StoreError::InvalidEntityEventPayload {
            message: error.to_string(),
        })?;
    let expected_kind = match mutation.entity_kind.as_str() {
        "node" => GraphEntityKind::Node,
        "edge" => GraphEntityKind::Edge,
        _ => {
            return Err(StoreError::InvalidEntityEventPayload {
                message: "unsupported benchmark entity kind".to_owned(),
            });
        }
    };
    if envelope.model_class.entity_kind() != expected_kind {
        return Err(StoreError::InvalidEntityEventPayload {
            message: "typed graph mutation entity kind mismatch".to_owned(),
        });
    }
    if mutation.payload != envelope.payload {
        return Err(StoreError::InvalidEntityEventPayload {
            message: "typed graph mutation payload mismatch".to_owned(),
        });
    }
    let entity_id =
        match envelope
            .inspect()
            .map_err(|error| StoreError::InvalidEntityEventPayload {
                message: error.to_string(),
            })? {
            kogwistar_contracts::TypedGraphEntity::Node(node) => node.common.id,
            kogwistar_contracts::TypedGraphEntity::Edge(edge) => edge.common.id,
        };
    if entity_id.as_deref() != Some(mutation.record.id.as_str()) {
        return Err(StoreError::InvalidEntityEventPayload {
            message: "typed graph mutation record ID mismatch".to_owned(),
        });
    }
    Ok(())
}

fn sample<F>(mut operation: F) -> Result<Duration, StoreError>
where
    F: FnMut() -> Result<(), StoreError>,
{
    let started = Instant::now();
    for _ in 0..ITERATIONS_PER_SAMPLE {
        operation()?;
    }
    Ok(started.elapsed())
}

fn median(mut values: Vec<Duration>) -> Duration {
    values.sort_unstable();
    values[values.len() / 2]
}

fn run_case(
    label: &str,
    entity_kind: &str,
    payload_index: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let payload = fixture_payload(payload_index)?;
    let mutation = mutation(entity_kind, &payload)?;
    let model_class = if entity_kind == "node" {
        GraphModelClass::Node
    } else {
        GraphModelClass::Edge
    };
    let envelope = GraphModelEnvelope::new(model_class, payload);

    for _ in 0..WARMUP_ITERATIONS {
        previous_validation(black_box(&mutation), black_box(&envelope))?;
        validate_typed_graph_mutation(black_box(&mutation), black_box(&envelope))?;
    }

    let mut previous_samples = Vec::with_capacity(SAMPLES);
    let mut optimized_samples = Vec::with_capacity(SAMPLES);
    for sample_index in 0..SAMPLES {
        let (first, second) = if sample_index % 2 == 0 {
            (
                sample(|| previous_validation(black_box(&mutation), black_box(&envelope)))?,
                sample(|| {
                    validate_typed_graph_mutation(black_box(&mutation), black_box(&envelope))
                })?,
            )
        } else {
            let optimized = sample(|| {
                validate_typed_graph_mutation(black_box(&mutation), black_box(&envelope))
            })?;
            let previous =
                sample(|| previous_validation(black_box(&mutation), black_box(&envelope)))?;
            (previous, optimized)
        };
        previous_samples.push(first);
        optimized_samples.push(second);
    }

    let previous = median(previous_samples);
    let optimized = median(optimized_samples);
    let previous_rate = ITERATIONS_PER_SAMPLE as f64 / previous.as_secs_f64();
    let optimized_rate = ITERATIONS_PER_SAMPLE as f64 / optimized.as_secs_f64();
    println!(
        "{label}: old={previous_rate:.0} ops/s ({:.3} ms), current={optimized_rate:.0} ops/s ({:.3} ms), change={:+.1}%",
        previous.as_secs_f64() * 1_000.0,
        optimized.as_secs_f64() * 1_000.0,
        (optimized_rate / previous_rate - 1.0) * 100.0,
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "typed mutation validation: {SAMPLES} alternating samples, {ITERATIONS_PER_SAMPLE} iterations/sample, {WARMUP_ITERATIONS} warmups; release profile recommended"
    );
    run_case("Node", "node", 0)?;
    run_case("Edge", "edge", 1)?;
    Ok(())
}
