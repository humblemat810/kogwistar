//! Rust-native SQLite metadata + Chroma projection trace benchmark.
//!
//! Reuses the public store contracts; no Python bridge is linked or invoked.
//! Python reference: scripts/native_graph_backend_benchmark.py.

use kogwistar_store::{
    DistanceMetric, GraphEntityKind, GraphReadStore, GraphRecord, GraphWriteStore,
};
use kogwistar_store_chroma::{ChromaGraphStore, ChromaStoreConfig};
use kogwistar_store_sqlite::{NewRawEntityEvent, SqliteStore};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant};

type BenchResult<T> = Result<T, Box<dyn Error>>;

fn append_canonical_json(value: &Value, output: &mut String) -> BenchResult<()> {
    match value {
        Value::Object(map) => {
            let mut keys = map.keys().collect::<Vec<_>>();
            keys.sort_unstable();
            output.push('{');
            for (index, key) in keys.iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                output.push_str(&serde_json::to_string(key)?);
                output.push(':');
                append_canonical_json(&map[*key], output)?;
            }
            output.push('}');
        }
        Value::Array(items) => {
            output.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                append_canonical_json(item, output)?;
            }
            output.push(']');
        }
        _ => output.push_str(&serde_json::to_string(value)?),
    }
    Ok(())
}

fn canonical_json(value: &Value) -> BenchResult<String> {
    let mut output = String::new();
    append_canonical_json(value, &mut output)?;
    Ok(output)
}

fn validate_trace(trace: &Value) -> BenchResult<()> {
    if trace["schema"] != "kogwistar-native-graph-backend-trace/v1" {
        return Err("unsupported graph trace schema".into());
    }
    if trace["trace_id"].as_str().is_none_or(str::is_empty)
        || trace["trace_sha256"].as_str().is_none_or(str::is_empty)
        || trace["root_payload"]["id"]
            .as_str()
            .is_none_or(str::is_empty)
    {
        return Err("trace identity or root payload is missing".into());
    }
    let operations = trace["operations"]
        .as_array()
        .ok_or("trace operations must be an array")?;
    if operations.is_empty() {
        return Err("trace operations must not be empty".into());
    }
    for (position, operation) in operations.iter().enumerate() {
        if operation["index"].as_u64() != Some((position / 3) as u64) {
            return Err("trace operation index is inconsistent".into());
        }
        let name = operation["operation"]
            .as_str()
            .ok_or("operation name missing")?;
        if !matches!(name, "write_record" | "read_event" | "read_projection") {
            return Err(format!("unsupported trace operation {name:?}").into());
        }
        if name == "write_record" {
            let node_id = operation["expected_entity_id"].as_str();
            let edge_id = operation["expected_edge_id"].as_str();
            let source_id = operation["expected_source_id"].as_str();
            if operation["node_payload"]["id"].as_str() != node_id
                || operation["edge_payload"]["id"].as_str() != edge_id
                || operation["edge_payload"]["source_ids"][0].as_str() != source_id
                || operation["edge_payload"]["target_ids"][0].as_str() != node_id
            {
                return Err("trace graph payload identity or endpoints mismatch".into());
            }
            let vector = operation["vector"]
                .as_array()
                .ok_or("trace vector missing")?;
            if vector.len() != operation["vector_dimension"].as_u64().unwrap_or_default() as usize {
                return Err("trace vector dimension mismatch".into());
            }
        }
    }
    let mut body = trace.clone();
    body.as_object_mut()
        .ok_or("trace document must be an object")?
        .remove("trace_sha256");
    let expected = trace["trace_sha256"]
        .as_str()
        .ok_or("trace digest missing")?;
    let actual = format!("{:x}", Sha256::digest(canonical_json(&body)?.as_bytes()));
    if actual != expected {
        return Err("trace digest mismatch".into());
    }
    Ok(())
}

fn graph_record(payload: &Value, vector: Option<Vec<f32>>) -> BenchResult<GraphRecord> {
    let id = payload["id"].as_str().ok_or("graph payload id missing")?;
    let metadata = payload["metadata"]
        .as_object()
        .cloned()
        .unwrap_or_else(Map::new);
    Ok(GraphRecord {
        id: id.to_owned(),
        document: Some(serde_json::to_string(payload)?),
        metadata,
        embedding: vector,
    })
}

fn append_event_pair(
    store: &SqliteStore,
    namespace: &str,
    trace_id: &str,
    node: &Value,
    edge: &Value,
) -> BenchResult<()> {
    let mut events = Vec::with_capacity(2);
    for (kind, payload) in [("node", node), ("edge", edge)] {
        let id = payload["id"].as_str().ok_or("graph payload id missing")?;
        events.push(NewRawEntityEvent {
            event_id: format!("{trace_id}:{kind}:{id}"),
            entity_kind: kind.to_owned(),
            entity_id: id.to_owned(),
            op: "ADD".to_owned(),
            payload_json: serde_json::to_string(payload)?,
        });
    }
    store.immediate_transaction(|uow| {
        for event in events {
            uow.append_raw_entity_event(namespace, event)?;
        }
        Ok(())
    })?;
    Ok(())
}

async fn project_record(
    chroma: &ChromaGraphStore,
    namespace: &str,
    kind: GraphEntityKind,
    payload: &Value,
    vector: Option<Vec<f32>>,
) -> BenchResult<()> {
    chroma
        .upsert_graph_record_of_kind(namespace, kind, graph_record(payload, vector)?)
        .await?;
    Ok(())
}

async fn read_projection(
    chroma: &ChromaGraphStore,
    namespace: &str,
    kind: GraphEntityKind,
    id: &str,
) -> BenchResult<Value> {
    let record = chroma
        .graph_record_of_kind(namespace, kind, id)
        .await?
        .ok_or_else(|| format!("Chroma projection {kind:?}/{id} missing"))?;
    let document = record
        .document
        .ok_or("Chroma projection document missing")?;
    let payload: Value = serde_json::from_str(&document)?;
    if payload["id"] != id {
        return Err("Chroma projection id mismatch".into());
    }
    Ok(payload)
}

#[cfg(target_os = "linux")]
fn peak_rss_bytes() -> Option<u64> {
    fs::read_to_string("/proc/self/status")
        .ok()?
        .lines()
        .find_map(|line| {
            line.strip_prefix("VmHWM:")?
                .split_whitespace()
                .next()?
                .parse::<u64>()
                .ok()
                .map(|value| value * 1024)
        })
}

#[cfg(windows)]
fn peak_rss_bytes() -> Option<u64> {
    use winapi::um::processthreadsapi::GetCurrentProcess;
    use winapi::um::psapi::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};

    let mut counters: PROCESS_MEMORY_COUNTERS = unsafe { std::mem::zeroed() };
    counters.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
    let buffer_size = counters.cb;
    let succeeded =
        unsafe { GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, buffer_size) };
    (succeeded != 0).then_some(counters.PeakWorkingSetSize as u64)
}

#[cfg(not(any(target_os = "linux", windows)))]
fn peak_rss_bytes() -> Option<u64> {
    None
}

fn percentile95(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[((sorted.len() as f64 * 0.95).ceil() as usize).saturating_sub(1)]
}

async fn execute_trace(
    sqlite: &SqliteStore,
    chroma: &ChromaGraphStore,
    trace: &Value,
    namespace: &str,
) -> BenchResult<Value> {
    let trace_id = trace["trace_id"].as_str().expect("validated trace id");
    let trace_sha = trace["trace_sha256"].as_str().expect("validated digest");
    let root = &trace["root_payload"];

    sqlite.immediate_transaction(|uow| {
        let id = root["id"].as_str().ok_or_else(|| {
            kogwistar_store_sqlite::SqliteStoreError::TransactionAborted(
                "root id missing".to_owned(),
            )
        })?;
        let payload_json = serde_json::to_string(root).map_err(|error| {
            kogwistar_store_sqlite::SqliteStoreError::TransactionAborted(error.to_string())
        })?;
        uow.append_raw_entity_event(
            namespace,
            NewRawEntityEvent {
                event_id: format!("{trace_id}:node:{id}"),
                entity_kind: "node".to_owned(),
                entity_id: id.to_owned(),
                op: "ADD".to_owned(),
                payload_json,
            },
        )?;
        Ok(())
    })?;
    let root_vector = trace["operations"]
        .as_array()
        .and_then(|items| {
            items
                .iter()
                .find(|item| item["operation"] == "write_record")
        })
        .and_then(|item| item["vector"].as_array())
        .ok_or("trace has no root projection vector")?
        .iter()
        .map(|value| {
            value
                .as_f64()
                .map(|number| number as f32)
                .ok_or("invalid root vector")
        })
        .collect::<Result<Vec<_>, _>>()?;
    project_record(
        chroma,
        namespace,
        GraphEntityKind::Node,
        root,
        Some(root_vector),
    )
    .await?;

    let operations = trace["operations"]
        .as_array()
        .expect("validated operations");
    let mut node_ids = BTreeSet::new();
    let mut edge_ids = BTreeSet::new();
    let mut latencies = BTreeMap::<String, Vec<f64>>::new();
    let wall_start = Instant::now();

    for operation in operations {
        let name = operation["operation"]
            .as_str()
            .expect("validated operation");
        let started = Instant::now();
        match name {
            "write_record" => {
                let node = &operation["node_payload"];
                let edge = &operation["edge_payload"];
                append_event_pair(sqlite, namespace, trace_id, node, edge)?;
                let vector = operation["vector"]
                    .as_array()
                    .ok_or("vector missing")?
                    .iter()
                    .map(|value| {
                        value
                            .as_f64()
                            .map(|number| number as f32)
                            .ok_or("invalid vector")
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                project_record(
                    chroma,
                    namespace,
                    GraphEntityKind::Node,
                    node,
                    Some(vector.clone()),
                )
                .await?;
                project_record(chroma, namespace, GraphEntityKind::Edge, edge, Some(vector))
                    .await?;
                node_ids.insert(node["id"].as_str().ok_or("node id missing")?.to_owned());
                edge_ids.insert(edge["id"].as_str().ok_or("edge id missing")?.to_owned());
            }
            "read_event" => {
                let index = operation["index"].as_i64().ok_or("trace index missing")?;
                let events = sqlite.replay_raw_events(namespace, 1 + 2 * index, 2)?;
                if events.len() != 2
                    || events[0].entity_kind != "node"
                    || events[0].entity_id != operation["expected_entity_id"]
                    || events[1].entity_kind != "edge"
                    || events[1].entity_id != operation["expected_edge_id"]
                {
                    return Err("canonical event read differs from shared trace".into());
                }
            }
            "read_projection" => {
                let node_id = operation["expected_entity_id"]
                    .as_str()
                    .ok_or("node id missing")?;
                let edge_id = operation["expected_edge_id"]
                    .as_str()
                    .ok_or("edge id missing")?;
                let node =
                    read_projection(chroma, namespace, GraphEntityKind::Node, node_id).await?;
                let edge =
                    read_projection(chroma, namespace, GraphEntityKind::Edge, edge_id).await?;
                if edge["source_ids"][0] != operation["expected_source_id"]
                    || edge["target_ids"][0] != node_id
                    || node["id"] != node_id
                {
                    return Err("Chroma projection topology differs from shared trace".into());
                }
            }
            _ => return Err(format!("unsupported operation {name:?}").into()),
        }
        latencies
            .entry(name.to_owned())
            .or_default()
            .push(started.elapsed().as_secs_f64() * 1000.0);
    }

    let mut canonical_records = BTreeMap::<String, Value>::new();
    canonical_records.insert(
        format!("node:{}", root["id"].as_str().ok_or("root id missing")?),
        root.clone(),
    );
    let mut projection_converged = true;
    for (kind, ids, entity_kind) in [
        ("node", node_ids, GraphEntityKind::Node),
        ("edge", edge_ids, GraphEntityKind::Edge),
    ] {
        for id in ids {
            match read_projection(chroma, namespace, entity_kind, &id).await {
                Ok(payload) => {
                    canonical_records.insert(format!("{kind}:{id}"), payload);
                }
                Err(_) => projection_converged = false,
            }
        }
    }
    let events = sqlite.replay_raw_events(namespace, 0, usize::MAX)?;
    let digest = Sha256::digest(canonical_json(&json!(canonical_records))?.as_bytes());
    let elapsed_ms = wall_start.elapsed().as_secs_f64() * 1000.0;
    let peak = peak_rss_bytes().ok_or("peak RSS unavailable; run in Linux container")?;
    Ok(json!({
        "trace_id": trace_id,
        "trace_sha256": trace_sha,
        "state_digest": format!("{digest:x}"),
        "canonical_event_count": events.len(),
        "projection_converged": projection_converged && canonical_records.len() == operations.iter().filter(|item| item["operation"] == "write_record").count() * 2 + 1,
        "peak_rss_bytes": peak,
        "elapsed_ms": elapsed_ms,
        "throughput_ops_per_sec": operations.len() as f64 * 1000.0 / elapsed_ms,
        "operation_p95_ms": latencies.iter().map(|(name, values)| (name, percentile95(values))).collect::<BTreeMap<_, _>>(),
        "resource_limits_verified": std::env::var("KOGWISTAR_BENCH_RESOURCE_LIMITS_VERIFIED").as_deref() == Ok("1"),
        "ffi_calls": 0,
    }))
}

#[tokio::main]
async fn main() -> BenchResult<()> {
    let mut args = std::env::args().skip(1);
    let trace_path = PathBuf::from(args.next().ok_or("expected trace JSON path")?);
    let sqlite_path = PathBuf::from(args.next().ok_or("expected fresh SQLite path")?);
    let namespace = args.next().ok_or("expected isolated namespace")?;
    if args.next().is_some() || namespace.trim().is_empty() {
        return Err("usage: <trace-json> <fresh-sqlite-path> <namespace>".into());
    }
    let trace: Value = serde_json::from_slice(&fs::read(trace_path)?)?;
    validate_trace(&trace)?;
    let endpoint = std::env::var("KOGWISTAR_TEST_CHROMA_URL")?;
    let node_config = ChromaStoreConfig {
        endpoint: endpoint.clone(),
        collection: std::env::var("KOGWISTAR_BENCH_CHROMA_NODE_COLLECTION")
            .unwrap_or_else(|_| "kogwistar_dlfc18_trace_nodes".to_owned()),
        metric: DistanceMetric::Cosine,
        request_timeout: Duration::from_secs(60),
        ..ChromaStoreConfig::default()
    };
    let edge_config = ChromaStoreConfig {
        collection: std::env::var("KOGWISTAR_BENCH_CHROMA_EDGE_COLLECTION")
            .unwrap_or_else(|_| "kogwistar_dlfc18_trace_edges".to_owned()),
        ..node_config.clone()
    };
    let sqlite = SqliteStore::open(sqlite_path)?;
    let chroma = ChromaGraphStore::connect_entity_collections(node_config, edge_config).await?;
    let report = execute_trace(&sqlite, &chroma, &trace, &namespace).await?;
    println!("{}", serde_json::to_string(&report)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{canonical_json, validate_trace};
    use serde_json::{Value, json};
    use sha2::{Digest, Sha256};

    fn valid_trace() -> Value {
        let mut trace = json!({
            "schema": "kogwistar-native-graph-backend-trace/v1",
            "trace_id": "trace-1",
            "trace_sha256": "",
            "root_payload": {"id": "root"},
            "operations": [
                {
                    "operation": "write_record",
                    "index": 0,
                    "expected_entity_id": "node-0",
                    "expected_edge_id": "edge-0",
                    "expected_source_id": "root",
                    "node_payload": {"id": "node-0", "metadata": {}},
                    "edge_payload": {
                        "id": "edge-0",
                        "source_ids": ["root"],
                        "target_ids": ["node-0"],
                        "metadata": {}
                    },
                    "vector": [0.25],
                    "vector_dimension": 1
                },
                {
                    "operation": "read_event",
                    "index": 0,
                    "expected_entity_id": "node-0",
                    "expected_edge_id": "edge-0",
                    "expected_source_id": "root"
                },
                {
                    "operation": "read_projection",
                    "index": 0,
                    "expected_entity_id": "node-0",
                    "expected_edge_id": "edge-0",
                    "expected_source_id": "root"
                }
            ]
        });
        trace.as_object_mut().unwrap().remove("trace_sha256");
        let digest = format!(
            "{:x}",
            Sha256::digest(canonical_json(&trace).unwrap().as_bytes())
        );
        trace
            .as_object_mut()
            .unwrap()
            .insert("trace_sha256".to_owned(), Value::String(digest));
        trace
    }

    #[test]
    fn accepts_shared_v1_trace_with_matching_digest_and_topology() {
        assert!(validate_trace(&valid_trace()).is_ok());
    }

    #[test]
    fn rejects_trace_digest_mismatch() {
        let mut trace = valid_trace();
        trace["trace_sha256"] = json!("stale");
        assert!(
            validate_trace(&trace)
                .unwrap_err()
                .to_string()
                .contains("digest")
        );
    }

    #[test]
    fn rejects_edge_endpoint_mismatch() {
        let mut trace = valid_trace();
        trace["operations"][0]["edge_payload"]["target_ids"][0] = json!("wrong");
        assert!(
            validate_trace(&trace)
                .unwrap_err()
                .to_string()
                .contains("endpoints")
        );
    }

    #[test]
    fn canonical_json_sorts_nested_object_keys_like_python() {
        let value = json!({"z": {"b": 2, "a": 1}, "a": [3, {"y": 5, "x": 4}]});
        assert_eq!(
            canonical_json(&value).unwrap(),
            r#"{"a":[3,{"x":4,"y":5}],"z":{"a":1,"b":2}}"#
        );
    }
}
