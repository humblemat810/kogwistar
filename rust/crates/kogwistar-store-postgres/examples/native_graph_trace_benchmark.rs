use kogwistar_store::{
    GraphMutation, GraphProjectionRead, GraphRecord, GraphScope, MetadataFilter,
};
use kogwistar_store_postgres::{GraphTableNames, PostgresStore};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

type BenchResult<T> = Result<T, Box<dyn Error>>;

fn validate_trace_document(trace: &Value) -> Result<(), String> {
    if trace["schema"] != "kogwistar-native-graph-backend-trace/v1" {
        return Err("unsupported native graph trace schema".to_owned());
    }
    if trace["trace_id"].as_str().is_none_or(str::is_empty)
        || trace["root_payload"]["id"]
            .as_str()
            .is_none_or(str::is_empty)
    {
        return Err("trace identity and root payload id are required".to_owned());
    }
    let root_id = trace["root_payload"]["id"]
        .as_str()
        .expect("validated root id");
    let operations = trace["operations"]
        .as_array()
        .ok_or_else(|| "trace operations must be an array".to_owned())?;
    if operations.is_empty() {
        return Err("trace operations must not be empty".to_owned());
    }
    let mut writes = std::collections::BTreeSet::new();
    for operation in operations {
        let operation_name = operation["operation"]
            .as_str()
            .ok_or_else(|| "trace operation name is required".to_owned())?;
        let index = operation["index"]
            .as_u64()
            .ok_or_else(|| "trace operation index must be non-negative".to_owned())?;
        let entity_id = operation["expected_entity_id"]
            .as_str()
            .ok_or_else(|| "expected_entity_id is required".to_owned())?;
        let edge_id = operation["expected_edge_id"]
            .as_str()
            .ok_or_else(|| "expected_edge_id is required".to_owned())?;
        let source_id = operation["expected_source_id"]
            .as_str()
            .ok_or_else(|| "expected_source_id is required".to_owned())?;
        let key = (index, entity_id.to_owned());
        match operation_name {
            "write_record" => {
                if operation["node_payload"]["id"] != entity_id {
                    return Err("node_payload id differs from expected_entity_id".to_owned());
                }
                if operation["edge_payload"]["id"] != edge_id {
                    return Err("edge_payload id differs from expected_edge_id".to_owned());
                }
                if operation["edge_payload"]["source_ids"][0] != source_id
                    || operation["edge_payload"]["target_ids"][0] != entity_id
                {
                    return Err("edge_payload source_ids/target_ids differ from trace".to_owned());
                }
                if !operation["vector"].is_array()
                    || operation["vector"].as_array().map(Vec::len)
                        != operation["vector_dimension"].as_u64().map(|v| v as usize)
                {
                    return Err("vector length differs from vector_dimension".to_owned());
                }
                writes.insert(key);
            }
            "read_event" | "read_projection" => {
                if !writes.contains(&key) {
                    return Err(format!("{operation_name} occurs before write_record"));
                }
            }
            other => return Err(format!("unsupported trace operation {other:?}")),
        }
    }
    if root_id.is_empty() {
        return Err("root payload id must not be empty".to_owned());
    }
    let expected_hash = trace["trace_sha256"]
        .as_str()
        .ok_or_else(|| "trace digest is required".to_owned())?;
    let mut body = trace.clone();
    body.as_object_mut()
        .ok_or_else(|| "trace document must be an object".to_owned())?
        .remove("trace_sha256");
    let actual_hash = format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&body).map_err(|error| error.to_string())?)
    );
    if actual_hash != expected_hash {
        return Err("trace digest does not match canonical trace document".to_owned());
    }
    Ok(())
}

fn projection_mutation(
    scope: &GraphScope,
    trace_id: &str,
    kind: &str,
    table: &str,
    payload: &Value,
    vector: Option<Vec<f32>>,
    embedding_dim: usize,
) -> GraphMutation {
    let id = payload["id"].as_str().expect("trace payload id validated");
    let metadata = payload["metadata"].as_object().cloned().unwrap_or_default();
    GraphMutation {
        scope: scope.clone(),
        table: table.to_owned(),
        entity_kind: kind.to_owned(),
        event_id: format!("{trace_id}:{kind}:{id}"),
        op: "ADD".to_owned(),
        payload: payload.clone(),
        record: GraphRecord {
            id: id.to_owned(),
            document: Some(payload.to_string()),
            metadata,
            embedding: vector,
        },
        embedding_dim,
    }
}

fn percentile95(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[((sorted.len() as f64 * 0.95).ceil() as usize).saturating_sub(1)]
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

#[cfg(target_os = "linux")]
fn process_cpu_time_ms() -> Option<f64> {
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) } != 0 {
        return None;
    }
    let user_ms = usage.ru_utime.tv_sec as f64 * 1_000.0 + usage.ru_utime.tv_usec as f64 / 1_000.0;
    let system_ms =
        usage.ru_stime.tv_sec as f64 * 1_000.0 + usage.ru_stime.tv_usec as f64 / 1_000.0;
    Some(user_ms + system_ms)
}

#[cfg(not(target_os = "linux"))]
fn process_cpu_time_ms() -> Option<f64> {
    None
}

#[cfg(not(target_os = "linux"))]
fn peak_rss_bytes() -> Option<u64> {
    None
}

async fn drop_schema(dsn: &str, schema: &str) -> BenchResult<()> {
    let config = dsn.parse::<tokio_postgres::Config>()?;
    let (client, connection) = config.connect(tokio_postgres::NoTls).await?;
    tokio::spawn(async move {
        let _ = connection.await;
    });
    client
        .batch_execute(&format!("DROP SCHEMA IF EXISTS \"{schema}\" CASCADE"))
        .await?;
    Ok(())
}

async fn execute_trace(store: &PostgresStore, trace: &Value) -> BenchResult<Value> {
    let trace_id = trace["trace_id"].as_str().expect("validated trace id");
    let trace_sha256 = trace["trace_sha256"].as_str().expect("validated digest");
    let operations = trace["operations"]
        .as_array()
        .expect("validated operations");
    let dimension = operations[0]["vector_dimension"].as_u64().unwrap_or(0) as usize;
    if dimension == 0 {
        return Err("trace must declare a positive vector dimension".into());
    }
    let scope = GraphScope {
        namespace: format!("bench-{trace_id}"),
        workspace_id: None,
        graph_space: None,
    };
    let mut root = trace["root_payload"].clone();
    if !root["metadata"].is_object() {
        root["metadata"] = serde_json::json!({});
    }
    let root_mutation = projection_mutation(
        &scope,
        trace_id,
        "node",
        "gke_nodes",
        &root,
        None,
        dimension,
    );
    store.apply_graph_mutation(root_mutation).await?;

    let mut latency: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    let wall_start = Instant::now();
    let cpu_start_ms = process_cpu_time_ms();
    let mut actual_rows = BTreeMap::<String, Value>::new();
    for operation in operations {
        let name = operation["operation"]
            .as_str()
            .expect("validated operation");
        let started = Instant::now();
        match name {
            "write_record" => {
                let vector = operation["vector"]
                    .as_array()
                    .ok_or("trace vector must be an array")?
                    .iter()
                    .map(|value| {
                        value
                            .as_f64()
                            .map(|number| number as f32)
                            .ok_or("invalid vector value")
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let node = projection_mutation(
                    &scope,
                    trace_id,
                    "node",
                    "gke_nodes",
                    &operation["node_payload"],
                    Some(vector.clone()),
                    dimension,
                );
                let edge = projection_mutation(
                    &scope,
                    trace_id,
                    "edge",
                    "gke_edges",
                    &operation["edge_payload"],
                    Some(vector),
                    dimension,
                );
                store.apply_graph_mutations_atomic(vec![node, edge]).await?;
            }
            "read_event" => {
                for (kind, id) in [
                    ("node", operation["expected_entity_id"].as_str().unwrap()),
                    ("edge", operation["expected_edge_id"].as_str().unwrap()),
                ] {
                    let found = store
                        .latest_entity_events(&scope.namespace, kind, &[id.to_owned()])
                        .await?;
                    let event = found
                        .get(id)
                        .ok_or("expected canonical graph event missing")?;
                    let payload: Value = serde_json::from_str(&event.payload_json)?;
                    if payload["id"] != id {
                        return Err("canonical event entity id mismatch".into());
                    }
                    if kind == "edge"
                        && (payload["source_ids"][0] != operation["expected_source_id"]
                            || payload["target_ids"][0] != operation["expected_entity_id"])
                    {
                        return Err("canonical edge event endpoint mismatch".into());
                    }
                }
            }
            "read_projection" => {
                for (kind, table, id) in [
                    (
                        "node",
                        "gke_nodes",
                        operation["expected_entity_id"].as_str().unwrap(),
                    ),
                    (
                        "edge",
                        "gke_edges",
                        operation["expected_edge_id"].as_str().unwrap(),
                    ),
                ] {
                    let rows = store
                        .graph_projection_records(GraphProjectionRead {
                            scope: scope.clone(),
                            table: table.to_owned(),
                            ids: Some(vec![id.to_owned()]),
                            metadata: MetadataFilter::default(),
                            limit: 1,
                        })
                        .await?;
                    let row = rows.first().ok_or("expected graph projection missing")?;
                    if row.id != id {
                        return Err("graph projection entity id mismatch".into());
                    }
                    let payload: Value = serde_json::from_str(
                        row.document
                            .as_deref()
                            .ok_or("graph projection document missing")?,
                    )?;
                    if kind == "edge"
                        && (payload["source_ids"][0] != operation["expected_source_id"]
                            || payload["target_ids"][0] != operation["expected_entity_id"])
                    {
                        return Err("graph projection edge endpoint mismatch".into());
                    }
                    actual_rows.insert(format!("{kind}:{id}"), payload);
                }
            }
            _ => return Err(format!("unsupported operation {name:?}").into()),
        }
        latency
            .entry(name)
            .or_default()
            .push(started.elapsed().as_secs_f64() * 1000.0);
    }
    let replayed = store
        .replay_raw_events(&scope.namespace, 0, usize::MAX)
        .await?;
    if replayed.len() != 1 + 2 * operations.len() / 3 {
        return Err("canonical event count differs from expected graph writes".into());
    }
    let root_id = root["id"].as_str().ok_or("root id missing")?;
    actual_rows.insert(format!("node:{root_id}"), root);
    let canonical_state = serde_json::to_vec(&actual_rows)?;
    let elapsed_ms = wall_start.elapsed().as_secs_f64() * 1000.0;
    let cpu_ms = process_cpu_time_ms()
        .zip(cpu_start_ms)
        .map(|(end, start)| end - start);
    Ok(serde_json::json!({
        "trace_id": trace_id,
        "trace_sha256": trace_sha256,
        "state_digest": format!("{:x}", Sha256::digest(canonical_state)),
        "canonical_event_count": replayed.len(),
        "peak_rss_bytes": peak_rss_bytes(),
        "projection_converged": true,
        "elapsed_ms": elapsed_ms,
        "cpu_ms": cpu_ms,
        "throughput_ops_per_sec": operations.len() as f64 * 1000.0 / elapsed_ms,
        "operation_p95_ms": latency.iter().map(|(name, values)| (*name, percentile95(values))).collect::<BTreeMap<_, _>>(),
        "resource_limits_verified": std::env::var("KOGWISTAR_BENCH_RESOURCE_LIMITS_VERIFIED").as_deref() == Ok("1"),
        "ffi_calls": 0,
    }))
}

#[tokio::main]
async fn main() -> BenchResult<()> {
    let mut args = std::env::args().skip(1);
    let trace_path = PathBuf::from(args.next().ok_or("expected shared trace JSON path")?);
    let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")?;
    let schema = args.next().ok_or("expected disposable PostgreSQL schema")?;
    if args.next().is_some()
        || schema.is_empty()
        || !schema
            .chars()
            .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '_')
    {
        return Err("expected: <trace-json> <lowercase-disposable-schema>".into());
    }
    let trace: Value = serde_json::from_slice(&fs::read(trace_path)?)?;
    validate_trace_document(&trace).map_err(std::io::Error::other)?;
    let outcome = async {
        let store = PostgresStore::from_dsn(&dsn, &schema)?;
        store.ensure_schema().await?;
        store
            .create_graph_schema(
                trace["operations"][0]["vector_dimension"]
                    .as_u64()
                    .ok_or("vector dimension missing")? as usize,
                GraphTableNames::default(),
            )
            .await?;
        let result = execute_trace(&store, &trace).await;
        drop(store);
        result
    }
    .await;
    let cleanup = drop_schema(&dsn, &schema).await;
    match (outcome, cleanup) {
        (Ok(result), Ok(())) => {
            println!("{}", result);
            Ok(())
        }
        (Err(error), Ok(())) => Err(error),
        (Ok(_), Err(error)) => Err(error),
        (Err(run_error), Err(cleanup_error)) => {
            Err(format!("{run_error}; schema cleanup failed: {cleanup_error}").into())
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};
    use sha2::{Digest, Sha256};

    use super::validate_trace_document;

    fn trace() -> Value {
        let mut value = json!({
            "schema": "kogwistar-native-graph-backend-trace/v1",
            "trace_id": "native-graph-1",
            "trace_sha256": "",
            "root_payload": {"id": "mixed-node-root"},
            "operations": [
                {
                    "operation": "write_record",
                    "index": 0,
                    "expected_entity_id": "mixed-node-00000",
                    "expected_edge_id": "mixed-edge-00000",
                    "expected_source_id": "mixed-node-root",
                    "node_payload": {"id": "mixed-node-00000", "metadata": {}},
                    "edge_payload": {
                        "id": "mixed-edge-00000",
                        "source_ids": ["mixed-node-root"],
                        "target_ids": ["mixed-node-00000"],
                        "metadata": {}
                    },
                    "vector_dimension": 2,
                    "vector": [0.25, 0.75]
                },
                {
                    "operation": "read_event",
                    "index": 0,
                    "expected_entity_id": "mixed-node-00000",
                    "expected_edge_id": "mixed-edge-00000",
                    "expected_source_id": "mixed-node-root"
                },
                {
                    "operation": "read_projection",
                    "index": 0,
                    "expected_entity_id": "mixed-node-00000",
                    "expected_edge_id": "mixed-edge-00000",
                    "expected_source_id": "mixed-node-root"
                }
            ]
        });
        value.as_object_mut().unwrap().remove("trace_sha256");
        let canonical = serde_json::to_vec(&value).unwrap();
        value["trace_sha256"] = json!(format!("{:x}", Sha256::digest(canonical)));
        value
    }

    #[test]
    fn shared_trace_contract_accepts_matching_graph_records() {
        assert!(validate_trace_document(&trace()).is_ok());
    }

    #[test]
    fn shared_trace_contract_rejects_mismatched_edge_endpoints() {
        let mut value = trace();
        value["operations"][0]["edge_payload"]["target_ids"] = json!(["another-node"]);

        let error = validate_trace_document(&value).unwrap_err();
        assert!(error.contains("target_ids"));
    }

    #[test]
    fn shared_trace_contract_rejects_reads_before_write() {
        let mut value = trace();
        value["operations"].as_array_mut().unwrap().swap(0, 1);

        let error = validate_trace_document(&value).unwrap_err();
        assert!(error.contains("before write_record"));
    }
}
