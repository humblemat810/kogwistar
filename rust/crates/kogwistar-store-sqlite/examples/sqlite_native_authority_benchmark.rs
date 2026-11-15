use kogwistar_store::NamedProjectionWrite;
use kogwistar_store_sqlite::{NewRawEntityEvent, SqliteStore};
use serde::Deserialize;
use serde_json::{Map, Value, json};
use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

type BenchResult<T> = Result<T, Box<dyn Error>>;

#[derive(Debug, Deserialize)]
struct WorkloadTrace {
    schema: String,
    profile: String,
    operations: Vec<TraceOperation>,
}

#[derive(Debug, Deserialize)]
struct TraceOperation {
    operation: String,
    index: usize,
    expected_entity_id: String,
}

fn validate_trace_contract(trace: &WorkloadTrace, items: usize) -> BenchResult<()> {
    if !matches!(
        trace.schema.as_str(),
        "kogwistar-mixed-rewrite-trace/v1" | "kogwistar-mixed-rewrite-trace/v2"
    ) || trace.profile != "read_after_write"
        || trace.operations.len() != items * 3
    {
        return Err("unsupported or malformed mixed workload trace".into());
    }
    Ok(())
}

fn measure<F>(operation: F) -> BenchResult<Value>
where
    F: FnOnce() -> BenchResult<usize>,
{
    let started = Instant::now();
    let operations = operation()?;
    let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
    Ok(json!({
        "elapsed_ms": elapsed_ms,
        "cpu_elapsed_ms": Value::Null,
        "operations": operations,
        "throughput_ops_per_sec": operations as f64 * 1_000.0 / elapsed_ms,
    }))
}

fn database_footprint_bytes(path: &Path) -> BenchResult<u64> {
    let mut total = 0;
    for suffix in ["", "-wal", "-shm"] {
        let candidate = if suffix.is_empty() {
            path.to_path_buf()
        } else {
            PathBuf::from(format!("{}{}", path.display(), suffix))
        };
        if let Ok(metadata) = fs::metadata(candidate) {
            total += metadata.len();
        }
    }
    Ok(total)
}

fn main() -> BenchResult<()> {
    let mut args = env::args().skip(1);
    let path = PathBuf::from(args.next().ok_or("expected database path")?);
    let items: usize = args.next().ok_or("expected item count")?.parse()?;
    let read_cycles: usize = args.next().ok_or("expected read-cycle count")?.parse()?;
    let trace_path = PathBuf::from(args.next().ok_or("expected workload trace path")?);
    if args.next().is_some() || items == 0 || read_cycles == 0 {
        return Err("expected: <database-path> <items> <read-cycles> <trace-path>".into());
    }
    let trace: WorkloadTrace = serde_json::from_str(&fs::read_to_string(trace_path)?)?;
    // The Python workload generator is the contract owner. v2 keeps the same
    // operation semantics as the original native benchmark while adding
    // explicit graph endpoint fields used by the shared fixture.
    validate_trace_contract(&trace, items)?;

    let store = SqliteStore::open(&path)?;
    let event_namespace = "benchmark";
    let projection_namespace = "benchmark_projection";

    let event_append = measure(|| {
        for index in 0..items {
            let entity_id = format!("node-{index:05}");
            store.append_raw_entity_event(
                event_namespace,
                NewRawEntityEvent {
                    event_id: format!("event-{index:05}"),
                    entity_kind: "node".to_owned(),
                    entity_id: entity_id.clone(),
                    op: "UPSERT".to_owned(),
                    payload_json: json!({"replacement": {"id": entity_id}}).to_string(),
                },
            )?;
        }
        Ok(items)
    })?;

    let event_replay = measure(|| {
        let mut count = 0;
        for _ in 0..read_cycles {
            count += store
                .replay_raw_events(event_namespace, 0, items + 1)?
                .len();
        }
        Ok(count)
    })?;

    let projection_replace = measure(|| {
        for index in 0..items {
            let mut payload = Map::new();
            payload.insert("index".to_owned(), json!(index));
            payload.insert("label".to_owned(), json!(format!("row-{index:05}")));
            store.replace_named_projection(
                projection_namespace,
                &format!("key-{index:05}"),
                NamedProjectionWrite {
                    payload,
                    last_authoritative_seq: index as i64 + 1,
                    last_materialized_seq: index as i64 + 1,
                    projection_schema_version: 1,
                    materialization_status: "ready".to_owned(),
                },
            )?;
        }
        Ok(items)
    })?;

    let projection_list = measure(|| {
        let mut count = 0;
        for _ in 0..read_cycles {
            count += store.list_named_projections(projection_namespace)?.len();
        }
        Ok(count)
    })?;

    let mixed_interleaved = measure(|| {
        let mut operations = 0;
        for step in &trace.operations {
            let index = step.index;
            let projection_key = format!("mixed-key-{index:05}");
            match step.operation.as_str() {
                "write_record" => {
                    let entity_id = step.expected_entity_id.clone();
                    store.immediate_transaction(|uow| {
                        uow.append_raw_entity_event(
                            "mixed_interleaved",
                            NewRawEntityEvent {
                                event_id: format!("mixed-event-{index:05}"),
                                entity_kind: "node".to_owned(),
                                entity_id: entity_id.clone(),
                                op: "UPSERT".to_owned(),
                                payload_json: json!({"id": entity_id, "revision": 1}).to_string(),
                            },
                        )?;
                        let payload = json!({"index": index, "label": format!("row-{index:05}")})
                            .as_object()
                            .cloned()
                            .unwrap_or_default();
                        uow.replace_named_projection(
                            "mixed_interleaved_projection",
                            &projection_key,
                            NamedProjectionWrite {
                                payload,
                                last_authoritative_seq: index as i64 + 1,
                                last_materialized_seq: index as i64 + 1,
                                projection_schema_version: 1,
                                materialization_status: "ready".to_owned(),
                            },
                        )?;
                        Ok(())
                    })?;
                }
                "read_event" => {
                    let event_rows =
                        store.replay_raw_events("mixed_interleaved", index as i64, 1)?;
                    if event_rows.len() != 1 || event_rows[0].entity_id != step.expected_entity_id {
                        return Err("interleaved event read did not observe its write".into());
                    }
                }
                "read_projection" => {
                    let projection = store
                        .get_named_projection("mixed_interleaved_projection", &projection_key)?;
                    if projection
                        .as_ref()
                        .and_then(|row| row.payload.get("index"))
                        .and_then(Value::as_u64)
                        != Some(index as u64)
                    {
                        return Err("interleaved projection read did not observe its write".into());
                    }
                }
                operation => {
                    return Err(format!("unknown workload operation {operation:?}").into());
                }
            }
            operations += 1;
        }
        Ok(operations)
    })?;

    let events: Vec<Value> = store
        .replay_raw_events(event_namespace, 0, items + 1)?
        .into_iter()
        .map(|event| {
            json!([
                event.seq,
                event.entity_kind,
                event.entity_id,
                event.op,
                event.payload_json
            ])
        })
        .collect();
    let projections: Vec<Value> = store
        .list_named_projections(projection_namespace)?
        .into_iter()
        .map(|projection| {
            json!({
                "namespace": projection.namespace,
                "key": projection.key,
                "payload": projection.payload,
                "last_authoritative_seq": projection.last_authoritative_seq,
                "last_materialized_seq": projection.last_materialized_seq,
                "projection_schema_version": projection.projection_schema_version,
                "materialization_status": projection.materialization_status,
            })
        })
        .collect();
    let mixed_events: Vec<Value> = store
        .replay_raw_events("mixed_interleaved", 0, items + 1)?
        .into_iter()
        .map(|event| {
            json!([
                event.seq,
                event.entity_kind,
                event.entity_id,
                event.op,
                event.payload_json
            ])
        })
        .collect();
    let mixed_projections: Vec<Value> = store
        .list_named_projections("mixed_interleaved_projection")?
        .into_iter()
        .map(|projection| {
            json!({
                "namespace": projection.namespace,
                "key": projection.key,
                "payload": projection.payload,
                "last_authoritative_seq": projection.last_authoritative_seq,
                "last_materialized_seq": projection.last_materialized_seq,
                "projection_schema_version": projection.projection_schema_version,
                "materialization_status": projection.materialization_status,
            })
        })
        .collect();
    let database_bytes = database_footprint_bytes(&path)?;

    println!(
        "{}",
        json!({
            "metrics": {
                "event_append": event_append,
                "event_replay": event_replay,
                "projection_replace": projection_replace,
                "projection_list": projection_list,
                "mixed_interleaved": mixed_interleaved,
            },
            "state": {
                "events": events,
                "projections": projections,
                "mixed_events": mixed_events,
                "mixed_projections": mixed_projections,
            },
            "database_bytes": database_bytes,
        })
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{TraceOperation, WorkloadTrace, validate_trace_contract};

    fn trace(schema: &str) -> WorkloadTrace {
        WorkloadTrace {
            schema: schema.to_owned(),
            profile: "read_after_write".to_owned(),
            operations: vec![
                TraceOperation {
                    operation: "write_record".to_owned(),
                    index: 0,
                    expected_entity_id: "mixed-node-00000".to_owned(),
                },
                TraceOperation {
                    operation: "read_event".to_owned(),
                    index: 0,
                    expected_entity_id: "mixed-node-00000".to_owned(),
                },
                TraceOperation {
                    operation: "read_projection".to_owned(),
                    index: 0,
                    expected_entity_id: "mixed-node-00000".to_owned(),
                },
            ],
        }
    }

    #[test]
    fn trace_contract_accepts_current_and_legacy_versions() {
        assert!(validate_trace_contract(&trace("kogwistar-mixed-rewrite-trace/v1"), 1).is_ok());
        assert!(validate_trace_contract(&trace("kogwistar-mixed-rewrite-trace/v2"), 1).is_ok());
    }

    #[test]
    fn trace_contract_rejects_unknown_versions() {
        assert!(validate_trace_contract(&trace("kogwistar-mixed-rewrite-trace/v3"), 1).is_err());
    }
}
