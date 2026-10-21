use kogwistar_store::NamedProjectionWrite;
use kogwistar_store_sqlite::{NewRawEntityEvent, SqliteStore};
use serde_json::{Map, Value, json};
use std::env;
use std::error::Error;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

type BenchResult<T> = Result<T, Box<dyn Error>>;

fn measure<F>(operation: F) -> BenchResult<Value>
where
    F: FnOnce() -> BenchResult<usize>,
{
    let started = Instant::now();
    let operations = operation()?;
    let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
    Ok(json!({
        "elapsed_ms": elapsed_ms,
        "operations": operations,
        "throughput_ops_per_sec": operations as f64 * 1_000.0 / elapsed_ms,
    }))
}

fn main() -> BenchResult<()> {
    let mut args = env::args().skip(1);
    let path = PathBuf::from(args.next().ok_or("expected database path")?);
    let items: usize = args.next().ok_or("expected item count")?.parse()?;
    let read_cycles: usize = args.next().ok_or("expected read-cycle count")?.parse()?;
    if args.next().is_some() || items == 0 || read_cycles == 0 {
        return Err("expected: <database-path> <items> <read-cycles>".into());
    }

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
    let database_bytes = fs::metadata(&path)?.len();

    println!(
        "{}",
        json!({
            "metrics": {
                "event_append": event_append,
                "event_replay": event_replay,
                "projection_replace": projection_replace,
                "projection_list": projection_list,
            },
            "state": {"events": events, "projections": projections},
            "database_bytes": database_bytes,
        })
    );
    Ok(())
}
