use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    routing::{get, post},
};
use kogwistar_store::NamedProjectionWrite;
use kogwistar_store_sqlite::{NewRawEntityEvent, SqliteStore};
use serde_json::{Map, Value, json};
use std::{
    collections::{BTreeMap, HashMap},
    env,
    net::SocketAddr,
    path::PathBuf,
    sync::{Arc, Barrier, Condvar, Mutex},
};

fn canonical_json(value: &Value) -> String {
    match value {
        Value::Null | Value::Bool(_) | Value::Number(_) => value.to_string(),
        Value::String(text) => {
            serde_json::to_string(text).expect("string serialization is infallible")
        }
        Value::Array(items) => format!(
            "[{}]",
            items
                .iter()
                .map(canonical_json)
                .collect::<Vec<_>>()
                .join(",")
        ),
        Value::Object(object) => {
            let sorted: BTreeMap<&String, &Value> = object.iter().collect();
            format!(
                "{{{}}}",
                sorted
                    .iter()
                    .map(|(key, value)| format!(
                        "{}:{}",
                        serde_json::to_string(key).expect("key serialization is infallible"),
                        canonical_json(value)
                    ))
                    .collect::<Vec<_>>()
                    .join(",")
            )
        }
    }
}

#[derive(Clone)]
struct AppState {
    store: Arc<SqliteStore>,
    database: Arc<PathBuf>,
    concurrent: Option<Arc<ConcurrentCoordinator>>,
}

struct ConcurrentCoordinator {
    steps: Mutex<HashMap<usize, Arc<ConcurrentStep>>>,
}

struct ConcurrentStep {
    before_commit: Barrier,
    after_commit: Barrier,
    committed: (Mutex<bool>, Condvar),
    before_readers: usize,
    after_readers: usize,
}

impl ConcurrentCoordinator {
    fn new() -> Self {
        Self {
            steps: Mutex::new(HashMap::new()),
        }
    }

    fn step(
        &self,
        index: usize,
        before_readers: usize,
        after_readers: usize,
    ) -> Result<Arc<ConcurrentStep>, String> {
        if before_readers > 4 || after_readers > 4 {
            return Err("reader counts must be between zero and four".to_owned());
        }
        let mut steps = self
            .steps
            .lock()
            .map_err(|_| "concurrent step registry is poisoned".to_owned())?;
        if let Some(step) = steps.get(&index) {
            if step.before_readers != before_readers || step.after_readers != after_readers {
                return Err("reader counts do not match the active trace step".to_owned());
            }
            return Ok(Arc::clone(step));
        }
        let step = Arc::new(ConcurrentStep {
            before_commit: Barrier::new(before_readers + 1),
            after_commit: Barrier::new(after_readers + 1),
            committed: (Mutex::new(false), Condvar::new()),
            before_readers,
            after_readers,
        });
        steps.insert(index, Arc::clone(&step));
        Ok(step)
    }

    fn finish_step(&self, index: usize) -> Result<(), String> {
        self.steps
            .lock()
            .map_err(|_| "concurrent step registry is poisoned".to_owned())?
            .remove(&index);
        Ok(())
    }

    fn publish_commit(step: &ConcurrentStep) -> Result<(), String> {
        let (committed, ready) = &step.committed;
        let mut state = committed
            .lock()
            .map_err(|_| "commit signal is poisoned".to_owned())?;
        *state = true;
        ready.notify_all();
        Ok(())
    }

    fn await_commit(step: &ConcurrentStep) -> Result<(), String> {
        let (committed, ready) = &step.committed;
        let mut state = committed
            .lock()
            .map_err(|_| "commit signal is poisoned".to_owned())?;
        while !*state {
            state = ready
                .wait(state)
                .map_err(|_| "commit signal is poisoned".to_owned())?;
        }
        Ok(())
    }
}

fn concurrent_tool_call(
    state: AppState,
    name: String,
    arguments: Map<String, Value>,
) -> Result<Value, String> {
    let coordinator = state
        .concurrent
        .as_ref()
        .ok_or_else(|| "concurrent MCP mode is not enabled".to_owned())?;
    match name.as_str() {
        "write_record" => {
            let index = arguments
                .get("index")
                .and_then(Value::as_u64)
                .ok_or_else(|| "index must be a non-negative integer".to_owned())?
                as usize;
            let entity_id = arguments
                .get("expected_entity_id")
                .and_then(Value::as_str)
                .ok_or_else(|| "expected_entity_id is required".to_owned())?
                .to_owned();
            let before_readers = arguments
                .get("before_readers")
                .and_then(Value::as_u64)
                .ok_or_else(|| "before_readers must be an integer".to_owned())?
                as usize;
            let after_readers = arguments
                .get("after_readers")
                .and_then(Value::as_u64)
                .ok_or_else(|| "after_readers must be an integer".to_owned())?
                as usize;
            let step = coordinator.step(index, before_readers, after_readers)?;
            let edge_id = arguments
                .get("expected_edge_id")
                .and_then(Value::as_str)
                .ok_or_else(|| "expected_edge_id is required".to_owned())?
                .to_owned();
            let source_id = arguments
                .get("expected_source_id")
                .and_then(Value::as_str)
                .ok_or_else(|| "expected_source_id is required".to_owned())?
                .to_owned();
            let node_payload = arguments
                .get("node_payload")
                .filter(|value| value.get("id").and_then(Value::as_str) == Some(&entity_id))
                .ok_or_else(|| "node_payload does not match expected_entity_id".to_owned())?
                .clone();
            let edge_payload = arguments
                .get("edge_payload")
                .filter(|value| {
                    value.get("id").and_then(Value::as_str) == Some(&edge_id)
                        && value.get("source_ids").and_then(Value::as_array)
                            == Some(&vec![json!(source_id)])
                        && value.get("target_ids").and_then(Value::as_array)
                            == Some(&vec![json!(entity_id)])
                })
                .ok_or_else(|| "edge_payload does not match expected graph endpoints".to_owned())?
                .clone();
            let entity_for_projection = entity_id.clone();
            let edge_for_projection = edge_id.clone();
            let source_for_projection = source_id.clone();
            let node_json = canonical_json(&node_payload);
            let edge_json = canonical_json(&edge_payload);
            let (node_seq, edge_seq) = state
                .store
                .immediate_transaction(|uow| {
                    let node_event = uow.append_raw_entity_event(
                        "mcp_benchmark",
                        NewRawEntityEvent {
                            event_id: format!("mcp-node-event-{index:05}"),
                            entity_kind: "node".to_owned(),
                            entity_id: entity_id.clone(),
                            op: "UPSERT".to_owned(),
                            payload_json: node_json,
                        },
                    )?;
                    let edge_event = uow.append_raw_entity_event(
                        "mcp_benchmark",
                        NewRawEntityEvent {
                            event_id: format!("mcp-edge-event-{index:05}"),
                            entity_kind: "edge".to_owned(),
                            entity_id: edge_id.clone(),
                            op: "UPSERT".to_owned(),
                            payload_json: edge_json,
                        },
                    )?;
                    uow.replace_named_projection(
                        "mcp_benchmark_projection",
                        &format!("node:{entity_for_projection}"),
                        NamedProjectionWrite {
                            payload: Map::from_iter([
                                ("entity_kind".to_owned(), json!("node")),
                                ("entity_id".to_owned(), json!(entity_for_projection)),
                                ("entity".to_owned(), node_payload),
                            ]),
                            last_authoritative_seq: node_event.event.seq,
                            last_materialized_seq: node_event.event.seq,
                            projection_schema_version: 1,
                            materialization_status: "ready".to_owned(),
                        },
                    )?;
                    uow.replace_named_projection(
                        "mcp_benchmark_projection",
                        &format!("edge:{edge_for_projection}"),
                        NamedProjectionWrite {
                            payload: Map::from_iter([
                                ("entity_kind".to_owned(), json!("edge")),
                                ("entity_id".to_owned(), json!(edge_for_projection)),
                                ("entity".to_owned(), edge_payload),
                            ]),
                            last_authoritative_seq: edge_event.event.seq,
                            last_materialized_seq: edge_event.event.seq,
                            projection_schema_version: 1,
                            materialization_status: "ready".to_owned(),
                        },
                    )?;
                    uow.replace_named_projection(
                        "mcp_benchmark_projection",
                        "mcp-head",
                        NamedProjectionWrite {
                            payload: Map::from_iter([
                                ("revision".to_owned(), json!(index + 1)),
                                ("entity_id".to_owned(), json!(entity_id)),
                                ("edge_id".to_owned(), json!(edge_id)),
                                (
                                    "previous_entity_id".to_owned(),
                                    json!(source_for_projection),
                                ),
                                ("node_seq".to_owned(), json!(node_event.event.seq)),
                                ("edge_seq".to_owned(), json!(edge_event.event.seq)),
                            ]),
                            last_authoritative_seq: edge_event.event.seq,
                            last_materialized_seq: edge_event.event.seq,
                            projection_schema_version: 1,
                            materialization_status: "ready".to_owned(),
                        },
                    )?;
                    step.before_commit.wait();
                    Ok((node_event.event.seq, edge_event.event.seq))
                })
                .map_err(|error| error.to_string())?;
            ConcurrentCoordinator::publish_commit(&step)?;
            step.after_commit.wait();
            coordinator.finish_step(index)?;
            Ok(
                json!({"seq": edge_seq, "node_seq": node_seq, "entity_id": entity_id, "edge_id": edge_id}),
            )
        }
        "read_head" => {
            let index = arguments
                .get("index")
                .and_then(Value::as_u64)
                .ok_or_else(|| "index must be a non-negative integer".to_owned())?
                as usize;
            let before_readers = arguments
                .get("before_readers")
                .and_then(Value::as_u64)
                .ok_or_else(|| "before_readers must be an integer".to_owned())?
                as usize;
            let after_readers = arguments
                .get("after_readers")
                .and_then(Value::as_u64)
                .ok_or_else(|| "after_readers must be an integer".to_owned())?
                as usize;
            let step = coordinator.step(index, before_readers, after_readers)?;
            let visibility = arguments
                .get("visibility")
                .and_then(Value::as_str)
                .ok_or_else(|| "visibility is required".to_owned())?;
            if !matches!(visibility, "before_commit" | "after_commit") {
                return Err("unsupported head visibility phase".to_owned());
            }
            let expected_revision = arguments
                .get("expected_revision")
                .and_then(Value::as_u64)
                .ok_or_else(|| "expected_revision must be non-negative".to_owned())?;
            let observed_revision = if visibility == "before_commit" {
                let observed = read_and_validate_head(&state.database, expected_revision)?;
                step.before_commit.wait();
                observed
            } else {
                ConcurrentCoordinator::await_commit(&step)?;
                let observed = read_and_validate_head(&state.database, expected_revision)?;
                step.after_commit.wait();
                observed
            };
            Ok(json!({"revision": observed_revision}))
        }
        _ => Err("unknown concurrent benchmark tool".to_owned()),
    }
}

fn read_and_validate_head(database: &PathBuf, expected_revision: u64) -> Result<u64, String> {
    let reader = SqliteStore::open(database).map_err(|error| error.to_string())?;
    let projection = reader
        .get_named_projection("mcp_benchmark_projection", "mcp-head")
        .map_err(|error| error.to_string())?;
    let revision = projection
        .as_ref()
        .and_then(|row| row.payload.get("revision"))
        .and_then(Value::as_u64)
        .ok_or_else(|| "head projection is missing a valid revision".to_owned())?;
    if revision != expected_revision {
        return Err(format!(
            "native reader observed revision {revision}; expected {expected_revision}"
        ));
    }
    let payload = projection
        .as_ref()
        .ok_or_else(|| "head projection disappeared".to_owned())?
        .payload
        .clone();
    let expected_entity = if revision == 0 {
        "mixed-node-root".to_owned()
    } else {
        format!("mixed-node-{:05}", revision - 1)
    };
    let expected_edge = if revision == 0 {
        Value::Null
    } else {
        json!(format!("mixed-edge-{:05}", revision - 1))
    };
    let expected_previous = if revision == 0 {
        Value::Null
    } else if revision == 1 {
        json!("mixed-node-root")
    } else {
        json!(format!("mixed-node-{:05}", revision - 2))
    };
    let expected_node_seq = if revision == 0 {
        1
    } else {
        (2 * revision) as i64
    };
    let expected_edge_seq = if revision == 0 {
        0
    } else {
        expected_node_seq + 1
    };
    if payload.get("entity_id").and_then(Value::as_str) != Some(&expected_entity)
        || payload.get("edge_id") != Some(&expected_edge)
        || payload.get("previous_entity_id") != Some(&expected_previous)
        || payload.get("node_seq").and_then(Value::as_i64) != Some(expected_node_seq)
        || payload.get("edge_seq").and_then(Value::as_i64) != Some(expected_edge_seq)
    {
        return Err("head projection graph fields do not match revision".to_owned());
    }
    if revision == 0 {
        let events = reader
            .replay_raw_events("mcp_benchmark", 0, 1)
            .map_err(|error| error.to_string())?;
        if events.len() != 1
            || events[0].seq != 1
            || events[0].entity_kind != "node"
            || events[0].entity_id != "mixed-node-root"
        {
            return Err("genesis graph node is missing or malformed".to_owned());
        }
    }
    if revision > 0 {
        let payload = &projection
            .as_ref()
            .ok_or_else(|| "head projection disappeared".to_owned())?
            .payload;
        let entity_id = payload
            .get("entity_id")
            .and_then(Value::as_str)
            .ok_or_else(|| "head projection entity_id is invalid".to_owned())?;
        let edge_id = payload
            .get("edge_id")
            .and_then(Value::as_str)
            .ok_or_else(|| "head projection edge_id is invalid".to_owned())?;
        let previous_id = payload
            .get("previous_entity_id")
            .and_then(Value::as_str)
            .ok_or_else(|| "head projection previous_entity_id is invalid".to_owned())?;
        let node_seq = payload
            .get("node_seq")
            .and_then(Value::as_i64)
            .ok_or_else(|| "head projection node_seq is invalid".to_owned())?;
        let edge_seq = payload
            .get("edge_seq")
            .and_then(Value::as_i64)
            .ok_or_else(|| "head projection edge_seq is invalid".to_owned())?;
        let events = reader
            .replay_raw_events("mcp_benchmark", node_seq - 1, 2)
            .map_err(|error| error.to_string())?;
        if events.len() != 2
            || events[0].seq != node_seq
            || events[0].entity_kind != "node"
            || events[0].entity_id != entity_id
            || events[1].seq != edge_seq
            || events[1].entity_kind != "edge"
            || events[1].entity_id != edge_id
        {
            return Err("head projection points to mismatched graph events".to_owned());
        }
        let edge: Value = serde_json::from_str(&events[1].payload_json)
            .map_err(|error| format!("edge event payload is invalid JSON: {error}"))?;
        if edge.get("source_ids").and_then(Value::as_array) != Some(&vec![json!(previous_id)])
            || edge.get("target_ids").and_then(Value::as_array) != Some(&vec![json!(entity_id)])
        {
            return Err("head projection edge endpoints are incorrect".to_owned());
        }
    }
    Ok(revision)
}

async fn mcp_call(
    State(state): State<AppState>,
    Json(request): Json<Value>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let id = request.get("id").cloned().unwrap_or(Value::Null);
    let invalid = |message: &str| {
        (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "jsonrpc": "2.0",
                "id": id,
                "error": {"code": -32602, "message": message}
            })),
        )
    };
    let internal_error = |message: String| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({
                "jsonrpc": "2.0",
                "id": id,
                "error": {"code": -32000, "message": message}
            })),
        )
    };
    if request.get("jsonrpc").and_then(Value::as_str) != Some("2.0")
        || request.get("method").and_then(Value::as_str) != Some("tools/call")
    {
        return Err(invalid("expected JSON-RPC tools/call request"));
    }
    let params = request
        .get("params")
        .and_then(Value::as_object)
        .ok_or_else(|| invalid("params must be an object"))?;
    let name = params
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid("tool name is required"))?;
    let args = params
        .get("arguments")
        .and_then(Value::as_object)
        .ok_or_else(|| invalid("arguments must be an object"))?;
    if state.concurrent.is_some() {
        let name = name.to_owned();
        let args = args.clone();
        let concurrent_state = state.clone();
        let result =
            tokio::task::spawn_blocking(move || concurrent_tool_call(concurrent_state, name, args))
                .await
                .map_err(|error| internal_error(format!("concurrent MCP task failed: {error}")))?
                .map_err(internal_error)?;
        return Ok(Json(json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": {"structuredContent": result, "content": [{"type": "text", "text": result.to_string()}]}
        })));
    }
    let index = args
        .get("index")
        .and_then(Value::as_u64)
        .ok_or_else(|| invalid("index must be a non-negative integer"))? as usize;
    let entity_id = args
        .get("expected_entity_id")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid("expected_entity_id is required"))?;
    let result = match name {
        "write_record" => {
            let edge_id = args
                .get("expected_edge_id")
                .and_then(Value::as_str)
                .ok_or_else(|| invalid("expected_edge_id is required"))?;
            let source_id = args
                .get("expected_source_id")
                .and_then(Value::as_str)
                .ok_or_else(|| invalid("expected_source_id is required"))?;
            let node_payload = args
                .get("node_payload")
                .filter(|value| value.get("id").and_then(Value::as_str) == Some(entity_id))
                .ok_or_else(|| invalid("node_payload does not match trace"))?;
            let edge_payload = args
                .get("edge_payload")
                .filter(|value| {
                    value.get("id").and_then(Value::as_str) == Some(edge_id)
                        && value.get("source_ids").and_then(Value::as_array)
                            == Some(&vec![json!(source_id)])
                        && value.get("target_ids").and_then(Value::as_array)
                            == Some(&vec![json!(entity_id)])
                })
                .ok_or_else(|| invalid("edge_payload does not match graph endpoints"))?;
            let node_json = canonical_json(node_payload);
            let edge_json = canonical_json(edge_payload);
            let node_value = node_payload.clone();
            let edge_value = edge_payload.clone();
            let entity_id_owned = entity_id.to_owned();
            let edge_id_owned = edge_id.to_owned();
            let (node_seq, edge_seq) = state
                .store
                .immediate_transaction(|uow| {
                    let node_event = uow.append_raw_entity_event(
                        "mcp_benchmark",
                        NewRawEntityEvent {
                            event_id: format!("mcp-node-event-{index:05}"),
                            entity_kind: "node".to_owned(),
                            entity_id: entity_id_owned.clone(),
                            op: "UPSERT".to_owned(),
                            payload_json: node_json,
                        },
                    )?;
                    let edge_event = uow.append_raw_entity_event(
                        "mcp_benchmark",
                        NewRawEntityEvent {
                            event_id: format!("mcp-edge-event-{index:05}"),
                            entity_kind: "edge".to_owned(),
                            entity_id: edge_id_owned.clone(),
                            op: "UPSERT".to_owned(),
                            payload_json: edge_json,
                        },
                    )?;
                    uow.replace_named_projection(
                        "mcp_benchmark_projection",
                        &format!("node:{entity_id_owned}"),
                        NamedProjectionWrite {
                            payload: Map::from_iter([
                                ("entity_kind".to_owned(), json!("node")),
                                ("entity_id".to_owned(), json!(entity_id_owned)),
                                ("entity".to_owned(), node_value),
                            ]),
                            last_authoritative_seq: node_event.event.seq,
                            last_materialized_seq: node_event.event.seq,
                            projection_schema_version: 1,
                            materialization_status: "ready".to_owned(),
                        },
                    )?;
                    uow.replace_named_projection(
                        "mcp_benchmark_projection",
                        &format!("edge:{edge_id_owned}"),
                        NamedProjectionWrite {
                            payload: Map::from_iter([
                                ("entity_kind".to_owned(), json!("edge")),
                                ("entity_id".to_owned(), json!(edge_id_owned)),
                                ("entity".to_owned(), edge_value),
                            ]),
                            last_authoritative_seq: edge_event.event.seq,
                            last_materialized_seq: edge_event.event.seq,
                            projection_schema_version: 1,
                            materialization_status: "ready".to_owned(),
                        },
                    )?;
                    Ok((node_event.event.seq, edge_event.event.seq))
                })
                .map_err(|error| internal_error(error.to_string()))?;
            json!({"seq": edge_seq, "node_seq": node_seq, "entity_id": entity_id, "edge_id": edge_id})
        }
        "read_event" => {
            let edge_id = args
                .get("expected_edge_id")
                .and_then(Value::as_str)
                .ok_or_else(|| invalid("expected_edge_id is required"))?;
            let source_id = args
                .get("expected_source_id")
                .and_then(Value::as_str)
                .ok_or_else(|| invalid("expected_source_id is required"))?;
            let node_seq = 2 + 2 * index as i64;
            let events = state
                .store
                .replay_raw_events("mcp_benchmark", node_seq - 1, 2)
                .map_err(|error| internal_error(error.to_string()))?;
            if events.len() != 2
                || events[0].seq != node_seq
                || events[0].entity_kind != "node"
                || events[0].entity_id != entity_id
                || events[1].seq != node_seq + 1
                || events[1].entity_kind != "edge"
                || events[1].entity_id != edge_id
            {
                return Err(invalid("graph events did not match trace"));
            }
            let edge: Value = serde_json::from_str(&events[1].payload_json)
                .map_err(|error| internal_error(error.to_string()))?;
            if edge.get("source_ids").and_then(Value::as_array) != Some(&vec![json!(source_id)])
                || edge.get("target_ids").and_then(Value::as_array) != Some(&vec![json!(entity_id)])
            {
                return Err(invalid("edge endpoints did not match trace"));
            }
            json!({"seq": events[1].seq, "entity_id": entity_id, "edge_id": edge_id})
        }
        "read_projection" => {
            let edge_id = args
                .get("expected_edge_id")
                .and_then(Value::as_str)
                .ok_or_else(|| invalid("expected_edge_id is required"))?;
            let source_id = args
                .get("expected_source_id")
                .and_then(Value::as_str)
                .ok_or_else(|| invalid("expected_source_id is required"))?;
            let projection = state
                .store
                .get_named_projection("mcp_benchmark_projection", &format!("node:{entity_id}"))
                .map_err(|error| internal_error(error.to_string()))?;
            let projection = projection.ok_or_else(|| invalid("expected projection is missing"))?;
            let edge_projection = state
                .store
                .get_named_projection("mcp_benchmark_projection", &format!("edge:{edge_id}"))
                .map_err(|error| internal_error(error.to_string()))?
                .ok_or_else(|| invalid("expected edge projection is missing"))?;
            if projection
                .payload
                .get("entity")
                .and_then(|value| value.get("id"))
                .and_then(Value::as_str)
                != Some(entity_id)
                || edge_projection
                    .payload
                    .get("entity")
                    .and_then(|value| value.get("source_ids"))
                    .and_then(Value::as_array)
                    != Some(&vec![json!(source_id)])
                || edge_projection
                    .payload
                    .get("entity")
                    .and_then(|value| value.get("target_ids"))
                    .and_then(Value::as_array)
                    != Some(&vec![json!(entity_id)])
            {
                return Err(invalid("projection did not match trace"));
            }
            json!({"index": index, "entity_id": entity_id, "edge_id": edge_id})
        }
        _ => return Err(invalid("unknown benchmark tool")),
    };
    Ok(Json(json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": {"structuredContent": result, "content": [{"type": "text", "text": result.to_string()}]}
    })))
}

async fn health() -> Json<Value> {
    Json(json!({"status": "ready"}))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let database = PathBuf::from(
        args.next()
            .ok_or("usage: <database-path> <socket-address> [concurrent]")?,
    );
    let address: SocketAddr = args
        .next()
        .ok_or("usage: <database-path> <socket-address> [concurrent]")?
        .parse()?;
    let concurrent = match args.next().as_deref() {
        None => None,
        Some("concurrent") => Some(Arc::new(ConcurrentCoordinator::new())),
        Some(_) => return Err("expected optional mode 'concurrent'".into()),
    };
    if args.next().is_some() {
        return Err("unexpected extra argument".into());
    }
    let store = SqliteStore::open(&database)?;
    let root_payload: Value = serde_json::from_str(
        &env::var("KOGWISTAR_MIXED_ROOT_PAYLOAD")
            .map_err(|_| "KOGWISTAR_MIXED_ROOT_PAYLOAD is required")?,
    )?;
    let root_json = canonical_json(&root_payload);
    store.immediate_transaction(|uow| {
        let event = uow.append_raw_entity_event(
            "mcp_benchmark",
            NewRawEntityEvent {
                event_id: "mcp-root-node".to_owned(),
                entity_kind: "node".to_owned(),
                entity_id: "mixed-node-root".to_owned(),
                op: "UPSERT".to_owned(),
                payload_json: root_json,
            },
        )?;
        uow.replace_named_projection(
            "mcp_benchmark_projection",
            "node:mixed-node-root",
            NamedProjectionWrite {
                payload: Map::from_iter([
                    ("entity_kind".to_owned(), json!("node")),
                    ("entity_id".to_owned(), json!("mixed-node-root")),
                    ("entity".to_owned(), root_payload),
                ]),
                last_authoritative_seq: event.event.seq,
                last_materialized_seq: event.event.seq,
                projection_schema_version: 1,
                materialization_status: "ready".to_owned(),
            },
        )?;
        if concurrent.is_some() {
            uow.replace_named_projection(
                "mcp_benchmark_projection",
                "mcp-head",
                NamedProjectionWrite {
                    payload: Map::from_iter([
                        ("revision".to_owned(), json!(0)),
                        ("entity_id".to_owned(), json!("mixed-node-root")),
                        ("edge_id".to_owned(), Value::Null),
                        ("previous_entity_id".to_owned(), Value::Null),
                        ("node_seq".to_owned(), json!(event.event.seq)),
                        ("edge_seq".to_owned(), json!(0)),
                    ]),
                    last_authoritative_seq: event.event.seq,
                    last_materialized_seq: event.event.seq,
                    projection_schema_version: 1,
                    materialization_status: "ready".to_owned(),
                },
            )?;
        }
        Ok::<(), kogwistar_store_sqlite::SqliteStoreError>(())
    })?;
    let app = Router::new()
        .route("/health", get(health))
        .route("/mcp", post(mcp_call))
        .with_state(AppState {
            store: Arc::new(store),
            database: Arc::new(database),
            concurrent,
        });
    axum::serve(tokio::net::TcpListener::bind(address).await?, app).await?;
    Ok(())
}
