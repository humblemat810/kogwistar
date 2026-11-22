use once_cell::sync::Lazy;
use pgvector::Vector;
use postgres::{Client, NoTls};
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use serde_json::Value;
use std::collections::HashMap;
use std::error::Error;
use std::sync::Mutex;

#[pyclass(skip_from_py_object, extends = PyException)]
#[derive(Clone, Debug)]
struct KogwistarGraphError {
    code: String,
    message: String,
}

#[pymethods]
impl KogwistarGraphError {
    #[new]
    fn new(code: String, message: String) -> Self {
        Self { code, message }
    }
    #[getter]
    fn code(&self) -> String {
        self.code.clone()
    }
    #[getter]
    fn message(&self) -> String {
        self.message.clone()
    }
    fn __str__(&self) -> String {
        format!("{}: {}", self.code, self.message)
    }
}

fn graph_err(code: &str, message: &str) -> PyErr {
    PyErr::new::<KogwistarGraphError, _>((code.to_string(), message.to_string()))
}

fn pg_err_with_source(code: &str, e: &postgres::Error) -> PyErr {
    let mut msg = format!("{}", e);
    let mut source = e.source();
    while let Some(s) = source {
        msg.push_str(&format!(" caused by: {}", s));
        source = s.source();
    }
    PyErr::new::<KogwistarGraphError, _>((code.to_string(), msg))
}

fn se<T>(r: Result<T, postgres::Error>) -> PyResult<T> {
    r.map_err(|e| pg_err_with_source("KOGWISTAR_STORE_GRAPH", &e))
}

struct ConnSlot(*mut Client);
unsafe impl Send for ConnSlot {}
unsafe impl Sync for ConnSlot {}

static CONNS: Lazy<Mutex<HashMap<String, ConnSlot>>> = Lazy::new(|| Mutex::new(HashMap::new()));

fn conn_key(connstr: &str) -> String {
    format!("conn|{}", connstr)
}

fn get_conn(connstr: &str) -> PyResult<*mut Client> {
    let key = conn_key(connstr);
    let map = CONNS
        .lock()
        .map_err(|e| graph_err("KOGWISTAR_STORE_GRAPH_LOCK", &format!("{:?}", e)))?;
    if let Some(slot) = map.get(&key) {
        return Ok(slot.0);
    }
    drop(map);
    let client = se(Client::connect(connstr, NoTls))?;
    let boxed = Box::new(client);
    let ptr = Box::into_raw(boxed);
    let mut map = CONNS
        .lock()
        .map_err(|e| graph_err("KOGWISTAR_STORE_GRAPH_LOCK", &format!("{:?}", e)))?;
    map.insert(key, ConnSlot(ptr));
    Ok(ptr)
}

fn with_conn<F, R>(connstr: &str, f: F) -> PyResult<R>
where
    F: FnOnce(&mut Client) -> PyResult<R>,
{
    let ptr = get_conn(connstr)?;
    let client = unsafe { &mut *ptr };
    f(client)
}

fn ensure_schema(conn: &mut Client) -> PyResult<()> {
    let stmts = [
        "CREATE SCHEMA IF NOT EXISTS kogwistar",
        "CREATE TABLE IF NOT EXISTS kogwistar.graph_nodes ( node_id TEXT PRIMARY KEY, namespace TEXT NOT NULL DEFAULT 'default', entity_kind TEXT NOT NULL, payload_json TEXT NOT NULL DEFAULT '{}', embedding vector(384), created_at TIMESTAMPTZ NOT NULL DEFAULT NOW() )",
        "CREATE TABLE IF NOT EXISTS kogwistar.graph_edges ( edge_id TEXT PRIMARY KEY, namespace TEXT NOT NULL DEFAULT 'default', source_node_id TEXT NOT NULL, target_node_id TEXT NOT NULL, relation_kind TEXT NOT NULL, payload_json TEXT NOT NULL DEFAULT '{}', created_at TIMESTAMPTZ NOT NULL DEFAULT NOW() )",
        "CREATE INDEX IF NOT EXISTS idx_graph_nodes_embedding ON kogwistar.graph_nodes USING ivfflat (embedding vector_cosine_ops)",
    ];
    for stmt in stmts.iter() {
        se(conn.batch_execute(stmt))?;
    }
    Ok(())
}

fn op_str(op: &Value, key: &str) -> PyResult<String> {
    op.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| {
            graph_err(
                "KOGWISTAR_STORE_GRAPH_INVALID_PAYLOAD",
                &format!("missing {}", key),
            )
        })
        .map(|s| s.to_string())
}

fn graph_add_node(conn: &mut Client, op: &Value) -> PyResult<Value> {
    let namespace = op
        .get("namespace")
        .and_then(Value::as_str)
        .unwrap_or("default");
    let node_id = op_str(op, "node_id")?;
    let entity_kind = op.get("entity_kind").and_then(Value::as_str).unwrap_or("");
    let payload_json = op
        .get("payload_json")
        .and_then(Value::as_str)
        .unwrap_or("{}");
    let embedding = op.get("embedding").and_then(Value::as_array);
    let emb_vec: Option<Vector> = embedding.map(|arr| {
        let vals: Vec<f32> = arr
            .iter()
            .filter_map(|v| v.as_f64().map(|x| x as f32))
            .collect();
        Vector::from(vals)
    });
    match emb_vec {
        Some(emb) => {
            se(conn.execute(
                "INSERT INTO kogwistar.graph_nodes(node_id, namespace, entity_kind, payload_json, embedding) VALUES ($1, $2, $3, $4, $5)",
                &[&node_id as &(dyn postgres::types::ToSql + Sync), &namespace as &(dyn postgres::types::ToSql + Sync), &entity_kind as &(dyn postgres::types::ToSql + Sync), &payload_json as &(dyn postgres::types::ToSql + Sync), &emb as &(dyn postgres::types::ToSql + Sync)],
            ))?;
        }
        None => {
            se(conn.execute(
                "INSERT INTO kogwistar.graph_nodes(node_id, namespace, entity_kind, payload_json) VALUES ($1, $2, $3, $4)",
                &[&node_id as &(dyn postgres::types::ToSql + Sync), &namespace as &(dyn postgres::types::ToSql + Sync), &entity_kind as &(dyn postgres::types::ToSql + Sync), &payload_json as &(dyn postgres::types::ToSql + Sync)],
            ))?;
        }
    }
    Ok(serde_json::json!({"inserted": true, "node_id": node_id}))
}

fn graph_add_edge(conn: &mut Client, op: &Value) -> PyResult<Value> {
    let namespace = op
        .get("namespace")
        .and_then(Value::as_str)
        .unwrap_or("default");
    let edge_id = op_str(op, "edge_id")?;
    let source_node_id = op_str(op, "source_node_id")?;
    let target_node_id = op_str(op, "target_node_id")?;
    let relation_kind = op_str(op, "relation_kind")?;
    let payload_json = op
        .get("payload_json")
        .and_then(Value::as_str)
        .unwrap_or("{}");
    se(conn.execute(
        "INSERT INTO kogwistar.graph_edges(edge_id, namespace, source_node_id, target_node_id, relation_kind, payload_json) VALUES ($1, $2, $3, $4, $5, $6)",
        &[&edge_id as &(dyn postgres::types::ToSql + Sync), &namespace as &(dyn postgres::types::ToSql + Sync), &source_node_id as &(dyn postgres::types::ToSql + Sync), &target_node_id as &(dyn postgres::types::ToSql + Sync), &relation_kind as &(dyn postgres::types::ToSql + Sync), &payload_json as &(dyn postgres::types::ToSql + Sync)],
    ))?;
    Ok(serde_json::json!({"inserted": true, "edge_id": edge_id}))
}

fn graph_get_node(conn: &mut Client, op: &Value) -> PyResult<Value> {
    let namespace = op
        .get("namespace")
        .and_then(Value::as_str)
        .unwrap_or("default");
    let node_id = op_str(op, "node_id")?;
    let row = se(conn.query_opt(
        "SELECT node_id, namespace, entity_kind, payload_json, created_at::text FROM kogwistar.graph_nodes WHERE node_id = $1 AND namespace = $2",
        &[&node_id as &(dyn postgres::types::ToSql + Sync), &namespace as &(dyn postgres::types::ToSql + Sync)],
    ))?;
    match row {
        Some(r) => Ok(serde_json::json!({
            "found": true,
            "node_id": r.get::<_, String>(0),
            "namespace": r.get::<_, String>(1),
            "entity_kind": r.get::<_, String>(2),
            "payload_json": r.get::<_, String>(3),
            "created_at": r.get::<_, String>(4),
        })),
        None => Ok(serde_json::json!({"found": false, "node_id": node_id})),
    }
}

fn graph_vector_search(conn: &mut Client, op: &Value) -> PyResult<Value> {
    let namespace = op
        .get("namespace")
        .and_then(Value::as_str)
        .unwrap_or("default");
    let embedding = op
        .get("embedding")
        .and_then(Value::as_array)
        .ok_or_else(|| graph_err("KOGWISTAR_STORE_GRAPH_INVALID_PAYLOAD", "missing embedding"))?;
    let vals: Vec<f32> = embedding
        .iter()
        .filter_map(|v| v.as_f64().map(|x| x as f32))
        .collect();
    let emb = Vector::from(vals);
    let limit: i64 = op.get("limit").and_then(Value::as_i64).unwrap_or(10);
    let rows = se(conn.query(
        "SELECT node_id, namespace, entity_kind, payload_json, 1 - (embedding <=> $1) AS cosine_similarity FROM kogwistar.graph_nodes WHERE namespace = $2 AND embedding IS NOT NULL ORDER BY embedding <=> $1 ASC LIMIT $3",
        &[&emb as &(dyn postgres::types::ToSql + Sync), &namespace as &(dyn postgres::types::ToSql + Sync), &limit as &(dyn postgres::types::ToSql + Sync)],
    ))?;
    let mut out: Vec<Value> = Vec::new();
    for row in rows {
        out.push(serde_json::json!({
            "node_id": row.get::<_, String>(0),
            "namespace": row.get::<_, String>(1),
            "entity_kind": row.get::<_, String>(2),
            "payload_json": row.get::<_, String>(3),
            "cosine_similarity": row.get::<_, f64>(4),
        }));
    }
    Ok(serde_json::json!(out))
}

#[pyfunction]
fn store_graph_json(payload: &str) -> PyResult<String> {
    let request: Value = serde_json::from_str(payload).map_err(|e| {
        graph_err(
            "KOGWISTAR_STORE_GRAPH_INVALID_PAYLOAD",
            &format!("invalid json: {}", e),
        )
    })?;
    let connstr = request
        .get("connstr")
        .and_then(Value::as_str)
        .ok_or_else(|| graph_err("KOGWISTAR_STORE_GRAPH_INVALID_PAYLOAD", "missing connstr"))?;
    let operation = request
        .get("operation")
        .cloned()
        .ok_or_else(|| graph_err("KOGWISTAR_STORE_GRAPH_INVALID_PAYLOAD", "missing operation"))?;
    let kind = operation.get("kind").and_then(Value::as_str).unwrap_or("");
    let result = with_conn(connstr, |conn| {
        ensure_schema(conn)?;
        match kind {
            "graph_add_node" => graph_add_node(conn, &operation),
            "graph_add_edge" => graph_add_edge(conn, &operation),
            "graph_get_node" => graph_get_node(conn, &operation),
            "graph_vector_search" => graph_vector_search(conn, &operation),
            _ => Err(graph_err(
                "KOGWISTAR_STORE_GRAPH_OPERATION_INVALID",
                &format!("unknown operation kind: {}", kind),
            )),
        }
    })?;
    Ok(serde_json::to_string(&result).unwrap())
}

#[pymodule]
fn _graph(module: &pyo3::Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    module.add_function(pyo3::wrap_pyfunction!(store_graph_json)(module.py())?)?;
    Ok(())
}
