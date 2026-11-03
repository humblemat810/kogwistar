//! Python-compatible lexical projection for the derived search index.
//!
//! PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/search_index_fts.rs#upsert_entries; python=kogwistar/engine_core/search_index/service.py#SearchIndexService.upsert_entries; fixture=contracts/golden/search-index-entry-v1.json; rust_test=rust/crates/kogwistar-api/src/search_index_fts.rs#rust_fts_projection_matches_python_index_schema; python_test=tests/core/test_search_index_contract.py#test_search_index_entries_match_shared_rust_golden_fixture

use std::path::Path;
use std::{collections::HashMap, sync::Arc};

use kogwistar_contracts::{
    SearchIndexEntryV1, SearchIndexUpsertV1, build_search_index_embedding_text,
    make_search_index_key,
};
use kogwistar_engine::embedding::EmbeddingProvider;
use kogwistar_store::{
    DistanceMetric, EventReadStore, EventWriteStore, GraphRecord, NewEntityEvent, NodeIndexStore,
    StoreError, StoreResult, VectorMatch, VectorQuery,
};
use rusqlite::{Connection, OptionalExtension, params};
use sha2::{Digest, Sha256};

const INDEX_TABLE_SCHEMA: &str = "CREATE TABLE IF NOT EXISTS semantic_index (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    index_key TEXT NOT NULL UNIQUE,
    node_id TEXT NOT NULL,
    canonical_title TEXT NOT NULL,
    keywords TEXT NOT NULL DEFAULT '',
    aliases TEXT NOT NULL DEFAULT '',
    provision TEXT NOT NULL,
    document_id TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(node_id, canonical_title, provision)
);";

const FTS_SCHEMA: &str = "CREATE VIRTUAL TABLE IF NOT EXISTS semantic_index_fts
USING fts5(index_key UNINDEXED, canonical_title, provision, keywords, aliases,
           content='semantic_index', content_rowid='id');
CREATE TRIGGER IF NOT EXISTS semantic_index_ai AFTER INSERT ON semantic_index BEGIN
    INSERT INTO semantic_index_fts(rowid,index_key,canonical_title,provision,keywords,aliases)
    VALUES(new.id,new.index_key,new.canonical_title,new.provision,new.keywords,new.aliases);
END;
CREATE TRIGGER IF NOT EXISTS semantic_index_ad AFTER DELETE ON semantic_index BEGIN
    INSERT INTO semantic_index_fts(semantic_index_fts,rowid,index_key,canonical_title,provision,keywords,aliases)
    VALUES('delete',old.id,old.index_key,old.canonical_title,old.provision,old.keywords,old.aliases);
END;
CREATE TRIGGER IF NOT EXISTS semantic_index_au AFTER UPDATE ON semantic_index BEGIN
    INSERT INTO semantic_index_fts(semantic_index_fts,rowid,index_key,canonical_title,provision,keywords,aliases)
    VALUES('delete',old.id,old.index_key,old.canonical_title,old.provision,old.keywords,old.aliases);
    INSERT INTO semantic_index_fts(rowid,index_key,canonical_title,provision,keywords,aliases)
    VALUES(new.id,new.index_key,new.canonical_title,new.provision,new.keywords,new.aliases);
END;";

#[derive(Clone, Debug, PartialEq)]
pub struct LexicalIndexHit {
    pub index_key: String,
    pub node_id: String,
    pub canonical_title: String,
    pub provision: String,
    pub document_id: Option<String>,
    pub bm25_score: f64,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct HybridSearchHit {
    pub index_key: String,
    pub node_id: String,
    pub canonical_title: String,
    pub provision: String,
    pub document_id: Option<String>,
    pub fts_score: f64,
    pub vec_score: f64,
    pub hybrid_score: f64,
}

pub fn parse_hybrid_search_query(path_and_query: &str) -> Result<(String, usize, bool), String> {
    let raw_query = path_and_query
        .split_once('?')
        .map_or("", |(_, query)| query);
    let mut query = None;
    let mut limit = 10usize;
    let mut resolve_node = false;
    for (key, value) in url::form_urlencoded::parse(raw_query.as_bytes()) {
        match key.as_ref() {
            "q" => query = Some(value.into_owned()),
            "limit" => {
                limit = value
                    .parse::<usize>()
                    .map_err(|_| "limit must be an integer".to_owned())?;
            }
            "resolve_node" => {
                resolve_node = match value.to_ascii_lowercase().as_str() {
                    "true" | "1" | "yes" => true,
                    "false" | "0" | "no" => false,
                    _ => return Err("resolve_node must be a boolean".to_owned()),
                };
            }
            _ => {}
        }
    }
    let query = query
        .filter(|query| !query.trim().is_empty())
        .ok_or_else(|| "q is required and must not be empty".to_owned())?;
    if !(1..=1000).contains(&limit) {
        return Err("limit must be between 1 and 1000".to_owned());
    }
    Ok((query, limit, resolve_node))
}

pub fn project_hybrid_search_result(
    query: &str,
    hits: &[HybridSearchHit],
    authorized_nodes: &[kogwistar_store::GraphRecord],
    resolve_node: bool,
    limit: usize,
) -> serde_json::Value {
    let nodes = authorized_nodes
        .iter()
        .map(|record| (record.id.as_str(), record))
        .collect::<HashMap<_, _>>();
    let results = hits
        .iter()
        .filter_map(|hit| {
            let node = nodes.get(hit.node_id.as_str())?;
            let mut value = serde_json::to_value(hit).ok()?;
            if resolve_node {
                value["documents"] = serde_json::to_value(&node.document).ok()?;
                value["metadatas"] = serde_json::to_value(&node.metadata).ok()?;
            }
            Some(value)
        })
        .take(limit)
        .collect::<Vec<_>>();
    serde_json::json!({"query": query, "results": results})
}

/// Combine Python-compatible FTS and vector candidates before the caller applies
/// canonical graph ACL filtering. Ranking weights and tie order match Python.
///
/// PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/search_index_fts.rs#fuse_hybrid_hits; python=kogwistar/engine_core/search_index/service.py#SearchIndexService._fuse_hybrid_rows; fixture=contracts/golden/search-index-hybrid-v1.json; rust_test=rust/crates/kogwistar-api/src/search_index_fts.rs#hybrid_fusion_matches_shared_python_golden_fixture; python_test=tests/core/test_search_index_contract.py#test_hybrid_search_fusion_matches_shared_rust_golden_fixture
pub fn fuse_hybrid_hits(
    lexical: &[LexicalIndexHit],
    vectors: &[VectorMatch],
) -> Vec<HybridSearchHit> {
    let lexical_scores = normalized_scores(
        lexical
            .iter()
            .map(|hit| (hit.index_key.as_str(), hit.bm25_score)),
    );
    let vector_scores = normalized_scores(
        vectors
            .iter()
            .filter_map(|item| vector_index_key(item).map(|key| (Arc::from(key), item.distance))),
    );

    let mut hits = Vec::with_capacity(lexical.len() + vectors.len());
    let mut positions = HashMap::<String, usize>::new();
    for item in lexical {
        let hit = HybridSearchHit {
            index_key: item.index_key.clone(),
            node_id: item.node_id.clone(),
            canonical_title: item.canonical_title.clone(),
            provision: item.provision.clone(),
            document_id: item.document_id.clone(),
            fts_score: lexical_scores
                .get(item.index_key.as_str())
                .copied()
                .unwrap_or(0.0),
            vec_score: 0.0,
            hybrid_score: 0.0,
        };
        if let Some(position) = positions.get(&hit.index_key).copied() {
            hits[position] = hit;
        } else {
            positions.insert(hit.index_key.clone(), hits.len());
            hits.push(hit);
        }
    }

    for item in vectors {
        let Some(index_key) = vector_index_key(item) else {
            continue;
        };
        let vec_score = vector_scores
            .get(index_key.as_str())
            .copied()
            .unwrap_or(0.0);
        if let Some(position) = positions.get(index_key.as_str()).copied() {
            hits[position].vec_score = vec_score;
        } else {
            let metadata = &item.record.metadata;
            let node_id = metadata
                .get("target_node_id")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default();
            let canonical_title = metadata
                .get("canonical_title")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default();
            let provision = metadata
                .get("provision")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default();
            positions.insert(index_key.clone(), hits.len());
            hits.push(HybridSearchHit {
                index_key,
                node_id: node_id.to_owned(),
                canonical_title: canonical_title.to_owned(),
                provision: provision.to_owned(),
                document_id: metadata
                    .get("doc_id")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned),
                fts_score: 0.0,
                vec_score,
                hybrid_score: 0.0,
            });
        }
    }

    for hit in &mut hits {
        hit.hybrid_score = 0.6 * hit.fts_score + 0.4 * hit.vec_score;
    }
    hits.sort_by(|left, right| right.hybrid_score.total_cmp(&left.hybrid_score));
    hits
}

fn vector_index_key(item: &VectorMatch) -> Option<String> {
    let metadata = &item.record.metadata;
    if let Some(key) = metadata
        .get("index_key")
        .and_then(serde_json::Value::as_str)
    {
        return Some(key.to_owned());
    }
    let node_id = metadata
        .get("target_node_id")
        .and_then(serde_json::Value::as_str)?;
    let title = metadata
        .get("canonical_title")
        .and_then(serde_json::Value::as_str)?;
    let provision = metadata
        .get("provision")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    Some(make_search_index_key(node_id, title, provision))
}

fn normalized_scores<K>(values: impl Iterator<Item = (K, f64)>) -> HashMap<Arc<str>, f64>
where
    K: Into<Arc<str>>,
{
    let values = values.collect::<Vec<_>>();
    let Some(minimum) = values.iter().map(|(_, score)| *score).reduce(f64::min) else {
        return HashMap::new();
    };
    let maximum = values
        .iter()
        .map(|(_, score)| *score)
        .reduce(f64::max)
        .unwrap_or(minimum);
    values
        .into_iter()
        .map(|(key, score)| {
            let normalized = if maximum == minimum {
                1.0
            } else {
                ((maximum - score) / (maximum - minimum)).clamp(0.0, 1.0)
            };
            (key.into(), normalized)
        })
        .collect()
}

pub fn open_index(path: impl AsRef<Path>) -> rusqlite::Result<Connection> {
    if path.as_ref() != Path::new(":memory:")
        && let Some(parent) = path
            .as_ref()
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)
            .map_err(|error| rusqlite::Error::ToSqlConversionFailure(Box::new(error)))?;
    }
    let connection = Connection::open(path)?;
    connection.execute_batch("PRAGMA foreign_keys = ON;")?;
    connection.execute_batch(INDEX_TABLE_SCHEMA)?;
    let existing_fts_schema = connection
        .query_row(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='semantic_index_fts'",
            [],
            |row| row.get::<_, String>(0),
        )
        .optional()?;
    if existing_fts_schema.is_some_and(|sql| {
        let sql = sql.to_ascii_lowercase();
        sql.contains("content=''") || sql.contains("content=\"\"")
    }) {
        connection.execute_batch(
            "DROP TRIGGER IF EXISTS semantic_index_ai;
             DROP TRIGGER IF EXISTS semantic_index_ad;
             DROP TRIGGER IF EXISTS semantic_index_au;
             DROP TABLE IF EXISTS semantic_index_fts;",
        )?;
    }
    connection.execute_batch(FTS_SCHEMA)?;
    connection.execute(
        "INSERT INTO semantic_index_fts(semantic_index_fts) VALUES('rebuild')",
        [],
    )?;
    Ok(connection)
}

pub fn upsert_entries(
    connection: &mut Connection,
    entries: &[SearchIndexEntryV1],
) -> rusqlite::Result<()> {
    let transaction = connection.transaction()?;
    {
        let mut statement = transaction.prepare_cached(
            "INSERT INTO semantic_index
                (index_key,node_id,canonical_title,keywords,aliases,provision,document_id)
             VALUES (?1,?2,?3,?4,?5,?6,?7)
             ON CONFLICT(index_key) DO UPDATE SET
                node_id=excluded.node_id,
                canonical_title=excluded.canonical_title,
                keywords=excluded.keywords,
                aliases=excluded.aliases,
                provision=excluded.provision,
                document_id=excluded.document_id,
                updated_at=CURRENT_TIMESTAMP",
        )?;
        for entry in entries {
            statement.execute(params![
                make_search_index_key(&entry.node_id, &entry.canonical_title, &entry.provision),
                entry.node_id,
                entry.canonical_title,
                entry.keywords.join(" "),
                entry.aliases.join(" "),
                entry.provision,
                entry.doc_id,
            ])?;
        }
    }
    transaction.commit()
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SearchIndexReplaySummary {
    pub events_read: usize,
    pub entries_projected: usize,
    pub prior_cursor: i64,
    pub new_cursor: i64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SearchIndexAdmissionSummary {
    pub entries_admitted: usize,
    pub events_projected: usize,
    pub last_event_seq: i64,
}

/// Authenticated HTTP admission shared by the native backend services.
/// The graph namespace is derived from verified principal scope, never from
/// request JSON, so callers cannot redirect durable index events.
///
/// PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/search_index_fts.rs#search_index_upsert_effect; python=kogwistar/server_mcp_with_admin.py#add_index_entries; fixture=contracts/golden/search-index-entry-v1.json; rust_test=rust/crates/kogwistar-api/src/search_index_fts.rs#search_index_upsert_effect_requires_write_scope_and_matches_python_response; python_test=tests/core/test_search_index_contract.py#test_search_index_entries_match_shared_rust_golden_fixture
pub(crate) async fn search_index_upsert_effect<E, V>(
    request: &crate::ApiEffectRequest,
    capabilities: &crate::SharedCapabilityState,
    event_store: &E,
    vector_store: &V,
    fts: Option<&std::sync::Mutex<Connection>>,
    provider: Option<&dyn EmbeddingProvider>,
) -> crate::ApiEffectResponse
where
    E: EventReadStore + EventWriteStore,
    V: NodeIndexStore,
{
    let scope = crate::security_scope_value(&request.principal);
    let namespace = scope["storage_namespace"]
        .as_str()
        .filter(|value| !value.trim().is_empty() && *value != "*");
    let namespace_authorized = namespace.is_some_and(|namespace| {
        crate::principal_strings(&request.principal, "ns")
            .iter()
            .any(|allowed| allowed == namespace || allowed == "*")
    });
    if crate::principal_role(&request.principal) != "rw"
        || !namespace_authorized
        || !crate::capability_allowed(request, capabilities, "write_graph", &["write_graph"])
    {
        return crate::effect_error(
            axum::http::StatusCode::FORBIDDEN,
            "KOGWISTAR_CAPABILITY_FORBIDDEN",
            "Search-index admission requires write_graph capability and an authorized storage namespace",
        );
    }
    let input: SearchIndexUpsertV1 = match serde_json::from_slice(&request.body) {
        Ok(input) => input,
        Err(error) => {
            return crate::effect_error(
                axum::http::StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                error.to_string(),
            );
        }
    };
    if input.index.is_empty() {
        return crate::json_effect(axum::http::StatusCode::OK, serde_json::json!({"ok": true}));
    }
    let (Some(fts), Some(provider), Some(namespace)) = (fts, provider, namespace) else {
        return crate::effect_error(
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            "KOGWISTAR_SEARCH_INDEX_UNAVAILABLE",
            "Search-index event, FTS, vector, or embedding services are not configured",
        );
    };
    match admit_search_index_entries(
        event_store,
        vector_store,
        fts,
        provider,
        namespace,
        &input.index,
    )
    .await
    {
        Ok(_) => crate::json_effect(axum::http::StatusCode::OK, serde_json::json!({"ok": true})),
        Err(error @ StoreError::InvalidEntityEventPayload { .. }) => crate::effect_error(
            axum::http::StatusCode::UNPROCESSABLE_ENTITY,
            "KOGWISTAR_INVALID_REQUEST",
            error.to_string(),
        ),
        Err(error) => crate::effect_error(
            axum::http::StatusCode::BAD_GATEWAY,
            "KOGWISTAR_SEARCH_INDEX_ADMISSION_FAILED",
            error.to_string(),
        ),
    }
}

/// Durably admit search-index entries, then project through their event
/// sequence. Stable event IDs make retries of the same namespace/entry
/// idempotent while a changed entry creates a new authoritative event.
///
/// PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/search_index_fts.rs#admit_search_index_entries; python=kogwistar/engine_core/search_index/service.py#SearchIndexService.upsert_entries; fixture=contracts/golden/search-index-entry-v1.json; rust_test=rust/crates/kogwistar-api/src/search_index_fts.rs#search_index_admission_is_durable_idempotent_and_retryable; python_test=tests/core/test_search_index_contract.py#test_search_index_entries_match_shared_rust_golden_fixture
pub async fn admit_search_index_entries<E, V>(
    event_store: &E,
    vector_store: &V,
    fts: &std::sync::Mutex<Connection>,
    provider: &dyn EmbeddingProvider,
    namespace: &str,
    entries: &[SearchIndexEntryV1],
) -> StoreResult<SearchIndexAdmissionSummary>
where
    E: EventReadStore + EventWriteStore,
    V: NodeIndexStore,
{
    if namespace.trim().is_empty() || entries.len() > 1000 {
        return Err(StoreError::InvalidEntityEventPayload {
            message: "search-index admission requires a namespace and at most 1000 entries"
                .to_owned(),
        });
    }
    let mut target_seq = 0;
    for entry in entries {
        if entry.node_id.trim().is_empty() || entry.canonical_title.trim().is_empty() {
            return Err(StoreError::InvalidEntityEventPayload {
                message: "search-index entry requires a node id and canonical title".to_owned(),
            });
        }
        let payload =
            serde_json::to_value(entry).map_err(|error| StoreError::InvalidEntityEventPayload {
                message: format!("search-index entry could not be serialized: {error}"),
            })?;
        let payload_bytes = serde_json::to_vec(&payload).map_err(|error| {
            StoreError::InvalidEntityEventPayload {
                message: format!("search-index event payload could not be serialized: {error}"),
            }
        })?;
        let mut digest = Sha256::new();
        digest.update(namespace.as_bytes());
        digest.update([0]);
        digest.update(&payload_bytes);
        let event_id = format!("search-index-v1-{:x}", digest.finalize());
        let entity_id =
            make_search_index_key(&entry.node_id, &entry.canonical_title, &entry.provision);
        let appended = event_store
            .append_entity_event(
                namespace,
                NewEntityEvent {
                    event_id,
                    entity_kind: "search_index".to_owned(),
                    entity_id: entity_id.clone(),
                    op: "search_index.upsert".to_owned(),
                    payload: payload.clone(),
                },
            )
            .await?;
        if appended.event.entity_kind != "search_index"
            || appended.event.entity_id != entity_id
            || appended.event.op != "search_index.upsert"
            || appended.event.payload != payload
        {
            return Err(StoreError::InvalidEntityEventPayload {
                message: "idempotent search-index event identity conflicts with its payload"
                    .to_owned(),
            });
        }
        target_seq = target_seq.max(appended.event.seq);
    }

    let mut events_projected = 0;
    while target_seq > 0 {
        let prior = event_store
            .replay_cursor(namespace, "search-index-fts-vector-v1")
            .await?;
        if prior.last_seq >= target_seq {
            break;
        }
        let replay = replay_search_index_events(
            event_store,
            vector_store,
            fts,
            provider,
            namespace,
            "search-index-fts-vector-v1",
            1000,
        )
        .await?;
        if replay.new_cursor <= prior.last_seq {
            return Err(StoreError::InvalidEntityEventPayload {
                message: "search-index replay made no progress toward the admitted events"
                    .to_owned(),
            });
        }
        events_projected += replay.events_read;
    }
    Ok(SearchIndexAdmissionSummary {
        entries_admitted: entries.len(),
        events_projected,
        last_event_seq: target_seq,
    })
}

/// Replay canonical search-index events into the disposable FTS and vector
/// projections. The cursor advances only after both projections succeed;
/// therefore a failure between writes safely retries idempotent upserts.
///
/// PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/search_index_fts.rs#replay_search_index_events; python=kogwistar/engine_core/subsystems/persist.py#PersistSubsystem.replay_namespace; fixture=contracts/golden/search-index-entry-v1.json; rust_test=rust/crates/kogwistar-api/src/search_index_fts.rs#search_index_admission_is_durable_idempotent_and_retryable; python_test=tests/cdc/test_search_index_cdc.py#test_search_index_replay
pub async fn replay_search_index_events<E, V>(
    event_store: &E,
    vector_store: &V,
    fts: &std::sync::Mutex<Connection>,
    provider: &dyn EmbeddingProvider,
    namespace: &str,
    consumer: &str,
    batch_limit: usize,
) -> StoreResult<SearchIndexReplaySummary>
where
    E: EventReadStore + EventWriteStore,
    V: NodeIndexStore,
{
    if namespace.trim().is_empty() || consumer.trim().is_empty() || batch_limit == 0 {
        return Err(StoreError::InvalidEntityEventPayload {
            message: "search-index replay requires namespace, consumer, and positive batch limit"
                .to_owned(),
        });
    }
    let cursor = event_store.replay_cursor(namespace, consumer).await?;
    let events = event_store
        .replay_events(namespace, cursor.last_seq, batch_limit)
        .await?;
    let mut entries = Vec::new();
    for event in &events {
        if event.entity_kind != "search_index" {
            continue;
        }
        if !event.op.eq_ignore_ascii_case("UPSERT")
            && !event.op.eq_ignore_ascii_case("search_index.upsert")
        {
            return Err(StoreError::UnsupportedEntityEventOperation {
                op: event.op.clone(),
            });
        }
        let entry: SearchIndexEntryV1 =
            serde_json::from_value(event.payload.clone()).map_err(|error| {
                StoreError::InvalidEntityEventPayload {
                    message: format!("invalid search-index event payload: {error}"),
                }
            })?;
        if entry.node_id.trim().is_empty()
            || entry.canonical_title.trim().is_empty()
            || event.entity_id
                != make_search_index_key(&entry.node_id, &entry.canonical_title, &entry.provision)
        {
            return Err(StoreError::InvalidEntityEventPayload {
                message: "search-index event identity does not match its payload".to_owned(),
            });
        }
        entries.push(entry);
    }

    let mut records = Vec::with_capacity(entries.len());
    if !entries.is_empty() {
        let texts = entries
            .iter()
            .map(build_search_index_embedding_text)
            .collect::<Vec<_>>();
        let embeddings = provider
            .embed(&texts)
            .await
            .map_err(|error| StoreError::Backend {
                backend: "search-index-embedding".to_owned(),
                message: error.to_string(),
            })?;
        if embeddings.len() != entries.len()
            || embeddings.iter().any(|vector| {
                vector.len() != provider.profile().dimension
                    || vector.iter().any(|component| !component.is_finite())
            })
        {
            return Err(StoreError::InvalidEntityEventPayload {
                message:
                    "search-index embedding provider returned invalid vector dimensions or values"
                        .to_owned(),
            });
        }
        for (entry, (document, embedding)) in entries.iter().zip(texts.into_iter().zip(embeddings))
        {
            let embedding = embedding
                .into_iter()
                .map(|value| value as f32)
                .collect::<Vec<_>>();
            if embedding.iter().any(|value| !value.is_finite()) {
                return Err(StoreError::InvalidEntityEventPayload {
                    message: "search-index embedding cannot be represented as f32".to_owned(),
                });
            }
            let key =
                make_search_index_key(&entry.node_id, &entry.canonical_title, &entry.provision);
            records.push(GraphRecord {
                id: format!("idx:{key}"),
                document: Some(document),
                metadata: serde_json::from_value(serde_json::json!({
                    "index_key": key,
                    "target_node_id": entry.node_id,
                    "canonical_title": entry.canonical_title,
                    "provision": entry.provision,
                    "keywords": serde_json::to_string(&entry.keywords).expect("string vec serializes"),
                    "aliases": serde_json::to_string(&entry.aliases).expect("string vec serializes"),
                    "doc_id": entry.doc_id,
                }))
                .map_err(|error| StoreError::InvalidEntityEventPayload {
                    message: error.to_string(),
                })?,
                embedding: Some(embedding),
            });
        }
        {
            let mut connection = fts.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            upsert_entries(&mut connection, &entries).map_err(|error| StoreError::Backend {
                backend: "search-index-fts".to_owned(),
                message: error.to_string(),
            })?;
        }
        vector_store
            .upsert_node_index("nodes_index", records)
            .await?;
    }

    let new_cursor = events.last().map_or(cursor.last_seq, |event| event.seq);
    if new_cursor > cursor.last_seq {
        event_store
            .advance_replay_cursor(namespace, consumer, new_cursor)
            .await?;
    }
    Ok(SearchIndexReplaySummary {
        events_read: events.len(),
        entries_projected: entries.len(),
        prior_cursor: cursor.last_seq,
        new_cursor,
    })
}

pub fn search_fts(
    connection: &Connection,
    query: &str,
    limit: usize,
) -> rusqlite::Result<Vec<LexicalIndexHit>> {
    let mut statement = connection.prepare(
        "SELECT s.index_key,s.node_id,s.canonical_title,s.provision,s.document_id,
                bm25(semantic_index_fts) AS fts_score
         FROM semantic_index AS s
         JOIN semantic_index_fts ON semantic_index_fts.rowid=s.id
         WHERE semantic_index_fts MATCH ?1
         ORDER BY fts_score
         LIMIT ?2",
    )?;
    let rows = statement.query_map(params![query, limit as i64], |row| {
        Ok(LexicalIndexHit {
            index_key: row.get(0)?,
            node_id: row.get(1)?,
            canonical_title: row.get(2)?,
            provision: row.get(3)?,
            document_id: row.get(4)?,
            bm25_score: row.get(5)?,
        })
    })?;
    rows.collect()
}

pub async fn hybrid_search<S: NodeIndexStore>(
    fts: &std::sync::Mutex<Connection>,
    vector_store: &S,
    index: &str,
    provider: &dyn EmbeddingProvider,
    query: &str,
    limit: usize,
) -> Result<Vec<HybridSearchHit>, String> {
    let lexical = {
        let connection = fts
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        search_fts(&connection, query, limit).map_err(|error| error.to_string())?
    };
    let mut batches = provider
        .embed(&[query.to_owned()])
        .await
        .map_err(|error| error.to_string())?;
    if batches.len() != 1 || batches[0].len() != provider.profile().dimension {
        return Err("embedding provider returned an unexpected query vector shape".to_owned());
    }
    let embedding = batches.pop().expect("one embedding batch");
    let embedding = embedding
        .into_iter()
        .map(|value| {
            if !value.is_finite() || value.abs() > f64::from(f32::MAX) {
                return Err(
                    "embedding provider returned a non-finite or out-of-range value".to_owned(),
                );
            }
            Ok(value as f32)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let metric = match provider
        .profile()
        .similarity_metric
        .to_ascii_lowercase()
        .as_str()
    {
        "cosine" => DistanceMetric::Cosine,
        "l2" => DistanceMetric::L2,
        "ip" => DistanceMetric::InnerProduct,
        _ => return Err("embedding profile has an unsupported similarity metric".to_owned()),
    };
    let vectors = vector_store
        .query_node_index(
            index,
            &VectorQuery {
                embedding,
                limit,
                metadata: Default::default(),
                metric,
            },
        )
        .await
        .map_err(|error| error.to_string())?;
    Ok(fuse_hybrid_hits(&lexical, &vectors))
}

#[cfg(test)]
mod tests {
    use super::{
        HybridSearchHit, fuse_hybrid_hits, open_index, parse_hybrid_search_query,
        project_hybrid_search_result, search_fts, upsert_entries,
    };
    use kogwistar_contracts::SearchIndexUpsertV1;
    use kogwistar_engine::embedding::{EmbeddingFuture, EmbeddingProfile, EmbeddingProvider};
    use kogwistar_store::{
        EventReadStore, GraphRecord, NodeIndexStore, StoreError, StoreResult, VectorMatch,
        VectorQuery,
    };
    use kogwistar_store_memory::InMemoryStore;
    use serde_json::Value;
    use std::path::PathBuf;
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    };

    struct FixedEmbeddingProvider {
        profile: EmbeddingProfile,
    }

    impl EmbeddingProvider for FixedEmbeddingProvider {
        fn profile(&self) -> &EmbeddingProfile {
            &self.profile
        }

        fn embed<'a>(&'a self, texts: &'a [String]) -> EmbeddingFuture<'a> {
            Box::pin(async move { Ok(texts.iter().map(|_| vec![0.6_f64, 0.8_f64]).collect()) })
        }
    }

    #[derive(Clone)]
    struct FailOnceNodeIndexStore {
        inner: InMemoryStore,
        fail_next: Arc<AtomicBool>,
    }

    impl NodeIndexStore for FailOnceNodeIndexStore {
        async fn upsert_node_index(
            &self,
            index: &str,
            records: Vec<GraphRecord>,
        ) -> StoreResult<()> {
            if self.fail_next.swap(false, Ordering::SeqCst) {
                return Err(StoreError::Backend {
                    backend: "test-vector-index".to_owned(),
                    message: "injected projection failure".to_owned(),
                });
            }
            self.inner.upsert_node_index(index, records).await
        }

        async fn get_node_index(
            &self,
            index: &str,
            ids: &[String],
        ) -> StoreResult<Vec<GraphRecord>> {
            self.inner.get_node_index(index, ids).await
        }

        async fn query_node_index(
            &self,
            index: &str,
            query: &VectorQuery,
        ) -> StoreResult<Vec<VectorMatch>> {
            self.inner.query_node_index(index, query).await
        }
    }

    #[test]
    fn hybrid_search_query_parses_defaults_and_encoded_values() {
        assert_eq!(
            parse_hybrid_search_query("/api/search_index_hybrid?q=graph+retrieval"),
            Ok(("graph retrieval".to_owned(), 10, false))
        );
        assert_eq!(
            parse_hybrid_search_query(
                "/api/search_index_hybrid?q=na%C3%AFve&limit=25&resolve_node=YES"
            ),
            Ok(("naïve".to_owned(), 25, true))
        );
    }

    #[test]
    fn hybrid_search_query_rejects_missing_query_bad_flags_and_out_of_range_limits() {
        for (query, expected) in [
            (
                "/api/search_index_hybrid",
                "q is required and must not be empty",
            ),
            (
                "/api/search_index_hybrid?q=%20%20",
                "q is required and must not be empty",
            ),
            (
                "/api/search_index_hybrid?q=x&limit=0",
                "limit must be between 1 and 1000",
            ),
            (
                "/api/search_index_hybrid?q=x&limit=1001",
                "limit must be between 1 and 1000",
            ),
            (
                "/api/search_index_hybrid?q=x&limit=many",
                "limit must be an integer",
            ),
            (
                "/api/search_index_hybrid?q=x&resolve_node=sometimes",
                "resolve_node must be a boolean",
            ),
        ] {
            assert_eq!(parse_hybrid_search_query(query).unwrap_err(), expected);
        }
    }

    #[test]
    fn hybrid_result_projection_acl_filters_before_applying_limit() {
        let hits = ["private", "visible-1", "visible-2"].map(|node_id| HybridSearchHit {
            index_key: format!("key-{node_id}"),
            node_id: node_id.to_owned(),
            canonical_title: node_id.to_owned(),
            provision: String::new(),
            document_id: None,
            fts_score: 0.0,
            vec_score: 0.0,
            hybrid_score: 0.0,
        });
        let authorized_nodes = ["visible-1", "visible-2"].map(|id| GraphRecord {
            id: id.to_owned(),
            document: Some(format!("document for {id}")),
            metadata: serde_json::Map::from_iter([(
                "canonical_id".to_owned(),
                Value::String(id.to_owned()),
            )]),
            embedding: None,
        });

        let projected = project_hybrid_search_result("q", &hits, &authorized_nodes, true, 1);
        let results = projected["results"].as_array().expect("results array");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["node_id"], "visible-1");
        assert_eq!(results[0]["documents"], "document for visible-1");
        assert!(results.iter().all(|hit| hit["node_id"] != "private"));
    }

    #[tokio::test]
    async fn hybrid_search_reads_fts_and_node_index_then_fuses_candidates() {
        let (request, _) = fixture();
        let entry = request
            .index
            .iter()
            .find(|entry| entry.node_id == "node-42")
            .expect("fixture includes node-42");
        let index_key = kogwistar_contracts::make_search_index_key(
            &entry.node_id,
            &entry.canonical_title,
            &entry.provision,
        );
        let vector_store = InMemoryStore::default();
        vector_store
            .upsert_node_index(
                "nodes_index",
                vec![GraphRecord {
                    id: index_key.clone(),
                    document: None,
                    metadata: serde_json::from_value(serde_json::json!({
                        "index_key": index_key,
                        "target_node_id": entry.node_id,
                        "canonical_title": entry.canonical_title,
                        "provision": entry.provision,
                        "doc_id": entry.doc_id,
                    }))
                    .unwrap(),
                    embedding: Some(vec![0.6, 0.8]),
                }],
            )
            .await
            .expect("seed vector index");
        let mut connection = open_index(":memory:").expect("open lexical index");
        upsert_entries(&mut connection, &request.index).expect("seed FTS index");
        let fts = Mutex::new(connection);
        let provider = FixedEmbeddingProvider {
            profile: EmbeddingProfile {
                provider: "test".to_owned(),
                model: "fixed-test-embedding".to_owned(),
                dimension: 2,
                similarity_metric: "COSINE".to_owned(),
                endpoint_fingerprint: None,
                max_sequence_length: None,
                crop_token_budget: None,
                tokenizer_fingerprint: None,
                crop_policy: None,
                embedding_kind: Some("single_vector".to_owned()),
                model_revision: None,
                preprocessing_fingerprint: None,
                max_image_patches: None,
            },
        };

        let hits =
            super::hybrid_search(&fts, &vector_store, "nodes_index", &provider, "permit", 10)
                .await
                .expect("query both indexes");
        let hit = hits
            .iter()
            .find(|hit| hit.node_id == "node-42")
            .expect("candidate appears in hybrid result");
        assert!(hit.fts_score > 0.0);
        assert!(hit.vec_score > 0.0);
        assert!(hit.hybrid_score > 0.0);
    }

    #[tokio::test]
    async fn search_index_admission_is_durable_idempotent_and_retryable() {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/search-index-entry-v1.json"
        )))
        .expect("search-index entry fixture");
        let request: SearchIndexUpsertV1 =
            serde_json::from_value(fixture["request"].clone()).expect("fixture request");
        let item = request.index[0].clone();
        let key = kogwistar_contracts::make_search_index_key(
            &item.node_id,
            &item.canonical_title,
            &item.provision,
        );
        let event_store = InMemoryStore::default();
        let vector_store = FailOnceNodeIndexStore {
            inner: InMemoryStore::default(),
            fail_next: Arc::new(AtomicBool::new(true)),
        };
        let connection = open_index(":memory:").expect("open lexical index");
        let fts = Mutex::new(connection);
        let provider = FixedEmbeddingProvider {
            profile: EmbeddingProfile {
                provider: "test".to_owned(),
                model: "fixed-test-embedding".to_owned(),
                dimension: 2,
                similarity_metric: "COSINE".to_owned(),
                endpoint_fingerprint: None,
                max_sequence_length: None,
                crop_token_budget: None,
                tokenizer_fingerprint: None,
                crop_policy: None,
                embedding_kind: Some("single_vector".to_owned()),
                model_revision: None,
                preprocessing_fingerprint: None,
                max_image_patches: None,
            },
        };

        assert!(
            super::admit_search_index_entries(
                &event_store,
                &vector_store,
                &fts,
                &provider,
                "workspace",
                std::slice::from_ref(&item),
            )
            .await
            .is_err()
        );
        assert_eq!(
            EventReadStore::replay_cursor(&event_store, "workspace", "search-index-fts-vector-v1")
                .await
                .expect("read cursor")
                .last_seq,
            0,
            "failed vector projection must not advance the canonical event cursor"
        );
        assert_eq!(
            search_fts(&fts.lock().expect("FTS lock"), &item.canonical_title, 10)
                .expect("FTS entry remains idempotently retryable")
                .len(),
            1
        );

        let replay = super::admit_search_index_entries(
            &event_store,
            &vector_store,
            &fts,
            &provider,
            "workspace",
            std::slice::from_ref(&item),
        )
        .await
        .expect("retry event admission and both projections");
        assert_eq!(replay.entries_admitted, 1);
        assert_eq!(replay.events_projected, 1);
        assert_eq!(replay.last_event_seq, 1);
        let vector_id = format!("idx:{key}");
        assert_eq!(
            vector_store
                .get_node_index("nodes_index", &[vector_id])
                .await
                .expect("read vector projection")
                .len(),
            1
        );
        let repeated = super::admit_search_index_entries(
            &event_store,
            &vector_store,
            &fts,
            &provider,
            "workspace",
            std::slice::from_ref(&item),
        )
        .await
        .expect("repeated admission is idempotent");
        assert_eq!(repeated.entries_admitted, 1);
        assert_eq!(repeated.events_projected, 0);
        assert_eq!(
            EventReadStore::replay_events(&event_store, "workspace", 0, 10)
                .await
                .expect("read durable events")
                .len(),
            1,
            "same namespace and entry must not append duplicate events"
        );

        let mut revised = item.clone();
        revised.keywords = vec!["latesttoken".to_owned()];
        revised.aliases.clear();
        let updated = super::admit_search_index_entries(
            &event_store,
            &vector_store,
            &fts,
            &provider,
            "workspace",
            std::slice::from_ref(&revised),
        )
        .await
        .expect("changed content appends and projects a newer event");
        assert_eq!(updated.last_event_seq, 2);
        assert_eq!(updated.events_projected, 1);
        let old_hits = search_fts(&fts.lock().expect("FTS lock"), "authorization", 10)
            .expect("query superseded lexical terms");
        assert!(old_hits.iter().all(|hit| hit.node_id != item.node_id));
        let latest_hits = search_fts(&fts.lock().expect("FTS lock"), "latesttoken", 10)
            .expect("query revised lexical terms");
        assert!(latest_hits.iter().any(|hit| hit.node_id == item.node_id));
        let projected = vector_store
            .get_node_index("nodes_index", &[format!("idx:{key}")])
            .await
            .expect("read updated vector projection");
        assert_eq!(projected.len(), 1);
        assert_eq!(projected[0].metadata["keywords"], "[\"latesttoken\"]");
        assert_eq!(
            EventReadStore::replay_events(&event_store, "workspace", 0, 10)
                .await
                .expect("read update history")
                .len(),
            2,
            "changed content is a new event, not a duplicate of the prior payload"
        );
    }

    #[tokio::test]
    async fn search_index_upsert_effect_requires_write_scope_and_matches_python_response() {
        let (upsert, _) = fixture();
        let body = serde_json::to_vec(&upsert).expect("fixture request serializes");
        let store = InMemoryStore::default();
        let fts = Mutex::new(open_index(":memory:").expect("open lexical index"));
        let provider = FixedEmbeddingProvider {
            profile: EmbeddingProfile {
                provider: "test".to_owned(),
                model: "fixed-test-embedding".to_owned(),
                dimension: 2,
                similarity_metric: "COSINE".to_owned(),
                endpoint_fingerprint: None,
                max_sequence_length: None,
                crop_token_budget: None,
                tokenizer_fingerprint: None,
                crop_policy: None,
                embedding_kind: Some("single_vector".to_owned()),
                model_revision: None,
                preprocessing_fingerprint: None,
                max_image_patches: None,
            },
        };
        let capabilities = Arc::new(Mutex::new(crate::CapabilityState::default()));

        let denied = super::search_index_upsert_effect(
            &crate::ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/add_index_entries".to_owned(),
                body: body.clone(),
                principal: serde_json::json!({
                    "sub": "reader",
                    "role": "ro",
                    "ns": ["docs"],
                    "storage_ns": "docs",
                    "capabilities": ["read_graph"]
                }),
            },
            &capabilities,
            &store,
            &store,
            None,
            None,
        )
        .await;
        assert_eq!(denied.status, axum::http::StatusCode::FORBIDDEN.as_u16());

        let wrong_namespace = super::search_index_upsert_effect(
            &crate::ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/add_index_entries".to_owned(),
                body: body.clone(),
                principal: serde_json::json!({
                    "sub": "writer",
                    "role": "rw",
                    "ns": ["other"],
                    "storage_ns": "docs",
                    "capabilities": ["write_graph"]
                }),
            },
            &capabilities,
            &store,
            &store,
            None,
            None,
        )
        .await;
        assert_eq!(
            wrong_namespace.status,
            axum::http::StatusCode::FORBIDDEN.as_u16()
        );

        let wildcard_without_storage_namespace = super::search_index_upsert_effect(
            &crate::ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/add_index_entries".to_owned(),
                body: body.clone(),
                principal: serde_json::json!({
                    "sub": "writer",
                    "role": "rw",
                    "ns": ["*"],
                    "capabilities": ["write_graph"]
                }),
            },
            &capabilities,
            &store,
            &store,
            None,
            None,
        )
        .await;
        assert_eq!(
            wildcard_without_storage_namespace.status,
            axum::http::StatusCode::FORBIDDEN.as_u16()
        );

        let accepted = super::search_index_upsert_effect(
            &crate::ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/add_index_entries".to_owned(),
                body,
                principal: serde_json::json!({
                    "sub": "writer",
                    "role": "rw",
                    "ns": ["docs"],
                    "storage_ns": "docs",
                    "capabilities": ["write_graph"]
                }),
            },
            &capabilities,
            &store,
            &store,
            Some(&fts),
            Some(&provider),
        )
        .await;
        assert_eq!(accepted.status, axum::http::StatusCode::OK.as_u16());
        assert_eq!(
            serde_json::from_slice::<Value>(&accepted.body).expect("response JSON"),
            serde_json::json!({"ok": true})
        );
        let events = EventReadStore::replay_events(&store, "docs", 0, 10)
            .await
            .expect("events are written only to the authorized namespace");
        assert_eq!(events.len(), upsert.index.len());
        assert!(
            events
                .iter()
                .all(|event| event.entity_kind == "search_index")
        );
    }

    fn fixture() -> (SearchIndexUpsertV1, Value) {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/search-index-entry-v1.json"
        )))
        .expect("search index fixture is valid");
        let request = serde_json::from_value(fixture["request"].clone())
            .expect("fixture request follows shared contract");
        (request, fixture)
    }

    #[test]
    fn rust_fts_projection_matches_python_index_schema() {
        let (request, fixture) = fixture();
        let mut connection = open_index(":memory:").expect("open in-memory index");
        upsert_entries(&mut connection, &request.index).expect("persist lexical entries");

        let count: i64 = connection
            .query_row("SELECT count(*) FROM semantic_index", [], |row| row.get(0))
            .expect("read persisted entry count");
        assert_eq!(count as usize, request.index.len());

        for (entry, expected) in request
            .index
            .iter()
            .zip(fixture["expected"].as_array().expect("expected entries"))
        {
            let key = kogwistar_contracts::make_search_index_key(
                &entry.node_id,
                &entry.canonical_title,
                &entry.provision,
            );
            let row: (String, String, String, String, Option<String>) = connection
                .query_row(
                    "SELECT index_key,node_id,canonical_title,provision,document_id
                     FROM semantic_index WHERE index_key=?1",
                    [&key],
                    |row| {
                        Ok((
                            row.get(0)?,
                            row.get(1)?,
                            row.get(2)?,
                            row.get(3)?,
                            row.get(4)?,
                        ))
                    },
                )
                .expect("lookup Python-compatible row");
            assert_eq!(row.0, expected["index_key"]);
            assert_eq!(row.1, entry.node_id);
            assert_eq!(row.2, entry.canonical_title);
            assert_eq!(row.3, entry.provision);
            assert_eq!(row.4.as_deref(), entry.doc_id.as_deref());
        }
        let hits = search_fts(&connection, "permit", 10).expect("FTS query");
        assert!(hits.iter().any(|hit| hit.node_id == "node-42"));
    }

    #[test]
    fn hybrid_fusion_matches_shared_python_golden_fixture() {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/search-index-hybrid-v1.json"
        )))
        .expect("shared hybrid-search fixture is valid JSON");
        let lexical = fixture["lexical_hits"]
            .as_array()
            .expect("fixture lexical hits")
            .iter()
            .map(|hit| super::LexicalIndexHit {
                index_key: hit["index_key"].as_str().unwrap().to_owned(),
                node_id: hit["node_id"].as_str().unwrap().to_owned(),
                canonical_title: hit["canonical_title"].as_str().unwrap().to_owned(),
                provision: hit["provision"].as_str().unwrap().to_owned(),
                document_id: hit["document_id"].as_str().map(str::to_owned),
                bm25_score: hit["bm25_score"].as_f64().unwrap(),
            })
            .collect::<Vec<_>>();
        let vectors = fixture["vector_hits"]
            .as_array()
            .expect("fixture vector hits")
            .iter()
            .map(|hit| VectorMatch {
                record: GraphRecord {
                    id: hit["id"].as_str().unwrap().to_owned(),
                    document: None,
                    metadata: serde_json::from_value(hit["metadata"].clone()).unwrap(),
                    embedding: None,
                },
                distance: hit["distance"].as_f64().unwrap(),
            })
            .collect::<Vec<_>>();
        let fused = fuse_hybrid_hits(&lexical, &vectors);
        let expected = fixture["expected"].as_array().unwrap();
        assert_eq!(fused.len(), expected.len());
        for (actual, expected) in fused.iter().zip(expected) {
            assert_eq!(actual.index_key, expected["index_key"]);
            assert_eq!(actual.node_id, expected["node_id"]);
            assert_eq!(actual.fts_score, expected["fts_score"].as_f64().unwrap());
            assert_eq!(actual.vec_score, expected["vec_score"].as_f64().unwrap());
            assert_eq!(
                actual.hybrid_score,
                expected["hybrid_score"].as_f64().unwrap()
            );
            assert_eq!(
                actual.document_id.as_deref(),
                expected["document_id"].as_str()
            );
        }
    }

    #[test]
    fn index_reopen_and_upsert_keep_fts_projection_in_sync() {
        let (request, _) = fixture();
        let path = PathBuf::from(std::env::temp_dir()).join(format!(
            "kogwistar-search-index-{}.sqlite3",
            uuid::Uuid::new_v4()
        ));
        {
            let mut connection = open_index(&path).expect("open persistent index");
            upsert_entries(&mut connection, &request.index).expect("initial upsert");
        }
        {
            let mut connection = open_index(&path).expect("reopen persistent index");
            upsert_entries(&mut connection, &request.index).expect("idempotent upsert");
            let hits = search_fts(&connection, "permit", 10).expect("query reopened FTS");
            assert!(hits.iter().any(|hit| hit.node_id == "node-42"));
            let count: i64 = connection
                .query_row("SELECT count(*) FROM semantic_index", [], |row| row.get(0))
                .expect("verify upsert identity");
            assert_eq!(count as usize, request.index.len());
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn update_and_delete_keep_external_content_fts_in_sync() {
        let (request, _) = fixture();
        let mut connection = open_index(":memory:").expect("open in-memory index");
        upsert_entries(&mut connection, &request.index).expect("initial upsert");

        let mut revised = request.index[0].clone();
        revised.keywords = vec!["revisedtoken".to_owned()];
        revised.aliases.clear();
        upsert_entries(&mut connection, &[revised.clone()]).expect("update entry");

        assert!(
            search_fts(&connection, "authorization", 10)
                .expect("query removed alias")
                .iter()
                .all(|hit| hit.node_id != revised.node_id)
        );
        assert!(
            search_fts(&connection, "revisedtoken", 10)
                .expect("query revised keyword")
                .iter()
                .any(|hit| hit.node_id == revised.node_id)
        );

        let index_key = kogwistar_contracts::make_search_index_key(
            &revised.node_id,
            &revised.canonical_title,
            &revised.provision,
        );
        connection
            .execute(
                "DELETE FROM semantic_index WHERE index_key=?1",
                [&index_key],
            )
            .expect("delete canonical row");
        assert!(
            search_fts(&connection, "revisedtoken", 10)
                .expect("query deleted entry")
                .iter()
                .all(|hit| hit.node_id != revised.node_id)
        );
    }

    #[test]
    fn opening_legacy_contentless_fts_database_migrates_and_rebuilds() {
        let path = std::env::temp_dir().join(format!(
            "kogwistar-search-index-legacy-{}.sqlite3",
            uuid::Uuid::new_v4()
        ));
        {
            let connection = rusqlite::Connection::open(&path).expect("legacy database");
            connection
                .execute_batch(
                    "CREATE TABLE semantic_index (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        index_key TEXT NOT NULL UNIQUE,
                        node_id TEXT NOT NULL,
                        canonical_title TEXT NOT NULL,
                        keywords TEXT NOT NULL DEFAULT '',
                        aliases TEXT NOT NULL DEFAULT '',
                        provision TEXT NOT NULL,
                        document_id TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(node_id, canonical_title, provision)
                    );
                    CREATE VIRTUAL TABLE semantic_index_fts USING fts5(
                        index_key UNINDEXED, canonical_title, provision, keywords, aliases, content=''
                    );",
                )
                .expect("create legacy contentless FTS schema");
            connection
                .execute(
                    "INSERT INTO semantic_index VALUES
                        (1,'node-42|Permit Review|Review the permit before work begins.',
                         'node-42','Permit Review','permit review','approval authorization',
                         'Review the permit before work begins.','safety-manual',CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)",
                    [],
                )
                .expect("insert canonical legacy row");
        }
        {
            let connection = open_index(&path).expect("migrate legacy index");
            let schema: String = connection
                .query_row(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name='semantic_index_fts'",
                    [],
                    |row| row.get(0),
                )
                .expect("read migrated FTS schema");
            assert!(schema.contains("content='semantic_index'"));
            assert!(
                search_fts(&connection, "permit", 10)
                    .expect("query rebuilt index")
                    .iter()
                    .any(|hit| hit.node_id == "node-42")
            );
        }
        let _ = std::fs::remove_file(&path);
    }
}
