//! Async-first storage contracts. Shadow handles expose reads only.
use kogwistar_contracts::{
    ContextSnapshotV1, ConversationContractError, ConversationSummaryV1, ConversationTranscriptV1,
    ConversationTurnV1, EntityEventEnvelope, GraphEntityKind, GraphModelEnvelope,
    ToolAuditContractError, ToolAuditItemV1,
};
use serde_json::{Map, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::future::Future;
use thiserror::Error;

pub type StoreResult<T> = Result<T, StoreError>;
pub type EntityEvent = EntityEventEnvelope;

/// Shared ADR-015 serving-projection shape. Both durable stores must emit the
/// same namespace, cursor metadata, and schema version for a reduced runtime.
pub const RUNTIME_CURRENT_STATE_NAMESPACE: &str = "workflow_runtime_current_state";
pub const RUNTIME_PROJECTION_SCHEMA_VERSION: i64 = 1;

pub fn runtime_checkpoint_namespace(conversation_id: &str) -> String {
    format!("{conversation_id}:workflow_checkpoint_latest")
}

pub fn runtime_status_namespace(conversation_id: &str) -> String {
    format!("{conversation_id}:workflow_run_status")
}

pub fn runtime_projection_write(
    payload: Map<String, Value>,
    seq: i64,
    materialization_status: impl Into<String>,
) -> NamedProjectionWrite {
    NamedProjectionWrite {
        payload,
        last_authoritative_seq: seq,
        last_materialized_seq: seq,
        projection_schema_version: RUNTIME_PROJECTION_SCHEMA_VERSION,
        materialization_status: materialization_status.into(),
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuthUser {
    pub user_id: String,
    pub email: String,
    pub display_name: Option<String>,
    pub is_active: bool,
    pub global_role: Option<String>,
    pub global_ns: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExternalIdentity {
    pub issuer: String,
    pub subject: String,
    pub user_id: String,
    pub email: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolveExternalIdentity {
    pub issuer: String,
    pub subject: String,
    pub email: String,
    pub display_name: Option<String>,
    pub new_user_id: String,
    pub default_role: String,
    pub default_ns: String,
}

/// Standalone auth-database contract. Auth storage deliberately remains
/// separate from engine metadata because Python configures it through
/// `AUTH_DB_URL` and defaults to an independent `auth.sqlite` database.
pub trait AuthIdentityStore: Send + Sync {
    fn auth_user(
        &self,
        user_id: &str,
    ) -> impl Future<Output = StoreResult<Option<AuthUser>>> + Send;
    fn external_identity(
        &self,
        issuer: &str,
        subject: &str,
    ) -> impl Future<Output = StoreResult<Option<ExternalIdentity>>> + Send;
    fn resolve_external_identity(
        &self,
        request: ResolveExternalIdentity,
    ) -> impl Future<Output = StoreResult<AuthUser>> + Send;
}

/// Backend-neutral helpers for ABI-preserving integer queue timestamps.
pub fn integer_timestamp(value: i64) -> Value {
    Value::from(value)
}
pub fn timestamp_i64(value: &Value) -> i64 {
    value.as_i64().unwrap_or_default()
}

#[derive(Clone, Debug, Error, PartialEq)]
pub enum StoreError {
    #[error("namespace must not be empty")]
    EmptyNamespace,
    #[error("graph record id must not be empty")]
    EmptyRecordId,
    #[error("vector dimension mismatch: expected {expected}, got {actual}")]
    VectorDimensionMismatch { expected: usize, actual: usize },
    #[error("vector contains a non-finite value")]
    NonFiniteVector,
    #[error("cosine distance is undefined for a zero vector")]
    ZeroNormVector,
    #[error("replay cursor {cursor} exceeds latest event sequence {latest}")]
    CursorOutOfRange { cursor: i64, latest: i64 },
    #[error("replay cursor cannot regress from {current} to {requested}")]
    CursorRegresses { current: i64, requested: i64 },
    #[error(
        "restored event sequence for namespace {namespace:?} must be positive and strictly increasing; got {seq}"
    )]
    InvalidRestoredEventSequence { namespace: String, seq: i64 },
    #[error("restored event id {event_id:?} is duplicated in namespace {namespace:?}")]
    DuplicateRestoredEventId { namespace: String, event_id: String },
    #[error(
        "event id {event_id:?} belongs to namespace {existing_namespace:?}, not {requested_namespace:?}"
    )]
    EventIdNamespaceCollision {
        event_id: String,
        existing_namespace: String,
        requested_namespace: String,
    },
    #[error("restored named projection key {key:?} is duplicated in namespace {namespace:?}")]
    DuplicateRestoredProjection { namespace: String, key: String },
    #[error("{identifier} must not be empty or contain a NUL byte")]
    InvalidIdentifier { identifier: &'static str },
    #[error("recovery batch limit must be positive")]
    InvalidRecoveryBatchLimit,
    #[error("invalid entity event payload: {message}")]
    InvalidEntityEventPayload { message: String },
    #[error("unsupported entity event operation {op:?}")]
    UnsupportedEntityEventOperation { op: String },
    #[error("{backend} store failure: {message}")]
    Backend { backend: String, message: String },
}

#[derive(Clone, Debug, PartialEq)]
pub struct GraphRecord {
    pub id: String,
    pub document: Option<String>,
    pub metadata: Map<String, Value>,
    pub embedding: Option<Vec<f32>>,
}

/// A document projection write owned by one concrete backend.
///
/// This is intentionally separate from graph mutation: document ingestion has
/// different lifecycle and embedding semantics, and must not be disguised as
/// a node write at a language seam.
#[derive(Clone, Debug, PartialEq)]
pub struct DocumentProjectionUpsert {
    pub scope: GraphScope,
    pub record: GraphRecord,
    pub embedding_dim: usize,
}

/// Deterministic adjacency result shared by native graph-read adapters.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GraphNeighbors {
    pub nodes: BTreeSet<String>,
    pub edges: BTreeSet<String>,
}

/// Query fields for the storage-neutral equivalent of Python `find_edges`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GraphFindEdgesQuery {
    pub relation: Option<String>,
    pub src_label_contains: Option<String>,
    pub tgt_label_contains: Option<String>,
    pub doc_id: Option<String>,
}

/// Query fields for the storage-neutral equivalent of Python
/// `GraphQuery.shortest_path`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphShortestPathQuery {
    pub src_id: String,
    pub dst_id: String,
    pub doc_id: Option<String>,
    pub max_depth: usize,
}

/// Query fields for the storage-neutral equivalent of Python `k_hop`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphKHopQuery {
    pub start_ids: Vec<String>,
    pub k: usize,
    pub doc_id: Option<String>,
}

/// Principal inputs required to evaluate persisted graph ACL records.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GraphAclPrincipal {
    pub principal_id: String,
    pub groups: BTreeSet<String>,
    pub security_scope: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct GraphAclRule {
    version: i64,
    mode: String,
    owner_id: Option<String>,
    security_scope: Option<String>,
    shared_principals: BTreeSet<String>,
    shared_groups: BTreeSet<String>,
    tombstoned: bool,
}

fn record_json(record: &GraphRecord) -> Option<Value> {
    record
        .document
        .as_deref()
        .and_then(|document| serde_json::from_str(document).ok())
}

fn record_field(record: &GraphRecord, field: &str) -> Option<Value> {
    if let Some(value) = record.metadata.get(field) {
        return Some(value.clone());
    }
    let document = record_json(record)?;
    for container in [document.get("metadata"), document.get("properties")] {
        if let Some(value) = container.and_then(|value| value.get(field)) {
            return Some(value.clone());
        }
    }
    None
}

fn value_string(record: &GraphRecord, field: &str) -> Option<String> {
    record_field(record, field).and_then(|value| value.as_str().map(str::to_owned))
}

fn value_strings(record: &GraphRecord, field: &str) -> BTreeSet<String> {
    record_field(record, field)
        .and_then(|value| value.as_array().cloned())
        .unwrap_or_default()
        .into_iter()
        .filter_map(|value| value.as_str().map(|item| item.to_ascii_lowercase()))
        .collect()
}

fn acl_rank(mode: &str) -> i64 {
    match mode {
        "public" => 0,
        "group" => 1,
        "shared" => 2,
        "scope" => 3,
        "private" => 4,
        _ => 99,
    }
}

fn is_acl_record(record: &GraphRecord) -> bool {
    value_string(record, "entity_type").as_deref() == Some("acl_record")
}

fn acl_rule_for(
    record: &GraphRecord,
    truth_graph: &str,
) -> Option<((String, String), GraphAclRule)> {
    if !is_acl_record(record)
        || value_string(record, "acl_truth_graph").as_deref() != Some(truth_graph)
    {
        return None;
    }
    let grain = value_string(record, "acl_target_grain")?;
    if grain != "node" && grain != "edge" {
        return None;
    }
    if value_string(record, "acl_target_item_id").is_some_and(|target_item| !target_item.is_empty())
    {
        return None;
    }
    let target_id = value_string(record, "acl_target_entity_id")?;
    let version = record_field(record, "acl_version")?.as_i64()?;
    let mode = value_string(record, "acl_mode")?.to_ascii_lowercase();
    Some((
        (grain, target_id),
        GraphAclRule {
            version,
            mode,
            owner_id: value_string(record, "owner_id"),
            security_scope: value_string(record, "security_scope"),
            shared_principals: value_strings(record, "shared_with_principals"),
            shared_groups: value_strings(record, "shared_with_groups"),
            tombstoned: record_field(record, "tombstoned")
                .and_then(|value| value.as_bool())
                .unwrap_or(false),
        },
    ))
}

fn acl_rule_visible(rule: &GraphAclRule, principal: &GraphAclPrincipal) -> bool {
    if rule.tombstoned {
        return false;
    }
    if rule.mode == "public" {
        return true;
    }
    if rule.owner_id.as_deref() == Some(principal.principal_id.as_str()) {
        return true;
    }
    match rule.mode.as_str() {
        "private" => false,
        "scope" => {
            principal.security_scope.is_some() && principal.security_scope == rule.security_scope
        }
        "shared" => rule
            .shared_principals
            .contains(&principal.principal_id.to_ascii_lowercase()),
        "group" => !rule.shared_groups.is_disjoint(&principal.groups),
        _ => false,
    }
}

/// Filter graph projections using the same fail-closed ACL rules as Python.
///
/// ACL records are persisted as graph nodes. Missing records deny access, and
/// ACL helper nodes never enter the returned graph projection.
pub fn graph_records_visible_by_acl(
    nodes: &[GraphRecord],
    edges: &[GraphRecord],
    truth_graph: &str,
    principal: &GraphAclPrincipal,
) -> (Vec<GraphRecord>, Vec<GraphRecord>) {
    let mut rules: BTreeMap<(String, String), GraphAclRule> = BTreeMap::new();
    for record in nodes {
        let Some((key, rule)) = acl_rule_for(record, truth_graph) else {
            continue;
        };
        let replace = rules.get(&key).is_none_or(|current| {
            (rule.version, acl_rank(&rule.mode)) > (current.version, acl_rank(&current.mode))
        });
        if replace {
            rules.insert(key, rule);
        }
    }
    let visible = |record: &GraphRecord, grain: &str| {
        if is_acl_record(record) {
            return false;
        }
        rules
            .get(&(grain.to_owned(), record.id.clone()))
            .is_some_and(|rule| acl_rule_visible(rule, principal))
    };
    (
        nodes
            .iter()
            .filter(|record| visible(record, "node"))
            .cloned()
            .collect(),
        edges
            .iter()
            .filter(|record| visible(record, "edge"))
            .cloned()
            .collect(),
    )
}

/// Project already-authorized graph records into the versioned tool-audit
/// payload shape used by the Python service.
///
/// The caller must provide records after namespace and ACL filtering.  This
/// helper deliberately does not inspect or infer authorization, so it cannot
/// become an alternate read path around the graph authority boundary.
pub fn tool_audit_items_from_visible_records(
    records: &[GraphRecord],
    conversation_id: Option<&str>,
    limit: usize,
) -> Result<Vec<ToolAuditItemV1>, ToolAuditContractError> {
    let mut items = Vec::new();
    for record in records {
        if is_tombstoned(&record.metadata) {
            continue;
        }
        let Some(entity_type) = value_string(record, "entity_type") else {
            continue;
        };
        if !matches!(entity_type.as_str(), "tool_call" | "tool_result") {
            continue;
        }
        let item_conversation_id = value_string(record, "conversation_id");
        if conversation_id.is_some_and(|expected| item_conversation_id.as_deref() != Some(expected))
        {
            continue;
        }
        let turn_index = record_field(record, "turn_index").and_then(|value| value.as_i64());
        let summary = value_string(record, "summary").unwrap_or_default();
        let tool_name = value_string(record, "tool_name").unwrap_or_default();
        let properties = record_json(record)
            .and_then(|document| document.get("properties").cloned())
            .and_then(|value| value.as_object().cloned())
            .unwrap_or_default()
            .into_iter()
            .collect();
        let item = ToolAuditItemV1 {
            schema_version: kogwistar_contracts::TOOL_AUDIT_WIRE_VERSION,
            node_id: record.id.clone(),
            entity_type,
            tool_name,
            conversation_id: item_conversation_id,
            turn_index,
            summary,
            properties,
        };
        item.validate()?;
        items.push(item);
        if items.len() >= limit {
            break;
        }
    }
    Ok(items)
}

/// Project already-authorized conversation nodes into the shared transcript
/// contract.  Conversation ordering is strict here: duplicate turn indexes
/// are rejected instead of silently changing the Python/Rust interpretation.
pub fn conversation_transcript_from_visible_records(
    records: &[GraphRecord],
    conversation_id: &str,
    limit: usize,
) -> Result<ConversationTranscriptV1, ConversationContractError> {
    let mut turns = records
        .iter()
        .filter(|record| !is_tombstoned(&record.metadata))
        .filter_map(|record| {
            let entity_type = value_string(record, "entity_type")?;
            if !matches!(entity_type.as_str(), "conversation_turn" | "assistant_turn") {
                return None;
            }
            if value_string(record, "conversation_id").as_deref() != Some(conversation_id) {
                return None;
            }
            let turn_index = record_field(record, "turn_index")?.as_i64()?;
            let role = value_string(record, "role")?;
            let content = value_string(record, "summary").unwrap_or_default();
            Some(ConversationTurnV1 {
                schema_version: kogwistar_contracts::CONVERSATION_WIRE_VERSION,
                node_id: record.id.clone(),
                turn_index,
                role,
                content,
                entity_type,
            })
        })
        .collect::<Vec<_>>();
    turns.sort_by(|left, right| {
        left.turn_index
            .cmp(&right.turn_index)
            .then_with(|| left.node_id.cmp(&right.node_id))
    });
    let mut transcript = ConversationTranscriptV1 {
        schema_version: kogwistar_contracts::CONVERSATION_WIRE_VERSION,
        conversation_id: conversation_id.to_owned(),
        turns,
    };
    transcript.validate()?;
    transcript.turns.truncate(limit);
    Ok(transcript)
}

/// Project a conversation start node and its authorized turns into the
/// versioned summary contract. The tail derives from the same strictly
/// validated transcript used by the transcript endpoint.
pub fn conversation_summary_from_visible_records(
    records: &[GraphRecord],
    conversation_id: &str,
) -> Result<ConversationSummaryV1, ConversationContractError> {
    let transcript =
        conversation_transcript_from_visible_records(records, conversation_id, usize::MAX)?;
    let start = records.iter().find(|record| {
        !is_tombstoned(&record.metadata)
            && value_string(record, "entity_type").as_deref() == Some("conversation_start")
            && value_string(record, "conversation_id").as_deref() == Some(conversation_id)
    });
    let Some(start) = start else {
        return Err(ConversationContractError::EmptyField {
            field: "start_node_id",
        });
    };
    let user_id = value_string(start, "user_id").unwrap_or_default();
    let status = value_string(start, "status").unwrap_or_else(|| "active".to_owned());
    let tail_node_id = transcript
        .turns
        .last()
        .map(|turn| turn.node_id.clone())
        .unwrap_or_else(|| start.id.clone());
    let summary = ConversationSummaryV1 {
        schema_version: kogwistar_contracts::CONVERSATION_WIRE_VERSION,
        conversation_id: conversation_id.to_owned(),
        user_id,
        status,
        start_node_id: start.id.clone(),
        tail_node_id,
        turn_count: transcript.turns.len() as i64,
    };
    summary.validate()?;
    Ok(summary)
}

/// Project the latest already-authorized context snapshot without choosing a
/// database-specific representation. A snapshot must carry an explicit order
/// field so a backend cannot silently select an arbitrary record.
pub fn context_snapshot_from_visible_records(
    records: &[GraphRecord],
    conversation_id: &str,
    run_id: Option<&str>,
    stage: Option<&str>,
) -> Result<ContextSnapshotV1, ConversationContractError> {
    let mut candidates = Vec::new();
    for record in records {
        if is_tombstoned(&record.metadata)
            || value_string(record, "entity_type").as_deref() != Some("context_snapshot")
            || value_string(record, "conversation_id").as_deref() != Some(conversation_id)
            || run_id
                .is_some_and(|expected| value_string(record, "run_id").as_deref() != Some(expected))
            || stage
                .is_some_and(|expected| value_string(record, "stage").as_deref() != Some(expected))
        {
            continue;
        }
        let order = ["snapshot_seq", "created_at_ms", "step_seq"]
            .into_iter()
            .find_map(|field| record_field(record, field).and_then(|value| value.as_i64()))
            .ok_or(ConversationContractError::MissingSnapshotOrder)?;
        candidates.push((order, record.id.clone(), record));
    }
    let Some((_, _, record)) = candidates
        .into_iter()
        .max_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)))
    else {
        return Err(ConversationContractError::EmptyField {
            field: "snapshot_node_id",
        });
    };
    let document = record_json(record).unwrap_or_else(|| Value::Object(Map::new()));
    let object = |field: &str| {
        document
            .get(field)
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .collect::<BTreeMap<_, _>>()
    };
    let snapshot = ContextSnapshotV1 {
        schema_version: kogwistar_contracts::CONVERSATION_WIRE_VERSION,
        snapshot_node_id: record.id.clone(),
        conversation_id: conversation_id.to_owned(),
        metadata: object("metadata"),
        properties: object("properties"),
    };
    snapshot.validate()?;
    Ok(snapshot)
}

/// Compute Python-compatible node/edge adjacency from canonical edge records.
///
/// This helper is deliberately storage-neutral: callers provide the already
/// scope-filtered node records and edge projections. It does not authorize access,
/// infer a namespace, or treat a projection row as authoritative outside the
/// caller's read boundary. Tombstoned rows are excluded so a stale projection
/// cannot reintroduce a deleted relationship.
pub fn graph_neighbors_from_records(
    rid: &str,
    nodes: &[GraphRecord],
    edges: &[GraphRecord],
    doc_id: Option<&str>,
) -> GraphNeighbors {
    graph_neighbors_from_records_with_jump(rid, nodes, edges, doc_id, true)
}

/// Compute adjacency with the caller-selected node-to-opposite-node jump.
///
/// Python's `GraphQuery.k_hop` intentionally disables this jump so traversal
/// advances through explicit edge IDs one layer at a time. Keeping the mode
/// in the storage kernel prevents a native caller from silently changing the
/// graph distance semantics.
pub fn graph_neighbors_from_records_with_jump(
    rid: &str,
    nodes: &[GraphRecord],
    edges: &[GraphRecord],
    doc_id: Option<&str>,
    allow_jump_edge: bool,
) -> GraphNeighbors {
    let mut result = GraphNeighbors::default();
    let node_ids: BTreeSet<&str> = nodes
        .iter()
        .filter(|node| !is_tombstoned(&node.metadata))
        .map(|node| node.id.as_str())
        .collect();
    let edge_ids: BTreeSet<&str> = edges
        .iter()
        .filter(|edge| !is_tombstoned(&edge.metadata))
        .map(|edge| edge.id.as_str())
        .collect();
    let is_node = node_ids.contains(rid);

    for edge in edges {
        if is_tombstoned(&edge.metadata) {
            continue;
        }
        let Some(document) = edge
            .document
            .as_deref()
            .and_then(|value| serde_json::from_str::<Value>(value).ok())
        else {
            continue;
        };
        if doc_id.is_some_and(|expected| {
            edge.metadata
                .get("doc_id")
                .and_then(Value::as_str)
                .or_else(|| document.get("doc_id").and_then(Value::as_str))
                != Some(expected)
        }) {
            continue;
        }

        let ids = |field: &str| {
            document
                .get(field)
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
        };
        let source_nodes: Vec<&str> = ids("source_ids").collect();
        let target_nodes: Vec<&str> = ids("target_ids").collect();
        let source_edges: Vec<&str> = ids("source_edge_ids").collect();
        let target_edges: Vec<&str> = ids("target_edge_ids").collect();

        if is_node && (source_nodes.contains(&rid) || target_nodes.contains(&rid)) {
            result.edges.insert(edge.id.clone());
            if allow_jump_edge {
                for endpoint in source_nodes.iter().chain(target_nodes.iter()) {
                    if *endpoint != rid && node_ids.contains(endpoint) {
                        result.nodes.insert((*endpoint).to_owned());
                    }
                }
            }
        }
        if edge.id == rid {
            for endpoint in source_nodes.iter().chain(target_nodes.iter()) {
                if node_ids.contains(endpoint) {
                    result.nodes.insert((*endpoint).to_owned());
                }
            }
            for endpoint in source_edges.iter().chain(target_edges.iter()) {
                if edge_ids.contains(endpoint) {
                    result.edges.insert((*endpoint).to_owned());
                }
            }
        }
    }

    result
}

/// Find visible edge IDs using the same exact relation/document and
/// case-insensitive endpoint-label semantics as Python `GraphQuery.find_edges`.
/// Callers must provide records after scope and ACL filtering.
pub fn graph_find_edges_from_records(
    query: &GraphFindEdgesQuery,
    nodes: &[GraphRecord],
    edges: &[GraphRecord],
) -> Vec<String> {
    let node_labels = nodes
        .iter()
        .filter(|node| !is_tombstoned(&node.metadata))
        .filter_map(|node| {
            let document = node.document.as_deref()?.parse::<Value>().ok()?;
            Some((
                node.id.clone(),
                document
                    .get("label")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
            ))
        })
        .collect::<BTreeMap<_, _>>();
    let contains = |label: &str, needle: &Option<String>| {
        needle.as_deref().is_none_or(|needle| {
            label
                .to_ascii_lowercase()
                .contains(&needle.to_ascii_lowercase())
        })
    };
    let mut result = edges
        .iter()
        .filter(|edge| !is_tombstoned(&edge.metadata))
        .filter_map(|edge| {
            let document = edge.document.as_deref()?.parse::<Value>().ok()?;
            let relation = document.get("relation").and_then(Value::as_str);
            if query
                .relation
                .as_deref()
                .is_some_and(|expected| relation != Some(expected))
            {
                return None;
            }
            let document_id = edge
                .metadata
                .get("doc_id")
                .and_then(Value::as_str)
                .or_else(|| document.get("doc_id").and_then(Value::as_str));
            if query
                .doc_id
                .as_deref()
                .is_some_and(|expected| document_id != Some(expected))
            {
                return None;
            }
            let ids = |field: &str| {
                document
                    .get(field)
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter_map(Value::as_str)
            };
            let source_labels = ids("source_ids")
                .filter_map(|id| node_labels.get(id).map(String::as_str))
                .collect::<Vec<_>>();
            let target_labels = ids("target_ids")
                .filter_map(|id| node_labels.get(id).map(String::as_str))
                .collect::<Vec<_>>();
            if !source_labels
                .iter()
                .any(|label| contains(label, &query.src_label_contains))
                || !target_labels
                    .iter()
                    .any(|label| contains(label, &query.tgt_label_contains))
            {
                return None;
            }
            Some(edge.id.clone())
        })
        .collect::<Vec<_>>();
    result.sort();
    result
}

/// Find a bounded path through visible node and edge records.
///
/// The caller must provide records after scope and ACL filtering. Traversal
/// reuses the same adjacency kernel as `kg_neighbors`, so a Rust-native query
/// cannot bypass the authority boundary by walking raw projections.
pub fn graph_shortest_path_from_records(
    query: &GraphShortestPathQuery,
    nodes: &[GraphRecord],
    edges: &[GraphRecord],
) -> Vec<String> {
    if query.src_id == query.dst_id {
        return vec![query.src_id.clone()];
    }
    let mut queue =
        std::collections::VecDeque::from([(query.src_id.clone(), vec![query.src_id.clone()])]);
    let mut seen = BTreeSet::from([query.src_id.clone()]);
    while let Some((current, path)) = queue.pop_front() {
        if path.len().saturating_sub(1) >= query.max_depth {
            continue;
        }
        let neighbors = graph_neighbors_from_records_with_jump(
            &current,
            nodes,
            edges,
            query.doc_id.as_deref(),
            false,
        );
        for next in neighbors.nodes.into_iter().chain(neighbors.edges) {
            if !seen.insert(next.clone()) {
                continue;
            }
            let mut candidate = path.clone();
            candidate.push(next.clone());
            if next == query.dst_id {
                return candidate;
            }
            queue.push_back((next, candidate));
        }
    }
    Vec::new()
}

/// Expand a bounded frontier in deterministic layers.
///
/// The result mirrors Python `GraphQuery.k_hop`: each layer contains the
/// nodes and edges discovered from the prior frontier, while visited IDs are
/// not re-expanded. Records must already be scope- and ACL-filtered.
pub fn graph_k_hop_from_records(
    query: &GraphKHopQuery,
    nodes: &[GraphRecord],
    edges: &[GraphRecord],
) -> Vec<GraphNeighbors> {
    let mut visited = BTreeSet::new();
    let mut frontier = query.start_ids.iter().cloned().collect::<BTreeSet<_>>();
    let mut layers = Vec::with_capacity(query.k);
    for _ in 0..query.k {
        let mut next_frontier = BTreeSet::new();
        let mut layer = GraphNeighbors::default();
        for rid in &frontier {
            if !visited.insert(rid.clone()) {
                continue;
            }
            let neighbors = graph_neighbors_from_records_with_jump(
                rid,
                nodes,
                edges,
                query.doc_id.as_deref(),
                false,
            );
            layer.nodes.extend(neighbors.nodes.iter().cloned());
            layer.edges.extend(neighbors.edges.iter().cloned());
            next_frontier.extend(neighbors.nodes);
            next_frontier.extend(neighbors.edges);
        }
        next_frontier.retain(|id| !visited.contains(id));
        layers.push(layer);
        frontier = next_frontier;
    }
    layers
}

fn is_tombstoned(metadata: &Map<String, Value>) -> bool {
    metadata
        .get("lifecycle_status")
        .and_then(Value::as_str)
        .is_some_and(|status| matches!(status, "tombstoned" | "deleted"))
}

/// Scope carried by the graph projection.  The Python pgvector schema keeps
/// graph rows in shared tables, so these values are represented by exact JSON
/// metadata predicates rather than by a Rust-owned table or schema.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct GraphScope {
    pub namespace: String,
    pub workspace_id: Option<String>,
    pub graph_space: Option<String>,
}

/// One authoritative graph change plus its materialized vector projection.
/// `table` names an existing Python pgvector collection (normally
/// `gke_nodes` or `gke_edges`); implementations must not create a parallel
/// graph schema for this capability.
#[derive(Clone, Debug, PartialEq)]
pub struct GraphMutation {
    pub scope: GraphScope,
    pub table: String,
    pub entity_kind: String,
    pub event_id: String,
    pub op: String,
    pub payload: Value,
    pub record: GraphRecord,
    pub embedding_dim: usize,
}

/// Return the canonical projection table for a persisted graph entity kind.
///
/// `document` and `domain` share the graph event/projection transaction path,
/// but are not Node/Edge lifecycle entities.
pub fn graph_projection_table(entity_kind: &str) -> StoreResult<&'static str> {
    match entity_kind {
        "node" => Ok("gke_nodes"),
        "edge" => Ok("gke_edges"),
        "document" => Ok("gke_documents"),
        "domain" => Ok("gke_domains"),
        _ => Err(StoreError::InvalidEntityEventPayload {
            message: format!("unsupported graph entity kind {entity_kind:?}"),
        }),
    }
}

/// Validate a projection target using the shared entity-kind/table mapping.
pub fn validate_graph_projection_target(table: &str, entity_kind: &str) -> StoreResult<()> {
    let expected_table = graph_projection_table(entity_kind)?;
    if table != expected_table {
        return Err(StoreError::InvalidEntityEventPayload {
            message: format!("graph entity kind {entity_kind:?} must use table {expected_table:?}"),
        });
    }
    Ok(())
}

/// Validate a node/edge-only lifecycle target, including its table mapping.
pub fn validate_graph_lifecycle_target(table: &str, entity_kind: &str) -> StoreResult<()> {
    if !matches!(entity_kind, "node" | "edge") {
        return Err(StoreError::InvalidEntityEventPayload {
            message: format!("graph lifecycle operations do not support {entity_kind:?}"),
        });
    }
    validate_graph_projection_target(table, entity_kind)
}

/// A graph mutation batch plus optional indexing intents that must share its
/// commit boundary.  Indexing remains explicit: a store must never infer a
/// queue operation from arbitrary graph metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct GraphMutationBatch {
    pub mutations: Vec<GraphMutation>,
    pub index_jobs: Vec<NewIndexJob>,
}

/// Validate the internal mutation before a store can own its effects.
///
/// The versioned wire DTO performs the same checks at the API boundary, but
/// Rust-native callers can reach a store without going through that adapter.
/// Keeping this guard at the effect owner prevents a direct caller from
/// bypassing the shared graph-mutation contract.
pub fn validate_graph_mutation(mutation: &GraphMutation) -> StoreResult<()> {
    if mutation.scope.namespace.is_empty() {
        return Err(StoreError::EmptyNamespace);
    }
    if mutation.event_id.is_empty() {
        return Err(StoreError::InvalidIdentifier {
            identifier: "graph mutation event_id",
        });
    }
    if mutation.record.id.is_empty() {
        return Err(StoreError::EmptyRecordId);
    }
    // Documents and domains share this internal transaction path without
    // being admitted to the public Node/Edge graph mutation authority.
    validate_graph_projection_target(&mutation.table, &mutation.entity_kind)?;
    if !matches!(mutation.op.as_str(), "ADD" | "REPLACE" | "TOMBSTONE") {
        return Err(StoreError::UnsupportedEntityEventOperation {
            op: mutation.op.clone(),
        });
    }
    if !mutation.payload.is_object() {
        return Err(StoreError::InvalidEntityEventPayload {
            message: "graph mutation payload must be a JSON object".to_owned(),
        });
    }
    if matches!(mutation.entity_kind.as_str(), "node" | "edge")
        && mutation.payload.get("id").and_then(Value::as_str) != Some(mutation.record.id.as_str())
    {
        return Err(StoreError::InvalidEntityEventPayload {
            message: "graph mutation payload id must match the projection record id".to_owned(),
        });
    }
    if let Some(embedding) = &mutation.record.embedding {
        if embedding.iter().any(|value| !value.is_finite()) {
            return Err(StoreError::NonFiniteVector);
        }
        if embedding.len() != mutation.embedding_dim {
            return Err(StoreError::VectorDimensionMismatch {
                expected: mutation.embedding_dim,
                actual: embedding.len(),
            });
        }
    }
    Ok(())
}

/// Validate a store-owned atomic batch before opening its transaction.
pub fn validate_graph_mutation_batch(batch: &GraphMutationBatch) -> StoreResult<()> {
    let first = batch
        .mutations
        .first()
        .ok_or(StoreError::InvalidEntityEventPayload {
            message: "graph mutation batches must not be empty".to_owned(),
        })?;
    for mutation in &batch.mutations {
        validate_graph_mutation(mutation)?;
        if mutation.scope != first.scope {
            return Err(StoreError::InvalidEntityEventPayload {
                message: "all atomic graph mutations must share one graph scope".to_owned(),
            });
        }
    }
    for job in &batch.index_jobs {
        if job.job_id.is_empty()
            || job.namespace.is_empty()
            || job.entity_kind.is_empty()
            || job.entity_id.is_empty()
            || job.index_kind.is_empty()
        {
            return Err(StoreError::InvalidIdentifier {
                identifier: "index job field",
            });
        }
        if job.namespace != first.scope.namespace {
            return Err(StoreError::InvalidEntityEventPayload {
                message: "index job namespace must match graph mutation namespace".to_owned(),
            });
        }
        if !matches!(job.op.as_str(), "UPSERT" | "DELETE") {
            return Err(StoreError::InvalidEntityEventPayload {
                message: "index job op must be UPSERT or DELETE".to_owned(),
            });
        }
        if job.max_retries < 0 {
            return Err(StoreError::InvalidEntityEventPayload {
                message: "index job max_retries must not be negative".to_owned(),
            });
        }
    }
    Ok(())
}

/// Validate the optional typed model that accompanies a graph mutation.
///
/// The legacy mutation payload remains unchanged.  This additive seam lets a
/// Rust-native caller opt into the same model contract as the Python writer
/// without forcing old event/replay callers to construct an envelope.
pub fn validate_typed_graph_mutation(
    mutation: &GraphMutation,
    envelope: &GraphModelEnvelope,
) -> StoreResult<()> {
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
                message: format!(
                    "typed graph mutation has unsupported entity kind {:?}",
                    mutation.entity_kind
                ),
            });
        }
    };
    if envelope.model_class.entity_kind() != expected_kind {
        return Err(StoreError::InvalidEntityEventPayload {
            message: format!(
                "typed graph mutation entity kind {:?} does not match model class {:?}",
                expected_kind, envelope.model_class
            ),
        });
    }
    if mutation.payload != envelope.payload {
        return Err(StoreError::InvalidEntityEventPayload {
            message: "typed graph mutation payload differs from its model envelope".to_owned(),
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
            message: "typed graph mutation record id differs from its model envelope".to_owned(),
        });
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq)]
pub struct AppliedGraphMutation {
    pub event: EntityEvent,
    pub inserted: bool,
    /// False for an idempotent event retry.  The pre-existing projection is
    /// deliberately left untouched in that case.
    pub mutated: bool,
}

/// Declared graph-authority level for a store implementation.
///
/// This is intentionally explicit and ordered by authority. A caller must
/// not infer graph support from an unrelated metadata or event API; the
/// default remains unsupported so older backends fail closed.
#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
pub enum GraphAuthorityCapability {
    #[default]
    Unsupported,
    SingleMutation,
    AtomicBatchWithIndexJobs,
}

impl GraphAuthorityCapability {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Unsupported => "unsupported",
            Self::SingleMutation => "single_mutation",
            Self::AtomicBatchWithIndexJobs => "atomic_batch_with_index_jobs",
        }
    }

    pub const fn supports_single_mutation(self) -> bool {
        !matches!(self, Self::Unsupported)
    }

    pub const fn supports_atomic_index_jobs(self) -> bool {
        matches!(self, Self::AtomicBatchWithIndexJobs)
    }
}

/// Read request for the materialized Python pgvector graph tables.  Metadata
/// predicates retain exact JSON equality semantics; scope predicates are
/// applied in addition to these user-visible filters.
#[derive(Clone, Debug, PartialEq)]
pub struct GraphProjectionRead {
    pub scope: GraphScope,
    pub table: String,
    pub ids: Option<Vec<String>>,
    pub metadata: MetadataFilter,
    pub limit: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GraphProjectionVectorQuery {
    pub scope: GraphScope,
    pub table: String,
    pub query: VectorQuery,
    pub embedding_dim: usize,
}

/// Narrow authoritative graph capability.  It intentionally combines event
/// append and projection materialization so implementations with a real UoW
/// can make both durable together.  The existing `GraphWriteStore` remains a
/// lightweight, non-evented Phase-2 capability.
pub trait GraphMutationStore: Send + Sync {
    /// Declare the strongest graph authority this store can actually provide.
    /// The default is unsupported for backward-compatible, fail-closed
    /// adoption by existing implementations.
    fn graph_authority_capability(&self) -> GraphAuthorityCapability {
        GraphAuthorityCapability::Unsupported
    }
    fn apply_graph_mutation(
        &self,
        mutation: GraphMutation,
    ) -> impl Future<Output = StoreResult<AppliedGraphMutation>> + Send;
    /// Apply a group under one authoritative transaction.
    ///
    /// The default is deliberately fail-closed.  Implementations must opt in
    /// only when they can preserve one commit boundary for the whole group;
    /// callers must not silently degrade an atomic request to per-item writes.
    fn apply_graph_mutations_atomic(
        &self,
        _mutations: Vec<GraphMutation>,
    ) -> impl Future<Output = StoreResult<Vec<AppliedGraphMutation>>> + Send {
        async {
            Err(StoreError::Backend {
                backend: "graph-mutation".to_owned(),
                message: "atomic graph mutation batches are not supported by this store".to_owned(),
            })
        }
    }
    /// Apply graph mutations and their explicit index intents atomically.
    /// The default is fail-closed because a queue write outside the graph
    /// transaction would create a false durability guarantee.
    fn apply_graph_mutations_with_index_jobs_atomic(
        &self,
        _batch: GraphMutationBatch,
    ) -> impl Future<Output = StoreResult<Vec<AppliedGraphMutation>>> + Send {
        async {
            Err(StoreError::Backend {
                backend: "graph-mutation".to_owned(),
                message: "atomic graph mutation plus index jobs are not supported by this store"
                    .to_owned(),
            })
        }
    }
    fn apply_typed_graph_mutation(
        &self,
        mutation: GraphMutation,
        envelope: GraphModelEnvelope,
    ) -> impl Future<Output = StoreResult<AppliedGraphMutation>> + Send {
        async move {
            if !self.graph_authority_capability().supports_single_mutation() {
                return Err(StoreError::Backend {
                    backend: "graph-mutation".to_owned(),
                    message: "typed graph mutations require declared graph authority".to_owned(),
                });
            }
            validate_typed_graph_mutation(&mutation, &envelope)?;
            self.apply_graph_mutation(mutation).await
        }
    }
    fn graph_projection_records(
        &self,
        read: GraphProjectionRead,
    ) -> impl Future<Output = StoreResult<Vec<GraphRecord>>> + Send;
    fn graph_projection_vector_query(
        &self,
        query: GraphProjectionVectorQuery,
    ) -> impl Future<Output = StoreResult<Vec<VectorMatch>>> + Send;
}

/// Declared document-projection authority for a store implementation.
///
/// This is intentionally explicit and fail-closed. A store must not advertise
/// document parity merely because it can persist events or named projections.
#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
pub enum DocumentAuthorityCapability {
    #[default]
    Unsupported,
    ProjectionUpsert,
}

impl DocumentAuthorityCapability {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Unsupported => "unsupported",
            Self::ProjectionUpsert => "projection_upsert",
        }
    }

    pub const fn supports_projection_upsert(self) -> bool {
        matches!(self, Self::ProjectionUpsert)
    }
}

/// Backend-owned document projection capability.
///
/// The default is fail-closed so stores without Python-compatible document
/// semantics cannot accidentally advertise the public document route.
pub trait DocumentStore: Send + Sync {
    fn upsert_document_projection(
        &self,
        _write: DocumentProjectionUpsert,
    ) -> impl Future<Output = StoreResult<()>> + Send {
        async {
            Err(StoreError::Backend {
                backend: "document".to_owned(),
                message: "document projection is not supported by this store".to_owned(),
            })
        }
    }

    fn document_authority_capability(&self) -> DocumentAuthorityCapability {
        DocumentAuthorityCapability::Unsupported
    }
}

impl GraphRecord {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            document: None,
            metadata: Map::new(),
            embedding: None,
        }
    }
}

/// Exact JSON equality predicates, combined with AND; missing keys do not match.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MetadataFilter {
    pub equals: BTreeMap<String, Value>,
}

impl MetadataFilter {
    pub fn matches(&self, metadata: &Map<String, Value>) -> bool {
        self.equals
            .iter()
            .all(|(key, expected)| metadata.get(key) == Some(expected))
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DistanceMetric {
    #[default]
    Cosine,
    L2,
    /// pgvector `<#>` ordering: negative inner product, ascending.
    InnerProduct,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VectorQuery {
    pub embedding: Vec<f32>,
    pub limit: usize,
    pub metadata: MetadataFilter,
    pub metric: DistanceMetric,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VectorMatch {
    pub record: GraphRecord,
    pub distance: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NewEntityEvent {
    pub event_id: String,
    pub entity_kind: String,
    pub entity_id: String,
    pub op: String,
    pub payload: Value,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AppendedEvent {
    pub event: EntityEvent,
    pub inserted: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReplayCursor {
    pub namespace: String,
    pub consumer: String,
    pub last_seq: i64,
}

/// Inputs shared by the bounded authoritative-event recovery capability.
/// `projection_namespace` deliberately remains independent from the event
/// namespace: consumers may materialize multiple isolated views of one log.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EntityRecoveryRequest {
    pub namespace: String,
    pub consumer: String,
    pub projection_namespace: String,
    pub projection_key: String,
    pub batch_limit: usize,
    /// Test-only transaction fault injection. Production callers leave false.
    pub abort_after_projection: bool,
}

/// Inputs for an explicit complete replay. This is never a startup action.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EntityRebuildRequest {
    pub namespace: String,
    pub consumer: String,
    pub projection_namespace: String,
    pub projection_key: String,
    /// Test-only transaction fault injection. Production callers leave false.
    pub abort_after_projection: bool,
}

/// Durable result of a bounded recovery or full rebuild.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EntityRecoveryReport {
    pub processed_count: usize,
    pub prior_cursor: i64,
    pub new_cursor: i64,
    pub latest_authoritative_seq: i64,
    pub caught_up: bool,
    /// Canonical, ASCII-safe JSON for byte-stable cross-language comparison.
    pub canonical_payload: String,
    pub digest: String,
}

pub fn validate_entity_recovery_request(request: &EntityRecoveryRequest) -> StoreResult<()> {
    validate_recovery_identifiers(
        &request.namespace,
        &request.consumer,
        &request.projection_namespace,
        &request.projection_key,
    )?;
    if request.batch_limit == 0 {
        return Err(StoreError::InvalidRecoveryBatchLimit);
    }
    Ok(())
}

pub fn validate_entity_rebuild_request(request: &EntityRebuildRequest) -> StoreResult<()> {
    validate_recovery_identifiers(
        &request.namespace,
        &request.consumer,
        &request.projection_namespace,
        &request.projection_key,
    )
}

fn validate_recovery_identifiers(
    namespace: &str,
    consumer: &str,
    projection_namespace: &str,
    projection_key: &str,
) -> StoreResult<()> {
    if namespace.is_empty() {
        return Err(StoreError::EmptyNamespace);
    }
    if namespace.contains('\0') {
        return Err(StoreError::InvalidIdentifier {
            identifier: "namespace",
        });
    }
    for (value, identifier) in [
        (consumer, "consumer"),
        (projection_namespace, "projection namespace"),
        (projection_key, "projection key"),
    ] {
        if value.is_empty() || value.contains('\0') {
            return Err(StoreError::InvalidIdentifier { identifier });
        }
    }
    Ok(())
}

/// Named-projection row, shaped exactly like Python meta-store read results.
#[derive(Clone, Debug, PartialEq)]
pub struct NamedProjection {
    pub namespace: String,
    pub key: String,
    pub payload: Map<String, Value>,
    pub last_authoritative_seq: i64,
    pub last_materialized_seq: i64,
    pub projection_schema_version: i64,
    pub materialization_status: String,
    pub updated_at_ms: i64,
}

/// Input shared by named-projection replace and compare-and-swap writes.
/// `updated_at_ms` is intentionally store-owned, matching Python's meta stores.
#[derive(Clone, Debug, PartialEq)]
pub struct NamedProjectionWrite {
    pub payload: Map<String, Value>,
    pub last_authoritative_seq: i64,
    pub last_materialized_seq: i64,
    pub projection_schema_version: i64,
    pub materialization_status: String,
}

/// Raw workflow-design checkpoint.  JSON text is opaque: callers that need
/// byte-for-byte Python compatibility must not parse or canonicalize it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkflowDesignSnapshot {
    pub workflow_id: String,
    pub version: i64,
    pub seq: i64,
    pub payload_json: String,
    pub schema_version: i64,
    pub created_at_ms: i64,
}

/// Input for a workflow-design checkpoint. `created_at_ms` remains store-owned.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkflowDesignSnapshotWrite {
    pub version: i64,
    pub seq: i64,
    pub payload_json: String,
    pub schema_version: i64,
}

/// Raw reversible workflow-design version transition.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkflowDesignDelta {
    pub workflow_id: String,
    pub version: i64,
    pub prev_version: i64,
    pub target_seq: i64,
    pub forward_json: String,
    pub inverse_json: String,
    pub schema_version: i64,
    pub created_at_ms: i64,
}

/// Input for a reversible transition. Both JSON fields stay opaque raw text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkflowDesignDeltaWrite {
    pub version: i64,
    pub prev_version: i64,
    pub target_seq: i64,
    pub forward_json: String,
    pub inverse_json: String,
    pub schema_version: i64,
}

/// Server-run row. JSON columns deliberately remain raw text in storage APIs;
/// Python-facing adapters decode them only at their ABI boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ServerRun {
    pub run_id: String,
    pub conversation_id: String,
    pub workflow_id: String,
    pub user_id: Option<String>,
    pub user_turn_node_id: Option<String>,
    pub assistant_turn_node_id: Option<String>,
    pub status: String,
    pub cancel_requested: bool,
    pub result_json: Option<String>,
    pub error_json: Option<String>,
    pub created_at_ms: i64,
    pub updated_at_ms: i64,
    pub started_at_ms: Option<i64>,
    pub finished_at_ms: Option<i64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ServerRunCreate {
    pub run_id: String,
    pub conversation_id: String,
    pub workflow_id: String,
    pub user_id: Option<String>,
    pub user_turn_node_id: String,
    pub status: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ServerRunUpdate {
    pub status: String,
    pub assistant_turn_node_id: Option<String>,
    pub result_json: Option<String>,
    pub error_json: Option<String>,
    pub started_at_ms: Option<i64>,
    pub finished_at_ms: Option<i64>,
    /// `None` means preserve current value, exactly as Python COALESCE does.
    pub cancel_requested: Option<bool>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ServerRunEvent {
    pub seq: i64,
    pub run_id: String,
    pub event_type: String,
    pub payload_json: String,
    pub created_at_ms: i64,
}

/// A completed recorded-only runtime transition.  This is intentionally a
/// narrow transaction capability rather than a scheduler trait: the caller has
/// already obtained any provider/tool result and owns no callbacks here.
#[derive(Clone, Debug, PartialEq)]
pub struct RecordedRuntimeTransitionWrite {
    pub run_id: String,
    pub workflow_id: String,
    pub conversation_id: String,
    pub transition_id: String,
    pub expected_event_seq: i64,
    pub event_type: String,
    /// Canonical transition request digest, used for exact retry detection.
    pub request_digest: String,
    /// Canonical persisted transition result/event payload.
    pub event_payload_json: String,
    pub server_status: String,
    pub result_json: Option<String>,
    pub error_json: Option<String>,
    pub checkpoint_namespace: String,
    pub checkpoint_key: String,
    pub checkpoint_projection: NamedProjectionWrite,
    pub status_namespace: String,
    pub status_key: String,
    pub status_projection: NamedProjectionWrite,
    /// Test-only rollback injection after all durable writes have been issued.
    pub abort_after_writes: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RecordedRuntimeTransitionResult {
    pub event_seq: i64,
    pub event_payload_json: String,
    pub idempotent: bool,
}

/// Durable derived-index queue row.  Timestamp values deliberately remain JSON
/// values: Python's SQLite queue exposes epoch seconds while PostgreSQL exposes
/// textual `TIMESTAMPTZ` values.  Normalising them here would break the facade
/// contract during the rollback window.
#[derive(Clone, Debug, PartialEq)]
pub struct IndexJob {
    pub job_id: String,
    pub namespace: String,
    pub entity_kind: String,
    pub entity_id: String,
    pub index_kind: String,
    pub coalesce_key: String,
    pub op: String,
    pub status: String,
    pub lease_until: Option<Value>,
    pub next_run_at: Option<Value>,
    pub max_retries: i64,
    pub retry_count: i64,
    pub last_error: Option<String>,
    pub payload_json: Option<String>,
    pub created_at: Value,
    pub updated_at: Value,
    pub claim_token: Option<String>,
    pub claim_attempts: i64,
    pub accepted_result_json: Option<String>,
    pub accepted_result_sha256: Option<String>,
    pub accepted_at: Option<Value>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AcceptedIndexJobResult {
    pub status: String,
    pub result_json: Option<String>,
    pub result_sha256: Option<String>,
    pub accepted_at: Option<Value>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NewIndexJob {
    pub job_id: String,
    pub namespace: String,
    pub entity_kind: String,
    pub entity_id: String,
    pub index_kind: String,
    pub op: String,
    pub payload_json: Option<String>,
    pub max_retries: i64,
}

/// Durable lane-message projection.  Lease values deliberately remain JSON
/// values: SQLite exposes epoch seconds; PostgreSQL exposes TIMESTAMPTZ text.
#[derive(Clone, Debug, PartialEq)]
pub struct ProjectedLaneMessage {
    pub message_id: String,
    pub namespace: String,
    pub purpose: String,
    pub inbox_id: String,
    pub conversation_id: String,
    pub recipient_id: String,
    pub sender_id: String,
    pub msg_type: String,
    pub status: String,
    pub seq: i64,
    pub conversation_seq: i64,
    pub claimed_by: Option<String>,
    pub lease_until: Option<Value>,
    pub retry_count: i64,
    pub created_at: i64,
    pub available_at: i64,
    pub run_id: Option<String>,
    pub step_id: Option<String>,
    pub correlation_id: Option<String>,
    pub payload_json: Option<String>,
    pub error_json: Option<String>,
    pub prev_message_id: Option<String>,
    pub next_message_id: Option<String>,
    pub inbox_tail_message_id: Option<String>,
    pub conversation_tail_message_id: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NewProjectedLaneMessage {
    pub message_id: String,
    pub namespace: String,
    pub purpose: String,
    pub inbox_id: String,
    pub conversation_id: String,
    pub recipient_id: String,
    pub sender_id: String,
    pub msg_type: String,
    pub status: String,
    pub created_at: i64,
    pub available_at: i64,
    pub run_id: Option<String>,
    pub step_id: Option<String>,
    pub correlation_id: Option<String>,
    pub payload_json: Option<String>,
    pub error_json: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LaneMessageFilter {
    pub namespace: Option<String>,
    pub purpose: Option<String>,
    pub inbox_id: Option<String>,
    pub conversation_id: Option<String>,
    pub status: Option<String>,
    pub msg_type: Option<String>,
    pub sender_id: Option<String>,
    pub recipient_id: Option<String>,
    pub correlation_id: Option<String>,
    /// Deliberately accepted but ignored: Python's baseline schema has no
    /// reply_to_message_id column.
    pub reply_to_message_id: Option<String>,
    pub created_at_gte: Option<i64>,
    pub created_at_lte: Option<i64>,
    pub available_at_gte: Option<i64>,
    pub available_at_lte: Option<i64>,
    pub limit: usize,
    pub newest_first: bool,
}

impl Default for LaneMessageFilter {
    fn default() -> Self {
        Self {
            namespace: None,
            purpose: None,
            inbox_id: None,
            conversation_id: None,
            status: None,
            msg_type: None,
            sender_id: None,
            recipient_id: None,
            correlation_id: None,
            reply_to_message_id: None,
            created_at_gte: None,
            created_at_lte: None,
            available_at_gte: None,
            available_at_lte: None,
            limit: 1000,
            newest_first: false,
        }
    }
}

impl LaneMessageFilter {
    pub fn namespace(namespace: impl Into<String>) -> Self {
        Self {
            namespace: Some(namespace.into()),
            limit: 1000,
            ..Self::default()
        }
    }
}

/// Atomic claim constraints for active-run control and other lane consumers.
///
/// Filtering belongs inside the claim transaction.  A caller must not list a
/// candidate and then claim it with a broader filter, or another consumer may
/// win the race and an unrelated message may be claimed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LaneMessageClaimFilter {
    pub namespace: String,
    pub inbox_id: String,
    pub claimed_by: String,
    pub message_ids: Option<Vec<String>>,
    pub run_id: Option<String>,
    pub msg_type: Option<String>,
    pub recipient_id: Option<String>,
    pub limit: usize,
    pub lease_seconds: i64,
}

pub trait LaneMessageReadStore: Send + Sync {
    fn projected_lane_message(
        &self,
        message_id: &str,
    ) -> impl Future<Output = StoreResult<Option<ProjectedLaneMessage>>> + Send;
    fn projected_lane_messages(
        &self,
        filter: LaneMessageFilter,
    ) -> impl Future<Output = StoreResult<Vec<ProjectedLaneMessage>>> + Send;
}

/// Durable projected lane queue.  `claimed_by` is ownership, not an advisory
/// token: stale workers must be harmless after an expired lease is reclaimed.
pub trait LaneMessageWriteStore: LaneMessageReadStore {
    fn project_lane_message(
        &self,
        row: NewProjectedLaneMessage,
    ) -> impl Future<Output = StoreResult<()>> + Send;
    fn update_projected_lane_message_status(
        &self,
        message_id: &str,
        status: &str,
        error_json: Option<String>,
    ) -> impl Future<Output = StoreResult<()>> + Send;
    fn update_projected_lane_message_links(
        &self,
        message_id: &str,
        prev_message_id: Option<String>,
        next_message_id: Option<String>,
        inbox_tail_message_id: Option<String>,
        conversation_tail_message_id: Option<String>,
    ) -> impl Future<Output = StoreResult<()>> + Send;
    fn clear_projected_lane_messages(
        &self,
        namespace: &str,
    ) -> impl Future<Output = StoreResult<u64>> + Send;
    fn claim_projected_lane_messages(
        &self,
        namespace: &str,
        inbox_id: &str,
        claimed_by: &str,
        limit: usize,
        lease_seconds: i64,
    ) -> impl Future<Output = StoreResult<Vec<ProjectedLaneMessage>>> + Send;
    fn claim_projected_lane_messages_filtered(
        &self,
        filter: LaneMessageClaimFilter,
    ) -> impl Future<Output = StoreResult<Vec<ProjectedLaneMessage>>> + Send;
    fn ack_projected_lane_message(
        &self,
        message_id: &str,
        claimed_by: &str,
    ) -> impl Future<Output = StoreResult<()>> + Send;
    fn requeue_projected_lane_message(
        &self,
        message_id: &str,
        claimed_by: &str,
        error_json: Option<String>,
        delay_seconds: i64,
    ) -> impl Future<Output = StoreResult<()>> + Send;
    fn dead_letter_projected_lane_message(
        &self,
        message_id: &str,
        claimed_by: &str,
        error_json: Option<String>,
    ) -> impl Future<Output = StoreResult<()>> + Send;
}

impl NewIndexJob {
    pub fn coalesce_key(&self) -> String {
        format!(
            "{}:{}:{}",
            self.entity_kind, self.entity_id, self.index_kind
        )
    }
}

/// Queue reads stay separate from graph/event reads so callers can depend on
/// this capability without gaining queue mutation authority.
pub trait IndexJobReadStore: Send + Sync {
    fn index_jobs(
        &self,
        namespace: Option<&str>,
        status: Option<&str>,
        entity_kind: Option<&str>,
        entity_id: Option<&str>,
        index_kind: Option<&str>,
        limit: usize,
    ) -> impl Future<Output = StoreResult<Vec<IndexJob>>> + Send;
}

/// Durable index-job state machine.  `None` token intentionally retains the
/// legacy Python administrative behaviour; supplied tokens protect stale
/// workers after a lease reclaim.
pub trait IndexJobWriteStore: IndexJobReadStore {
    fn enqueue_index_job(
        &self,
        job: NewIndexJob,
    ) -> impl Future<Output = StoreResult<String>> + Send;
    fn claim_index_jobs(
        &self,
        limit: usize,
        lease_seconds: i64,
        namespace: Option<&str>,
    ) -> impl Future<Output = StoreResult<Vec<IndexJob>>> + Send;
    fn mark_index_job_done(
        &self,
        job_id: &str,
        claim_token: Option<&str>,
    ) -> impl Future<Output = StoreResult<bool>> + Send;
    fn accept_index_job_result(
        &self,
        job_id: &str,
        claim_token: &str,
        result_json: &str,
        result_sha256: &str,
    ) -> impl Future<Output = StoreResult<AcceptedIndexJobResult>> + Send;
    fn index_job_result(
        &self,
        job_id: &str,
    ) -> impl Future<Output = StoreResult<Option<AcceptedIndexJobResult>>> + Send;
    fn mark_index_job_failed(
        &self,
        job_id: &str,
        error: &str,
        final_: bool,
        claim_token: Option<&str>,
    ) -> impl Future<Output = StoreResult<()>> + Send;
    fn bump_retry_and_requeue(
        &self,
        job_id: &str,
        error: &str,
        next_run_at_seconds: i64,
        claim_token: Option<&str>,
    ) -> impl Future<Output = StoreResult<()>> + Send;
    fn renew_index_job_lease(
        &self,
        job_id: &str,
        claim_token: &str,
        lease_seconds: i64,
    ) -> impl Future<Output = StoreResult<bool>> + Send;
    fn requeue_index_job_at_tail(
        &self,
        job_id: &str,
        payload_json: String,
        delay_seconds: i64,
        claim_token: Option<&str>,
    ) -> impl Future<Output = StoreResult<()>> + Send;
}

/// Bounded run-registry capability. It intentionally has no transition/CAS
/// policy: Python backends accept every status value and missing updates noop.
pub trait ServerRunReadStore: Send + Sync {
    fn server_run(
        &self,
        run_id: &str,
    ) -> impl Future<Output = StoreResult<Option<ServerRun>>> + Send;
    fn server_runs(
        &self,
        status: Option<&str>,
        workflow_id: Option<&str>,
        conversation_id: Option<&str>,
        limit: usize,
    ) -> impl Future<Output = StoreResult<Vec<ServerRun>>> + Send;
    fn server_run_events(
        &self,
        run_id: &str,
        after_seq: i64,
        limit: usize,
    ) -> impl Future<Output = StoreResult<Vec<ServerRunEvent>>> + Send;
}

pub trait ServerRunWriteStore: ServerRunReadStore {
    fn create_server_run(
        &self,
        run: ServerRunCreate,
    ) -> impl Future<Output = StoreResult<()>> + Send;
    fn append_server_run_event(
        &self,
        run_id: &str,
        event_type: &str,
        payload_json: String,
    ) -> impl Future<Output = StoreResult<ServerRunEvent>> + Send;
    fn update_server_run(
        &self,
        run_id: &str,
        update: ServerRunUpdate,
    ) -> impl Future<Output = StoreResult<()>> + Send;
    fn request_server_run_cancel(
        &self,
        run_id: &str,
    ) -> impl Future<Output = StoreResult<()>> + Send;
}

pub trait GraphReadStore: Send + Sync {
    fn graph_record(
        &self,
        namespace: &str,
        id: &str,
    ) -> impl Future<Output = StoreResult<Option<GraphRecord>>> + Send;
    /// Namespace records in implementation-defined stable read order.
    ///
    /// The Phase-2 in-memory implementation preserves first insertion order:
    /// upserting an existing id keeps its position, while delete then reinsert
    /// appends it. Consumers must not infer bytewise-id ordering from this API.
    fn graph_records(
        &self,
        namespace: &str,
        metadata: &MetadataFilter,
    ) -> impl Future<Output = StoreResult<Vec<GraphRecord>>> + Send;
    /// Metadata matches in implementation-defined stable rank order.
    ///
    /// The Phase-2 in-memory implementation's cosine mode follows the public
    /// Python memory backend: ascending distance with insertion-order ties;
    /// absent, dimension-mismatched, and zero-norm candidate vectors rank at
    /// distance `2.0`. L2 retains its strict vector contract.
    fn vector_query(
        &self,
        namespace: &str,
        query: &VectorQuery,
    ) -> impl Future<Output = StoreResult<Vec<VectorMatch>>> + Send;
}

pub trait GraphWriteStore: GraphReadStore {
    fn upsert_graph_record(
        &self,
        namespace: &str,
        record: GraphRecord,
    ) -> impl Future<Output = StoreResult<()>> + Send;
    fn delete_graph_record(
        &self,
        namespace: &str,
        id: &str,
    ) -> impl Future<Output = StoreResult<bool>> + Send;
}

pub trait EventReadStore: Send + Sync {
    /// Events strictly after `after_seq`, ascending namespace-local sequence.
    fn replay_events(
        &self,
        namespace: &str,
        after_seq: i64,
        limit: usize,
    ) -> impl Future<Output = StoreResult<Vec<EntityEvent>>> + Send;
    fn replay_cursor(
        &self,
        namespace: &str,
        consumer: &str,
    ) -> impl Future<Output = StoreResult<ReplayCursor>> + Send;
    fn latest_event_seq(&self, namespace: &str) -> impl Future<Output = StoreResult<i64>> + Send;
}

pub trait EventWriteStore: EventReadStore {
    fn append_entity_event(
        &self,
        namespace: &str,
        event: NewEntityEvent,
    ) -> impl Future<Output = StoreResult<AppendedEvent>> + Send;
    fn advance_replay_cursor(
        &self,
        namespace: &str,
        consumer: &str,
        last_seq: i64,
    ) -> impl Future<Output = StoreResult<ReplayCursor>> + Send;
}

/// Explicit destructive event-history capability. Kept separate from ordinary
/// append/cursor writes so consumers opt in to branch pruning deliberately.
pub trait EventPruneStore: EventReadStore {
    /// Delete only rows whose namespace-local sequence is strictly greater
    /// than `to_seq`, returning the exact affected-row count.
    fn prune_entity_events_after(
        &self,
        namespace: &str,
        to_seq: i64,
    ) -> impl Future<Output = StoreResult<u64>> + Send;
}

pub trait ProjectionReadStore: Send + Sync {
    fn named_projection(
        &self,
        namespace: &str,
        key: &str,
    ) -> impl Future<Output = StoreResult<Option<NamedProjection>>> + Send;
    /// Namespace rows, ascending bytewise key, matching Python meta-store list.
    fn named_projections(
        &self,
        namespace: &str,
    ) -> impl Future<Output = StoreResult<Vec<NamedProjection>>> + Send;
}

/// Mutation capability for generic named projections.  Read consumers keep
/// depending on [`ProjectionReadStore`], so shadow inspection stays read-only.
pub trait ProjectionWriteStore: ProjectionReadStore {
    fn replace_named_projection(
        &self,
        namespace: &str,
        key: &str,
        projection: NamedProjectionWrite,
    ) -> impl Future<Output = StoreResult<()>> + Send;

    /// Creates only when both expected sequences are `None`; otherwise updates
    /// only when both persisted sequence values match exactly.
    fn compare_and_swap_named_projection(
        &self,
        namespace: &str,
        key: &str,
        expected_last_authoritative_seq: Option<i64>,
        expected_last_materialized_seq: Option<i64>,
        projection: NamedProjectionWrite,
    ) -> impl Future<Output = StoreResult<bool>> + Send;

    fn clear_named_projection(
        &self,
        namespace: &str,
        key: &str,
    ) -> impl Future<Output = StoreResult<()>> + Send;

    fn clear_projection_namespace(
        &self,
        namespace: &str,
    ) -> impl Future<Output = StoreResult<()>> + Send;
}

/// Read capability for workflow-design checkpoints and reversible deltas.
pub trait WorkflowDesignHistoryReadStore: Send + Sync {
    fn workflow_design_snapshot(
        &self,
        workflow_id: &str,
        max_version: i64,
        schema_version: i64,
    ) -> impl Future<Output = StoreResult<Option<WorkflowDesignSnapshot>>> + Send;

    fn workflow_design_delta(
        &self,
        workflow_id: &str,
        version: i64,
        schema_version: i64,
    ) -> impl Future<Output = StoreResult<Option<WorkflowDesignDelta>>> + Send;
}

/// Mutation capability for workflow-design history. The concrete durable
/// stores also expose these operations on their UoWs for atomic branches.
pub trait WorkflowDesignHistoryWriteStore: WorkflowDesignHistoryReadStore {
    fn put_workflow_design_snapshot(
        &self,
        workflow_id: &str,
        snapshot: WorkflowDesignSnapshotWrite,
    ) -> impl Future<Output = StoreResult<()>> + Send;

    fn clear_workflow_design_snapshots(
        &self,
        workflow_id: &str,
    ) -> impl Future<Output = StoreResult<()>> + Send;

    fn put_workflow_design_delta(
        &self,
        workflow_id: &str,
        delta: WorkflowDesignDeltaWrite,
    ) -> impl Future<Output = StoreResult<()>> + Send;

    fn clear_workflow_design_deltas(
        &self,
        workflow_id: &str,
    ) -> impl Future<Output = StoreResult<()>> + Send;
}

/// Capability-scoped shadow façade. No mutation trait is implemented.
#[derive(Clone, Debug)]
pub struct ShadowInspection<S> {
    inner: S,
}
impl<S> ShadowInspection<S> {
    fn new(inner: S) -> Self {
        Self { inner }
    }
}

pub trait ShadowInspectable:
    Clone + GraphReadStore + EventReadStore + ProjectionReadStore + Sized
{
    fn shadow_inspection(&self) -> ShadowInspection<Self> {
        ShadowInspection::new(self.clone())
    }
}
impl<T> ShadowInspectable for T where
    T: Clone + GraphReadStore + EventReadStore + ProjectionReadStore
{
}

impl<S: GraphReadStore + Send + Sync> GraphReadStore for ShadowInspection<S> {
    async fn graph_record(&self, ns: &str, id: &str) -> StoreResult<Option<GraphRecord>> {
        self.inner.graph_record(ns, id).await
    }
    async fn graph_records(
        &self,
        ns: &str,
        filter: &MetadataFilter,
    ) -> StoreResult<Vec<GraphRecord>> {
        self.inner.graph_records(ns, filter).await
    }
    async fn vector_query(&self, ns: &str, query: &VectorQuery) -> StoreResult<Vec<VectorMatch>> {
        self.inner.vector_query(ns, query).await
    }
}

impl<S: EventReadStore + Send + Sync> EventReadStore for ShadowInspection<S> {
    async fn replay_events(
        &self,
        ns: &str,
        after: i64,
        limit: usize,
    ) -> StoreResult<Vec<EntityEvent>> {
        self.inner.replay_events(ns, after, limit).await
    }
    async fn replay_cursor(&self, ns: &str, consumer: &str) -> StoreResult<ReplayCursor> {
        self.inner.replay_cursor(ns, consumer).await
    }
    async fn latest_event_seq(&self, ns: &str) -> StoreResult<i64> {
        self.inner.latest_event_seq(ns).await
    }
}

impl<S: ProjectionReadStore + Send + Sync> ProjectionReadStore for ShadowInspection<S> {
    async fn named_projection(
        &self,
        namespace: &str,
        key: &str,
    ) -> StoreResult<Option<NamedProjection>> {
        self.inner.named_projection(namespace, key).await
    }
    async fn named_projections(&self, namespace: &str) -> StoreResult<Vec<NamedProjection>> {
        self.inner.named_projections(namespace).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edge_record(id: &str, document: Value) -> GraphRecord {
        GraphRecord {
            id: id.to_owned(),
            document: Some(document.to_string()),
            metadata: Map::new(),
            embedding: None,
        }
    }

    fn node_record(id: &str) -> GraphRecord {
        GraphRecord::new(id)
    }

    #[test]
    fn document_authority_capability_is_explicit_and_fail_closed() {
        assert_eq!(
            DocumentAuthorityCapability::default(),
            DocumentAuthorityCapability::Unsupported
        );
        assert_eq!(
            DocumentAuthorityCapability::Unsupported.as_str(),
            "unsupported"
        );
        assert!(!DocumentAuthorityCapability::Unsupported.supports_projection_upsert());
        assert!(DocumentAuthorityCapability::ProjectionUpsert.supports_projection_upsert());
    }

    #[test]
    fn graph_neighbors_match_node_and_edge_hypergraph_semantics() {
        let edges = vec![
            edge_record(
                "e0",
                serde_json::json!({"source_ids": [], "target_ids": []}),
            ),
            edge_record(
                "e1",
                serde_json::json!({
                    "source_ids": ["n1", "n2"],
                    "target_ids": ["n3"],
                    "source_edge_ids": ["e0"],
                    "target_edge_ids": ["e2"]
                }),
            ),
            edge_record(
                "e2",
                serde_json::json!({"source_ids": [], "target_ids": []}),
            ),
        ];
        let nodes = vec![node_record("n1"), node_record("n2"), node_record("n3")];

        assert_eq!(
            graph_neighbors_from_records("n1", &nodes, &edges, None),
            GraphNeighbors {
                nodes: BTreeSet::from(["n2".to_owned(), "n3".to_owned()]),
                edges: BTreeSet::from(["e1".to_owned()]),
            }
        );
        assert_eq!(
            graph_neighbors_from_records("e1", &nodes, &edges, None),
            GraphNeighbors {
                nodes: BTreeSet::from(["n1".to_owned(), "n2".to_owned(), "n3".to_owned()]),
                edges: BTreeSet::from(["e0".to_owned(), "e2".to_owned()]),
            }
        );
    }

    #[test]
    fn graph_neighbors_excludes_stale_and_other_document_edges() {
        let mut stale = edge_record(
            "stale",
            serde_json::json!({"source_ids": ["n1"], "target_ids": ["n2"]}),
        );
        stale.metadata.insert(
            "lifecycle_status".to_owned(),
            serde_json::json!("tombstoned"),
        );
        let other = edge_record(
            "other",
            serde_json::json!({
                "doc_id": "doc-other",
                "source_ids": ["n1"],
                "target_ids": ["n3"]
            }),
        );
        let nodes = vec![node_record("n1"), node_record("n2"), node_record("n3")];

        assert_eq!(
            graph_neighbors_from_records("n1", &nodes, &[stale, other], Some("doc-requested")),
            GraphNeighbors::default()
        );
    }

    #[test]
    fn graph_neighbors_excludes_tombstoned_and_dangling_endpoints() {
        let mut tombstoned_node = node_record("n2");
        tombstoned_node.metadata.insert(
            "lifecycle_status".to_owned(),
            serde_json::json!("tombstoned"),
        );
        let edges = vec![edge_record(
            "e1",
            serde_json::json!({
                "source_ids": ["n1", "n2", "missing"],
                "target_ids": [],
                "source_edge_ids": ["missing-edge"]
            }),
        )];
        let nodes = vec![node_record("n1"), tombstoned_node];

        assert_eq!(
            graph_neighbors_from_records("n1", &nodes, &edges, None),
            GraphNeighbors {
                nodes: BTreeSet::new(),
                edges: BTreeSet::from(["e1".to_owned()]),
            }
        );
        assert_eq!(
            graph_neighbors_from_records("e1", &nodes, &edges, None),
            GraphNeighbors {
                nodes: BTreeSet::from(["n1".to_owned()]),
                edges: BTreeSet::new(),
            }
        );
    }

    #[test]
    fn graph_find_edges_matches_relation_document_and_label_filters() {
        let mut source = node_record("n1");
        source.document = Some(serde_json::json!({"label": "Alpha source"}).to_string());
        let mut target = node_record("n2");
        target.document = Some(serde_json::json!({"label": "Beta target"}).to_string());
        let edges = vec![
            edge_record(
                "e1",
                serde_json::json!({
                    "relation": "links",
                    "doc_id": "doc-1",
                    "source_ids": ["n1"],
                    "target_ids": ["n2"]
                }),
            ),
            edge_record(
                "e2",
                serde_json::json!({
                    "relation": "mentions",
                    "doc_id": "doc-1",
                    "source_ids": ["n1"],
                    "target_ids": ["n2"]
                }),
            ),
        ];
        assert_eq!(
            graph_find_edges_from_records(
                &GraphFindEdgesQuery {
                    relation: Some("links".to_owned()),
                    src_label_contains: Some("alpha".to_owned()),
                    tgt_label_contains: Some("TARGET".to_owned()),
                    doc_id: Some("doc-1".to_owned()),
                },
                &[source, target],
                &edges,
            ),
            vec!["e1".to_owned()]
        );
    }

    #[test]
    fn graph_shortest_path_is_bounded_and_respects_document_and_tombstone_filters() {
        let nodes = vec![node_record("n1"), node_record("n2"), node_record("n3")];
        let edges = vec![
            edge_record(
                "e1",
                serde_json::json!({"doc_id": "doc-1", "source_ids": ["n1"], "target_ids": ["n2"]}),
            ),
            edge_record(
                "e2",
                serde_json::json!({"doc_id": "doc-1", "source_ids": ["n2"], "target_ids": ["n3"]}),
            ),
            edge_record(
                "other-doc",
                serde_json::json!({"doc_id": "doc-2", "source_ids": ["n1"], "target_ids": ["n3"]}),
            ),
        ];
        let query = GraphShortestPathQuery {
            src_id: "n1".to_owned(),
            dst_id: "n3".to_owned(),
            doc_id: Some("doc-1".to_owned()),
            max_depth: 4,
        };
        assert_eq!(
            graph_shortest_path_from_records(&query, &nodes, &edges),
            vec!["n1", "e1", "n2", "e2", "n3"]
        );
        let too_shallow = GraphShortestPathQuery {
            max_depth: 1,
            ..query.clone()
        };
        assert!(graph_shortest_path_from_records(&too_shallow, &nodes, &edges).is_empty());
        let mut tombstoned = edges[1].clone();
        tombstoned.metadata.insert(
            "lifecycle_status".to_owned(),
            serde_json::json!("tombstoned"),
        );
        assert!(
            graph_shortest_path_from_records(&query, &nodes, &[edges[0].clone(), tombstoned])
                .is_empty()
        );
    }

    #[test]
    fn graph_k_hop_returns_stable_layers_without_reexpanding_visited_ids() {
        let nodes = vec![node_record("n1"), node_record("n2"), node_record("n3")];
        let edges = vec![
            edge_record(
                "e1",
                serde_json::json!({"doc_id": "doc-1", "source_ids": ["n1"], "target_ids": ["n2"]}),
            ),
            edge_record(
                "e2",
                serde_json::json!({"doc_id": "doc-1", "source_ids": ["n2"], "target_ids": ["n3"]}),
            ),
        ];
        let layers = graph_k_hop_from_records(
            &GraphKHopQuery {
                start_ids: vec!["n1".to_owned()],
                k: 2,
                doc_id: Some("doc-1".to_owned()),
            },
            &nodes,
            &edges,
        );
        assert_eq!(layers.len(), 2);
        assert!(layers[0].nodes.is_empty());
        assert_eq!(layers[0].edges, BTreeSet::from(["e1".to_owned()]));
        assert_eq!(
            layers[1].nodes,
            BTreeSet::from(["n1".to_owned(), "n2".to_owned()])
        );
        assert!(layers[1].edges.is_empty());
    }

    #[test]
    fn graph_traversal_matches_shared_golden_contract() {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/graph-traversal-v1.json"
        )))
        .expect("valid graph traversal fixture");
        assert_eq!(fixture["schema_version"], 1);
        let string_set = |value: &Value| {
            value
                .as_array()
                .expect("fixture string array")
                .iter()
                .map(|item| item.as_str().expect("fixture string").to_owned())
                .collect::<BTreeSet<_>>()
        };
        let nodes = fixture["graph"]["nodes"]
            .as_array()
            .expect("fixture nodes")
            .iter()
            .map(|value| {
                let id = value.as_str().expect("node ID");
                let mut record = node_record(id);
                let label = fixture["graph"]["node_labels"][id]
                    .as_str()
                    .expect("node label");
                record.document = Some(serde_json::json!({"label": label}).to_string());
                record
            })
            .collect::<Vec<_>>();
        let edges = fixture["graph"]["edges"]
            .as_array()
            .expect("fixture edges")
            .iter()
            .map(|value| {
                edge_record(
                    value["id"].as_str().expect("edge ID"),
                    serde_json::json!({
                        "doc_id": value["doc_id"],
                        "relation": value["relation"],
                        "source_ids": value["source_ids"],
                        "target_ids": value["target_ids"]
                    }),
                )
            })
            .collect::<Vec<_>>();
        let cases = fixture["cases"].as_array().expect("fixture cases");

        let node_case = &cases[0];
        let neighbors = graph_neighbors_from_records_with_jump(
            node_case["rid"].as_str().expect("neighbor ID"),
            &nodes,
            &edges,
            Some("D1"),
            false,
        );
        assert_eq!(
            neighbors.nodes,
            string_set(&node_case["expected_neighbors"]["nodes"])
        );
        assert_eq!(
            neighbors.edges,
            string_set(&node_case["expected_neighbors"]["edges"])
        );

        let edge_case = &cases[1];
        let neighbors = graph_neighbors_from_records(
            edge_case["rid"].as_str().expect("edge ID"),
            &nodes,
            &edges,
            Some("D1"),
        );
        assert_eq!(
            neighbors.nodes,
            string_set(&edge_case["expected_neighbors"]["nodes"])
        );
        assert_eq!(
            neighbors.edges,
            string_set(&edge_case["expected_neighbors"]["edges"])
        );

        let path_case = &cases[2];
        let path = graph_shortest_path_from_records(
            &GraphShortestPathQuery {
                src_id: path_case["src_id"].as_str().unwrap().to_owned(),
                dst_id: path_case["dst_id"].as_str().unwrap().to_owned(),
                doc_id: Some("D1".to_owned()),
                max_depth: 8,
            },
            &nodes,
            &edges,
        );
        let expected_path = path_case["expected_path"]
            .as_array()
            .expect("fixture path")
            .iter()
            .map(|value| value.as_str().expect("fixture path ID").to_owned())
            .collect::<Vec<_>>();
        assert_eq!(path, expected_path);

        let hop_case = &cases[3];
        let layers = graph_k_hop_from_records(
            &GraphKHopQuery {
                start_ids: vec!["A".to_owned()],
                k: hop_case["k"].as_u64().unwrap() as usize,
                doc_id: Some("D1".to_owned()),
            },
            &nodes,
            &edges,
        );
        let expected_layers = hop_case["expected_layers"]
            .as_array()
            .expect("fixture layers");
        assert_eq!(layers.len(), expected_layers.len());
        for (actual, expected) in layers.iter().zip(expected_layers) {
            assert_eq!(actual.nodes, string_set(&expected["nodes"]));
            assert_eq!(actual.edges, string_set(&expected["edges"]));
        }

        let find_case = &cases[4];
        let matching_edges = graph_find_edges_from_records(
            &GraphFindEdgesQuery {
                relation: find_case["relation"].as_str().map(str::to_owned),
                src_label_contains: find_case["source_label_contains"]
                    .as_str()
                    .map(str::to_owned),
                tgt_label_contains: find_case["target_label_contains"]
                    .as_str()
                    .map(str::to_owned),
                doc_id: find_case["doc_id"].as_str().map(str::to_owned),
            },
            &nodes,
            &edges,
        );
        let expected_edges = find_case["expected_edge_ids"]
            .as_array()
            .expect("fixture edge IDs")
            .iter()
            .map(|value| value.as_str().expect("fixture edge ID").to_owned())
            .collect::<Vec<_>>();
        assert_eq!(matching_edges, expected_edges);
    }

    #[test]
    fn graph_record_acl_matches_python_visibility_and_fails_closed() {
        let mut acl = GraphRecord::new("acl-n1");
        acl.metadata
            .insert("entity_type".into(), "acl_record".into());
        acl.metadata
            .insert("acl_truth_graph".into(), "knowledge".into());
        acl.metadata
            .insert("acl_target_entity_id".into(), "n1".into());
        acl.metadata
            .insert("acl_target_grain".into(), "node".into());
        acl.metadata.insert("acl_version".into(), 1.into());
        acl.metadata.insert("acl_mode".into(), "shared".into());
        acl.document = Some(
            serde_json::json!({
                "properties": {"shared_with_principals": ["alice"]}
            })
            .to_string(),
        );
        let nodes = vec![acl, GraphRecord::new("n1"), GraphRecord::new("n2")];
        let edges = vec![GraphRecord::new("e1")];
        let alice = GraphAclPrincipal {
            principal_id: "alice".to_owned(),
            ..Default::default()
        };
        let bob = GraphAclPrincipal {
            principal_id: "bob".to_owned(),
            ..Default::default()
        };
        let (alice_nodes, alice_edges) =
            graph_records_visible_by_acl(&nodes, &edges, "knowledge", &alice);
        assert_eq!(
            alice_nodes
                .iter()
                .map(|r| r.id.as_str())
                .collect::<Vec<_>>(),
            ["n1"]
        );
        assert!(alice_edges.is_empty());
        let (bob_nodes, bob_edges) =
            graph_records_visible_by_acl(&nodes, &edges, "knowledge", &bob);
        assert!(bob_nodes.is_empty());
        assert!(bob_edges.is_empty());
    }

    #[test]
    fn graph_record_acl_covers_owner_scope_group_private_and_stale_item_rules() {
        let acl_record = |id: &str, target: &str, mode: &str, version: i64| {
            let mut record = GraphRecord::new(id);
            record
                .metadata
                .insert("entity_type".into(), "acl_record".into());
            record
                .metadata
                .insert("acl_truth_graph".into(), "knowledge".into());
            record
                .metadata
                .insert("acl_target_entity_id".into(), target.into());
            record
                .metadata
                .insert("acl_target_grain".into(), "node".into());
            record.metadata.insert("acl_version".into(), version.into());
            record.metadata.insert("acl_mode".into(), mode.into());
            record
        };
        let mut owner = acl_record("acl-owner", "owner", "private", 1);
        owner.metadata.insert("owner_id".into(), "alice".into());
        let mut scoped = acl_record("acl-scope", "scope", "scope", 1);
        scoped
            .metadata
            .insert("security_scope".into(), "team-a".into());
        let mut grouped = acl_record("acl-group", "group", "group", 1);
        grouped.document = Some(
            serde_json::json!({"properties": {"shared_with_groups": ["reviewers"]}}).to_string(),
        );
        let mut stale_item = acl_record("acl-span", "span", "public", 1);
        stale_item
            .metadata
            .insert("acl_target_item_id".into(), "span-1".into());
        let nodes = vec![
            owner,
            scoped,
            grouped,
            stale_item,
            GraphRecord::new("owner"),
            GraphRecord::new("scope"),
            GraphRecord::new("group"),
            GraphRecord::new("span"),
        ];
        let principal = GraphAclPrincipal {
            principal_id: "alice".into(),
            groups: BTreeSet::from(["reviewers".into()]),
            security_scope: Some("team-a".into()),
        };
        let (visible, _) = graph_records_visible_by_acl(&nodes, &[], "knowledge", &principal);
        assert_eq!(
            visible
                .iter()
                .map(|record| record.id.as_str())
                .collect::<Vec<_>>(),
            ["owner", "scope", "group"]
        );
    }

    #[test]
    fn graph_acl_matches_shared_golden_contract() {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/graph-acl-v1.json"
        )))
        .expect("valid graph ACL fixture");
        assert_eq!(fixture["schema_version"], 1);
        let principal = GraphAclPrincipal {
            principal_id: fixture["principal"]["id"]
                .as_str()
                .expect("principal ID")
                .to_owned(),
            groups: fixture["principal"]["groups"]
                .as_array()
                .expect("principal groups")
                .iter()
                .map(|value| value.as_str().expect("group").to_owned())
                .collect(),
            security_scope: fixture["principal"]["security_scope"]
                .as_str()
                .map(str::to_owned),
        };
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        for target in fixture["targets"].as_array().expect("targets") {
            let id = target["id"].as_str().expect("target ID");
            let grain = target["grain"].as_str().expect("target grain");
            let target_record = GraphRecord::new(id);
            let mut acl = GraphRecord::new(format!("acl-{id}"));
            acl.metadata
                .insert("entity_type".into(), "acl_record".into());
            acl.metadata
                .insert("acl_truth_graph".into(), fixture["truth_graph"].clone());
            acl.metadata
                .insert("acl_target_entity_id".into(), id.into());
            acl.metadata.insert("acl_target_grain".into(), grain.into());
            acl.metadata.insert("acl_version".into(), 1.into());
            if let Some(mode) = target["mode"].as_str() {
                acl.metadata.insert("acl_mode".into(), mode.into());
            }
            if let Some(owner) = target["owner_id"].as_str() {
                acl.metadata.insert("owner_id".into(), owner.into());
            }
            if let Some(scope) = target["security_scope"].as_str() {
                acl.metadata.insert("security_scope".into(), scope.into());
            }
            let mut properties = serde_json::Map::new();
            if let Some(values) = target["shared_with_principals"].as_array() {
                properties.insert(
                    "shared_with_principals".into(),
                    Value::Array(values.clone()),
                );
            }
            if let Some(values) = target["shared_with_groups"].as_array() {
                properties.insert("shared_with_groups".into(), Value::Array(values.clone()));
            }
            if target["tombstoned"].as_bool().unwrap_or(false) {
                acl.metadata.insert("tombstoned".into(), true.into());
            }
            if !properties.is_empty() {
                acl.document = Some(serde_json::json!({"properties": properties}).to_string());
            }
            nodes.push(acl);
            if grain == "node" {
                nodes.push(target_record);
            } else {
                edges.push(target_record);
            }
        }
        let (visible_nodes, visible_edges) =
            graph_records_visible_by_acl(&nodes, &edges, "knowledge", &principal);
        let node_ids = visible_nodes
            .iter()
            .map(|record| record.id.as_str())
            .collect::<Vec<_>>();
        let edge_ids = visible_edges
            .iter()
            .map(|record| record.id.as_str())
            .collect::<Vec<_>>();
        let expected_nodes = fixture["expected_visible"]["nodes"]
            .as_array()
            .expect("expected nodes")
            .iter()
            .map(|value| value.as_str().expect("node ID").to_owned())
            .collect::<Vec<_>>();
        let expected_edges = fixture["expected_visible"]["edges"]
            .as_array()
            .expect("expected edges")
            .iter()
            .map(|value| value.as_str().expect("edge ID").to_owned())
            .collect::<Vec<_>>();
        assert_eq!(node_ids, expected_nodes);
        assert_eq!(edge_ids, expected_edges);
    }

    #[test]
    fn tool_audit_projection_is_scope_filtered_and_canonical() {
        let mut call = GraphRecord::new("tool-call-1");
        call.metadata
            .insert("entity_type".into(), "tool_call".into());
        call.metadata
            .insert("tool_name".into(), "graph.read".into());
        call.metadata
            .insert("conversation_id".into(), "conversation-1".into());
        call.metadata.insert("turn_index".into(), 2.into());
        call.document = Some(
            serde_json::json!({
                "properties": {"status": "completed", "count": 1}
            })
            .to_string(),
        );

        let mut other = GraphRecord::new("tool-result-other");
        other
            .metadata
            .insert("entity_type".into(), "tool_result".into());
        other
            .metadata
            .insert("tool_name".into(), "graph.read".into());
        other
            .metadata
            .insert("conversation_id".into(), "conversation-2".into());

        let mut tombstoned = GraphRecord::new("tool-call-deleted");
        tombstoned
            .metadata
            .insert("entity_type".into(), "tool_call".into());
        tombstoned
            .metadata
            .insert("tool_name".into(), "secret".into());
        tombstoned
            .metadata
            .insert("conversation_id".into(), "conversation-1".into());
        tombstoned
            .metadata
            .insert("lifecycle_status".into(), "tombstoned".into());

        let items = tool_audit_items_from_visible_records(
            &[call, other, tombstoned],
            Some("conversation-1"),
            10,
        )
        .unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].node_id, "tool-call-1");
        assert_eq!(items[0].turn_index, Some(2));
        assert_eq!(items[0].properties["status"], "completed");
    }

    #[test]
    fn tool_audit_projection_fails_closed_for_malformed_visible_records() {
        let mut malformed = GraphRecord::new("tool-call-invalid");
        malformed
            .metadata
            .insert("entity_type".into(), "tool_call".into());
        malformed
            .metadata
            .insert("conversation_id".into(), "conversation-1".into());
        let error = tool_audit_items_from_visible_records(&[malformed], None, 10).unwrap_err();
        assert_eq!(
            error,
            kogwistar_contracts::ToolAuditContractError::EmptyField { field: "tool_name" }
        );
    }

    #[test]
    fn conversation_transcript_projection_is_ordered_and_scope_filtered() {
        let mut assistant = GraphRecord::new("assistant-1");
        assistant
            .metadata
            .insert("entity_type".into(), "assistant_turn".into());
        assistant
            .metadata
            .insert("conversation_id".into(), "conversation-1".into());
        assistant.metadata.insert("turn_index".into(), 1.into());
        assistant.metadata.insert("role".into(), "assistant".into());
        assistant.metadata.insert("summary".into(), "answer".into());

        let mut user = GraphRecord::new("user-0");
        user.metadata
            .insert("entity_type".into(), "conversation_turn".into());
        user.metadata
            .insert("conversation_id".into(), "conversation-1".into());
        user.metadata.insert("turn_index".into(), 0.into());
        user.metadata.insert("role".into(), "user".into());
        user.metadata.insert("summary".into(), "question".into());

        let mut other = user.clone();
        other.id = "other-user".into();
        other
            .metadata
            .insert("conversation_id".into(), "conversation-2".into());

        let transcript = conversation_transcript_from_visible_records(
            &[assistant, other, user],
            "conversation-1",
            10,
        )
        .unwrap();
        assert_eq!(transcript.turns.len(), 2);
        assert_eq!(transcript.turns[0].node_id, "user-0");
        assert_eq!(transcript.turns[1].node_id, "assistant-1");
    }

    #[test]
    fn conversation_transcript_projection_rejects_duplicate_turn_indexes() {
        let mut first = GraphRecord::new("turn-a");
        first.metadata.extend(
            serde_json::json!({
                "entity_type": "conversation_turn",
                "conversation_id": "conversation-1",
                "turn_index": 0,
                "role": "user",
                "summary": "one"
            })
            .as_object()
            .unwrap()
            .clone(),
        );
        let mut second = first.clone();
        second.id = "turn-b".into();
        let error =
            conversation_transcript_from_visible_records(&[first, second], "conversation-1", 0)
                .unwrap_err();
        assert_eq!(error, ConversationContractError::InvalidTurnOrder);
    }

    #[test]
    fn conversation_summary_projection_derives_owner_status_and_tail() {
        let mut start = GraphRecord::new("conversation-start");
        start.metadata.extend(
            serde_json::json!({
                "entity_type": "conversation_start",
                "conversation_id": "conversation-1",
                "user_id": "alice",
                "status": "active"
            })
            .as_object()
            .unwrap()
            .clone(),
        );
        let mut turn = GraphRecord::new("turn-0");
        turn.metadata.extend(
            serde_json::json!({
                "entity_type": "conversation_turn",
                "conversation_id": "conversation-1",
                "turn_index": 0,
                "role": "user",
                "summary": "question"
            })
            .as_object()
            .unwrap()
            .clone(),
        );
        let summary =
            conversation_summary_from_visible_records(&[start, turn], "conversation-1").unwrap();
        assert_eq!(summary.user_id, "alice");
        assert_eq!(summary.start_node_id, "conversation-start");
        assert_eq!(summary.tail_node_id, "turn-0");
        assert_eq!(summary.turn_count, 1);
    }

    #[test]
    fn context_snapshot_projection_selects_latest_matching_scope() {
        let mut older = GraphRecord::new("snapshot-old");
        older
            .metadata
            .insert("entity_type".into(), "context_snapshot".into());
        older
            .metadata
            .insert("conversation_id".into(), "conversation-1".into());
        older.metadata.insert("run_id".into(), "run-1".into());
        older.metadata.insert("stage".into(), "answer".into());
        older.metadata.insert("snapshot_seq".into(), 3.into());
        older.document = Some(
            serde_json::json!({
                "metadata": {"stage": "answer"},
                "properties": {"value": "old"}
            })
            .to_string(),
        );

        let mut newer = older.clone();
        newer.id = "snapshot-new".into();
        newer.metadata.insert("snapshot_seq".into(), 4.into());
        newer.document = Some(
            serde_json::json!({
                "metadata": {"stage": "answer"},
                "properties": {"value": "new"}
            })
            .to_string(),
        );

        let snapshot = context_snapshot_from_visible_records(
            &[older, newer],
            "conversation-1",
            Some("run-1"),
            Some("answer"),
        )
        .unwrap();
        assert_eq!(snapshot.snapshot_node_id, "snapshot-new");
        assert_eq!(snapshot.properties["value"], "new");
    }

    #[test]
    fn context_snapshot_projection_rejects_unordered_candidates() {
        let mut record = GraphRecord::new("snapshot-invalid");
        record
            .metadata
            .insert("entity_type".into(), "context_snapshot".into());
        record
            .metadata
            .insert("conversation_id".into(), "conversation-1".into());
        let error = context_snapshot_from_visible_records(&[record], "conversation-1", None, None)
            .unwrap_err();
        assert_eq!(error, ConversationContractError::MissingSnapshotOrder);
    }
    use kogwistar_contracts::{GraphModelClass, GraphModelEnvelope};

    fn fixture_payload(index: usize) -> Value {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/python-graph-model-payloads-v1.json"
        )))
        .expect("valid graph model fixture");
        fixture["cases"][index]["payload"].clone()
    }

    fn typed_mutation(entity_kind: &str, payload: Value) -> GraphMutation {
        let id = payload["id"].as_str().expect("fixture id").to_owned();
        GraphMutation {
            scope: GraphScope {
                namespace: "ns".to_owned(),
                workspace_id: Some("ws".to_owned()),
                graph_space: Some("facts".to_owned()),
            },
            table: if entity_kind == "edge" {
                "gke_edges".to_owned()
            } else {
                "gke_nodes".to_owned()
            },
            entity_kind: entity_kind.to_owned(),
            event_id: "event-1".to_owned(),
            op: "ADD".to_owned(),
            payload: payload.clone(),
            record: GraphRecord {
                id,
                document: None,
                metadata: payload["metadata"].as_object().cloned().unwrap_or_default(),
                embedding: None,
            },
            embedding_dim: 0,
        }
    }

    #[test]
    fn runtime_projection_contract_has_one_stable_shape() {
        let mut payload = Map::new();
        payload.insert("status".to_owned(), Value::String("running".to_owned()));
        let projection = runtime_projection_write(payload, 7, "ready");
        assert_eq!(
            runtime_checkpoint_namespace("conv"),
            "conv:workflow_checkpoint_latest"
        );
        assert_eq!(runtime_status_namespace("conv"), "conv:workflow_run_status");
        assert_eq!(projection.last_authoritative_seq, 7);
        assert_eq!(projection.last_materialized_seq, 7);
        assert_eq!(
            projection.projection_schema_version,
            RUNTIME_PROJECTION_SCHEMA_VERSION
        );
        assert_eq!(projection.materialization_status, "ready");
    }

    #[test]
    fn typed_graph_mutation_accepts_matching_payload_and_identity() {
        let payload = fixture_payload(1);
        let mutation = typed_mutation("edge", payload.clone());
        let envelope = GraphModelEnvelope::new(GraphModelClass::Edge, payload);
        validate_typed_graph_mutation(&mutation, &envelope).expect("valid typed mutation");
    }

    #[test]
    fn typed_graph_mutation_rejects_kind_and_identity_confusion() {
        let edge_payload = fixture_payload(1);
        let edge_mutation = typed_mutation("node", edge_payload.clone());
        let edge_envelope = GraphModelEnvelope::new(GraphModelClass::Edge, edge_payload);
        assert!(matches!(
            validate_typed_graph_mutation(&edge_mutation, &edge_envelope),
            Err(StoreError::InvalidEntityEventPayload { .. })
        ));

        let node_payload = fixture_payload(0);
        let mut node_mutation = typed_mutation("node", node_payload.clone());
        node_mutation.record.id = "different-id".to_owned();
        let node_envelope = GraphModelEnvelope::new(GraphModelClass::Node, node_payload);
        assert!(matches!(
            validate_typed_graph_mutation(&node_mutation, &node_envelope),
            Err(StoreError::InvalidEntityEventPayload { .. })
        ));
    }

    #[test]
    fn typed_graph_mutation_cannot_bypass_base_mutation_validation() {
        let payload = fixture_payload(0);
        let envelope = GraphModelEnvelope::new(GraphModelClass::Node, payload.clone());
        let valid = typed_mutation("node", payload);

        let mut invalid = valid.clone();
        invalid.table = "gke_edges".to_owned();
        assert!(validate_typed_graph_mutation(&invalid, &envelope).is_err());

        let mut invalid = valid.clone();
        invalid.scope.namespace.clear();
        assert!(validate_typed_graph_mutation(&invalid, &envelope).is_err());

        let mut invalid = valid.clone();
        invalid.op = "DELETE".to_owned();
        assert!(validate_typed_graph_mutation(&invalid, &envelope).is_err());

        let mut invalid = valid.clone();
        invalid.payload = Value::Array(Vec::new());
        assert!(validate_typed_graph_mutation(&invalid, &envelope).is_err());
    }

    #[test]
    fn graph_projection_target_mapping_is_shared_and_exact() {
        let targets = [
            ("node", "gke_nodes"),
            ("edge", "gke_edges"),
            ("document", "gke_documents"),
            ("domain", "gke_domains"),
        ];
        for (entity_kind, table) in targets {
            assert_eq!(graph_projection_table(entity_kind), Ok(table));
            validate_graph_projection_target(table, entity_kind)
                .expect("matching projection target should be accepted");
            let mutation = GraphMutation {
                scope: GraphScope {
                    namespace: "ns".to_owned(),
                    workspace_id: None,
                    graph_space: None,
                },
                table: table.to_owned(),
                entity_kind: entity_kind.to_owned(),
                event_id: format!("{entity_kind}-event"),
                op: "ADD".to_owned(),
                payload: if matches!(entity_kind, "node" | "edge") {
                    serde_json::json!({"id": format!("{entity_kind}-id")})
                } else {
                    serde_json::json!({})
                },
                record: GraphRecord::new(format!("{entity_kind}-id")),
                embedding_dim: 0,
            };
            validate_graph_mutation(&mutation).expect("projection mutation should be valid");
            for (_, other_table) in targets {
                if other_table != table {
                    assert!(validate_graph_projection_target(other_table, entity_kind).is_err());
                }
            }
        }
        assert!(graph_projection_table("unknown").is_err());
        assert!(validate_graph_lifecycle_target("gke_documents", "document").is_err());
        assert!(validate_graph_lifecycle_target("gke_domains", "domain").is_err());
        assert!(validate_graph_lifecycle_target("gke_nodes", "node").is_ok());
        assert!(validate_graph_lifecycle_target("gke_edges", "edge").is_ok());
    }

    #[test]
    fn direct_graph_mutation_validation_rejects_effect_bypass_inputs() {
        let mut mutation = typed_mutation("node", fixture_payload(0));
        mutation.table = "gke_edges".to_owned();
        assert!(matches!(
            validate_graph_mutation(&mutation),
            Err(StoreError::InvalidEntityEventPayload { .. })
        ));

        mutation.table = "gke_nodes".to_owned();
        mutation.payload = Value::Array(Vec::new());
        assert!(matches!(
            validate_graph_mutation(&mutation),
            Err(StoreError::InvalidEntityEventPayload { .. })
        ));
    }

    #[test]
    fn graph_mutation_batch_validation_rejects_scope_and_index_drift() {
        let first = typed_mutation("node", fixture_payload(0));
        let mut second = typed_mutation("edge", fixture_payload(1));
        second.scope.namespace = "other".to_owned();
        assert!(matches!(
            validate_graph_mutation_batch(&GraphMutationBatch {
                mutations: vec![first.clone(), second],
                index_jobs: Vec::new(),
            }),
            Err(StoreError::InvalidEntityEventPayload { .. })
        ));

        assert!(matches!(
            validate_graph_mutation_batch(&GraphMutationBatch {
                mutations: vec![first],
                index_jobs: vec![NewIndexJob {
                    job_id: "job-1".to_owned(),
                    namespace: "other".to_owned(),
                    entity_kind: "node".to_owned(),
                    entity_id: "n-1".to_owned(),
                    index_kind: "doc".to_owned(),
                    op: "UPSERT".to_owned(),
                    payload_json: None,
                    max_retries: 0,
                }],
            }),
            Err(StoreError::InvalidEntityEventPayload { .. })
        ));
    }

    #[test]
    fn graph_mutation_rejects_payload_and_projection_identity_drift() {
        let payload = fixture_payload(0);
        let mut mutation = typed_mutation("node", payload);
        mutation.record.id = "different-node".to_owned();

        assert!(matches!(
            validate_graph_mutation(&mutation),
            Err(StoreError::InvalidEntityEventPayload { .. })
        ));
    }
}
