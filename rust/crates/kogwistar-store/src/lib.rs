//! Async-first storage contracts. Shadow handles expose reads only.
pub use kogwistar_contracts::GraphEntityKind;
use kogwistar_contracts::{
    ContextSnapshotV1, ConversationContractError, ConversationSummaryV1, ConversationTranscriptV1,
    ConversationTurnV1, EntityEventEnvelope, GraphModelClass, GraphModelEnvelope,
    PersistentGraphEntityV1, ToolAuditContractError, ToolAuditItemV1,
};
use serde::Serialize;
use serde_json::{Map, Value, json};
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
    /// Read the Python-compatible per-workflow role from the shared auth DB.
    fn workflow_acl_role(
        &self,
        _workflow_id: &str,
        _user_id: &str,
    ) -> impl Future<Output = StoreResult<Option<String>>> + Send {
        async { Ok(None) }
    }
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConversationListEntry {
    pub id: String,
    pub start_node_id: String,
    pub status: String,
    pub turn_count: i64,
}

/// Incrementally projects conversation-list rows from already-authorized pages.
/// Retains only turn counts and matching start records, not every graph node.
#[derive(Clone, Debug, Default)]
pub struct ConversationListAccumulator {
    user_id: String,
    turn_counts: BTreeMap<String, i64>,
    starts: Vec<(String, String, String)>,
}

impl ConversationListAccumulator {
    pub fn new(user_id: impl Into<String>) -> Self {
        Self {
            user_id: user_id.into(),
            ..Self::default()
        }
    }

    /// Add one page after the caller has applied namespace and ACL filtering.
    pub fn extend_visible_records(&mut self, records: &[GraphRecord]) {
        for record in records
            .iter()
            .filter(|record| !is_tombstoned(&record.metadata))
        {
            match value_string(record, "entity_type").as_deref() {
                Some("conversation_turn" | "assistant_turn")
                    if record_field(record, "turn_index")
                        .and_then(|value| value.as_i64())
                        .is_some() =>
                {
                    if let Some(conversation_id) = value_string(record, "conversation_id") {
                        *self.turn_counts.entry(conversation_id).or_default() += 1;
                    }
                }
                Some("conversation_start")
                    if value_string(record, "user_id").as_deref()
                        == Some(self.user_id.as_str()) =>
                {
                    if let Some(conversation_id) = value_string(record, "conversation_id") {
                        let status =
                            value_string(record, "status").unwrap_or_else(|| "active".to_owned());
                        self.starts
                            .push((conversation_id, record.id.clone(), status));
                    }
                }
                _ => {}
            }
        }
    }

    pub fn finish(self) -> Vec<ConversationListEntry> {
        let mut entries = self
            .starts
            .into_iter()
            .map(
                |(conversation_id, start_node_id, status)| ConversationListEntry {
                    turn_count: self
                        .turn_counts
                        .get(&conversation_id)
                        .copied()
                        .unwrap_or_default(),
                    id: conversation_id,
                    start_node_id,
                    status,
                },
            )
            .collect::<Vec<_>>();
        entries.sort_by_key(|entry| std::cmp::Reverse(entry.turn_count));
        entries
    }
}

/// Project conversation-list source rows to the small field set needed after
/// graph scanning. Source documents and embeddings are deliberately discarded.
pub fn compact_conversation_list_candidates(records: &[GraphRecord]) -> Vec<GraphRecord> {
    const FIELDS: &[&str] = &[
        "entity_type",
        "turn_index",
        "conversation_id",
        "user_id",
        "status",
        "lifecycle_status",
    ];
    records
        .iter()
        .filter_map(|record| {
            let entity_type = value_string(record, "entity_type")?;
            if !matches!(
                entity_type.as_str(),
                "conversation_start" | "conversation_turn" | "assistant_turn"
            ) {
                return None;
            }
            let metadata = FIELDS
                .iter()
                .filter_map(|field| {
                    record_field(record, field).map(|value| ((*field).to_owned(), value))
                })
                .collect();
            Some(GraphRecord {
                id: record.id.clone(),
                document: None,
                metadata,
                embedding: None,
            })
        })
        .collect()
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

type GraphAclKey = (String, String, Option<String>);

/// Compact, reusable ACL rules collected independently of graph-record pages.
/// This is required when a target and its ACL record are in different pages.
#[derive(Clone, Debug, Default)]
pub struct GraphAclIndex {
    rules: BTreeMap<GraphAclKey, GraphAclRule>,
}

impl GraphAclIndex {
    pub fn extend_from_records(&mut self, records: &[GraphRecord], truth_graph: &str) {
        for record in records {
            let Some((key, rule)) = acl_rule_for(record, truth_graph) else {
                continue;
            };
            let replace = self.rules.get(&key).is_none_or(|current| {
                (current.tombstoned && !rule.tombstoned)
                    || (current.tombstoned == rule.tombstoned
                        && (rule.version, acl_rank(&rule.mode))
                            > (current.version, acl_rank(&current.mode)))
            });
            if replace {
                self.rules.insert(key, rule);
            }
        }
    }
}

/// ACL-filtered graph records paired with the complete authorization context
/// that produced them. Use `filter_records` for later bounded projections so
/// ACL helper records removed from the visible records are not needed again.
#[derive(Clone, Debug)]
pub struct GraphAclProjection {
    nodes: Vec<GraphRecord>,
    edges: Vec<GraphRecord>,
    index: GraphAclIndex,
    principal: GraphAclPrincipal,
}

impl GraphAclProjection {
    /// Build an authorized projection from a complete graph snapshot.
    pub fn from_complete_snapshot(
        nodes: &[GraphRecord],
        edges: &[GraphRecord],
        truth_graph: &str,
        principal: &GraphAclPrincipal,
    ) -> Self {
        let mut index = GraphAclIndex::default();
        index.extend_from_records(nodes, truth_graph);
        Self::from_indexed_records(nodes, edges, index, principal)
    }

    /// Build an authorized projection using ACL rules collected from the
    /// complete relevant set, including records on other pages.
    pub fn from_indexed_records(
        nodes: &[GraphRecord],
        edges: &[GraphRecord],
        index: GraphAclIndex,
        principal: &GraphAclPrincipal,
    ) -> Self {
        let (nodes, edges) =
            graph_records_visible_by_acl_with_index(nodes, edges, &index, principal);
        Self {
            nodes,
            edges,
            index,
            principal: principal.clone(),
        }
    }

    /// Re-filter a record subset without losing the original ACL context.
    pub fn filter_records(&self, nodes: &[GraphRecord], edges: &[GraphRecord]) -> Self {
        Self::from_indexed_records(nodes, edges, self.index.clone(), &self.principal)
    }

    pub fn nodes(&self) -> &[GraphRecord] {
        &self.nodes
    }

    pub fn edges(&self) -> &[GraphRecord] {
        &self.edges
    }
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
    document.get(field).cloned()
}

fn graph_visualization_edge_relation(record: &GraphRecord) -> Option<Value> {
    ["relation", "label"]
        .into_iter()
        .filter_map(|field| record_field(record, field))
        .find(|value| value.as_str().is_some_and(|text| !text.is_empty()))
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

fn acl_rule_for(record: &GraphRecord, truth_graph: &str) -> Option<(GraphAclKey, GraphAclRule)> {
    if !is_acl_record(record)
        || value_string(record, "acl_truth_graph").as_deref() != Some(truth_graph)
    {
        return None;
    }
    let grain = value_string(record, "acl_target_grain")?;
    if !matches!(grain.as_str(), "node" | "edge" | "grounding" | "span") {
        return None;
    }
    let target_id = value_string(record, "acl_target_entity_id")?;
    let target_item_id = value_string(record, "acl_target_item_id")
        .filter(|target_item_id| !target_item_id.is_empty());
    let version = record_field(record, "acl_version")?.as_i64()?;
    let mode = value_string(record, "acl_mode")?.to_ascii_lowercase();
    Some((
        (grain, target_id, target_item_id),
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

fn indexed_acl_visible(
    index: &GraphAclIndex,
    grain: &str,
    entity_id: &str,
    target_item_id: Option<&str>,
    principal: &GraphAclPrincipal,
) -> bool {
    let key = (
        grain.to_owned(),
        entity_id.to_owned(),
        target_item_id.map(str::to_owned),
    );
    index
        .rules
        .get(&key)
        .is_some_and(|rule| acl_rule_visible(rule, principal))
}

fn record_array_field(record: &GraphRecord, field: &str) -> Option<Vec<Value>> {
    let Some(value) = record_field(record, field) else {
        return Some(Vec::new());
    };
    match value {
        Value::Array(values) => Some(values),
        Value::String(encoded) => serde_json::from_str::<Value>(&encoded)
            .ok()?
            .as_array()
            .cloned(),
        Value::Null => Some(Vec::new()),
        _ => None,
    }
}

fn record_string_array_field(record: &GraphRecord, field: &str) -> Option<Vec<String>> {
    record_array_field(record, field)?
        .into_iter()
        .map(|value| value.as_str().map(str::to_owned))
        .collect()
}

fn graph_entity_usage_acl_visible(
    record: &GraphRecord,
    entity_kind: &str,
    index: &GraphAclIndex,
    principal: &GraphAclPrincipal,
) -> bool {
    let Some(mentions) = record_array_field(record, "mentions") else {
        // Malformed mention metadata must not bypass item-level ACL checks.
        return false;
    };
    for (mention_index, mention) in mentions.iter().enumerate() {
        let Some(span_values) = mention.get("spans").and_then(Value::as_array) else {
            return false;
        };
        let grounding_id = format!(
            "__kogwistar_acl_usage_v2__:{entity_kind}:{}::mention::{mention_index}",
            record.id
        );
        let legacy_grounding_id = format!("{}::mention::{mention_index}", record.id);
        if !indexed_acl_usage_visible(
            index,
            "grounding",
            &record.id,
            &grounding_id,
            &legacy_grounding_id,
            principal,
        ) {
            return false;
        }
        for span_index in 0..span_values.len() {
            let span_id = format!("{grounding_id}::span::{span_index}");
            let legacy_span_id = format!("{legacy_grounding_id}::span::{span_index}");
            if !indexed_acl_usage_visible(
                index,
                "span",
                &record.id,
                &span_id,
                &legacy_span_id,
                principal,
            ) {
                return false;
            }
        }
    }
    true
}

fn indexed_acl_usage_visible(
    index: &GraphAclIndex,
    grain: &str,
    entity_id: &str,
    versioned_item_id: &str,
    legacy_item_id: &str,
    principal: &GraphAclPrincipal,
) -> bool {
    let versioned_key = (
        grain.to_owned(),
        entity_id.to_owned(),
        Some(versioned_item_id.to_owned()),
    );
    if index.rules.contains_key(&versioned_key) {
        return indexed_acl_visible(index, grain, entity_id, Some(versioned_item_id), principal);
    }
    // Persisted Python ACL rows used untyped mention IDs before v2. Continue
    // reading them unless a v2 row exists for this exact graph entity.
    indexed_acl_visible(index, grain, entity_id, Some(legacy_item_id), principal)
}

/// Filter a complete graph snapshot using the same fail-closed ACL rules as Python.
// PYTHON_PARITY_V1: rust=rust/crates/kogwistar-store/src/lib.rs#graph_records_visible_by_acl; python=kogwistar/acl/graph.py#ACLGraph.decide; fixture=contracts/golden/graph-acl-v1.json; rust_test=rust/crates/kogwistar-store/src/lib.rs#graph_acl_matches_shared_golden_contract; python_test=tests/core/test_graph_acl_golden.py#test_python_acl_visibility_matches_shared_golden_contract
///
/// ACL records are persisted as graph nodes. Missing records deny access, and
/// ACL helper nodes never enter the returned graph projection. Because the
/// result omits those helper nodes, callers must not rebuild an ACL index from
/// the result; retain the index and use
/// [`graph_records_visible_by_acl_with_index`] for later pages or projections.
pub fn graph_records_visible_by_acl(
    nodes: &[GraphRecord],
    edges: &[GraphRecord],
    truth_graph: &str,
    principal: &GraphAclPrincipal,
) -> (Vec<GraphRecord>, Vec<GraphRecord>) {
    let mut index = GraphAclIndex::default();
    index.extend_from_records(nodes, truth_graph);
    graph_records_visible_by_acl_with_index(nodes, edges, &index, principal)
}

/// Filter a page using an index built from the complete relevant ACL set.
/// ACL helper records are never returned as graph entities. Reuse this index
/// when filtering subsequent pages or projections; never derive it from a
/// previously filtered result.
pub fn graph_records_visible_by_acl_with_index(
    nodes: &[GraphRecord],
    edges: &[GraphRecord],
    index: &GraphAclIndex,
    principal: &GraphAclPrincipal,
) -> (Vec<GraphRecord>, Vec<GraphRecord>) {
    let visible = |record: &GraphRecord, grain: &str| {
        // Python's default graph reads resolve entities with `active_only`.
        if is_acl_record(record) || is_tombstoned(&record.metadata) {
            return false;
        }
        indexed_acl_visible(index, grain, &record.id, None, principal)
            && graph_entity_usage_acl_visible(record, grain, index, principal)
    };
    let visible_nodes = nodes
        .iter()
        .filter(|record| visible(record, "node"))
        .cloned()
        .collect::<Vec<_>>();
    let edge_base_visible = |edge: &GraphRecord| {
        if !visible(edge, "edge") {
            return false;
        }
        let Some(source_ids) = record_string_array_field(edge, "source_ids") else {
            return false;
        };
        let Some(target_ids) = record_string_array_field(edge, "target_ids") else {
            return false;
        };
        let endpoint_ids = source_ids.into_iter().chain(target_ids);
        // Python counterpart: ACLAwareReadSubsystem._edge_visible requires every
        // referenced node to resolve and pass ACL checks before returning the edge.
        endpoint_ids.into_iter().all(|endpoint_id| {
            nodes
                .iter()
                .find(|node| node.id == endpoint_id && !is_acl_record(node))
                .is_some_and(|node| visible(node, "node"))
        })
    };
    let mut visible_edge_ids = edges
        .iter()
        .filter(|edge| !is_tombstoned(&edge.metadata) && edge_base_visible(edge))
        .map(|edge| edge.id.clone())
        .collect::<Vec<_>>();
    loop {
        let current = visible_edge_ids.iter().cloned().collect::<BTreeSet<_>>();
        let next = visible_edge_ids
            .iter()
            .filter(|edge_id| {
                let Some(edge) = edges.iter().find(|edge| edge.id == **edge_id) else {
                    return false;
                };
                let Some(source_edge_ids) = record_string_array_field(edge, "source_edge_ids")
                else {
                    return false;
                };
                let Some(target_edge_ids) = record_string_array_field(edge, "target_edge_ids")
                else {
                    return false;
                };
                source_edge_ids
                    .into_iter()
                    .chain(target_edge_ids)
                    .all(|endpoint_id| {
                        let Some(referenced) =
                            edges.iter().find(|candidate| candidate.id == endpoint_id)
                        else {
                            return false;
                        };
                        !is_tombstoned(&referenced.metadata) && current.contains(&referenced.id)
                    })
            })
            .cloned()
            .collect::<Vec<_>>();
        if next == visible_edge_ids {
            break;
        }
        visible_edge_ids = next;
    }
    let visible_edge_ids = visible_edge_ids.into_iter().collect::<BTreeSet<_>>();
    let visible_edges = edges
        .iter()
        .filter(|edge| visible_edge_ids.contains(&edge.id))
        .cloned()
        .collect();
    (visible_nodes, visible_edges)
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
    if limit == 0 {
        return Ok(items);
    }
    for expected_type in ["tool_call", "tool_result"] {
        for record in records {
            if is_tombstoned(&record.metadata)
                || value_string(record, "entity_type").as_deref() != Some(expected_type)
            {
                continue;
            }
            let item_conversation_id = value_string(record, "conversation_id");
            if conversation_id
                .is_some_and(|expected| item_conversation_id.as_deref() != Some(expected))
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
                entity_type: expected_type.to_owned(),
                tool_name,
                conversation_id: item_conversation_id,
                turn_index,
                summary,
                properties,
            };
            item.validate()?;
            items.push(item);
            if items.len() >= limit {
                return Ok(items);
            }
        }
    }
    Ok(items)
}

/// Project already-authorized conversation nodes into the shared transcript
/// contract. Stored projections require unique indexes; the wire DTO itself
/// permits ties to match Python's nondecreasing-index validation.
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
    if turns
        .windows(2)
        .any(|pair| pair[0].turn_index == pair[1].turn_index)
    {
        return Err(ConversationContractError::InvalidTurnOrder);
    }
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

/// Project the legacy conversation-list response from already-authorized
/// records. The caller must apply namespace and ACL filtering before calling.
pub fn conversation_list_for_user_from_visible_records(
    records: &[GraphRecord],
    user_id: &str,
) -> Vec<ConversationListEntry> {
    let mut accumulator = ConversationListAccumulator::new(user_id);
    accumulator.extend_visible_records(records);
    accumulator.finish()
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
        let order = ["run_step_seq", "snapshot_seq", "created_at_ms", "step_seq"]
            .into_iter()
            .find_map(|field| record_field(record, field).and_then(|value| value.as_i64()))
            .ok_or(ConversationContractError::MissingSnapshotOrder)?;
        let attempt = record_field(record, "attempt_seq")
            .and_then(|value| value.as_i64())
            .unwrap_or_default();
        candidates.push((order, attempt, record.id.clone(), record));
    }
    let Some((_, _, _, record)) = candidates.into_iter().max_by(|left, right| {
        left.0
            .cmp(&right.0)
            .then_with(|| left.1.cmp(&right.1))
            .then_with(|| left.2.cmp(&right.2))
    }) else {
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

/// Resolve a node ID to its active redirect target using a scoped graph snapshot.
///
/// This mirrors Python's `resolve_mode="redirect"`: active nodes resolve to
/// themselves, tombstones without a target and broken/cyclic chains resolve to
/// no node. Callers must still apply ACL filtering to the returned target.
pub fn resolve_graph_node_redirect_id(rid: &str, nodes: &[GraphRecord]) -> Option<String> {
    resolve_graph_entity_redirect_id(rid, nodes)
}

/// Resolve an edge ID to its active redirect target from an edge snapshot.
/// The supplied records must be edges; callers remain responsible for ACL checks.
pub fn resolve_graph_edge_redirect_id(rid: &str, edges: &[GraphRecord]) -> Option<String> {
    resolve_graph_entity_redirect_id(rid, edges)
}

fn resolve_graph_entity_redirect_id(rid: &str, records: &[GraphRecord]) -> Option<String> {
    let mut current = rid.trim().to_owned();
    if current.is_empty() {
        return None;
    }

    let mut visited = BTreeSet::new();
    loop {
        if !visited.insert(current.clone()) {
            return None;
        }
        let record = records.iter().find(|record| record.id == current)?;
        if let Some(target) = record
            .metadata
            .get("redirect_to_id")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|target| !target.is_empty())
        {
            current = target.to_owned();
            continue;
        }
        if is_tombstoned(&record.metadata) {
            return None;
        }
        return Some(current);
    }
}

/// Compute adjacency with the caller-selected node-to-opposite-node jump.
///
/// Python's `GraphQuery.k_hop` intentionally disables this jump so traversal
/// advances through explicit edge IDs one layer at a time. Keeping the mode
/// in the storage kernel prevents a native caller from silently changing the
/// graph distance semantics.
// PYTHON_PARITY_V1: rust=rust/crates/kogwistar-store/src/lib.rs#graph_neighbors_from_records_with_jump; python=kogwistar/graph_query.py#GraphQuery.neighbors; fixture=contracts/golden/graph-neighbors-cross-kind-id-v1.json; rust_test=rust/crates/kogwistar-store/src/lib.rs#graph_neighbors_preserve_cross_kind_same_id_from_shared_fixture; python_test=tests/core/test_edge_endpoint_rows.py#test_graph_query_sync_endpoint_merge_preserves_cross_kind_same_id
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
    let labels_match = |labels: &[&str], needle: Option<&str>| {
        needle.is_none_or(|needle| {
            let needle = needle.to_lowercase();
            labels
                .iter()
                .any(|label| label.to_lowercase().contains(&needle))
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
            if !labels_match(&source_labels, query.src_label_contains.as_deref())
                || !labels_match(&target_labels, query.tgt_label_contains.as_deref())
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

/// Project scope- and ACL-filtered graph records to the Python D3 MCP shape.
///
/// This is a serialization kernel only: callers own scope selection and ACL
/// filtering. Missing or inaccessible endpoints are omitted from links so a
/// visible edge cannot disclose identifiers for hidden entities.
fn graph_record_has_reference(
    record: &GraphRecord,
    insertion_method: &str,
    doc_id: Option<&str>,
) -> bool {
    let Some(document) = record_json(record) else {
        return false;
    };
    let references = document
        .get("references")
        .filter(|value| value.as_array().is_some_and(|items| !items.is_empty()))
        .or_else(|| document.get("mentions"))
        .and_then(Value::as_array);
    references.is_some_and(|references| {
        references.iter().any(|reference| {
            reference.get("insertion_method").and_then(Value::as_str) == Some(insertion_method)
                && doc_id.is_none_or(|expected_doc| {
                    reference.get("doc_id").and_then(Value::as_str) == Some(expected_doc)
                        || reference
                            .get("document_page_url")
                            .and_then(Value::as_str)
                            .is_some_and(|url| url.contains(expected_doc))
                })
        })
    })
}

fn graph_viz_selected_records<'a>(
    records: &'a [GraphRecord],
    doc_id: Option<&str>,
    insertion_method: Option<&str>,
) -> Vec<&'a GraphRecord> {
    // Python's `_ids_by_doc` treats an empty query value like an omitted filter.
    let selection_doc_id = doc_id.filter(|value| !value.is_empty());
    let base = records
        .iter()
        .filter(|record| {
            !is_tombstoned(&record.metadata)
                && selection_doc_id.is_none_or(|expected| {
                    record_field(record, "doc_id")
                        .and_then(|value| value.as_str().map(str::to_owned))
                        .as_deref()
                        == Some(expected)
                })
        })
        .collect::<Vec<_>>();
    // Python's `_filter_by_insertion_method` treats an empty query as no filter.
    let Some(expected_method) = insertion_method.filter(|value| !value.is_empty()) else {
        return base;
    };
    let matching = base
        .iter()
        .copied()
        .filter(|record| graph_record_has_reference(record, expected_method, selection_doc_id))
        .collect::<Vec<_>>();
    if matching.is_empty() { base } else { matching }
}

fn graph_viz_endpoint_complete_edges<'a>(
    nodes: &[GraphRecord],
    edges: &'a [GraphRecord],
    doc_id: Option<&str>,
    insertion_method: Option<&str>,
) -> Vec<&'a GraphRecord> {
    let node_ids = graph_viz_selected_records(nodes, doc_id, insertion_method)
        .into_iter()
        .map(|node| node.id.as_str())
        .collect::<BTreeSet<_>>();
    let candidates = graph_viz_selected_records(edges, doc_id, insertion_method);
    let mut visible_edge_ids = candidates
        .iter()
        .map(|edge| edge.id.as_str())
        .collect::<BTreeSet<_>>();

    loop {
        let next = candidates
            .iter()
            .filter(|edge| {
                ["source_ids", "target_ids"]
                    .into_iter()
                    .flat_map(|field| graph_record_string_list(edge, field))
                    .all(|endpoint_id| node_ids.contains(endpoint_id.as_str()))
                    && ["source_edge_ids", "target_edge_ids"]
                        .into_iter()
                        .flat_map(|field| graph_record_string_list(edge, field))
                        .all(|endpoint_id| visible_edge_ids.contains(endpoint_id.as_str()))
            })
            .map(|edge| edge.id.as_str())
            .collect::<BTreeSet<_>>();
        if next == visible_edge_ids {
            break;
        }
        visible_edge_ids = next;
    }

    candidates
        .into_iter()
        .filter(|edge| visible_edge_ids.contains(edge.id.as_str()))
        .collect()
}

pub fn graph_viz_d3_from_records(
    nodes: &[GraphRecord],
    edges: &[GraphRecord],
    doc_id: Option<&str>,
    mode: &str,
    insertion_method: Option<&str>,
) -> Value {
    let mut node_values = BTreeMap::<String, Value>::new();
    let mut node_order = Vec::<String>::new();
    let selected_nodes = graph_viz_selected_records(nodes, doc_id, insertion_method);
    for node in selected_nodes {
        let mut value = record_json(node)
            .filter(Value::is_object)
            .unwrap_or_else(|| Value::Object(Map::new()));
        let object = value.as_object_mut().expect("object was checked");
        object.insert("id".to_owned(), Value::String(node.id.clone()));
        object.insert(
            "label".to_owned(),
            record_field(node, "label")
                .filter(|value| !value.is_null())
                .unwrap_or_else(|| Value::String(node.id.clone())),
        );
        object.insert("type".to_owned(), Value::String("entity".to_owned()));
        if let Some(summary) = record_field(node, "summary") {
            object.insert("summary".to_owned(), summary);
        }
        object.insert(
            "properties".to_owned(),
            record_field(node, "properties")
                .filter(Value::is_object)
                .unwrap_or_else(|| Value::Object(Map::new())),
        );
        if !node_values.contains_key(&node.id) {
            node_order.push(node.id.clone());
        }
        node_values.insert(node.id.clone(), value);
    }
    let node_ids = node_values.keys().cloned().collect::<BTreeSet<_>>();
    let visible_edges = graph_viz_endpoint_complete_edges(nodes, edges, doc_id, insertion_method);
    let edge_ids = visible_edges
        .iter()
        .map(|record| record.id.clone())
        .collect::<BTreeSet<_>>();
    let reify = mode.eq_ignore_ascii_case("reify");
    let mut links = Vec::new();

    for edge in &visible_edges {
        let relation = graph_visualization_edge_relation(edge);
        let edge_label = relation
            .clone()
            .unwrap_or_else(|| Value::String("edge".to_owned()));
        let properties = record_field(edge, "properties")
            .filter(Value::is_object)
            .unwrap_or_else(|| Value::Object(Map::new()));
        let sources = graph_record_string_list(edge, "source_ids");
        let targets = graph_record_string_list(edge, "target_ids");
        let source_edges = graph_record_string_list(edge, "source_edge_ids");
        let target_edges = graph_record_string_list(edge, "target_edge_ids");

        if reify {
            if !node_values.contains_key(&edge.id) {
                let mut edge_node = record_json(edge)
                    .filter(Value::is_object)
                    .unwrap_or_else(|| Value::Object(Map::new()));
                let object = edge_node.as_object_mut().expect("object was checked");
                object.insert("id".to_owned(), Value::String(edge.id.clone()));
                object.insert("label".to_owned(), edge_label);
                object.insert("type".to_owned(), Value::String("edge-node".to_owned()));
                object.insert(
                    "summary".to_owned(),
                    record_field(edge, "summary").unwrap_or(Value::Null),
                );
                object.insert("properties".to_owned(), properties.clone());
                node_order.push(edge.id.clone());
                node_values.insert(edge.id.clone(), edge_node);
            }
            for source in sources.iter().filter(|id| node_ids.contains(*id)) {
                links.push(json!({"source": source, "target": edge.id, "relation": relation, "role": "src", "properties": properties}));
            }
            for source in source_edges.iter().filter(|id| edge_ids.contains(*id)) {
                links.push(json!({"source": source, "target": edge.id, "relation": relation, "role": "src", "properties": properties}));
            }
            for target in targets.iter().filter(|id| node_ids.contains(*id)) {
                links.push(json!({"source": edge.id, "target": target, "relation": relation, "role": "tgt", "properties": properties}));
            }
            for target in target_edges.iter().filter(|id| edge_ids.contains(*id)) {
                links.push(json!({"source": edge.id, "target": target, "relation": relation, "role": "tgt", "properties": properties}));
            }
        } else {
            for source in sources.iter().filter(|id| node_ids.contains(*id)) {
                for target in targets.iter().filter(|id| node_ids.contains(*id)) {
                    links.push(json!({"source": source, "target": target, "relation": relation, "properties": properties}));
                }
            }
        }
    }
    json!({
        "nodes": node_order.into_iter().filter_map(|id| node_values.remove(&id)).collect::<Vec<_>>(),
        "links": links,
        "mode": mode,
        "doc_id": doc_id,
    })
}

/// Project the D3 projection into Python Cytoscape's elements envelope.
pub fn graph_viz_cytoscape_from_records(
    nodes: &[GraphRecord],
    edges: &[GraphRecord],
    doc_id: Option<&str>,
    mode: &str,
    insertion_method: Option<&str>,
) -> Value {
    let d3 = graph_viz_d3_from_records(nodes, edges, doc_id, mode, insertion_method);
    let mut elements = Vec::new();
    let visible_node_ids = d3["nodes"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|node| node["type"] == "entity")
        .filter_map(|node| node["id"].as_str().map(str::to_owned))
        .collect::<BTreeSet<_>>();
    let selected_edges = graph_viz_endpoint_complete_edges(nodes, edges, doc_id, insertion_method);
    let visible_edge_ids = selected_edges
        .iter()
        .map(|edge| edge.id.clone())
        .collect::<BTreeSet<_>>();
    let d3_nodes = d3["nodes"]
        .as_array()
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    for node in d3_nodes.iter().filter(|node| node["type"] == "entity") {
        let data = if node["type"] == "entity" {
            json!({"id": node["id"], "label": node["label"], "type": "entity"})
        } else {
            json!({"id": node["id"], "label": node["label"], "type": "edge-node"})
        };
        elements.push(json!({
            "data": data,
            "classes": if node["type"] == "edge-node" { "edge-node" } else { "" },
        }));
    }
    if mode.eq_ignore_ascii_case("reify") {
        let links = d3["links"].as_array().into_iter().flatten().fold(
            BTreeMap::<String, Vec<&Value>>::new(),
            |mut grouped, link| {
                let role = link["role"].as_str().unwrap_or_default();
                let edge_id = if role == "src" {
                    link["target"].as_str().unwrap_or_default()
                } else {
                    link["source"].as_str().unwrap_or_default()
                };
                grouped.entry(edge_id.to_owned()).or_default().push(link);
                grouped
            },
        );
        for edge in &selected_edges {
            if let Some(node) = d3_nodes
                .iter()
                .find(|node| node["type"] == "edge-node" && node["id"] == edge.id)
            {
                elements.push(json!({
                    "data": {"id": node["id"], "label": node["label"], "type": "edge-node"},
                    "classes": "edge-node",
                }));
            }
            for link in links.get(&edge.id).into_iter().flatten() {
                let role = link["role"].as_str().unwrap_or_default();
                let edge_id = if role == "src" {
                    link["target"].as_str().unwrap_or_default()
                } else {
                    link["source"].as_str().unwrap_or_default()
                };
                let endpoint_id = if role == "src" {
                    link["source"].as_str().unwrap_or_default()
                } else {
                    link["target"].as_str().unwrap_or_default()
                };
                let endpoint_is_edge = visible_edge_ids.contains(endpoint_id);
                let endpoint_role = match (role, endpoint_is_edge) {
                    ("src", true) => "srcE",
                    ("src", false) => "src",
                    (_, true) => "tgtE",
                    (_, false) => "tgt",
                };
                elements.push(json!({
                    "data": {
                        "id": format!("{edge_id}::{endpoint_role}::{endpoint_id}"),
                        "source": link["source"],
                        "target": link["target"],
                        "label": link["relation"],
                    },
                    "classes": role,
                }));
            }
        }
    } else {
        for edge in &selected_edges {
            let relation = graph_visualization_edge_relation(edge);
            for source in graph_record_string_list(edge, "source_ids")
                .iter()
                .filter(|id| visible_node_ids.contains(*id))
            {
                for target in graph_record_string_list(edge, "target_ids")
                    .iter()
                    .filter(|id| visible_node_ids.contains(*id))
                {
                    elements.push(json!({
                        "data": {
                            "id": format!("{}::{}->{}", edge.id, source, target),
                            "source": source,
                            "target": target,
                            "label": relation,
                        },
                        "classes": "",
                    }));
                }
            }
        }
    }
    json!({"elements": elements, "mode": mode, "doc_id": doc_id})
}

fn graph_record_string_list(record: &GraphRecord, field: &str) -> Vec<String> {
    let value = record_field(record, field).and_then(|value| match value {
        Value::String(encoded) => serde_json::from_str::<Value>(&encoded).ok(),
        value => Some(value),
    });
    value
        .and_then(|value| value.as_array().cloned())
        .unwrap_or_default()
        .into_iter()
        .filter_map(|value| value.as_str().map(str::to_owned))
        .filter(|value| !value.is_empty())
        .collect()
}

fn is_tombstoned(metadata: &Map<String, Value>) -> bool {
    metadata
        .get("lifecycle_status")
        .and_then(Value::as_str)
        .is_some_and(|status| matches!(status, "tombstoned" | "deleted"))
}

/// Storage scope carried by the graph projection. Namespace, workspace, and
/// graph-space are storage coordinates, not graph roles or entity subtypes.
/// Python's `kg_graph_type` identifies the graph role and supplies the default
/// ACL `truth_graph`; it is deliberately independent of this storage scope.
/// The Python pgvector schema keeps graph rows in shared tables, so these scope
/// values are represented by exact JSON metadata predicates rather than by a
/// Rust-owned table or schema. ACL truth-graph selection is policy context and
/// is carried separately by the ACL-aware mutation contract.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct GraphScope {
    pub namespace: String,
    pub workspace_id: Option<String>,
    pub graph_space: Option<String>,
}

impl GraphScope {
    /// Reject metadata that claims a storage-scope dimension absent from this
    /// request, so backends cannot accidentally widen a scoped record.
    pub fn validate_metadata_scope(&self, metadata: &Map<String, Value>) -> StoreResult<()> {
        for (key, expected) in [
            ("namespace", Some(self.namespace.as_str())),
            ("workspace_id", self.workspace_id.as_deref()),
            ("graph_space", self.graph_space.as_deref()),
        ] {
            match expected {
                Some(expected)
                    if metadata
                        .get(key)
                        .is_some_and(|actual| actual.as_str() != Some(expected)) =>
                {
                    return Err(StoreError::InvalidEntityEventPayload {
                        message: format!(
                            "graph metadata scope {key:?} conflicts with request scope"
                        ),
                    });
                }
                None if metadata.get(key).is_some_and(|actual| !actual.is_null()) => {
                    return Err(StoreError::InvalidEntityEventPayload {
                        message: format!(
                            "graph metadata scope {key:?} is absent from request scope"
                        ),
                    });
                }
                _ => {}
            }
        }
        Ok(())
    }
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
    mutation
        .scope
        .validate_metadata_scope(&mutation.record.metadata)?;
    if let Some(metadata) = mutation.payload.get("metadata") {
        let metadata =
            metadata
                .as_object()
                .ok_or_else(|| StoreError::InvalidEntityEventPayload {
                    message: "graph mutation payload metadata must be an object".to_owned(),
                })?;
        mutation.scope.validate_metadata_scope(metadata)?;
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

/// Ensure an event-ID retry is the same graph effect before treating it as
/// idempotent. Raw event stores intentionally return the first event for a
/// repeated ID; graph mutation owners must apply this stricter comparison.
pub fn validate_graph_mutation_replay(
    mutation: &GraphMutation,
    existing: &EntityEvent,
) -> StoreResult<()> {
    if existing.namespace != mutation.scope.namespace
        || existing.event_id != mutation.event_id
        || existing.entity_kind != mutation.entity_kind
        || existing.entity_id != mutation.record.id
        || existing.op != mutation.op
        || existing.payload != mutation.payload
    {
        return Err(StoreError::InvalidEntityEventPayload {
            message: format!(
                "graph mutation event id {:?} was reused with different input",
                mutation.event_id
            ),
        });
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
        if !(0..=i32::MAX as i64).contains(&job.max_retries) {
            return Err(StoreError::InvalidEntityEventPayload {
                message: "index job max_retries must fit a non-negative PostgreSQL INTEGER"
                    .to_owned(),
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
    let persistent_entity = match envelope.model_class {
        GraphModelClass::Node | GraphModelClass::Edge => {
            Some(envelope.decode_persistent_v1().map_err(|error| {
                StoreError::InvalidEntityEventPayload {
                    message: error.to_string(),
                }
            })?)
        }
        _ => {
            envelope.validate_family_v1().map_err(|error| {
                StoreError::InvalidEntityEventPayload {
                    message: error.to_string(),
                }
            })?;
            None
        }
    };
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
    let entity_id = match persistent_entity {
        Some(entity) => match entity {
            PersistentGraphEntityV1::Node(node) => node.core.id,
            PersistentGraphEntityV1::Edge(edge) => edge.core.id,
        },
        None => {
            match envelope
                .inspect()
                .map_err(|error| StoreError::InvalidEntityEventPayload {
                    message: error.to_string(),
                })? {
                kogwistar_contracts::TypedGraphEntity::Node(node) => node.common.id,
                kogwistar_contracts::TypedGraphEntity::Edge(edge) => edge.common.id,
            }
        }
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

/// Keyset-paginated graph projection read. Existing `GraphProjectionRead`
/// remains unchanged for source compatibility.
#[derive(Clone, Debug, PartialEq)]
pub struct GraphProjectionPageRead {
    pub read: GraphProjectionRead,
    pub after_id: Option<String>,
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
    /// Read one keyset page. The default fails closed so older store
    /// implementations remain source-compatible without pretending to page.
    fn graph_projection_page(
        &self,
        _read: GraphProjectionPageRead,
    ) -> impl Future<Output = StoreResult<Vec<GraphRecord>>> + Send {
        async {
            Err(StoreError::Backend {
                backend: "graph-projection-pagination".to_owned(),
                message: "keyset pagination is not supported by this store".to_owned(),
            })
        }
    }
    fn graph_projection_vector_query(
        &self,
        query: GraphProjectionVectorQuery,
    ) -> impl Future<Output = StoreResult<Vec<VectorMatch>>> + Send;
}

/// Visit graph projection pages in stable ID order without retaining all rows.
///
/// The helper rejects backends that return unordered or non-advancing pages,
/// preventing silent omissions and infinite loops at language/backend seams.
pub async fn for_each_graph_projection_page<S, F>(
    store: &S,
    read: GraphProjectionRead,
    mut visit: F,
) -> StoreResult<()>
where
    S: GraphMutationStore + Sync,
    F: FnMut(&[GraphRecord]) -> StoreResult<()>,
{
    let page_size = read.limit;
    if page_size == 0 {
        return Ok(());
    }

    let mut after_id = None;
    loop {
        let page = store
            .graph_projection_page(GraphProjectionPageRead {
                read: read.clone(),
                after_id: after_id.clone(),
            })
            .await?;
        let Some(first) = page.first() else {
            break;
        };
        if after_id.as_ref().is_some_and(|cursor| first.id <= *cursor)
            || page.windows(2).any(|pair| pair[0].id >= pair[1].id)
            || page.len() > page_size
        {
            return Err(StoreError::Backend {
                backend: "graph-projection-pagination".to_owned(),
                message: "backend returned a non-advancing or unordered page".to_owned(),
            });
        }
        let has_more = page.len() == page_size;
        after_id = page.last().map(|record| record.id.clone());
        visit(&page)?;
        if !has_more {
            break;
        }
    }
    Ok(())
}

/// Read every page into one vector. Prefer `for_each_graph_projection_page`
/// when the consumer can reduce records incrementally.
pub async fn graph_projection_records_all<S>(
    store: &S,
    read: GraphProjectionRead,
) -> StoreResult<Vec<GraphRecord>>
where
    S: GraphMutationStore + Sync,
{
    let mut records = Vec::new();
    for_each_graph_projection_page(store, read, |page| {
        records.extend_from_slice(page);
        Ok(())
    })
    .await?;
    Ok(records)
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
    fn supports_document_projection_lookup(&self) -> bool {
        false
    }

    fn document_projection_ids(
        &self,
        _scope: GraphScope,
        _ids: Vec<String>,
    ) -> impl Future<Output = StoreResult<Vec<String>>> + Send {
        async {
            Err(StoreError::Backend {
                backend: "document".to_owned(),
                message: "document projection lookup is not supported by this store".to_owned(),
            })
        }
    }

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

/// Convert a standardized lower-is-better distance to a higher-is-better score.
///
/// Store adapters normalize backend-specific encodings before returning a
/// [`VectorMatch`]: cosine distance remains `1 - cosine`, L2 is Euclidean
/// distance, and inner product is represented as negative dot product. This
/// keeps score policy independent of the selected storage backend.
pub fn similarity_from_distance(distance: Option<f64>, metric: DistanceMetric) -> Option<f64> {
    let distance = distance.filter(|value| value.is_finite())?;
    Some(match metric {
        DistanceMetric::Cosine => 1.0 - distance,
        DistanceMetric::L2 => 1.0 / (1.0 + distance.max(0.0)),
        DistanceMetric::InnerProduct => -distance,
    })
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

/// Lossless event archive input shared by metadata backends. `namespace` is
/// supplied separately to preserve the established SQLite restore API.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RestoredRawEntityEvent {
    pub seq: i64,
    pub event_id: String,
    pub entity_kind: String,
    pub entity_id: String,
    pub op: String,
    pub payload_json: String,
    pub created_at: i64,
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

/// Minimal queue context passed to result-provider callbacks.
///
/// Unlike [`IndexJob`], this type intentionally carries no lease token,
/// acknowledgement state, retry counters, or accepted-result fields. Queue
/// authority stays with the worker while providers receive only work context.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct IndexJobProviderInput {
    pub schema_version: u32,
    pub job_id: String,
    pub namespace: String,
    pub entity_kind: String,
    pub entity_id: String,
    pub index_kind: String,
    pub coalesce_key: String,
    pub op: String,
    pub payload_json: Option<String>,
}

impl From<&IndexJob> for IndexJobProviderInput {
    fn from(job: &IndexJob) -> Self {
        Self {
            schema_version: 1,
            job_id: job.job_id.clone(),
            namespace: job.namespace.clone(),
            entity_kind: job.entity_kind.clone(),
            entity_id: job.entity_id.clone(),
            index_kind: job.index_kind.clone(),
            coalesce_key: job.coalesce_key.clone(),
            op: job.op.clone(),
            payload_json: job.payload_json.clone(),
        }
    }
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

/// Restricts a queue claim to explicitly supported job kinds. Implementations
/// must apply this filter inside the same atomic claim operation; claiming
/// broadly and returning unrelated work is not equivalent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexJobClaimFilter {
    pub namespace: Option<String>,
    pub index_kinds: Vec<String>,
}

/// Durable index-job state machine. `None` token intentionally retains the
/// legacy Python administrative behavior. A supplied token authorizes queue
/// mutations only while it matches the current claim and its lease is live.
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
    /// Atomically claim only jobs matching the supplied index kinds.
    ///
    /// The default fails closed so older stores do not accidentally lease work
    /// they cannot execute.
    fn claim_index_jobs_filtered(
        &self,
        _limit: usize,
        _lease_seconds: i64,
        _filter: IndexJobClaimFilter,
    ) -> impl Future<Output = StoreResult<Vec<IndexJob>>> + Send {
        async {
            Err(StoreError::Backend {
                backend: "index-job-queue".to_owned(),
                message: "filtered index-job claims are not supported by this store".to_owned(),
            })
        }
    }
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
    /// Read by both namespace and entity kind. The default fails closed so
    /// stores with an ambiguous legacy identity model cannot claim parity.
    fn graph_record_of_kind(
        &self,
        _namespace: &str,
        _kind: GraphEntityKind,
        _id: &str,
    ) -> impl Future<Output = StoreResult<Option<GraphRecord>>> + Send {
        async {
            Err(StoreError::Backend {
                backend: "graph-read".to_owned(),
                message: "kind-aware graph record reads are not supported by this store".to_owned(),
            })
        }
    }
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
    /// Read a graph entity kind without mixing nodes and edges.
    fn graph_records_of_kind(
        &self,
        _namespace: &str,
        _kind: GraphEntityKind,
        _metadata: &MetadataFilter,
    ) -> impl Future<Output = StoreResult<Vec<GraphRecord>>> + Send {
        async {
            Err(StoreError::Backend {
                backend: "graph-read".to_owned(),
                message: "kind-aware graph record listing is not supported by this store"
                    .to_owned(),
            })
        }
    }
    /// Read a bounded set of records by typed identity. Missing IDs are
    /// omitted, duplicates are returned once, and result order follows the
    /// first occurrence in `ids`. The default preserves compatibility for
    /// stores that only implement single-record reads; backends can override
    /// it with one native batch request.
    fn graph_records_by_ids_of_kind(
        &self,
        namespace: &str,
        kind: GraphEntityKind,
        ids: &[String],
    ) -> impl Future<Output = StoreResult<Vec<GraphRecord>>> + Send {
        let namespace = namespace.to_owned();
        let ids = ids.to_vec();
        async move {
            if ids.len() > 10_000 {
                return Err(StoreError::Backend {
                    backend: "graph-read".to_owned(),
                    message: "typed ID batch exceeds the 10000-record limit".to_owned(),
                });
            }
            let mut seen = BTreeSet::new();
            let mut records = Vec::with_capacity(ids.len());
            for id in ids {
                if id.trim().is_empty() {
                    return Err(StoreError::InvalidIdentifier {
                        identifier: "graph record id",
                    });
                }
                if !seen.insert(id.clone()) {
                    continue;
                }
                if let Some(record) = self.graph_record_of_kind(&namespace, kind, &id).await? {
                    records.push(record);
                }
            }
            Ok(records)
        }
    }
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
    /// Run vector search within one graph entity kind.
    fn vector_query_of_kind(
        &self,
        _namespace: &str,
        _kind: GraphEntityKind,
        _query: &VectorQuery,
    ) -> impl Future<Output = StoreResult<Vec<VectorMatch>>> + Send {
        async {
            Err(StoreError::Backend {
                backend: "graph-read".to_owned(),
                message: "kind-aware vector search is not supported by this store".to_owned(),
            })
        }
    }
    /// Run a typed vector query with an inclusive numeric metadata upper bound.
    /// The default fails closed rather than applying the bound after top-N,
    /// which could silently change the candidate set.
    fn vector_query_of_kind_numeric_lte(
        &self,
        _namespace: &str,
        _kind: GraphEntityKind,
        _query: &VectorQuery,
        _field: &str,
        _maximum: f64,
    ) -> impl Future<Output = StoreResult<Vec<VectorMatch>>> + Send {
        async {
            Err(StoreError::Backend {
                backend: "graph-read".to_owned(),
                message: "numeric-bounded vector search is not supported by this store".to_owned(),
            })
        }
    }
}

/// An independently named vector collection, separate from canonical graph
/// entities. Derived indexes such as Python's `node_index` must not be stored
/// as fake Nodes or Edges: graph reads, ACL traversal, and lifecycle writes
/// treat those entity kinds as authoritative graph data.
pub trait VectorCollectionStore: Send + Sync {
    /// Upsert one bounded batch within one namespace and collection. The
    /// backend's transaction/eventual-consistency guarantees remain explicit;
    /// this method does not imply graph-mutation atomicity.
    fn upsert_vector_collection(
        &self,
        namespace: &str,
        collection: &str,
        records: Vec<GraphRecord>,
    ) -> impl Future<Output = StoreResult<()>> + Send;

    /// Delete the requested IDs from only the selected namespace/collection.
    fn delete_vector_collection_records(
        &self,
        namespace: &str,
        collection: &str,
        ids: Vec<String>,
    ) -> impl Future<Output = StoreResult<()>> + Send;

    /// Query only the selected namespace/collection using the shared distance
    /// contract and stable backend ordering.
    fn query_vector_collection(
        &self,
        namespace: &str,
        collection: &str,
        query: &VectorQuery,
    ) -> impl Future<Output = StoreResult<Vec<VectorMatch>>> + Send;
}

/// Backend-native access to Python's dedicated `node_index` projection.
///
/// This is intentionally separate from `VectorCollectionStore`: PostgreSQL
/// stores this projection in the graph-schema-local `gke_node_index` table,
/// while Chroma and memory use their dedicated index collections. `index`
/// names that backend-native target; it is not a graph entity kind and writes
/// through this contract do not imply canonical graph mutation atomicity.
pub trait NodeIndexStore: Send + Sync {
    fn upsert_node_index(
        &self,
        index: &str,
        records: Vec<GraphRecord>,
    ) -> impl Future<Output = StoreResult<()>> + Send;

    fn get_node_index(
        &self,
        index: &str,
        ids: &[String],
    ) -> impl Future<Output = StoreResult<Vec<GraphRecord>>> + Send;

    fn query_node_index(
        &self,
        index: &str,
        query: &VectorQuery,
    ) -> impl Future<Output = StoreResult<Vec<VectorMatch>>> + Send;
}

pub trait GraphWriteStore: GraphReadStore {
    fn upsert_graph_record(
        &self,
        namespace: &str,
        record: GraphRecord,
    ) -> impl Future<Output = StoreResult<()>> + Send;
    /// Write a graph record with an explicit entity kind.
    fn upsert_graph_record_of_kind(
        &self,
        namespace: &str,
        _kind: GraphEntityKind,
        record: GraphRecord,
    ) -> impl Future<Output = StoreResult<()>> + Send {
        async move { self.upsert_graph_record(namespace, record).await }
    }
    fn delete_graph_record(
        &self,
        namespace: &str,
        id: &str,
    ) -> impl Future<Output = StoreResult<bool>> + Send;
    /// Delete by namespace, entity kind, and ID to avoid cross-kind deletion.
    fn delete_graph_record_of_kind(
        &self,
        namespace: &str,
        _kind: GraphEntityKind,
        id: &str,
    ) -> impl Future<Output = StoreResult<bool>> + Send {
        async move { self.delete_graph_record(namespace, id).await }
    }
}

/// Minimal materialized-projection mutation seam for durable index workers.
///
/// The worker does not query graph state. Requiring `GraphWriteStore` also
/// requires unrelated graph reads and prevents native projection stores from
/// participating without pretending to implement those queries.
pub trait GraphProjectionWriteStore: Send + Sync {
    fn upsert_graph_projection_record(
        &self,
        namespace: &str,
        entity_kind: &str,
        record: GraphRecord,
    ) -> impl Future<Output = StoreResult<()>> + Send;

    fn delete_graph_projection_record(
        &self,
        namespace: &str,
        entity_kind: &str,
        id: &str,
    ) -> impl Future<Output = StoreResult<bool>> + Send;

    /// Report whether an accepted projection action changed the materialized
    /// store or was safely superseded by a newer canonical revision. Existing
    /// implementations remain source-compatible and default to `Applied`.
    fn upsert_graph_projection_record_with_outcome(
        &self,
        namespace: &str,
        entity_kind: &str,
        record: GraphRecord,
    ) -> impl Future<Output = StoreResult<ProjectionWriteOutcome>> + Send {
        async move {
            self.upsert_graph_projection_record(namespace, entity_kind, record)
                .await?;
            Ok(ProjectionWriteOutcome::Applied)
        }
    }

    fn delete_graph_projection_record_with_outcome(
        &self,
        namespace: &str,
        entity_kind: &str,
        id: &str,
    ) -> impl Future<Output = StoreResult<ProjectionWriteOutcome>> + Send {
        async move {
            self.delete_graph_projection_record(namespace, entity_kind, id)
                .await?;
            Ok(ProjectionWriteOutcome::Applied)
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProjectionWriteOutcome {
    Applied,
    Superseded,
}

impl<T: GraphWriteStore> GraphProjectionWriteStore for T {
    // Preserve the trait's explicit opaque-future + Send contract here.
    #[allow(clippy::manual_async_fn)]
    fn upsert_graph_projection_record(
        &self,
        namespace: &str,
        entity_kind: &str,
        record: GraphRecord,
    ) -> impl Future<Output = StoreResult<()>> + Send {
        async move {
            let kind = match entity_kind {
                "node" => GraphEntityKind::Node,
                "edge" => GraphEntityKind::Edge,
                _ => {
                    return Err(StoreError::Backend {
                        backend: "graph-projection".to_owned(),
                        message: "graph projection entity kind must be 'node' or 'edge'".to_owned(),
                    });
                }
            };
            self.upsert_graph_record_of_kind(namespace, kind, record)
                .await
        }
    }

    #[allow(clippy::manual_async_fn)]
    fn delete_graph_projection_record(
        &self,
        namespace: &str,
        entity_kind: &str,
        id: &str,
    ) -> impl Future<Output = StoreResult<bool>> + Send {
        async move {
            let kind = match entity_kind {
                "node" => GraphEntityKind::Node,
                "edge" => GraphEntityKind::Edge,
                _ => {
                    return Err(StoreError::Backend {
                        backend: "graph-projection".to_owned(),
                        message: "graph projection entity kind must be 'node' or 'edge'".to_owned(),
                    });
                }
            };
            self.delete_graph_record_of_kind(namespace, kind, id).await
        }
    }
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
    async fn vector_query_of_kind_numeric_lte(
        &self,
        ns: &str,
        kind: GraphEntityKind,
        query: &VectorQuery,
        field: &str,
        maximum: f64,
    ) -> StoreResult<Vec<VectorMatch>> {
        self.inner
            .vector_query_of_kind_numeric_lte(ns, kind, query, field, maximum)
            .await
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

    #[test]
    fn graph_scope_metadata_rejects_unrequested_workspace_and_graph_space() {
        let scope = GraphScope {
            namespace: "tenant-a".to_owned(),
            workspace_id: None,
            graph_space: None,
        };
        assert!(
            scope
                .validate_metadata_scope(&Map::from_iter([(
                    "workspace_id".to_owned(),
                    Value::String("workspace-a".to_owned()),
                )]))
                .is_err()
        );
        assert!(
            scope
                .validate_metadata_scope(&Map::from_iter([(
                    "graph_space".to_owned(),
                    Value::String("conversation".to_owned()),
                )]))
                .is_err()
        );
        assert!(
            scope
                .validate_metadata_scope(&Map::from_iter([(
                    "workspace_id".to_owned(),
                    Value::Null,
                )]))
                .is_ok()
        );
    }

    #[test]
    fn standardized_vector_distance_scores_are_backend_neutral() {
        assert_eq!(
            similarity_from_distance(Some(0.25), DistanceMetric::Cosine),
            Some(0.75)
        );
        assert_eq!(
            similarity_from_distance(Some(3.0), DistanceMetric::L2),
            Some(0.25)
        );
        assert_eq!(
            similarity_from_distance(Some(-0.8), DistanceMetric::InnerProduct),
            Some(0.8)
        );
        assert_eq!(similarity_from_distance(None, DistanceMetric::Cosine), None);
        assert_eq!(
            similarity_from_distance(Some(f64::NAN), DistanceMetric::Cosine),
            None
        );
    }

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

    // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-store/src/lib.rs#graph_viz_d3_from_records; python=kogwistar/visualization/graph_viz.py#to_d3_force; fixture=contracts/golden/python-graph-model-payloads-v1.json; rust_test=rust/crates/kogwistar-store/src/lib.rs#graph_visualization_model_payloads_match_python_fixture; python_test=tests/test_visualization.py#test_graph_visualization_payloads_match_shared_python_model_fixture
    #[test]
    fn graph_visualization_model_payloads_match_python_fixture() {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/python-graph-model-payloads-v1.json"
        )))
        .expect("Python graph-model fixture is valid JSON");
        let payload = |kind: &str| {
            fixture["cases"]
                .as_array()
                .expect("fixture cases")
                .iter()
                .find(|case| case["kind"] == kind)
                .expect("fixture graph model")["payload"]
                .clone()
        };
        let source = payload("node");
        let mut target = source.clone();
        target["id"] = json!("parity-target");
        target["label"] = json!("Target node");
        let mut edge = payload("edge");
        edge["source_ids"] = json!([source["id"]]);
        edge["target_ids"] = json!([target["id"]]);

        let record = |payload: &Value| GraphRecord {
            id: payload["id"].as_str().expect("model ID").to_owned(),
            document: Some(payload.to_string()),
            metadata: payload["metadata"].as_object().cloned().unwrap_or_default(),
            embedding: None,
        };
        let nodes = vec![record(&source), record(&target)];
        let edges = vec![record(&edge)];
        let d3 = graph_viz_d3_from_records(&nodes, &edges, None, "reify", None);

        let mut expected_nodes = vec![source.clone(), target.clone(), edge.clone()];
        for node in &mut expected_nodes[..2] {
            node["type"] = json!("entity");
            node["properties"] = node["properties"]
                .as_object()
                .cloned()
                .map(Value::Object)
                .unwrap_or_else(|| json!({}));
        }
        expected_nodes[2]["type"] = json!("edge-node");
        expected_nodes[2]["label"] = edge["relation"].clone();
        expected_nodes.sort_by_key(|node| node["id"].as_str().unwrap_or_default().to_owned());
        let mut actual_nodes = d3["nodes"].as_array().unwrap().clone();
        actual_nodes.sort_by_key(|node| node["id"].as_str().unwrap_or_default().to_owned());
        assert_eq!(actual_nodes, expected_nodes);

        let mut actual_links = d3["links"].as_array().unwrap().clone();
        actual_links.sort_by_key(|link| {
            (
                link["source"].as_str().unwrap_or_default().to_owned(),
                link["target"].as_str().unwrap_or_default().to_owned(),
            )
        });
        let mut expected_links = vec![
            json!({"source":source["id"],"target":edge["id"],"relation":edge["relation"],"role":"src","properties":edge["properties"]}),
            json!({"source":edge["id"],"target":target["id"],"relation":edge["relation"],"role":"tgt","properties":edge["properties"]}),
        ];
        expected_links.sort_by_key(|link| {
            (
                link["source"].as_str().unwrap_or_default().to_owned(),
                link["target"].as_str().unwrap_or_default().to_owned(),
            )
        });
        assert_eq!(actual_links, expected_links);

        let classic_d3 = graph_viz_d3_from_records(&nodes, &edges, None, "classic", None);
        let mut expected_classic_nodes = vec![source.clone(), target.clone()];
        for node in &mut expected_classic_nodes {
            node["type"] = json!("entity");
            node["properties"] = node["properties"]
                .as_object()
                .cloned()
                .map(Value::Object)
                .unwrap_or_else(|| json!({}));
        }
        expected_classic_nodes
            .sort_by_key(|node| node["id"].as_str().unwrap_or_default().to_owned());
        let mut actual_classic_nodes = classic_d3["nodes"].as_array().unwrap().clone();
        actual_classic_nodes.sort_by_key(|node| node["id"].as_str().unwrap_or_default().to_owned());
        assert_eq!(actual_classic_nodes, expected_classic_nodes);
        assert_eq!(
            classic_d3["links"],
            json!([{
                "source": source["id"],
                "target": target["id"],
                "relation": edge["relation"],
                "properties": edge["properties"]
            }])
        );

        let cytoscape = graph_viz_cytoscape_from_records(&nodes, &edges, None, "reify", None);
        let mut expected_elements = vec![
            json!({"data":{"id":source["id"],"label":source["label"],"type":"entity"},"classes":""}),
            json!({"data":{"id":target["id"],"label":target["label"],"type":"entity"},"classes":""}),
            json!({"data":{"id":edge["id"],"label":edge["relation"],"type":"edge-node"},"classes":"edge-node"}),
            json!({"data":{"id":format!("{}::src::{}",edge["id"].as_str().unwrap(),source["id"].as_str().unwrap()),"source":source["id"],"target":edge["id"],"label":edge["relation"]},"classes":"src"}),
            json!({"data":{"id":format!("{}::tgt::{}",edge["id"].as_str().unwrap(),target["id"].as_str().unwrap()),"source":edge["id"],"target":target["id"],"label":edge["relation"]},"classes":"tgt"}),
        ];
        expected_elements.sort_by_key(|element| {
            element["data"]["id"]
                .as_str()
                .unwrap_or_default()
                .to_owned()
        });
        let mut actual_elements = cytoscape["elements"].as_array().unwrap().clone();
        actual_elements.sort_by_key(|element| {
            element["data"]["id"]
                .as_str()
                .unwrap_or_default()
                .to_owned()
        });
        assert_eq!(actual_elements, expected_elements);
        assert_eq!(cytoscape["mode"], "reify");
        assert!(cytoscape["doc_id"].is_null());

        let classic_cytoscape =
            graph_viz_cytoscape_from_records(&nodes, &edges, None, "classic", None);
        assert_eq!(classic_cytoscape["elements"].as_array().unwrap().len(), 3);
        let expected_classic_edge_id = format!(
            "{}::{}->{}",
            edge["id"].as_str().unwrap(),
            source["id"].as_str().unwrap(),
            target["id"].as_str().unwrap()
        );
        assert!(
            classic_cytoscape["elements"]
                .as_array()
                .unwrap()
                .iter()
                .any(|element| {
                    element["data"]["id"].as_str() == Some(expected_classic_edge_id.as_str())
                })
        );
        assert_eq!(classic_cytoscape["mode"], "classic");

        let incomplete_edge = record(&payload("edge"));
        let incomplete_nodes = vec![record(&source)];
        let incomplete_edges = vec![incomplete_edge];
        let incomplete_d3 =
            graph_viz_d3_from_records(&incomplete_nodes, &incomplete_edges, None, "reify", None);
        assert_eq!(incomplete_d3["nodes"].as_array().unwrap().len(), 1);
        assert!(incomplete_d3["links"].as_array().unwrap().is_empty());
        let incomplete_cytoscape = graph_viz_cytoscape_from_records(
            &incomplete_nodes,
            &incomplete_edges,
            None,
            "reify",
            None,
        );
        assert_eq!(
            incomplete_cytoscape["elements"].as_array().unwrap().len(),
            1
        );
    }

    // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-store/src/lib.rs#graph_visualization_edge_label_fallback_matches_python_contract; python=kogwistar/visualization/graph_viz.py#to_d3_force; fixture=contracts/golden/visualization-label-fallback-v1.json; rust_test=rust/crates/kogwistar-store/src/lib.rs#graph_visualization_edge_label_fallback_matches_python_contract; python_test=tests/test_visualization.py#test_visualization_edge_label_fallback_matches_shared_contract
    #[test]
    fn graph_visualization_edge_label_fallback_matches_python_contract() {
        let contract: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/visualization-label-fallback-v1.json"
        )))
        .expect("visualization label fallback contract is valid JSON");
        let model_fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/python-graph-model-payloads-v1.json"
        )))
        .expect("Python graph model fixture is valid JSON");
        let model_payload = |kind: &str| {
            model_fixture["cases"]
                .as_array()
                .expect("model fixture cases")
                .iter()
                .find(|case| case["kind"] == kind)
                .expect("graph model fixture case")["payload"]
                .clone()
        };
        let record = |payload: &Value| GraphRecord {
            id: payload["id"].as_str().expect("record ID").to_owned(),
            document: Some(payload.to_string()),
            metadata: payload["metadata"].as_object().cloned().unwrap_or_default(),
            embedding: None,
        };
        let source = model_payload("node");
        let mut target = source.clone();
        target["id"] = json!("target");
        let nodes = vec![record(&source), record(&target)];
        for case in contract["cases"].as_array().expect("contract cases") {
            let mut edge = model_payload("edge");
            edge["id"] = case["name"].clone();
            edge["relation"] = case["relation"].clone();
            edge["label"] = case["label"].clone();
            edge["source_ids"] = json!([source["id"]]);
            edge["target_ids"] = json!([target["id"]]);
            let edges = vec![record(&edge)];
            let expected_label = &case["expected"];

            let d3 = graph_viz_d3_from_records(&nodes, &edges, None, "reify", None);
            assert!(
                d3["nodes"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|node| { node["id"] == edge["id"] && &node["label"] == expected_label })
            );
            assert!(
                d3["links"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .all(|link| { &link["relation"] == expected_label })
            );

            let classic = graph_viz_d3_from_records(&nodes, &edges, None, "classic", None);
            assert_eq!(classic["links"][0]["relation"], *expected_label);

            let reified_cytoscape =
                graph_viz_cytoscape_from_records(&nodes, &edges, None, "reify", None);
            assert!(
                reified_cytoscape["elements"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter(|element| {
                        element["data"]["source"] == edge["id"]
                            || element["data"]["target"] == edge["id"]
                    })
                    .all(|element| &element["data"]["label"] == expected_label)
            );

            let classic_cytoscape =
                graph_viz_cytoscape_from_records(&nodes, &edges, None, "classic", None);
            assert!(
                classic_cytoscape["elements"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter(|element| element["data"]["source"] == source["id"])
                    .all(|element| &element["data"]["label"] == expected_label)
            );
        }
    }

    // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-store/src/lib.rs#graph_visualization_empty_doc_id_matches_python_contract; python=kogwistar/visualization/graph_viz.py#_ids_by_doc; fixture=contracts/golden/visualization-doc-filter-v1.json; rust_test=rust/crates/kogwistar-store/src/lib.rs#graph_visualization_empty_doc_id_matches_python_contract; python_test=tests/test_visualization.py#test_visualization_empty_doc_id_matches_python_route_selection_contract
    #[test]
    fn graph_visualization_empty_doc_id_matches_python_contract() {
        let contract: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/visualization-doc-filter-v1.json"
        )))
        .expect("visualization doc filter contract is valid JSON");
        let model_fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/python-graph-model-payloads-v1.json"
        )))
        .expect("Python graph model fixture is valid JSON");
        let node_template = model_fixture["cases"]
            .as_array()
            .expect("model fixture cases")
            .iter()
            .find(|case| case["kind"] == "node")
            .expect("node model fixture")["payload"]
            .clone();
        let edge_template = model_fixture["cases"]
            .as_array()
            .expect("model fixture cases")
            .iter()
            .find(|case| case["kind"] == "edge")
            .expect("edge model fixture")["payload"]
            .clone();
        let record = |payload: &Value| GraphRecord {
            id: payload["id"].as_str().expect("record ID").to_owned(),
            document: Some(payload.to_string()),
            metadata: payload["metadata"].as_object().cloned().unwrap_or_default(),
            embedding: None,
        };
        let mut first_node = node_template.clone();
        first_node["id"] = json!("node-a");
        first_node["doc_id"] = json!("doc-a");
        first_node["metadata"]["doc_id"] = json!("doc-a");
        let mut second_node = node_template;
        second_node["id"] = json!("node-b");
        second_node["doc_id"] = json!("doc-b");
        second_node["metadata"]["doc_id"] = json!("doc-b");
        let nodes = vec![record(&first_node), record(&second_node)];
        let mut first_edge = edge_template.clone();
        first_edge["id"] = json!("edge-a");
        first_edge["doc_id"] = json!("doc-a");
        first_edge["metadata"]["doc_id"] = json!("doc-a");
        first_edge["source_ids"] = json!(["node-a"]);
        first_edge["target_ids"] = json!(["node-a"]);
        let mut second_edge = edge_template;
        second_edge["id"] = json!("edge-b");
        second_edge["doc_id"] = json!("doc-b");
        second_edge["metadata"]["doc_id"] = json!("doc-b");
        second_edge["source_ids"] = json!(["node-b"]);
        second_edge["target_ids"] = json!(["node-b"]);
        let edges = vec![record(&first_edge), record(&second_edge)];

        for case in contract["cases"].as_array().expect("contract cases") {
            let doc_id = case["doc_id"].as_str();
            let payload = graph_viz_d3_from_records(&nodes, &edges, doc_id, "reify", None);
            assert_eq!(payload["doc_id"], case["response_doc_id"]);
            let mut actual_doc_ids = payload["nodes"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|node| node["type"] == "entity")
                .map(|node| node["doc_id"].as_str().unwrap().to_owned())
                .collect::<Vec<_>>();
            actual_doc_ids.sort();
            let mut expected_doc_ids = case["selected_doc_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|doc_id| doc_id.as_str().unwrap().to_owned())
                .collect::<Vec<_>>();
            expected_doc_ids.sort();
            assert_eq!(actual_doc_ids, expected_doc_ids, "case {}", case["name"]);

            let mut actual_edge_doc_ids = payload["nodes"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|node| node["type"] == "edge-node")
                .map(|node| node["doc_id"].as_str().unwrap().to_owned())
                .collect::<Vec<_>>();
            actual_edge_doc_ids.sort();
            let mut expected_edge_doc_ids = case["selected_edge_doc_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|doc_id| doc_id.as_str().unwrap().to_owned())
                .collect::<Vec<_>>();
            expected_edge_doc_ids.sort();
            assert_eq!(
                actual_edge_doc_ids, expected_edge_doc_ids,
                "edge selection case {}",
                case["name"]
            );
        }
    }

    // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-store/src/lib.rs#graph_visualization_hyperedges_match_python_contract; python=kogwistar/visualization/graph_viz.py#to_d3_force; fixture=contracts/golden/visualization-hyperedge-v1.json; rust_test=rust/crates/kogwistar-store/src/lib.rs#graph_visualization_hyperedges_match_python_contract; python_test=tests/test_visualization.py#test_hyperedge_visualization_matches_shared_python_contract
    #[test]
    fn graph_visualization_hyperedges_match_python_contract() {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/visualization-hyperedge-v1.json"
        )))
        .expect("visualization hyperedge contract is valid JSON");
        let records = |field: &str| {
            fixture[field]
                .as_array()
                .expect("fixture graph records")
                .iter()
                .map(|payload| GraphRecord {
                    id: payload["id"].as_str().expect("record ID").to_owned(),
                    document: Some(payload.to_string()),
                    metadata: Map::new(),
                    embedding: None,
                })
                .collect::<Vec<_>>()
        };
        let nodes = records("nodes");
        let edges = records("edges");
        let expected = &fixture["expected"];

        let d3 = graph_viz_d3_from_records(&nodes, &edges, None, "reify", None);
        let actual_node_ids = d3["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|node| node["id"].as_str().unwrap())
            .collect::<Vec<_>>();
        let expected_node_ids = expected["reified_d3_node_order"]
            .as_array()
            .unwrap()
            .iter()
            .map(|id| id.as_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(actual_node_ids, expected_node_ids);
        let actual_reified_links = d3["links"]
            .as_array()
            .unwrap()
            .iter()
            .map(|link| {
                (
                    link["source"].as_str().unwrap(),
                    link["target"].as_str().unwrap(),
                    link["role"].as_str().unwrap(),
                    link["relation"].as_str().unwrap(),
                )
            })
            .collect::<BTreeSet<_>>();
        let expected_reified_links = expected["reified_d3_links"]
            .as_array()
            .unwrap()
            .iter()
            .map(|link| {
                (
                    link["source"].as_str().unwrap(),
                    link["target"].as_str().unwrap(),
                    link["role"].as_str().unwrap(),
                    link["relation"].as_str().unwrap(),
                )
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(actual_reified_links, expected_reified_links);

        let classic_d3 = graph_viz_d3_from_records(&nodes, &edges, None, "classic", None);
        assert_eq!(
            classic_d3["nodes"]
                .as_array()
                .unwrap()
                .iter()
                .map(|node| node["id"].as_str().unwrap())
                .collect::<Vec<_>>(),
            expected["classic_d3_node_order"]
                .as_array()
                .unwrap()
                .iter()
                .map(|id| id.as_str().unwrap())
                .collect::<Vec<_>>()
        );
        let actual_classic_links = classic_d3["links"]
            .as_array()
            .unwrap()
            .iter()
            .map(|link| {
                (
                    link["source"].as_str().unwrap(),
                    link["target"].as_str().unwrap(),
                    link["relation"].as_str().unwrap(),
                )
            })
            .collect::<BTreeSet<_>>();
        let expected_classic_links = expected["classic_d3_links"]
            .as_array()
            .unwrap()
            .iter()
            .map(|link| {
                (
                    link["source"].as_str().unwrap(),
                    link["target"].as_str().unwrap(),
                    link["relation"].as_str().unwrap(),
                )
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(actual_classic_links, expected_classic_links);

        let cytoscape_links =
            |mode: &str| graph_viz_cytoscape_from_records(&nodes, &edges, None, mode, None);
        let reified_cytoscape = cytoscape_links("reify");
        assert_eq!(
            reified_cytoscape["elements"]
                .as_array()
                .unwrap()
                .iter()
                .map(|element| element["data"]["id"].as_str().unwrap())
                .collect::<Vec<_>>(),
            expected["reified_cytoscape_element_order"]
                .as_array()
                .unwrap()
                .iter()
                .map(|id| id.as_str().unwrap())
                .collect::<Vec<_>>()
        );
        let actual_reified_cytoscape_links = reified_cytoscape["elements"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|element| matches!(element["classes"].as_str(), Some("src" | "tgt")))
            .map(|element| {
                (
                    element["data"]["id"].as_str().unwrap(),
                    element["data"]["source"].as_str().unwrap(),
                    element["data"]["target"].as_str().unwrap(),
                    element["data"]["label"].as_str().unwrap(),
                    element["classes"].as_str().unwrap(),
                )
            })
            .collect::<BTreeSet<_>>();
        let expected_reified_cytoscape_links = expected["reified_cytoscape_links"]
            .as_array()
            .unwrap()
            .iter()
            .map(|element| {
                (
                    element["id"].as_str().unwrap(),
                    element["source"].as_str().unwrap(),
                    element["target"].as_str().unwrap(),
                    element["label"].as_str().unwrap(),
                    element["class"].as_str().unwrap(),
                )
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(
            actual_reified_cytoscape_links,
            expected_reified_cytoscape_links
        );

        let classic_cytoscape = cytoscape_links("classic");
        assert_eq!(
            classic_cytoscape["elements"]
                .as_array()
                .unwrap()
                .iter()
                .map(|element| element["data"]["id"].as_str().unwrap())
                .collect::<Vec<_>>(),
            expected["classic_cytoscape_element_order"]
                .as_array()
                .unwrap()
                .iter()
                .map(|id| id.as_str().unwrap())
                .collect::<Vec<_>>()
        );
        let actual_classic_cytoscape_links = classic_cytoscape["elements"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|element| element["data"].get("source").is_some())
            .map(|element| {
                (
                    element["data"]["id"].as_str().unwrap(),
                    element["data"]["source"].as_str().unwrap(),
                    element["data"]["target"].as_str().unwrap(),
                    element["data"]["label"].as_str().unwrap(),
                    element["classes"].as_str().unwrap(),
                )
            })
            .collect::<BTreeSet<_>>();
        let expected_classic_cytoscape_links = expected["classic_cytoscape_links"]
            .as_array()
            .unwrap()
            .iter()
            .map(|element| {
                (
                    element["id"].as_str().unwrap(),
                    element["source"].as_str().unwrap(),
                    element["target"].as_str().unwrap(),
                    element["label"].as_str().unwrap(),
                    element["class"].as_str().unwrap(),
                )
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(
            actual_classic_cytoscape_links,
            expected_classic_cytoscape_links
        );
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
    fn node_redirect_resolution_matches_python_lifecycle_modes() {
        let mut redirected = node_record("old");
        redirected
            .metadata
            .insert("lifecycle_status".to_owned(), json!("tombstoned"));
        redirected
            .metadata
            .insert("redirect_to_id".to_owned(), json!("middle"));
        let mut middle = node_record("middle");
        middle
            .metadata
            .insert("lifecycle_status".to_owned(), json!("tombstoned"));
        middle
            .metadata
            .insert("redirect_to_id".to_owned(), json!("current"));
        let nodes = vec![redirected.clone(), middle, node_record("current")];

        assert_eq!(
            resolve_graph_node_redirect_id("old", &nodes),
            Some("current".into())
        );
        assert_eq!(
            resolve_graph_node_redirect_id("current", &nodes),
            Some("current".into())
        );
        assert_eq!(resolve_graph_node_redirect_id("missing", &nodes), None);

        let mut tombstone = node_record("gone");
        tombstone
            .metadata
            .insert("lifecycle_status".to_owned(), json!("tombstoned"));
        let mut broken = redirected;
        broken
            .metadata
            .insert("redirect_to_id".to_owned(), json!("missing"));
        assert_eq!(resolve_graph_node_redirect_id("gone", &[tombstone]), None);
        assert_eq!(resolve_graph_node_redirect_id("old", &[broken]), None);

        let mut cycle_a = node_record("cycle-a");
        cycle_a
            .metadata
            .insert("redirect_to_id".to_owned(), json!("cycle-b"));
        let mut cycle_b = node_record("cycle-b");
        cycle_b
            .metadata
            .insert("redirect_to_id".to_owned(), json!("cycle-a"));
        assert_eq!(
            resolve_graph_node_redirect_id("cycle-a", &[cycle_a, cycle_b]),
            None
        );
    }

    #[test]
    fn edge_redirect_resolution_matches_python_lifecycle_modes() {
        let mut redirected = edge_record("old-edge", json!({"source_ids": [], "target_ids": []}));
        redirected
            .metadata
            .insert("lifecycle_status".to_owned(), json!("tombstoned"));
        redirected
            .metadata
            .insert("redirect_to_id".to_owned(), json!("middle-edge"));
        let mut middle = edge_record("middle-edge", json!({"source_ids": [], "target_ids": []}));
        middle
            .metadata
            .insert("lifecycle_status".to_owned(), json!("tombstoned"));
        middle
            .metadata
            .insert("redirect_to_id".to_owned(), json!("current-edge"));
        let edges = vec![
            redirected.clone(),
            middle,
            edge_record("current-edge", json!({})),
        ];

        assert_eq!(
            resolve_graph_edge_redirect_id("old-edge", &edges),
            Some("current-edge".into())
        );
        assert_eq!(
            resolve_graph_edge_redirect_id("current-edge", &edges),
            Some("current-edge".into())
        );
        assert_eq!(resolve_graph_edge_redirect_id("missing", &edges), None);

        let mut tombstone = edge_record("gone-edge", json!({}));
        tombstone
            .metadata
            .insert("lifecycle_status".to_owned(), json!("tombstoned"));
        let mut broken = redirected;
        broken
            .metadata
            .insert("redirect_to_id".to_owned(), json!("missing-edge"));
        assert_eq!(
            resolve_graph_edge_redirect_id("gone-edge", &[tombstone]),
            None
        );
        assert_eq!(resolve_graph_edge_redirect_id("old-edge", &[broken]), None);

        let mut cycle_a = edge_record("cycle-edge-a", json!({}));
        cycle_a
            .metadata
            .insert("redirect_to_id".to_owned(), json!("cycle-edge-b"));
        let mut cycle_b = edge_record("cycle-edge-b", json!({}));
        cycle_b
            .metadata
            .insert("redirect_to_id".to_owned(), json!("cycle-edge-a"));
        assert_eq!(
            resolve_graph_edge_redirect_id("cycle-edge-a", &[cycle_a, cycle_b]),
            None
        );
    }

    #[test]
    fn graph_neighbors_preserve_cross_kind_same_id_from_shared_fixture() {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/graph-neighbors-cross-kind-id-v1.json"
        )))
        .expect("cross-kind graph-neighbor fixture is valid JSON");
        let shared_id = fixture["expected"]["node_ids"][0]
            .as_str()
            .expect("expected shared node ID");
        let parent_edge = &fixture["parent_edge"];
        let nodes = vec![node_record(shared_id)];
        let edges = vec![
            edge_record(parent_edge["id"].as_str().unwrap(), parent_edge.clone()),
            edge_record(shared_id, json!({"source_ids": [], "target_ids": []})),
        ];

        let actual =
            graph_neighbors_from_records(fixture["rid"].as_str().unwrap(), &nodes, &edges, None);
        let expected = GraphNeighbors {
            nodes: fixture["expected"]["node_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|id| id.as_str().unwrap().to_owned())
                .collect(),
            edges: fixture["expected"]["edge_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|id| id.as_str().unwrap().to_owned())
                .collect(),
        };

        assert_eq!(actual, expected);
    }

    #[test]
    fn graph_visualization_projects_acl_visible_records_in_both_formats() {
        let mut source = node_record("n1");
        source.document =
            Some(json!({"label":"Source","summary":"source summary","doc_id":"d1"}).to_string());
        let mut target = node_record("n2");
        target.document = Some(json!({"label":"Target","doc_id":"d1"}).to_string());
        let mut hidden = node_record("private-node");
        hidden.document = Some(json!({"label":"Hidden","doc_id":"d1"}).to_string());
        let visible_edge = edge_record(
            "e1",
            json!({
                "relation":"connects",
                "doc_id":"d1",
                "source_ids":["n1", "private-node"],
                "target_ids":["n2"],
                "properties":{"weight":2}
            }),
        );
        let unrelated_edge = edge_record(
            "e2",
            json!({"relation":"other","doc_id":"d2","source_ids":["n1"],"target_ids":["n2"]}),
        );
        let mut tombstoned = edge_record(
            "e3",
            json!({"relation":"deleted","doc_id":"d1","source_ids":["n1"],"target_ids":["n2"]}),
        );
        tombstoned
            .metadata
            .insert("lifecycle_status".to_owned(), json!("tombstoned"));
        let nodes = vec![source, target, hidden];
        let edges = vec![visible_edge.clone(), unrelated_edge, tombstoned];

        let d3 = graph_viz_d3_from_records(&nodes[..2], &edges, Some("d1"), "reify", None);
        assert_eq!(d3["nodes"].as_array().unwrap().len(), 2);
        assert!(d3["links"].as_array().unwrap().is_empty());
        assert!(!d3.to_string().contains("private-node"));
        assert!(!d3.to_string().contains("e2"));
        assert!(!d3.to_string().contains("e3"));

        let cy = graph_viz_cytoscape_from_records(&nodes[..2], &edges, Some("d1"), "classic", None);
        let elements = cy["elements"].as_array().unwrap();
        assert_eq!(elements.len(), 2);
        assert_eq!(cy["mode"], "classic");
        assert_eq!(cy["doc_id"], "d1");

        let complete_edge = edge_record(
            "e1",
            json!({
                "relation":"connects",
                "doc_id":"d1",
                "source_ids":["n1"],
                "target_ids":["n2"],
                "properties":{"weight":2}
            }),
        );
        let source_edge = edge_record(
            "e4",
            json!({
                "relation":"composes",
                "doc_id":"d1",
                "source_edge_ids":["e1"],
                "target_edge_ids":["e5"]
            }),
        );
        let target_edge = edge_record("e5", json!({"relation":"terminal","doc_id":"d1"}));
        let reified = graph_viz_cytoscape_from_records(
            &nodes[..2],
            &[complete_edge, source_edge, target_edge],
            Some("d1"),
            "reify",
            None,
        );
        let reified_elements = reified["elements"].as_array().unwrap();
        assert!(reified_elements.iter().any(|element| {
            element["data"]["id"] == "e4::srcE::e1"
                && element["data"]["source"] == "e1"
                && element["data"]["target"] == "e4"
        }));
        assert!(reified_elements.iter().any(|element| {
            element["data"]["id"] == "e4::tgtE::e5"
                && element["data"]["source"] == "e4"
                && element["data"]["target"] == "e5"
        }));
    }

    // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-store/src/lib.rs#graph_record_has_reference; python=kogwistar/visualization/graph_viz.py#_filter_by_insertion_method; fixture=contracts/golden/visualization-insertion-filter-v1.json; rust_test=rust/crates/kogwistar-store/src/lib.rs#graph_visualization_filters_by_reference_method_and_preserves_python_fallback; python_test=tests/test_visualization.py#test_visualization_insertion_method_filter_matches_shared_rust_fixture
    #[test]
    fn graph_visualization_filters_by_reference_method_and_preserves_python_fallback() {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/visualization-insertion-filter-v1.json"
        )))
        .unwrap();
        let make_record = |value: &Value, edge: bool| {
            let id = value["id"].as_str().unwrap();
            let document = if edge {
                json!({
                    "label": id,
                    "source_ids": ["n1"],
                    "target_ids": ["n2"],
                    "mentions": value["references"].clone()
                })
            } else {
                json!({
                    "label": id,
                    "references": value["references"].clone()
                })
            };
            let mut record = GraphRecord {
                id: id.to_owned(),
                document: Some(document.to_string()),
                metadata: Map::from_iter([("doc_id".to_owned(), fixture["doc_id"].clone())]),
                embedding: None,
            };
            if edge {
                record
                    .metadata
                    .insert("entity_type".to_owned(), json!("edge"));
            }
            record
        };
        let nodes = fixture["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|record| make_record(record, false))
            .collect::<Vec<_>>();
        let edges = fixture["edges"]
            .as_array()
            .unwrap()
            .iter()
            .map(|record| make_record(record, true))
            .collect::<Vec<_>>();
        let doc_id = fixture["doc_id"].as_str();
        let insertion_method = fixture["insertion_method"].as_str();

        let filtered_node_ids = graph_viz_selected_records(&nodes, doc_id, insertion_method)
            .into_iter()
            .map(|node| node.id.as_str())
            .collect::<BTreeSet<_>>();
        let filtered_edge_ids = graph_viz_selected_records(&edges, doc_id, insertion_method)
            .into_iter()
            .map(|edge| edge.id.as_str())
            .collect::<BTreeSet<_>>();
        assert_eq!(
            filtered_node_ids,
            fixture["expected_matches_by_kind"]["node"]
                .as_array()
                .unwrap()
                .iter()
                .map(|id| id.as_str().unwrap())
                .collect::<BTreeSet<_>>()
        );
        assert_eq!(
            filtered_edge_ids,
            fixture["expected_matches_by_kind"]["edge"]
                .as_array()
                .unwrap()
                .iter()
                .map(|id| id.as_str().unwrap())
                .collect::<BTreeSet<_>>()
        );

        let mut empty_method_ids = graph_viz_selected_records(&nodes, doc_id, Some(""))
            .into_iter()
            .chain(graph_viz_selected_records(&edges, doc_id, Some("")))
            .map(|record| record.id.as_str())
            .collect::<Vec<_>>();
        empty_method_ids.sort_unstable();
        assert_eq!(
            empty_method_ids,
            fixture["expected_empty_method_filter"]
                .as_array()
                .unwrap()
                .iter()
                .map(|id| id.as_str().unwrap())
                .collect::<Vec<_>>()
        );

        let mut fallback_ids = graph_viz_selected_records(&nodes, doc_id, Some("missing-method"))
            .into_iter()
            .chain(graph_viz_selected_records(
                &edges,
                doc_id,
                Some("missing-method"),
            ))
            .map(|record| record.id.as_str())
            .collect::<Vec<_>>();
        fallback_ids.sort_unstable();
        assert_eq!(
            fallback_ids,
            fixture["expected_empty_match_fallback"]
                .as_array()
                .unwrap()
                .iter()
                .map(|id| id.as_str().unwrap())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn graph_neighbors_excludes_stale_and_other_document_edges() {
        let mut stale = edge_record(
            "stale",
            serde_json::json!({
                "doc_id": "doc-requested",
                "source_ids": ["n1"],
                "target_ids": ["n2"]
            }),
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
    fn graph_acl_traversal_matches_shared_python_contract() {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/graph-acl-traversal-v1.json"
        )))
        .expect("valid graph ACL traversal fixture");
        assert_eq!(fixture["schema_version"], 1);
        let truth_graph = fixture["truth_graph"].as_str().expect("truth graph");
        let edge_spec = &fixture["edge"];
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        for node_spec in fixture["nodes"].as_array().expect("nodes") {
            let id = node_spec["id"].as_str().expect("node ID");
            let record = node_record(id);
            nodes.push(record.clone());

            let mut acl = GraphRecord::new(format!("acl-{id}"));
            acl.metadata
                .insert("entity_type".into(), "acl_record".into());
            acl.metadata
                .insert("acl_truth_graph".into(), truth_graph.into());
            acl.metadata
                .insert("acl_target_entity_id".into(), id.into());
            acl.metadata
                .insert("acl_target_grain".into(), "node".into());
            acl.metadata.insert("acl_version".into(), 1.into());
            acl.metadata
                .insert("acl_mode".into(), node_spec["mode"].clone());
            if let Some(owner) = node_spec["owner_id"].as_str() {
                acl.metadata.insert("owner_id".into(), owner.into());
            }
            nodes.push(acl);
        }

        let edge_id = edge_spec["id"].as_str().expect("edge ID");
        let edge_document = serde_json::json!({
            "doc_id": fixture["doc_id"],
            "relation": edge_spec["relation"],
            "source_ids": edge_spec["source_ids"],
            "target_ids": edge_spec["target_ids"],
            "source_edge_ids": [],
            "target_edge_ids": []
        });
        let edge = edge_record(edge_id, edge_document);
        edges.push(edge.clone());
        let mut edge_acl = GraphRecord::new(format!("acl-{edge_id}"));
        edge_acl
            .metadata
            .insert("entity_type".into(), "acl_record".into());
        edge_acl
            .metadata
            .insert("acl_truth_graph".into(), truth_graph.into());
        edge_acl
            .metadata
            .insert("acl_target_entity_id".into(), edge_id.into());
        edge_acl
            .metadata
            .insert("acl_target_grain".into(), "edge".into());
        edge_acl.metadata.insert("acl_version".into(), 1.into());
        edge_acl
            .metadata
            .insert("acl_mode".into(), edge_spec["mode"].clone());
        if let Some(owner) = edge_spec["owner_id"].as_str() {
            edge_acl.metadata.insert("owner_id".into(), owner.into());
        }
        nodes.push(edge_acl);

        for principal_spec in fixture["principals"].as_array().expect("principals") {
            let principal = GraphAclPrincipal {
                principal_id: principal_spec["id"]
                    .as_str()
                    .expect("principal ID")
                    .to_owned(),
                ..Default::default()
            };
            let (visible_nodes, visible_edges) =
                graph_records_visible_by_acl(&nodes, &edges, truth_graph, &principal);
            let neighbors = graph_neighbors_from_records(
                fixture["nodes"][0]["id"].as_str().expect("source ID"),
                &visible_nodes,
                &visible_edges,
                fixture["doc_id"].as_str(),
            );
            let expected_nodes = principal_spec["expected_neighbors"]["nodes"]
                .as_array()
                .expect("expected nodes")
                .iter()
                .map(|value| value.as_str().expect("node ID").to_owned())
                .collect::<BTreeSet<_>>();
            let expected_edges = principal_spec["expected_neighbors"]["edges"]
                .as_array()
                .expect("expected edges")
                .iter()
                .map(|value| value.as_str().expect("edge ID").to_owned())
                .collect::<BTreeSet<_>>();
            assert_eq!(neighbors.nodes, expected_nodes, "{}", principal_spec["id"]);
            assert_eq!(neighbors.edges, expected_edges, "{}", principal_spec["id"]);

            let k_hop = graph_k_hop_from_records(
                &GraphKHopQuery {
                    start_ids: vec![
                        fixture["nodes"][0]["id"]
                            .as_str()
                            .expect("source ID")
                            .to_owned(),
                    ],
                    k: 2,
                    doc_id: fixture["doc_id"].as_str().map(str::to_owned),
                },
                &visible_nodes,
                &visible_edges,
            );
            let expected_k_hop = principal_spec["expected_k_hop"]
                .as_array()
                .expect("expected k-hop layers")
                .iter()
                .map(|layer| {
                    serde_json::json!({
                        "nodes": layer["nodes"],
                        "edges": layer["edges"]
                    })
                })
                .collect::<Vec<_>>();
            let actual_k_hop = k_hop
                .iter()
                .map(|layer| {
                    serde_json::json!({
                        "nodes": layer.nodes,
                        "edges": layer.edges
                    })
                })
                .collect::<Vec<_>>();
            assert_eq!(actual_k_hop, expected_k_hop, "{}", principal_spec["id"]);

            let shortest_path = graph_shortest_path_from_records(
                &GraphShortestPathQuery {
                    src_id: fixture["nodes"][0]["id"]
                        .as_str()
                        .expect("source ID")
                        .to_owned(),
                    dst_id: fixture["nodes"][1]["id"]
                        .as_str()
                        .expect("target ID")
                        .to_owned(),
                    doc_id: fixture["doc_id"].as_str().map(str::to_owned),
                    max_depth: 8,
                },
                &visible_nodes,
                &visible_edges,
            );
            assert_eq!(
                shortest_path,
                principal_spec["expected_shortest_path"]
                    .as_array()
                    .expect("expected path")
                    .iter()
                    .map(|value| value.as_str().expect("path ID").to_owned())
                    .collect::<Vec<_>>(),
                "{}",
                principal_spec["id"]
            );

            let find_edges = graph_find_edges_from_records(
                &GraphFindEdgesQuery {
                    relation: edge_spec["relation"].as_str().map(str::to_owned),
                    doc_id: fixture["doc_id"].as_str().map(str::to_owned),
                    ..Default::default()
                },
                &visible_nodes,
                &visible_edges,
            );
            assert_eq!(
                find_edges,
                principal_spec["expected_find_edges"]
                    .as_array()
                    .expect("expected matching edges")
                    .iter()
                    .map(|value| value.as_str().expect("edge ID").to_owned())
                    .collect::<Vec<_>>(),
                "{}",
                principal_spec["id"]
            );
        }
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
    fn acl_projection_reuses_index_after_helper_records_are_removed() {
        let mut acl = GraphRecord::new("acl-n1");
        acl.metadata.extend(
            serde_json::json!({
                "entity_type": "acl_record",
                "acl_truth_graph": "knowledge",
                "acl_target_entity_id": "n1",
                "acl_target_grain": "node",
                "acl_version": 1,
                "acl_mode": "shared"
            })
            .as_object()
            .expect("ACL metadata is an object")
            .clone(),
        );
        acl.document = Some(
            serde_json::json!({
                "properties": {"shared_with_principals": ["alice"]}
            })
            .to_string(),
        );
        let complete_snapshot = vec![acl, GraphRecord::new("n1"), GraphRecord::new("n2")];
        let alice = GraphAclPrincipal {
            principal_id: "alice".to_owned(),
            ..Default::default()
        };

        let projection = GraphAclProjection::from_complete_snapshot(
            &complete_snapshot,
            &[],
            "knowledge",
            &alice,
        );
        assert_eq!(
            projection
                .nodes()
                .iter()
                .map(|record| record.id.as_str())
                .collect::<Vec<_>>(),
            ["n1"]
        );
        assert!(projection.edges().is_empty());

        // The filtered view deliberately lacks the ACL helper node. A later
        // projection reuses the full snapshot's index through the wrapper.
        let reprojected = projection.filter_records(projection.nodes(), &[]);
        assert_eq!(reprojected.nodes(), projection.nodes());

        // Rebuilding from the filtered view has no ACL rule and correctly
        // fails closed; services must not mistake that for the canonical ACL.
        let (incorrectly_rebuilt_nodes, _) =
            graph_records_visible_by_acl(projection.nodes(), &[], "knowledge", &alice);
        assert!(incorrectly_rebuilt_nodes.is_empty());
    }

    #[test]
    fn paged_conversation_acl_uses_rules_from_other_pages() {
        let acl_record = |id: &str, target_id: &str| {
            let mut record = GraphRecord::new(id);
            record.metadata.extend(
                serde_json::json!({
                    "entity_type": "acl_record",
                    "acl_truth_graph": "conversation",
                    "acl_target_entity_id": target_id,
                    "acl_target_grain": "node",
                    "acl_version": 1,
                    "acl_mode": "private",
                    "owner_id": "alice"
                })
                .as_object()
                .expect("ACL metadata is an object")
                .clone(),
            );
            record
        };
        let mut start = GraphRecord::new("start-alice");
        start.document = Some(
            serde_json::json!({
                "metadata": {
                    "entity_type": "conversation_start",
                    "conversation_id": "conversation-1",
                    "user_id": "alice",
                    "status": "active"
                }
            })
            .to_string(),
        );
        let mut turn = GraphRecord::new("turn-1");
        turn.metadata.extend(
            serde_json::json!({
                "entity_type": "conversation_turn",
                "conversation_id": "conversation-1",
                "turn_index": 0
            })
            .as_object()
            .expect("turn metadata is an object")
            .clone(),
        );
        let acl_page = [
            acl_record("acl-start", "start-alice"),
            acl_record("acl-turn", "turn-1"),
        ];
        let data_page = [start, turn];

        let mut acl_index = GraphAclIndex::default();
        acl_index.extend_from_records(&acl_page, "conversation");
        let compact = compact_conversation_list_candidates(&data_page);
        assert!(compact.iter().all(|record| record.document.is_none()));

        let alice = GraphAclPrincipal {
            principal_id: "alice".to_owned(),
            ..Default::default()
        };
        let bob = GraphAclPrincipal {
            principal_id: "bob".to_owned(),
            ..Default::default()
        };
        let (visible_alice, _) =
            graph_records_visible_by_acl_with_index(&compact, &[], &acl_index, &alice);
        let (visible_bob, _) =
            graph_records_visible_by_acl_with_index(&compact, &[], &acl_index, &bob);
        let mut alice_list = ConversationListAccumulator::new("alice");
        alice_list.extend_visible_records(&visible_alice);

        assert_eq!(
            alice_list.finish(),
            vec![ConversationListEntry {
                id: "conversation-1".to_owned(),
                start_node_id: "start-alice".to_owned(),
                status: "active".to_owned(),
                turn_count: 1,
            }]
        );
        let mut bob_list = ConversationListAccumulator::new("bob");
        bob_list.extend_visible_records(&visible_bob);
        assert!(bob_list.finish().is_empty());
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
    fn graph_acl_latest_version_is_scoped_to_selected_truth_graph() {
        let entity_id = "shared-entity-id";
        let acl = |record_id: &str, truth_graph: &str, version: i64, mode: &str| {
            let mut record = GraphRecord::new(record_id);
            record.metadata.extend(
                serde_json::json!({
                    "entity_type": "acl_record",
                    "acl_truth_graph": truth_graph,
                    "acl_target_entity_id": entity_id,
                    "acl_target_grain": "node",
                    "acl_version": version,
                    "acl_mode": mode,
                    "owner_id": if mode == "private" { Some("wisdom-owner") } else { None },
                })
                .as_object()
                .expect("ACL metadata")
                .clone(),
            );
            record
        };
        let records = vec![
            GraphRecord::new(entity_id),
            acl("knowledge-v1", "knowledge", 1, "public"),
            acl("wisdom-v1", "wisdom", 1, "public"),
            acl("wisdom-v2", "wisdom", 2, "private"),
        ];
        let reader = GraphAclPrincipal {
            principal_id: "reader".into(),
            ..Default::default()
        };
        let owner = GraphAclPrincipal {
            principal_id: "wisdom-owner".into(),
            ..Default::default()
        };

        let (knowledge_reader, _) =
            graph_records_visible_by_acl(&records, &[], "knowledge", &reader);
        let (wisdom_reader, _) = graph_records_visible_by_acl(&records, &[], "wisdom", &reader);
        let (wisdom_owner, _) = graph_records_visible_by_acl(&records, &[], "wisdom", &owner);

        assert!(knowledge_reader.iter().any(|record| record.id == entity_id));
        assert!(!wisdom_reader.iter().any(|record| record.id == entity_id));
        assert!(wisdom_owner.iter().any(|record| record.id == entity_id));
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

        let isolation = &fixture["graph_type_isolation"];
        let entity_id = isolation["entity_id"].as_str().expect("entity ID");
        let mut scoped_nodes = vec![GraphRecord::new(entity_id)];
        for case in isolation["cases"].as_array().expect("graph-type cases") {
            let mut acl = GraphRecord::new(format!(
                "acl-{}",
                case["truth_graph"].as_str().expect("truth graph")
            ));
            acl.metadata.extend(
                serde_json::json!({
                    "entity_type": "acl_record",
                    "acl_truth_graph": case["truth_graph"],
                    "acl_target_entity_id": entity_id,
                    "acl_target_grain": case["grain"],
                    "acl_version": 1,
                    "acl_mode": case["mode"],
                    "owner_id": case.get("owner_id")
                })
                .as_object()
                .expect("ACL metadata object")
                .clone(),
            );
            scoped_nodes.push(acl);
        }
        for case in isolation["cases"].as_array().expect("graph-type cases") {
            let truth_graph = case["truth_graph"].as_str().expect("truth graph");
            let (visible, _) =
                graph_records_visible_by_acl(&scoped_nodes, &[], truth_graph, &principal);
            assert_eq!(
                visible.iter().any(|record| record.id == entity_id),
                case["expected_visible"]
                    .as_bool()
                    .expect("visibility expectation"),
                "ACL decision leaked across graph type {truth_graph}"
            );
        }
    }

    #[test]
    fn wisdom_acl_keeps_same_id_node_and_edge_grains_independent() {
        let entity_id = "same-id-in-wisdom";
        let acl = |id: &str, grain: &str, mode: &str, owner: Option<&str>| {
            let mut record = GraphRecord::new(format!("acl-{grain}"));
            record
                .metadata
                .insert("entity_type".into(), "acl_record".into());
            record
                .metadata
                .insert("acl_truth_graph".into(), "wisdom".into());
            record
                .metadata
                .insert("acl_target_entity_id".into(), id.into());
            record
                .metadata
                .insert("acl_target_grain".into(), grain.into());
            record
                .metadata
                .insert("acl_version".into(), serde_json::json!(1));
            record.metadata.insert("acl_mode".into(), mode.into());
            if let Some(owner) = owner {
                record.metadata.insert("owner_id".into(), owner.into());
            }
            record
        };
        let node = GraphRecord::new(entity_id);
        let mut edge = GraphRecord::new(entity_id);
        edge.metadata
            .insert("source_ids".into(), serde_json::json!([]));
        edge.metadata
            .insert("target_ids".into(), serde_json::json!([]));
        let nodes = vec![
            node,
            acl(entity_id, "node", "public", None),
            acl(entity_id, "edge", "private", Some("alice")),
        ];

        let reader = GraphAclPrincipal {
            principal_id: "bob".to_owned(),
            groups: BTreeSet::new(),
            security_scope: None,
        };
        let (reader_nodes, reader_edges) =
            graph_records_visible_by_acl(&nodes, &[edge.clone()], "wisdom", &reader);
        assert_eq!(
            reader_nodes
                .iter()
                .map(|record| record.id.as_str())
                .collect::<Vec<_>>(),
            [entity_id]
        );
        assert!(reader_edges.is_empty());

        let owner = GraphAclPrincipal {
            principal_id: "alice".to_owned(),
            groups: BTreeSet::new(),
            security_scope: None,
        };
        let (owner_nodes, owner_edges) =
            graph_records_visible_by_acl(&nodes, &[edge], "wisdom", &owner);
        assert_eq!(
            owner_nodes
                .iter()
                .map(|record| record.id.as_str())
                .collect::<Vec<_>>(),
            [entity_id]
        );
        assert_eq!(
            owner_edges
                .iter()
                .map(|record| record.id.as_str())
                .collect::<Vec<_>>(),
            [entity_id]
        );
    }

    #[test]
    fn wisdom_node_acl_does_not_override_same_id_edge_acl() {
        let entity_id = "same-id-in-wisdom";
        let acl = |grain: &str, mode: &str| {
            let mut record = GraphRecord::new(format!("acl-{grain}"));
            record.metadata.extend(
                serde_json::json!({
                    "entity_type": "acl_record",
                    "acl_truth_graph": "wisdom",
                    "acl_target_entity_id": entity_id,
                    "acl_target_grain": grain,
                    "acl_version": 1,
                    "acl_mode": mode
                })
                .as_object()
                .expect("ACL metadata")
                .clone(),
            );
            record
        };
        let node = GraphRecord::new(entity_id);
        let mut edge = GraphRecord::new(entity_id);
        edge.metadata
            .insert("source_ids".into(), serde_json::json!([]));
        edge.metadata
            .insert("target_ids".into(), serde_json::json!([]));
        let records = vec![node, acl("node", "private"), acl("edge", "public")];
        let reader = GraphAclPrincipal {
            principal_id: "reader".into(),
            ..Default::default()
        };

        let (visible_nodes, visible_edges) =
            graph_records_visible_by_acl(&records, &[edge], "wisdom", &reader);

        assert!(visible_nodes.is_empty());
        assert_eq!(
            visible_edges
                .iter()
                .map(|record| record.id.as_str())
                .collect::<Vec<_>>(),
            [entity_id]
        );
    }

    #[test]
    fn wisdom_shared_and_group_acl_grants_are_truth_graph_scoped() {
        let direct_id = "wisdom-direct-share-node";
        let group_id = "wisdom-group-share-node";
        let acl = |id: &str, mode: &str, properties: Value| {
            let mut record = GraphRecord::new(format!("acl-{id}"));
            record.metadata.extend(
                serde_json::json!({
                    "entity_type": "acl_record",
                    "acl_truth_graph": "wisdom",
                    "acl_target_entity_id": id,
                    "acl_target_grain": "node",
                    "acl_version": 1,
                    "acl_mode": mode
                })
                .as_object()
                .expect("ACL metadata object")
                .clone(),
            );
            record.document = Some(serde_json::json!({"properties": properties}).to_string());
            record
        };
        let nodes = vec![
            GraphRecord::new(direct_id),
            acl(
                direct_id,
                "shared",
                serde_json::json!({"shared_with_principals": ["alice"]}),
            ),
            GraphRecord::new(group_id),
            acl(
                group_id,
                "group",
                serde_json::json!({"shared_with_groups": ["wisdom-reviewers"]}),
            ),
        ];
        let alice = GraphAclPrincipal {
            principal_id: "alice".to_owned(),
            groups: BTreeSet::new(),
            security_scope: None,
        };
        let (wisdom_direct_nodes, _) = graph_records_visible_by_acl(&nodes, &[], "wisdom", &alice);
        assert!(wisdom_direct_nodes.iter().any(|node| node.id == direct_id));
        assert!(!wisdom_direct_nodes.iter().any(|node| node.id == group_id));
        let (other_graph_nodes, _) = graph_records_visible_by_acl(&nodes, &[], "knowledge", &alice);
        assert!(other_graph_nodes.is_empty());

        let reviewer = GraphAclPrincipal {
            principal_id: "bob".to_owned(),
            groups: BTreeSet::from(["wisdom-reviewers".to_owned()]),
            security_scope: None,
        };
        let (wisdom_group_nodes, _) =
            graph_records_visible_by_acl(&nodes, &[], "wisdom", &reviewer);
        assert!(wisdom_group_nodes.iter().any(|node| node.id == group_id));
        assert!(!wisdom_group_nodes.iter().any(|node| node.id == direct_id));
    }

    #[test]
    fn graph_node_item_usage_matches_shared_python_contract() {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/graph-acl-item-usage-v1.json"
        )))
        .expect("valid graph ACL item-use fixture");
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
        for case in fixture["cases"].as_array().expect("cases") {
            let node_id = case["node_id"].as_str().expect("node ID");
            let mention_span_counts = case["mention_span_counts"]
                .as_array()
                .expect("mention span counts");
            let mentions = mention_span_counts
                .iter()
                .map(|span_count| {
                    let span_count = span_count.as_u64().expect("span count") as usize;
                    serde_json::json!({"spans": vec![serde_json::json!({}); span_count]})
                })
                .collect::<Vec<_>>();
            let mut records = Vec::new();
            for acl_record in case["acl_records"].as_array().expect("ACL records") {
                let mut record = GraphRecord::new(format!("acl-{node_id}-{}", records.len()));
                record
                    .metadata
                    .insert("entity_type".into(), "acl_record".into());
                record
                    .metadata
                    .insert("acl_truth_graph".into(), fixture["truth_graph"].clone());
                record
                    .metadata
                    .insert("acl_target_entity_id".into(), node_id.into());
                record.metadata.insert(
                    "acl_target_grain".into(),
                    acl_record["grain"].as_str().expect("ACL grain").into(),
                );
                record
                    .metadata
                    .insert("acl_version".into(), serde_json::json!(1));
                record.metadata.insert(
                    "acl_mode".into(),
                    acl_record["mode"].as_str().expect("ACL mode").into(),
                );
                if let Some(item_id) = acl_record["target_item_id"].as_str() {
                    record
                        .metadata
                        .insert("acl_target_item_id".into(), item_id.into());
                }
                if let Some(owner_id) = acl_record["owner_id"].as_str() {
                    record.metadata.insert("owner_id".into(), owner_id.into());
                }
                if acl_record.get("shared_with_principals").is_some()
                    || acl_record.get("shared_with_groups").is_some()
                {
                    record.document = Some(
                        serde_json::json!({
                            "properties": {
                                "shared_with_principals": acl_record
                                    .get("shared_with_principals")
                                    .cloned()
                                    .unwrap_or_else(|| serde_json::json!([])),
                                "shared_with_groups": acl_record
                                    .get("shared_with_groups")
                                    .cloned()
                                    .unwrap_or_else(|| serde_json::json!([]))
                            }
                        })
                        .to_string(),
                    );
                }
                records.push(record);
            }
            let mut target = GraphRecord::new(node_id);
            let encoded_mentions = serde_json::to_string(&mentions).unwrap();
            if case["mentions_storage"].as_str() == Some("document_root") {
                target.document =
                    Some(serde_json::json!({"mentions": encoded_mentions}).to_string());
            } else {
                target.metadata.insert(
                    "mentions".into(),
                    serde_json::Value::String(encoded_mentions),
                );
            }
            records.push(target);
            let (visible, _) = graph_records_visible_by_acl(&records, &[], "knowledge", &principal);
            let expected = case["expected_visible"]
                .as_bool()
                .expect("expected visibility");
            assert_eq!(
                visible.iter().any(|record| record.id == node_id),
                expected,
                "{node_id}"
            );
        }
    }

    #[test]
    fn graph_edge_visibility_requires_visible_usage_and_existing_endpoints() {
        let principal = GraphAclPrincipal {
            principal_id: "alice".into(),
            groups: BTreeSet::new(),
            security_scope: None,
        };
        let acl_record = |id: &str, grain: &str, item_id: Option<&str>, mode: &str| {
            let mut record =
                GraphRecord::new(format!("acl-{id}-{grain}-{}", item_id.unwrap_or("entity")));
            record
                .metadata
                .insert("entity_type".into(), "acl_record".into());
            record
                .metadata
                .insert("acl_truth_graph".into(), "knowledge".into());
            record
                .metadata
                .insert("acl_target_entity_id".into(), id.into());
            record
                .metadata
                .insert("acl_target_grain".into(), grain.into());
            record
                .metadata
                .insert("acl_version".into(), serde_json::json!(1));
            record.metadata.insert("acl_mode".into(), mode.into());
            if let Some(item_id) = item_id {
                record
                    .metadata
                    .insert("acl_target_item_id".into(), item_id.into());
            }
            record
        };

        let edge_id = "edge-private-span";
        let grounding_id = format!("{edge_id}::mention::0");
        let span_id = format!("{grounding_id}::span::0");
        let mut edge = GraphRecord::new(edge_id);
        edge.metadata
            .insert("source_ids".into(), serde_json::json!(["visible-node"]));
        edge.metadata
            .insert("mentions".into(), serde_json::json!([{"spans": [{}]}]));
        let mut nodes = vec![
            acl_record(edge_id, "edge", None, "public"),
            acl_record(edge_id, "grounding", Some(&grounding_id), "public"),
            acl_record(edge_id, "span", Some(&span_id), "private"),
            acl_record("visible-node", "node", None, "public"),
            GraphRecord::new("visible-node"),
        ];
        let (visible_nodes, visible_edges) =
            graph_records_visible_by_acl(&nodes, &[edge], "knowledge", &principal);
        assert_eq!(visible_nodes.len(), 1);
        assert!(
            visible_edges.is_empty(),
            "private edge span must hide the edge"
        );

        let mut endpoint_edge = GraphRecord::new("edge-hidden-endpoint");
        endpoint_edge
            .metadata
            .insert("source_ids".into(), serde_json::json!(["hidden-node"]));
        nodes = vec![
            acl_record("edge-hidden-endpoint", "edge", None, "public"),
            acl_record("hidden-node", "node", None, "private"),
            GraphRecord::new("hidden-node"),
        ];
        let (_, visible_edges) =
            graph_records_visible_by_acl(&nodes, &[endpoint_edge.clone()], "knowledge", &principal);
        assert!(
            visible_edges.is_empty(),
            "an edge must not disclose an existing hidden endpoint"
        );

        nodes = vec![
            acl_record("edge-hidden-endpoint", "edge", None, "public"),
            acl_record("visible-node", "node", None, "public"),
            GraphRecord::new("visible-node"),
        ];
        endpoint_edge
            .metadata
            .insert("source_ids".into(), serde_json::json!(["visible-node"]));
        let (_, visible_edges) =
            graph_records_visible_by_acl(&nodes, &[endpoint_edge], "knowledge", &principal);
        assert_eq!(
            visible_edges.len(),
            1,
            "legacy edges without mention metadata remain readable with visible endpoints"
        );

        let mut malformed_reference = GraphRecord::new("edge-malformed-reference");
        malformed_reference
            .metadata
            .insert("source_edge_ids".into(), "not-json".into());
        nodes.push(acl_record(
            "edge-malformed-reference",
            "edge",
            None,
            "public",
        ));
        let (_, visible_edges) =
            graph_records_visible_by_acl(&nodes, &[malformed_reference], "knowledge", &principal);
        assert!(
            visible_edges.is_empty(),
            "malformed edge-reference metadata must fail closed"
        );
    }

    #[test]
    fn graph_acl_projection_excludes_tombstoned_entities_and_their_edges() {
        let acl_record = |id: &str, grain: &str| {
            let mut record = GraphRecord::new(format!("acl-{id}-{grain}"));
            record.metadata.extend(
                serde_json::json!({
                    "entity_type": "acl_record",
                    "acl_truth_graph": "knowledge",
                    "acl_target_entity_id": id,
                    "acl_target_grain": grain,
                    "acl_version": 1,
                    "acl_mode": "public"
                })
                .as_object()
                .expect("ACL metadata is an object")
                .clone(),
            );
            record
        };
        let mut tombstoned_node = GraphRecord::new("node-deleted");
        tombstoned_node
            .metadata
            .insert("lifecycle_status".into(), "tombstoned".into());
        let mut edge = GraphRecord::new("edge-to-deleted-node");
        edge.metadata
            .insert("source_ids".into(), serde_json::json!(["node-active"]));
        edge.metadata
            .insert("target_ids".into(), serde_json::json!(["node-deleted"]));
        let nodes = vec![
            acl_record("node-active", "node"),
            acl_record("node-deleted", "node"),
            acl_record("edge-to-deleted-node", "edge"),
            GraphRecord::new("node-active"),
            tombstoned_node,
        ];
        let principal = GraphAclPrincipal::default();

        let (visible_nodes, visible_edges) =
            graph_records_visible_by_acl(&nodes, &[edge], "knowledge", &principal);

        assert_eq!(
            visible_nodes
                .iter()
                .map(|record| record.id.as_str())
                .collect::<Vec<_>>(),
            ["node-active"]
        );
        assert!(visible_edges.is_empty());
    }

    #[test]
    fn graph_edge_visibility_fails_closed_on_non_string_endpoint_ids() {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/graph-acl-malformed-endpoint-v1.json"
        )))
        .expect("valid malformed endpoint ACL fixture");
        let edge_id = "malformed-endpoint-edge";
        let endpoint_id = fixture["source_endpoint_id"].as_str().unwrap();
        let acl_record = |id: &str, grain: &str, mode: &str| {
            let mut record = GraphRecord::new(format!("acl-{id}-{grain}"));
            record
                .metadata
                .insert("entity_type".into(), "acl_record".into());
            record
                .metadata
                .insert("acl_truth_graph".into(), fixture["truth_graph"].clone());
            record
                .metadata
                .insert("acl_target_entity_id".into(), id.into());
            record
                .metadata
                .insert("acl_target_grain".into(), grain.into());
            record
                .metadata
                .insert("acl_version".into(), serde_json::json!(1));
            record.metadata.insert("acl_mode".into(), mode.into());
            if mode == "private" {
                record
                    .metadata
                    .insert("owner_id".into(), "someone-else".into());
            }
            record
        };

        let mut edge = GraphRecord::new(edge_id);
        edge.metadata.insert(
            "source_ids".into(),
            serde_json::json!([fixture["source_endpoint_value"]]),
        );
        edge.metadata
            .insert("target_ids".into(), serde_json::json!([]));
        let nodes = vec![
            acl_record(edge_id, "edge", "public"),
            acl_record(endpoint_id, "node", "private"),
            GraphRecord::new(endpoint_id),
        ];
        let principal = GraphAclPrincipal {
            principal_id: fixture["principal"]["id"].as_str().unwrap().into(),
            groups: BTreeSet::new(),
            security_scope: fixture["principal"]["security_scope"]
                .as_str()
                .map(str::to_owned),
        };
        let (_, visible_edges) = graph_records_visible_by_acl(
            &nodes,
            &[edge],
            fixture["truth_graph"].as_str().unwrap(),
            &principal,
        );
        assert_eq!(
            !visible_edges.is_empty(),
            fixture["expected_edge_visible"].as_bool().unwrap()
        );
    }

    #[test]
    fn graph_edge_usage_matches_shared_python_golden_contract() {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/graph-acl-edge-usage-v1.json"
        )))
        .expect("valid graph edge ACL fixture");
        assert_eq!(fixture["schema_version"], 1);
        let principal = GraphAclPrincipal {
            principal_id: fixture["principal"]["id"]
                .as_str()
                .expect("principal ID")
                .to_owned(),
            groups: BTreeSet::new(),
            security_scope: fixture["principal"]["security_scope"]
                .as_str()
                .map(str::to_owned),
        };
        let acl_record = |entity_id: &str, grain: &str, item_id: Option<&str>, mode: &str| {
            let mut record = GraphRecord::new(format!(
                "acl-{entity_id}-{grain}-{}",
                item_id.unwrap_or("entity")
            ));
            record
                .metadata
                .insert("entity_type".into(), "acl_record".into());
            record
                .metadata
                .insert("acl_truth_graph".into(), fixture["truth_graph"].clone());
            record
                .metadata
                .insert("acl_target_entity_id".into(), entity_id.into());
            record
                .metadata
                .insert("acl_target_grain".into(), grain.into());
            record
                .metadata
                .insert("acl_version".into(), serde_json::json!(1));
            record.metadata.insert("acl_mode".into(), mode.into());
            if let Some(item_id) = item_id {
                record
                    .metadata
                    .insert("acl_target_item_id".into(), item_id.into());
            }
            if mode == "private" {
                record.metadata.insert(
                    "owner_id".into(),
                    fixture["owner_id"].as_str().expect("owner ID").into(),
                );
            } else if mode == "shared" {
                record.metadata.insert(
                    "shared_with_principals".into(),
                    fixture["shared_with_principals"].clone(),
                );
            }
            record
        };

        for case in fixture["cases"].as_array().expect("fixture cases") {
            let edge_id = case["edge_id"].as_str().expect("edge ID");
            let source_id = case["source_node_id"].as_str().expect("source ID");
            let target_id = case["target_node_id"].as_str().expect("target ID");
            let mut nodes = vec![acl_record(edge_id, "edge", None, "shared")];
            for (endpoint_id, exists, mode) in [
                (
                    source_id,
                    case["source_node_exists"].as_bool().unwrap_or(true),
                    case["source_node_mode"].as_str().expect("source ACL mode"),
                ),
                (
                    target_id,
                    case["target_node_exists"].as_bool().unwrap_or(true),
                    case["target_node_mode"].as_str().expect("target ACL mode"),
                ),
            ] {
                if exists {
                    nodes.push(acl_record(endpoint_id, "node", None, mode));
                    nodes.push(GraphRecord::new(endpoint_id));
                }
            }
            let mut edge = GraphRecord::new(edge_id);
            edge.metadata
                .insert("source_ids".into(), serde_json::json!([source_id]));
            edge.metadata
                .insert("target_ids".into(), serde_json::json!([target_id]));
            let mut mentions = Vec::new();
            for (mention_index, span_count) in case["mention_span_counts"]
                .as_array()
                .expect("mention span counts")
                .iter()
                .enumerate()
            {
                let grounding_id = format!("{edge_id}::mention::{mention_index}");
                nodes.push(acl_record(
                    edge_id,
                    "grounding",
                    Some(&grounding_id),
                    "public",
                ));
                let span_count = span_count.as_u64().expect("span count") as usize;
                let mut spans = Vec::new();
                for span_index in 0..span_count {
                    let span_id = format!("{grounding_id}::span::{span_index}");
                    let private = case["private_span"].as_array().is_some_and(|indices| {
                        indices[0].as_u64() == Some(mention_index as u64)
                            && indices[1].as_u64() == Some(span_index as u64)
                    });
                    nodes.push(acl_record(
                        edge_id,
                        "span",
                        Some(&span_id),
                        if private { "private" } else { "public" },
                    ));
                    spans.push(serde_json::json!({}));
                }
                mentions.push(serde_json::json!({"spans": spans}));
            }
            if !mentions.is_empty() {
                edge.metadata
                    .insert("mentions".into(), serde_json::json!(mentions));
            }
            let mut edges = vec![edge];
            if let Some(referenced_edge_id) = case["source_edge_id"].as_str() {
                edges[0].metadata.insert(
                    "source_edge_ids".into(),
                    serde_json::json!([referenced_edge_id]),
                );
                if case["source_edge_exists"].as_bool().unwrap_or(true) {
                    nodes.push(acl_record(referenced_edge_id, "edge", None, "private"));
                    edges.push(GraphRecord::new(referenced_edge_id));
                }
            }
            let (_, visible_edges) = graph_records_visible_by_acl(
                &nodes,
                &edges,
                fixture["truth_graph"].as_str().expect("truth graph"),
                &principal,
            );
            assert_eq!(
                visible_edges.iter().any(|record| record.id == edge_id),
                case["expected_visible"]
                    .as_bool()
                    .expect("expected visibility"),
                "{edge_id}"
            );
        }
    }

    #[test]
    fn graph_node_malformed_item_usage_fails_closed_but_legacy_absence_remains_readable() {
        let principal = GraphAclPrincipal {
            principal_id: "alice".into(),
            groups: BTreeSet::new(),
            security_scope: None,
        };
        let acl = |node_id: &str| {
            let mut record = GraphRecord::new(format!("acl-{node_id}"));
            record
                .metadata
                .insert("entity_type".into(), "acl_record".into());
            record
                .metadata
                .insert("acl_truth_graph".into(), "knowledge".into());
            record
                .metadata
                .insert("acl_target_entity_id".into(), node_id.into());
            record
                .metadata
                .insert("acl_target_grain".into(), "node".into());
            record
                .metadata
                .insert("acl_version".into(), serde_json::json!(1));
            record.metadata.insert("acl_mode".into(), "public".into());
            record
        };
        let mut malformed_json = GraphRecord::new("malformed-json");
        malformed_json
            .metadata
            .insert("mentions".into(), "not-json".into());
        let mut wrong_shape = GraphRecord::new("wrong-shape");
        wrong_shape
            .metadata
            .insert("mentions".into(), serde_json::json!({"unexpected": true}));
        let legacy_without_mentions = GraphRecord::new("legacy-without-mentions");
        let nodes = vec![
            acl("malformed-json"),
            malformed_json,
            acl("wrong-shape"),
            wrong_shape,
            acl("legacy-without-mentions"),
            legacy_without_mentions,
        ];

        let (visible, _) = graph_records_visible_by_acl(&nodes, &[], "knowledge", &principal);
        let visible_ids = visible
            .iter()
            .map(|record| record.id.as_str())
            .collect::<BTreeSet<_>>();
        assert_eq!(visible_ids, BTreeSet::from(["legacy-without-mentions"]));
    }

    #[test]
    fn graph_acl_history_matches_shared_golden_contract() {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/graph-acl-history-v1.json"
        )))
        .expect("valid graph ACL history fixture");
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
            for (index, rule) in target["acl_records"]
                .as_array()
                .expect("ACL record history")
                .iter()
                .enumerate()
            {
                let mut acl = GraphRecord::new(format!("acl-{id}-{index}"));
                acl.metadata
                    .insert("entity_type".into(), "acl_record".into());
                acl.metadata
                    .insert("acl_truth_graph".into(), fixture["truth_graph"].clone());
                acl.metadata
                    .insert("acl_target_entity_id".into(), id.into());
                acl.metadata.insert("acl_target_grain".into(), grain.into());
                acl.metadata.insert(
                    "acl_version".into(),
                    rule["version"].as_i64().expect("version").into(),
                );
                acl.metadata.insert(
                    "acl_mode".into(),
                    rule["mode"].as_str().expect("mode").into(),
                );
                if let Some(owner) = rule["owner_id"].as_str() {
                    acl.metadata.insert("owner_id".into(), owner.into());
                }
                if let Some(scope) = rule["security_scope"].as_str() {
                    acl.metadata.insert("security_scope".into(), scope.into());
                }
                if rule["tombstoned"].as_bool().unwrap_or(false) {
                    acl.metadata.insert("tombstoned".into(), true.into());
                }
                let mut properties = serde_json::Map::new();
                for field in ["shared_with_principals", "shared_with_groups"] {
                    if let Some(values) = rule[field].as_array() {
                        properties.insert(field.into(), Value::Array(values.clone()));
                    }
                }
                if !properties.is_empty() {
                    acl.document = Some(serde_json::json!({"properties": properties}).to_string());
                }
                nodes.push(acl);
            }
            if grain == "node" {
                nodes.push(target_record);
            } else {
                edges.push(target_record);
            }
        }

        let (visible_nodes, visible_edges) = graph_records_visible_by_acl(
            &nodes,
            &edges,
            fixture["truth_graph"].as_str().expect("truth graph"),
            &principal,
        );
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
            .map(|value| value.as_str().expect("node ID"))
            .collect::<Vec<_>>();
        let expected_edges = fixture["expected_visible"]["edges"]
            .as_array()
            .expect("expected edges")
            .iter()
            .map(|value| value.as_str().expect("edge ID"))
            .collect::<Vec<_>>();
        assert_eq!(node_ids, expected_nodes);
        assert_eq!(edge_ids, expected_edges);

        let reversed_nodes = nodes.iter().rev().cloned().collect::<Vec<_>>();
        let reversed_edges = edges.iter().rev().cloned().collect::<Vec<_>>();
        let (reordered_visible_nodes, reordered_visible_edges) = graph_records_visible_by_acl(
            &reversed_nodes,
            &reversed_edges,
            fixture["truth_graph"].as_str().expect("truth graph"),
            &principal,
        );
        let mut reordered_node_ids = reordered_visible_nodes
            .iter()
            .map(|record| record.id.as_str())
            .collect::<Vec<_>>();
        let mut reordered_edge_ids = reordered_visible_edges
            .iter()
            .map(|record| record.id.as_str())
            .collect::<Vec<_>>();
        reordered_node_ids.sort_unstable();
        reordered_edge_ids.sort_unstable();
        let mut expected_node_ids = expected_nodes.clone();
        let mut expected_edge_ids = expected_edges.clone();
        expected_node_ids.sort_unstable();
        expected_edge_ids.sort_unstable();
        assert_eq!(reordered_node_ids, expected_node_ids);
        assert_eq!(reordered_edge_ids, expected_edge_ids);
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

        let mut result = GraphRecord::new("tool-result-1");
        result
            .metadata
            .insert("entity_type".into(), "tool_result".into());
        result
            .metadata
            .insert("tool_name".into(), "graph.read".into());
        result
            .metadata
            .insert("conversation_id".into(), "conversation-1".into());
        result.document = Some(
            serde_json::json!({
                "properties": {"status": "completed"}
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
            &[result, call, other, tombstoned],
            Some("conversation-1"),
            10,
        )
        .unwrap();
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].node_id, "tool-call-1");
        assert_eq!(items[0].turn_index, Some(2));
        assert_eq!(items[0].properties["status"], "completed");
        assert_eq!(items[1].node_id, "tool-result-1");
        assert!(
            tool_audit_items_from_visible_records(&[GraphRecord::new("ignored")], None, 0)
                .unwrap()
                .is_empty()
        );
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
    fn conversation_list_projection_is_user_scoped_and_counts_visible_turns() {
        let mut start_alice = GraphRecord::new("start-alice");
        start_alice.metadata.extend(
            serde_json::json!({
                "entity_type": "conversation_start",
                "conversation_id": "conversation-alice",
                "user_id": "alice",
                "status": "active"
            })
            .as_object()
            .unwrap()
            .clone(),
        );
        let mut start_bob = start_alice.clone();
        start_bob.id = "start-bob".to_owned();
        start_bob
            .metadata
            .insert("conversation_id".to_owned(), "conversation-bob".into());
        start_bob
            .metadata
            .insert("user_id".to_owned(), "bob".into());
        let mut first_turn = GraphRecord::new("turn-1");
        first_turn.metadata.extend(
            serde_json::json!({
                "entity_type": "conversation_turn",
                "conversation_id": "conversation-alice",
                "turn_index": 0
            })
            .as_object()
            .unwrap()
            .clone(),
        );
        let mut second_turn = first_turn.clone();
        second_turn.id = "turn-2".to_owned();
        second_turn
            .metadata
            .insert("entity_type".to_owned(), "assistant_turn".into());
        second_turn
            .metadata
            .insert("turn_index".to_owned(), 1.into());

        let records = [start_alice, start_bob, first_turn, second_turn];
        let entries = conversation_list_for_user_from_visible_records(&records, "alice");
        assert_eq!(
            entries,
            vec![ConversationListEntry {
                id: "conversation-alice".to_owned(),
                start_node_id: "start-alice".to_owned(),
                status: "active".to_owned(),
                turn_count: 2,
            }]
        );

        let mut chunked = ConversationListAccumulator::new("alice");
        chunked.extend_visible_records(&records[..1]);
        chunked.extend_visible_records(&records[1..3]);
        chunked.extend_visible_records(&records[3..]);
        assert_eq!(chunked.finish(), entries);
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
    fn context_snapshot_projection_matches_python_run_and_attempt_order_fixture() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/context-snapshot-order-v1.json"
        )))
        .expect("valid shared Python context-snapshot ordering fixture");
        let conversation_id = fixture["conversation_id"].as_str().unwrap();
        let run_id = fixture["run_id"].as_str().unwrap();
        let stage = fixture["stage"].as_str().unwrap();
        let records = fixture["candidates"]
            .as_array()
            .unwrap()
            .iter()
            .map(|candidate| {
                let id = candidate["id"].as_str().unwrap().to_owned();
                let metadata = serde_json::json!({
                    "entity_type": "context_snapshot",
                    "conversation_id": conversation_id,
                    "run_id": run_id,
                    "stage": stage,
                    "run_step_seq": candidate["run_step_seq"],
                    "attempt_seq": candidate["attempt_seq"]
                })
                .as_object()
                .unwrap()
                .clone();
                GraphRecord {
                    id,
                    document: Some(
                        serde_json::json!({
                            "metadata": metadata,
                            "properties": candidate["properties"]
                        })
                        .to_string(),
                    ),
                    metadata,
                    embedding: None,
                }
            })
            .collect::<Vec<_>>();

        let snapshot = context_snapshot_from_visible_records(
            &records,
            conversation_id,
            Some(run_id),
            Some(stage),
        )
        .expect("Python run_step_seq and attempt_seq metadata is orderable");
        assert_eq!(
            snapshot.snapshot_node_id,
            fixture["expected_snapshot_node_id"].as_str().unwrap()
        );
        assert_eq!(snapshot.properties["prompt_messages"], "latest retry");
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
        let wrong_edge_envelope =
            GraphModelEnvelope::new(GraphModelClass::Edge, node_payload.clone());
        let wrong_edge_mutation = typed_mutation("node", node_payload);
        let error = validate_typed_graph_mutation(&wrong_edge_mutation, &wrong_edge_envelope)
            .expect_err("malformed edge family must fail before the later kind mismatch");
        assert!(matches!(
            error,
            StoreError::InvalidEntityEventPayload { message } if message.contains("source_ids")
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
    fn direct_graph_mutation_validation_rejects_unrequested_scope_dimensions() {
        let mut mutation = typed_mutation("node", fixture_payload(0));
        mutation.scope.workspace_id = None;
        mutation.scope.graph_space = None;
        mutation.record.metadata.insert(
            "workspace_id".to_owned(),
            Value::String("foreign".to_owned()),
        );
        assert!(validate_graph_mutation(&mutation).is_err());

        mutation.record.metadata.remove("workspace_id");
        mutation.payload["metadata"]["graph_space"] = Value::String("foreign".to_owned());
        assert!(validate_graph_mutation(&mutation).is_err());
    }

    #[test]
    fn graph_mutation_replay_requires_identical_effect_identity_and_payload() {
        let mutation = typed_mutation("node", fixture_payload(0));
        let event = EntityEvent {
            namespace: mutation.scope.namespace.clone(),
            seq: 1,
            event_id: mutation.event_id.clone(),
            entity_kind: mutation.entity_kind.clone(),
            entity_id: mutation.record.id.clone(),
            op: mutation.op.clone(),
            payload: mutation.payload.clone(),
        };
        assert_eq!(validate_graph_mutation_replay(&mutation, &event), Ok(()));

        let mut altered = event.clone();
        altered.entity_kind = "edge".to_owned();
        assert!(validate_graph_mutation_replay(&mutation, &altered).is_err());
        let mut altered = event.clone();
        altered.entity_id.push_str("-other");
        assert!(validate_graph_mutation_replay(&mutation, &altered).is_err());
        let mut altered = event.clone();
        altered.op = "REPLACE".to_owned();
        assert!(validate_graph_mutation_replay(&mutation, &altered).is_err());
        let mut altered = event.clone();
        altered.payload["summary"] = Value::String("different".to_owned());
        assert!(validate_graph_mutation_replay(&mutation, &altered).is_err());
        let mut altered = event.clone();
        altered.event_id.push_str("-other");
        assert!(validate_graph_mutation_replay(&mutation, &altered).is_err());
        let mut altered = event;
        altered.namespace.push_str("-other");
        assert!(validate_graph_mutation_replay(&mutation, &altered).is_err());
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
    fn opaque_payload_metadata_does_not_select_storage_scope() {
        // Python's kg_graph_type is engine configuration, not an entity field.
        // A similarly named payload value must remain opaque to the store.
        for metadata_value in [
            "knowledge",
            "conversation",
            "workflow",
            "wisdom",
            "custom-role",
        ] {
            for (entity_kind, fixture_index) in [("node", 0), ("edge", 1)] {
                let mut mutation = typed_mutation(entity_kind, fixture_payload(fixture_index));
                mutation.payload["metadata"]["engine_type"] = Value::String(metadata_value.into());
                mutation.record.metadata.insert(
                    "engine_type".to_owned(),
                    Value::String(metadata_value.into()),
                );

                assert_eq!(mutation.scope.graph_space.as_deref(), Some("facts"));
                assert!(
                    validate_graph_mutation_batch(&GraphMutationBatch {
                        mutations: vec![mutation],
                        index_jobs: Vec::new(),
                    })
                    .is_ok(),
                    "opaque metadata {metadata_value:?} must not specialize {entity_kind} storage"
                );
            }
        }
    }

    #[test]
    fn graph_mutation_batch_rejects_index_retry_counts_outside_portable_range() {
        let mutation = typed_mutation("node", fixture_payload(0));
        for max_retries in [-1, i32::MAX as i64 + 1] {
            let result = validate_graph_mutation_batch(&GraphMutationBatch {
                mutations: vec![mutation.clone()],
                index_jobs: vec![NewIndexJob {
                    job_id: "job-1".to_owned(),
                    namespace: mutation.scope.namespace.clone(),
                    entity_kind: "node".to_owned(),
                    entity_id: mutation.record.id.clone(),
                    index_kind: "doc".to_owned(),
                    op: "UPSERT".to_owned(),
                    payload_json: None,
                    max_retries,
                }],
            });
            assert!(matches!(
                result,
                Err(StoreError::InvalidEntityEventPayload { .. })
            ));
        }

        let boundary = validate_graph_mutation_batch(&GraphMutationBatch {
            mutations: vec![mutation.clone()],
            index_jobs: vec![NewIndexJob {
                job_id: "job-boundary".to_owned(),
                namespace: mutation.scope.namespace.clone(),
                entity_kind: "node".to_owned(),
                entity_id: mutation.record.id.clone(),
                index_kind: "doc".to_owned(),
                op: "UPSERT".to_owned(),
                payload_json: None,
                max_retries: i32::MAX as i64,
            }],
        });
        assert!(boundary.is_ok());
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
