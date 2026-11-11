// Graph ACL and authorized conversation/read projections. Python parity: kogwistar.acl.graph, graph_query, and server conversation-query services.
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
