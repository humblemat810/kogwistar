fn graph_record_string_ids(record: &GraphRecord, field: &str) -> Vec<String> {
    let document = record
        .document
        .as_deref()
        .and_then(|document| serde_json::from_str::<Value>(document).ok());
    let field_value = record
        .metadata
        .get(field)
        .or_else(|| document.as_ref().and_then(|document| document.get(field)));
    let decoded_field = field_value.and_then(|value| match value {
        Value::String(encoded) => serde_json::from_str::<Value>(encoded).ok(),
        value => Some(value.clone()),
    });
    decoded_field
        .as_ref()
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .collect()
}

fn graph_record_string_id(record: &GraphRecord, field: &str) -> Option<String> {
    let document = record
        .document
        .as_deref()
        .and_then(|document| serde_json::from_str::<Value>(document).ok());
    record
        .metadata
        .get(field)
        .or_else(|| document.as_ref().and_then(|document| document.get(field)))
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .map(str::to_owned)
}

fn brute_force_pair_kinds(pair_kind: &str) -> Option<(&'static str, &'static str)> {
    match pair_kind {
        "node_node" => Some(("node", "node")),
        "edge_edge" => Some(("edge", "edge")),
        "node_edge" => Some(("node", "edge")),
        "any_any" => Some(("any", "any")),
        "any_node" => Some(("any", "node")),
        "any_edge" => Some(("any", "edge")),
        "node_any" => Some(("node", "any")),
        "edge_any" => Some(("edge", "any")),
        _ => None,
    }
}

fn brute_force_document_indexes_from_records(
    nodes: &[GraphRecord],
    edges: &[GraphRecord],
) -> (
    BTreeMap<String, BTreeSet<String>>,
    BTreeMap<String, BTreeSet<String>>,
) {
    let mut node_ids_by_doc = BTreeMap::<String, BTreeSet<String>>::new();
    let mut docs_by_node = BTreeMap::<String, BTreeSet<String>>::new();
    for node in nodes {
        let mut doc_ids = graph_record_string_ids(node, "doc_ids")
            .into_iter()
            .collect::<BTreeSet<_>>();
        if let Some(doc_id) = graph_record_string_id(node, "doc_id") {
            doc_ids.insert(doc_id);
        }
        for doc_id in &doc_ids {
            node_ids_by_doc
                .entry(doc_id.clone())
                .or_default()
                .insert(node.id.clone());
        }
        docs_by_node.insert(node.id.clone(), doc_ids);
    }
    let mut edge_ids_by_doc = BTreeMap::<String, BTreeSet<String>>::new();
    for edge in edges {
        let mut doc_ids = BTreeSet::new();
        for endpoint_id in graph_record_string_ids(edge, "source_ids")
            .into_iter()
            .chain(graph_record_string_ids(edge, "target_ids"))
        {
            doc_ids.extend(docs_by_node.get(&endpoint_id).into_iter().flatten().cloned());
        }
        if let Some(doc_id) = graph_record_string_id(edge, "doc_id") {
            doc_ids.insert(doc_id);
        }
        for doc_id in doc_ids {
            edge_ids_by_doc
                .entry(doc_id)
                .or_default()
                .insert(edge.id.clone());
        }
    }
    (node_ids_by_doc, edge_ids_by_doc)
}

fn graph_record_has_document_method(
    record: &GraphRecord,
    doc_id: &str,
    insertion_method: &str,
) -> bool {
    let Some(document) = record
        .document
        .as_deref()
        .and_then(|document| serde_json::from_str::<Value>(document).ok())
    else {
        return false;
    };
    document
        .get("mentions")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|mention| mention.get("spans").and_then(Value::as_array))
        .flatten()
        .any(|span| {
            let span_doc = span
                .get("doc_id")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .map(str::to_owned)
                .or_else(|| {
                    span.get("document_page_url")
                        .and_then(Value::as_str)
                        .and_then(|url| url.split("document/").nth(1))
                        .map(|suffix| {
                            suffix
                                .chars()
                                .take_while(|ch| ch.is_ascii_alphanumeric() || *ch == '-')
                                .collect::<String>()
                        })
                        .filter(|value| !value.is_empty())
                });
            let method = span
                .get("insertion_method")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .or_else(|| {
                    span.get("verification")
                        .and_then(|verification| verification.get("method"))
                        .and_then(Value::as_str)
                        .filter(|value| !value.is_empty())
                })
                .unwrap_or("unknown");
            span_doc.as_deref() == Some(doc_id) && method == insertion_method
        })
}

fn graph_load_persisted_ids_from_records(
    nodes: &[GraphRecord],
    edges: &[GraphRecord],
    input: &LoadPersistedGraphRequest,
) -> (Vec<String>, Vec<String>) {
    if let Some(method) = input.insertion_method.as_deref() {
        let node_ids = nodes
            .iter()
            .filter(|record| {
                !matches!(
                    record.metadata.get("lifecycle_status").and_then(Value::as_str),
                    Some("deleted" | "tombstoned")
                )
            })
            .filter(|record| {
                input
                    .doc_ids
                    .iter()
                    .any(|doc_id| graph_record_has_document_method(record, doc_id, method))
            })
            .map(|record| record.id.clone())
            .collect::<BTreeSet<_>>();
        let edge_ids = edges
            .iter()
            .filter(|record| {
                !matches!(
                    record.metadata.get("lifecycle_status").and_then(Value::as_str),
                    Some("deleted" | "tombstoned")
                )
            })
            .filter(|record| {
                input
                    .doc_ids
                    .iter()
                    .any(|doc_id| graph_record_has_document_method(record, doc_id, method))
            })
            .map(|record| record.id.clone())
            .collect::<BTreeSet<_>>();
        return (node_ids.into_iter().collect(), edge_ids.into_iter().collect());
    }

    let mut docs_by_node = BTreeMap::<String, BTreeSet<String>>::new();
    let mut node_ids = BTreeSet::new();
    for node in nodes {
        if matches!(
            node.metadata.get("lifecycle_status").and_then(Value::as_str),
            Some("deleted" | "tombstoned")
        ) {
            continue;
        }
        let mut doc_ids = graph_record_string_ids(node, "doc_ids")
            .into_iter()
            .collect::<BTreeSet<_>>();
        if let Some(doc_id) = graph_record_string_id(node, "doc_id") {
            doc_ids.insert(doc_id);
        }
        if doc_ids.iter().any(|doc_id| input.doc_ids.contains(doc_id)) {
            node_ids.insert(node.id.clone());
        }
        docs_by_node.insert(node.id.clone(), doc_ids);
    }

    let docs_by_edge = edges
        .iter()
        .filter(|edge| {
            !matches!(
                edge.metadata.get("lifecycle_status").and_then(Value::as_str),
                Some("deleted" | "tombstoned")
            )
        })
        .map(|edge| {
            let mut doc_ids = graph_record_string_ids(edge, "doc_ids")
                .into_iter()
                .collect::<BTreeSet<_>>();
            if let Some(doc_id) = graph_record_string_id(edge, "doc_id") {
                doc_ids.insert(doc_id);
            }
            for endpoint_id in graph_record_string_ids(edge, "source_ids")
                .into_iter()
                .chain(graph_record_string_ids(edge, "target_ids"))
            {
                doc_ids.extend(docs_by_node.get(&endpoint_id).into_iter().flatten().cloned());
            }
            (edge.id.as_str(), doc_ids)
        })
        .collect::<BTreeMap<_, _>>();
    let edge_ids = docs_by_edge
        .into_iter()
        .filter(|(_, doc_ids)| doc_ids.iter().any(|doc_id| input.doc_ids.contains(doc_id)))
        .map(|(edge_id, _)| edge_id.to_owned())
        .collect::<BTreeSet<_>>();
    (node_ids.into_iter().collect(), edge_ids.into_iter().collect())
}

fn brute_force_proposal_pairs(
    doc_ids: &[String],
    node_ids_by_doc: &BTreeMap<String, BTreeSet<String>>,
    edge_ids_by_doc: &BTreeMap<String, BTreeSet<String>>,
    nodes: &BTreeMap<String, GraphRecord>,
    edges: &BTreeMap<String, GraphRecord>,
    input: &ProposeBruteForceRequest,
) -> Result<Vec<Value>, String> {
    const MAX_OUTPUT_PAIRS: usize = 50_000;
    const MAX_PAIR_EVALUATIONS: usize = 1_000_000;
    let (left_selector, right_selector) = brute_force_pair_kinds(&input.pair_kind)
        .ok_or_else(|| format!("unsupported pair_kind {:?}", input.pair_kind))?;
    let where_filter = input.where_filter.as_ref();
    let matching_ids = |records: &BTreeMap<String, GraphRecord>| -> Result<_, String> {
        let mut ids = BTreeSet::new();
        for (id, record) in records {
            let payload = json!({"metadata": record.metadata, "where": where_filter});
            if kogwistar_contracts::metadata_filter_matches_from_str(&payload.to_string())
                .map_err(|error| error.to_string())?
            {
                ids.insert(id.clone());
            }
        }
        Ok(ids)
    };
    let where_node_ids = if where_filter.is_some() {
        matching_ids(nodes)?
    } else {
        BTreeSet::new()
    };
    let where_edge_ids = if where_filter.is_some() {
        matching_ids(edges)?
    } else {
        BTreeSet::new()
    };
    let accepts_where = |id: &str, kind: &str| {
        if where_filter.is_none() {
            return true;
        }
        if kind == "node" {
            where_node_ids.is_empty() || where_node_ids.contains(id)
        } else {
            where_edge_ids.is_empty() || where_edge_ids.contains(id)
        }
    };
    let pool = |doc_id: &str, selector: &str| -> Vec<(String, &'static str)> {
        let mut result = Vec::new();
        if selector == "node" || selector == "any" {
            result.extend(
                node_ids_by_doc
                    .get(doc_id)
                    .into_iter()
                    .flatten()
                    .filter(|id| nodes.contains_key(*id))
                    .map(|id| (id.clone(), "node")),
            );
        }
        if selector == "edge" || selector == "any" {
            result.extend(
                edge_ids_by_doc
                    .get(doc_id)
                    .into_iter()
                    .flatten()
                    .filter(|id| edges.contains_key(*id))
                    .map(|id| (id.clone(), "edge")),
            );
        }
        result
    };
    let per_bucket = input.limit_per_bucket.filter(|limit| *limit > 0);
    let mut seen = BTreeSet::<((String, String), (String, String))>::new();
    let mut pairs = Vec::new();
    let mut pair_evaluations = 0_usize;
    let mut add_bucket = |left_doc: &str, right_doc: &str| -> Result<(), String> {
        let left = pool(left_doc, left_selector);
        let right = pool(right_doc, right_selector);
        let mut bucket_count = 0_usize;
        for (left_index, (left_id, left_kind)) in left.iter().enumerate() {
            for (right_index, (right_id, right_kind)) in right.iter().enumerate() {
                pair_evaluations += 1;
                if pair_evaluations > MAX_PAIR_EVALUATIONS {
                    return Err(format!(
                        "brute-force proposal exceeds the {MAX_PAIR_EVALUATIONS}-candidate evaluation limit"
                    ));
                }
                if left_id == right_id
                    || (left_doc == right_doc
                        && left_selector == right_selector
                        && right_index <= left_index)
                {
                    continue;
                }
                if per_bucket.is_some_and(|limit| bucket_count >= limit as usize) {
                    break;
                }
                bucket_count += 1;
                let key = (
                    (left_kind.to_string(), left_id.clone()),
                    (right_kind.to_string(), right_id.clone()),
                );
                let reverse = (key.1.clone(), key.0.clone());
                if seen.contains(&key) || seen.contains(&reverse) {
                    continue;
                }
                seen.insert(key);
                if accepts_where(left_id, left_kind) && accepts_where(right_id, right_kind) {
                    pairs.push(json!({
                        "left_id": left_id,
                        "left_kind": left_kind,
                        "right_id": right_id,
                        "right_kind": right_kind,
                    }));
                    if pairs.len() > MAX_OUTPUT_PAIRS {
                        return Err(format!(
                            "brute-force proposal exceeds the {MAX_OUTPUT_PAIRS}-pair response limit"
                        ));
                    }
                }
            }
        }
        Ok(())
    };

    if let Some(anchor) = input.anchor_doc_id.as_deref() {
        for doc_id in doc_ids {
            if doc_id == anchor && !input.cross_doc_only {
                add_bucket(anchor, anchor)?;
            } else if doc_id != anchor {
                add_bucket(anchor, doc_id)?;
                if !input.anchor_only && !input.cross_doc_only {
                    add_bucket(doc_id, doc_id)?;
                }
            }
        }
    } else {
        for (index, left_doc) in doc_ids.iter().enumerate() {
            if !input.cross_doc_only {
                add_bucket(left_doc, left_doc)?;
            }
            for right_doc in doc_ids.iter().skip(index + 1) {
                add_bucket(left_doc, right_doc)?;
            }
        }
    }
    Ok(pairs)
}

enum NativeEvidenceRecordError {
    Rejected,
    Unavailable,
}

impl PostgresRunApplicationService {
    // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/postgres_graph_service.rs#graph_propose_vector_effect; python=kogwistar/server/mcp_tools.py#propose_vector; fixture=contracts/golden/graph-vector-proposal-acl-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/graph_query_contracts.rs#postgres_graph_vector_query_is_scoped_acl_filtered_and_rank_stable_when_dsn_available; python_test=tests/mcp/test_chat_mcp_tools.py#test_propose_vector_returns_only_acl_visible_pairs
    async fn graph_propose_vector_effect(
        &self,
        request: &ApiEffectRequest,
    ) -> ApiEffectResponse {
        if !capability_allowed(request, &self.capabilities, "read_graph", &["read_graph"]) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Vector graph proposal requires read_graph capability",
            );
        }
        let input = match GraphVectorProposalRequest::parse(&request.body) {
            Ok(input) => input,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error,
                );
            }
        };
        let where_filter = input
            .where_filter
            .as_ref()
            .filter(|value| value.is_object());
        if let Err(error) = kogwistar_contracts::metadata_filter_matches_from_str(
            &json!({"metadata": {}, "where": where_filter}).to_string(),
        ) {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                error.to_string(),
            );
        }
        let scope_value = security_scope_value(&request.principal);
        let scope = GraphScope {
            namespace: scope_value["storage_namespace"]
                .as_str()
                .unwrap_or("default")
                .to_owned(),
            workspace_id: request
                .principal
                .get("workspace_id")
                .or_else(|| request.principal.get("workspace"))
                .and_then(Value::as_str)
                .map(str::to_owned),
            graph_space: request
                .principal
                .get("graph_space")
                .and_then(Value::as_str)
                .map(str::to_owned),
        };
        let (all_nodes, all_edges) = match self.graph_records_snapshot("knowledge", scope).await {
            Ok(records) => records,
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error,
                );
            }
        };
        let principal = graph_acl_principal_from_claims(
            &request.principal,
            scope_value["security_scope"].as_str(),
        );
        let (visible_nodes, visible_edges) =
            graph_records_visible_by_acl(&all_nodes, &all_edges, "knowledge", &principal);
        let pairs = match graph_vector_proposal_pairs(
            &all_nodes,
            &all_edges,
            &visible_nodes,
            &visible_edges,
            &input,
            where_filter,
        ) {
            Ok(pairs) => pairs,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error,
                );
            }
        };
        json_effect(
            StatusCode::OK,
            rust_native_attested(json!({"pairs": pairs})),
        )
    }

    // PYTHON_PARITY: kogwistar/server/mcp_tools.py::kg_propose_bruteforce
    async fn graph_propose_bruteforce_effect(
        &self,
        request: &ApiEffectRequest,
    ) -> ApiEffectResponse {
        if !capability_allowed(request, &self.capabilities, "read_graph", &["read_graph"]) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Brute-force graph proposal requires read_graph capability",
            );
        }
        let input = match serde_json::from_slice::<ProposeBruteForceRequest>(&request.body) {
            Ok(input) => input,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    format!("Invalid brute-force proposal request: {error}"),
                );
            }
        };
        if brute_force_pair_kinds(&input.pair_kind).is_none()
            || input.allowed_docs.as_ref().is_some_and(|docs| {
                docs.len() > 1000 || docs.iter().any(|doc| doc.trim().is_empty())
            })
            || input
                .anchor_doc_id
                .as_ref()
                .is_some_and(|doc| doc.trim().is_empty())
            || input
                .limit_per_bucket
                .is_some_and(|limit| limit > 5000)
            || input
                .where_filter
                .as_ref()
                .is_some_and(|value| !value.is_object())
        {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                "pair_kind, document IDs, where, or limit_per_bucket are invalid or exceed their bounds",
            );
        }
        if let Some(where_filter) = input.where_filter.as_ref() {
            if let Err(error) = kogwistar_contracts::metadata_filter_matches_from_str(
                &json!({"metadata": {}, "where": where_filter}).to_string(),
            ) {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error.to_string(),
                );
            }
        }

        let scope_value = security_scope_value(&request.principal);
        let scope = GraphScope {
            namespace: scope_value["storage_namespace"]
                .as_str()
                .unwrap_or("default")
                .to_owned(),
            workspace_id: request
                .principal
                .get("workspace_id")
                .or_else(|| request.principal.get("workspace"))
                .and_then(Value::as_str)
                .map(str::to_owned),
            graph_space: request
                .principal
                .get("graph_space")
                .and_then(Value::as_str)
                .map(str::to_owned),
        };
        let acl_principal = graph_acl_principal_from_claims(
            &request.principal,
            scope_value["security_scope"].as_str(),
        );
        let result = self
            .store
            .transaction(move |uow| {
                Box::pin(async move {
                    const MAX_INDEX_ROWS: usize = 100_000;
                    const MAX_CANDIDATE_ENTITIES: usize = 100_000;
                    let indexes = GraphIndexTableNames::default();
                    let read_index = |table: &str| GraphProjectionRead {
                        scope: scope.clone(),
                        table: table.to_owned(),
                        ids: None,
                        metadata: MetadataFilter::default(),
                        limit: MAX_INDEX_ROWS + 1,
                    };
                    // Python's `node_docs` join index is scope-free; use it
                    // only for candidate IDs, then reload graph entities with `scope` below.
                    let node_rows = uow
                        .graph_node_document_candidates(
                            &MetadataFilter::default(),
                            MAX_INDEX_ROWS + 1,
                        )
                        .await
                        .map_err(|error| PostgresStoreError::Backend(error.to_string()))?;
                    let edge_rows = uow
                        .graph_metadata_projection_records(read_index(&indexes.edge_endpoints))
                        .await
                        .map_err(|error| PostgresStoreError::Backend(error.to_string()))?;
                    let document_rows = uow
                        .graph_metadata_projection_records(read_index("gke_documents"))
                        .await
                        .map_err(|error| PostgresStoreError::Backend(error.to_string()))?;
                    if node_rows.len() > MAX_INDEX_ROWS
                        || edge_rows.len() > MAX_INDEX_ROWS
                        || document_rows.len() > MAX_INDEX_ROWS
                    {
                        return Err(PostgresStoreError::Backend(
                            "document index scan exceeds 100000 rows".to_owned(),
                        ));
                    }
                    let mut node_ids_by_doc = BTreeMap::<String, BTreeSet<String>>::new();
                    let mut edge_ids_by_doc = BTreeMap::<String, BTreeSet<String>>::new();
                    for row in node_rows {
                        if let (Some(doc_id), Some(node_id)) = (
                            graph_record_string_id(&row, "doc_id"),
                            graph_record_string_id(&row, "node_id"),
                        ) {
                            node_ids_by_doc.entry(doc_id).or_default().insert(node_id);
                        }
                    }
                    for row in edge_rows {
                        if let (Some(doc_id), Some(edge_id)) = (
                            graph_record_string_id(&row, "doc_id"),
                            graph_record_string_id(&row, "edge_id"),
                        ) {
                            edge_ids_by_doc.entry(doc_id).or_default().insert(edge_id);
                        }
                    }
                    let mut docs = match input.allowed_docs.as_ref().filter(|docs| !docs.is_empty()) {
                        Some(docs) => docs.iter().cloned().collect::<Vec<_>>(),
                        None => node_ids_by_doc
                            .keys()
                            .chain(edge_ids_by_doc.keys())
                            .chain(document_rows.iter().map(|record| &record.id))
                            .cloned()
                            .collect::<BTreeSet<_>>()
                            .into_iter()
                            .collect(),
                    };
                    if let Some(anchor) = input.anchor_doc_id.as_ref()
                        && !docs.contains(anchor)
                    {
                        docs.insert(0, anchor.clone());
                    }
                    if docs.len() > 1000 {
                        return Err(PostgresStoreError::Backend(
                            "brute-force proposal is limited to 1000 documents".to_owned(),
                        ));
                    }
                    if docs.is_empty() {
                        return Ok(Vec::new());
                    }
                    let selected_docs = docs.iter().collect::<BTreeSet<_>>();
                    let node_ids = node_ids_by_doc
                        .iter()
                        .filter(|(doc, _)| selected_docs.contains(doc))
                        .flat_map(|(_, ids)| ids.iter().cloned())
                        .collect::<BTreeSet<_>>();
                    let edge_ids = edge_ids_by_doc
                        .iter()
                        .filter(|(doc, _)| selected_docs.contains(doc))
                        .flat_map(|(_, ids)| ids.iter().cloned())
                        .collect::<BTreeSet<_>>();
                    if node_ids.len().saturating_add(edge_ids.len()) > MAX_CANDIDATE_ENTITIES {
                        return Err(PostgresStoreError::Backend(
                            "brute-force proposal exceeds 100000 indexed entities".to_owned(),
                        ));
                    }
                    let mut nodes = Vec::new();
                    for batch in node_ids.iter().cloned().collect::<Vec<_>>().chunks(1000) {
                        nodes.extend(
                            uow.graph_projection_records(GraphProjectionRead {
                                scope: scope.clone(),
                                table: indexes.nodes.clone(),
                                ids: Some(batch.to_vec()),
                                metadata: MetadataFilter::default(),
                                limit: batch.len(),
                            })
                            .await
                            .map_err(|error| PostgresStoreError::Backend(error.to_string()))?,
                        );
                    }
                    let mut edges = Vec::new();
                    for batch in edge_ids.iter().cloned().collect::<Vec<_>>().chunks(1000) {
                        edges.extend(
                            uow.graph_projection_records(GraphProjectionRead {
                                scope: scope.clone(),
                                table: indexes.edges.clone(),
                                ids: Some(batch.to_vec()),
                                metadata: MetadataFilter::default(),
                                limit: batch.len(),
                            })
                            .await
                            .map_err(|error| PostgresStoreError::Backend(error.to_string()))?,
                        );
                    }

                    // Match Python's document lookup: current Stage-1 candidates are
                    // eligible, but only after fencing them against canonical events.
                    let mut staged_nodes = BTreeMap::new();
                    let mut staged_edges = BTreeMap::new();
                    for (kind, ids_by_doc, staged_records) in [
                        ("node", &mut node_ids_by_doc, &mut staged_nodes),
                        ("edge", &mut edge_ids_by_doc, &mut staged_edges),
                    ] {
                        let staged = uow
                            .stage1_graph_projections_for_documents(&scope.namespace, kind, &docs)
                            .await
                            .map_err(|error| PostgresStoreError::Backend(error.to_string()))?;
                        let staged = staged
                            .into_iter()
                            .filter(|row| {
                                row.namespace == scope.namespace
                                    && row.materialization_status == "pending"
                                    && row.metadata.get("doc_id").and_then(Value::as_str)
                                        .is_some_and(|doc_id| docs.iter().any(|doc| doc == doc_id))
                                    && scope.workspace_id.as_ref().map_or(true, |workspace| {
                                        row.metadata
                                            .get("workspace_id")
                                            .and_then(Value::as_str)
                                            == Some(workspace)
                                    })
                                    && scope.graph_space.as_ref().map_or(true, |graph_space| {
                                        row.metadata
                                            .get("graph_space")
                                            .and_then(Value::as_str)
                                            == Some(graph_space)
                                    })
                            })
                            .collect::<Vec<_>>();
                        if staged.len() > MAX_CANDIDATE_ENTITIES {
                            return Err(PostgresStoreError::Backend(
                                "Stage-1 proposal scan exceeds 100000 candidates".to_owned(),
                            ));
                        }
                        let staged_ids = staged
                            .iter()
                            .map(|row| row.entity_id.clone())
                            .collect::<Vec<_>>();
                        let latest = uow
                            .latest_entity_events(&scope.namespace, kind, &staged_ids)
                            .await
                            .map_err(|error| PostgresStoreError::Backend(error.to_string()))?;
                        for row in staged {
                            let Some(event) = latest.get(&row.entity_id) else {
                                continue;
                            };
                            let payload = serde_json::from_str::<Value>(&event.payload_json)
                                .map_err(|error| {
                                    PostgresStoreError::Backend(format!(
                                        "invalid canonical event payload: {error}"
                                    ))
                                })?;
                            let canonical = read_canonical_entity_revision(
                                &[EntityEvent {
                                    namespace: event.namespace.clone(),
                                    seq: event.seq,
                                    event_id: event.event_id.clone(),
                                    entity_kind: event.entity_kind.clone(),
                                    entity_id: event.entity_id.clone(),
                                    op: event.op.clone(),
                                    payload,
                                }],
                                &scope.namespace,
                                kind,
                                &row.entity_id,
                            )
                            .map_err(|error| {
                                PostgresStoreError::Backend(error.to_string())
                            })?;
                            let Some(canonical) = canonical.filter(|revision| {
                                !revision.is_deleted() && revision.revision == row.revision
                            }) else {
                                continue;
                            };
                            let expected_fingerprint =
                                kogwistar_engine::canonical_revision::canonical_revision_job_payload(
                                    &canonical,
                                );
                            let expected_fingerprint = serde_json::from_str::<Value>(
                                &expected_fingerprint,
                            )
                            .ok()
                            .and_then(|value| {
                                value.get("source_fingerprint")?.as_str().map(str::to_owned)
                            });
                            if expected_fingerprint.as_deref()
                                != Some(row.source_fingerprint.as_str())
                            {
                                continue;
                            }
                            let doc_id = row
                                .metadata
                                .get("doc_id")
                                .and_then(Value::as_str)
                                .expect("filtered Stage-1 row has a document ID")
                                .to_owned();
                            ids_by_doc
                                .entry(doc_id)
                                .or_default()
                                .insert(row.entity_id.clone());
                            staged_records.insert(
                                row.entity_id.clone(),
                                GraphRecord {
                                    id: row.entity_id,
                                    document: Some(row.document),
                                    metadata: row.metadata,
                                    embedding: None,
                                },
                            );
                        }
                    }
                    if node_ids_by_doc
                        .values()
                        .map(BTreeSet::len)
                        .sum::<usize>()
                        .saturating_add(
                            edge_ids_by_doc.values().map(BTreeSet::len).sum::<usize>(),
                        )
                        > MAX_CANDIDATE_ENTITIES
                    {
                        return Err(PostgresStoreError::Backend(
                            "brute-force proposal exceeds 100000 indexed and staged entities"
                                .to_owned(),
                        ));
                    }
                    let existing_node_ids = nodes
                        .iter()
                        .map(|record| record.id.clone())
                        .collect::<BTreeSet<_>>();
                    nodes.extend(
                        staged_nodes
                            .into_iter()
                            .filter_map(|(id, record)| (!existing_node_ids.contains(&id)).then_some(record)),
                    );
                    let existing_edge_ids = edges
                        .iter()
                        .map(|record| record.id.clone())
                        .collect::<BTreeSet<_>>();
                    edges.extend(
                        staged_edges
                            .into_iter()
                            .filter_map(|(id, record)| (!existing_edge_ids.contains(&id)).then_some(record)),
                    );
                    let entity_ids = nodes
                        .iter()
                        .chain(edges.iter())
                        .map(|record| record.id.clone())
                        .collect::<BTreeSet<_>>();
                    let mut acl_index = GraphAclIndex::default();
                    let entity_ids = entity_ids.into_iter().collect::<Vec<_>>();
                    for batch in entity_ids.chunks(10_000) {
                        let acl_records = uow
                            .graph_acl_records_for_entities(&scope, "knowledge", batch)
                            .await
                            .map_err(|error| PostgresStoreError::Backend(error.to_string()))?;
                        acl_index.extend_from_records(&acl_records, "knowledge");
                    }
                    let (nodes, edges) = graph_records_visible_by_acl_with_index(
                        &nodes,
                        &edges,
                        &acl_index,
                        &acl_principal,
                    );
                    let nodes = nodes
                        .into_iter()
                        .filter(|record| {
                            !matches!(
                                record
                                    .metadata
                                    .get("lifecycle_status")
                                    .and_then(Value::as_str),
                                Some("deleted" | "tombstoned")
                            )
                        })
                        .map(|record| (record.id.clone(), record))
                        .collect::<BTreeMap<_, _>>();
                    let edges = edges
                        .into_iter()
                        .filter(|record| {
                            !matches!(
                                record
                                    .metadata
                                    .get("lifecycle_status")
                                    .and_then(Value::as_str),
                                Some("deleted" | "tombstoned")
                            )
                        })
                        .map(|record| (record.id.clone(), record))
                        .collect::<BTreeMap<_, _>>();
                    brute_force_proposal_pairs(
                        &docs,
                        &node_ids_by_doc,
                        &edge_ids_by_doc,
                        &nodes,
                        &edges,
                        &input,
                    )
                    .map_err(PostgresStoreError::Backend)
                })
            })
            .await;
        match result {
            Ok(pairs) => json_effect(
                StatusCode::OK,
                rust_native_attested(json!({"pairs": pairs})),
            ),
            Err(error) => effect_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "KOGWISTAR_STORE_ERROR",
                error.to_string(),
            ),
        }
    }

    // PYTHON_PARITY: kogwistar/server/mcp_tools.py::_load_persisted_graph
    async fn graph_load_persisted_effect(
        &self,
        request: &ApiEffectRequest,
    ) -> ApiEffectResponse {
        if !capability_allowed(request, &self.capabilities, "read_graph", &["read_graph"]) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Persisted graph lookup requires read_graph capability",
            );
        }
        let input = match serde_json::from_slice::<LoadPersistedGraphRequest>(&request.body) {
            Ok(input) => input,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    format!("Invalid persisted graph lookup request: {error}"),
                );
            }
        };
        if input.doc_ids.is_empty()
            || input.doc_ids.len() > 1000
            || input.doc_ids.iter().any(|doc| doc.trim().is_empty())
            || input
                .insertion_method
                .as_ref()
                .is_some_and(|method| method.trim().is_empty())
        {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                "doc_ids must contain 1 to 1000 non-empty IDs and insertion_method must be non-empty when provided",
            );
        }

        let scope_value = security_scope_value(&request.principal);
        let scope = GraphScope {
            namespace: scope_value["storage_namespace"]
                .as_str()
                .unwrap_or("default")
                .to_owned(),
            workspace_id: request
                .principal
                .get("workspace_id")
                .or_else(|| request.principal.get("workspace"))
                .and_then(Value::as_str)
                .map(str::to_owned),
            graph_space: request
                .principal
                .get("graph_space")
                .and_then(Value::as_str)
                .map(str::to_owned),
        };
        let acl_principal = graph_acl_principal_from_claims(
            &request.principal,
            scope_value["security_scope"].as_str(),
        );
        let docs = input.doc_ids.clone();
        let insertion_method = input.insertion_method.clone();
        let namespace = scope.namespace.clone();
        let result = self
            .store
            .transaction(move |uow| {
                Box::pin(async move {
                    let indexes = GraphIndexTableNames::default();
                    let mut node_ids = BTreeSet::new();
                    let mut edge_ids = BTreeSet::new();
                    let mut valid_stage1_nodes = BTreeMap::new();
                    let mut valid_stage1_edges = BTreeMap::new();
                    for doc_id in &docs {
                        let index_filter = |with_method: bool| {
                            let mut equals = BTreeMap::new();
                            equals.insert("doc_id".to_owned(), json!(doc_id));
                            if with_method {
                                if let Some(method) = insertion_method.as_ref() {
                                    equals.insert("insertion_method".to_owned(), json!(method));
                                }
                            }
                            MetadataFilter { equals }
                        };
                        let node_table = if insertion_method.is_some() {
                            &indexes.node_refs
                        } else {
                            &indexes.node_docs
                        };
                        let edge_table = if insertion_method.is_some() {
                            &indexes.edge_refs
                        } else {
                            &indexes.edge_endpoints
                        };
                        let node_filter = index_filter(insertion_method.is_some());
                        let node_rows = if insertion_method.is_none() {
                            uow.graph_node_document_candidates(&node_filter, 100_001)
                                .await
                        } else {
                            uow.graph_metadata_projection_records(GraphProjectionRead {
                                scope: scope.clone(),
                                table: node_table.clone(),
                                ids: None,
                                metadata: node_filter,
                                limit: 100_001,
                            })
                            .await
                        }
                        .map_err(|error| PostgresStoreError::Backend(error.to_string()))?;
                        let edge_rows = uow
                            .graph_metadata_projection_records(GraphProjectionRead {
                                scope: scope.clone(),
                                table: edge_table.clone(),
                                ids: None,
                                metadata: index_filter(insertion_method.is_some()),
                                limit: 100_001,
                            })
                            .await
                            .map_err(|error| PostgresStoreError::Backend(error.to_string()))?;
                        if node_rows.len() > 100_000 || edge_rows.len() > 100_000 {
                            return Err(PostgresStoreError::Backend(
                                "document join-index result exceeds 100000 records".to_owned(),
                            ));
                        }
                        for row in node_rows {
                            if let Some(id) = graph_record_string_id(&row, "node_id") {
                                node_ids.insert(id);
                            }
                        }
                        for row in edge_rows {
                            if let Some(id) = graph_record_string_id(&row, "edge_id") {
                                edge_ids.insert(id);
                            }
                        }
                    }

                    if insertion_method.is_none() {
                        for (kind, ids, valid_rows) in [
                            ("node", &mut node_ids, &mut valid_stage1_nodes),
                            ("edge", &mut edge_ids, &mut valid_stage1_edges),
                        ] {
                            let staged = uow
                                .stage1_graph_projections_for_documents(&namespace, kind, &docs)
                                .await.map_err(|error| PostgresStoreError::Backend(error.to_string()))?;
                            let staged = staged.into_iter().filter(|row| {
                                row.materialization_status == "pending"
                                    && row.metadata.get("doc_id").and_then(Value::as_str)
                                        .is_some_and(|doc_id| docs.iter().any(|requested| requested == doc_id))
                            }).collect::<Vec<_>>();
                            let entity_ids = staged.iter().map(|row| row.entity_id.clone()).collect::<Vec<_>>();
                            let latest = uow.latest_entity_events(&namespace, kind, &entity_ids)
                                .await.map_err(|error| PostgresStoreError::Backend(error.to_string()))?;
                            for row in staged {
                                let Some(event) = latest.get(&row.entity_id) else { continue };
                                let payload = serde_json::from_str::<Value>(&event.payload_json)
                                    .map_err(|error| PostgresStoreError::Backend(format!("invalid canonical event payload: {error}")))?;
                                let canonical = read_canonical_entity_revision(&[EntityEvent {
                                    namespace: event.namespace.clone(), seq: event.seq,
                                    event_id: event.event_id.clone(), entity_kind: event.entity_kind.clone(),
                                    entity_id: event.entity_id.clone(), op: event.op.clone(), payload,
                                }], &namespace, kind, &row.entity_id)
                                    .map_err(|error| PostgresStoreError::Backend(error.to_string()))?;
                                let Some(canonical) = canonical.filter(|revision| {
                                    !revision.is_deleted() && revision.revision == row.revision
                                }) else { continue };
                                let canonical_payload = kogwistar_engine::canonical_revision::canonical_revision_job_payload(&canonical);
                                let expected = serde_json::from_str::<Value>(&canonical_payload)
                                    .ok().and_then(|value| value.get("source_fingerprint").and_then(Value::as_str).map(str::to_owned));
                                if expected.as_deref() != Some(row.source_fingerprint.as_str()) { continue; }
                                ids.insert(row.entity_id.clone());
                                valid_rows.insert(row.entity_id.clone(), row);
                            }
                        }
                    }

                    if node_ids.len().saturating_add(edge_ids.len()) > 100_000 {
                        return Err(PostgresStoreError::Backend(
                            "persisted graph result exceeds 100000 entities".to_owned(),
                        ));
                    }
                    let read = |table: &str, ids: &[String]| GraphProjectionRead {
                        scope: scope.clone(), table: table.to_owned(),
                        ids: Some(ids.to_vec()),
                        metadata: MetadataFilter::default(), limit: ids.len(),
                    };
                    let node_ids_vec = node_ids.iter().cloned().collect::<Vec<_>>();
                    let edge_ids_vec = edge_ids.iter().cloned().collect::<Vec<_>>();
                    let mut nodes = if node_ids_vec.is_empty() {
                        Vec::new()
                    } else {
                        uow.graph_projection_records(read(&indexes.nodes, &node_ids_vec)).await
                            .map_err(|error| PostgresStoreError::Backend(error.to_string()))?
                    };
                    let edges = if edge_ids_vec.is_empty() {
                        Vec::new()
                    } else {
                        uow.graph_projection_records(read(&indexes.edges, &edge_ids_vec)).await
                            .map_err(|error| PostgresStoreError::Backend(error.to_string()))?
                    };
                    let found_node_ids = nodes.iter().map(|record| record.id.clone()).collect::<BTreeSet<_>>();
                    for (entity_id, row) in valid_stage1_nodes {
                        if node_ids.contains(&entity_id) && !found_node_ids.contains(&entity_id) {
                            nodes.push(GraphRecord { id: row.entity_id, document: Some(row.document), metadata: row.metadata, embedding: None });
                        }
                    }
                    let found_edge_ids = edges.iter().map(|record| record.id.clone()).collect::<BTreeSet<_>>();
                    let mut all_edges = edges;
                    for (entity_id, row) in valid_stage1_edges {
                        if edge_ids.contains(&entity_id) && !found_edge_ids.contains(&entity_id) {
                            all_edges.push(GraphRecord { id: row.entity_id, document: Some(row.document), metadata: row.metadata, embedding: None });
                        }
                    }
                    let endpoint_ids = all_edges.iter().flat_map(|edge| {
                        graph_record_string_ids(edge, "source_ids").into_iter()
                            .chain(graph_record_string_ids(edge, "target_ids"))
                    }).collect::<BTreeSet<_>>();
                    let selected_node_ids = node_ids.iter().cloned().collect::<BTreeSet<_>>();
                    let extra_endpoint_ids = endpoint_ids.difference(&selected_node_ids).cloned().collect::<Vec<_>>();
                    if !extra_endpoint_ids.is_empty() {
                        nodes.extend(uow.graph_projection_records(read(&indexes.nodes, &extra_endpoint_ids)).await
                            .map_err(|error| PostgresStoreError::Backend(error.to_string()))?);
                    }
                    let acl_ids = nodes.iter().chain(all_edges.iter()).map(|record| record.id.clone()).collect::<BTreeSet<_>>();
                    let mut acl_index = GraphAclIndex::default();
                    let acl_entity_ids = acl_ids.into_iter().collect::<Vec<_>>();
                    for batch in acl_entity_ids.chunks(10_000) {
                        let acl_records = uow.graph_acl_records_for_entities(
                            &scope, "knowledge", batch,
                        ).await.map_err(|error| PostgresStoreError::Backend(error.to_string()))?;
                        acl_index.extend_from_records(&acl_records, "knowledge");
                    }
                    let (visible_nodes, visible_edges) = graph_records_visible_by_acl_with_index(
                        &nodes, &all_edges, &acl_index, &acl_principal,
                    );
                    let node_ids = visible_nodes.iter().filter(|record| selected_node_ids.contains(&record.id))
                        .filter(|record| !matches!(record.metadata.get("lifecycle_status").and_then(Value::as_str), Some("deleted" | "tombstoned")))
                        .map(|record| record.id.clone()).collect::<BTreeSet<_>>();
                    let selected_edge_ids = edge_ids;
                    let edge_ids = visible_edges.iter().filter(|record| selected_edge_ids.contains(&record.id))
                        .filter(|record| !matches!(record.metadata.get("lifecycle_status").and_then(Value::as_str), Some("deleted" | "tombstoned")))
                        .map(|record| record.id.clone()).collect::<BTreeSet<_>>();
                    Ok((node_ids.into_iter().collect::<Vec<_>>(), edge_ids.into_iter().collect::<Vec<_>>()))
                })
            })
            .await;
        match result {
            Ok((node_ids, edge_ids)) => json_effect(StatusCode::OK, rust_native_attested(json!({
                "node_ids": node_ids,
                "edge_ids": edge_ids,
            }))),
            Err(error) => effect_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "KOGWISTAR_STORE_ERROR",
                error.to_string(),
            ),
        }
    }

    // PYTHON_PARITY: kogwistar/server/bootstrap.py::build_graph_engine(graph_type=...)
    fn graph_store_for_type(&self, graph_type: &str) -> &PostgresStore {
        match graph_type.to_ascii_lowercase().as_str() {
            "conversation" => &self.conversation_store,
            "workflow" => &self.workflow_store,
            "wisdom" => &self.wisdom_store,
            _ => &self.store,
        }
    }

    async fn authorized_evidence_records(
        &self,
        principal: &Value,
        node_ids: &[String],
        edge_ids: &[String],
        truth_graph: &str,
    ) -> Result<Value, NativeEvidenceRecordError> {
        if node_ids.len().saturating_add(edge_ids.len()) > 100 {
            return Err(NativeEvidenceRecordError::Rejected);
        }
        let scope_value = security_scope_value(principal);
        let scope = GraphScope {
            namespace: scope_value["storage_namespace"]
                .as_str()
                .unwrap_or("default")
                .to_owned(),
            workspace_id: principal
                .get("workspace_id")
                .or_else(|| principal.get("workspace"))
                .and_then(Value::as_str)
                .map(str::to_owned),
            graph_space: principal
                .get("graph_space")
                .and_then(Value::as_str)
                .map(str::to_owned),
        };
        let scope_value = scope_value.clone();
        let principal = principal.clone();
        let truth_graph = truth_graph.to_owned();
        let node_ids = node_ids.to_vec();
        let edge_ids = edge_ids.to_vec();
        self.graph_store_for_type(&truth_graph)
            .transaction(move |uow| {
                Box::pin(async move {
                    let read = |table: &str, ids: &[String]| GraphProjectionRead {
                        scope: scope.clone(),
                        table: table.to_owned(),
                        ids: (!ids.is_empty()).then(|| ids.to_vec()),
                        metadata: MetadataFilter::default(),
                        limit: ids.len().max(1),
                    };
                    let mut nodes = if node_ids.is_empty() {
                        Vec::new()
                    } else {
                        uow.graph_projection_records(read("gke_nodes", &node_ids))
                            .await?
                    };
                    let edges = if edge_ids.is_empty() {
                        Vec::new()
                    } else {
                        uow.graph_projection_records(read("gke_edges", &edge_ids))
                            .await?
                    };
                    let endpoint_ids = edges
                        .iter()
                        .flat_map(|edge| {
                            graph_record_string_ids(edge, "source_ids")
                                .into_iter()
                                .chain(graph_record_string_ids(edge, "target_ids"))
                        })
                        .collect::<BTreeSet<_>>();
                    if endpoint_ids.len() > 1000 {
                        return Ok(Err(NativeEvidenceRecordError::Rejected));
                    }
                    let selected_node_ids = node_ids.iter().cloned().collect::<BTreeSet<_>>();
                    let additional_endpoint_ids = endpoint_ids
                        .difference(&selected_node_ids)
                        .cloned()
                        .collect::<Vec<_>>();
                    if !additional_endpoint_ids.is_empty() {
                        nodes.extend(
                            uow.graph_projection_records(read(
                                "gke_nodes",
                                &additional_endpoint_ids,
                            ))
                            .await?,
                        );
                    }
                    let mut unique_nodes = BTreeMap::new();
                    for node in nodes {
                        unique_nodes.insert(node.id.clone(), node);
                    }
                    let nodes = unique_nodes.into_values().collect::<Vec<_>>();
                    let entity_ids = nodes
                        .iter()
                        .chain(edges.iter())
                        .map(|record| record.id.clone())
                        .collect::<BTreeSet<_>>();
                    let acl_records = uow
                        .graph_acl_records_for_entities(
                            &scope,
                            &truth_graph,
                            &entity_ids.into_iter().collect::<Vec<_>>(),
                        )
                        .await?;
                    let mut acl_index = GraphAclIndex::default();
                    acl_index.extend_from_records(&acl_records, &truth_graph);
                    let acl_principal = graph_acl_principal_from_claims(
                        &principal,
                        scope_value["security_scope"].as_str(),
                    );
                    let (visible_nodes, visible_edges) = graph_records_visible_by_acl_with_index(
                        &nodes,
                        &edges,
                        &acl_index,
                        &acl_principal,
                    );
                    let selected_node_ids = node_ids.iter().collect::<BTreeSet<_>>();
                    let selected_edge_ids = edge_ids.iter().collect::<BTreeSet<_>>();
                    let node_records = visible_nodes
                        .iter()
                        .filter(|record| {
                            selected_node_ids.contains(&record.id)
                                && !matches!(
                                    record
                                        .metadata
                                        .get("lifecycle_status")
                                        .and_then(Value::as_str),
                                    Some("tombstoned" | "deleted")
                                )
                        })
                        .map(|record| {
                            json!({
                                "id": record.id,
                                "document": record.document,
                                "metadata": record.metadata,
                            })
                        })
                        .collect::<Vec<_>>();
                    let edge_records = visible_edges
                        .iter()
                        .filter(|record| {
                            selected_edge_ids.contains(&record.id)
                                && !matches!(
                                    record
                                        .metadata
                                        .get("lifecycle_status")
                                        .and_then(Value::as_str),
                                    Some("tombstoned" | "deleted")
                                )
                        })
                        .map(|record| {
                            json!({
                                "id": record.id,
                                "document": record.document,
                                "metadata": record.metadata,
                            })
                        })
                        .collect::<Vec<_>>();
                    if node_records.len() != selected_node_ids.len()
                        || edge_records.len() != selected_edge_ids.len()
                    {
                        return Ok(Err(NativeEvidenceRecordError::Rejected));
                    }
                    Ok(Ok(json!({"nodes": node_records, "edges": edge_records})))
                })
            })
            .await
            .map_err(|_| NativeEvidenceRecordError::Unavailable)?
    }

    async fn graph_records_snapshot(
        &self,
        graph_type: &str,
        scope: GraphScope,
    ) -> Result<(Vec<GraphRecord>, Vec<GraphRecord>), String> {
        self.graph_records_snapshot_from(self.graph_store_for_type(graph_type), scope)
            .await
    }

    async fn graph_records_snapshot_from(
        &self,
        store: &PostgresStore,
        scope: GraphScope,
    ) -> Result<(Vec<GraphRecord>, Vec<GraphRecord>), String> {
        let reads = ["gke_nodes", "gke_edges"]
            .into_iter()
            .map(|table| GraphProjectionRead {
                scope: scope.clone(),
                table: table.to_owned(),
                ids: None,
                metadata: MetadataFilter::default(),
                limit: 10_000,
            })
            .collect();
        let mut snapshots = store
            .graph_projection_snapshot(reads)
            .await
            .map_err(|error| error.to_string())?
            .into_iter();
        let nodes = snapshots
            .next()
            .ok_or_else(|| "graph snapshot omitted node projection".to_owned())?;
        let edges = snapshots
            .next()
            .ok_or_else(|| "graph snapshot omitted edge projection".to_owned())?;
        if snapshots.next().is_some() {
            return Err("graph snapshot returned unexpected projection data".to_owned());
        }
        Ok((nodes, edges))
    }

    fn authorize_context_snapshot(
        &self,
        request: &ApiEffectRequest,
    ) -> Result<ContextSnapshotAuthorization, ApiEffectResponse> {
        if !capability_allowed(
            request,
            &self.capabilities,
            "project_view",
            &["project_view"],
        ) {
            return Err(effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Context snapshot reads require project_view capability",
            ));
        }
        Ok(ContextSnapshotAuthorization {
            principal: request.principal.clone(),
        })
    }

    fn authorize_conversation_transcript(
        &self,
        request: &ApiEffectRequest,
    ) -> Result<ConversationTranscriptAuthorization, ApiEffectResponse> {
        if !capability_allowed(request, &self.capabilities, "read_graph", &["read_graph"]) {
            return Err(effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Conversation transcript reads require read_graph capability",
            ));
        }
        Ok(ConversationTranscriptAuthorization {
            principal: request.principal.clone(),
        })
    }

    async fn authorized_conversation_nodes(
        &self,
        request_principal: &Value,
    ) -> Result<Vec<GraphRecord>, String> {
        // Match Python's empty-graph read behavior without implicitly creating
        // the optional graph projection schema during an operator read.
        if !self
            .conversation_store
            .has_graph_projection_rows()
            .await
            .map_err(|error| error.to_string())?
        {
            return Ok(Vec::new());
        }
        let (scope, acl_principal) = conversation_graph_scope_and_acl_principal(request_principal);
        let records = graph_projection_records_all(
            self.graph_store_for_type("conversation"),
            GraphProjectionRead {
                scope,
                table: "gke_nodes".to_owned(),
                ids: None,
                metadata: MetadataFilter::default(),
                limit: 10_000,
            },
        )
        .await
        .map_err(|error| error.to_string())?;
        let (visible_nodes, _) =
            graph_records_visible_by_acl(&records, &[], "conversation", &acl_principal);
        Ok(visible_nodes)
    }

    async fn authorized_tool_audit_nodes(
        &self,
        request_principal: &Value,
        conversation_id: Option<&str>,
        limit: usize,
    ) -> Result<Vec<GraphRecord>, String> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let (scope, acl_principal) = conversation_graph_scope_and_acl_principal(request_principal);
        let mut candidates = Vec::new();
        for entity_type in ["tool_call", "tool_result"] {
            let mut equals = BTreeMap::new();
            equals.insert("entity_type".to_owned(), json!(entity_type));
            if let Some(conversation_id) = conversation_id.filter(|value| !value.is_empty()) {
                equals.insert("conversation_id".to_owned(), json!(conversation_id));
            }
            candidates.extend(
                graph_projection_records_all(
                    self.graph_store_for_type("conversation"),
                    GraphProjectionRead {
                        scope: scope.clone(),
                        table: "gke_nodes".to_owned(),
                        ids: None,
                        metadata: MetadataFilter { equals },
                        limit,
                    },
                )
                .await
                .map_err(|error| error.to_string())?,
            );
        }
        let acl_records = graph_projection_records_all(
            self.graph_store_for_type("conversation"),
            GraphProjectionRead {
                scope,
                table: "gke_nodes".to_owned(),
                ids: None,
                metadata: MetadataFilter {
                    equals: BTreeMap::from([("entity_type".to_owned(), json!("acl_record"))]),
                },
                limit: 10_000,
            },
        )
        .await
        .map_err(|error| error.to_string())?;
        let mut acl_index = GraphAclIndex::default();
        acl_index.extend_from_records(&acl_records, "conversation");
        let (visible_nodes, _) =
            graph_records_visible_by_acl_with_index(&candidates, &[], &acl_index, &acl_principal);
        Ok(visible_nodes)
    }

    async fn tool_audit_effect(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
        let namespaces = principal_strings(&request.principal, "ns");
        let namespaces = if namespaces.is_empty() {
            vec!["docs".to_owned()]
        } else {
            namespaces
        };
        if !namespaces
            .iter()
            .any(|namespace| matches!(namespace.as_str(), "*" | "conversation" | "workflow"))
        {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_NAMESPACE_FORBIDDEN",
                "Tool audit reads require conversation or workflow namespace",
            );
        }
        if !capability_allowed(
            request,
            &self.capabilities,
            "project_view",
            &["project_view", "read_graph"],
        ) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Reading tool audit requires project_view capability",
            );
        }
        let (conversation_id, limit) = match tool_audit_query(&request.path_and_query) {
            Ok(query) => query,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error,
                );
            }
        };
        let visible_nodes = match self
            .authorized_tool_audit_nodes(&request.principal, conversation_id.as_deref(), limit)
            .await
        {
            Ok(nodes) => nodes,
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error,
                );
            }
        };
        let items = match tool_audit_items_from_visible_records(
            &visible_nodes,
            conversation_id.as_deref(),
            limit,
        ) {
            Ok(items) => items,
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_TOOL_AUDIT_INVALID",
                    error.to_string(),
                );
            }
        };
        let response = ToolAuditResponseV1 {
            schema_version: kogwistar_contracts::TOOL_AUDIT_WIRE_VERSION,
            conversation_id,
            items,
        };
        if let Err(error) = response.validate() {
            return effect_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "KOGWISTAR_TOOL_AUDIT_INVALID",
                error.to_string(),
            );
        }
        let mut payload =
            serde_json::to_value(response).expect("validated tool audit response serializes");
        if let Some(object) = payload.as_object_mut() {
            object.remove("schema_version");
            if let Some(items) = object.get_mut("items").and_then(Value::as_array_mut) {
                for item in items {
                    if let Some(item) = item.as_object_mut() {
                        item.remove("schema_version");
                    }
                }
            }
        }
        json_effect(StatusCode::OK, payload)
    }

    async fn conversation_transcript_projection(
        &self,
        request: &ApiEffectRequest,
        input: InternalConversationTranscriptRequest,
        authorization: ConversationTranscriptAuthorization,
    ) -> Result<ConversationTranscriptV1, ApiEffectResponse> {
        if authorization.principal != request.principal {
            return Err(effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Conversation transcript authorization does not match the request principal",
            ));
        }
        if let Err(error) = validate_internal_conversation_transcript_request(&input) {
            return Err(effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                error,
            ));
        }
        let visible_nodes = self
            .authorized_conversation_nodes(&request.principal)
            .await
            .map_err(|error| {
                effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error,
                )
            })?;
        conversation_transcript_from_visible_records(
            &visible_nodes,
            &input.conversation_id,
            input.limit,
        )
        .map_err(|error| {
            effect_error(
                StatusCode::NOT_FOUND,
                "KOGWISTAR_CONVERSATION_NOT_FOUND",
                error.to_string(),
            )
        })
    }

    async fn internal_conversation_transcript_effect(
        &self,
        request: &ApiEffectRequest,
    ) -> ApiEffectResponse {
        let authorization = match self.authorize_conversation_transcript(request) {
            Ok(value) => value,
            Err(response) => return response,
        };
        let input = match parse_internal_conversation_transcript_request(&request.body) {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error,
                );
            }
        };
        match self
            .conversation_transcript_projection(request, input, authorization)
            .await
        {
            Ok(transcript) => json_effect(
                StatusCode::OK,
                rust_native_attested(
                    serde_json::to_value(transcript).expect("transcript serializes"),
                ),
            ),
            Err(response) => response,
        }
    }

    async fn context_snapshot_projection(
        &self,
        request: &ApiEffectRequest,
        input: InternalContextSnapshotRequest,
        authorization: ContextSnapshotAuthorization,
    ) -> Result<ContextSnapshotV1, ApiEffectResponse> {
        if authorization.principal != request.principal {
            return Err(effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Context snapshot authorization does not match the request principal",
            ));
        }
        if let Err(error) = validate_internal_context_snapshot_request(&input) {
            return Err(effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                error,
            ));
        }
        let visible_nodes = self
            .authorized_conversation_nodes(&request.principal)
            .await
            .map_err(|error| {
                effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error,
                )
            })?;
        kogwistar_store::context_snapshot_from_visible_records(
            &visible_nodes,
            &input.conversation_id,
            input.run_id.as_deref(),
            input.stage.as_deref(),
        )
        .map_err(|error| {
            effect_error(
                StatusCode::NOT_FOUND,
                "KOGWISTAR_CONTEXT_SNAPSHOT_NOT_FOUND",
                error.to_string(),
            )
        })
    }

    async fn internal_context_snapshot_effect(
        &self,
        request: &ApiEffectRequest,
    ) -> ApiEffectResponse {
        let authorization = match self.authorize_context_snapshot(request) {
            Ok(value) => value,
            Err(response) => return response,
        };
        let input = match parse_internal_context_snapshot_request(&request.body) {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error,
                );
            }
        };
        match self
            .context_snapshot_projection(request, input, authorization)
            .await
        {
            Ok(snapshot) => json_effect(
                StatusCode::OK,
                rust_native_attested(serde_json::to_value(snapshot).expect("snapshot serializes")),
            ),
            Err(response) => response,
        }
    }

    async fn latest_snapshot_effect(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
        let namespaces = principal_strings(&request.principal, "ns");
        let namespaces = if namespaces.is_empty() {
            vec!["docs".to_owned()]
        } else {
            namespaces
        };
        if !namespaces
            .iter()
            .any(|namespace| matches!(namespace.as_str(), "*" | "conversation" | "workflow"))
        {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_NAMESPACE_FORBIDDEN",
                "Context snapshot reads require conversation or workflow namespace",
            );
        }
        let Some(conversation_id) = latest_snapshot_conversation_id(&request.path_and_query) else {
            return effect_error(
                StatusCode::NOT_FOUND,
                "KOGWISTAR_ROUTE_NOT_FOUND",
                "Conversation snapshot route was not found",
            );
        };
        let (run_id, stage) = context_snapshot_query(&request.path_and_query);
        let authorization = match self.authorize_context_snapshot(request) {
            Ok(value) => value,
            Err(response) => return response,
        };
        match self
            .context_snapshot_projection(
                request,
                InternalContextSnapshotRequest {
                    conversation_id,
                    run_id,
                    stage,
                },
                authorization,
            )
            .await
        {
            Ok(snapshot) => json_effect(StatusCode::OK, public_context_snapshot_value(snapshot)),
            Err(response) => response,
        }
    }

    async fn conversation_list_effect(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
        let namespaces = principal_strings(&request.principal, "ns");
        let namespaces = if namespaces.is_empty() {
            vec!["docs".to_owned()]
        } else {
            namespaces
        };
        if !namespaces
            .iter()
            .any(|namespace| matches!(namespace.as_str(), "*" | "conversation"))
        {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_NAMESPACE_FORBIDDEN",
                "Conversation reads require conversation namespace",
            );
        }
        if !capability_allowed(request, &self.capabilities, "read_graph", &["read_graph"]) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Conversation reads require read_graph capability",
            );
        }
        let user_id = request
            .principal
            .get("sub")
            .or_else(|| request.principal.get("user_id"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty());
        let Some(user_id) = user_id else {
            return effect_error(
                StatusCode::UNAUTHORIZED,
                "KOGWISTAR_USER_IDENTITY_REQUIRED",
                "User Identity required",
            );
        };
        let (scope, acl_principal) = conversation_graph_scope_and_acl_principal(&request.principal);
        let mut acl_index = GraphAclIndex::default();
        let mut candidates = Vec::new();
        let read_result = for_each_graph_projection_page(
            self.graph_store_for_type("conversation"),
            GraphProjectionRead {
                scope,
                table: "gke_nodes".to_owned(),
                ids: None,
                metadata: MetadataFilter::default(),
                limit: 1024,
            },
            |page| {
                acl_index.extend_from_records(page, "conversation");
                candidates.extend(compact_conversation_list_candidates(page));
                Ok(())
            },
        )
        .await;
        if let Err(error) = read_result {
            return effect_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "KOGWISTAR_STORE_ERROR",
                error.to_string(),
            );
        }
        let (visible_nodes, _) =
            graph_records_visible_by_acl_with_index(&candidates, &[], &acl_index, &acl_principal);
        let mut conversations = ConversationListAccumulator::new(user_id);
        conversations.extend_visible_records(&visible_nodes);
        let conversations = conversations
            .finish()
            .into_iter()
            .map(|entry| {
                json!({
                    "id": entry.id,
                    "start_node_id": entry.start_node_id,
                    "status": entry.status,
                    "turn_count": entry.turn_count,
                })
            })
            .collect::<Vec<_>>();
        json_effect(StatusCode::OK, json!({"conversations": conversations}))
    }

    async fn conversation_detail_effect(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
        let namespaces = principal_strings(&request.principal, "ns");
        let namespaces = if namespaces.is_empty() {
            vec!["docs".to_owned()]
        } else {
            namespaces
        };
        if !namespaces
            .iter()
            .any(|namespace| matches!(namespace.as_str(), "*" | "conversation"))
        {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_NAMESPACE_FORBIDDEN",
                "Conversation reads require conversation namespace",
            );
        }
        if !capability_allowed(request, &self.capabilities, "read_graph", &["read_graph"]) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Conversation reads require read_graph capability",
            );
        }
        let Some(conversation_id) = conversation_detail_id(&request.path_and_query) else {
            return effect_error(
                StatusCode::NOT_FOUND,
                "KOGWISTAR_ROUTE_NOT_FOUND",
                "Conversation route was not found",
            );
        };
        let visible_nodes = match self.authorized_conversation_nodes(&request.principal).await {
            Ok(nodes) => nodes,
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error,
                );
            }
        };
        let summary =
            match conversation_summary_from_visible_records(&visible_nodes, &conversation_id) {
                Ok(summary) => summary,
                Err(error) => {
                    return effect_error(
                        StatusCode::NOT_FOUND,
                        "KOGWISTAR_CONVERSATION_NOT_FOUND",
                        error.to_string(),
                    );
                }
            };
        let mut payload = rust_native_attested(
            serde_json::to_value(summary).expect("conversation summary serializes"),
        );
        if let Some(object) = payload.as_object_mut() {
            object.remove("schema_version");
            object.remove("implementation_authority");
            object.remove("seam_attestation_version");
            object.remove("seam_crossings");
        }
        json_effect(StatusCode::OK, payload)
    }

    async fn conversation_transcript_effect(
        &self,
        request: &ApiEffectRequest,
    ) -> ApiEffectResponse {
        let namespaces = principal_strings(&request.principal, "ns");
        let namespaces = if namespaces.is_empty() {
            vec!["docs".to_owned()]
        } else {
            namespaces
        };
        if !namespaces
            .iter()
            .any(|namespace| matches!(namespace.as_str(), "*" | "conversation"))
        {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_NAMESPACE_FORBIDDEN",
                "Conversation reads require conversation namespace",
            );
        }
        let Some(conversation_id) = conversation_transcript_id(&request.path_and_query) else {
            return effect_error(
                StatusCode::NOT_FOUND,
                "KOGWISTAR_ROUTE_NOT_FOUND",
                "Conversation transcript route was not found",
            );
        };
        let authorization = match self.authorize_conversation_transcript(request) {
            Ok(value) => value,
            Err(response) => return response,
        };
        match self
            .conversation_transcript_projection(
                request,
                InternalConversationTranscriptRequest {
                    conversation_id,
                    limit: 10_000,
                },
                authorization,
            )
            .await
        {
            Ok(transcript) => {
                let mut payload = serde_json::to_value(transcript).expect("transcript serializes");
                if let Some(object) = payload.as_object_mut() {
                    object.remove("schema_version");
                    if let Some(turns) = object.get_mut("turns").and_then(Value::as_array_mut) {
                        for turn in turns {
                            if let Some(turn) = turn.as_object_mut() {
                                turn.remove("schema_version");
                            }
                        }
                    }
                }
                json_effect(StatusCode::OK, payload)
            }
            Err(response) => response,
        }
    }

    async fn graph_neighbors_effect(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
        if !capability_allowed(request, &self.capabilities, "read_graph", &["read_graph"]) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Graph read requires read_graph capability",
            );
        }
        let input = match parse_graph_neighbors_request(&request.body) {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error.to_string(),
                );
            }
        };
        let rid = input.rid.trim();
        let doc_id = input.doc_id.as_deref();
        let scope_value = security_scope_value(&request.principal);
        let scope = GraphScope {
            namespace: scope_value["storage_namespace"]
                .as_str()
                .unwrap_or("default")
                .to_owned(),
            workspace_id: request
                .principal
                .get("workspace_id")
                .or_else(|| request.principal.get("workspace"))
                .and_then(Value::as_str)
                .map(str::to_owned),
            graph_space: request
                .principal
                .get("graph_space")
                .and_then(Value::as_str)
                .map(str::to_owned),
        };
        let (nodes, edges) = match self.graph_records_snapshot(&input.truth_graph, scope).await {
            Ok(records) => records,
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error,
                );
            }
        };
        let resolved_rid = match input.resolve_mode {
            GraphNeighborsResolveModeV1::ActiveOnly => Some(rid.to_owned()),
            GraphNeighborsResolveModeV1::Redirect => {
                if nodes.iter().any(|node| node.id == rid) {
                    resolve_graph_node_redirect_id(rid, &nodes)
                } else {
                    resolve_graph_edge_redirect_id(rid, &edges)
                }
            }
        };
        let principal = graph_acl_principal_from_claims(
            &request.principal,
            scope_value["security_scope"].as_str(),
        );
        let (nodes, edges) =
            graph_records_visible_by_acl(&nodes, &edges, input.truth_graph.trim(), &principal);
        let neighbors = graph_neighbors_from_records(
            resolved_rid.as_deref().unwrap_or_default(),
            &nodes,
            &edges,
            doc_id,
        );
        json_effect(
            StatusCode::OK,
            rust_native_attested(json!({
                "nodes": neighbors.nodes.into_iter().collect::<Vec<_>>(),
                "edges": neighbors.edges.into_iter().collect::<Vec<_>>(),
            })),
        )
    }

    async fn authorized_vector_candidate_ids(
        &self,
        store: &PostgresStore,
        scope: &GraphScope,
        truth_graph: &str,
        entity_kind: &str,
        candidates_by_query: &[Vec<GraphRecord>],
        principal: &GraphAclPrincipal,
    ) -> Result<BTreeSet<String>, String> {
        let mut seen_candidate_ids = BTreeSet::new();
        let candidates = candidates_by_query
            .iter()
            .flatten()
            .filter(|record| seen_candidate_ids.insert(record.id.clone()))
            .cloned()
            .collect::<Vec<_>>();
        let (nodes, edges) = if entity_kind == "node" {
            (candidates, Vec::new())
        } else {
            let mut edges = candidates;
            let mut known_edge_ids = edges
                .iter()
                .map(|record| record.id.clone())
                .collect::<BTreeSet<_>>();
            let mut frontier = edges
                .iter()
                .flat_map(|record| {
                    graph_record_string_ids(record, "source_edge_ids")
                        .into_iter()
                        .chain(graph_record_string_ids(record, "target_edge_ids"))
                })
                .collect::<BTreeSet<_>>();
            while !frontier.is_empty() {
                let pending = frontier
                    .iter()
                    .filter(|edge_id| !known_edge_ids.contains(*edge_id))
                    .cloned()
                    .collect::<Vec<_>>();
                frontier.clear();
                if pending.is_empty() {
                    break;
                }
                if known_edge_ids.len().saturating_add(pending.len()) > 10_000 {
                    return Err("edge ACL dependency closure exceeds 10000 records".to_owned());
                }
                let referenced = graph_projection_records_all(
                    store,
                    GraphProjectionRead {
                        scope: scope.clone(),
                        table: "gke_edges".to_owned(),
                        ids: Some(pending),
                        metadata: MetadataFilter::default(),
                        limit: 10_000,
                    },
                )
                .await
                .map_err(|error| error.to_string())?;
                for record in referenced {
                    if !known_edge_ids.insert(record.id.clone()) {
                        continue;
                    }
                    frontier.extend(graph_record_string_ids(&record, "source_edge_ids"));
                    frontier.extend(graph_record_string_ids(&record, "target_edge_ids"));
                    edges.push(record);
                }
            }
            let node_ids = edges
                .iter()
                .flat_map(|record| {
                    graph_record_string_ids(record, "source_ids")
                        .into_iter()
                        .chain(graph_record_string_ids(record, "target_ids"))
                })
                .collect::<BTreeSet<_>>();
            let nodes = if node_ids.is_empty() {
                Vec::new()
            } else {
                graph_projection_records_all(
                    store,
                    GraphProjectionRead {
                        scope: scope.clone(),
                        table: "gke_nodes".to_owned(),
                        ids: Some(node_ids.into_iter().collect()),
                        metadata: MetadataFilter::default(),
                        limit: 10_000,
                    },
                )
                .await
                .map_err(|error| error.to_string())?
            };
            (nodes, edges)
        };
        let entity_ids = nodes
            .iter()
            .chain(edges.iter())
            .map(|record| record.id.clone())
            .collect::<BTreeSet<_>>();
        let acl_records = store
            .graph_acl_records_for_entities(
                scope,
                truth_graph,
                &entity_ids.into_iter().collect::<Vec<_>>(),
            )
            .await
            .map_err(|error| error.to_string())?;
        let mut acl_index = GraphAclIndex::default();
        acl_index.extend_from_records(&acl_records, truth_graph);
        let (visible_nodes, visible_edges) =
            graph_records_visible_by_acl_with_index(&nodes, &edges, &acl_index, principal);
        Ok(visible_nodes
            .iter()
            .chain(visible_edges.iter())
            .map(|record| record.id.clone())
            .collect())
    }

    async fn search_index_hybrid_effect(
        &self,
        request: &ApiEffectRequest,
    ) -> ApiEffectResponse {
        if !capability_allowed(request, &self.capabilities, "read_graph", &["read_graph"]) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Search-index query requires read_graph capability",
            );
        }
        let (query, limit, resolve_node) = match crate::search_index_fts::parse_hybrid_search_query(
            &request.path_and_query,
        ) {
            Ok(parsed) => parsed,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error,
                );
            }
        };
        let (Some(provider), Some(fts)) =
            (self.embedding_provider.as_ref(), self.search_index_fts.as_ref())
        else {
            return effect_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "KOGWISTAR_SEARCH_INDEX_UNAVAILABLE",
                "Search-index embedding or FTS storage is not configured",
            );
        };
        let graph_store = self.graph_store_for_type(&self.graph_role);
        let hits = match crate::search_index_fts::hybrid_search(
            fts,
            graph_store,
            "gke_node_index",
            provider.as_ref(),
            &query,
            limit,
        )
        .await
        {
            Ok(hits) => hits,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_SEARCH_INDEX_ERROR",
                    error,
                );
            }
        };
        let scope_value = security_scope_value(&request.principal);
        let scope = GraphScope {
            namespace: scope_value["storage_namespace"]
                .as_str()
                .unwrap_or("default")
                .to_owned(),
            workspace_id: request
                .principal
                .get("workspace_id")
                .or_else(|| request.principal.get("workspace"))
                .and_then(Value::as_str)
                .map(str::to_owned),
            graph_space: request
                .principal
                .get("graph_space")
                .and_then(Value::as_str)
                .map(str::to_owned),
        };
        let node_ids = hits
            .iter()
            .map(|hit| hit.node_id.clone())
            .filter(|id| !id.is_empty())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let mut canonical_nodes = if node_ids.is_empty() {
            Vec::new()
        } else {
            match graph_projection_records_all(
                graph_store,
                GraphProjectionRead {
                    scope: scope.clone(),
                    table: "gke_nodes".to_owned(),
                    ids: Some(node_ids.clone()),
                    metadata: MetadataFilter::default(),
                    limit: node_ids.len(),
                },
            )
            .await
            {
                Ok(nodes) => nodes,
                Err(error) => {
                    return effect_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "KOGWISTAR_GRAPH_READ_ERROR",
                        error.to_string(),
                    );
                }
            }
        };
        canonical_nodes.retain(|record| {
            !matches!(
                record
                    .metadata
                    .get("lifecycle_status")
                    .and_then(Value::as_str),
                Some("tombstoned" | "deleted")
            )
        });
        let principal = graph_acl_principal_from_claims(
            &request.principal,
            scope_value["security_scope"].as_str(),
        );
        let visible_ids = match self
            .authorized_vector_candidate_ids(
                graph_store,
                &scope,
                &self.graph_role,
                "node",
                &[canonical_nodes.clone()],
                &principal,
            )
            .await
        {
            Ok(ids) => ids,
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_ACL_QUERY_ERROR",
                    error,
                );
            }
        };
        canonical_nodes.retain(|record| visible_ids.contains(&record.id));
        json_effect(
            StatusCode::OK,
            crate::search_index_fts::project_hybrid_search_result(
                &query,
                &hits,
                &canonical_nodes,
                resolve_node,
                limit,
            ),
        )
    }

    // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/postgres_graph_service.rs#graph_vector_query_effect; python=kogwistar/engine_core/subsystems/acl.py#ACLAwareReadSubsystem.query_nodes; fixture=contracts/golden/graph-acl-vector-query-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/graph_query_contracts.rs#postgres_graph_vector_query_is_scoped_acl_filtered_and_rank_stable_when_dsn_available; python_test=tests/mcp/test_chat_mcp_tools.py#test_acl_vector_query_matches_shared_python_rust_ranking_fixture
    async fn graph_vector_query_effect(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
        if !capability_allowed(request, &self.capabilities, "read_graph", &["read_graph"]) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Graph vector query requires read_graph capability",
            );
        }
        let input = match parse_graph_vector_query_request(&request.body) {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error,
                );
            }
        };
        let scope_value = security_scope_value(&request.principal);
        let scope = GraphScope {
            namespace: scope_value["storage_namespace"]
                .as_str()
                .unwrap_or("default")
                .to_owned(),
            workspace_id: request
                .principal
                .get("workspace_id")
                .or_else(|| request.principal.get("workspace"))
                .and_then(Value::as_str)
                .map(str::to_owned),
            graph_space: request
                .principal
                .get("graph_space")
                .and_then(Value::as_str)
                .map(str::to_owned),
        };
        let table = if input.entity_kind == "node" {
            "gke_nodes"
        } else {
            "gke_edges"
        };
        let principal = graph_acl_principal_from_claims(
            &request.principal,
            scope_value["security_scope"].as_str(),
        );

        let mut all_ids = Vec::with_capacity(input.query_embeddings.len());
        let mut all_documents = Vec::with_capacity(input.query_embeddings.len());
        let mut all_metadatas = Vec::with_capacity(input.query_embeddings.len());
        let mut all_embeddings = Vec::with_capacity(input.query_embeddings.len());
        let mut all_distances = Vec::with_capacity(input.query_embeddings.len());
        let queries = input
            .query_embeddings
            .iter()
            .map(|embedding| GraphProjectionVectorQuery {
                scope: scope.clone(),
                table: table.to_owned(),
                query: VectorQuery {
                    embedding: embedding.clone(),
                    limit: input.n_results,
                    metadata: input.where_filter.clone(),
                    metric: input.metric,
                },
                embedding_dim: embedding.len(),
            })
            .collect();
        let graph_store = self.graph_store_for_type(&input.truth_graph);
        let matches_by_query = match graph_store
            .graph_projection_vector_query_batch_excluding_metadata(
                queries,
                BTreeMap::from([("entity_type".to_owned(), json!("acl_record"))]),
            )
            .await
        {
            Ok(matches) => matches,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_VECTOR_QUERY_ERROR",
                    error.to_string(),
                );
            }
        };
        let candidate_rows = matches_by_query
            .iter()
            .map(|matches| {
                matches
                    .iter()
                    .map(|matched| matched.record.clone())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let visible_ids = match self
            .authorized_vector_candidate_ids(
                graph_store,
                &scope,
                &input.truth_graph,
                &input.entity_kind,
                &candidate_rows,
                &principal,
            )
            .await
        {
            Ok(ids) => ids,
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_ACL_QUERY_ERROR",
                    error,
                );
            }
        };
        for matches in matches_by_query {
            let visible_matches = matches
                .into_iter()
                .filter(|matched| {
                    !matched
                        .record
                        .metadata
                        .get("lifecycle_status")
                        .and_then(Value::as_str)
                        .is_some_and(|status| matches!(status, "tombstoned" | "deleted"))
                        && visible_ids.contains(&matched.record.id)
                })
                .collect::<Vec<_>>();
            all_ids.push(
                visible_matches
                    .iter()
                    .map(|matched| json!(matched.record.id))
                    .collect::<Vec<_>>(),
            );
            if input.include.iter().any(|field| field == "documents") {
                all_documents.push(
                    visible_matches
                        .iter()
                        .map(|matched| json!(matched.record.document))
                        .collect::<Vec<_>>(),
                );
            }
            if input.include.iter().any(|field| field == "metadatas") {
                all_metadatas.push(
                    visible_matches
                        .iter()
                        .map(|matched| json!(matched.record.metadata))
                        .collect::<Vec<_>>(),
                );
            }
            if input.include.iter().any(|field| field == "embeddings") {
                all_embeddings.push(
                    visible_matches
                        .iter()
                        .map(|matched| json!(matched.record.embedding))
                        .collect::<Vec<_>>(),
                );
            }
            if input.include.iter().any(|field| field == "distances") {
                all_distances.push(
                    visible_matches
                        .iter()
                        .map(|matched| json!(matched.distance))
                        .collect::<Vec<_>>(),
                );
            }
        }
        let mut response = serde_json::Map::new();
        response.insert("ids".to_owned(), json!(all_ids));
        if input.include.iter().any(|field| field == "documents") {
            response.insert("documents".to_owned(), json!(all_documents));
        }
        if input.include.iter().any(|field| field == "metadatas") {
            response.insert("metadatas".to_owned(), json!(all_metadatas));
        }
        if input.include.iter().any(|field| field == "embeddings") {
            response.insert("embeddings".to_owned(), json!(all_embeddings));
        }
        if input.include.iter().any(|field| field == "distances") {
            response.insert("distances".to_owned(), json!(all_distances));
        }
        json_effect(
            StatusCode::OK,
            rust_native_attested(Value::Object(response)),
        )
    }

    async fn graph_find_edges_effect(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
        if !capability_allowed(request, &self.capabilities, "read_graph", &["read_graph"]) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Graph read requires read_graph capability",
            );
        }
        let input = match parse_graph_find_edges_request(&request.body) {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error,
                );
            }
        };
        let scope_value = security_scope_value(&request.principal);
        let scope = GraphScope {
            namespace: scope_value["storage_namespace"]
                .as_str()
                .unwrap_or("default")
                .to_owned(),
            workspace_id: request
                .principal
                .get("workspace_id")
                .or_else(|| request.principal.get("workspace"))
                .and_then(Value::as_str)
                .map(str::to_owned),
            graph_space: request
                .principal
                .get("graph_space")
                .and_then(Value::as_str)
                .map(str::to_owned),
        };
        let (nodes, edges) = match self.graph_records_snapshot("knowledge", scope).await {
            Ok(records) => records,
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error,
                );
            }
        };
        let principal = graph_acl_principal_from_claims(
            &request.principal,
            scope_value["security_scope"].as_str(),
        );
        let (nodes, edges) = graph_records_visible_by_acl(&nodes, &edges, "knowledge", &principal);
        let query = GraphFindEdgesQuery {
            relation: input.relation,
            src_label_contains: input.src_label_contains,
            tgt_label_contains: input.tgt_label_contains,
            doc_id: input.doc_id,
        };
        json_effect(
            StatusCode::OK,
            rust_native_attested(json!({
                "edges": graph_find_edges_from_records(&query, &nodes, &edges),
            })),
        )
    }

    async fn graph_shortest_path_effect(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
        if !capability_allowed(request, &self.capabilities, "read_graph", &["read_graph"]) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Graph read requires read_graph capability",
            );
        }
        let input: GraphShortestPathRequest = match serde_json::from_slice(&request.body) {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error.to_string(),
                );
            }
        };
        if input.src_id.trim().is_empty() || input.dst_id.trim().is_empty() || input.max_depth > 128
        {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                "src_id and dst_id are required and max_depth must be <= 128",
            );
        }
        let scope_value = security_scope_value(&request.principal);
        let scope = GraphScope {
            namespace: scope_value["storage_namespace"]
                .as_str()
                .unwrap_or("default")
                .to_owned(),
            workspace_id: request
                .principal
                .get("workspace_id")
                .or_else(|| request.principal.get("workspace"))
                .and_then(Value::as_str)
                .map(str::to_owned),
            graph_space: request
                .principal
                .get("graph_space")
                .and_then(Value::as_str)
                .map(str::to_owned),
        };
        let (nodes, edges) = match self.graph_records_snapshot("knowledge", scope).await {
            Ok(records) => records,
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error,
                );
            }
        };
        let principal = graph_acl_principal_from_claims(
            &request.principal,
            scope_value["security_scope"].as_str(),
        );
        let (nodes, edges) = graph_records_visible_by_acl(&nodes, &edges, "knowledge", &principal);
        let query = GraphShortestPathQuery {
            src_id: input.src_id,
            dst_id: input.dst_id,
            doc_id: input.doc_id,
            max_depth: input.max_depth,
        };
        json_effect(
            StatusCode::OK,
            rust_native_attested(json!({
                "path": graph_shortest_path_from_records(&query, &nodes, &edges),
            })),
        )
    }

    async fn graph_k_hop_effect(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
        if !capability_allowed(request, &self.capabilities, "read_graph", &["read_graph"]) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Graph read requires read_graph capability",
            );
        }
        let input: GraphKHopRequest = match serde_json::from_slice(&request.body) {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error.to_string(),
                );
            }
        };
        if input.start_ids.is_empty() || input.start_ids.iter().any(|id| id.trim().is_empty()) {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                "start_ids must contain at least one non-empty ID",
            );
        }
        if input.k > 32 {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                "k must be <= 32",
            );
        }
        let scope_value = security_scope_value(&request.principal);
        let scope = GraphScope {
            namespace: scope_value["storage_namespace"]
                .as_str()
                .unwrap_or("default")
                .to_owned(),
            workspace_id: request
                .principal
                .get("workspace_id")
                .or_else(|| request.principal.get("workspace"))
                .and_then(Value::as_str)
                .map(str::to_owned),
            graph_space: request
                .principal
                .get("graph_space")
                .and_then(Value::as_str)
                .map(str::to_owned),
        };
        let (nodes, edges) = match self.graph_records_snapshot("knowledge", scope).await {
            Ok(records) => records,
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error,
                );
            }
        };
        let principal = graph_acl_principal_from_claims(
            &request.principal,
            scope_value["security_scope"].as_str(),
        );
        let (nodes, edges) = graph_records_visible_by_acl(&nodes, &edges, "knowledge", &principal);
        let query = GraphKHopQuery {
            start_ids: input.start_ids,
            k: input.k,
            doc_id: input.doc_id,
        };
        let layers = graph_k_hop_from_records(&query, &nodes, &edges)
            .into_iter()
            .map(|layer| {
                json!({
                    "nodes": layer.nodes.into_iter().collect::<Vec<_>>(),
                    "edges": layer.edges.into_iter().collect::<Vec<_>>(),
                })
            })
            .collect::<Vec<_>>();
        json_effect(
            StatusCode::OK,
            rust_native_attested(json!({
                "layers": layers,
            })),
        )
    }

    async fn graph_visualization_effect(
        &self,
        request: &ApiEffectRequest,
        cytoscape: bool,
    ) -> ApiEffectResponse {
        if !capability_allowed(request, &self.capabilities, "read_graph", &["read_graph"]) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Graph read requires read_graph capability",
            );
        }
        let input: GraphVisualizationRequest = match serde_json::from_slice(&request.body) {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error.to_string(),
                );
            }
        };
        let scope_value = security_scope_value(&request.principal);
        let scope = GraphScope {
            namespace: scope_value["storage_namespace"]
                .as_str()
                .unwrap_or("default")
                .to_owned(),
            workspace_id: request
                .principal
                .get("workspace_id")
                .or_else(|| request.principal.get("workspace"))
                .and_then(Value::as_str)
                .map(str::to_owned),
            graph_space: request
                .principal
                .get("graph_space")
                .and_then(Value::as_str)
                .map(str::to_owned),
        };
        let truth_graph = input.truth_graph();
        let store = self.graph_store_for_type(truth_graph);
        let (nodes, edges) = match self.graph_records_snapshot_from(store, scope).await {
            Ok(records) => records,
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error,
                );
            }
        };
        let principal = graph_acl_principal_from_claims(
            &request.principal,
            scope_value["security_scope"].as_str(),
        );
        let (nodes, edges) = graph_records_visible_by_acl(&nodes, &edges, truth_graph, &principal);
        let result = if cytoscape {
            graph_viz_cytoscape_from_records(
                &nodes,
                &edges,
                input.doc_id.as_deref(),
                &input.mode,
                input.insertion_method.as_deref(),
            )
        } else {
            graph_viz_d3_from_records(
                &nodes,
                &edges,
                input.doc_id.as_deref(),
                &input.mode,
                input.insertion_method.as_deref(),
            )
        };
        json_effect(StatusCode::OK, rust_native_attested(result))
    }

    async fn load_resource_snapshot_rows(
        &self,
    ) -> Result<
        (
            Vec<ServerRun>,
            Vec<ProjectedLaneMessage>,
            Vec<ServerRunEvent>,
        ),
        ApiEffectResponse,
    > {
        let runs = self
            .store
            .list_server_runs(None, None, None, 10_000)
            .await
            .map_err(|error| {
                effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error.to_string(),
                )
            })?;
        let lanes = self
            .store
            .list_projected_lane_messages(LaneMessageFilter {
                namespace: Some("workflow".to_owned()),
                limit: 10_000,
                ..LaneMessageFilter::default()
            })
            .await
            .map_err(|error| {
                effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error.to_string(),
                )
            })?;
        let mut events = Vec::new();
        for run in &runs {
            let mut run_events = self
                .store
                .list_server_run_events(&run.run_id, 0, 200)
                .await
                .map_err(|error| {
                    effect_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "KOGWISTAR_STORE_ERROR",
                        error.to_string(),
                    )
                })?;
            events.append(&mut run_events);
        }
        Ok((runs, lanes, events))
    }

    async fn invoke_builtin_tool_syscall(
        &self,
        request: &ApiEffectRequest,
        input: &SyscallInput,
    ) -> ApiEffectResponse {
        let Some(tool_name) = input
            .args
            .get("tool_name")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .map(str::to_ascii_lowercase)
        else {
            return effect_error(
                StatusCode::BAD_REQUEST,
                "KOGWISTAR_INVALID_REQUEST",
                "tool_name is required",
            );
        };

        let result = match tool_name.as_str() {
            "list_transcript" => {
                let Some(conversation_id) = input
                    .args
                    .get("conversation_id")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                else {
                    return effect_error(
                        StatusCode::BAD_REQUEST,
                        "KOGWISTAR_INVALID_REQUEST",
                        "conversation_id is required for list_transcript",
                    );
                };
                let authorization = match self.authorize_conversation_transcript(request) {
                    Ok(value) => value,
                    Err(response) => return response,
                };
                let transcript = match self
                    .conversation_transcript_projection(
                        request,
                        InternalConversationTranscriptRequest {
                            conversation_id: conversation_id.to_owned(),
                            limit: 10_000,
                        },
                        authorization,
                    )
                    .await
                {
                    Ok(value) => value,
                    Err(response) => return response,
                };
                let mut result = legacy_receive_message_value(transcript);
                result
                    .as_object_mut()
                    .expect("legacy transcript result is an object")
                    .insert("tool_name".to_owned(), Value::String(tool_name.clone()));
                result
            }
            "latest_snapshot" => {
                let Some(conversation_id) = input
                    .args
                    .get("conversation_id")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                else {
                    return effect_error(
                        StatusCode::BAD_REQUEST,
                        "KOGWISTAR_INVALID_REQUEST",
                        "conversation_id is required for latest_snapshot",
                    );
                };
                let authorization = match self.authorize_context_snapshot(request) {
                    Ok(value) => value,
                    Err(response) => return response,
                };
                let run_id = match context_snapshot_syscall_field(&input.args, "run_id") {
                    Ok(value) => value,
                    Err(error) => {
                        return effect_error(
                            StatusCode::UNPROCESSABLE_ENTITY,
                            "KOGWISTAR_INVALID_REQUEST",
                            error,
                        );
                    }
                };
                let stage = match context_snapshot_syscall_field(&input.args, "stage") {
                    Ok(value) => value,
                    Err(error) => {
                        return effect_error(
                            StatusCode::UNPROCESSABLE_ENTITY,
                            "KOGWISTAR_INVALID_REQUEST",
                            error,
                        );
                    }
                };
                let snapshot = match self
                    .context_snapshot_projection(
                        request,
                        InternalContextSnapshotRequest {
                            conversation_id: conversation_id.to_owned(),
                            run_id,
                            stage,
                        },
                        authorization,
                    )
                    .await
                {
                    Ok(value) => value,
                    Err(response) => return response,
                };
                json!({
                    "tool_name": tool_name,
                    "conversation_id": conversation_id,
                    "snapshot": public_context_snapshot_value(snapshot),
                })
            }
            "capability_snapshot" => {
                if !capability_allowed(
                    request,
                    &self.capabilities,
                    "project_view",
                    &["project_view", "read_security_scope"],
                ) {
                    return effect_error(
                        StatusCode::FORBIDDEN,
                        "KOGWISTAR_CAPABILITY_FORBIDDEN",
                        format!(
                            "built-in tool {tool_name:?} requires project_view and read_security_scope"
                        ),
                    );
                }
                let state = self
                    .capabilities
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                json!({
                    "tool_name": tool_name,
                    "snapshot": capability_snapshot(&request.principal, &state),
                })
            }
            "resource_snapshot" => {
                if !capability_allowed(
                    request,
                    &self.capabilities,
                    "project_view",
                    &["project_view", "read_security_scope"],
                ) {
                    return effect_error(
                        StatusCode::FORBIDDEN,
                        "KOGWISTAR_CAPABILITY_FORBIDDEN",
                        format!(
                            "built-in tool {tool_name:?} requires project_view and read_security_scope"
                        ),
                    );
                }
                let (runs, lanes, events) = match self.load_resource_snapshot_rows().await {
                    Ok(rows) => rows,
                    Err(response) => return response,
                };
                let service_rows = match self.store.list_named_projections("service_registry").await
                {
                    Ok(rows) => rows,
                    Err(error) => {
                        return effect_error(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            "KOGWISTAR_STORE_ERROR",
                            error.to_string(),
                        );
                    }
                };
                let services = service_rows
                    .into_iter()
                    .map(|row| Value::Object(row.payload))
                    .collect::<Vec<_>>();
                let usage = usage_history(&events, 10_000);
                let mut snapshot = resource_value(
                    &runs,
                    &lanes,
                    &usage,
                    &events,
                    configured_storage_usage_bytes(),
                );
                snapshot["services"] = resource_service_value(&services);
                snapshot
                    .as_object_mut()
                    .expect("resource snapshot is an object")
                    .remove("migration");
                json!({"tool_name": tool_name, "snapshot": snapshot})
            }
            _ => {
                return effect_error(
                    StatusCode::NOT_FOUND,
                    "KOGWISTAR_TOOL_NOT_FOUND",
                    format!("unknown built-in tool: {tool_name}"),
                );
            }
        };
        syscall_result(
            &input.version,
            "invoke_tool",
            json_effect(StatusCode::OK, result),
        )
    }

    async fn syscall_effect(&self, request: &ApiEffectRequest, op: &str) -> ApiEffectResponse {
        let input = match syscall_input(request, op) {
            Ok(input) => input,
            Err(response) => return response,
        };
        let response = match op {
            "create_conversation" => self.create_conversation_syscall(request, &input).await,
            "spawn_process" => {
                if !syscall_allowed(
                    request,
                    &self.capabilities,
                    op,
                    true,
                    &["spawn_process", "workflow.run.write"],
                ) {
                    syscall_forbidden(op)
                } else {
                    let delegated_args = match prepare_delegated_run_args(
                        &Value::Object(input.args.clone()),
                        &request.principal,
                        &self.capabilities,
                    ) {
                        Ok(value) => value,
                        Err(error) => {
                            return effect_error(
                                StatusCode::UNPROCESSABLE_ENTITY,
                                "KOGWISTAR_INVALID_DELEGATION",
                                error,
                            );
                        }
                    };
                    let nested = ApiEffectRequest {
                        contract_version: request.contract_version,
                        method: "POST".to_owned(),
                        path_and_query: "/api/workflow/runs".to_owned(),
                        body: serde_json::to_vec(&delegated_args)
                            .expect("delegated syscall args serialize"),
                        principal: delegated_child_principal(&request.principal, &delegated_args),
                    };
                    syscall_result(&input.version, op, self.submit_run(&nested).await)
                }
            }
            "terminate_process" => {
                if !syscall_allowed(
                    request,
                    &self.capabilities,
                    op,
                    true,
                    &["workflow.run.write"],
                ) {
                    syscall_forbidden(op)
                } else if let Some(run_id) = input.args["run_id"].as_str() {
                    match self.store.request_server_run_cancel(run_id).await {
                        Ok(()) => match self.store.get_server_run(run_id).await {
                            Ok(Some(run)) => syscall_result(
                                &input.version,
                                op,
                                json_effect(StatusCode::ACCEPTED, server_run_value(run)),
                            ),
                            Ok(None) => effect_error(
                                StatusCode::NOT_FOUND,
                                "KOGWISTAR_RUN_NOT_FOUND",
                                format!("Unknown run_id: {run_id}"),
                            ),
                            Err(error) => effect_error(
                                StatusCode::INTERNAL_SERVER_ERROR,
                                "KOGWISTAR_STORE_ERROR",
                                error.to_string(),
                            ),
                        },
                        Err(error) => effect_error(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            "KOGWISTAR_STORE_ERROR",
                            error.to_string(),
                        ),
                    }
                } else {
                    effect_error(
                        StatusCode::BAD_REQUEST,
                        "KOGWISTAR_INVALID_REQUEST",
                        "run_id is required",
                    )
                }
            }
            "checkpoint" => {
                if !syscall_allowed(
                    request,
                    &self.capabilities,
                    op,
                    false,
                    &["workflow.run.read"],
                ) {
                    syscall_forbidden(op)
                } else if let (Some(run_id), Some(step_seq)) = (
                    input.args["run_id"].as_str(),
                    input.args["step_seq"].as_i64(),
                ) {
                    match self.store.list_server_run_events(run_id, 0, 100_000).await {
                        Ok(events) => syscall_result(
                            &input.version,
                            op,
                            runtime_inspection_effect(
                                request,
                                run_id,
                                &["checkpoints", &step_seq.to_string()],
                                events,
                            ),
                        ),
                        Err(error) => effect_error(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            "KOGWISTAR_STORE_ERROR",
                            error.to_string(),
                        ),
                    }
                } else {
                    effect_error(
                        StatusCode::BAD_REQUEST,
                        "KOGWISTAR_INVALID_REQUEST",
                        "run_id and integer step_seq are required",
                    )
                }
            }
            "receive_message" => {
                if !syscall_allowed_in_namespace(
                    request,
                    &self.capabilities,
                    op,
                    false,
                    &["read_graph"],
                    "conversation",
                ) {
                    syscall_forbidden(op)
                } else if let Some(conversation_id) = input
                    .args
                    .get("conversation_id")
                    .and_then(Value::as_str)
                {
                    let authorization = match self.authorize_conversation_transcript(request) {
                        Ok(value) => value,
                        Err(response) => return syscall_result(&input.version, op, response),
                    };
                    match self
                        .conversation_transcript_projection(
                            request,
                            InternalConversationTranscriptRequest {
                                conversation_id: conversation_id.to_owned(),
                                limit: 10_000,
                            },
                            authorization,
                        )
                        .await
                    {
                        Ok(transcript) => syscall_result(
                            &input.version,
                            op,
                            json_effect(StatusCode::OK, legacy_receive_message_value(transcript)),
                        ),
                        Err(response) => syscall_result(&input.version, op, response),
                    }
                } else {
                    json_effect(
                        StatusCode::NOT_FOUND,
                        json!({"detail": "'conversation_id'"}),
                    )
                }
            }
            "send_message" => {
                if !syscall_allowed_in_namespace(
                    request,
                    &self.capabilities,
                    op,
                    true,
                    &["send_message"],
                    "conversation",
                ) {
                    syscall_forbidden(op)
                } else {
                    let response = self.send_message_syscall(request, &input).await;
                    syscall_result(&input.version, op, response)
                }
            }
            "mount_memory" => {
                if !syscall_allowed_in_namespace(
                    request,
                    &self.capabilities,
                    op,
                    false,
                    &["project_view"],
                    "conversation",
                ) {
                    syscall_forbidden(op)
                } else if let Some(conversation_id) = input.args["conversation_id"].as_str() {
                    let authorization = match self.authorize_context_snapshot(request) {
                        Ok(value) => value,
                        Err(response) => return syscall_result(&input.version, op, response),
                    };
                    let run_id = match context_snapshot_syscall_field(&input.args, "run_id") {
                        Ok(value) => value,
                        Err(error) => {
                            return syscall_result(
                                &input.version,
                                op,
                                effect_error(
                                    StatusCode::UNPROCESSABLE_ENTITY,
                                    "KOGWISTAR_INVALID_REQUEST",
                                    error,
                                ),
                            );
                        }
                    };
                    let stage = match context_snapshot_syscall_field(&input.args, "stage") {
                        Ok(value) => value,
                        Err(error) => {
                            return syscall_result(
                                &input.version,
                                op,
                                effect_error(
                                    StatusCode::UNPROCESSABLE_ENTITY,
                                    "KOGWISTAR_INVALID_REQUEST",
                                    error,
                                ),
                            );
                        }
                    };
                    match self
                        .context_snapshot_projection(
                            request,
                            InternalContextSnapshotRequest {
                                conversation_id: conversation_id.to_owned(),
                                run_id,
                                stage,
                            },
                            authorization,
                        )
                        .await
                    {
                        Ok(snapshot) => syscall_result(
                            &input.version,
                            op,
                            json_effect(StatusCode::OK, public_context_snapshot_value(snapshot)),
                        ),
                        Err(response) => syscall_result(&input.version, op, response),
                    }
                } else {
                    effect_error(
                        StatusCode::BAD_REQUEST,
                        "KOGWISTAR_INVALID_REQUEST",
                        "conversation_id is required",
                    )
                }
            }
            "invoke_tool" => {
                if !syscall_allowed_in_namespace(
                    request,
                    &self.capabilities,
                    op,
                    true,
                    &[],
                    "conversation",
                ) {
                    syscall_forbidden(op)
                } else {
                    self.invoke_builtin_tool_syscall(request, &input).await
                }
            }
            "project_view" => {
                if !syscall_allowed(
                    request,
                    &self.capabilities,
                    op,
                    false,
                    &["read_security_scope", "project_view"],
                ) {
                    syscall_forbidden(op)
                } else {
                    let (runs, lanes, events) = match self.load_resource_snapshot_rows().await {
                        Ok(rows) => rows,
                        Err(response) => return response,
                    };
                    let service_rows =
                        match self.store.list_named_projections("service_registry").await {
                            Ok(rows) => rows,
                            Err(error) => {
                                return effect_error(
                                    StatusCode::INTERNAL_SERVER_ERROR,
                                    "KOGWISTAR_STORE_ERROR",
                                    error.to_string(),
                                );
                            }
                        };
                    let services = service_rows
                        .into_iter()
                        .map(|row| Value::Object(row.payload))
                        .collect::<Vec<_>>();
                    syscall_result(
                        &input.version,
                        op,
                        operational_effect(
                            request,
                            "/api/workflow/resources",
                            runs,
                            lanes,
                            events,
                            services,
                            Vec::new(),
                        ),
                    )
                }
            }
            "resume" => {
                if !syscall_allowed(
                    request,
                    &self.capabilities,
                    op,
                    true,
                    &["workflow.run.write"],
                ) {
                    syscall_forbidden(op)
                } else if let Some(run_id) = input.args["run_id"].as_str() {
                    let nested = ApiEffectRequest {
                        contract_version: request.contract_version,
                        method: "POST".to_owned(),
                        path_and_query: format!("/api/workflow/runs/{run_id}/resume"),
                        body: serde_json::to_vec(&input.args).expect("syscall args serialize"),
                        principal: request.principal.clone(),
                    };
                    syscall_result(
                        &input.version,
                        op,
                        self.resume_runtime_run(run_id, &nested).await,
                    )
                } else {
                    effect_error(
                        StatusCode::BAD_REQUEST,
                        "KOGWISTAR_INVALID_REQUEST",
                        "run_id is required",
                    )
                }
            }
            "request_approval" => {
                if !syscall_allowed(request, &self.capabilities, op, true, &["approve_action"]) {
                    syscall_forbidden(op)
                } else {
                    request_approval_syscall(request, &self.capabilities, &input)
                }
            }
            _ => unavailable_effect(request.clone()),
        };
        syscall_audit(
            &self.capabilities,
            &input.version,
            op,
            syscall_audit_status(&response),
        );
        response
    }
}
