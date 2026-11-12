const INTERNAL_DOCUMENT_PROJECTION_ROUTE: &str = "/internal/document/projection";
const INTERNAL_DOCUMENT_LOOKUP_ROUTE: &str = "/internal/document/lookup";
const PUBLIC_DOCUMENT_UPSERT_ROUTE: &str = "/api/document";
const PUBLIC_GRAPH_UPSERT_ROUTE: &str = "/api/graph/upsert";
const PUBLIC_DOCUMENT_TREE_UPSERT_ROUTE: &str = "/api/document.upsert_tree";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DocumentLookupRequest {
    file_name: String,
}

async fn document_id_lookup_effect<S: DocumentStore>(
    request: &ApiEffectRequest,
    store: &S,
    capabilities: &SharedCapabilityState,
) -> ApiEffectResponse {
    if !capability_allowed(request, capabilities, "read_graph", &["read_graph"]) {
        return effect_error(
            StatusCode::FORBIDDEN,
            "KOGWISTAR_CAPABILITY_FORBIDDEN",
            "Document lookup requires read_graph capability",
        );
    }
    if !store.supports_document_projection_lookup() {
        return effect_error(
            StatusCode::NOT_IMPLEMENTED,
            "KOGWISTAR_DOCUMENT_LOOKUP_UNSUPPORTED",
            "The configured document store does not support scoped document lookup",
        );
    }
    let input = match serde_json::from_slice::<DocumentLookupRequest>(&request.body) {
        Ok(input) if !input.file_name.trim().is_empty() => input,
        Ok(_) => {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                "file_name must be a non-empty string",
            );
        }
        Err(error) => {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                error.to_string(),
            );
        }
    };
    let security_scope = security_scope_value(&request.principal);
    let scope = GraphScope {
        namespace: security_scope["storage_namespace"]
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
    match store
        .document_projection_ids(scope, vec![input.file_name])
        .await
    {
        Ok(ids) => json_effect(StatusCode::OK, json!({"ids": ids})),
        Err(error) => effect_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "KOGWISTAR_STORE_ERROR",
            error.to_string(),
        ),
    }
}

fn is_document_projection_route(method: &str, path_and_query: &str) -> bool {
    method == "POST"
        && matches!(
            path_and_query.split('?').next(),
            Some(INTERNAL_DOCUMENT_PROJECTION_ROUTE | PUBLIC_DOCUMENT_UPSERT_ROUTE)
        )
}

fn is_public_graph_upsert_route(method: &str, path_and_query: &str) -> bool {
    method == "POST" && path_and_query.split('?').next() == Some(PUBLIC_GRAPH_UPSERT_ROUTE)
}

fn graph_upsert_commit_error(error: SqliteStoreError) -> ApiEffectResponse {
    match error {
        SqliteStoreError::TransactionAborted(message) => effect_error(
            StatusCode::CONFLICT,
            "KOGWISTAR_GRAPH_UPSERT_CONFLICT",
            message,
        ),
        SqliteStoreError::Io(error) => effect_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "KOGWISTAR_STORE_ERROR",
            error.to_string(),
        ),
        error => effect_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "KOGWISTAR_STORE_ERROR",
            error.to_string(),
        ),
    }
}

fn is_public_document_tree_upsert_route(method: &str, path_and_query: &str) -> bool {
    method == "POST" && path_and_query.split('?').next() == Some(PUBLIC_DOCUMENT_TREE_UPSERT_ROUTE)
}

fn document_tree_upsert_graph_request(
    request: &ApiEffectRequest,
) -> Result<(ApiEffectRequest, usize, usize), ApiEffectResponse> {
    let payload: Value = serde_json::from_slice(&request.body).map_err(|error| {
        effect_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            "KOGWISTAR_INVALID_REQUEST",
            error.to_string(),
        )
    })?;
    let Some(payload) = payload.as_object() else {
        return Err(effect_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            "KOGWISTAR_INVALID_REQUEST",
            "document tree upsert body must be an object",
        ));
    };
    let doc_id = payload.get("doc_id").and_then(Value::as_str);
    if !doc_id.is_some_and(|value| !value.trim().is_empty()) {
        return Err(effect_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            "KOGWISTAR_INVALID_REQUEST",
            "doc_id must be a non-empty string",
        ));
    }
    let nodes = payload
        .get("nodes")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                "nodes must be an array",
            )
        })?;
    let edges = match payload.get("edges") {
        None => Vec::new(),
        Some(Value::Array(edges)) => edges.clone(),
        Some(_) => {
            return Err(effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                "edges must be an array",
            ));
        }
    };
    if nodes.iter().chain(&edges).any(|entity| !entity.is_object()) {
        return Err(effect_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            "KOGWISTAR_INVALID_REQUEST",
            "nodes and edges must contain objects",
        ));
    }
    let node_count = nodes.len();
    let edge_count = edges.len();
    let graph_payload = json!({
        "contract_version": 1,
        "doc_id": doc_id,
        "content": null,
        "doc_type": "text",
        "insertion_method": payload
            .get("insertion_method")
            .cloned()
            .unwrap_or_else(|| json!("document_parser_v1")),
        "nodes": nodes,
        "edges": edges,
    });
    let mut graph_request = request.clone();
    graph_request.path_and_query = PUBLIC_GRAPH_UPSERT_ROUTE.to_owned();
    graph_request.body = serde_json::to_vec(&graph_payload).map_err(|error| {
        effect_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "KOGWISTAR_REQUEST_ENCODING_FAILED",
            error.to_string(),
        )
    })?;
    Ok((graph_request, node_count, edge_count))
}

fn document_projection_input(
    request: &ApiEffectRequest,
) -> Result<DocumentUpsertV1, ApiEffectResponse> {
    let input: DocumentUpsertV1 = serde_json::from_slice(&request.body).map_err(|error| {
        effect_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            "KOGWISTAR_INVALID_REQUEST",
            error.to_string(),
        )
    })?;
    input.validate().map_err(|error| {
        effect_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            "KOGWISTAR_INVALID_REQUEST",
            error.to_string(),
        )
    })?;
    Ok(input)
}

// PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/document_graph.rs#document_projection_write; python=kogwistar/server_mcp_with_admin.py#document_upsert; fixture=contracts/golden/document-upsert-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/part_03.rs#postgres_document_projection_route_persists_reopens_and_retries_when_dsn_available; python_test=tests/server/test_document_api_auth.py#test_document_upsert_keeps_rw_write_path
fn document_projection_write(
    request: &ApiEffectRequest,
    capabilities: &SharedCapabilityState,
) -> Result<DocumentProjectionUpsert, ApiEffectResponse> {
    if !capability_allowed(request, capabilities, "write_graph", &["write_graph"]) {
        return Err(effect_error(
            StatusCode::FORBIDDEN,
            "KOGWISTAR_CAPABILITY_FORBIDDEN",
            "Document projection requires write_graph capability",
        ));
    }
    let input = document_projection_input(request)?;
    if input.doc_type != "text" {
        return Err(effect_error(
            StatusCode::NOT_IMPLEMENTED,
            "KOGWISTAR_DOCUMENT_PROJECTION_UNSUPPORTED",
            "Rust document upsert currently supports normalized text content only",
        ));
    }
    let scope = security_scope_value(&request.principal);
    let mut metadata = input.metadata;
    metadata.insert("doc_type".to_owned(), Value::String(input.doc_type));
    metadata.insert(
        "insertion_method".to_owned(),
        Value::String(input.insertion_method),
    );
    Ok(DocumentProjectionUpsert {
        scope: GraphScope {
            namespace: scope["storage_namespace"]
                .as_str()
                .unwrap_or("docs")
                .to_owned(),
            workspace_id: scope["workspace"].as_str().map(str::to_owned),
            graph_space: scope["graph_space"].as_str().map(str::to_owned),
        },
        record: GraphRecord {
            id: input.doc_id,
            document: Some(input.content),
            metadata,
            embedding: None,
        },
        embedding_dim: 0,
    })
}

async fn embed_document_projection(
    mut write: DocumentProjectionUpsert,
    provider: Option<&dyn EmbeddingProvider>,
) -> Result<DocumentProjectionUpsert, ()> {
    let Some(provider) = provider else {
        return Ok(write);
    };
    let profile = provider.profile();
    profile.validate().map_err(|_| ())?;
    let text = write.record.document.as_ref().ok_or(())?.clone();
    let vectors = provider.embed(std::slice::from_ref(&text)).await.map_err(|_| ())?;
    if vectors.len() != 1 || vectors[0].len() != profile.dimension {
        return Err(());
    }
    let vector = vectors[0]
        .iter()
        .map(|value| {
            let value = *value as f32;
            value.is_finite().then_some(value).ok_or(())
        })
        .collect::<Result<Vec<_>, _>>()?;
    write.record.embedding = Some(vector);
    write.embedding_dim = profile.dimension;
    Ok(write)
}

fn graph_upsert_request(
    request: &ApiEffectRequest,
) -> Result<GraphExtractionRequestV1, ApiEffectResponse> {
    let input: GraphExtractionRequestV1 =
        serde_json::from_slice(&request.body).map_err(|error| {
            effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                error.to_string(),
            )
        })?;
    input.validate().map_err(|error| {
        effect_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            "KOGWISTAR_INVALID_REQUEST",
            error.to_string(),
        )
    })?;
    if input.doc_type != "text" {
        return Err(effect_error(
            StatusCode::NOT_IMPLEMENTED,
            "KOGWISTAR_GRAPH_UPSERT_UNSUPPORTED",
            "Rust graph upsert currently supports normalized text documents only",
        ));
    }
    Ok(input)
}

fn graph_upsert_should_write_document(
    inline_content: Option<&str>,
    stored_content: Option<&str>,
) -> bool {
    inline_content.is_some() || !stored_content.is_some_and(|content| !content.is_empty())
}

fn graph_upsert_endpoint_snapshot_matches(
    actual: &[GraphRecord],
    expected: &[GraphRecord],
) -> bool {
    if actual.len() != expected.len() {
        return false;
    }
    let expected_by_id = expected
        .iter()
        .map(|record| (record.id.as_str(), record))
        .collect::<BTreeMap<_, _>>();
    actual.iter().all(|record| {
        expected_by_id
            .get(record.id.as_str())
            .is_some_and(|expected| {
                record.document == expected.document && record.metadata == expected.metadata
            })
    })
}

#[derive(Clone, Debug, Default)]
pub(crate) struct GraphUpsertExternalEndpoints {
    pub node_ids: std::collections::BTreeSet<String>,
    pub edge_ids: std::collections::BTreeSet<String>,
}

pub(crate) fn graph_upsert_external_endpoints(
    input: &GraphExtractionRequestV1,
) -> Result<GraphUpsertExternalEndpoints, String> {
    let local_node_ids = input
        .nodes
        .iter()
        .flat_map(|node| [node.get("local_id"), node.get("id")])
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect::<std::collections::BTreeSet<_>>();
    let local_edge_ids = input
        .edges
        .iter()
        .flat_map(|edge| [edge.get("local_id"), edge.get("id")])
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect::<std::collections::BTreeSet<_>>();
    let mut external = GraphUpsertExternalEndpoints::default();
    for edge in &input.edges {
        for field in ["source_ids", "target_ids"] {
            let endpoints = edge
                .get(field)
                .and_then(Value::as_array)
                .ok_or_else(|| format!("edge {field} must be an array"))?;
            for endpoint in endpoints {
                let id = endpoint
                    .as_str()
                    .ok_or_else(|| format!("edge {field} entries must be strings"))?;
                if id.is_empty() {
                    return Err(format!("edge {field} endpoint IDs must not be empty"));
                }
                if !local_node_ids.contains(id) && !id.starts_with("nn:") {
                    external.node_ids.insert(id.to_owned());
                }
            }
        }
        for field in ["source_edge_ids", "target_edge_ids"] {
            let Some(endpoints) = edge.get(field).and_then(Value::as_array) else {
                continue;
            };
            for endpoint in endpoints {
                let id = endpoint
                    .as_str()
                    .ok_or_else(|| format!("edge {field} entries must be strings"))?;
                if id.is_empty() {
                    return Err(format!("edge {field} endpoint IDs must not be empty"));
                }
                if !local_edge_ids.contains(id) && !id.starts_with("ne:") {
                    external.edge_ids.insert(id.to_owned());
                }
            }
        }
    }
    Ok(external)
}

fn normalize_extraction_entity(
    raw: &serde_json::Map<String, Value>,
    doc_id: &str,
    insertion_method: &str,
    id: String,
    is_edge: bool,
) -> Result<Value, String> {
    let mut entity = raw.clone();
    entity.remove("local_id");
    entity.remove("references");
    entity.remove("groundings");
    entity.insert("id".to_owned(), Value::String(id));
    entity.insert("doc_id".to_owned(), Value::String(doc_id.to_owned()));
    entity.remove("embedding");
    entity.insert("metadata".to_owned(), Value::Object(serde_json::Map::new()));
    let metadata = entity
        .get_mut("metadata")
        .and_then(Value::as_object_mut)
        .ok_or_else(|| "graph metadata must be an object".to_owned())?;
    metadata
        .entry("lifecycle_status".to_owned())
        .or_insert_with(|| Value::String("active".to_owned()));
    metadata
        .entry("redirect_to_id".to_owned())
        .or_insert(Value::Null);

    let raw_mentions = entity
        .remove("mentions")
        .or_else(|| raw.get("groundings").cloned());
    let mentions = match raw_mentions {
        Some(Value::Array(mentions)) => mentions,
        Some(_) => return Err("mentions must be an array".to_owned()),
        None => raw
            .get("references")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .map(|span| json!({"spans": [span]}))
            .collect(),
    };
    let mentions = mentions
        .into_iter()
        .map(|mention| normalize_extraction_mention(mention, doc_id, insertion_method))
        .collect::<Result<Vec<_>, _>>()?;
    if mentions.is_empty() {
        return Err("graph entities require at least one source mention".to_owned());
    }
    entity.insert("mentions".to_owned(), Value::Array(mentions));
    if is_edge {
        for field in ["source_ids", "target_ids"] {
            if !entity.get(field).is_some_and(Value::is_array) {
                return Err(format!("edge {field} must be an array"));
            }
        }
        for field in ["source_edge_ids", "target_edge_ids"] {
            entity.entry(field.to_owned()).or_insert(Value::Null);
        }
    }
    Ok(Value::Object(entity))
}

fn normalize_extraction_mention(
    mut mention: Value,
    doc_id: &str,
    insertion_method: &str,
) -> Result<Value, String> {
    if mention.get("spans").is_none() && mention.get("start_char").is_some() {
        mention = json!({"spans": [mention]});
    }
    let spans = mention
        .get_mut("spans")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| "each graph mention must contain spans".to_owned())?;
    for (index, span) in spans.iter_mut().enumerate() {
        let object = span
            .as_object_mut()
            .ok_or_else(|| "mention spans must be objects".to_owned())?;
        object
            .entry("doc_id".to_owned())
            .or_insert_with(|| Value::String(doc_id.to_owned()));
        object
            .entry("collection_page_url".to_owned())
            .or_insert_with(|| Value::String(format!("doc://{doc_id}")));
        object
            .entry("document_page_url".to_owned())
            .or_insert_with(|| Value::String(format!("doc://{doc_id}#span-{index}")));
        object
            .entry("insertion_method".to_owned())
            .or_insert_with(|| Value::String(insertion_method.to_owned()));
        object
            .entry("page_number".to_owned())
            .or_insert(Value::from(1));
        object
            .entry("context_before".to_owned())
            .or_insert_with(|| Value::String(String::new()));
        object
            .entry("context_after".to_owned())
            .or_insert_with(|| Value::String(String::new()));
        object.entry("chunk_id".to_owned()).or_insert(Value::Null);
        object
            .entry("source_cluster_id".to_owned())
            .or_insert(Value::Null);
    }
    Ok(mention)
}

fn graph_upsert_mutation_batch(
    input: &GraphExtractionRequestV1,
    source_text: &str,
    security_scope: &Value,
    external_endpoints: &GraphUpsertExternalEndpoints,
    graph_role: &str,
) -> Result<(GraphMutationBatchInput, Vec<String>, Vec<String>), String> {
    use std::collections::BTreeMap;

    input
        .validate_source_spans(&BTreeMap::from([(
            input.doc_id.clone(),
            source_text.to_owned(),
        )]))
        .map_err(|error| error.to_string())?;

    let mut node_ids = BTreeMap::<String, String>::new();
    let mut edge_ids = BTreeMap::<String, String>::new();
    let mut normalized_nodes = Vec::with_capacity(input.nodes.len());
    let mut normalized_edges = Vec::with_capacity(input.edges.len());
    let mut result_nodes = Vec::with_capacity(input.nodes.len());
    let mut result_edges = Vec::with_capacity(input.edges.len());
    let mut edge_ids_in_input = Vec::with_capacity(input.edges.len());
    for raw in &input.nodes {
        let token = raw
            .get("local_id")
            .or_else(|| raw.get("id"))
            .and_then(Value::as_str);
        let id = match token {
            Some(value) if value.starts_with("nn:") => node_ids
                .entry(value.to_owned())
                .or_insert_with(|| uuid::Uuid::new_v4().to_string())
                .clone(),
            Some(value) if !value.is_empty() => value.to_owned(),
            Some(_) => return Err("node id must not be empty".to_owned()),
            None => uuid::Uuid::new_v4().to_string(),
        };
        for alias in [raw.get("local_id"), raw.get("id")].into_iter().flatten() {
            if let Some(alias) = alias.as_str() {
                node_ids.insert(alias.to_owned(), id.clone());
            }
        }
        result_nodes.push(id.clone());
        normalized_nodes.push(normalize_extraction_entity(
            raw,
            &input.doc_id,
            &input.insertion_method,
            id,
            false,
        )?);
    }
    for raw in &input.edges {
        let token = raw
            .get("local_id")
            .or_else(|| raw.get("id"))
            .and_then(Value::as_str);
        let id = match token {
            Some(value) if value.starts_with("ne:") => edge_ids
                .entry(value.to_owned())
                .or_insert_with(|| uuid::Uuid::new_v4().to_string())
                .clone(),
            Some(value) if !value.is_empty() => value.to_owned(),
            Some(_) => return Err("edge id must not be empty".to_owned()),
            None => uuid::Uuid::new_v4().to_string(),
        };
        for alias in [raw.get("local_id"), raw.get("id")].into_iter().flatten() {
            if let Some(alias) = alias.as_str() {
                edge_ids.insert(alias.to_owned(), id.clone());
            }
        }
        edge_ids_in_input.push(id.clone());
        result_edges.push(id.clone());
    }
    let mut pending_edges = Vec::with_capacity(input.edges.len());
    for (index, raw) in input.edges.iter().enumerate() {
        let id = edge_ids_in_input[index].clone();
        let mut entity =
            normalize_extraction_entity(raw, &input.doc_id, &input.insertion_method, id, true)?;
        for field in ["source_ids", "target_ids"] {
            let endpoints = entity[field]
                .as_array_mut()
                .ok_or_else(|| format!("edge {field} must be an array"))?;
            for endpoint in endpoints {
                let token = endpoint
                    .as_str()
                    .ok_or_else(|| format!("edge {field} entries must be strings"))?;
                *endpoint = Value::String(
                    node_ids
                        .get(token)
                        .cloned()
                        .or_else(|| {
                            external_endpoints
                                .node_ids
                                .contains(token)
                                .then(|| token.to_owned())
                        })
                        .ok_or_else(|| {
                            format!("edge endpoint {token:?} is not a node in this request")
                        })?,
                );
            }
        }
        pending_edges.push(entity);
    }
    let mut materialized_edge_ids = BTreeSet::new();
    while !pending_edges.is_empty() {
        let mut progress = false;
        let mut next = Vec::new();
        for mut entity in pending_edges {
            let mut dependencies_ready = true;
            for field in ["source_edge_ids", "target_edge_ids"] {
                if let Some(endpoints) = entity[field].as_array_mut() {
                    for endpoint in endpoints {
                        let token = endpoint
                            .as_str()
                            .ok_or_else(|| format!("edge {field} entries must be strings"))?;
                        let local_edge_id = edge_ids.get(token);
                        let resolved = local_edge_id.map(String::as_str).unwrap_or(token);
                        let resolved_local = local_edge_id.is_some()
                            || edge_ids.values().any(|edge_id| edge_id == token);
                        if !resolved_local && !external_endpoints.edge_ids.contains(token) {
                            return Err(format!(
                                "hyperedge endpoint {token:?} is not an edge in this request"
                            ));
                        }
                        if resolved_local && !materialized_edge_ids.contains(resolved) {
                            dependencies_ready = false;
                        }
                        *endpoint = Value::String(resolved.to_owned());
                    }
                }
            }
            if dependencies_ready {
                materialized_edge_ids.insert(
                    entity["id"]
                        .as_str()
                        .ok_or_else(|| "normalized edge has no id".to_owned())?
                        .to_owned(),
                );
                normalized_edges.push(entity);
                progress = true;
            } else {
                next.push(entity);
            }
        }
        if !progress {
            return Err("hyperedge endpoints contain a cycle or unresolved dependency".to_owned());
        }
        pending_edges = next;
    }

    for payload in normalized_nodes.iter_mut().chain(&mut normalized_edges) {
        payload["metadata"]["engine_type"] = Value::String(graph_role.to_owned());
    }

    let scope = GraphScope {
        namespace: security_scope["storage_namespace"]
            .as_str()
            .unwrap_or("docs")
            .to_owned(),
        workspace_id: security_scope["workspace"].as_str().map(str::to_owned),
        graph_space: security_scope["graph_space"].as_str().map(str::to_owned),
    };
    let mut mutations = Vec::with_capacity(normalized_nodes.len() + normalized_edges.len());
    for (entity_kind, table, model_class, entities) in [
        ("node", "gke_nodes", GraphModelClass::Node, normalized_nodes),
        ("edge", "gke_edges", GraphModelClass::Edge, normalized_edges),
    ] {
        for payload in entities {
            let id = payload["id"]
                .as_str()
                .ok_or_else(|| "normalized graph entity has no id".to_owned())?
                .to_owned();
            let envelope = GraphModelEnvelope::new(model_class, payload.clone());
            envelope
                .decode_persistent_v1()
                .map_err(|error| error.to_string())?;
            let mut document_payload = payload.clone();
            let object = document_payload
                .as_object_mut()
                .ok_or_else(|| "normalized graph entity must be an object".to_owned())?;
            if entity_kind == "node" {
                object.remove("metadata");
                object.remove("embedding");
            } else {
                object.insert("embedding".to_owned(), Value::Null);
            }
            let document =
                serde_json::to_string(&document_payload).map_err(|error| error.to_string())?;
            let mut metadata = payload
                .as_object()
                .and_then(|object| object.get("metadata"))
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default();
            for field in [
                "doc_id",
                "label",
                "type",
                "summary",
                "domain_id",
                "canonical_entity_id",
            ] {
                if let Some(value) = payload.get(field) {
                    metadata.insert(field.to_owned(), value.clone());
                }
            }
            if let Some(properties) = payload.get("properties") {
                metadata.insert(
                    "properties".to_owned(),
                    Value::String(
                        serde_json::to_string(properties).map_err(|error| error.to_string())?,
                    ),
                );
            }
            if let Some(mentions) = payload.get("mentions") {
                metadata.insert(
                    "mentions".to_owned(),
                    Value::String(
                        serde_json::to_string(mentions).map_err(|error| error.to_string())?,
                    ),
                );
            }
            if entity_kind == "edge" {
                for field in ["relation", "source_ids", "target_ids"] {
                    if let Some(value) = payload.get(field) {
                        metadata.insert(
                            field.to_owned(),
                            Value::String(
                                serde_json::to_string(value).map_err(|error| error.to_string())?,
                            ),
                        );
                    }
                }
            }
            metadata.insert(
                "namespace".to_owned(),
                Value::String(scope.namespace.clone()),
            );
            metadata.insert(
                "_class_name".to_owned(),
                Value::String(
                    if entity_kind == "node" {
                        "Node"
                    } else {
                        "Edge"
                    }
                    .to_owned(),
                ),
            );
            let mut record_metadata = serde_json::Map::new();
            for (key, value) in metadata {
                record_metadata.insert(key, value);
            }
            mutations.push(GraphMutationInput {
                scope: kogwistar_contracts::GraphScopeV1 {
                    namespace: scope.namespace.clone(),
                    workspace_id: scope.workspace_id.clone(),
                    graph_space: scope.graph_space.clone(),
                },
                table: table.to_owned(),
                entity_kind: entity_kind.to_owned(),
                event_id: uuid::Uuid::new_v4().to_string(),
                op: "ADD".to_owned(),
                payload: payload.clone(),
                record: kogwistar_contracts::GraphRecordV1 {
                    id,
                    document: Some(document),
                    metadata: record_metadata,
                    embedding: None,
                },
                embedding_dim: 0,
                model_envelope: Some(envelope),
            });
        }
    }
    let mut index_jobs = Vec::with_capacity(mutations.len());
    for mutation in &mutations {
        index_jobs.push(GraphIndexJobInput {
            job_id: uuid::Uuid::new_v4().to_string(),
            namespace: scope.namespace.clone(),
            entity_kind: mutation.entity_kind.clone(),
            entity_id: mutation.record.id.clone(),
            index_kind: "node_embedding".to_owned(),
            op: "UPSERT".to_owned(),
            payload_json: None,
            max_retries: 10,
        });
    }
    let batch = GraphMutationBatchInput {
        contract_version: 1,
        atomic: true,
        mutations,
        index_jobs,
    };
    if !batch.mutations.is_empty() {
        batch.validate().map_err(|error| error.to_string())?;
    }
    Ok((batch, result_nodes, result_edges))
}

fn normalize_graph_role(role: &str) -> Result<String, String> {
    if role.trim().is_empty() {
        return Err("graph role must not be empty".to_owned());
    }
    Ok(kogwistar_contracts::normalize_graph_kind(Some(role)))
}

impl SqliteRunApplicationService {
    async fn sqlite_public_graph_upsert_effect(
        &self,
        request: &ApiEffectRequest,
    ) -> ApiEffectResponse {
        if self.chroma_graph.is_none() {
            return effect_error(
                StatusCode::NOT_IMPLEMENTED,
                "KOGWISTAR_GRAPH_AUTHORITY_UNSUPPORTED",
                "Graph upsert requires SQLite canonical events and a configured Chroma projection",
            );
        }
        if principal_role(&request.principal) != "rw"
            || !capability_allowed_for_principal(
                &request.principal,
                &self.capabilities,
                "write_graph",
                &["write_graph"],
            )
        {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Graph upsert requires rw and write_graph capability",
            );
        }
        let Some(owner_id) = authenticated_run_owner_id(&request.principal) else {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_GRAPH_OWNER_REQUIRED",
                "Graph upsert requires an authenticated owner identity",
            );
        };
        let input = match graph_upsert_request(request) {
            Ok(input) => input,
            Err(response) => return response,
        };
        let security_scope = security_scope_value(&request.principal);
        let scope = GraphScope {
            namespace: security_scope["storage_namespace"]
                .as_str()
                .unwrap_or("docs")
                .to_owned(),
            workspace_id: security_scope["workspace"].as_str().map(str::to_owned),
            graph_space: security_scope["graph_space"].as_str().map(str::to_owned),
        };
        let external_endpoints = match graph_upsert_external_endpoints(&input) {
            Ok(endpoints) => endpoints,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error,
                );
            }
        };
        let owner_id = owner_id.to_owned();
        let acl_principal = graph_acl_principal_from_claims(
            &request.principal,
            security_scope["security_scope"].as_str(),
        );
        let acl_security_scope = security_scope["security_scope"].as_str().map(str::to_owned);
        let effect_id = uuid::Uuid::new_v4().to_string();
        let graph_role = self.graph_role.clone();
        let committed = self.store.immediate_transaction(|uow| {
            let existing_document = uow.source_document_projection(&scope, &input.doc_id)?;
            let source_text = input
                .content
                .clone()
                .or_else(|| {
                    existing_document
                        .as_ref()
                        .and_then(|record| record.document.clone())
                })
                .unwrap_or_default();
            let should_write_document = graph_upsert_should_write_document(
                input.content.as_deref(),
                existing_document
                    .as_ref()
                    .and_then(|record| record.document.as_deref()),
            );
            let (batch, node_ids, edge_ids) = graph_upsert_mutation_batch(
                &input,
                &source_text,
                &security_scope,
                &external_endpoints,
                &self.graph_role,
            )
            .map_err(SqliteStoreError::TransactionAborted)?;
            let mutations = graph_mutation_batch_value(batch.clone())
                .map_err(SqliteStoreError::TransactionAborted)?;
            validate_graph_mutation_envelopes(&batch, &mutations)
                .map_err(SqliteStoreError::TransactionAborted)?;
            if mutations
                .iter()
                .any(|mutation| !graph_mutation_scope_allowed(&mutation.scope, &security_scope))
            {
                return Err(SqliteStoreError::TransactionAborted(
                    "graph mutation scope is outside the principal storage scope".to_owned(),
                ));
            }

            if !mutations.is_empty() {
                let references = graph_mutation_references(&mutations);
                let target_nodes = mutations
                    .iter()
                    .filter(|mutation| mutation.entity_kind == "node")
                    .map(|mutation| mutation.record.id.clone())
                    .collect::<BTreeSet<_>>();
                let target_edges = mutations
                    .iter()
                    .filter(|mutation| mutation.entity_kind == "edge")
                    .map(|mutation| mutation.record.id.clone())
                    .collect::<BTreeSet<_>>();
                let mut external_edges = BTreeMap::<String, GraphRecord>::new();
                let mut endpoint_refs = references;
                loop {
                    let pending = endpoint_refs
                        .edge_ids
                        .iter()
                        .filter(|id| {
                            !target_edges.contains(*id) && !external_edges.contains_key(*id)
                        })
                        .cloned()
                        .collect::<Vec<_>>();
                    if pending.is_empty() {
                        break;
                    }
                    if endpoint_refs
                        .node_ids
                        .len()
                        .saturating_add(endpoint_refs.edge_ids.len())
                        > 10_000
                    {
                        return Err(SqliteStoreError::TransactionAborted(
                            "graph endpoint closure exceeds 10000 entities".to_owned(),
                        ));
                    }
                    let records = sqlite_canonical_graph_records(uow, &scope, "edge", &pending)?;
                    if records.len() != pending.len() {
                        return Err(SqliteStoreError::TransactionAborted(
                            "one or more external graph edges are missing from the principal scope"
                                .to_owned(),
                        ));
                    }
                    for record in records {
                        endpoint_refs.node_ids.extend(
                            ["source_ids", "target_ids"]
                                .into_iter()
                                .flat_map(|field| graph_record_string_ids(&record, field)),
                        );
                        endpoint_refs.edge_ids.extend(
                            ["source_edge_ids", "target_edge_ids"]
                                .into_iter()
                                .flat_map(|field| graph_record_string_ids(&record, field)),
                        );
                        external_edges.insert(record.id.clone(), record);
                    }
                }
                endpoint_refs
                    .node_ids
                    .retain(|id| !target_nodes.contains(id));
                endpoint_refs
                    .edge_ids
                    .retain(|id| !target_edges.contains(id));
                if endpoint_refs
                    .node_ids
                    .len()
                    .saturating_add(endpoint_refs.edge_ids.len())
                    > 10_000
                {
                    return Err(SqliteStoreError::TransactionAborted(
                        "graph endpoint closure exceeds 10000 entities".to_owned(),
                    ));
                }
                let mut node_ids_to_check =
                    endpoint_refs.node_ids.iter().cloned().collect::<Vec<_>>();
                node_ids_to_check.extend(target_nodes.iter().cloned());
                node_ids_to_check.sort();
                node_ids_to_check.dedup();
                let mut edge_ids_to_check =
                    endpoint_refs.edge_ids.iter().cloned().collect::<Vec<_>>();
                edge_ids_to_check.extend(target_edges.iter().cloned());
                edge_ids_to_check.sort();
                edge_ids_to_check.dedup();
                let nodes =
                    sqlite_canonical_graph_records(uow, &scope, "node", &node_ids_to_check)?;
                let edges =
                    sqlite_canonical_graph_records(uow, &scope, "edge", &edge_ids_to_check)?;
                if endpoint_refs
                    .node_ids
                    .iter()
                    .any(|id| !nodes.iter().any(|node| &node.id == id))
                    || endpoint_refs
                        .edge_ids
                        .iter()
                        .any(|id| !edges.iter().any(|edge| &edge.id == id))
                {
                    return Err(SqliteStoreError::TransactionAborted(
                        "one or more graph endpoints are missing from the principal scope"
                            .to_owned(),
                    ));
                }
                let existing_ids = nodes
                    .iter()
                    .chain(&edges)
                    .map(|record| record.id.clone())
                    .collect::<Vec<_>>();
                let acl_records =
                    uow.graph_acl_records_for_entities(&scope, "knowledge", &existing_ids)?;
                let mut acl_index = GraphAclIndex::default();
                acl_index.extend_from_records(&acl_records, "knowledge");
                let (visible_nodes, visible_edges) = graph_records_visible_by_acl_with_index(
                    &nodes,
                    &edges,
                    &acl_index,
                    &acl_principal,
                );
                let expected = nodes
                    .iter()
                    .map(|record| ("node", record.id.as_str()))
                    .chain(edges.iter().map(|record| ("edge", record.id.as_str())))
                    .collect::<BTreeSet<_>>();
                let visible = visible_nodes
                    .iter()
                    .map(|record| ("node", record.id.as_str()))
                    .chain(
                        visible_edges
                            .iter()
                            .map(|record| ("edge", record.id.as_str())),
                    )
                    .collect::<BTreeSet<_>>();
                if expected != visible
                    || nodes.iter().any(|record| {
                        target_nodes.contains(&record.id) && wisdom_record_is_tombstoned(record)
                    })
                    || edges.iter().any(|record| {
                        target_edges.contains(&record.id) && wisdom_record_is_tombstoned(record)
                    })
                {
                    return Err(SqliteStoreError::TransactionAborted(
                        "one or more graph endpoints are hidden or tombstoned".to_owned(),
                    ));
                }
                let existing_targets = nodes
                    .iter()
                    .filter(|record| target_nodes.contains(&record.id))
                    .map(|record| ("node".to_owned(), record.id.clone()))
                    .chain(
                        edges
                            .iter()
                            .filter(|record| target_edges.contains(&record.id))
                            .map(|record| ("edge".to_owned(), record.id.clone())),
                    )
                    .collect::<BTreeSet<_>>();
                let new_mutations = mutations
                    .iter()
                    .filter(|mutation| {
                        !existing_targets
                            .contains(&(mutation.entity_kind.clone(), mutation.record.id.clone()))
                    })
                    .cloned()
                    .collect::<Vec<_>>();
                let (acl_mutations, acl_jobs) = derive_default_acl_graph_effects_from_records(
                    &scope,
                    &new_mutations,
                    acl_targets_for_graph_mutations(&new_mutations).0,
                    acl_records,
                    &graph_role,
                    &owner_id,
                    acl_security_scope.as_deref(),
                    &effect_id,
                )
                .map_err(SqliteStoreError::TransactionAborted)?;
                let mut all_mutations = mutations;
                all_mutations.extend(acl_mutations);
                if should_write_document {
                    let mut metadata = serde_json::Map::new();
                    metadata.insert("doc_type".to_owned(), json!(input.doc_type));
                    metadata.insert("insertion_method".to_owned(), json!(input.insertion_method));
                    uow.upsert_source_document_projection(DocumentProjectionUpsert {
                        scope: scope.clone(),
                        record: GraphRecord {
                            id: input.doc_id.clone(),
                            document: Some(source_text.clone()),
                            metadata,
                            embedding: None,
                        },
                        embedding_dim: 0,
                    })?;
                }
                // SQLite creates the canonical revision-fenced embedding intent for each
                // graph event itself; the generic PostgreSQL DTO's duplicate intents are
                // not copied into this backend's transaction.
                uow.apply_canonical_graph_mutation_batch(GraphMutationBatch {
                    mutations: all_mutations,
                    index_jobs: acl_jobs,
                })?;
                return Ok((input.doc_id.clone(), node_ids, edge_ids));
            }

            if should_write_document {
                let mut metadata = serde_json::Map::new();
                metadata.insert("doc_type".to_owned(), json!(input.doc_type));
                metadata.insert("insertion_method".to_owned(), json!(input.insertion_method));
                uow.upsert_source_document_projection(DocumentProjectionUpsert {
                    scope: scope.clone(),
                    record: GraphRecord {
                        id: input.doc_id.clone(),
                        document: Some(source_text),
                        metadata,
                        embedding: None,
                    },
                    embedding_dim: 0,
                })?;
            }
            Ok((input.doc_id.clone(), node_ids, edge_ids))
        });
        match committed {
            Ok((document_id, node_ids, edge_ids)) => json_effect(
                StatusCode::OK,
                json!({
                    "document_id": document_id,
                    "node_ids": node_ids,
                    "edge_ids": edge_ids,
                    "nodes_added": node_ids.len(),
                    "edges_added": edge_ids.len(),
                }),
            ),
            Err(error) => graph_upsert_commit_error(error),
        }
    }
}

#[cfg(test)]
mod graph_upsert_tests {
    use super::*;
    use std::collections::BTreeMap;

    fn extraction_fixture() -> GraphExtractionRequestV1 {
        serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/graph-extraction-v1.json"
        )))
        .expect("shared graph extraction fixture")
    }

    fn principal_scope() -> Value {
        json!({
            "storage_namespace": "tenant-a",
            "workspace": "workspace-a",
            "graph_space": "facts",
        })
    }

    #[test]
    fn graph_upsert_storage_error_matches_shared_python_contract() {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/persistence-error-v1.json"
        )))
        .expect("shared persistence error fixture");
        let message = fixture["message"].as_str().expect("fixture message");
        let response = graph_upsert_commit_error(SqliteStoreError::Io(
            std::io::Error::other(message.to_owned()),
        ));
        assert_eq!(response.status, fixture["rust"]["status"].as_u64().unwrap() as u16);
        let body: Value = serde_json::from_slice(&response.body).expect("error response JSON");
        assert_eq!(body["code"], fixture["rust"]["code"]);
        assert_eq!(body["message"], fixture["rust"]["message"]);

        let generic = graph_upsert_commit_error(SqliteStoreError::UnsupportedSchemaVersion {
            found: 2,
            supported: 1,
        });
        let generic_contract = &fixture["generic_store_error"]["rust"];
        assert_eq!(generic.status, generic_contract["status"].as_u64().unwrap() as u16);
        let body: Value = serde_json::from_slice(&generic.body).expect("generic error JSON");
        assert_eq!(body["code"], generic_contract["code"]);
        assert_eq!(body["message"], generic_contract["message"]);

        let conflict = graph_upsert_commit_error(SqliteStoreError::TransactionAborted(
            "duplicate graph mutation".to_owned(),
        ));
        assert_eq!(
            conflict.status,
            fixture["transaction_conflict"]["status"]
                .as_u64()
                .unwrap() as u16
        );
        let body: Value = serde_json::from_slice(&conflict.body).expect("conflict response JSON");
        assert_eq!(body["code"], fixture["transaction_conflict"]["code"]);
    }

    #[test]
    fn graph_upsert_transaction_abort_matches_comparison_fixture() {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/transaction-abort-error-v1.json"
        )))
        .expect("shared transaction-abort comparison fixture");
        let message = fixture["message"].as_str().expect("fixture message");
        let response = graph_upsert_commit_error(SqliteStoreError::TransactionAborted(
            message.to_owned(),
        ));
        assert_eq!(response.status, fixture["rust"]["status"].as_u64().unwrap() as u16);
        let body: Value = serde_json::from_slice(&response.body).expect("error response JSON");
        assert_eq!(body["code"], fixture["rust"]["code"]);
        assert_eq!(body["message"], fixture["rust"]["message"]);
        assert_eq!(fixture["parity"], false);
    }

    #[test]
    fn graph_role_configuration_matches_python_aliases_and_fails_closed_when_blank() {
        for (input, expected) in [
            ("knowledge", "knowledge"),
            (" KG ", "knowledge"),
            ("chat", "conversation"),
            ("conv", "conversation"),
            ("dialog", "conversation"),
            ("wf", "workflow"),
            ("w", "wisdom"),
            ("custom-role", "custom-role"),
        ] {
            assert_eq!(normalize_graph_role(input).unwrap(), expected);
        }
        assert!(normalize_graph_role("  ").is_err());
    }

    #[test]
    fn legacy_document_tree_request_maps_to_native_graph_contract() {
        let request = ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/document.upsert_tree".to_owned(),
            body: serde_json::to_vec(&json!({
                "doc_id": "doc-tree",
                "nodes": [{"id": "node-1"}],
                "edges": [{"id": "edge-1"}],
                "legacy_extra": "ignored by the Python request model",
            }))
            .unwrap(),
            principal: json!({"sub": "alice"}),
        };
        let (native, node_count, edge_count) =
            document_tree_upsert_graph_request(&request).expect("legacy request maps");
        let body: Value = serde_json::from_slice(&native.body).unwrap();
        assert_eq!(native.path_and_query, PUBLIC_GRAPH_UPSERT_ROUTE);
        assert_eq!(body["doc_id"], "doc-tree");
        assert_eq!(body["doc_type"], "text");
        assert_eq!(body["insertion_method"], "document_parser_v1");
        assert_eq!(body["content"], Value::Null);
        assert_eq!((node_count, edge_count), (1, 1));
        assert_eq!(native.principal, request.principal);
    }

    #[test]
    fn legacy_document_tree_request_rejects_invalid_entity_envelopes() {
        for payload in [
            json!({"doc_id": "doc-tree"}),
            json!({"doc_id": "doc-tree", "nodes": [null]}),
            json!({"doc_id": "doc-tree", "nodes": [], "edges": "not-an-array"}),
            json!({"doc_id": "", "nodes": []}),
        ] {
            let request = ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/document.upsert_tree".to_owned(),
                body: serde_json::to_vec(&payload).unwrap(),
                principal: Value::Null,
            };
            let error = document_tree_upsert_graph_request(&request)
                .expect_err("invalid legacy request must fail closed");
            assert_eq!(error.status, StatusCode::UNPROCESSABLE_ENTITY);
        }
    }

    #[test]
    fn graph_upsert_builds_typed_atomic_batch_from_shared_fixture() {
        let input = extraction_fixture();
        let source = input.content.as_deref().expect("fixture source text");
        let (batch, nodes, edges) = graph_upsert_mutation_batch(
            &input,
            source,
            &principal_scope(),
            &GraphUpsertExternalEndpoints::default(),
            "knowledge",
        )
        .expect("valid fixture converts to a native graph batch");
        assert_eq!(nodes.len(), 2);
        assert_eq!(edges.len(), 1);
        assert!(nodes.iter().all(|id| !id.starts_with("nn:")));
        assert!(edges.iter().all(|id| !id.starts_with("ne:")));
        assert_eq!(batch.mutations.len(), 3);
        assert_eq!(batch.index_jobs.len(), 3);
        batch.validate().expect("atomic batch is valid");
        let mutations = graph_mutation_batch_value(batch.clone()).expect("store DTO conversion");
        validate_graph_mutation_envelopes(&batch, &mutations)
            .expect("persistent graph envelopes match their mutations");
        let edge = batch
            .mutations
            .iter()
            .find(|mutation| mutation.entity_kind == "edge")
            .expect("edge mutation");
        assert_eq!(edge.payload["source_ids"][0], nodes[0]);
        assert_eq!(edge.payload["target_ids"][0], nodes[1]);
        assert_eq!(edge.scope.namespace, "tenant-a");
        assert_eq!(edge.scope.workspace_id.as_deref(), Some("workspace-a"));
        assert_eq!(edge.scope.graph_space.as_deref(), Some("facts"));
    }

    #[test]
    fn graph_role_changes_metadata_without_changing_generic_storage_contract() {
        let input = extraction_fixture();
        let source = input.content.as_deref().expect("fixture source text");
        for role in [
            "knowledge",
            "conversation",
            "workflow",
            "wisdom",
            "acl",
            "custom:research",
        ] {
            let (batch, _, _) = graph_upsert_mutation_batch(
                &input,
                source,
                &principal_scope(),
                &GraphUpsertExternalEndpoints::default(),
                role,
            )
            .expect("each graph role uses the generic graph-upsert contract");
            assert_eq!(batch.mutations.len(), 3);
            for mutation in &batch.mutations {
                let expected_table = match mutation.entity_kind.as_str() {
                    "node" => "gke_nodes",
                    "edge" => "gke_edges",
                    other => panic!("unexpected entity kind {other:?}"),
                };
                assert_eq!(mutation.table, expected_table);
                assert_eq!(mutation.scope.namespace, "tenant-a");
                assert_eq!(mutation.scope.workspace_id.as_deref(), Some("workspace-a"));
                assert_eq!(mutation.scope.graph_space.as_deref(), Some("facts"));
                assert_eq!(mutation.payload["metadata"]["engine_type"], role);
                assert_eq!(
                    mutation
                        .record
                        .metadata
                        .get("_class_name")
                        .and_then(Value::as_str),
                    Some(if mutation.entity_kind == "node" {
                        "Node"
                    } else {
                        "Edge"
                    })
                );
            }
        }
    }

    #[test]
    fn graph_upsert_can_validate_against_host_loaded_existing_document_text() {
        let mut input = extraction_fixture();
        let source = input.content.take().expect("fixture source text");
        let (batch, nodes, edges) = graph_upsert_mutation_batch(
            &input,
            &source,
            &principal_scope(),
            &GraphUpsertExternalEndpoints::default(),
            "knowledge",
        )
        .expect("stored source text validates an extraction without inline content");
        assert_eq!((batch.mutations.len(), nodes.len(), edges.len()), (3, 2, 1));
    }

    #[test]
    fn graph_upsert_creates_placeholder_when_stored_document_has_no_text() {
        assert!(graph_upsert_should_write_document(None, None));
        assert!(graph_upsert_should_write_document(None, Some("")));
        assert!(!graph_upsert_should_write_document(
            None,
            Some("stored source")
        ));
        assert!(graph_upsert_should_write_document(
            Some(""),
            Some("stored source")
        ));
    }

    #[test]
    fn graph_upsert_checks_exact_source_and_does_not_accept_caller_acl_metadata() {
        let mut input = extraction_fixture();
        input.nodes[0].insert(
            "metadata".to_owned(),
            json!({"lifecycle_status":"tombstoned", "acl_mode":"public", "engine_type":"attacker"}),
        );
        let source = input.content.as_deref().expect("fixture source text");
        let (batch, _, _) = graph_upsert_mutation_batch(
            &input,
            source,
            &principal_scope(),
            &GraphUpsertExternalEndpoints::default(),
            "conversation",
        )
        .expect("LLM-only input is normalized without trusting metadata");
        let node = batch
            .mutations
            .iter()
            .find(|mutation| mutation.entity_kind == "node")
            .expect("node mutation");
        assert_eq!(node.payload["metadata"]["lifecycle_status"], "active");
        assert_eq!(node.payload["metadata"]["engine_type"], "conversation");
        assert!(node.payload["metadata"].get("acl_mode").is_none());

        let mut tampered = extraction_fixture();
        tampered.nodes[0]["references"][0]["excerpt"] = json!("Fabricated");
        let sources = BTreeMap::from([(
            tampered.doc_id.clone(),
            tampered.content.clone().expect("fixture source text"),
        )]);
        assert!(tampered.validate_source_spans(&sources).is_err());
    }

    #[test]
    fn graph_upsert_rejects_dangling_node_endpoints() {
        let mut input = extraction_fixture();
        input.edges[0]["source_ids"][0] = json!("nn:unknown");
        let source = input.content.as_deref().expect("fixture source text");
        assert!(
            graph_upsert_mutation_batch(
                &input,
                source,
                &principal_scope(),
                &GraphUpsertExternalEndpoints::default(),
                "knowledge",
            )
            .is_err()
        );
    }

    #[test]
    fn graph_upsert_orders_hyperedges_by_dependency_not_input_order() {
        let mut input = extraction_fixture();
        input.edges[0]["source_edge_ids"] = json!(["ne:inner"]);
        let mut inner = input.edges[0].clone();
        inner.insert("local_id".to_owned(), json!("ne:inner"));
        inner.insert("id".to_owned(), json!("ne:inner"));
        inner.insert("source_edge_ids".to_owned(), Value::Null);
        inner.insert("target_edge_ids".to_owned(), Value::Null);
        input.edges.push(inner);
        let source = input.content.as_deref().expect("fixture source text");
        let (batch, _, edge_ids) = graph_upsert_mutation_batch(
            &input,
            source,
            &principal_scope(),
            &GraphUpsertExternalEndpoints::default(),
            "knowledge",
        )
        .expect("forward-declared hyperedge is topologically ordered");
        assert_eq!(edge_ids.len(), 2);
        let ordered = batch
            .mutations
            .iter()
            .filter(|mutation| mutation.entity_kind == "edge")
            .map(|mutation| mutation.record.id.as_str())
            .collect::<Vec<_>>();
        assert_eq!(ordered, [edge_ids[1].as_str(), edge_ids[0].as_str()]);
        input.edges[1]["source_edge_ids"] = json!(["ne:connects"]);
        assert!(
            graph_upsert_mutation_batch(
                &input,
                source,
                &principal_scope(),
                &GraphUpsertExternalEndpoints::default(),
                "knowledge",
            )
            .is_err_and(|error| error.contains("cycle"))
        );
    }

    #[test]
    fn graph_upsert_preserves_declared_external_node_endpoint_for_host_validation() {
        let input = extraction_fixture();
        let source = input.content.clone().expect("fixture source text");
        let external_id = "persisted-node".to_owned();
        let external_endpoints = GraphUpsertExternalEndpoints {
            node_ids: std::collections::BTreeSet::from([external_id.clone()]),
            edge_ids: std::collections::BTreeSet::new(),
        };
        let mut input = input;
        input.edges[0]["source_ids"][0] = Value::String(external_id.clone());
        let (batch, _, _) =
            graph_upsert_mutation_batch(
                &input,
                &source,
                &principal_scope(),
                &external_endpoints,
                "knowledge",
            )
                .expect("host-validated external node references remain canonical");
        let edge = batch
            .mutations
            .iter()
            .find(|mutation| mutation.entity_kind == "edge")
            .expect("edge mutation");
        assert_eq!(edge.payload["source_ids"][0], external_id);
    }

    #[test]
    fn graph_upsert_preserves_declared_external_edge_endpoint_for_host_validation() {
        let mut input = extraction_fixture();
        let mut nested = input.edges[0].clone();
        nested.insert("id".to_owned(), json!("ne:nested-local"));
        nested.insert("source_edge_ids".to_owned(), json!(["persisted-edge"]));
        input.edges.push(nested);
        let source = input.content.clone().expect("fixture source text");
        let external_endpoints = GraphUpsertExternalEndpoints {
            node_ids: std::collections::BTreeSet::new(),
            edge_ids: std::collections::BTreeSet::from(["persisted-edge".to_owned()]),
        };
        let (batch, _, _) =
            graph_upsert_mutation_batch(
                &input,
                &source,
                &principal_scope(),
                &external_endpoints,
                "knowledge",
            )
                .expect("host-validated external edge references remain canonical");
        let nested_edge = batch
            .mutations
            .iter()
            .find(|mutation| {
                mutation
                    .payload
                    .get("source_edge_ids")
                    .and_then(Value::as_array)
                    .is_some_and(|ids| ids.iter().any(|id| id == "persisted-edge"))
            })
            .expect("nested edge mutation");
        assert!(
            nested_edge.payload["source_edge_ids"]
                .as_array()
                .expect("edge endpoint list")
                .iter()
                .any(|id| id == "persisted-edge")
        );
    }
}

fn request_approval_syscall(
    request: &ApiEffectRequest,
    shared: &SharedCapabilityState,
    input: &SyscallInput,
) -> ApiEffectResponse {
    let action = input.args["action"]
        .as_str()
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    if action.is_empty() {
        return effect_error(
            StatusCode::BAD_REQUEST,
            "KOGWISTAR_INVALID_REQUEST",
            "action is required",
        );
    }
    if action == "deny" {
        return json_effect(
            StatusCode::OK,
            json!({
                "version": input.version,
                "op": input.op,
                "status": "blocked",
                "result": {},
                "error": {"reason": input.args["reason"].as_str().unwrap_or("request denied")},
            }),
        );
    }
    if !matches!(action.as_str(), "grant" | "revoke") {
        return json_effect(
            StatusCode::OK,
            json!({
                "version": input.version,
                "op": input.op,
                "status": "ok",
                "result": {"status":"requested", "reason": input.args["reason"].as_str().unwrap_or("approval pending")},
            }),
        );
    }
    let capability = input.args["capability"]
        .as_str()
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    if capability.is_empty() {
        return effect_error(
            StatusCode::BAD_REQUEST,
            "KOGWISTAR_INVALID_REQUEST",
            format!("capability is required for {action}"),
        );
    }
    let subject = input.args["subject"]
        .as_str()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_ascii_lowercase)
        .unwrap_or_else(|| principal_subject(&request.principal));
    let mut state = shared
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let payload = if action == "grant" {
        let approval_action = input.args["approval_action"]
            .as_str()
            .unwrap_or("request_approval")
            .to_ascii_lowercase();
        state
            .approvals
            .entry((subject.clone(), approval_action.clone()))
            .or_default()
            .insert(capability.clone());
        json!({
            "status":"approved",
            "approval": {"subject":subject,"action":approval_action,"capabilities":[capability]},
        })
    } else {
        state
            .revoked
            .entry(subject.clone())
            .or_default()
            .insert(capability.clone());
        json!({
            "status":"revoked",
            "revocation": {"subject":subject,"capability":capability},
        })
    };
    json_effect(
        StatusCode::OK,
        json!({"version":input.version,"op":input.op,"status":"ok","result":payload}),
    )
}

fn syscall_read_effect(
    request: &ApiEffectRequest,
    shared: &SharedCapabilityState,
) -> Option<ApiEffectResponse> {
    let path = request
        .path_and_query
        .split('?')
        .next()
        .unwrap_or(&request.path_and_query);
    match (request.method.as_str(), path) {
        ("GET", "/api/syscall/v1") => {
            let mut state = shared
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|value| value.as_millis() as u64)
                .unwrap_or_default();
            state.syscall_audit.push(json!({
                "ts_ms": now,
                "version": "v1",
                "op": "list_syscalls",
                "status": "ok",
            }));
            Some(json_effect(
                StatusCode::OK,
                json!({"version": "v1", "ops": SYSCALL_OPS}),
            ))
        }
        ("GET", "/api/syscall/v1/audit") => {
            let limit =
                query_integer(&request.path_and_query, "limit", 200).clamp(0, 10_000) as usize;
            let state = shared
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let keep_from = state.syscall_audit.len().saturating_sub(limit);
            Some(json_effect(
                StatusCode::OK,
                json!({"version": "v1", "events": &state.syscall_audit[keep_from..]}),
            ))
        }
        _ => None,
    }
}

// PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/document_graph.rs#designer_capabilities_effect; python=kogwistar/server_mcp_with_admin.py#designer_capabilities; fixture=contracts/golden/workflow-designer-capabilities-schema-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/part_03.rs#sqlite_visibility_and_capabilities_enforce_claims_across_rest_and_mcp; python_test=tests/server/test_chat_server_api.py#test_designer_capabilities_endpoint
fn designer_capabilities_effect(
    request: &ApiEffectRequest,
    shared: &SharedCapabilityState,
) -> Option<ApiEffectResponse> {
    if request.method != "GET"
        || request.path_and_query.split('?').next() != Some("/designer/capabilities")
    {
        return None;
    }
    let explicit = if request.principal.get("capabilities").is_some() {
        principal_strings(&request.principal, "capabilities")
    } else {
        principal_strings(&request.principal, "caps")
    };
    let namespaces = principal_strings(&request.principal, "ns");
    if !namespaces
        .iter()
        .any(|namespace| namespace == "workflow" || namespace == "*")
        || !explicit
            .iter()
            .any(|capability| capability == "workflow.design.inspect")
        || !capability_allowed(
            request,
            shared,
            "workflow.design.inspect",
            &["workflow.design.inspect"],
        )
    {
        return Some(effect_error(
            StatusCode::FORBIDDEN,
            "KOGWISTAR_CAPABILITY_FORBIDDEN",
            "Inspecting workflow design requires workflow.design.inspect capability",
        ));
    }
    Some(json_effect(
        StatusCode::OK,
        json!({
            "schema_version": "workflow-designer-capabilities/v1",
            "projection_schema": "workflow_design_v1",
            "design_features": {
                "undo_redo": true,
                "delta_history": true,
                "snapshot_restore": true,
                "dry_run_validation": false,
            },
            "custom_ops": {
                "allow_unregistered_ops_in_design": true,
                "allow_execution_of_unregistered_ops": false,
                "binding_statuses": ["resolved", "unresolved", "sandboxed", "plugin"],
            },
            "node_types": [{
                "type": "workflow_node",
                "display_name": "Workflow Node",
                "metadata_schema": {
                    "type": "object",
                    "title": "WorkflowNodeMetadata",
                    "description": "Workflow *design* node metadata (stored in workflow_engine).\n\nRequired keys:\n  - entity_type=\"workflow_node\"\n  - workflow_id\n  - wf_op (unless wf_terminal=True)\n\nOptional:\n  - wf_version: version of the wf_op implementation selected by the resolver;\n    the exact version is part of the frozen execution/replay contract\n  - wf_environment_id: required worker/runtime dependency environment\n  - wf_executor_target: optional per-node worker implementation; omitted\n    preserves legacy operation-prefix routing and Python-worker defaults\n  - wf_fanout: allow multiple outgoing edges to fire in parallel\n  - wf_start: marks the start node in this workflow_id\n  - wf_terminal: marks terminal node\n\nThis is distinct from ``WorkflowDesignArtifact.workflow_version``, which\nversions the workflow design as a whole and does not select step handlers.",
                    "additionalProperties": true,
                    "required": ["workflow_id"],
                    "properties": {
                        "entity_type": {"default": "workflow_node", "description": "Must be \"workflow_node\"", "title": "Entity Type", "type": "string"},
                        "wf_environment_id": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": null, "title": "Wf Environment Id"},
                        "wf_executor_target": {"anyOf": [{"enum": ["python-worker", "rust-native"], "type": "string"}, {"type": "null"}], "default": null, "title": "Wf Executor Target"},
                        "wf_fanout": {"default": false, "title": "Wf Fanout", "type": "boolean"},
                        "wf_join": {"default": false, "title": "Wf Join", "type": "boolean"},
                        "wf_op": {"default": "noop", "title": "Wf Op", "type": "string"},
                        "wf_start": {"default": false, "title": "Wf Start", "type": "boolean"},
                        "wf_terminal": {"default": false, "title": "Wf Terminal", "type": "boolean"},
                        "wf_version": {"default": "v1", "title": "Wf Version", "type": "string"},
                        "workflow_id": {"title": "Workflow Id", "type": "string"}
                    },
                },
                "flags": {
                    "supports_start": true,
                    "supports_terminal": true,
                    "supports_fanout": true,
                    "supports_join": true,
                },
            }],
            "edge_types": [{
                "type": "workflow_edge",
                "display_name": "Workflow Edge",
                "metadata_schema": {
                    "type": "object",
                    "title": "WorkflowEdgeMetadata",
                    "description": "Workflow *design* edge metadata (stored in workflow_engine).\n\nKeys:\n  - entity_type=\"workflow_edge\"\n  - workflow_id\n  - wf_predicate: symbolic predicate name (None means unconditional)\n  - wf_priority: lower first\n  - wf_is_default: used if no predicate matches\n  - wf_multiplicity: \"one\"|\"many\"  (allows fanout)",
                    "additionalProperties": true,
                    "required": ["workflow_id"],
                    "properties": {
                        "entity_type": {"default": "workflow_edge", "description": "Must be \"workflow_edge\"", "title": "Entity Type", "type": "string"},
                        "wf_is_default": {"default": false, "title": "Wf Is Default", "type": "boolean"},
                        "wf_multiplicity": {"default": "one", "enum": ["one", "many"], "title": "Wf Multiplicity", "type": "string"},
                        "wf_predicate": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": null, "title": "Wf Predicate"},
                        "wf_priority": {"default": 100, "title": "Wf Priority", "type": "integer"},
                        "workflow_id": {"title": "Workflow Id", "type": "string"}
                    },
                },
                "flags": {
                    "supports_predicate": true,
                    "supports_priority": true,
                    "supports_default": true,
                    "supports_multiplicity": true,
                },
            }],
            "runtime": {
                "resolver_found": true,
                "builtin_ops": ["llm_call", "start"],
                "nested_ops": [],
                "sandboxed_ops": [],
                "sandbox": {
                    "supports_sandboxed_ops": false,
                    "runtime_configured": false,
                },
                "state_schema": {},
            },
        }),
    ))
}

fn operational_path(path_and_query: &str) -> Option<&str> {
    let path = path_and_query.split('?').next().unwrap_or(path_and_query);
    matches!(
        path,
        "/api/workflow/operator/dashboard"
            | "/api/workflow/operator/inbox"
            | "/api/workflow/process-table"
            | "/api/workflow/blocked-runs"
            | "/api/workflow/lane/progress"
            | "/api/workflow/resources"
            | "/api/workflow/budget"
            | "/api/workflow/budget/history"
            | "/api/workflow/scheduler/timeline"
            | "/api/workflow/dead-letters"
            | "/api/workflow/catalog/ops"
    )
    .then_some(path)
}

fn operator_inbox_effect(
    request: &ApiEffectRequest,
    lanes: &[ProjectedLaneMessage],
    visible_nodes: &[GraphRecord],
) -> ApiEffectResponse {
    let query_value = |name: &str| {
        let query = request
            .path_and_query
            .split_once('?')
            .map(|(_, query)| query)
            .unwrap_or_default();
        url::form_urlencoded::parse(query.as_bytes())
            .find_map(|(key, value)| (key == name && !value.is_empty()).then(|| value.into_owned()))
    };
    let inbox_id = query_value("inbox_id");
    let status = query_value("status");
    let limit = query_integer(&request.path_and_query, "limit", 200).clamp(1, 10_000) as usize;
    let scope = security_scope_value(&request.principal);
    let actual_scope = scope["security_scope"]
        .as_str()
        .unwrap_or_default()
        .to_ascii_lowercase();
    let visible = visible_nodes
        .iter()
        .map(|record| (record.id.as_str(), &record.metadata))
        .collect::<std::collections::HashMap<_, _>>();
    let mut messages = Vec::new();
    for lane in lanes {
        if inbox_id.as_deref().is_some_and(|value| value != lane.inbox_id)
            || status.as_deref().is_some_and(|value| value != lane.status)
        {
            continue;
        }
        let Some(metadata) = visible.get(lane.message_id.as_str()) else {
            continue;
        };
        let message_scope = metadata
            .get("security_scope")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let normalized_message_scope = message_scope.trim().to_ascii_lowercase();
        let shared = metadata.get("shared_scope").and_then(Value::as_bool) == Some(true)
            || metadata.get("shared_inbox").and_then(Value::as_bool) == Some(true);
        if !shared
            && !normalized_message_scope.is_empty()
            && normalized_message_scope != "*"
            && normalized_message_scope != actual_scope
        {
            continue;
        }
        messages.push(json!({
            "message_id": lane.message_id,
            "inbox_id": lane.inbox_id,
            "conversation_id": lane.conversation_id,
            "recipient_id": lane.recipient_id,
            "sender_id": lane.sender_id,
            "msg_type": lane.msg_type,
            "status": lane.status,
            "seq": lane.seq,
            "conversation_seq": lane.conversation_seq,
            "claimed_by": lane.claimed_by,
            "lease_until": lane.lease_until,
            "retry_count": lane.retry_count,
            "created_at": lane.created_at,
            "available_at": lane.available_at,
            "run_id": lane.run_id,
            "step_id": lane.step_id,
            "correlation_id": lane.correlation_id,
            "message_security_scope": message_scope,
            "visibility": metadata.get("visibility").and_then(Value::as_str).unwrap_or("private"),
            "storage_namespace": scope["storage_namespace"],
            "execution_namespace": scope["execution_namespace"],
            "security_scope": scope["security_scope"],
        }));
        if messages.len() >= limit {
            break;
        }
    }
    json_effect(StatusCode::OK, json!({"messages": messages}))
}

fn operator_inbox_authorization_error(
    request: &ApiEffectRequest,
    capabilities: &SharedCapabilityState,
) -> Option<ApiEffectResponse> {
    if !matches!(principal_role(&request.principal).as_str(), "ro" | "rw") {
        return Some(effect_error(
            StatusCode::FORBIDDEN,
            "KOGWISTAR_ROLE_FORBIDDEN",
            "Operator inbox reads require a read-capable workflow role",
        ));
    }
    if !principal_strings(&request.principal, "ns")
        .iter()
        .any(|namespace| namespace == "workflow" || namespace == "*")
    {
        return Some(effect_error(
            StatusCode::FORBIDDEN,
            "KOGWISTAR_NAMESPACE_FORBIDDEN",
            "Operator inbox reads require workflow namespace",
        ));
    }
    (!capability_allowed(
        request,
        capabilities,
        "project_view",
        &["project_view"],
    ))
    .then(|| {
        effect_error(
            StatusCode::FORBIDDEN,
            "KOGWISTAR_CAPABILITY_FORBIDDEN",
            "Reading operator inbox requires project_view capability",
        )
    })
}

fn lane_progress_value(lane: &ProjectedLaneMessage) -> Value {
    json!({
        "event_type": format!("worker.{}", lane.status),
        "message_id": lane.message_id,
        "conversation_id": lane.conversation_id,
        "inbox_id": lane.inbox_id,
        "recipient_id": lane.recipient_id,
        "sender_id": lane.sender_id,
        "status": lane.status,
        "msg_type": lane.msg_type,
        "seq": lane.seq,
        "conversation_seq": lane.conversation_seq,
        "claimed_by": lane.claimed_by,
        "retry_count": lane.retry_count,
        "run_id": lane.run_id,
        "step_id": lane.step_id,
        "correlation_id": lane.correlation_id,
        "created_at": lane.created_at,
        "available_at": lane.available_at,
    })
}

fn status_counts<'a>(values: impl Iterator<Item = &'a str>) -> serde_json::Map<String, Value> {
    let mut counts = serde_json::Map::new();
    for status in values {
        let count = counts.get(status).and_then(Value::as_u64).unwrap_or(0) + 1;
        counts.insert(status.to_owned(), json!(count));
    }
    counts
}

fn message_queue_snapshot_value(lanes: &[ProjectedLaneMessage]) -> Value {
    let mut by_inbox = serde_json::Map::new();
    for lane in lanes {
        let count = by_inbox
            .get(&lane.inbox_id)
            .and_then(Value::as_u64)
            .unwrap_or(0)
            + 1;
        by_inbox.insert(lane.inbox_id.clone(), json!(count));
    }
    let failed = lanes
        .iter()
        .filter(|lane| lane.status == "failed")
        .take(20)
        .map(|lane| {
            json!({
                "message_id": lane.message_id,
                "inbox_id": lane.inbox_id,
                "conversation_id": lane.conversation_id,
                "correlation_id": lane.correlation_id,
            })
        })
        .collect::<Vec<_>>();
    json!({
        "total": lanes.len(),
        "by_status": status_counts(lanes.iter().map(|lane| lane.status.as_str())),
        "by_inbox": by_inbox,
        "failed": failed,
    })
}

fn usage_history(events: &[ServerRunEvent], limit: usize) -> Vec<Value> {
    let mut values = events
        .iter()
        .filter(|event| event.event_type == "workflow.usage.v1")
        .map(|event| {
            let payload =
                serde_json::from_str::<Value>(&event.payload_json).unwrap_or_else(|_| json!({}));
            json!({
                "workspace_id": "workflow",
                "kind": "usage",
                "amount": payload["usage"]["total_tokens"].as_i64().unwrap_or_else(|| {
                    payload["usage"]["input_tokens"].as_i64().unwrap_or_default()
                        + payload["usage"]["output_tokens"].as_i64().unwrap_or_default()
                }),
                "source": payload["effect_id"],
                "run_id": event.run_id,
                "event_seq": event.seq,
                "created_at_ms": event.created_at_ms,
                "usage": payload["usage"],
            })
        })
        .collect::<Vec<_>>();
    let keep_from = values.len().saturating_sub(limit);
    values.drain(0..keep_from);
    values
}

pub(crate) fn persisted_directory_bytes(paths: &[std::path::PathBuf]) -> u64 {
    fn directory_bytes(root: &std::path::Path) -> u64 {
        let mut pending = vec![root.to_path_buf()];
        let mut total = 0_u64;
        while let Some(directory) = pending.pop() {
            let Ok(entries) = std::fs::read_dir(directory) else {
                continue;
            };
            for entry in entries.flatten() {
                let Ok(file_type) = entry.file_type() else {
                    continue;
                };
                if file_type.is_dir() {
                    pending.push(entry.path());
                } else if (file_type.is_file()
                    || (file_type.is_symlink()
                        && entry.metadata().is_ok_and(|metadata| metadata.is_file())))
                    && let Ok(metadata) = entry.metadata()
                {
                    total = total.saturating_add(metadata.len());
                }
            }
        }
        total
    }

    paths.iter().fold(0_u64, |total, path| {
        total.saturating_add(directory_bytes(path))
    })
}

pub(crate) fn configured_storage_usage_bytes() -> u64 {
    let configured =
        |name: &str, default: &str| std::env::var(name).unwrap_or_else(|_| default.to_owned());
    let directories = [
        configured("KOGWISTAR_PERSIST_DIRECTORY", ".kogwistar"),
        configured(
            "KOGWISTAR_CONVERSATION_PERSIST_DIRECTORY",
            ".kogwistar/conversation",
        ),
        configured(
            "KOGWISTAR_WORKFLOW_PERSIST_DIRECTORY",
            ".kogwistar/workflow",
        ),
    ]
    .into_iter()
    .map(std::path::PathBuf::from)
    .collect::<Vec<_>>();
    persisted_directory_bytes(&directories)
}

fn resource_value(
    runs: &[ServerRun],
    lanes: &[ProjectedLaneMessage],
    usage: &[Value],
    events: &[ServerRunEvent],
    storage_usage_bytes: u64,
) -> Value {
    let run_counts = status_counts(runs.iter().map(|run| run.status.as_str()));
    let lane_counts = status_counts(lanes.iter().map(|lane| lane.status.as_str()));
    let active = lanes.iter().filter(|lane| lane.status == "claimed").count();
    let queued = lanes.iter().filter(|lane| lane.status == "pending").count();
    let total_amount = usage
        .iter()
        .filter_map(|event| event["amount"].as_i64())
        .sum::<i64>();
    let now_seconds = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|value| value.as_secs() as i64)
        .unwrap_or_default();
    let oldest_pending_seconds = lanes
        .iter()
        .filter(|lane| lane.status == "pending")
        .map(|lane| lane.available_at)
        .min();
    let parity_mismatches = events
        .iter()
        .filter(|event| {
            matches!(
                event.event_type.as_str(),
                "rust.parity_mismatch" | "workflow.parity_mismatch.v1"
            )
        })
        .count();
    json!({
        "scheduler": {
            "max_active": active.max(1),
            "max_queue": Value::Null,
            "active": active,
            "queued": queued,
            "active_by_class": {},
            "queued_by_class": {},
            "dead_letter_count": lane_counts.get("dead-letter").and_then(Value::as_u64).unwrap_or(0),
            "paused_count": runs.iter().filter(|run| run.status == "suspended").count(),
            "pause_requested_count": 0,
        },
        "runs": {
            "total_runs": runs.len(),
            "by_status": run_counts,
            "terminal_runs": runs.iter().filter(|run| matches!(run.status.as_str(), "succeeded" | "failed" | "cancelled")).count(),
        },
        "services": {"total_services": 0, "by_health": {}},
        "storage_usage_bytes": storage_usage_bytes,
        "cost_ledger": {
            "workspace_id": "workflow",
            "event_count": usage.len(),
            "total_amount": total_amount,
            "by_kind": {"usage": usage.len()},
        },
        "budget_model": {"token_kind": "token", "time_kind": "ms", "storage_kind": "bytes"},
        "policy_infra": {"cpu_millicores": Value::Null, "memory_mb": Value::Null},
        "migration": {
            "implementation_mode": "rust",
            "contract_version": 1,
            "schema_version": 1,
            "parity_mismatch_count": parity_mismatches,
            "queue_lag": {
                "pending_count": queued,
                "oldest_pending_age_seconds": oldest_pending_seconds
                    .map(|created| now_seconds.saturating_sub(created)),
            },
            "replay_lag": {
                "events_behind": 0,
                "mode": "transactional_projection",
            },
        },
    })
}

#[cfg(test)]
mod storage_usage_tests {
    use super::{persisted_directory_bytes, resource_value};

    #[test]
    fn persisted_directory_bytes_sums_files_and_ignores_missing_paths() {
        let root =
            std::env::temp_dir().join(format!("kogwistar-storage-usage-{}", uuid::Uuid::new_v4()));
        let nested = root.join("nested");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(root.join("one.bin"), [1_u8; 7]).unwrap();
        std::fs::write(nested.join("two.bin"), [2_u8; 11]).unwrap();

        let bytes = persisted_directory_bytes(&[root.clone(), root.join("missing")]);

        assert_eq!(bytes, 18);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn resource_snapshot_keeps_storage_usage_as_an_integer() {
        let snapshot = resource_value(&[], &[], &[], &[], 18);
        assert_eq!(snapshot["storage_usage_bytes"], 18);
    }
}

fn resource_service_value(services: &[Value]) -> Value {
    let mut by_health = BTreeMap::<String, usize>::new();
    for service in services {
        let health = service
            .get("health_status")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .unwrap_or("unknown");
        *by_health.entry(health.to_owned()).or_default() += 1;
    }
    json!({"total_services": services.len(), "by_health": by_health})
}

fn process_table_effect(
    request: &ApiEffectRequest,
    runs: Vec<ServerRun>,
    mut services: Vec<Value>,
    run_events: Vec<ServerRunEvent>,
    service_event_rows: Vec<(i64, String, String, String, String)>,
) -> ApiEffectResponse {
    let limit = query_integer(&request.path_and_query, "limit", 100).clamp(1, 10_000) as usize;
    let query_value = |name: &str| {
        let query = request
            .path_and_query
            .split_once('?')
            .map(|(_, query)| query)
            .unwrap_or_default();
        url::form_urlencoded::parse(query.as_bytes())
            .find_map(|(key, value)| (key == name && !value.is_empty()).then(|| value.into_owned()))
    };
    let status_filter = query_value("status");
    let workflow_filter = query_value("workflow_id");
    let conversation_filter = query_value("conversation_id");
    let scope = security_scope_value(&request.principal);

    let mut processes = runs
        .into_iter()
        .filter(|run| status_filter.as_deref().is_none_or(|status| run.status == status))
        .filter(|run| workflow_filter.as_deref().is_none_or(|workflow| run.workflow_id == workflow))
        .filter(|run| {
            conversation_filter
                .as_deref()
                .is_none_or(|conversation| run.conversation_id == conversation)
        })
        .take(limit)
        .map(|run| {
            let related_events = run_events
                .iter()
                .filter(|event| event.run_id == run.run_id)
                .take(50)
                .collect::<Vec<_>>();
            let last_event = related_events.last();
            json!({
                "process_id": run.run_id,
                "process_kind": "workflow_run",
                "status": run.status,
                "workflow_id": run.workflow_id,
                "conversation_id": run.conversation_id,
                "user_id": run.user_id,
                "user_turn_node_id": run.user_turn_node_id,
                "assistant_turn_node_id": run.assistant_turn_node_id,
                "created_at_ms": run.created_at_ms,
                "updated_at_ms": run.updated_at_ms,
                "started_at_ms": run.started_at_ms,
                "finished_at_ms": run.finished_at_ms,
                "terminal": matches!(run.status.as_str(), "succeeded" | "failed" | "cancelled"),
                "event_count": related_events.len(),
                "last_event_type": last_event.map(|event| event.event_type.as_str()),
                "last_event_seq": last_event.map(|event| event.seq),
                "owner": run.user_id,
                "namespace": scope["execution_namespace"],
                "storage_namespace": scope["storage_namespace"],
                "security_scope": scope["security_scope"],
            })
        })
        .collect::<Vec<_>>();

    services.sort_by(|left, right| left["service_id"].as_str().cmp(&right["service_id"].as_str()));
    services.truncate(limit);
    let mut current_service_events = BTreeMap::<String, String>::new();
    for (_seq, entity_kind, entity_id, op, payload_json) in service_event_rows {
        if entity_kind != "node" {
            continue;
        }
        match op.to_ascii_uppercase().as_str() {
            "DELETE" | "TOMBSTONE" | "REMOVE" => {
                current_service_events.remove(&entity_id);
                continue;
            }
            "ADD" | "REPLACE" | "UPSERT" => {}
            _ => continue,
        }
        let Ok(payload) = serde_json::from_str::<Value>(&payload_json) else {
            continue;
        };
        let Some(metadata) = payload.get("metadata").and_then(Value::as_object) else {
            continue;
        };
        if metadata.get("entity_type").and_then(Value::as_str) == Some("service_event")
            && let Some(service_id) = metadata.get("service_id").and_then(Value::as_str)
        {
            current_service_events.insert(entity_id, service_id.to_owned());
        }
    }
    let mut service_event_counts = BTreeMap::<String, usize>::new();
    for service_id in current_service_events.into_values() {
        *service_event_counts.entry(service_id).or_default() += 1;
    }

    for service in services {
        let service_id = service["service_id"].as_str().unwrap_or_default();
        let service_status = service["lifecycle_status"].as_str().unwrap_or_default();
        let target_ref = service["target_ref"].as_str().unwrap_or_default();
        let target_config = service.get("target_config").and_then(Value::as_object);
        let service_conversation = target_config
            .and_then(|config| config.get("conversation_id"))
            .and_then(Value::as_str);
        if status_filter.as_deref().is_some_and(|status| status != service_status)
            || workflow_filter
                .as_deref()
                .is_some_and(|workflow| workflow != target_ref)
            || conversation_filter
                .as_deref()
                .is_some_and(|conversation| Some(conversation) != service_conversation)
        {
            continue;
        }
        processes.push(json!({
            "process_id": service_id,
            "process_kind": "service",
            "status": service_status,
            "workflow_id": if service["target_kind"] == "workflow" { service.get("target_ref") } else { None },
            "conversation_id": service_conversation,
            "user_id": Value::Null,
            "user_turn_node_id": Value::Null,
            "assistant_turn_node_id": Value::Null,
            "created_at_ms": service.get("created_at_ms"),
            "updated_at_ms": service.get("updated_at_ms"),
            "started_at_ms": service.get("current_child_started_at_ms"),
            "finished_at_ms": Value::Null,
            "terminal": !service["enabled"].as_bool().unwrap_or(false),
            "event_count": service_event_counts.get(service_id).copied().unwrap_or_default().min(200),
            "last_event_type": Value::Null,
            "last_event_seq": Value::Null,
            "owner": Value::Null,
            "namespace": service.get("execution_namespace"),
            "storage_namespace": service.get("storage_namespace"),
            "security_scope": service.get("security_scope"),
            "service_kind": service.get("service_kind"),
            "health_status": service.get("health_status"),
            "last_heartbeat_ms": service.get("last_heartbeat_ms"),
            "restart_count": service.get("restart_count"),
            "last_trigger_type": service.get("last_trigger_type"),
            "current_child_run_id": service.get("current_child_run_id"),
            "current_child_status": service.get("current_child_status"),
        }));
    }
    processes.sort_by(|left, right| {
        let left_key = (
            left["updated_at_ms"].as_i64().unwrap_or_default(),
            left["process_kind"].as_str().unwrap_or_default(),
            left["process_id"].as_str().unwrap_or_default(),
        );
        let right_key = (
            right["updated_at_ms"].as_i64().unwrap_or_default(),
            right["process_kind"].as_str().unwrap_or_default(),
            right["process_id"].as_str().unwrap_or_default(),
        );
        right_key.cmp(&left_key)
    });
    if request
        .path_and_query
        .split('?')
        .next()
        == Some("/api/workflow/blocked-runs")
    {
        processes.retain(|process| {
            matches!(
                process["status"].as_str(),
                Some("blocked" | "waiting" | "suspended" | "cancelling")
            )
        });
    }
    json_effect(StatusCode::OK, json!({"processes": processes}))
}

// PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/document_graph.rs#operational_effect; python=kogwistar/server/chat_service.py#ChatRunService.operator_dashboard; fixture=contracts/golden/operator-dashboard-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/part_07.rs#sqlite_operator_dashboard_requires_project_view_and_separates_blocked_runs_from_inbox; python_test=tests/server/test_service_daemon_model.py#test_service_operator_dashboard_round_trip
fn operational_effect(
    request: &ApiEffectRequest,
    path: &str,
    runs: Vec<ServerRun>,
    lanes: Vec<ProjectedLaneMessage>,
    events: Vec<ServerRunEvent>,
    services: Vec<Value>,
    service_event_rows: Vec<(i64, String, String, String, String)>,
) -> ApiEffectResponse {
    let limit = query_integer(&request.path_and_query, "limit", 200).clamp(0, 10_000) as usize;
    let usage = usage_history(&events, limit.max(1));
    let storage_usage_bytes = if matches!(
        path,
        "/api/workflow/resources" | "/api/workflow/operator/dashboard"
    ) {
        configured_storage_usage_bytes()
    } else {
        0
    };
    let mut resources = resource_value(&runs, &lanes, &usage, &events, storage_usage_bytes);
    resources["services"] = resource_service_value(&services);
    match path {
        "/api/workflow/process-table" | "/api/workflow/blocked-runs" => process_table_effect(
            request,
            runs,
            services,
            events,
            service_event_rows,
        ),
        "/api/workflow/catalog/ops" => json_effect(
            StatusCode::OK,
            json!([
                {
                    "op": "start",
                    "label": "Start",
                    "description": "Entry point for workflow execution.",
                    "input_schema": {},
                    "output_schema": {"type": "object"},
                    "config_schema": {"type": "object"},
                },
                {
                    "op": "llm_call",
                    "label": "LLM Call",
                    "description": "Calls an LLM and returns structured output.",
                    "input_schema": {
                        "type": "object",
                        "properties": {"prompt": {"type": "string"}},
                        "required": ["prompt"],
                    },
                    "output_schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                    },
                    "config_schema": {
                        "type": "object",
                        "properties": {
                            "model": {"type": "string"},
                            "temperature": {"type": "number"},
                        },
                    },
                },
            ]),
        ),
        // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/document_graph.rs#operational_effect; python=kogwistar/server/chat_service.py#ChatRunService.resource_snapshot; fixture=contracts/golden/workflow-resources-http-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/part_07.rs#sqlite_resources_route_matches_python_contract_and_requires_scope_capabilities; python_test=tests/server/test_chat_server_api.py#test_runtime_resource_snapshot_reports_pressure
        "/api/workflow/resources" => json_effect(StatusCode::OK, resources),
        "/api/workflow/budget" => json_effect(
            StatusCode::OK,
            json!({
                "cost_ledger": resources["cost_ledger"],
                "budget_model": resources["budget_model"],
            }),
        ),
        "/api/workflow/budget/history" => json_effect(
            StatusCode::OK,
            json!({"cost_ledger": resources["cost_ledger"], "events": usage}),
        ),
        "/api/workflow/scheduler/timeline" => {
            let run_id = query_string(&request.path_and_query, "run_id");
            let mut timeline = events
                .iter()
                .filter(|event| run_id.as_ref().is_none_or(|id| id == &event.run_id))
                .cloned()
                .map(server_run_event_value)
                .collect::<Vec<_>>();
            let keep_from = timeline.len().saturating_sub(limit);
            timeline.drain(0..keep_from);
            json_effect(
                StatusCode::OK,
                json!({"run_id": run_id, "events": timeline}),
            )
        }
        "/api/workflow/dead-letters" => json_effect(
            StatusCode::OK,
            json!({
                "runs": lanes.iter()
                    .filter(|lane| lane.status == "dead-letter")
                    .take(limit)
                    .map(lane_progress_value)
                    .collect::<Vec<_>>(),
                "limit": limit,
            }),
        ),
        // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/document_graph.rs#operational_effect; python=kogwistar/server/runtime_api.py#create_runtime_router.get_lane_message_progress; python_service=kogwistar/server/chat_service.py#ChatRunService.lane_message_progress; fixture=contracts/golden/workflow-lane-progress-http-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/part_05.rs#sqlite_run_service_handles_get_events_and_cancel; python_test=tests/server/test_chat_server_api.py#test_workflow_lane_progress_matches_shared_contract
        "/api/workflow/lane/progress" => {
            let run_id = query_string(&request.path_and_query, "run_id");
            let conversation_id = query_string(&request.path_and_query, "conversation_id");
            let mut items = events
                .iter()
                .filter(|event| run_id.as_ref().is_some_and(|id| id == &event.run_id))
                .map(|event| {
                    let mut value = server_run_event_value(event.clone());
                    if let Some(object) = value.as_object_mut() {
                        object.insert("event_type".to_owned(), json!(event.event_type));
                    }
                    value
                })
                .collect::<Vec<_>>();
            if conversation_id.is_some() {
                items.extend(
                    lanes
                        .iter()
                        .filter(|lane| {
                            run_id
                                .as_ref()
                                .is_none_or(|id| lane.run_id.as_ref() == Some(id))
                                && conversation_id
                                    .as_ref()
                                    .is_none_or(|id| &lane.conversation_id == id)
                        })
                        .map(lane_progress_value),
                );
            }
            items.truncate(limit);
            json_effect(
                StatusCode::OK,
                json!({"total": items.len(), "items": items}),
            )
        }
        "/api/workflow/operator/dashboard" => {
            let process_table_response = process_table_effect(
                request,
                runs.clone(),
                services.clone(),
                events.clone(),
                service_event_rows.clone(),
            );
            let process_table_payload = serde_json::from_slice::<Value>(&process_table_response.body)
                .unwrap_or_else(|_| json!({}));
            let process_table = process_table_payload["processes"]
                .as_array()
                .cloned()
                .unwrap_or_default();
            let blocked_runs = process_table
                .iter()
                .filter(|process| {
                    matches!(
                        process["status"].as_str(),
                        Some("blocked" | "waiting" | "suspended" | "cancelling")
                    )
                })
                .take(limit)
                .cloned()
                .collect::<Vec<_>>();
            let blocked_flow_nodes = blocked_runs
                .iter()
                .map(|process| {
                    json!({
                        "id": process["process_id"],
                        "kind": process["process_kind"],
                        "status": process["status"],
                        "workflow_id": process["workflow_id"],
                        "conversation_id": process["conversation_id"],
                    })
                })
                .collect::<Vec<_>>();
            let blocked_count = blocked_flow_nodes.len();
            let dead_letters = lanes
                .iter()
                .filter(|lane| lane.status == "dead-letter")
                .take(limit)
                .map(lane_progress_value)
                .collect::<Vec<_>>();
            let message_queue = message_queue_snapshot_value(&lanes);
            json_effect(
                StatusCode::OK,
                json!({
                    "process_table": process_table,
                    "operator_inbox": [],
                    "blocked_runs": blocked_runs,
                    "blocked_flow_graph": {"nodes": blocked_flow_nodes, "edges": [], "blocked_count": blocked_count},
                    "message_queue": message_queue,
                    "dead_letters": {"runs": dead_letters, "limit": limit},
                    "capabilities": {},
                    "resources": resources,
                }),
            )
        }
        _ => unreachable!("unsupported operational path"),
    }
}

fn operator_dashboard_with_inbox(
    mut dashboard: ApiEffectResponse,
    messages: Vec<Value>,
) -> ApiEffectResponse {
    let Ok(mut value) = serde_json::from_slice::<Value>(&dashboard.body) else {
        return dashboard;
    };
    let Some(object) = value.as_object_mut() else {
        return dashboard;
    };
    object.insert("operator_inbox".to_owned(), json!(messages));
    dashboard.body = serde_json::to_vec(&value).expect("JSON value always serializes");
    dashboard
}

fn operator_dashboard_with_capabilities(
    mut dashboard: ApiEffectResponse,
    capabilities: Value,
) -> ApiEffectResponse {
    let Ok(mut value) = serde_json::from_slice::<Value>(&dashboard.body) else {
        return dashboard;
    };
    let Some(object) = value.as_object_mut() else {
        return dashboard;
    };
    object.insert("capabilities".to_owned(), capabilities);
    dashboard.body = serde_json::to_vec(&value).expect("JSON value always serializes");
    dashboard
}

fn runtime_run_route(path_and_query: &str) -> Option<(String, Vec<String>)> {
    let path = path_and_query.split('?').next().unwrap_or(path_and_query);
    for prefix in ["/api/runs/", "/api/workflow/runs/"] {
        if let Some(rest) = path.strip_prefix(prefix) {
            let segments = rest.split('/').collect::<Vec<_>>();
            if let Some((run_id, tail)) = segments.split_first()
                && !run_id.is_empty()
            {
                let run_id = percent_encoding::percent_decode_str(run_id)
                    .decode_utf8()
                    .ok()?
                    .into_owned();
                return Some((
                    run_id,
                    tail.iter().map(|value| (*value).to_owned()).collect(),
                ));
            }
        }
    }
    None
}

fn runtime_run_namespace(path_and_query: &str) -> Option<&'static str> {
    let path = path_and_query.split('?').next().unwrap_or(path_and_query);
    if path.starts_with("/api/workflow/runs/") {
        Some("workflow")
    } else if path.starts_with("/api/runs/") {
        Some("conversation")
    } else {
        None
    }
}

fn dead_letter_replay_run_id(path_and_query: &str) -> Option<String> {
    let path = path_and_query.split('?').next().unwrap_or(path_and_query);
    let rest = path.strip_prefix("/api/workflow/dead-letters/")?;
    let run_id = rest.strip_suffix("/replay")?;
    (!run_id.is_empty() && !run_id.contains('/')).then(|| run_id.to_owned())
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum ServiceRoute {
    Get,
    Events,
    Repair,
}

fn service_route(path_and_query: &str) -> Option<(String, ServiceRoute)> {
    let path = path_and_query.split('?').next().unwrap_or(path_and_query);
    let rest = path.strip_prefix("/api/workflow/services/")?;
    let mut segments = rest.split('/');
    let service_id = segments.next()?;
    if service_id.is_empty() {
        return None;
    }
    let route = match segments.collect::<Vec<_>>().as_slice() {
        [] => ServiceRoute::Get,
        ["events"] => ServiceRoute::Events,
        ["repair"] => ServiceRoute::Repair,
        _ => return None,
    };
    Some((service_id.to_owned(), route))
}

fn service_repair_route(path_and_query: &str) -> Option<Option<String>> {
    let path = path_and_query.split('?').next().unwrap_or(path_and_query);
    if path == "/api/workflow/services/repair" {
        return Some(None);
    }
    let (service_id, route) = service_route(path_and_query)?;
    (route == ServiceRoute::Repair).then_some(Some(service_id))
}

fn service_enabled_route(path_and_query: &str) -> Option<(String, bool)> {
    let path = path_and_query.split('?').next().unwrap_or(path_and_query);
    let rest = path.strip_prefix("/api/workflow/services/")?;
    let (service_id, action) = rest.split_once('/')?;
    if service_id.is_empty() || action.contains('/') {
        return None;
    }
    match action {
        "enable" => Some((service_id.to_owned(), true)),
        "disable" => Some((service_id.to_owned(), false)),
        _ => None,
    }
}

fn latest_service_definition(
    events: impl IntoIterator<Item = (i64, String, String, String, String)>,
    service_id: &str,
) -> Option<Value> {
    let mut current = BTreeMap::<String, (i64, Value)>::new();
    for (seq, entity_kind, entity_id, op, payload_json) in events {
        if entity_kind != "node" {
            continue;
        }
        match op.to_ascii_uppercase().as_str() {
            "DELETE" | "TOMBSTONE" | "REMOVE" => {
                current.remove(&entity_id);
                continue;
            }
            "ADD" | "REPLACE" | "UPSERT" => {}
            _ => continue,
        }
        let Ok(payload) = serde_json::from_str::<Value>(&payload_json) else {
            continue;
        };
        if payload.is_object() {
            current.insert(entity_id, (seq, payload));
        }
    }
    current
        .into_values()
        .filter(|(_, payload)| {
            payload["metadata"]["entity_type"].as_str() == Some("service_definition")
                && payload["metadata"]["service_id"].as_str() == Some(service_id)
        })
        .max_by_key(|(seq, payload)| {
            (
                payload["metadata"]["updated_at_ms"]
                    .as_i64()
                    .or_else(|| payload["properties"]["updated_at_ms"].as_i64())
                    .unwrap_or_default(),
                *seq,
            )
        })
        .map(|(_, payload)| payload)
}

fn service_enabled_nodes(
    mut definition: Value,
    service_id: &str,
    enabled: bool,
    timestamp: u64,
) -> Result<(String, Value, String, Value), String> {
    let definition_id = format!(
        "service_def:{service_id}:{timestamp}:{}",
        uuid::Uuid::new_v4().simple()
    );
    let definition_object = definition
        .as_object_mut()
        .ok_or_else(|| "service definition must be an object".to_owned())?;
    definition_object.insert("id".to_owned(), json!(definition_id));
    let metadata = definition_object
        .get_mut("metadata")
        .and_then(Value::as_object_mut)
        .ok_or_else(|| "service definition metadata must be an object".to_owned())?;
    metadata.insert("enabled".to_owned(), json!(enabled));
    metadata.insert("updated_at_ms".to_owned(), json!(timestamp));
    let properties = definition_object
        .get_mut("properties")
        .and_then(Value::as_object_mut)
        .ok_or_else(|| "service definition properties must be an object".to_owned())?;
    properties.insert("enabled".to_owned(), json!(enabled));
    properties.insert("updated_at_ms".to_owned(), json!(timestamp));

    let event_type = if enabled {
        "service.enabled"
    } else {
        "service.stopped"
    };
    let event_id = format!(
        "service_evt:{service_id}:{timestamp}:{}",
        uuid::Uuid::new_v4().simple()
    );
    let event = json!({
        "id": event_id,
        "label": format!("service_event:{event_type}"),
        "type": "entity",
        "summary": format!("Service event {event_type} for {service_id}"),
        "mentions": [{"spans": [{
            "collection_page_url": format!("workflow/_wf:{service_id}"),
            "context_after": "",
            "context_before": "",
            "doc_id": format!("_wf:{service_id}"),
            "document_page_url": format!("workflow/_wf:{service_id}"),
            "end_char": 1,
            "excerpt": "",
            "page_number": 1,
            "source_cluster_id": Value::Null,
            "start_char": 0,
        }]}],
        "properties": {
            "payload_json": json!({"enabled": enabled}).to_string(),
            "enabled": enabled,
        },
        "metadata": {
            "entity_type": "service_event",
            "artifact_kind": "service_event",
            "service_id": service_id,
            "service_event_type": event_type,
            "ts_ms": timestamp,
            "in_conversation_chain": false,
            "lifecycle_status": "active",
            "redirect_to_id": Value::Null,
        },
    });
    Ok((definition_id, definition, event_id, event))
}

fn service_event_node(
    service_id: &str,
    event_type: &str,
    payload: &Value,
    timestamp: u64,
) -> (String, Value) {
    let event_id = format!(
        "service_evt:{service_id}:{timestamp}:{}",
        uuid::Uuid::new_v4().simple()
    );
    (
        event_id.clone(),
        json!({
            "id": event_id,
            "label": format!("service_event:{event_type}"),
            "type": "entity",
            "summary": format!("Service event {event_type} for {service_id}"),
            "mentions": [{"spans": [{
                "collection_page_url": format!("workflow/_wf:{service_id}"),
                "context_after": "",
                "context_before": "",
                "doc_id": format!("_wf:{service_id}"),
                "document_page_url": format!("workflow/_wf:{service_id}"),
                "end_char": 1,
                "excerpt": "",
                "page_number": 1,
                "source_cluster_id": Value::Null,
                "start_char": 0,
            }]}],
            "properties": {"payload_json": payload.to_string()},
            "metadata": {
                "entity_type": "service_event",
                "artifact_kind": "service_event",
                "service_id": service_id,
                "service_event_type": event_type,
                "ts_ms": timestamp,
                "in_conversation_chain": false,
                "lifecycle_status": "active",
                "redirect_to_id": Value::Null,
            },
        }),
    )
}

#[derive(Clone, Debug, Deserialize)]
struct ServiceDeclareInput {
    service_id: String,
    #[serde(default = "default_service_kind")]
    service_kind: String,
    #[serde(default = "default_service_target_kind")]
    target_kind: String,
    target_ref: String,
    #[serde(default)]
    target_config: serde_json::Map<String, Value>,
    #[serde(default = "default_true")]
    enabled: bool,
    #[serde(default)]
    autostart: bool,
    #[serde(default)]
    restart_policy: serde_json::Map<String, Value>,
    #[serde(default = "default_heartbeat_ttl_ms")]
    heartbeat_ttl_ms: i64,
    #[serde(default)]
    trigger_specs: Vec<Value>,
}

fn default_service_kind() -> String {
    "daemon".to_owned()
}

fn default_service_target_kind() -> String {
    "workflow".to_owned()
}

fn default_heartbeat_ttl_ms() -> i64 {
    60_000
}

#[derive(Clone, Debug, Deserialize)]
struct ServiceTriggerInput {
    trigger_type: String,
    #[serde(default)]
    payload: serde_json::Map<String, Value>,
}

fn service_trigger_route(path_and_query: &str) -> Option<String> {
    let path = path_and_query.split('?').next().unwrap_or(path_and_query);
    let rest = path.strip_prefix("/api/workflow/services/")?;
    let (service_id, action) = rest.split_once('/')?;
    (!service_id.is_empty() && action == "trigger").then(|| service_id.to_owned())
}

fn validate_service_trigger(input: &mut ServiceTriggerInput) -> Result<(), &'static str> {
    input.trigger_type = input.trigger_type.trim().to_ascii_lowercase();
    if matches!(
        input.trigger_type.as_str(),
        "schedule"
            | "message arrival"
            | "graph change"
            | "external event"
            | "autostart"
            | "restart"
    ) {
        Ok(())
    } else {
        Err("unsupported trigger_type")
    }
}

fn service_definition_runtime(definition: &Value) -> Result<(String, Value), &'static str> {
    let properties = definition
        .get("properties")
        .and_then(Value::as_object)
        .ok_or("service definition properties must be an object")?;
    let target_kind = properties
        .get("target_kind")
        .and_then(Value::as_str)
        .unwrap_or("workflow")
        .to_owned();
    let target_config = properties
        .get("target_config_json")
        .and_then(Value::as_str)
        .and_then(|value| serde_json::from_str::<Value>(value).ok())
        .filter(Value::is_object)
        .unwrap_or_else(|| json!({}));
    Ok((target_kind, target_config))
}

fn service_trigger_is_suppressed(
    definition: &Value,
    projection: &Value,
    trigger_type: &str,
    timestamp_ms: u64,
) -> bool {
    let properties = &definition["properties"];
    if properties["enabled"].as_bool() == Some(false) {
        return true;
    }
    let trigger_specs = properties["trigger_specs_json"]
        .as_str()
        .and_then(|raw| serde_json::from_str::<Vec<Value>>(raw).ok())
        .unwrap_or_default();
    let matching = trigger_specs.into_iter().find(|spec| {
        spec["type"]
            .as_str()
            .is_some_and(|value| value.trim().eq_ignore_ascii_case(trigger_type))
    });
    let Some(spec) = matching else {
        return false;
    };
    if spec["enabled"].as_bool() == Some(false) {
        return true;
    }
    let cooldown_ms = spec["cooldown_ms"].as_u64().unwrap_or_default();
    let debounce_ms = spec["debounce_ms"].as_u64().unwrap_or_default();
    let not_before_ms = cooldown_ms.max(debounce_ms);
    let last_triggered_at_ms = projection["last_triggered_at_ms"]
        .as_u64()
        .unwrap_or_default();
    not_before_ms > 0
        && last_triggered_at_ms > 0
        && timestamp_ms.saturating_sub(last_triggered_at_ms) < not_before_ms
}

fn validate_service_declare(input: &mut ServiceDeclareInput) -> Result<(), &'static str> {
    input.service_id = input.service_id.trim().to_owned();
    input.service_kind = input.service_kind.trim().to_owned();
    input.target_kind = input.target_kind.trim().to_owned();
    input.target_ref = input.target_ref.trim().to_owned();
    if input.service_id.is_empty()
        || input.service_kind.is_empty()
        || input.target_kind.is_empty()
        || input.target_ref.is_empty()
    {
        return Err("service_id, service_kind, target_kind, and target_ref are required");
    }
    input.heartbeat_ttl_ms = input.heartbeat_ttl_ms.max(1);
    for spec in &input.trigger_specs {
        let Some(kind) = spec.get("type").and_then(Value::as_str) else {
            return Err("trigger_specs[].type is required");
        };
        if kind.trim().is_empty() {
            return Err("trigger_specs[].type is required");
        }
    }
    Ok(())
}

fn service_definition_node(input: &ServiceDeclareInput, timestamp: u64) -> (String, Value) {
    let service_id = &input.service_id;
    let entity_id = format!(
        "service_def:{service_id}:{timestamp}:{}",
        uuid::Uuid::new_v4().simple()
    );
    (
        entity_id.clone(),
        json!({
            "id": entity_id,
            "label": format!("service_definition:{service_id}"),
            "type": "entity",
            "summary": format!("Service definition {service_id}"),
            "mentions": [{"spans": [{
                "collection_page_url": format!("workflow/_wf:{service_id}"),
                "context_after": "",
                "context_before": "",
                "doc_id": format!("_wf:{service_id}"),
                "document_page_url": format!("workflow/_wf:{service_id}"),
                "end_char": 1,
                "excerpt": "",
                "page_number": 1,
                "source_cluster_id": Value::Null,
                "start_char": 0,
            }]}],
            "properties": {
                "service_id": service_id,
                "service_kind": input.service_kind,
                "target_kind": input.target_kind,
                "target_ref": input.target_ref,
                "target_config_json": Value::Object(input.target_config.clone()).to_string(),
                "enabled": input.enabled,
                "autostart": input.autostart,
                "restart_policy_json": Value::Object(input.restart_policy.clone()).to_string(),
                "heartbeat_ttl_ms": input.heartbeat_ttl_ms,
                "trigger_specs_json": Value::Array(input.trigger_specs.clone()).to_string(),
                "security_scope": "workflow",
                "storage_namespace": "workflow",
                "execution_namespace": "workflow",
                "created_at_ms": timestamp,
                "updated_at_ms": timestamp,
            },
            "metadata": {
                "entity_type": "service_definition",
                "artifact_kind": "service_definition",
                "service_id": service_id,
                "service_kind": input.service_kind,
                "target_kind": input.target_kind,
                "target_ref": input.target_ref,
                "enabled": input.enabled,
                "updated_at_ms": timestamp,
                "created_at_ms": timestamp,
                "in_conversation_chain": false,
                "lifecycle_status": "active",
                "redirect_to_id": Value::Null,
            },
        }),
    )
}

// PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/document_graph.rs#service_event_values; python=kogwistar/server/service_daemon.py#ServiceSupervisor.list_service_events; fixture=contracts/golden/workflow-service-events-http-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/part_04.rs#sqlite_service_events_fold_authoritative_history_for_rest_and_mcp; python_test=tests/server/test_service_daemon_model.py#test_service_runtime_api_round_trip
fn service_event_values(
    events: impl IntoIterator<Item = (i64, String, String, String, String)>,
    service_id: &str,
    limit: usize,
) -> Vec<Value> {
    let mut current = BTreeMap::<String, (i64, Value)>::new();
    for (seq, entity_kind, entity_id, op, payload_json) in events {
        if entity_kind != "node" {
            continue;
        }
        match op.to_ascii_uppercase().as_str() {
            "DELETE" | "TOMBSTONE" | "REMOVE" => {
                current.remove(&entity_id);
                continue;
            }
            "ADD" | "REPLACE" | "UPSERT" => {}
            _ => continue,
        }
        let Ok(payload) = serde_json::from_str::<Value>(&payload_json) else {
            continue;
        };
        if !payload.is_object() {
            continue;
        }
        current.insert(entity_id, (seq, payload));
    }

    let mut values = current
        .into_iter()
        .filter_map(|(entity_id, (seq, payload))| {
            let metadata = payload.get("metadata")?.as_object()?;
            if metadata.get("entity_type").and_then(Value::as_str) != Some("service_event")
                || metadata.get("service_id").and_then(Value::as_str) != Some(service_id)
            {
                return None;
            }
            let event_type = metadata
                .get("service_event_type")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let ts_ms = metadata
                .get("ts_ms")
                .and_then(Value::as_i64)
                .unwrap_or_default();
            let properties = payload.get("properties").and_then(Value::as_object);
            let event_payload = properties
                .and_then(|value| value.get("payload_json"))
                .and_then(Value::as_str)
                .and_then(|value| serde_json::from_str::<Value>(value).ok())
                .filter(Value::is_object)
                .unwrap_or_else(|| json!({}));
            let event_id = payload
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or(&entity_id)
                .to_owned();
            Some((
                ts_ms,
                seq,
                json!({
                    "event_id": event_id,
                    "service_id": service_id,
                    "event_type": event_type,
                    "ts_ms": ts_ms,
                    "payload": event_payload,
                }),
            ))
        })
        .collect::<Vec<_>>();
    values.sort_by_key(|left| (left.0, left.1));
    values.truncate(limit);
    values.into_iter().map(|(_, _, value)| value).collect()
}

fn service_projection_values(
    events: impl IntoIterator<Item = (i64, String, String, String, String)>,
) -> Result<BTreeMap<String, (i64, Value)>, String> {
    let mut current = BTreeMap::<String, (i64, Value)>::new();
    let mut latest_seq = 0_i64;
    for (seq, entity_kind, entity_id, op, payload_json) in events {
        latest_seq = latest_seq.max(seq);
        if entity_kind != "node" {
            continue;
        }
        match op.to_ascii_uppercase().as_str() {
            "DELETE" | "TOMBSTONE" | "REMOVE" => {
                current.remove(&entity_id);
                continue;
            }
            "ADD" | "REPLACE" | "UPSERT" => {}
            _ => continue,
        }
        let Ok(payload) = serde_json::from_str::<Value>(&payload_json) else {
            continue;
        };
        if payload.is_object() {
            current.insert(entity_id, (seq, payload));
        }
    }

    let mut definitions = BTreeMap::<String, (i64, i64, Value)>::new();
    let mut service_events = BTreeMap::<String, Vec<(i64, Value)>>::new();
    for (_entity_id, (seq, payload)) in current {
        let Some(metadata) = payload.get("metadata").and_then(Value::as_object) else {
            continue;
        };
        let Some(service_id) = metadata.get("service_id").and_then(Value::as_str) else {
            continue;
        };
        match metadata.get("entity_type").and_then(Value::as_str) {
            Some("service_definition") => {
                let updated_at_ms = metadata
                    .get("updated_at_ms")
                    .and_then(Value::as_i64)
                    .unwrap_or_default();
                let replace =
                    definitions
                        .get(service_id)
                        .is_none_or(|(current_updated, current_seq, _)| {
                            (updated_at_ms, seq) > (*current_updated, *current_seq)
                        });
                if replace {
                    definitions.insert(service_id.to_owned(), (updated_at_ms, seq, payload));
                }
            }
            Some("service_event") => service_events
                .entry(service_id.to_owned())
                .or_default()
                .push((seq, payload)),
            _ => {}
        }
    }

    let mut projections = BTreeMap::new();
    for (service_id, (_updated_at_ms, _definition_seq, definition)) in definitions {
        let metadata = definition["metadata"]
            .as_object()
            .ok_or_else(|| "service definition metadata must be an object".to_owned())?;
        let properties = definition["properties"]
            .as_object()
            .ok_or_else(|| "service definition properties must be an object".to_owned())?;
        let json_property = |name: &str, fallback: Value| {
            properties
                .get(name)
                .and_then(Value::as_str)
                .and_then(|raw| serde_json::from_str::<Value>(raw).ok())
                .unwrap_or(fallback)
        };
        let enabled = properties
            .get("enabled")
            .and_then(Value::as_bool)
            .unwrap_or(true);
        let created_at_ms = properties
            .get("created_at_ms")
            .and_then(Value::as_i64)
            .or_else(|| metadata.get("created_at_ms").and_then(Value::as_i64))
            .unwrap_or_default();
        let updated_at_ms = properties
            .get("updated_at_ms")
            .and_then(Value::as_i64)
            .or_else(|| metadata.get("updated_at_ms").and_then(Value::as_i64))
            .unwrap_or(created_at_ms);
        let mut projection = json!({
            "service_id": service_id,
            "service_kind": properties.get("service_kind").and_then(Value::as_str).unwrap_or("service"),
            "target_kind": properties.get("target_kind").and_then(Value::as_str).unwrap_or("workflow"),
            "target_ref": properties.get("target_ref").and_then(Value::as_str).unwrap_or_default(),
            "enabled": enabled,
            "autostart": properties.get("autostart").and_then(Value::as_bool).unwrap_or(false),
            "storage_namespace": properties.get("storage_namespace").and_then(Value::as_str).unwrap_or("workflow"),
            "execution_namespace": properties.get("execution_namespace").and_then(Value::as_str).unwrap_or("workflow"),
            "security_scope": properties.get("security_scope").and_then(Value::as_str).unwrap_or("workflow"),
            "lifecycle_status": if enabled { "degraded" } else { "stopped" },
            "health_status": if enabled { "degraded" } else { "stopped" },
            "last_heartbeat_ms": Value::Null,
            "restart_count": 0,
            "current_child_run_id": Value::Null,
            "current_child_status": Value::Null,
            "current_child_started_at_ms": Value::Null,
            "last_handled_child_run_id": Value::Null,
            "last_trigger_type": Value::Null,
            "last_triggered_at_ms": Value::Null,
            "heartbeat_ttl_ms": properties.get("heartbeat_ttl_ms").and_then(Value::as_i64).unwrap_or(60_000).max(1),
            "restart_policy": json_property("restart_policy_json", json!({})),
            "trigger_specs": json_property("trigger_specs_json", json!([])),
            "last_message_seen_ms": 0,
            "last_graph_event_seq": 0,
            "next_due_at_ms": Value::Null,
            "restart_not_before_ms": Value::Null,
            "created_at_ms": created_at_ms,
            "updated_at_ms": updated_at_ms,
        });
        let mut history = service_events.remove(&service_id).unwrap_or_default();
        history.sort_by_key(|(seq, _)| *seq);
        for (seq, event) in history {
            let metadata = &event["metadata"];
            let event_type = metadata["service_event_type"].as_str().unwrap_or_default();
            let ts_ms = metadata["ts_ms"].as_i64().unwrap_or_default();
            let payload = event["properties"]["payload_json"]
                .as_str()
                .and_then(|raw| serde_json::from_str::<Value>(raw).ok())
                .unwrap_or_else(|| json!({}));
            projection["updated_at_ms"] = json!(
                projection["updated_at_ms"]
                    .as_i64()
                    .unwrap_or_default()
                    .max(ts_ms)
            );
            match event_type {
                "service.heartbeat" => projection["last_heartbeat_ms"] = json!(ts_ms),
                "service.enabled" => {
                    projection["enabled"] = json!(payload["enabled"].as_bool().unwrap_or(true));
                }
                "service.triggered" => {
                    projection["last_trigger_type"] = payload["trigger_type"].clone();
                    projection["last_triggered_at_ms"] = json!(ts_ms);
                    projection["lifecycle_status"] = json!("starting");
                }
                "service.starting" => {
                    projection["lifecycle_status"] = json!("starting");
                    projection["health_status"] = json!("healthy");
                }
                "service.healthy" => {
                    projection["lifecycle_status"] = json!("healthy");
                    projection["health_status"] =
                        json!(payload["health_status"].as_str().unwrap_or("healthy"));
                }
                "service.degraded" => {
                    projection["lifecycle_status"] = json!("degraded");
                    projection["health_status"] =
                        json!(payload["health_status"].as_str().unwrap_or("degraded"));
                }
                "service.run_spawned" => {
                    projection["current_child_run_id"] = payload["run_id"].clone();
                    projection["current_child_started_at_ms"] = json!(ts_ms);
                    projection["current_child_status"] = json!("queued");
                    projection["lifecycle_status"] = json!("starting");
                }
                "service.restarting" => {
                    projection["lifecycle_status"] = json!("restarting");
                    projection["health_status"] = json!("degraded");
                    projection["restart_not_before_ms"] = payload["restart_not_before_ms"].clone();
                    projection["restart_count"] =
                        json!(payload["restart_count"].as_i64().unwrap_or_else(|| {
                            projection["restart_count"].as_i64().unwrap_or_default()
                        }));
                }
                "service.stopped" => {
                    let still_enabled = payload["enabled"]
                        .as_bool()
                        .unwrap_or_else(|| projection["enabled"].as_bool().unwrap_or(false));
                    projection["enabled"] = json!(still_enabled);
                    if !still_enabled {
                        projection["lifecycle_status"] = json!("stopped");
                        projection["health_status"] = json!("stopped");
                    }
                }
                "service.run_failed" => {
                    projection["current_child_status"] = payload["status"].clone();
                    projection["last_handled_child_run_id"] = payload["run_id"].clone();
                }
                _ => {}
            }
            let _ = seq;
        }
        projections.insert(service_id, (latest_seq, projection));
    }
    Ok(projections)
}

// PYTHON_REFERENCE_V1: rust=rust/crates/kogwistar-api/src/sections/document_graph.rs#service_projection_from_history; python=kogwistar/server/service_daemon.py#ServiceSupervisor.get_service+ServiceSupervisor._rebuild_projection; fixture=contracts/golden/workflow-service-get-http-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/part_04.rs#sqlite_service_get_rebuilds_missing_projection_from_history; python_test=tests/server/test_service_daemon_model.py#test_service_get_rebuilds_missing_projection_from_history; note=canonical raw history rebuilds the disposable serving projection when missing.
fn service_projection_from_history(
    events: impl IntoIterator<Item = (i64, String, String, String, String)>,
    service_id: &str,
) -> Result<Option<(i64, Value)>, String> {
    Ok(service_projection_values(events)?.remove(service_id))
}

fn service_projection_write(last_seq: i64, projection: &Value) -> NamedProjectionWrite {
    NamedProjectionWrite {
        payload: projection.as_object().cloned().unwrap_or_default(),
        last_authoritative_seq: last_seq,
        last_materialized_seq: last_seq,
        projection_schema_version: 1,
        materialization_status: "ready".to_owned(),
    }
}
