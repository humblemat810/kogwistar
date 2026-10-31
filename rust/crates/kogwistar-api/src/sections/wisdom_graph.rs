const PUBLIC_WISDOM_GRAPH_UPSERT_ROUTE: &str = "/api/wisdom/graph/upsert";

fn is_public_wisdom_graph_upsert_route(method: &str, path_and_query: &str) -> bool {
    method == "POST" && path_and_query.split('?').next() == Some(PUBLIC_WISDOM_GRAPH_UPSERT_ROUTE)
}

#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct WisdomGraphUpsertRequest {
    inp: WisdomPureGraphInput,
}

#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct WisdomPureGraphInput {
    #[serde(default)]
    content: Option<String>,
    #[serde(default = "default_wisdom_insertion_method")]
    insertion_method: String,
    #[serde(default)]
    nodes: Vec<Value>,
    #[serde(default)]
    edges: Vec<Value>,
}

fn default_wisdom_insertion_method() -> String {
    "api_upsert".to_owned()
}

struct PreparedWisdomGraph {
    mutations: Vec<GraphMutationInput>,
    node_ids: Vec<String>,
    edge_ids: Vec<String>,
}

#[derive(Default)]
struct WisdomGraphReferences {
    node_ids: BTreeSet<String>,
    edge_ids: BTreeSet<String>,
}

fn wisdom_graph_references(mutations: &[GraphMutation]) -> WisdomGraphReferences {
    let local_nodes = mutations
        .iter()
        .filter(|mutation| mutation.entity_kind == "node")
        .map(|mutation| mutation.record.id.as_str())
        .collect::<BTreeSet<_>>();
    let local_edges = mutations
        .iter()
        .filter(|mutation| mutation.entity_kind == "edge")
        .map(|mutation| mutation.record.id.as_str())
        .collect::<BTreeSet<_>>();
    let mut references = WisdomGraphReferences::default();
    for mutation in mutations.iter().filter(|mutation| mutation.entity_kind == "edge") {
        for field in ["source_ids", "target_ids"] {
            references.node_ids.extend(
                graph_record_string_ids_from_value(&mutation.payload, field)
                    .into_iter()
                    .filter(|id| !local_nodes.contains(id.as_str())),
            );
        }
        for field in ["source_edge_ids", "target_edge_ids"] {
            references.edge_ids.extend(
                graph_record_string_ids_from_value(&mutation.payload, field)
                    .into_iter()
                    .filter(|id| !local_edges.contains(id.as_str())),
            );
        }
    }
    references
}

fn wisdom_graph_record_ids(record: &GraphRecord, field: &str) -> Vec<String> {
    graph_record_string_ids(record, field)
}

fn parse_wisdom_graph_upsert(body: &[u8]) -> Result<WisdomGraphUpsertRequest, String> {
    let input: WisdomGraphUpsertRequest =
        serde_json::from_slice(body).map_err(|error| error.to_string())?;
    if input.inp.nodes.iter().chain(&input.inp.edges).any(|item| !item.is_object()) {
        return Err("nodes and edges must contain objects".to_owned());
    }
    Ok(input)
}

fn wisdom_graph_entity_id(
    raw: &Value,
    kind: &str,
    local_ids: &mut BTreeMap<String, String>,
) -> Result<String, String> {
    let token = raw
        .get("id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("{kind} id must be a non-empty string"))?;
    let local_prefix = if kind == "node" { "nn:" } else { "ne:" };
    let wrong_prefix = if kind == "node" { "ne:" } else { "nn:" };
    if token.starts_with(wrong_prefix) {
        return Err(format!("{kind} id uses the wrong batch-local prefix"));
    }
    if token.starts_with(local_prefix) {
        return Ok(local_ids
            .entry(token.to_owned())
            .or_insert_with(|| uuid::Uuid::new_v4().to_string())
            .clone());
    }
    Ok(token.to_owned())
}

fn wisdom_metadata_value(
    supplied: &BTreeMap<String, Value>,
    namespace: &str,
    graph_space: Option<&str>,
) -> Map<String, Value> {
    const RESERVED: &[&str] = &[
        "acl_mode",
        "acl_target_entity_id",
        "acl_target_grain",
        "acl_truth_graph",
        "created_by",
        "lifecycle_status",
        "owner_id",
        "redirect_to_id",
        "security_scope",
        "tombstoned",
        "workspace_id",
        "graph_space",
        "namespace",
    ];
    let mut metadata = supplied
        .iter()
        .filter(|(key, _)| !RESERVED.contains(&key.as_str()) && !key.starts_with("acl_"))
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect::<Map<_, _>>();
    metadata.insert("namespace".to_owned(), json!(namespace));
    if let Some(graph_space) = graph_space {
        metadata.insert("graph_space".to_owned(), json!(graph_space));
    }
    metadata
}

fn wisdom_graph_mutation(
    entity_kind: &str,
    payload: Value,
    embedding: Option<Vec<f32>>,
    scope: &GraphScope,
) -> Result<GraphMutationInput, String> {
    let id = payload
        .get("id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .ok_or_else(|| format!("{entity_kind} id is missing"))?
        .to_owned();
    let model_class = if entity_kind == "node" {
        GraphModelClass::PureChromaNode
    } else {
        GraphModelClass::PureChromaEdge
    };
    let envelope = GraphModelEnvelope::new(model_class, payload.clone());
    envelope
        .validate_family_v1()
        .map_err(|error| error.to_string())?;
    if payload.get("label").and_then(Value::as_str).is_none()
        || payload.get("summary").and_then(Value::as_str).is_none()
    {
        return Err(format!("{entity_kind} label and summary are required"));
    }
    let mut metadata = payload
        .get("metadata")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    for field in ["label", "type", "summary", "doc_id", "domain_id", "canonical_entity_id"] {
        if let Some(value) = payload.get(field).filter(|value| !value.is_null()) {
            metadata.insert(field.to_owned(), value.clone());
        }
    }
    if let Some(properties) = payload.get("properties").filter(|value| !value.is_null()) {
        metadata.insert("properties".to_owned(), properties.clone());
    }
    if entity_kind == "edge" {
        for field in ["relation", "source_ids", "target_ids", "source_edge_ids", "target_edge_ids"] {
            if let Some(value) = payload.get(field).filter(|value| !value.is_null()) {
                metadata.insert(field.to_owned(), value.clone());
            }
        }
    }
    metadata.insert("namespace".to_owned(), json!(scope.namespace));
    metadata.insert("_class_name".to_owned(), json!(if entity_kind == "node" { "PureChromaNode" } else { "PureChromaEdge" }));
    if let Some(workspace_id) = &scope.workspace_id {
        metadata.insert("workspace_id".to_owned(), json!(workspace_id));
    }
    if let Some(graph_space) = &scope.graph_space {
        metadata.insert("graph_space".to_owned(), json!(graph_space));
    }

    let mut document_payload = payload.clone();
    if let Some(object) = document_payload.as_object_mut() {
        object.remove("embedding");
        if entity_kind == "node" {
            object.remove("metadata");
        }
    }
    let document = serde_json::to_string(&document_payload).map_err(|error| error.to_string())?;
    let dimension = embedding.as_ref().map_or(0, Vec::len);
    Ok(GraphMutationInput {
        scope: kogwistar_contracts::GraphScopeV1 {
            namespace: scope.namespace.clone(),
            workspace_id: scope.workspace_id.clone(),
            graph_space: scope.graph_space.clone(),
        },
        table: if entity_kind == "node" { "gke_nodes" } else { "gke_edges" }.to_owned(),
        entity_kind: entity_kind.to_owned(),
        event_id: uuid::Uuid::new_v4().to_string(),
        op: "ADD".to_owned(),
        payload,
        record: kogwistar_contracts::GraphRecordV1 {
            id,
            document: Some(document),
            metadata,
            embedding,
        },
        embedding_dim: dimension,
        model_envelope: Some(envelope),
    })
}

fn prepare_wisdom_graph_mutations(
    input: WisdomPureGraphInput,
    scope: &GraphScope,
) -> Result<PreparedWisdomGraph, String> {
    let _ = (&input.content, &input.insertion_method);
    let mut node_local_ids = BTreeMap::new();
    let mut edge_local_ids = BTreeMap::new();
    let mut node_ids = Vec::with_capacity(input.nodes.len());
    let mut edge_ids = Vec::with_capacity(input.edges.len());
    let mut normalized_nodes = Vec::with_capacity(input.nodes.len());
    let mut normalized_edges = Vec::with_capacity(input.edges.len());
    let mut seen_nodes = BTreeSet::new();
    let mut seen_edges = BTreeSet::new();

    for raw in &input.nodes {
        let id = wisdom_graph_entity_id(raw, "node", &mut node_local_ids)?;
        if !seen_nodes.insert(id.clone()) {
            return Err(format!("duplicate node id {id:?}"));
        }
        let mut object = raw
            .as_object()
            .cloned()
            .ok_or_else(|| "node must be an object".to_owned())?;
        object.insert("id".to_owned(), json!(id));
        object.insert("doc_id".to_owned(), Value::Null);
        object.insert(
            "metadata".to_owned(),
            Value::Object(wisdom_metadata_value(
                &serde_json::from_value(
                    object.get("metadata").cloned().unwrap_or_else(|| json!({})),
                )
                .map_err(|error| format!("node metadata is invalid: {error}"))?,
                &scope.namespace,
                scope.graph_space.as_deref(),
            )),
        );
        let model: kogwistar_contracts::PureChromaNodeV1 =
            serde_json::from_value(Value::Object(object))
                .map_err(|error| format!("invalid pure node: {error}"))?;
        let mut model = model;
        model.fields.embedding = None;
        let payload = serde_json::to_value(&model).map_err(|error| error.to_string())?;
        node_ids.push(id);
        normalized_nodes.push(wisdom_graph_mutation("node", payload, None, scope)?);
    }

    for raw in &input.edges {
        let id = wisdom_graph_entity_id(raw, "edge", &mut edge_local_ids)?;
        if !seen_edges.insert(id.clone()) {
            return Err(format!("duplicate edge id {id:?}"));
        }
        edge_ids.push(id);
    }
    for (raw, id) in input.edges.iter().zip(&edge_ids) {
        let mut object = raw
            .as_object()
            .cloned()
            .ok_or_else(|| "edge must be an object".to_owned())?;
        object.insert("id".to_owned(), json!(id));
        object.insert("doc_id".to_owned(), Value::Null);
        object.entry("source_edge_ids".to_owned()).or_insert(Value::Null);
        object.entry("target_edge_ids".to_owned()).or_insert(Value::Null);
        let metadata = serde_json::from_value(
            object.get("metadata").cloned().unwrap_or_else(|| json!({})),
        )
        .map_err(|error| format!("edge metadata is invalid: {error}"))?;
        object.insert(
            "metadata".to_owned(),
            Value::Object(wisdom_metadata_value(
                &metadata,
                &scope.namespace,
                scope.graph_space.as_deref(),
            )),
        );
        let mut model: kogwistar_contracts::PureChromaEdgeV1 =
            serde_json::from_value(Value::Object(object))
                .map_err(|error| format!("invalid pure edge: {error}"))?;
        for endpoint in model.source_ids.iter_mut().chain(&mut model.target_ids) {
            if let Some(id) = node_local_ids.get(&endpoint.0) {
                endpoint.0 = id.clone();
            } else if endpoint.0.starts_with("nn:") || endpoint.0.starts_with("ne:") {
                return Err(format!("unknown batch-local node endpoint {:?}", endpoint.0));
            }
        }
        for endpoints in [&mut model.source_edge_ids, &mut model.target_edge_ids] {
            if let kogwistar_contracts::RequiredNullable::Value(ids) = endpoints {
                for endpoint in ids {
                    if let Some(id) = edge_local_ids.get(&endpoint.0) {
                        endpoint.0 = id.clone();
                    } else if endpoint.0.starts_with("nn:") || endpoint.0.starts_with("ne:") {
                        return Err(format!("unknown batch-local edge endpoint {:?}", endpoint.0));
                    }
                }
            }
        }
        model.fields.embedding = None;
        let payload = serde_json::to_value(&model).map_err(|error| error.to_string())?;
        normalized_edges.push(wisdom_graph_mutation("edge", payload, None, scope)?);
    }

    let local_edge_ids = edge_ids.iter().cloned().collect::<BTreeSet<_>>();
    let mut pending = normalized_edges;
    let mut ordered_edges = Vec::with_capacity(pending.len());
    let mut materialized = BTreeSet::new();
    while !pending.is_empty() {
        let mut next = Vec::new();
        let mut progress = false;
        for mutation in pending {
            let dependencies = ["source_edge_ids", "target_edge_ids"]
                .into_iter()
                .flat_map(|field| graph_record_string_ids_from_value(&mutation.payload, field))
                .filter(|id| local_edge_ids.contains(id))
                .collect::<BTreeSet<_>>();
            if dependencies.iter().all(|id| materialized.contains(id)) {
                materialized.insert(mutation.record.id.clone());
                ordered_edges.push(mutation);
                progress = true;
            } else {
                next.push(mutation);
            }
        }
        if !progress {
            return Err("hyperedge endpoints contain a cycle or unresolved dependency".to_owned());
        }
        pending = next;
    }
    let mutations = normalized_nodes.into_iter().chain(ordered_edges).collect();
    Ok(PreparedWisdomGraph {
        mutations,
        node_ids,
        edge_ids,
    })
}

fn graph_record_string_ids_from_value(value: &Value, field: &str) -> Vec<String> {
    value
        .get(field)
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect()
}

impl PostgresRunApplicationService {
    async fn embed_wisdom_graph_mutations(
        &self,
        input: &mut GraphMutationBatchInput,
    ) -> Result<(), String> {
        let provider = self
            .embedding_provider
            .as_ref()
            .ok_or_else(|| "wisdom graph writes require a configured embedding provider".to_owned())?;
        let profile = provider.profile();
        profile.validate().map_err(|error| error.to_string())?;

        let mut positions = Vec::new();
        let mut texts = Vec::new();
        for (position, mutation) in input.mutations.iter().enumerate() {
            let label = mutation
                .payload
                .get("label")
                .and_then(Value::as_str)
                .ok_or_else(|| "wisdom graph entity is missing label".to_owned())?;
            let summary = mutation
                .payload
                .get("summary")
                .and_then(Value::as_str)
                .ok_or_else(|| "wisdom graph entity is missing summary".to_owned())?;
            if mutation.embedding_dim != 0 && mutation.embedding_dim != profile.dimension {
                return Err("embedding profile dimension conflicts with wisdom graph mutation".to_owned());
            }
            positions.push(position);
            texts.push(format!("{label}: {summary}"));
        }
        if texts.is_empty() {
            return Ok(());
        }
        let vectors = provider
            .embed(&texts)
            .await
            .map_err(|_| "embedding provider failed".to_owned())?;
        if vectors.len() != positions.len() {
            return Err("embedding provider returned an unexpected vector count".to_owned());
        }
        for (position, vector) in positions.into_iter().zip(vectors) {
            if vector.len() != profile.dimension || vector.iter().any(|value| !value.is_finite()) {
                return Err("embedding provider returned an invalid vector".to_owned());
            }
            let vector = vector.into_iter().map(|value| value as f32).collect::<Vec<_>>();
            if vector.iter().any(|value| !value.is_finite()) {
                return Err("embedding provider vector exceeds graph storage range".to_owned());
            }
            input.mutations[position].record.embedding = Some(vector);
            input.mutations[position].embedding_dim = profile.dimension;
        }
        input.validate().map_err(|error| error.to_string())
    }

    async fn public_wisdom_graph_upsert_effect(
        &self,
        request: &ApiEffectRequest,
    ) -> ApiEffectResponse {
        if !self.wisdom_store.graph_authority_capability().supports_atomic_index_jobs() {
            return effect_error(
                StatusCode::NOT_IMPLEMENTED,
                "KOGWISTAR_GRAPH_AUTHORITY_UNSUPPORTED",
                "The selected Rust store does not own atomic wisdom graph writes",
            );
        }
        if principal_role(&request.principal) != "rw"
            || !principal_strings(&request.principal, "ns")
                .iter()
                .any(|namespace| namespace == "wisdom" || namespace == "*")
            || !capability_allowed(request, &self.capabilities, "write_graph", &["write_graph"])
        {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Wisdom graph upsert requires rw, wisdom namespace, and write_graph capability",
            );
        }
        let Some(_authenticated_owner_id) = authenticated_run_owner_id(&request.principal) else {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_GRAPH_OWNER_REQUIRED",
                "Wisdom graph upsert requires an authenticated owner identity",
            );
        };
        let scope_values = security_scope_value(&request.principal);
        let scope = GraphScope {
            namespace: scope_values["storage_namespace"]
                .as_str()
                .unwrap_or("default")
                .to_owned(),
            workspace_id: scope_values["workspace"].as_str().map(str::to_owned),
            graph_space: scope_values["graph_space"].as_str().map(str::to_owned),
        };
        let input = match parse_wisdom_graph_upsert(&request.body) {
            Ok(input) => input,
            Err(error) => {
                return effect_error(StatusCode::UNPROCESSABLE_ENTITY, "KOGWISTAR_INVALID_REQUEST", error);
            }
        };
        let prepared = match prepare_wisdom_graph_mutations(input.inp, &scope) {
            Ok(prepared) => prepared,
            Err(error) => {
                return effect_error(StatusCode::UNPROCESSABLE_ENTITY, "KOGWISTAR_INVALID_REQUEST", error);
            }
        };
        if prepared.mutations.is_empty() {
            return json_effect(StatusCode::OK, json!({
                "node_ids": prepared.node_ids,
                "edge_ids": prepared.edge_ids,
                "nodes_added": 0,
                "edges_added": 0,
            }));
        }
        let mut batch = GraphMutationBatchInput {
            contract_version: 1,
            atomic: true,
            mutations: prepared.mutations,
            index_jobs: Vec::new(),
        };
        if let Err(error) = self.embed_wisdom_graph_mutations(&mut batch).await {
            return effect_error(StatusCode::BAD_GATEWAY, "KOGWISTAR_EMBEDDING_FAILED", error);
        }
        if batch.mutations.iter().any(|mutation| mutation.record.embedding.is_none()) {
            return effect_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "KOGWISTAR_EMBEDDING_UNAVAILABLE",
                "Wisdom graph writes require an embedding for every node and edge",
            );
        }
        let mutations = match graph_mutation_batch_value(batch.clone()) {
            Ok(mutations) => mutations,
            Err(error) => {
                return effect_error(StatusCode::UNPROCESSABLE_ENTITY, "KOGWISTAR_INVALID_REQUEST", error);
            }
        };
        if let Err(error) = validate_graph_mutation_envelopes(&batch, &mutations) {
            return effect_error(StatusCode::UNPROCESSABLE_ENTITY, "KOGWISTAR_INVALID_REQUEST", error);
        }
        let acl_principal = graph_acl_principal_from_claims(
            &request.principal,
            scope_values["security_scope"].as_str(),
        );
        let owner_id = acl_principal.principal_id.clone();
        let security_scope = scope_values["security_scope"].as_str().map(str::to_owned);
        let effect_id = uuid::Uuid::new_v4().to_string();
        let target_node_ids = mutations
            .iter()
            .filter(|mutation| mutation.entity_kind == "node")
            .map(|mutation| mutation.record.id.clone())
            .collect::<BTreeSet<_>>();
        let target_edge_ids = mutations
            .iter()
            .filter(|mutation| mutation.entity_kind == "edge")
            .map(|mutation| mutation.record.id.clone())
            .collect::<BTreeSet<_>>();
        let mut references = wisdom_graph_references(&mutations);
        let mut external_edges = BTreeMap::<String, GraphRecord>::new();
        loop {
            let pending_edge_ids = references
                .edge_ids
                .iter()
                .filter(|id| !external_edges.contains_key(*id))
                .cloned()
                .collect::<Vec<_>>();
            if pending_edge_ids.is_empty() {
                break;
            }
            if references.node_ids.len().saturating_add(references.edge_ids.len()) > 10_000 {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_GRAPH_ENDPOINT_LIMIT",
                    "Wisdom graph endpoint closure exceeds 10000 entities",
                );
            }
            let records = match self
                .wisdom_store
                .graph_projection_records(GraphProjectionRead {
                    scope: scope.clone(),
                    table: "gke_edges".to_owned(),
                    ids: Some(pending_edge_ids.clone()),
                    metadata: MetadataFilter::default(),
                    limit: pending_edge_ids.len(),
                })
                .await
            {
                Ok(records) => records,
                Err(error) => {
                    return effect_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "KOGWISTAR_GRAPH_ENDPOINT_UNAVAILABLE",
                        error.to_string(),
                    );
                }
            };
            if records.len() != pending_edge_ids.len() {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_GRAPH_ENDPOINT_INVALID",
                    "One or more external wisdom edges are missing from the caller scope",
                );
            }
            for record in records {
                references.node_ids.extend(
                    ["source_ids", "target_ids"]
                        .into_iter()
                        .flat_map(|field| wisdom_graph_record_ids(&record, field)),
                );
                references.edge_ids.extend(
                    ["source_edge_ids", "target_edge_ids"]
                        .into_iter()
                        .flat_map(|field| wisdom_graph_record_ids(&record, field)),
                );
                external_edges.insert(record.id.clone(), record);
            }
            references.edge_ids.retain(|id| !target_edge_ids.contains(id));
        }
        references.node_ids.retain(|id| !target_node_ids.contains(id));
        if references.node_ids.len().saturating_add(references.edge_ids.len()) > 10_000 {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_GRAPH_ENDPOINT_LIMIT",
                "Wisdom graph endpoint closure exceeds 10000 entities",
            );
        }
        let mut validation_node_ids = references.node_ids.clone();
        validation_node_ids.extend(target_node_ids.iter().cloned());
        let mut validation_edge_ids = references.edge_ids.clone();
        validation_edge_ids.extend(target_edge_ids.iter().cloned());
        let expected_nodes = match self
            .wisdom_store
            .graph_projection_records(GraphProjectionRead {
                scope: scope.clone(),
                table: "gke_nodes".to_owned(),
                ids: Some(validation_node_ids.iter().cloned().collect()),
                metadata: MetadataFilter::default(),
                limit: validation_node_ids.len(),
            })
            .await
        {
            Ok(records) if references.node_ids.iter().all(|id| records.iter().any(|r| &r.id == id)) => records,
            Ok(_) => return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_GRAPH_ENDPOINT_INVALID",
                "One or more external wisdom nodes are missing from the caller scope",
            ),
            Err(error) => return effect_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "KOGWISTAR_GRAPH_ENDPOINT_UNAVAILABLE",
                error.to_string(),
            ),
        };
        let expected_edges = match self
            .wisdom_store
            .graph_projection_records(GraphProjectionRead {
                scope: scope.clone(),
                table: "gke_edges".to_owned(),
                ids: Some(validation_edge_ids.iter().cloned().collect()),
                metadata: MetadataFilter::default(),
                limit: validation_edge_ids.len(),
            })
            .await
        {
            Ok(records) if references.edge_ids.iter().all(|id| records.iter().any(|r| &r.id == id)) => records,
            Ok(_) => return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_GRAPH_ENDPOINT_INVALID",
                "One or more external wisdom edges are missing from the caller scope",
            ),
            Err(error) => return effect_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "KOGWISTAR_GRAPH_ENDPOINT_UNAVAILABLE",
                error.to_string(),
            ),
        };
        let existing_target_ids = expected_nodes
            .iter()
            .filter(|record| target_node_ids.contains(&record.id))
            .map(|record| ("node".to_owned(), record.id.clone()))
            .chain(
                expected_edges
                    .iter()
                    .filter(|record| target_edge_ids.contains(&record.id))
                    .map(|record| ("edge".to_owned(), record.id.clone())),
            )
            .collect::<BTreeSet<_>>();
        let new_mutations = mutations
            .iter()
            .filter(|mutation| {
                !existing_target_ids.contains(&(mutation.entity_kind.clone(), mutation.record.id.clone()))
            })
            .cloned()
            .collect::<Vec<_>>();
        let mut identity_locks = validation_node_ids
            .iter()
            .map(|id| ("node".to_owned(), id.clone()))
            .chain(validation_edge_ids.iter().map(|id| ("edge".to_owned(), id.clone())))
            .collect::<BTreeSet<_>>();
        let applied = self
            .wisdom_store
            .transaction(move |uow| {
                let mutations = mutations.clone();
                let new_mutations = new_mutations.clone();
                let expected_nodes = expected_nodes.clone();
                let expected_edges = expected_edges.clone();
                let validation_node_ids = validation_node_ids.iter().cloned().collect::<Vec<_>>();
                let validation_edge_ids = validation_edge_ids.iter().cloned().collect::<Vec<_>>();
                let target_node_ids = target_node_ids.clone();
                let target_edge_ids = target_edge_ids.clone();
                let acl_principal = acl_principal.clone();
                let scope = scope.clone();
                let owner_id = owner_id.clone();
                let security_scope = security_scope.clone();
                let effect_id = effect_id.clone();
                Box::pin(async move {
                    for (kind, id) in std::mem::take(&mut identity_locks) {
                        uow.lock_entity_identity(&scope.namespace, &kind, &id).await?;
                    }
                    let nodes = if validation_node_ids.is_empty() {
                        Vec::new()
                    } else {
                        uow.graph_projection_records_for_update(GraphProjectionRead {
                            scope: scope.clone(),
                            table: "gke_nodes".to_owned(),
                            ids: Some(validation_node_ids.clone()),
                            metadata: MetadataFilter::default(),
                            limit: validation_node_ids.len(),
                        }).await?
                    };
                    let edges = if validation_edge_ids.is_empty() {
                        Vec::new()
                    } else {
                        uow.graph_projection_records_for_update(GraphProjectionRead {
                            scope: scope.clone(),
                            table: "gke_edges".to_owned(),
                            ids: Some(validation_edge_ids.clone()),
                            metadata: MetadataFilter::default(),
                            limit: validation_edge_ids.len(),
                        }).await?
                    };
                    if !graph_upsert_endpoint_snapshot_matches(&nodes, &expected_nodes)
                        || !graph_upsert_endpoint_snapshot_matches(&edges, &expected_edges)
                    {
                        return Err(kogwistar_store_postgres::PostgresStoreError::Backend(
                            "wisdom graph entities changed during authorization".to_owned(),
                        ));
                    }
                    let mut all_existing_ids = nodes.iter().chain(&edges)
                        .map(|record| record.id.clone()).collect::<Vec<_>>();
                    all_existing_ids.sort();
                    all_existing_ids.dedup();
                    let acl_records = uow.graph_acl_records_for_entities(&scope, "wisdom", &all_existing_ids).await?;
                    let mut acl_index = GraphAclIndex::default();
                    acl_index.extend_from_records(&acl_records, "wisdom");
                    let (visible_nodes, visible_edges) = graph_records_visible_by_acl_with_index(
                        &nodes, &edges, &acl_index, &acl_principal,
                    );
                    let expected_entity_keys = nodes.iter().map(|record| ("node", record.id.as_str()))
                        .chain(edges.iter().map(|record| ("edge", record.id.as_str())))
                        .collect::<BTreeSet<_>>();
                    let visible_entity_keys = visible_nodes.iter().map(|record| ("node", record.id.as_str()))
                        .chain(visible_edges.iter().map(|record| ("edge", record.id.as_str())))
                        .collect::<BTreeSet<_>>();
                    if visible_entity_keys != expected_entity_keys
                        || nodes.iter().any(|record| {
                            target_node_ids.contains(&record.id)
                                && matches!(record.metadata.get("lifecycle_status").and_then(Value::as_str), Some("tombstoned" | "deleted"))
                        })
                        || edges.iter().any(|record| {
                            target_edge_ids.contains(&record.id)
                                && matches!(record.metadata.get("lifecycle_status").and_then(Value::as_str), Some("tombstoned" | "deleted"))
                        })
                    {
                        return Err(kogwistar_store_postgres::PostgresStoreError::Backend(
                            "one or more wisdom graph entities are missing, hidden, or tombstoned".to_owned(),
                        ));
                    }
                    let (acl_mutations, acl_jobs) = resolve_default_acl_graph_effects(
                        uow,
                        &scope,
                        &new_mutations,
                        "wisdom",
                        &owner_id,
                        security_scope.as_deref(),
                        &effect_id,
                    )
                    .await?;
                    for mutation in mutations.into_iter().chain(acl_mutations) {
                        uow.apply_graph_mutation(mutation).await?;
                    }
                    for job in acl_jobs {
                        uow.enqueue_index_job(job).await?;
                    }
                    Ok(())
                })
            })
            .await;
        match applied {
            Ok(()) => json_effect(StatusCode::OK, json!({
                "node_ids": prepared.node_ids,
                "edge_ids": prepared.edge_ids,
                "nodes_added": prepared.node_ids.len(),
                "edges_added": prepared.edge_ids.len(),
            })),
            Err(error) => effect_error(
                StatusCode::CONFLICT,
                "KOGWISTAR_WISDOM_GRAPH_UPSERT_CONFLICT",
                error.to_string(),
            ),
        }
    }
}

#[cfg(test)]
mod wisdom_graph_tests {
    use super::*;

    fn fixture_node(id: &str) -> Value {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/python-graph-model-payloads-v1.json"
        )))
        .expect("shared graph model fixture is valid");
        let mut node = fixture["cases"][0]["payload"].clone();
        node["id"] = json!(id);
        node["embedding"] = Value::Null;
        node
    }

    fn fixture_edge(id: &str, sources: Value, source_edges: Value) -> Value {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/python-graph-model-payloads-v1.json"
        )))
        .expect("shared graph model fixture is valid");
        let mut edge = fixture["cases"][1]["payload"].clone();
        edge["id"] = json!(id);
        edge["source_ids"] = sources;
        edge["target_ids"] = json!([]);
        edge["source_edge_ids"] = source_edges;
        edge["target_edge_ids"] = json!([]);
        edge["embedding"] = Value::Null;
        edge
    }

    fn test_scope() -> GraphScope {
        GraphScope {
            namespace: "wisdom".to_owned(),
            workspace_id: Some("workspace-a".to_owned()),
            graph_space: Some("wisdom".to_owned()),
        }
    }

    #[test]
    fn batch_local_ids_resolve_and_reserved_metadata_is_host_controlled() {
        let mut node = fixture_node("nn:node-a");
        node["metadata"]["acl_mode"] = json!("public");
        node["metadata"]["owner_id"] = json!("attacker");
        node["metadata"]["workspace_id"] = json!("other-workspace");
        let edge = fixture_edge("ne:edge-a", json!(["nn:node-a"]), json!([]));
        let prepared = prepare_wisdom_graph_mutations(
            WisdomPureGraphInput {
                content: None,
                insertion_method: "api_upsert".to_owned(),
                nodes: vec![node],
                edges: vec![edge],
            },
            &test_scope(),
        )
        .expect("valid batch-local graph references normalize");
        assert_eq!(prepared.node_ids.len(), 1);
        assert_eq!(prepared.edge_ids.len(), 1);
        assert_ne!(prepared.node_ids[0], "nn:node-a");
        assert_ne!(prepared.edge_ids[0], "ne:edge-a");
        let node_mutation = &prepared.mutations[0];
        let edge_mutation = &prepared.mutations[1];
        assert_eq!(edge_mutation.payload["source_ids"][0], prepared.node_ids[0]);
        assert_eq!(node_mutation.record.metadata["workspace_id"], "workspace-a");
        assert!(!node_mutation.record.metadata.contains_key("acl_mode"));
        assert!(!node_mutation.record.metadata.contains_key("owner_id"));
    }

    #[test]
    fn hyperedge_batch_dependencies_are_applied_before_dependents() {
        let base = fixture_edge("ne:base", json!([]), json!([]));
        let dependent = fixture_edge("ne:dependent", json!([]), json!(["ne:base"]));
        let prepared = prepare_wisdom_graph_mutations(
            WisdomPureGraphInput {
                content: None,
                insertion_method: "api_upsert".to_owned(),
                nodes: Vec::new(),
                edges: vec![dependent, base],
            },
            &test_scope(),
        )
        .expect("acyclic hyperedge dependencies normalize");
        let ordered_ids = prepared
            .mutations
            .iter()
            .map(|mutation| mutation.record.id.as_str())
            .collect::<Vec<_>>();
        let base_id = prepared.edge_ids[1].as_str();
        let dependent_id = prepared.edge_ids[0].as_str();
        assert!(
            ordered_ids.iter().position(|id| *id == base_id)
                < ordered_ids.iter().position(|id| *id == dependent_id)
        );
    }

    #[test]
    fn unknown_batch_local_endpoint_is_rejected() {
        let edge = fixture_edge("edge-a", json!(["nn:missing"]), json!([]));
        let result = prepare_wisdom_graph_mutations(
            WisdomPureGraphInput {
                content: None,
                insertion_method: "api_upsert".to_owned(),
                nodes: Vec::new(),
                edges: vec![edge],
            },
            &test_scope(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn supplied_embeddings_are_discarded_for_python_pure_graph_parity() {
        let mut node = fixture_node("node-a");
        node["embedding"] = json!([1.0, 0.0, 0.0]);
        let mut edge = fixture_edge("edge-a", json!(["node-a"]), json!([]));
        edge["embedding"] = json!([0.0, 1.0, 0.0]);
        let prepared = prepare_wisdom_graph_mutations(
            WisdomPureGraphInput {
                content: None,
                insertion_method: "api_upsert".to_owned(),
                nodes: vec![node],
                edges: vec![edge],
            },
            &test_scope(),
        )
        .expect("well-formed pure graph parses");

        assert!(prepared.mutations.iter().all(|mutation| {
            mutation.record.embedding.is_none() && mutation.payload["embedding"].is_null()
        }));
    }
}
