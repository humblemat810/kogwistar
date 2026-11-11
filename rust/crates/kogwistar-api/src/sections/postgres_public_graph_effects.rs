// Public graph and document-tree write effects.
// Python counterparts: engine_core.subsystems.ingest/write graph-upsert paths.
impl PostgresRunApplicationService {
    async fn public_graph_upsert_effect(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
        let graph_store = self.graph_store_for_type(&self.graph_role);
        if !graph_store
            .graph_authority_capability()
            .supports_atomic_index_jobs()
            || !self
                .document_authority_capability()
                .supports_projection_upsert()
        {
            return effect_error(
                StatusCode::NOT_IMPLEMENTED,
                "KOGWISTAR_GRAPH_AUTHORITY_UNSUPPORTED",
                "The selected Rust store does not own atomic graph and document projections",
            );
        }
        if !capability_allowed(request, &self.capabilities, "write_graph", &["write_graph"]) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Graph upsert requires write_graph capability",
            );
        }
        let input = match graph_upsert_request(request) {
            Ok(input) => input,
            Err(response) => return response,
        };
        let security_scope = security_scope_value(&request.principal);
        let document_scope = GraphScope {
            namespace: security_scope["storage_namespace"]
                .as_str()
                .unwrap_or("docs")
                .to_owned(),
            workspace_id: security_scope["workspace"].as_str().map(str::to_owned),
            graph_space: security_scope["graph_space"].as_str().map(str::to_owned),
        };
        let existing_document = if input.content.is_none() {
            match graph_store
                .graph_projection_records(GraphProjectionRead {
                    scope: document_scope.clone(),
                    table: "gke_documents".to_owned(),
                    ids: Some(vec![input.doc_id.clone()]),
                    metadata: MetadataFilter::default(),
                    limit: 1,
                })
                .await
            {
                Ok(records) => records.into_iter().next(),
                Err(error) => {
                    return effect_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "KOGWISTAR_DOCUMENT_SOURCE_UNAVAILABLE",
                        error.to_string(),
                    );
                }
            }
        } else {
            None
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
        let mut endpoint_node_ids = external_endpoints.node_ids.clone();
        let mut endpoint_edge_ids = external_endpoints.edge_ids.clone();
        let mut endpoint_edges = BTreeMap::<String, GraphRecord>::new();
        loop {
            let pending_edge_ids = endpoint_edge_ids
                .iter()
                .filter(|id| !endpoint_edges.contains_key(*id))
                .cloned()
                .collect::<Vec<_>>();
            if pending_edge_ids.is_empty() {
                break;
            }
            if endpoint_node_ids
                .len()
                .saturating_add(endpoint_edge_ids.len())
                > 10_000
            {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_GRAPH_ENDPOINT_LIMIT",
                    "External graph endpoint closure exceeds 10000 entities",
                );
            }
            let records = match graph_store
                .graph_projection_records(GraphProjectionRead {
                    scope: document_scope.clone(),
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
                    "One or more external graph endpoints are missing from the principal scope",
                );
            }
            for record in records {
                endpoint_node_ids.extend(
                    graph_record_string_ids(&record, "source_ids")
                        .into_iter()
                        .chain(graph_record_string_ids(&record, "target_ids")),
                );
                endpoint_edge_ids.extend(
                    graph_record_string_ids(&record, "source_edge_ids")
                        .into_iter()
                        .chain(graph_record_string_ids(&record, "target_edge_ids")),
                );
                endpoint_edges.insert(record.id.clone(), record);
            }
        }
        if endpoint_node_ids
            .len()
            .saturating_add(endpoint_edge_ids.len())
            > 10_000
        {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_GRAPH_ENDPOINT_LIMIT",
                "External graph endpoint closure exceeds 10000 entities",
            );
        }
        let endpoint_nodes = if endpoint_node_ids.is_empty() {
            Vec::new()
        } else {
            match graph_store
                .graph_projection_records(GraphProjectionRead {
                    scope: document_scope.clone(),
                    table: "gke_nodes".to_owned(),
                    ids: Some(endpoint_node_ids.iter().cloned().collect()),
                    metadata: MetadataFilter::default(),
                    limit: endpoint_node_ids.len(),
                })
                .await
            {
                Ok(records) if records.len() == endpoint_node_ids.len() => records,
                Ok(_) => {
                    return effect_error(
                        StatusCode::UNPROCESSABLE_ENTITY,
                        "KOGWISTAR_GRAPH_ENDPOINT_INVALID",
                        "One or more external graph endpoints are missing from the principal scope",
                    );
                }
                Err(error) => {
                    return effect_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "KOGWISTAR_GRAPH_ENDPOINT_UNAVAILABLE",
                        error.to_string(),
                    );
                }
            }
        };
        let endpoint_edges = endpoint_edges.into_values().collect::<Vec<_>>();
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
        let (batch, node_ids, edge_ids) = match graph_upsert_mutation_batch(
            &input,
            &source_text,
            &security_scope,
            &external_endpoints,
            &self.graph_role,
        ) {
            Ok(batch) => batch,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error,
                );
            }
        };
        if batch.mutations.is_empty() {
            if should_write_document {
                let mut metadata = serde_json::Map::new();
                metadata.insert("doc_type".to_owned(), json!(input.doc_type));
                metadata.insert("insertion_method".to_owned(), json!(input.insertion_method));
                if let Err(error) = graph_store
                    .upsert_document_projection(DocumentProjectionUpsert {
                        scope: document_scope,
                        record: GraphRecord {
                            id: input.doc_id.clone(),
                            document: Some(source_text),
                            metadata,
                            embedding: None,
                        },
                        embedding_dim: 0,
                    })
                    .await
                {
                    return effect_error(
                        StatusCode::CONFLICT,
                        "KOGWISTAR_DOCUMENT_PROJECTION_FAILED",
                        error.to_string(),
                    );
                }
            }
            return json_effect(
                StatusCode::OK,
                json!({
                    "document_id": input.doc_id,
                    "node_ids": [],
                    "edge_ids": [],
                    "nodes_added": 0,
                    "edges_added": 0,
                }),
            );
        }
        let mutations = match graph_mutation_batch_value(batch.clone()) {
            Ok(mutations) => mutations,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error,
                );
            }
        };
        if let Err(error) = validate_graph_mutation_envelopes(&batch, &mutations) {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                error,
            );
        }
        if mutations
            .iter()
            .any(|mutation| !graph_mutation_scope_allowed(&mutation.scope, &security_scope))
        {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_SCOPE_FORBIDDEN",
                "Graph mutation scope is outside the principal storage scope",
            );
        }
        let mut document_metadata = serde_json::Map::new();
        document_metadata.insert("doc_type".to_owned(), json!(input.doc_type));
        document_metadata.insert("insertion_method".to_owned(), json!(input.insertion_method));
        let document_write = should_write_document.then(|| PostgresGraphProjectionUpsert {
            scope: document_scope.clone(),
            table: "gke_documents".to_owned(),
            record: GraphRecord {
                id: input.doc_id.clone(),
                document: Some(source_text.clone()),
                metadata: document_metadata,
                embedding: None,
            },
            embedding_dim: 0,
        });
        let locked_document_read = GraphProjectionRead {
            scope: document_scope.clone(),
            table: "gke_documents".to_owned(),
            ids: Some(vec![input.doc_id.clone()]),
            metadata: MetadataFilter::default(),
            limit: 1,
        };
        let index_jobs = match graph_index_jobs_value(&batch, &mutations) {
            Ok(jobs) => jobs,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error,
                );
            }
        };
        let acl_principal = graph_acl_principal_from_claims(
            &request.principal,
            security_scope["security_scope"].as_str(),
        );
        let Some(_authenticated_owner_id) = authenticated_run_owner_id(&request.principal) else {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_GRAPH_OWNER_REQUIRED",
                "Graph upsert requires an authenticated owner identity",
            );
        };
        let owner_id = acl_principal.principal_id.clone();
        let acl_security_scope = security_scope["security_scope"].as_str().map(str::to_owned);
        let acl_effect_id = uuid::Uuid::new_v4().to_string();
        let endpoint_scope = document_scope.clone();
        let graph_role = self.graph_role.clone();
        let endpoint_node_ids = endpoint_node_ids.into_iter().collect::<Vec<_>>();
        let endpoint_edge_ids = endpoint_edges
            .iter()
            .map(|record| record.id.clone())
            .collect::<Vec<_>>();
        let expected_endpoint_nodes = endpoint_nodes;
        let expected_endpoint_edges = endpoint_edges;
        let mut identity_locks = endpoint_node_ids
            .iter()
            .map(|id| ("node".to_owned(), id.clone()))
            .chain(
                endpoint_edge_ids
                    .iter()
                    .map(|id| ("edge".to_owned(), id.clone())),
            )
            .chain(
                mutations
                    .iter()
                    .map(|mutation| (mutation.entity_kind.clone(), mutation.record.id.clone())),
            )
            .collect::<BTreeSet<_>>();
        match graph_store
            .transaction(move |uow| {
                Box::pin(async move {
                    for (kind, id) in std::mem::take(&mut identity_locks) {
                        uow.lock_entity_identity(&endpoint_scope.namespace, &kind, &id)
                            .await?;
                    }
                    if !endpoint_node_ids.is_empty() || !endpoint_edge_ids.is_empty() {
                        let nodes = uow
                            .graph_projection_records_for_update(GraphProjectionRead {
                                scope: endpoint_scope.clone(),
                                table: "gke_nodes".to_owned(),
                                ids: Some(endpoint_node_ids.clone()),
                                metadata: MetadataFilter::default(),
                                limit: endpoint_node_ids.len(),
                            })
                            .await?;
                        let edges = uow
                            .graph_projection_records_for_update(GraphProjectionRead {
                                scope: endpoint_scope.clone(),
                                table: "gke_edges".to_owned(),
                                ids: Some(endpoint_edge_ids.clone()),
                                metadata: MetadataFilter::default(),
                                limit: endpoint_edge_ids.len(),
                            })
                            .await?;
                        if !graph_upsert_endpoint_snapshot_matches(&nodes, &expected_endpoint_nodes)
                            || !graph_upsert_endpoint_snapshot_matches(
                                &edges,
                                &expected_endpoint_edges,
                            )
                        {
                            return Err(kogwistar_store_postgres::PostgresStoreError::Backend(
                                "graph endpoint closure changed during authorization".to_owned(),
                            ));
                        }
                        let node_id_set = endpoint_node_ids.iter().collect::<BTreeSet<_>>();
                        let edge_id_set = endpoint_edge_ids.iter().collect::<BTreeSet<_>>();
                        for edge in &edges {
                            if ["source_ids", "target_ids"].into_iter().any(|field| {
                                graph_record_string_ids(edge, field)
                                    .iter()
                                    .any(|id| !node_id_set.contains(id))
                            }) || ["source_edge_ids", "target_edge_ids"].into_iter().any(
                                |field| {
                                    graph_record_string_ids(edge, field)
                                        .iter()
                                        .any(|id| !edge_id_set.contains(id))
                                },
                            ) {
                                return Err(kogwistar_store_postgres::PostgresStoreError::Backend(
                                    "external graph endpoint closure is incomplete".to_owned(),
                                ));
                            }
                        }
                        let mut all_endpoint_ids = endpoint_node_ids.clone();
                        all_endpoint_ids.extend(endpoint_edge_ids.clone());
                        let acl_records = uow
                            .graph_acl_records_for_entities(
                                &endpoint_scope,
                                &graph_role,
                                &all_endpoint_ids,
                            )
                            .await?;
                        let mut acl_index = GraphAclIndex::default();
                        acl_index.extend_from_records(&acl_records, &graph_role);
                        let (visible_nodes, visible_edges) =
                            graph_records_visible_by_acl_with_index(
                                &nodes,
                                &edges,
                                &acl_index,
                                &acl_principal,
                            );
                        let mut visible_ids = visible_nodes
                            .iter()
                            .filter(|record| {
                                !record
                                    .metadata
                                    .get("lifecycle_status")
                                    .and_then(Value::as_str)
                                    .is_some_and(|status| {
                                        matches!(status, "tombstoned" | "deleted")
                                    })
                            })
                            .map(|record| record.id.clone())
                            .collect::<BTreeSet<_>>();
                        visible_ids.extend(visible_edges.iter().map(|record| record.id.clone()));
                        if visible_ids.len() != all_endpoint_ids.len()
                            || all_endpoint_ids.iter().any(|id| !visible_ids.contains(id))
                        {
                            return Err(kogwistar_store_postgres::PostgresStoreError::Backend(
                                "one or more graph endpoints are missing or not visible".to_owned(),
                            ));
                        }
                    }
                    let (acl_mutations, acl_jobs) = resolve_default_acl_graph_effects(
                        uow,
                        &endpoint_scope,
                        &mutations,
                        &graph_role,
                        &owner_id,
                        acl_security_scope.as_deref(),
                        &acl_effect_id,
                    )
                    .await?;
                    if let Some(expected_document) = existing_document {
                        let locked = uow
                            .graph_projection_records_for_update(locked_document_read)
                            .await?;
                        if locked
                            .first()
                            .is_none_or(|current| current.document != expected_document.document)
                        {
                            return Err(kogwistar_store_postgres::PostgresStoreError::Backend(
                                "source document changed while graph evidence was being validated"
                                    .to_owned(),
                            ));
                        }
                    }
                    if let Some(document_write) = document_write {
                        uow.upsert_graph_projection(document_write).await?;
                    }
                    for mutation in mutations {
                        uow.apply_graph_mutation(mutation).await?;
                    }
                    for mutation in acl_mutations {
                        uow.apply_graph_mutation(mutation).await?;
                    }
                    for job in index_jobs.into_iter().chain(acl_jobs) {
                        uow.enqueue_index_job(job).await?;
                    }
                    Ok(())
                })
            })
            .await
        {
            Ok(()) => json_effect(
                StatusCode::OK,
                json!({
                    "document_id": input.doc_id,
                    "node_ids": node_ids,
                    "edge_ids": edge_ids,
                    "nodes_added": node_ids.len(),
                    "edges_added": edge_ids.len(),
                }),
            ),
            Err(error) => effect_error(
                StatusCode::CONFLICT,
                "KOGWISTAR_GRAPH_UPSERT_CONFLICT",
                error.to_string(),
            ),
        }
    }

    // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/postgres_runtime_service.rs#public_document_tree_upsert_effect; python=kogwistar/server_mcp_with_admin.py#document_upsert_tree; fixture=contracts/golden/document-upsert-tree-success-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/part_03.rs#postgres_internal_graph_mutation_route_persists_atomically_when_dsn_available; python_test=tests/server/test_chat_server_api.py#test_document_upsert_tree_uses_document_graph_extraction_entrypoint
    async fn public_document_tree_upsert_effect(
        &self,
        request: &ApiEffectRequest,
    ) -> ApiEffectResponse {
        if !self
            .store
            .graph_authority_capability()
            .supports_atomic_index_jobs()
            || !self
                .document_authority_capability()
                .supports_projection_upsert()
        {
            return effect_error(
                StatusCode::NOT_IMPLEMENTED,
                "KOGWISTAR_GRAPH_AUTHORITY_UNSUPPORTED",
                "The selected Rust store does not own atomic graph and document projections",
            );
        }
        if !capability_allowed(request, &self.capabilities, "write_graph", &["write_graph"]) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Graph upsert requires write_graph capability",
            );
        }
        let (graph_request, inserted_nodes, inserted_edges) =
            match document_tree_upsert_graph_request(request) {
                Ok(parsed) => parsed,
                Err(response) => return response,
            };
        let response = self.public_graph_upsert_effect(&graph_request).await;
        if !(200..300).contains(&response.status) {
            return response;
        }
        let result: Value = match serde_json::from_slice(&response.body) {
            Ok(result) => result,
            Err(error) => {
                return effect_error(
                    StatusCode::BAD_GATEWAY,
                    "KOGWISTAR_GRAPH_UPSERT_RESPONSE_INVALID",
                    error.to_string(),
                );
            }
        };
        let Some(document_id) = result.get("document_id").and_then(Value::as_str) else {
            return effect_error(
                StatusCode::BAD_GATEWAY,
                "KOGWISTAR_GRAPH_UPSERT_RESPONSE_INVALID",
                "graph upsert response did not include document_id",
            );
        };
        let engine_result = json!({
            "document_id": document_id,
            "node_ids": result.get("node_ids").cloned().unwrap_or_else(|| json!([])),
            "edge_ids": result.get("edge_ids").cloned().unwrap_or_else(|| json!([])),
            "nodes_added": result.get("nodes_added").cloned().unwrap_or_else(|| json!(0)),
            "edges_added": result.get("edges_added").cloned().unwrap_or_else(|| json!(0)),
        });
        json_effect(
            StatusCode::OK,
            json!({
                "status": "ok",
                "inserted_nodes": inserted_nodes,
                "inserted_edges": inserted_edges,
                "engine_result": engine_result,
            }),
        )
    }
}
