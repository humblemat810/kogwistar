// Native SQLite-backed answer context, durable semantic events, and persistence.
// Python parity: kogwistar/server/chat_service.py and chat_service_run_execution.py
// (answer submission, run execution, context snapshots, and answer persistence).
impl SqliteRunApplicationService {
    async fn native_answer_view_context(
        &self,
        request: &kogwistar_runtime::RuntimeNativeStepRequest,
        caller: Option<&Value>,
    ) -> Result<Value, NativeEvidenceContextError> {
        let run = self
            .store
            .get_server_run(&request.run_id)
            .map_err(|_| NativeEvidenceContextError::OwnerAuthorization)?
            .ok_or(NativeEvidenceContextError::OwnerAuthorization)?;
        let owner_id = run
            .user_id
            .as_deref()
            .filter(|value| !value.trim().is_empty())
            .ok_or(NativeEvidenceContextError::OwnerAuthorization)?;
        if let Some(caller) = caller {
            if authenticated_run_owner_id(caller) != Some(owner_id) {
                return Err(NativeEvidenceContextError::CallerNotRunOwner);
            }
        }
        let mut principal = self
            .resolve_inline_native_owner_principal(&request.run_id, "read_graph")
            .await
            .map_err(|_| NativeEvidenceContextError::OwnerAuthorization)?;
        principal["actor_id"] = json!(owner_id);

        let turn_id = request
            .turn_node_id
            .as_deref()
            .or_else(|| request.state.get("turn_node_id").and_then(Value::as_str))
            .filter(|id| !id.trim().is_empty())
            .ok_or(NativeEvidenceContextError::GraphReadRejected)?;
        if request
            .state
            .get("turn_node_id")
            .and_then(Value::as_str)
            .is_some_and(|state_turn_id| state_turn_id != turn_id)
        {
            return Err(NativeEvidenceContextError::GraphReadRejected);
        }

        let graph_request = ApiEffectRequest {
            contract_version: 1,
            method: "GET".to_owned(),
            path_and_query: String::new(),
            body: Vec::new(),
            principal,
        };
        let graph_snapshot = self
            .chroma_graph_snapshot(&graph_request, "conversation")
            .await
            .map_err(|response| {
                if response.status >= StatusCode::INTERNAL_SERVER_ERROR.as_u16() {
                    NativeEvidenceContextError::GraphReadUnavailable
                } else {
                    NativeEvidenceContextError::GraphReadRejected
                }
            })?;
        sqlite_native_answer_view_from_visible_records(
            graph_snapshot.nodes(),
            graph_snapshot.edges(),
            &request.conversation_id,
            turn_id,
        )
    }

    async fn native_candidate_context(
        &self,
        request: &kogwistar_runtime::RuntimeNativeStepRequest,
        caller: Option<&Value>,
    ) -> Result<Value, NativeEvidenceContextError> {
        let run = self
            .store
            .get_server_run(&request.run_id)
            .map_err(|_| NativeEvidenceContextError::OwnerAuthorization)?
            .ok_or(NativeEvidenceContextError::OwnerAuthorization)?;
        let owner_id = run
            .user_id
            .as_deref()
            .filter(|value| !value.trim().is_empty())
            .ok_or(NativeEvidenceContextError::OwnerAuthorization)?;
        if let Some(caller) = caller {
            if authenticated_run_owner_id(caller) != Some(owner_id) {
                return Err(NativeEvidenceContextError::CallerNotRunOwner);
            }
        }
        let mut principal = self
            .resolve_inline_native_owner_principal(&request.run_id, "read_graph")
            .await
            .map_err(|_| NativeEvidenceContextError::OwnerAuthorization)?;
        principal["actor_id"] = json!(owner_id);

        let question = request
            .state
            .get("question")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty() && value.len() <= 100_000)
            .ok_or(NativeEvidenceContextError::GraphReadRejected)?;
        let max_candidates = request
            .state
            .get("max_candidates")
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .filter(|value| *value <= 200)
            .ok_or(NativeEvidenceContextError::GraphReadRejected)?;
        let max_retrieval_level = request
            .state
            .get("max_retrieval_level")
            .and_then(Value::as_i64)
            .ok_or(NativeEvidenceContextError::GraphReadRejected)?;
        if max_candidates == 0 {
            return Ok(Value::Array(Vec::new()));
        }
        // Chroma numeric filters use f64; reject bounds that cannot be represented exactly.
        if max_retrieval_level.unsigned_abs() > (1_u64 << 53) {
            return Err(NativeEvidenceContextError::GraphReadRejected);
        }
        let provider = self
            .embedding_provider
            .as_ref()
            .ok_or(NativeEvidenceContextError::GraphReadRejected)?;
        let profile = provider.profile();
        profile
            .validate()
            .map_err(|_| NativeEvidenceContextError::GraphReadRejected)?;
        if profile.dimension > 8192 {
            return Err(NativeEvidenceContextError::GraphReadRejected);
        }
        let mut vectors = provider
            .embed(&[question.to_owned()])
            .await
            .map_err(|_| NativeEvidenceContextError::GraphReadUnavailable)?;
        if vectors.len() != 1 || vectors[0].len() != profile.dimension {
            return Err(NativeEvidenceContextError::GraphReadRejected);
        }
        let metric = match profile.similarity_metric.to_ascii_lowercase().as_str() {
            "cosine" => DistanceMetric::Cosine,
            "l2" => DistanceMetric::L2,
            "ip" => DistanceMetric::InnerProduct,
            _ => return Err(NativeEvidenceContextError::GraphReadRejected),
        };
        let embedding = vectors
            .pop()
            .expect("the embedding count was validated")
            .into_iter()
            .map(|value| {
                let value = value as f32;
                value
                    .is_finite()
                    .then_some(value)
                    .ok_or(NativeEvidenceContextError::GraphReadRejected)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let scope_value = security_scope_value(&principal);
        let namespace = scope_value["storage_namespace"]
            .as_str()
            .unwrap_or("default")
            .to_owned();
        let mut metadata = MetadataFilter::default();
        if let Some(workspace) = principal
            .get("workspace_id")
            .or_else(|| principal.get("workspace"))
            .and_then(Value::as_str)
        {
            metadata
                .equals
                .insert("workspace_id".to_owned(), json!(workspace));
        }
        if let Some(graph_space) = principal.get("graph_space").and_then(Value::as_str) {
            metadata
                .equals
                .insert("graph_space".to_owned(), json!(graph_space));
        }
        let graph = self
            .chroma_graph
            .as_ref()
            .ok_or(NativeEvidenceContextError::GraphReadUnavailable)?;
        let candidates = graph
            .vector_query_of_kind_numeric_lte(
                &namespace,
                GraphEntityKind::Node,
                &VectorQuery {
                    embedding,
                    limit: max_candidates,
                    metadata,
                    metric,
                },
                "level_from_root",
                max_retrieval_level as f64,
            )
            .await
            .map_err(|_| NativeEvidenceContextError::GraphReadUnavailable)?;
        let mut candidate_records = candidates
            .into_iter()
            .map(|matched| matched.record)
            .collect::<Vec<_>>();
        let mut no_edges = Vec::new();
        self.filter_current_chroma_stage2(&namespace, &mut candidate_records, &mut no_edges)
            .map_err(|_| NativeEvidenceContextError::GraphReadUnavailable)?;

        let mut acl_filter = MetadataFilter {
            equals: BTreeMap::from([
                ("entity_type".to_owned(), json!("acl_record")),
                ("acl_truth_graph".to_owned(), json!("knowledge")),
            ]),
        };
        if let Some(workspace) = principal
            .get("workspace_id")
            .or_else(|| principal.get("workspace"))
            .and_then(Value::as_str)
        {
            acl_filter
                .equals
                .insert("workspace_id".to_owned(), json!(workspace));
        }
        if let Some(graph_space) = principal.get("graph_space").and_then(Value::as_str) {
            acl_filter
                .equals
                .insert("graph_space".to_owned(), json!(graph_space));
        }
        let mut acl_records = graph
            .graph_records_of_kind_bounded(&namespace, GraphEntityKind::Node, &acl_filter, 10_000)
            .await
            .map_err(|_| NativeEvidenceContextError::GraphReadUnavailable)?;
        self.filter_current_chroma_stage2(&namespace, &mut acl_records, &mut Vec::new())
            .map_err(|_| NativeEvidenceContextError::GraphReadUnavailable)?;
        let acl_principal =
            graph_acl_principal_from_claims(&principal, scope_value["security_scope"].as_str());
        Ok(sqlite_native_candidate_values(
            &candidate_records,
            &acl_records,
            "knowledge",
            &acl_principal,
        ))
    }

    async fn native_evidence_context(
        &self,
        request: &kogwistar_runtime::RuntimeNativeStepRequest,
        caller: Option<&Value>,
    ) -> Result<Value, NativeEvidenceContextError> {
        let run = self
            .store
            .get_server_run(&request.run_id)
            .map_err(|_| NativeEvidenceContextError::OwnerAuthorization)?
            .ok_or(NativeEvidenceContextError::OwnerAuthorization)?;
        let owner_id = run
            .user_id
            .as_deref()
            .filter(|value| !value.trim().is_empty())
            .ok_or(NativeEvidenceContextError::OwnerAuthorization)?;
        if caller.is_some_and(|caller| authenticated_run_owner_id(caller) != Some(owner_id)) {
            return Err(NativeEvidenceContextError::CallerNotRunOwner);
        }
        let mut principal = self
            .resolve_inline_native_owner_principal(&request.run_id, "read_graph")
            .await
            .map_err(|_| NativeEvidenceContextError::OwnerAuthorization)?;
        principal["actor_id"] = json!(owner_id);

        let read_ids = |field: &str| -> Result<Vec<String>, NativeEvidenceContextError> {
            let Some(value) = request.state.get(field) else {
                return Ok(Vec::new());
            };
            let values = value
                .as_array()
                .filter(|values| values.len() <= 100)
                .ok_or(NativeEvidenceContextError::GraphReadRejected)?;
            let ids = values
                .iter()
                .map(|value| {
                    value
                        .as_str()
                        .filter(|id| !id.trim().is_empty())
                        .map(str::to_owned)
                        .ok_or(NativeEvidenceContextError::GraphReadRejected)
                })
                .collect::<Result<Vec<_>, _>>()?;
            if ids.iter().collect::<BTreeSet<_>>().len() != ids.len() {
                return Err(NativeEvidenceContextError::GraphReadRejected);
            }
            Ok(ids)
        };
        let node_ids = read_ids("used_node_ids")?;
        let edge_ids = read_ids("used_edge_ids")?;
        if node_ids.len().saturating_add(edge_ids.len()) > 100 {
            return Err(NativeEvidenceContextError::GraphReadRejected);
        }

        let graph_request = ApiEffectRequest {
            contract_version: 1,
            method: "GET".to_owned(),
            path_and_query: String::new(),
            body: Vec::new(),
            principal,
        };
        let snapshot = self
            .chroma_graph_snapshot(&graph_request, "knowledge")
            .await
            .map_err(|response| {
                if response.status >= StatusCode::INTERNAL_SERVER_ERROR.as_u16() {
                    NativeEvidenceContextError::GraphReadUnavailable
                } else {
                    NativeEvidenceContextError::GraphReadRejected
                }
            })?;
        let selected_nodes = snapshot
            .nodes()
            .iter()
            .filter(|record| node_ids.contains(&record.id))
            .collect::<Vec<_>>();
        let selected_edges = snapshot
            .edges()
            .iter()
            .filter(|record| edge_ids.contains(&record.id))
            .collect::<Vec<_>>();
        if selected_nodes.len() != node_ids.len() || selected_edges.len() != edge_ids.len() {
            return Err(NativeEvidenceContextError::GraphReadRejected);
        }
        let project = |records: Vec<&GraphRecord>| {
            records
                .into_iter()
                .map(|record| {
                    json!({"id":record.id,"document":record.document,"metadata":record.metadata})
                })
                .collect::<Vec<_>>()
        };
        Ok(json!({"nodes":project(selected_nodes),"edges":project(selected_edges)}))
    }

    async fn native_pointer_context(
        &self,
        request: &kogwistar_runtime::RuntimeNativeStepRequest,
        caller: Option<&Value>,
    ) -> Result<Value, NativeEvidenceContextError> {
        let run = self
            .store
            .get_server_run(&request.run_id)
            .map_err(|_| NativeEvidenceContextError::OwnerAuthorization)?
            .ok_or(NativeEvidenceContextError::OwnerAuthorization)?;
        let owner_id = run
            .user_id
            .as_deref()
            .filter(|value| !value.trim().is_empty())
            .ok_or(NativeEvidenceContextError::OwnerAuthorization)?;
        if caller.is_some_and(|caller| authenticated_run_owner_id(caller) != Some(owner_id)) {
            return Err(NativeEvidenceContextError::CallerNotRunOwner);
        }
        let mut principal = self
            .resolve_inline_native_owner_principal(&request.run_id, "read_graph")
            .await
            .map_err(|_| NativeEvidenceContextError::OwnerAuthorization)?;
        principal["actor_id"] = json!(owner_id);

        let read_ids = |field: &str| -> Result<Vec<String>, NativeEvidenceContextError> {
            let Some(value) = request.state.get(field) else {
                return Ok(Vec::new());
            };
            value
                .as_array()
                .filter(|values| values.len() <= 100)
                .ok_or(NativeEvidenceContextError::GraphReadRejected)?
                .iter()
                .map(|value| {
                    value
                        .as_str()
                        .filter(|id| !id.trim().is_empty())
                        .map(str::to_owned)
                        .ok_or(NativeEvidenceContextError::GraphReadRejected)
                })
                .collect()
        };
        let used_ids = read_ids("used_node_ids")?;
        let conversation_id = request
            .state
            .get("conversation_id")
            .and_then(Value::as_str)
            .filter(|id| *id == request.conversation_id)
            .ok_or(NativeEvidenceContextError::GraphReadRejected)?;
        let run_node_id = request
            .state
            .get("run_node_id")
            .and_then(Value::as_str)
            .filter(|id| !id.trim().is_empty())
            .ok_or(NativeEvidenceContextError::GraphReadRejected)?;
        let pointer_ids = used_ids
            .iter()
            .map(|id| {
                crate::native_answer_persist::stable_conversation_id(
                    "ptr",
                    &[
                        ("scope", &format!("conv:{conversation_id}")),
                        ("pointer_kind", "kg_node"),
                        ("target_kind", "node"),
                        ("target_id", id),
                    ],
                )
            })
            .collect::<BTreeSet<_>>();
        let mut wanted_nodes = pointer_ids.clone();
        wanted_nodes.insert(run_node_id.to_owned());
        let wanted_edges = pointer_ids
            .iter()
            .map(|pointer_id| {
                crate::native_answer_persist::stable_conversation_id(
                    "e",
                    &[
                        ("scope", &format!("conv:{conversation_id}")),
                        ("rel", "used_evidence"),
                        ("src", run_node_id),
                        ("dst", pointer_id),
                    ],
                )
            })
            .collect::<BTreeSet<_>>();

        let scope = security_scope_value(&principal);
        let namespace = scope["storage_namespace"].as_str().unwrap_or("default");
        let mut filter = MetadataFilter::default();
        if let Some(workspace) = principal
            .get("workspace_id")
            .or_else(|| principal.get("workspace"))
            .and_then(Value::as_str)
        {
            filter
                .equals
                .insert("workspace_id".to_owned(), json!(workspace));
        }
        if let Some(graph_space) = principal.get("graph_space").and_then(Value::as_str) {
            filter
                .equals
                .insert("graph_space".to_owned(), json!(graph_space));
        }
        let graph = self
            .chroma_graph
            .as_ref()
            .ok_or(NativeEvidenceContextError::GraphReadUnavailable)?;
        const MAX_POINTER_SNAPSHOT_RECORDS: usize = 10_000;
        let (mut nodes, mut edges) = tokio::try_join!(
            graph.graph_records_of_kind_bounded(
                namespace,
                GraphEntityKind::Node,
                &filter,
                MAX_POINTER_SNAPSHOT_RECORDS,
            ),
            graph.graph_records_of_kind_bounded(
                namespace,
                GraphEntityKind::Edge,
                &filter,
                MAX_POINTER_SNAPSHOT_RECORDS,
            ),
        )
        .map_err(|_| NativeEvidenceContextError::GraphReadUnavailable)?;
        self.filter_current_chroma_stage2(namespace, &mut nodes, &mut edges)
            .map_err(|_| NativeEvidenceContextError::GraphReadUnavailable)?;
        for (kind, records) in [("node", &mut nodes), ("edge", &mut edges)] {
            self.merge_current_stage1_records(
                namespace,
                kind,
                &filter,
                records,
                MAX_POINTER_SNAPSHOT_RECORDS,
            )
            .map_err(|_| NativeEvidenceContextError::GraphReadUnavailable)?;
        }
        let acl_principal =
            graph_acl_principal_from_claims(&principal, scope["security_scope"].as_str());
        let knowledge =
            GraphAclProjection::from_complete_snapshot(&nodes, &edges, "knowledge", &acl_principal);
        let conversation = GraphAclProjection::from_complete_snapshot(
            &nodes,
            &edges,
            "conversation",
            &acl_principal,
        );
        let selected = used_ids.iter().cloned().collect::<BTreeSet<_>>();
        let knowledge_records = knowledge
            .nodes()
            .iter()
            .filter(|record| selected.contains(&record.id))
            .collect::<Vec<_>>();
        let visible_knowledge = knowledge_records
            .iter()
            .filter(|record| {
                !matches!(
                    record
                        .metadata
                        .get("lifecycle_status")
                        .and_then(Value::as_str),
                    Some("tombstoned" | "deleted")
                )
            })
            .map(|record| record.id.clone())
            .collect::<BTreeSet<_>>();
        if visible_knowledge != selected {
            return Err(NativeEvidenceContextError::GraphReadRejected);
        }
        let conversation_nodes = conversation
            .nodes()
            .iter()
            .filter(|record| wanted_nodes.contains(&record.id))
            .collect::<Vec<_>>();
        let conversation_edges = conversation
            .edges()
            .iter()
            .filter(|record| wanted_edges.contains(&record.id))
            .collect::<Vec<_>>();
        let raw_node_ids = nodes
            .iter()
            .filter(|record| wanted_nodes.contains(&record.id))
            .map(|record| record.id.as_str())
            .collect::<BTreeSet<_>>();
        let raw_edge_ids = edges
            .iter()
            .filter(|record| wanted_edges.contains(&record.id))
            .map(|record| record.id.as_str())
            .collect::<BTreeSet<_>>();
        if conversation_nodes
            .iter()
            .map(|record| record.id.as_str())
            .collect::<BTreeSet<_>>()
            != raw_node_ids
            || conversation_edges
                .iter()
                .map(|record| record.id.as_str())
                .collect::<BTreeSet<_>>()
                != raw_edge_ids
        {
            return Err(NativeEvidenceContextError::GraphReadRejected);
        }
        let run_record = conversation_nodes
            .iter()
            .find(|record| record.id == run_node_id)
            .ok_or(NativeEvidenceContextError::GraphReadRejected)?;
        if run_record
            .metadata
            .get("entity_type")
            .and_then(Value::as_str)
            != Some("agent_run")
            || run_record
                .metadata
                .get("conversation_id")
                .and_then(Value::as_str)
                != Some(conversation_id)
            || matches!(
                run_record
                    .metadata
                    .get("lifecycle_status")
                    .and_then(Value::as_str),
                Some("tombstoned" | "deleted")
            )
        {
            return Err(NativeEvidenceContextError::GraphReadRejected);
        }
        let project = |records: Vec<&GraphRecord>| {
            records
                .into_iter()
                .map(|record| {
                    json!({"id": record.id, "document": record.document, "metadata": record.metadata})
                })
                .collect::<Vec<_>>()
        };
        Ok(json!({
            "knowledge_nodes": project(knowledge_records),
            "conversation_nodes": project(conversation_nodes),
            "conversation_edges": project(conversation_edges)
        }))
    }

    async fn native_turn_link_context(
        &self,
        request: &kogwistar_runtime::RuntimeNativeStepRequest,
        caller: Option<&Value>,
    ) -> Result<Value, NativeEvidenceContextError> {
        let response_id = request
            .state
            .get("answer")
            .and_then(Value::as_object)
            .and_then(|answer| {
                answer
                    .get("response_node_id")
                    .or_else(|| answer.get("assistant_turn_node_id"))
            })
            .and_then(Value::as_str)
            .filter(|id| !id.trim().is_empty())
            .or_else(|| {
                request
                    .state
                    .get("agentic_answering_result")
                    .and_then(Value::as_object)
                    .and_then(|result| {
                        result
                            .get("assistant_turn_node_id")
                            .or_else(|| result.get("response_node_id"))
                    })
                    .and_then(Value::as_str)
                    .filter(|id| !id.trim().is_empty())
            });
        let Some(response_id) = response_id else {
            return Ok(json!({"nodes": [], "edges": []}));
        };
        let user_turn_id = request
            .state
            .get("turn_node_id")
            .and_then(Value::as_str)
            .filter(|id| !id.trim().is_empty())
            .ok_or(NativeEvidenceContextError::GraphReadRejected)?;
        let conversation_id = request
            .state
            .get("conversation_id")
            .and_then(Value::as_str)
            .filter(|id| *id == request.conversation_id)
            .ok_or(NativeEvidenceContextError::GraphReadRejected)?;
        let run = self
            .store
            .get_server_run(&request.run_id)
            .map_err(|_| NativeEvidenceContextError::OwnerAuthorization)?
            .ok_or(NativeEvidenceContextError::OwnerAuthorization)?;
        let owner_id = run
            .user_id
            .as_deref()
            .filter(|value| !value.trim().is_empty())
            .ok_or(NativeEvidenceContextError::OwnerAuthorization)?;
        if caller.is_some_and(|caller| authenticated_run_owner_id(caller) != Some(owner_id)) {
            return Err(NativeEvidenceContextError::CallerNotRunOwner);
        }
        let mut principal = self
            .resolve_inline_native_owner_principal(&request.run_id, "read_graph")
            .await
            .map_err(|_| NativeEvidenceContextError::OwnerAuthorization)?;
        principal["actor_id"] = json!(owner_id);
        let graph_request = ApiEffectRequest {
            contract_version: 1,
            method: "GET".to_owned(),
            path_and_query: String::new(),
            body: Vec::new(),
            principal,
        };
        let snapshot = self
            .chroma_graph_snapshot(&graph_request, "conversation")
            .await
            .map_err(|response| {
                if response.status >= StatusCode::INTERNAL_SERVER_ERROR.as_u16() {
                    NativeEvidenceContextError::GraphReadUnavailable
                } else {
                    NativeEvidenceContextError::GraphReadRejected
                }
            })?;
        let nodes = snapshot
            .nodes()
            .iter()
            .filter(|record| record.id == user_turn_id || record.id == response_id)
            .collect::<Vec<_>>();
        if nodes.len() != 2 {
            return Err(NativeEvidenceContextError::GraphReadRejected);
        }
        for (expected_id, expected_type) in [
            (user_turn_id, "conversation_turn"),
            (response_id, "assistant_turn"),
        ] {
            let record = nodes
                .iter()
                .find(|record| record.id == expected_id)
                .ok_or(NativeEvidenceContextError::GraphReadRejected)?;
            let actual_type = record.metadata.get("entity_type").and_then(Value::as_str);
            let type_matches = actual_type == Some(expected_type)
                || (expected_type == "assistant_turn" && actual_type == Some("conversation_turn"));
            if !type_matches
                || record
                    .metadata
                    .get("conversation_id")
                    .and_then(Value::as_str)
                    != Some(conversation_id)
                || matches!(
                    record
                        .metadata
                        .get("lifecycle_status")
                        .and_then(Value::as_str),
                    Some("tombstoned" | "deleted")
                )
            {
                return Err(NativeEvidenceContextError::GraphReadRejected);
            }
        }
        Ok(json!({
            "nodes": nodes.into_iter().map(|record| json!({
                "id": record.id,
                "document": record.document,
                "metadata": record.metadata
            })).collect::<Vec<_>>(),
            "edges": []
        }))
    }

    async fn persist_sqlite_native_answer_snapshot(
        &self,
        request: &kogwistar_runtime::RuntimeNativeStepRequest,
    ) -> Result<(), String> {
        if self.chroma_graph.is_none() {
            return Err("native answer snapshots require SQLite canonical graph events with Chroma projection".to_owned());
        }
        let model = self
            .chat_completion_provider
            .as_ref()
            .ok_or_else(|| "Rust-native structured-answer provider is not configured".to_owned())?
            .model()
            .to_owned();
        let batch =
            crate::native_answer_generation::answer_context_snapshot_batch(request, &model)?;
        self.persist_sqlite_native_context_snapshot_batch(request, batch)
            .await
    }

    async fn persist_sqlite_native_selection_context_snapshot(
        &self,
        request: &kogwistar_runtime::RuntimeNativeStepRequest,
    ) -> Result<(), String> {
        kogwistar_runtime::validate_native_step_registration(request, &self.native_operations)
            .map_err(|_| {
                "native evidence selection is not registered for this operation version".to_owned()
            })?;
        if self.chroma_graph.is_none() {
            return Err("native selection snapshots require SQLite canonical graph events with Chroma projection".to_owned());
        }
        let batch = crate::native_answer_generation::selection_context_snapshot_batch(request)?;
        self.persist_sqlite_native_context_snapshot_batch(request, batch)
            .await
    }

    async fn persist_sqlite_native_context_snapshot_batch(
        &self,
        request: &kogwistar_runtime::RuntimeNativeStepRequest,
        batch: kogwistar_contracts::GraphMutationBatchV2,
    ) -> Result<(), String> {
        let (mutations, mut index_jobs) = graph_mutation_batch_v2_value(batch)
            .map_err(|_| "native context snapshot graph mutation is invalid".to_owned())?;
        validate_runtime_graph_mutations(&mutations)
            .map_err(|_| "native context snapshot graph mutation failed validation".to_owned())?;
        let principal = self
            .resolve_inline_native_owner_principal(&request.run_id, "write_graph")
            .await?;
        if !capability_allowed_for_principal(
            &principal,
            &self.capabilities,
            "write_graph.native_context_snapshot",
            &["write_graph"],
        ) || !principal_strings(&principal, "capabilities")
            .iter()
            .any(|capability| capability == "write_graph")
        {
            return Err(
                "native context snapshot requires the run owner's current write_graph capability"
                    .to_owned(),
            );
        }
        let security_scope = security_scope_value(&principal);
        if mutations
            .iter()
            .any(|mutation| !graph_mutation_scope_allowed(&mutation.scope, &security_scope))
        {
            return Err(
                "native context snapshot scope is outside the run owner's admitted scope"
                    .to_owned(),
            );
        }
        let Some(scope) = mutations.first().map(|mutation| mutation.scope.clone()) else {
            return Err("native context snapshot has no graph mutations".to_owned());
        };
        let owner_id = authenticated_run_owner_id(&principal)
            .ok_or_else(|| "native context snapshot requires an authenticated owner".to_owned())?
            .to_owned();
        let owner_acl_id = owner_id.to_ascii_lowercase();
        let acl_scope = principal
            .get("security_scope")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_ascii_lowercase);
        let effect_id = format!(
            "native-context-snapshot|{}|{}|{}",
            request.run_id, request.step_seq, request.token_id
        );
        let run_id = request.run_id.clone();
        let capabilities = self.capabilities.clone();
        self.store
            .immediate_transaction(move |uow| {
                let run = uow.get_server_run(&run_id)?.ok_or_else(|| {
                    SqliteStoreError::RecordedRuntimeConflict(
                        "native context snapshot run is missing".to_owned(),
                    )
                })?;
                if run.user_id.as_deref() != Some(owner_id.as_str()) {
                    return Err(SqliteStoreError::RecordedRuntimeConflict(
                        "native context snapshot run owner changed before commit".to_owned(),
                    ));
                }
                if !capability_allowed_for_principal(
                    &principal,
                    &capabilities,
                    "write_graph.native_context_snapshot.commit",
                    &["write_graph"],
                ) || !principal_strings(&principal, "capabilities")
                    .iter()
                    .any(|capability| capability == "write_graph")
                {
                    return Err(SqliteStoreError::RecordedRuntimeConflict(
                        "native context snapshot write_graph capability was revoked".to_owned(),
                    ));
                }
                if mutations
                    .iter()
                    .any(|mutation| !graph_mutation_scope_allowed(&mutation.scope, &security_scope))
                {
                    return Err(SqliteStoreError::RecordedRuntimeConflict(
                        "native context snapshot scope changed before commit".to_owned(),
                    ));
                }
                let acl_targets = acl_targets_for_graph_mutations(&mutations).0;
                let target_ids = acl_target_entity_ids(&acl_targets);
                let existing =
                    uow.graph_acl_records_for_entities(&scope, "conversation", &target_ids)?;
                let (acl_mutations, acl_jobs) = derive_default_acl_graph_effects_from_records(
                    &scope,
                    &mutations,
                    acl_targets,
                    existing,
                    "conversation",
                    &owner_acl_id,
                    acl_scope.as_deref(),
                    &effect_id,
                )
                .map_err(SqliteStoreError::TransactionAborted)?;
                let mut all_mutations = mutations;
                all_mutations.extend(acl_mutations);
                index_jobs.extend(acl_jobs);
                uow.apply_canonical_graph_mutation_batch(GraphMutationBatch {
                    mutations: all_mutations,
                    index_jobs,
                })?;
                Ok(())
            })
            .map_err(|_| "native answer context snapshot could not be committed".to_owned())
    }

    async fn persist_sqlite_native_provider_event(
        &self,
        event: kogwistar_contracts::DurableProviderEventV1,
        conversation_id: String,
        trace_payload: Value,
    ) -> Result<(), String> {
        let run_id = event.trace.run_id.clone();
        let principal = self
            .resolve_inline_native_owner_principal(&run_id, "write_graph")
            .await?;
        if !capability_allowed_for_principal(
            &principal,
            &self.capabilities,
            "write_graph.provider_event",
            &["write_graph"],
        ) || !principal_strings(&principal, "capabilities")
            .iter()
            .any(|capability| capability == "write_graph")
        {
            return Err("provider event publication requires the run owner's current write_graph capability".to_owned());
        }
        if self.chroma_graph.is_none() {
            return Err("provider event publication requires SQLite canonical graph events with Chroma projection".to_owned());
        }
        let security_scope = security_scope_value(&principal);
        let scope = GraphScope {
            namespace: security_scope["storage_namespace"]
                .as_str()
                .filter(|value| !value.trim().is_empty())
                .ok_or_else(|| "provider event owner has no storage namespace".to_owned())?
                .to_owned(),
            workspace_id: security_scope["workspace"]
                .as_str()
                .filter(|value| !value.is_empty())
                .map(str::to_owned),
            graph_space: Some("conversation".to_owned()),
        };
        let publication = kogwistar_contracts::DurableProviderEventPublicationV1 {
            namespace: scope.namespace.clone(),
            conversation_id: conversation_id.clone(),
            event,
        };
        let mutations = provider_event_graph_mutations(
            &publication,
            &scope,
            security_scope["security_scope"]
                .as_str()
                .unwrap_or_default(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|_| "system clock is before Unix epoch".to_owned())?
                .as_secs() as i64,
        );
        validate_provider_event_publications(
            std::slice::from_ref(&publication),
            &mutations,
            &run_id,
        )?;
        if mutations
            .iter()
            .any(|mutation| !graph_mutation_scope_allowed(&mutation.scope, &security_scope))
        {
            return Err(
                "provider event publication is outside the run owner's admitted scope".to_owned(),
            );
        }
        let owner_id = authenticated_run_owner_id(&principal)
            .ok_or_else(|| "provider event publication requires an authenticated owner".to_owned())?
            .to_owned();
        let owner_acl_id = graph_acl_principal_id(&principal);
        let acl_scope = principal
            .get("security_scope")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_ascii_lowercase);
        let message_id = format!("msg:{}", publication.event.event_id);
        let event_payload = serde_json::to_string(&publication.event)
            .map_err(|_| "provider event could not be serialized".to_owned())?;
        let run_event_payload = serde_json::to_string(&trace_payload)
            .map_err(|_| "provider event trace could not be serialized".to_owned())?;
        let run_event_type = "workflow.provider_event.v1";
        let capabilities = self.capabilities.clone();
        let event_run_id = run_id.clone();
        let event_conversation_id = conversation_id.clone();
        self.store
            .immediate_transaction(move |uow| {
                let run = uow.get_server_run(&event_run_id)?.ok_or_else(|| {
                    SqliteStoreError::RecordedRuntimeConflict(
                        "provider event run is missing".to_owned(),
                    )
                })?;
                if run.user_id.as_deref() != Some(owner_id.as_str())
                    || run.conversation_id != event_conversation_id
                {
                    return Err(SqliteStoreError::RecordedRuntimeConflict(
                        "provider event owner or conversation changed before commit".to_owned(),
                    ));
                }
                if !capability_allowed_for_principal(
                    &principal,
                    &capabilities,
                    "write_graph.provider_event.commit",
                    &["write_graph"],
                ) || !principal_strings(&principal, "capabilities")
                    .iter()
                    .any(|capability| capability == "write_graph")
                {
                    return Err(SqliteStoreError::RecordedRuntimeConflict(
                        "provider event write_graph capability was revoked before commit"
                            .to_owned(),
                    ));
                }
                if mutations
                    .iter()
                    .any(|mutation| !graph_mutation_scope_allowed(&mutation.scope, &security_scope))
                {
                    return Err(SqliteStoreError::RecordedRuntimeConflict(
                        "provider event scope changed before commit".to_owned(),
                    ));
                }
                if let Some(existing) = uow.get_projected_lane_message(&message_id)? {
                    let identical = existing.namespace == publication.namespace
                        && existing.inbox_id == "agent-provider-events"
                        && existing.conversation_id == publication.conversation_id
                        && existing.run_id.as_deref() == Some(event_run_id.as_str())
                        && existing.correlation_id.as_deref()
                            == Some(publication.event.event_id.as_str())
                        && existing.payload_json.as_deref() == Some(event_payload.as_str());
                    if identical {
                        return Ok(());
                    }
                    return Err(SqliteStoreError::RecordedRuntimeConflict(
                        "provider event lane identity conflicts with existing content".to_owned(),
                    ));
                }
                let message_mutation = mutations
                    .iter()
                    .find(|mutation| {
                        mutation.entity_kind == "node" && mutation.record.id == message_id
                    })
                    .cloned()
                    .ok_or_else(|| {
                        SqliteStoreError::TransactionAborted(
                            "provider event message node is missing".to_owned(),
                        )
                    })?;
                let (acl_mutations, mut acl_jobs) = derive_default_acl_graph_effects_from_records(
                    &scope,
                    std::slice::from_ref(&message_mutation),
                    acl_targets_for_graph_mutations(std::slice::from_ref(&message_mutation)).0,
                    uow.graph_acl_records_for_entities(
                        &scope,
                        "conversation",
                        &[message_id.clone()],
                    )?,
                    "conversation",
                    &owner_acl_id,
                    acl_scope.as_deref(),
                    &publication.event.event_id,
                )
                .map_err(SqliteStoreError::TransactionAborted)?;
                let mut graph_mutations = mutations;
                graph_mutations.extend(acl_mutations);
                uow.apply_canonical_graph_mutation_batch(GraphMutationBatch {
                    mutations: graph_mutations,
                    index_jobs: std::mem::take(&mut acl_jobs),
                })?;
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_err(|_| {
                        SqliteStoreError::TransactionAborted(
                            "system clock is before Unix epoch".to_owned(),
                        )
                    })?
                    .as_secs() as i64;
                uow.project_lane_message(NewProjectedLaneMessage {
                    message_id,
                    namespace: publication.namespace,
                    purpose: "agent_event".to_owned(),
                    inbox_id: "agent-provider-events".to_owned(),
                    conversation_id: publication.conversation_id,
                    recipient_id: "agent-event-consumer".to_owned(),
                    sender_id: "agent-runtime".to_owned(),
                    msg_type: "agent.provider_event".to_owned(),
                    status: "pending".to_owned(),
                    created_at: now,
                    available_at: now,
                    run_id: Some(event_run_id.clone()),
                    step_id: Some(publication.event.trace.step_id.clone()),
                    correlation_id: Some(publication.event.event_id.clone()),
                    payload_json: Some(event_payload),
                    error_json: None,
                })?;
                uow.append_server_run_event(&event_run_id, run_event_type, run_event_payload)?;
                Ok(())
            })
            .map_err(|_| "provider event SQLite transaction failed".to_owned())
    }

    async fn execute_sqlite_native_provider_event_request(
        &self,
        handoff: RecordedWorkerHandoff,
        mut request: kogwistar_runtime::RuntimeNativeStepRequest,
    ) -> Result<RuntimeWorkerResultRequest, String> {
        let Some(provider) = self.chat_completion_provider.as_ref() else {
            return Ok(failed_native_result(
                handoff,
                &request,
                "KOGWISTAR_CHAT_PROVIDER_NOT_CONFIGURED",
                "Rust-native provider-event streaming requires a configured chat provider",
            ));
        };
        let provider_request: kogwistar_contracts::ProviderRequestV1 = serde_json::from_value(
            request
                .state
                .get("provider_request")
                .cloned()
                .ok_or_else(|| "agent.event.request requires provider_request state".to_owned())?,
        )
        .map_err(|_| "provider_request does not match the v1 contract".to_owned())?;
        provider_request
            .validate()
            .map_err(|_| "provider_request does not match the v1 contract".to_owned())?;
        if provider_request.model != provider.model() {
            return Err("provider_request model does not match the configured model".to_owned());
        }
        let principal = self
            .resolve_inline_native_owner_principal(&request.run_id, "write_graph")
            .await
            .map_err(|_| "provider-event request owner authorization is unavailable".to_owned())?;
        if !capability_allowed_for_principal(
            &principal,
            &self.capabilities,
            "write_graph.provider_event.request",
            &["write_graph"],
        ) || !principal_strings(&principal, "capabilities")
            .iter()
            .any(|capability| capability == "write_graph")
        {
            return Err(
                "provider-event request requires current run-owner write_graph authority"
                    .to_owned(),
            );
        }
        if self.chroma_graph.is_none() {
            return Err(
                "provider-event request requires SQLite canonical events with configured Chroma projection"
                    .to_owned(),
            );
        }
        let run = self
            .store
            .get_server_run(&request.run_id)
            .map_err(|_| "provider-event run could not be verified".to_owned())?
            .ok_or_else(|| "provider-event run is missing".to_owned())?;
        if run.conversation_id != request.conversation_id {
            return Err("provider-event conversation does not match the admitted run".to_owned());
        }

        use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
        let client_sequence = Arc::new(AtomicI64::new(
            request
                .state
                .get("agent_event_client_sse_sequence")
                .and_then(Value::as_i64)
                .filter(|sequence| *sequence >= 0)
                .unwrap_or(0),
        ));
        let semantic_count = Arc::new(AtomicI64::new(0));
        let provider_sequence = Arc::new(AtomicI64::new(0));
        let saw_failure = Arc::new(AtomicBool::new(false));
        let summaries = Arc::new(Mutex::new(Vec::<Value>::new()));
        let sink_error = Arc::new(Mutex::new(None::<String>));
        let provider_id = "openai_compatible_chat_completions";
        let model = provider.model().to_owned();
        let stream_result = provider
            .stream_request_events(&provider_request, |event| {
                let client_sequence = Arc::clone(&client_sequence);
                let semantic_count = Arc::clone(&semantic_count);
                let provider_sequence = Arc::clone(&provider_sequence);
                let saw_failure = Arc::clone(&saw_failure);
                let summaries = Arc::clone(&summaries);
                let sink_error = Arc::clone(&sink_error);
                let request = request.clone();
                let model = model.clone();
                async move {
                    if let kogwistar_contracts::ProviderEventV1::TextDelta { delta, .. } = &event {
                        if delta.chars().count() > 4096 {
                            return Err(
                                "transient text delta exceeds the configured size limit".to_owned(),
                            );
                        }
                        let sequence = client_sequence
                            .fetch_add(1, Ordering::Relaxed)
                            .saturating_add(1);
                        let _ = self.transient_run_deltas.publish(&request.run_id, sequence, delta);
                        return Ok(());
                    }
                    let event_value = serde_json::to_value(&event)
                        .map_err(|_| "provider event could not be encoded".to_owned())?;
                    provider_sequence.fetch_max(
                        event_value
                            .get("provider_sequence")
                            .and_then(Value::as_i64)
                            .unwrap_or(0),
                        Ordering::Relaxed,
                    );
                    if matches!(event, kogwistar_contracts::ProviderEventV1::ResponseFailed { .. }) {
                        saw_failure.store(true, Ordering::Relaxed);
                    }
                    let sequence = client_sequence.load(Ordering::Relaxed);
                    match self
                        .persist_sqlite_native_stream_semantic_event(
                            &request,
                            event,
                            provider_id,
                            &model,
                            sequence,
                        )
                        .await
                    {
                        Ok(summary) => {
                            semantic_count.fetch_add(1, Ordering::Relaxed);
                            if let Ok(mut values) = summaries.lock() {
                                values.push(summary);
                            }
                            Ok(())
                        }
                        Err(error) => {
                            if let Ok(mut saved) = sink_error.lock() {
                                *saved = Some(error.clone());
                            }
                            Err(error)
                        }
                    }
                }
            })
            .await;
        if let Some(error) = sink_error.lock().ok().and_then(|mut saved| saved.take()) {
            return Err(error);
        }
        if (stream_result.is_err() || semantic_count.load(Ordering::Relaxed) == 0)
            && !saw_failure.load(Ordering::Relaxed)
        {
            let failure = kogwistar_contracts::ProviderEventV1::ResponseFailed {
                contract_version: kogwistar_contracts::PROVIDER_EVENT_CONTRACT_VERSION.to_owned(),
                provider_event_id: None,
                provider_sequence: Some(provider_sequence.load(Ordering::Relaxed)),
                response_id: None,
                error_code: if stream_result.is_err() {
                    "provider_stream_error".to_owned()
                } else {
                    "empty_provider_stream".to_owned()
                },
                message: "provider request did not complete with semantic output".to_owned(),
                retryable: false,
            };
            let summary = self
                .persist_sqlite_native_stream_semantic_event(
                    &request,
                    failure,
                    provider_id,
                    &model,
                    client_sequence.load(Ordering::Relaxed),
                )
                .await?;
            semantic_count.fetch_add(1, Ordering::Relaxed);
            summaries
                .lock()
                .map_err(|_| "provider trace summary is unavailable".to_owned())?
                .push(summary);
        }
        request.ephemeral_context.insert(
            "provider_event_request_result".to_owned(),
            json!({
                "provider_sequence": provider_sequence.load(Ordering::Relaxed),
                "client_sse_sequence": client_sequence.load(Ordering::Relaxed),
                "semantic_event_count": semantic_count.load(Ordering::Relaxed),
            }),
        );
        request.ephemeral_context.insert(
            "_provider_event_trace_summaries".to_owned(),
            Value::Array(
                summaries
                    .lock()
                    .map_err(|_| "provider trace summary is unavailable".to_owned())?
                    .clone(),
            ),
        );
        request
            .ephemeral_context
            .insert("provider_model".to_owned(), json!(model));
        native_result_from_request(handoff, request, &self.native_operations)
    }

    // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/sqlite_native_answer_context.rs#persist_sqlite_native_stream_semantic_event; python=kogwistar/agent/event_journal.py#LaneProviderEventJournal.publish; fixture=contracts/golden/agent-durable-provider-event-v1.json; rust_test=rust/crates/kogwistar-contracts/src/agent_event.rs#durable_provider_event_matches_python_golden_fixture; python_test=tests/agent/test_provider_event_cross_language_contract.py#test_python_durable_provider_event_matches_shared_rust_golden_fixture
    async fn persist_sqlite_native_stream_semantic_event(
        &self,
        request: &kogwistar_runtime::RuntimeNativeStepRequest,
        event: kogwistar_contracts::ProviderEventV1,
        provider_id: &str,
        model: &str,
        client_sse_sequence: i64,
    ) -> Result<Value, String> {
        let summary = crate::native_answer_generation::semantic_event_trace_summary(&event)
            .ok_or_else(|| "provider emitted a non-semantic event".to_owned())?;
        let event_value = serde_json::to_value(&event)
            .map_err(|_| "provider event could not be encoded".to_owned())?;
        let attempt = kogwistar_runtime::native_trace_context_fields(request)
            .get("attempt")
            .and_then(Value::as_i64)
            .filter(|attempt| *attempt > 0)
            .unwrap_or(1);
        let trace_context = kogwistar_runtime::native_trace_context_fields(request);
        let event_trace = kogwistar_contracts::EventTraceMetadataV1 {
            run_id: request.run_id.clone(),
            turn_id: request
                .turn_node_id
                .clone()
                .or_else(|| {
                    request
                        .state
                        .get("turn_node_id")
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                })
                .unwrap_or_default(),
            workflow_node_id: request.node_id.clone(),
            step_id: request.step_seq.to_string(),
            attempt,
            producer_step_seq: Some(request.step_seq),
            producer_attempt: Some(attempt),
            trace_id: trace_context
                .get("trace_id")
                .and_then(Value::as_str)
                .map(str::to_owned),
            span_id: trace_context
                .get("span_id")
                .and_then(Value::as_str)
                .map(str::to_owned),
            parent_span_id: trace_context
                .get("parent_span_id")
                .and_then(Value::as_str)
                .map(str::to_owned),
            provider_id: provider_id.to_owned(),
            provider_response_id: event_value
                .get("response_id")
                .and_then(Value::as_str)
                .map(str::to_owned),
            provider_sequence: event_value
                .get("provider_sequence")
                .and_then(Value::as_i64),
            client_sse_sequence: Some(client_sse_sequence),
        };
        let durable_event = kogwistar_contracts::durable_provider_event(&event_trace, event);
        let mut trace_payload = json!({
            "run_id": request.run_id,
            "trace_id": request.run_id,
            "workflow_id": request.workflow_id,
            "conversation_id": request.conversation_id,
            "turn_node_id": request.turn_node_id,
            "step_id": request.node_id,
            "step_seq": request.step_seq,
            "token_id": request.token_id,
            "parent_token_id": request.parent_token_id,
            "executor_target": "rust-native",
            "provider_id": provider_id,
            "provider_model": model,
            "client_sse_sequence": client_sse_sequence,
            "provider_event": summary,
        });
        if let Some(fields) = trace_payload.as_object_mut() {
            fields.extend(trace_context);
        }
        self.persist_sqlite_native_provider_event(
            durable_event,
            request.conversation_id.clone(),
            trace_payload,
        )
        .await?;
        Ok(summary)
    }

    async fn execute_sqlite_native_answer_generation(
        &self,
        handoff: RecordedWorkerHandoff,
        mut request: kogwistar_runtime::RuntimeNativeStepRequest,
    ) -> Result<RuntimeWorkerResultRequest, String> {
        kogwistar_runtime::validate_native_step_registration(&request, &self.native_operations)
            .map_err(|error| format!("native execution rejected: {error}"))?;
        let Some(provider) = self.chat_completion_provider.as_ref() else {
            return Ok(failed_native_result(
                handoff,
                &request,
                "KOGWISTAR_CHAT_PROVIDER_NOT_CONFIGURED",
                "Rust-native answer generation requires a configured structured chat provider",
            ));
        };
        self.persist_sqlite_native_answer_snapshot(&request).await?;
        let runtime = request
            .state
            .get("_rt")
            .and_then(Value::as_object)
            .ok_or_else(|| "native answer generation requires runtime context".to_owned())?;
        let evidence_pack = runtime.get("evidence_pack").ok_or_else(|| {
            "native answer generation requires a materialized evidence pack".to_owned()
        })?;
        let question = request
            .state
            .get("question")
            .and_then(Value::as_str)
            .ok_or_else(|| "native answer generation requires a question".to_owned())?;
        let system_prompt = request
            .state
            .get("system_prompt")
            .and_then(Value::as_str)
            .ok_or_else(|| "native answer generation requires a system prompt".to_owned())?;
        let used_node_ids = request
            .state
            .get("used_node_ids")
            .and_then(Value::as_array)
            .ok_or_else(|| "native answer generation requires selected evidence IDs".to_owned())?
            .iter()
            .map(|id| {
                id.as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| "selected evidence IDs must be strings".to_owned())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let run_id = request.run_id.clone();
        let workflow_id = request.workflow_id.clone();
        let conversation_id = request.conversation_id.clone();
        let node_id = request.node_id.clone();
        let step_seq = request.step_seq;
        let turn_node_id = request
            .turn_node_id
            .clone()
            .or_else(|| {
                request
                    .state
                    .get("turn_node_id")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            })
            .ok_or_else(|| "native answer generation requires a turn node ID".to_owned())?;
        let trace_context = kogwistar_runtime::native_trace_context_fields(&request);
        let model = provider.model().to_owned();
        let event_model = model.clone();
        let service = self;
        let (answer, event_summaries) = provider
            .answer_with_event_sink(
                system_prompt,
                question,
                evidence_pack,
                &used_node_ids,
                move |event| {
                    let trace_context = trace_context.clone();
                    let service = service;
                    let run_id = run_id.clone();
                    let workflow_id = workflow_id.clone();
                    let conversation_id = conversation_id.clone();
                    let node_id = node_id.clone();
                    let turn_node_id = turn_node_id.clone();
                    let model = event_model.clone();
                    async move {
                        if let kogwistar_contracts::ProviderEventV1::TextDelta { delta, .. } =
                            &event
                        {
                            if delta.chars().count() > 4096 {
                                return Err(
                                    "transient text delta exceeds the configured size limit"
                                        .to_owned(),
                                );
                            }
                            let _ = service.transient_run_deltas.publish_next(&run_id, delta);
                            return Ok(());
                        }
                        let Some(summary) =
                            crate::native_answer_generation::semantic_event_trace_summary(&event)
                        else {
                            return Ok(());
                        };
                        let event_value = serde_json::to_value(&event)
                            .map_err(|_| "provider event could not be encoded".to_owned())?;
                        let attempt = trace_context
                            .get("attempt")
                            .and_then(Value::as_i64)
                            .filter(|value| *value > 0)
                            .unwrap_or(1);
                        let client_sse_sequence = service
                            .transient_run_deltas
                            .last_sequence(&run_id)
                            .unwrap_or_default();
                        let trace = kogwistar_contracts::EventTraceMetadataV1 {
                            run_id: run_id.clone(),
                            turn_id: turn_node_id.clone(),
                            workflow_node_id: node_id.clone(),
                            step_id: step_seq.to_string(),
                            attempt,
                            producer_step_seq: Some(step_seq),
                            producer_attempt: Some(attempt),
                            trace_id: trace_context
                                .get("trace_id")
                                .and_then(Value::as_str)
                                .map(str::to_owned),
                            span_id: trace_context
                                .get("span_id")
                                .and_then(Value::as_str)
                                .map(str::to_owned),
                            parent_span_id: trace_context
                                .get("parent_span_id")
                                .and_then(Value::as_str)
                                .map(str::to_owned),
                            provider_id: "openai_compatible_chat_completions".to_owned(),
                            provider_response_id: event_value
                                .get("response_id")
                                .and_then(Value::as_str)
                                .map(str::to_owned),
                            provider_sequence: event_value
                                .get("provider_sequence")
                                .and_then(Value::as_i64),
                            client_sse_sequence: Some(client_sse_sequence),
                        };
                        let durable = kogwistar_contracts::durable_provider_event(&trace, event);
                        let mut payload = json!({
                            "run_id": run_id,
                            "trace_id": run_id,
                            "workflow_id": workflow_id,
                            "conversation_id": conversation_id,
                            "turn_node_id": turn_node_id,
                            "step_id": node_id,
                            "step_seq": step_seq,
                            "executor_target": "rust-native",
                            "provider_id": "openai_compatible_chat_completions",
                            "provider_model": model,
                            "client_sse_sequence": client_sse_sequence,
                            "provider_event": summary,
                        });
                        if let Some(fields) = payload.as_object_mut() {
                            fields.extend(trace_context);
                        }
                        service
                            .persist_sqlite_native_provider_event(durable, conversation_id, payload)
                            .await
                    }
                },
            )
            .await?;
        request
            .ephemeral_context
            .insert("provider_answer".to_owned(), answer);
        request.ephemeral_context.insert(
            "_provider_event_trace_summaries".to_owned(),
            Value::Array(event_summaries),
        );
        request
            .ephemeral_context
            .insert("provider_model".to_owned(), json!(model));
        native_result_from_request(handoff, request, &self.native_operations)
    }

}
