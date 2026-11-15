    #[tokio::test]
    #[ignore = "requires a disposable live Chroma service; set KOGWISTAR_TEST_CHROMA_URL"]
    async fn sqlite_chroma_native_worker_promotes_to_real_chroma_after_restart() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use jsonwebtoken::{EncodingKey, Header, encode};
        use kogwistar_store::{GraphMutation, GraphMutationBatch, GraphScope, GraphWriteStore};
        use tower::ServiceExt;

        let endpoint = std::env::var("KOGWISTAR_TEST_CHROMA_URL")
            .expect("ignored live Chroma test requires KOGWISTAR_TEST_CHROMA_URL");
        let suffix = uuid::Uuid::new_v4().simple().to_string();
        let namespace = format!("rust-live-{suffix}");
        let entity_id = format!("node-{suffix}");
        let node_collection = format!("kgw-rust-n-{suffix}");
        let edge_collection = format!("kgw-rust-e-{suffix}");
        let path = std::env::temp_dir().join(format!("kogwistar-live-chroma-{suffix}.sqlite3"));
        let graph_config = SqliteChromaGraphConfig {
            endpoint: endpoint.clone(),
            metric: DistanceMetric::Cosine,
            node_collection: node_collection.clone(),
            edge_collection: edge_collection.clone(),
        };
        let provider = Arc::new(FixedEmbeddingProvider {
            profile: kogwistar_engine::embedding::EmbeddingProfile {
                provider: "test".to_owned(),
                model: "fixed-v1".to_owned(),
                dimension: 2,
                similarity_metric: "cosine".to_owned(),
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
            vector: vec![0.6, 0.8],
            calls: Arc::new(Mutex::new(Vec::new())),
            fail: None,
        });
        let source_text = format!("Rust live Chroma restart probe {suffix}");
        let model_fixture: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/python-graph-model-payloads-v1.json"
        ))
        .unwrap();
        let mut payload = model_fixture["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|case| case["kind"] == "node")
            .unwrap()["payload"]
            .clone();
        payload["id"] = json!(entity_id);
        payload["document"] = json!(source_text);
        payload["metadata"]["namespace"] = json!(namespace);
        let mut expected_document_payload = payload.clone();
        expected_document_payload
            .as_object_mut()
            .unwrap()
            .remove("embedding");
        expected_document_payload
            .as_object_mut()
            .unwrap()
            .remove("metadata");
        let expected_document =
            kogwistar_contracts::canonical_json(&expected_document_payload, false);
        let service = open_sqlite_application_service(&path, Some(graph_config.clone()))
            .await
            .unwrap()
            .with_embedding_provider(provider.clone());
        let appended = service
            .store
            .append_raw_entity_event(
                &namespace,
                SqliteNewRawEntityEvent {
                    event_id: format!("add-{suffix}"),
                    entity_kind: "node".to_owned(),
                    entity_id: entity_id.clone(),
                    op: "ADD".to_owned(),
                    payload_json: payload.to_string(),
                },
            )
            .unwrap();
        let revision = sqlite_canonical_entity_revision(
            &service.store,
            &namespace,
            "node",
            &entity_id,
        )
        .unwrap()
        .unwrap();
        let job_payload =
            kogwistar_engine::canonical_revision::canonical_revision_job_payload(&revision);
        service
            .store
            .enqueue_index_job(NewIndexJob {
                job_id: format!("embed-{suffix}"),
                namespace: namespace.clone(),
                entity_kind: "node".to_owned(),
                entity_id: entity_id.clone(),
                index_kind: "node_embedding".to_owned(),
                op: "UPSERT".to_owned(),
                payload_json: Some(job_payload),
                max_retries: 3,
            })
            .unwrap();

        drop(service);
        let service = open_sqlite_application_service(&path, Some(graph_config))
            .await
            .unwrap()
            .with_embedding_provider(provider);
        let pending = service
            .store
            .list_index_jobs(Some(&namespace), Some("PENDING"), None, None, None, 10)
            .unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].entity_id, entity_id);
        assert_eq!(
            sqlite_canonical_entity_revision(&service.store, &namespace, "node", &entity_id)
                .unwrap()
                .unwrap()
                .revision,
            appended.event.seq
        );

        let summary = service
            .run_embedding_projection_jobs_once(1, 30, Some(&namespace))
            .await
            .unwrap();
        assert_eq!(summary.claimed, 1);
        assert_eq!(
            summary.projected, 1,
            "Rust Chroma promotion failed; summary={summary:?}, jobs={:?}",
            service
                .store
                .list_index_jobs(Some(&namespace), None, None, None, None, 10)
                .unwrap()
        );
        assert!(service
            .store
            .get_stage1_node_projection(&namespace, &format!("node:{entity_id}"))
            .unwrap()
            .is_none());

        let chroma_config = |collection: String| kogwistar_store_chroma::ChromaStoreConfig {
            endpoint: endpoint.clone(),
            metric: DistanceMetric::Cosine,
            collection,
            ..Default::default()
        };
        let graph = kogwistar_store_chroma::ChromaGraphStore::connect_entity_collections(
            chroma_config(node_collection),
            chroma_config(edge_collection),
        )
        .await
        .unwrap();
        let stored = graph
            .graph_record_of_kind(&namespace, GraphEntityKind::Node, &entity_id)
            .await
            .unwrap()
            .expect("the Rust worker persisted the node in real Chroma");
        assert_eq!(stored.document.as_deref(), Some(expected_document.as_str()));
        assert_eq!(stored.embedding.as_deref(), Some([0.6_f32, 0.8_f32].as_slice()));
        assert_eq!(stored.metadata["_kogwistar_stage2_ready"], true);
        assert_eq!(
            stored.metadata["_kogwistar_source_revision"],
            appended.event.seq
        );

        let live_workspace = format!("workspace-{suffix}");
        let conversation_id = format!("conversation-{suffix}");
        let upsert_conversation_node = |id: String, mut metadata: serde_json::Map<String, Value>| async {
            metadata.insert("workspace_id".to_owned(), json!(live_workspace));
            metadata.insert("graph_space".to_owned(), json!("conversation"));
            let payload = json!({"id":id,"metadata":metadata});
            service
                .store
                .apply_canonical_graph_mutation_batch(GraphMutationBatch {
                    mutations: vec![GraphMutation {
                        scope: GraphScope {
                            namespace: namespace.clone(),
                            workspace_id: Some(live_workspace.clone()),
                            graph_space: Some("conversation".to_owned()),
                        },
                        table: "gke_nodes".to_owned(),
                        entity_kind: "node".to_owned(),
                        event_id: format!("conversation-seed-{id}"),
                        op: "ADD".to_owned(),
                        payload: payload.clone(),
                        record: GraphRecord {
                            id: id.clone(),
                            document: None,
                            metadata: metadata.clone(),
                            embedding: None,
                        },
                        embedding_dim: 2,
                    }],
                    index_jobs: Vec::new(),
                })
                .unwrap();
            graph
                .upsert_graph_record_of_kind(
                    &namespace,
                    GraphEntityKind::Node,
                    GraphRecord {
                        id,
                        document: None,
                        metadata,
                        embedding: Some(vec![0.6, 0.8]),
                    },
                )
                .await
                .unwrap();
        };
        let conversation_node = |entity_type: &str, extra: Value| {
            let mut metadata = extra.as_object().cloned().unwrap_or_default();
            metadata.insert("entity_type".to_owned(), json!(entity_type));
            metadata.insert("conversation_id".to_owned(), json!(conversation_id));
            metadata
        };
        upsert_conversation_node(
            format!("start-{suffix}"),
            conversation_node(
                "conversation_start",
                json!({"user_id":"reader-live","status":"active"}),
            ),
        )
        .await;
        for (id, turn_index, role, summary) in [
            ("turn-one", 1_i64, "user", "live question"),
            ("turn-two", 2_i64, "assistant", "live answer"),
            ("turn-private", 3_i64, "assistant", "must stay hidden"),
        ] {
            let node_id = format!("{id}-{suffix}");
            upsert_conversation_node(
                node_id.clone(),
                conversation_node(
                    "conversation_turn",
                    json!({"turn_index":turn_index,"role":role,"summary":summary}),
                ),
            )
            .await;
            let is_private = id == "turn-private";
            upsert_conversation_node(
                format!("acl-{id}-{suffix}"),
                conversation_node(
                    "acl_record",
                    json!({
                        "acl_truth_graph":"conversation",
                        "acl_target_grain":"node",
                        "acl_target_entity_id":node_id,
                        "acl_version":1,
                        "acl_mode":if is_private {"private"} else {"public"},
                        "acl_owner_id":if is_private {"other-reader"} else {"reader-live"}
                    }),
                ),
            )
            .await;
        }
        let mut conversation_start_acl = conversation_node(
            "acl_record",
            json!({
                "acl_truth_graph":"conversation",
                "acl_target_grain":"node",
                "acl_target_entity_id":format!("start-{suffix}"),
                "acl_version":1,
                "acl_mode":"public",
                "acl_owner_id":"reader-live"
            }),
        );
        upsert_conversation_node(
            format!("acl-start-{suffix}"),
            std::mem::take(&mut conversation_start_acl),
        )
        .await;

        let conversation_principal = json!({
            "sub":"reader-live",
            "storage_ns":namespace,
            "workspace_id":live_workspace,
            "graph_space":"conversation",
            "ns":["conversation"],
            "capabilities":["read_graph"]
        });
        let transcript_request = ApiEffectRequest {
            contract_version: 1,
            method: "GET".to_owned(),
            path_and_query: format!("/api/conversations/{conversation_id}/turns"),
            body: Vec::new(),
            principal: conversation_principal,
        };
        assert!(service.supports_http_operation("GET", &transcript_request.path_and_query));
        let transcript = service.execute(transcript_request).await;
        assert_eq!(transcript.status, StatusCode::OK.as_u16());
        let transcript: Value = serde_json::from_slice(&transcript.body).unwrap();
        assert_eq!(transcript["conversation_id"], conversation_id);
        assert_eq!(transcript["turns"].as_array().unwrap().len(), 2);
        assert_eq!(transcript["turns"][0]["content"], "live question");
        assert!(!transcript.to_string().contains("must stay hidden"));

        let auth = AuthConfig {
            mode: "dev".to_owned(),
            algorithm: Some("HS256".to_owned()),
            key: Some(format!("chroma-live-secret-{suffix}")),
            ..AuthConfig::default()
        };
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let token = encode(
            &Header::new(jsonwebtoken::Algorithm::HS256),
            &json!({
                "sub":"reader-live", "role":"ro", "storage_ns":namespace,
                "workspace_id":live_workspace, "graph_space":"conversation",
                "ns":["conversation"], "capabilities":["read_graph"], "exp":now + 300
            }),
            &EncodingKey::from_secret(auth.key.as_deref().unwrap().as_bytes()),
        )
        .unwrap();
        let router = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: vec![],
                implementation: ImplementationSnapshot::default(),
                auth,
            },
            Arc::new(service.clone()),
        );
        let mcp = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp/conversation")
                    .header("content-type", "application/json")
                    .header("authorization", format!("Bearer {token}"))
                    .body(Body::from(format!(
                        "{{\"jsonrpc\":\"2.0\",\"id\":\"live-transcript\",\"method\":\"tools/call\",\"params\":{{\"name\":\"conversation.get_transcript\",\"arguments\":{{\"conversation_id\":\"{conversation_id}\"}}}}}}"
                    )))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(mcp.status(), StatusCode::OK);
        let mcp_body = to_bytes(mcp.into_body(), 65_536).await.unwrap();
        let mcp: Value = serde_json::from_slice(&mcp_body).unwrap();
        assert_eq!(mcp["result"]["conversation_id"], conversation_id);
        assert_eq!(mcp["result"]["turns"].as_array().unwrap().len(), 2);
        assert!(!mcp.to_string().contains("must stay hidden"));

        let http_transcript = router
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri(format!("/api/conversations/{conversation_id}/turns"))
                    .header("authorization", format!("Bearer {token}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(http_transcript.status(), StatusCode::OK);
        let http_transcript_body = to_bytes(http_transcript.into_body(), 65_536).await.unwrap();
        let http_transcript: Value = serde_json::from_slice(&http_transcript_body).unwrap();
        assert_eq!(http_transcript["conversation_id"], conversation_id);
        assert_eq!(http_transcript["turns"].as_array().unwrap().len(), 2);
        assert!(!http_transcript.to_string().contains("must stay hidden"));

        drop(graph);
        drop(service);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    #[ignore = "requires a disposable live Chroma service; set KOGWISTAR_TEST_CHROMA_URL"]
    async fn sqlite_chroma_public_graph_upsert_survives_restart_and_projects_to_chroma() {
        let endpoint = std::env::var("KOGWISTAR_TEST_CHROMA_URL")
            .expect("ignored live Chroma test requires KOGWISTAR_TEST_CHROMA_URL");
        let suffix = uuid::Uuid::new_v4().simple().to_string();
        let namespace = format!("rust-public-upsert-{suffix}");
        let node_collection = format!("kgw-pub-n-{suffix}");
        let edge_collection = format!("kgw-pub-e-{suffix}");
        let path = std::env::temp_dir().join(format!("kogwistar-public-chroma-{suffix}.sqlite3"));
        let graph_config = SqliteChromaGraphConfig {
            endpoint: endpoint.clone(),
            metric: DistanceMetric::Cosine,
            node_collection: node_collection.clone(),
            edge_collection: edge_collection.clone(),
        };
        let provider = Arc::new(FixedEmbeddingProvider {
            profile: kogwistar_engine::embedding::EmbeddingProfile {
                provider: "test".to_owned(),
                model: "fixed-v1".to_owned(),
                dimension: 2,
                similarity_metric: "cosine".to_owned(),
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
            vector: vec![0.6, 0.8],
            calls: Arc::new(Mutex::new(Vec::new())),
            fail: None,
        });
        let request: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/graph-extraction-v1.json"
        ))
        .unwrap();
        let service = open_sqlite_application_service(&path, Some(graph_config.clone()))
            .await
            .unwrap()
            .with_embedding_provider(provider.clone());
        let response = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: PUBLIC_GRAPH_UPSERT_ROUTE.to_owned(),
                body: serde_json::to_vec(&request).unwrap(),
                principal: json!({
                    "sub": "live-owner", "user_id": "live-owner", "role": "rw",
                    "ns": [namespace], "storage_ns": namespace,
                    "capabilities": ["write_graph"]
                }),
            })
            .await;
        assert_eq!(
            response.status,
            StatusCode::OK.as_u16(),
            "{}",
            String::from_utf8_lossy(&response.body)
        );
        let result: Value = serde_json::from_slice(&response.body).unwrap();
        let node_ids = result["node_ids"].as_array().unwrap();
        let edge_ids = result["edge_ids"].as_array().unwrap();
        assert_eq!(node_ids.len(), 2);
        assert_eq!(edge_ids.len(), 1);
        let pending = service
            .store
            .list_index_jobs(Some(&namespace), Some("PENDING"), None, None, None, 200)
            .unwrap();
        assert!(
            pending.iter().any(|job| job.entity_id == node_ids[0]),
            "first node intent missing: result={result}, pending={pending:?}"
        );
        assert!(
            pending.iter().any(|job| job.entity_id == node_ids[1]),
            "second node intent missing: result={result}, pending={pending:?}"
        );
        assert!(
            pending
                .iter()
                .any(|job| job.entity_id == edge_ids[0] && job.index_kind == "node_embedding"),
            "edge embedding intent missing: result={result}, pending={pending:?}"
        );
        let embedding_intent_count = pending
            .iter()
            .filter(|job| job.index_kind == "node_embedding")
            .count();
        drop(service);

        let service = open_sqlite_application_service(&path, Some(graph_config.clone()))
            .await
            .unwrap()
            .with_embedding_provider(provider);
        let summary = service
            .run_embedding_projection_jobs_once(200, 30, Some(&namespace))
            .await
            .unwrap();
        assert_eq!(
            summary.claimed, embedding_intent_count,
            "all durable embedding intents are processed"
        );
        assert!(
            summary.projected >= 3,
            "all extracted Node/Edge records reach Chroma: {summary:?}"
        );

        let chroma_config = |collection: String| kogwistar_store_chroma::ChromaStoreConfig {
            endpoint: endpoint.clone(),
            metric: DistanceMetric::Cosine,
            collection,
            ..Default::default()
        };
        let graph = kogwistar_store_chroma::ChromaGraphStore::connect_entity_collections(
            chroma_config(node_collection),
            chroma_config(edge_collection),
        )
        .await
        .unwrap();
        for id in node_ids {
            let stored = graph
                .graph_record_of_kind(
                    &namespace,
                    kogwistar_store::GraphEntityKind::Node,
                    id.as_str().unwrap(),
                )
                .await
                .unwrap()
                .expect("canonical node intent is projected to Chroma");
            assert!(
                stored
                    .document
                    .as_deref()
                    .unwrap_or_default()
                    .contains("summary")
            );
            assert_eq!(stored.embedding.as_deref(), Some([0.6_f32, 0.8_f32].as_slice()));
            assert_eq!(stored.metadata["_kogwistar_stage2_ready"], true);
        }
        let edge = graph
            .graph_record_of_kind(
                &namespace,
                kogwistar_store::GraphEntityKind::Edge,
                edge_ids[0].as_str().unwrap(),
            )
            .await
            .unwrap()
            .expect("canonical edge intent is projected to Chroma");
        assert_eq!(edge.embedding.as_deref(), Some([0.6_f32, 0.8_f32].as_slice()));
        assert_eq!(edge.metadata["_kogwistar_stage2_ready"], true);

        let visualization = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: INTERNAL_GRAPH_VIZ_D3_ROUTE.to_owned(),
                body: serde_json::to_vec(&json!({
                    "doc_id": "doc-1",
                    "mode": "reify",
                    "insertion_method": "api_upsert",
                    "graph_type": "knowledge"
                }))
                .unwrap(),
                principal: json!({
                    "sub": "live-owner",
                    "user_id": "live-owner",
                    "role": "ro",
                    "ns": [namespace],
                    "storage_ns": namespace,
                    "capabilities": ["read_graph"]
                }),
            })
            .await;
        assert_eq!(
            visualization.status,
            StatusCode::OK.as_u16(),
            "{}",
            String::from_utf8_lossy(&visualization.body)
        );
        let visualization_payload: Value =
            serde_json::from_slice(&visualization.body).unwrap();
        let mut visualized_ids = visualization_payload["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|node| node["id"].as_str())
            .collect::<Vec<_>>();
        visualized_ids.sort_unstable();
        let mut expected_visualized_ids = node_ids
            .iter()
            .chain(edge_ids.iter())
            .map(|id| id.as_str().unwrap())
            .collect::<Vec<_>>();
        expected_visualized_ids.sort_unstable();
        assert_eq!(visualized_ids, expected_visualized_ids);

        let unauthorized_visualization = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: INTERNAL_GRAPH_VIZ_D3_ROUTE.to_owned(),
                body: serde_json::to_vec(&json!({
                    "doc_id": "doc-1",
                    "mode": "reify",
                    "insertion_method": "api_upsert",
                    "graph_type": "knowledge"
                }))
                .unwrap(),
                principal: json!({
                    "sub": "other-owner",
                    "user_id": "other-owner",
                    "role": "ro",
                    "ns": [namespace],
                    "storage_ns": namespace,
                    "capabilities": ["read_graph"]
                }),
            })
            .await;
        assert_eq!(
            unauthorized_visualization.status,
            StatusCode::OK.as_u16(),
            "{}",
            String::from_utf8_lossy(&unauthorized_visualization.body)
        );
        let unauthorized_payload: Value =
            serde_json::from_slice(&unauthorized_visualization.body).unwrap();
        assert!(
            unauthorized_payload["nodes"].as_array().unwrap().is_empty(),
            "another principal must not read the graph owner's visualization"
        );

        drop(graph);
        drop(service);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn sqlite_chroma_snapshot_merges_only_current_scoped_stage1_records() {
        use kogwistar_store::MetadataFilter;
        use kogwistar_store_sqlite::NewRawEntityEvent;

        let path = std::env::temp_dir().join(format!(
            "kogwistar-sqlite-stage1-snapshot-{}.sqlite3",
            uuid::Uuid::new_v4()
        ));
        let service = SqliteRunApplicationService::open(&path).unwrap();
        let namespace = "snapshot-tenant";
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/python-graph-model-payloads-v1.json"
        ))
        .unwrap();
        let mut payload = fixture["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|case| case["kind"] == "node")
            .unwrap()["payload"]
            .clone();
        payload["id"] = json!("stage1-visible");
        payload["metadata"]["workspace_id"] = json!("workspace-a");
        service
            .store
            .append_raw_entity_event(
                namespace,
                NewRawEntityEvent {
                    event_id: "stage1-visible-add".to_owned(),
                    entity_kind: "node".to_owned(),
                    entity_id: "stage1-visible".to_owned(),
                    op: "ADD".to_owned(),
                    payload_json: payload.to_string(),
                },
            )
            .unwrap();
        let canonical = sqlite_canonical_entity_revision(
            &service.store,
            namespace,
            "node",
            "stage1-visible",
        )
        .unwrap()
        .unwrap();
        let fingerprint = RevisionFencedChromaProjectionStore::canonical_fingerprint(&canonical)
            .unwrap();
        let stage1 = RevisionFencedChromaProjectionStore::restore_stage1_from_revision(
            &canonical,
            &fingerprint,
            None,
        )
        .unwrap();
        service
            .store
            .replace_stage1_node_projection(namespace, "node:stage1-visible", stage1)
            .unwrap();

        let mut records = Vec::new();
        service
            .merge_current_stage1_records(
                namespace,
                "node",
                &MetadataFilter {
                    equals: [("workspace_id".to_owned(), json!("workspace-a"))]
                        .into_iter()
                        .collect(),
                },
                &mut records,
                10,
            )
            .unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].id, "stage1-visible");
        assert_eq!(records[0].metadata["workspace_id"], "workspace-a");

        let mut wrong_scope = Vec::new();
        service
            .merge_current_stage1_records(
                namespace,
                "node",
                &MetadataFilter {
                    equals: [("workspace_id".to_owned(), json!("workspace-b"))]
                        .into_iter()
                        .collect(),
                },
                &mut wrong_scope,
                10,
            )
            .unwrap();
        assert!(wrong_scope.is_empty());

        service
            .store
            .append_raw_entity_event(
                namespace,
                NewRawEntityEvent {
                    event_id: "stage1-visible-tombstone".to_owned(),
                    entity_kind: "node".to_owned(),
                    entity_id: "stage1-visible".to_owned(),
                    op: "TOMBSTONE".to_owned(),
                    payload_json: json!({"id":"stage1-visible"}).to_string(),
                },
            )
            .unwrap();
        let mut deleted = Vec::new();
        service
            .merge_current_stage1_records(
                namespace,
                "node",
                &MetadataFilter::default(),
                &mut deleted,
                10,
            )
            .unwrap();
        assert!(deleted.is_empty(), "a stale Stage-1 row must not revive a tombstone");

        drop(service);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn sqlite_chroma_reconciliation_removes_orphaned_tombstone_projection() {
        use kogwistar_store::NamedProjectionWrite;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let namespace = "reconcile-tenant";
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let (stop_chroma, mut stop_chroma_rx) = tokio::sync::oneshot::channel();
        let chroma = tokio::spawn(async move {
            let mut observed = Vec::new();
            let mut orphan_deleted = false;
            loop {
                let accepted = tokio::select! {
                    _ = &mut stop_chroma_rx => break,
                    accepted = listener.accept() => accepted.unwrap(),
                };
                let (mut stream, _) = accepted;
                let mut bytes = Vec::new();
                let mut chunk = [0_u8; 4096];
                let (split, length) = loop {
                    let count = stream.read(&mut chunk).await.unwrap();
                    assert!(count > 0, "Chroma client closed before sending a full request");
                    bytes.extend_from_slice(&chunk[..count]);
                    if let Some(split) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
                        let head = String::from_utf8_lossy(&bytes[..split]);
                        let length = head
                            .lines()
                            .find_map(|line| {
                                let (name, value) = line.split_once(':')?;
                                name.eq_ignore_ascii_case("content-length")
                                    .then(|| value.trim().parse::<usize>().ok())
                                    .flatten()
                            })
                            .unwrap_or(0);
                        if bytes.len() >= split + 4 + length {
                            break (split, length);
                        }
                    }
                };
                let head = String::from_utf8_lossy(&bytes[..split]);
                let request_line = head.lines().next().unwrap().to_owned();
                let body: Value = serde_json::from_slice(&bytes[split + 4..split + 4 + length])
                    .unwrap();
                let response = if request_line.contains("/collections HTTP") {
                    json!({
                        "id": format!("collection-{}", body["name"].as_str().unwrap()),
                        "metadata": {"hnsw:space":"cosine"}
                    })
                } else if request_line.contains("/get")
                    && request_line.contains("collection-reconcile-nodes")
                {
                    let row = |id: &str, revision: i64| {
                        let original = json!({
                            "namespace": namespace,
                            "_kogwistar_stage2_ready": true,
                            "_kogwistar_source_revision": revision,
                            "_kogwistar_source_fingerprint": "stale-fingerprint"
                        });
                        json!({
                            "id": id,
                            "document": "stale",
                            "metadata": {
                                "_kogwistar_namespace": namespace,
                                "_kogwistar_original_metadata": original.to_string()
                            },
                            "embedding": [0.2, 0.8]
                        })
                    };
                    let rows = if let Some(ids) = body.get("ids").and_then(Value::as_array) {
                        ids.iter()
                            .filter_map(Value::as_str)
                            .filter(|id| *id != "orphan" || !orphan_deleted)
                            .map(|id| row(id, 1))
                            .collect::<Vec<_>>()
                    } else {
                        let mut all_rows = Vec::new();
                        if !orphan_deleted {
                            all_rows.push(row("orphan", 1));
                        }
                        all_rows.push(row("stale-active", 1));
                        let offset = body["offset"].as_u64().unwrap_or(0) as usize;
                        let limit = body["limit"].as_u64().unwrap_or(2) as usize;
                        all_rows.into_iter().skip(offset).take(limit).collect()
                    };
                    json!({
                        "ids": rows.iter().map(|row| row["id"].clone()).collect::<Vec<_>>(),
                        "documents": rows.iter().map(|row| row["document"].clone()).collect::<Vec<_>>(),
                        "metadatas": rows.iter().map(|row| row["metadata"].clone()).collect::<Vec<_>>(),
                        "embeddings": rows.iter().map(|row| row["embedding"].clone()).collect::<Vec<_>>()
                    })
                } else {
                    json!({"ids":[],"documents":[],"metadatas":[],"embeddings":[]})
                };
                if request_line.contains("/delete") && body["ids"] == json!(["orphan"]) {
                    orphan_deleted = true;
                }
                let response = response.to_string();
                let response = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                    response.len(),
                    response
                );
                stream.write_all(response.as_bytes()).await.unwrap();
                observed.push((request_line, body));
            }
            observed
        });

        let path = std::env::temp_dir().join(format!(
            "kogwistar-sqlite-chroma-reconcile-{}.sqlite3",
            uuid::Uuid::new_v4()
        ));
        let graph_config = SqliteChromaGraphConfig {
            endpoint,
            metric: DistanceMetric::Cosine,
            node_collection: "reconcile-nodes".to_owned(),
            edge_collection: "reconcile-edges".to_owned(),
        };
        let service = open_sqlite_application_service(
            &path,
            Some(graph_config.clone()),
        )
        .await
        .unwrap();
        service
            .store
            .append_raw_entity_event(
                namespace,
                SqliteNewRawEntityEvent {
                    event_id: "orphan-tombstone".to_owned(),
                    entity_kind: "node".to_owned(),
                    entity_id: "orphan".to_owned(),
                    op: "TOMBSTONE".to_owned(),
                    payload_json: json!({"id":"orphan"}).to_string(),
                },
            )
            .unwrap();
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/python-graph-model-payloads-v1.json"
        ))
        .unwrap();
        let mut active_payload = fixture["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|case| case["kind"] == "node")
            .unwrap()["payload"]
            .clone();
        active_payload["id"] = json!("stale-active");
        service
            .store
            .append_raw_entity_event(
                namespace,
                SqliteNewRawEntityEvent {
                    event_id: "stale-active-add".to_owned(),
                    entity_kind: "node".to_owned(),
                    entity_id: "stale-active".to_owned(),
                    op: "ADD".to_owned(),
                    payload_json: active_payload.to_string(),
                },
            )
            .unwrap();
        let mut stage1_only_payload = active_payload.clone();
        stage1_only_payload["id"] = json!("stage1-only-stale");
        service
            .store
            .append_raw_entity_event(
                namespace,
                SqliteNewRawEntityEvent {
                    event_id: "stage1-only-add".to_owned(),
                    entity_kind: "node".to_owned(),
                    entity_id: "stage1-only-stale".to_owned(),
                    op: "ADD".to_owned(),
                    payload_json: stage1_only_payload.to_string(),
                },
            )
            .unwrap();
        service
            .store
            .replace_stage1_node_projection(
                namespace,
                "node:stage1-only-stale",
                NamedProjectionWrite {
                    payload: json!({
                        "id":"stage1-only-stale",
                        "entity_kind":"node",
                        "source_fingerprint":"stale-fingerprint"
                    })
                    .as_object()
                    .unwrap()
                    .clone(),
                    last_authoritative_seq: 2,
                    last_materialized_seq: 2,
                    projection_schema_version: 2,
                    materialization_status: "pending".to_owned(),
                },
            )
            .unwrap();

        // Reconciliation must recover from durable SQLite state after process
        // restart; Chroma remains an external, eventually consistent projection.
        drop(service);
        let service = open_sqlite_application_service(&path, Some(graph_config))
            .await
            .unwrap();
        assert_eq!(
            ApplicationService::reconcile_eventual_projection_once(
                &service,
                namespace.to_owned(),
                2,
            )
            .await
            .unwrap(),
            3
        );
        let pending_jobs = service
            .store
            .list_index_jobs(Some(namespace), Some("PENDING"), None, None, None, 10)
            .unwrap()
            .len();
        assert_eq!(
            ApplicationService::reconcile_eventual_projection_once(
                &service,
                namespace.to_owned(),
                2,
            )
            .await
            .unwrap(),
            0
        );
        assert_eq!(
            service
                .store
                .list_index_jobs(Some(namespace), Some("PENDING"), None, None, None, 10)
                .unwrap()
                .len(),
            pending_jobs,
            "repeat reconciliation must not enqueue an already-staged current revision"
        );
        let recovered_stage1 = service
            .store
            .get_stage1_node_projection(namespace, "node:stage1-only-stale")
            .unwrap()
            .expect("active stale Stage-1 row is rebuilt without a Chroma row");
        assert_ne!(
            recovered_stage1
                .payload
                .get("source_fingerprint")
                .and_then(Value::as_str),
            Some("stale-fingerprint")
        );
        assert!(service
            .store
            .get_stage1_node_projection(namespace, "node:stale-active")
            .unwrap()
            .is_some());
        assert!(service
            .store
            .list_index_jobs(Some(namespace), Some("PENDING"), None, None, None, 10)
            .unwrap()
            .iter()
            .any(|job| job.entity_id == "stale-active" && job.op == "UPSERT"));
        assert!(service
            .store
            .list_index_jobs(Some(namespace), Some("PENDING"), None, None, None, 10)
            .unwrap()
            .iter()
            .any(|job| job.entity_id == "stage1-only-stale" && job.op == "UPSERT"));
        let _ = stop_chroma.send(());
        let requests = chroma.await.unwrap();
        let (delete_path, delete_body) = requests
            .iter()
            .find(|(line, _)| line.contains("/delete"))
            .expect("orphaned Stage-2 row is physically removed");
        assert!(delete_path.contains("reconcile-nodes"));
        assert_eq!(delete_body["ids"], json!(["orphan"]));
        assert_eq!(delete_body["where"]["$or"][0]["_kogwistar_namespace"]["$eq"], namespace);
        assert!(requests.iter().any(|(line, body)| {
            line.contains("/get") && body.get("offset").and_then(Value::as_u64) == Some(2)
        }));
    }

    #[tokio::test]
    async fn sqlite_chroma_projection_fence_replaces_stale_delete_after_resurrection() {
        use kogwistar_store::NamedProjectionWrite;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let race_store: Arc<Mutex<Option<kogwistar_store_sqlite::SqliteStore>>> =
            Arc::new(Mutex::new(None));
        let race_store_for_chroma = race_store.clone();
        let (stop_chroma, mut stop_chroma_rx) = tokio::sync::oneshot::channel();
        let chroma = tokio::spawn(async move {
            let mut observed = Vec::new();
            loop {
                let accepted = tokio::select! {
                    _ = &mut stop_chroma_rx => break,
                    accepted = listener.accept() => accepted.unwrap(),
                };
                let (mut stream, _) = accepted;
                let mut bytes = Vec::new();
                let mut chunk = [0_u8; 4096];
                let (split, length) = loop {
                    let count = stream.read(&mut chunk).await.unwrap();
                    assert!(count > 0);
                    bytes.extend_from_slice(&chunk[..count]);
                    if let Some(split) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
                        let head = String::from_utf8_lossy(&bytes[..split]);
                        let length = head
                            .lines()
                            .find_map(|line| {
                                let (name, value) = line.split_once(':')?;
                                name.eq_ignore_ascii_case("content-length")
                                    .then(|| value.trim().parse::<usize>().ok())
                                    .flatten()
                            })
                            .unwrap_or(0);
                        if bytes.len() >= split + 4 + length {
                            break (split, length);
                        }
                    }
                };
                let head = String::from_utf8_lossy(&bytes[..split]);
                let request_line = head.lines().next().unwrap().to_owned();
                let body: Value = serde_json::from_slice(&bytes[split + 4..split + 4 + length])
                    .unwrap();
                if request_line.contains("/upsert") {
                    let store = race_store_for_chroma.lock().unwrap().clone().unwrap();
                    store
                        .append_raw_entity_event(
                            "fenced-tenant",
                            SqliteNewRawEntityEvent {
                                event_id: "resurrected-node-changed-during-upsert".to_owned(),
                                entity_kind: "node".to_owned(),
                                entity_id: "resurrected-node".to_owned(),
                                op: "REPLACE".to_owned(),
                                payload_json: json!({
                                    "id":"resurrected-node",
                                    "document":"newer source during Chroma write"
                                })
                                .to_string(),
                            },
                        )
                        .unwrap();
                    let latest = sqlite_canonical_entity_revision(
                        &store,
                        "fenced-tenant",
                        "node",
                        "resurrected-node",
                    )
                    .unwrap()
                    .unwrap();
                    let job_payload =
                        kogwistar_engine::canonical_revision::canonical_revision_job_payload(
                            &latest,
                        );
                    let fingerprint = serde_json::from_str::<Value>(&job_payload).unwrap()
                        ["source_fingerprint"]
                        .as_str()
                        .unwrap()
                        .to_owned();
                    store
                        .replace_stage1_node_projection(
                            "fenced-tenant",
                            "node:resurrected-node",
                            NamedProjectionWrite {
                                payload: json!({
                                    "document":"newer source during Chroma write",
                                    "metadata":{},
                                    "source_fingerprint":fingerprint
                                })
                                .as_object()
                                .unwrap()
                                .clone(),
                                last_authoritative_seq: latest.revision,
                                last_materialized_seq: latest.revision,
                                projection_schema_version: 1,
                                materialization_status: "ready".to_owned(),
                            },
                        )
                        .unwrap();
                }
                let response = if request_line.contains("/get") && body["ids"].is_array() {
                    json!({
                        "ids":body["ids"],
                        "documents":[null],
                        "metadatas":[{"_kogwistar_namespace":"fenced-tenant"}]
                    })
                    .to_string()
                } else {
                    body["name"].as_str().map_or_else(
                        || json!({}).to_string(),
                        |name| {
                            json!({
                                "id": format!("collection-{name}"),
                                "metadata": {"hnsw:space":"cosine"}
                            })
                            .to_string()
                        },
                    )
                };
                let response = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                    response.len(),
                    response
                );
                stream.write_all(response.as_bytes()).await.unwrap();
                observed.push((request_line, body));
            }
            observed
        });

        let path = std::env::temp_dir().join(format!(
            "kogwistar-sqlite-chroma-stale-delete-{}.sqlite3",
            uuid::Uuid::new_v4()
        ));
        let calls = Arc::new(Mutex::new(Vec::new()));
        let provider = Arc::new(FixedEmbeddingProvider {
            profile: kogwistar_engine::embedding::EmbeddingProfile {
                provider: "test".to_owned(),
                model: "fixed-v1".to_owned(),
                dimension: 2,
                similarity_metric: "cosine".to_owned(),
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
            vector: vec![0.6, 0.8],
            calls: calls.clone(),
            fail: None,
        });
        let service = open_sqlite_application_service(
            &path,
            Some(SqliteChromaGraphConfig {
                endpoint,
                metric: DistanceMetric::Cosine,
                node_collection: "fenced-nodes".to_owned(),
                edge_collection: "fenced-edges".to_owned(),
            }),
        )
        .await
        .unwrap()
        .with_embedding_provider(provider);
        *race_store.lock().unwrap() = Some(service.store.clone());

        let namespace = "fenced-tenant";
        let entity_id = "resurrected-node";
        let deleted = service
            .store
            .append_raw_entity_event(
                namespace,
                SqliteNewRawEntityEvent {
                    event_id: "resurrected-node-delete".to_owned(),
                    entity_kind: "node".to_owned(),
                    entity_id: entity_id.to_owned(),
                    op: "DELETE".to_owned(),
                    payload_json: json!({"id":entity_id}).to_string(),
                },
            )
            .unwrap();
        let deleted_revision = kogwistar_engine::canonical_revision::read_canonical_entity_revision(
            &[EntityEvent {
                namespace: deleted.event.namespace.clone(),
                seq: deleted.event.seq,
                event_id: deleted.event.event_id.clone(),
                entity_kind: deleted.event.entity_kind.clone(),
                entity_id: deleted.event.entity_id.clone(),
                op: deleted.event.op.clone(),
                payload: serde_json::from_str(&deleted.event.payload_json).unwrap(),
            }],
            namespace,
            "node",
            entity_id,
        )
        .unwrap()
        .unwrap();
        let stale_delete_payload =
            kogwistar_engine::canonical_revision::canonical_revision_job_payload(&deleted_revision);

        let active = service
            .store
            .append_raw_entity_event(
                namespace,
                SqliteNewRawEntityEvent {
                    event_id: "resurrected-node-reactivate".to_owned(),
                    entity_kind: "node".to_owned(),
                    entity_id: entity_id.to_owned(),
                    op: "REPLACE".to_owned(),
                    payload_json: json!({"id":entity_id,"document":"current source"}).to_string(),
                },
            )
            .unwrap();
        let active_revision = sqlite_canonical_entity_revision(
            &service.store,
            namespace,
            "node",
            entity_id,
        )
        .unwrap()
        .unwrap();
        let active_job_payload =
            kogwistar_engine::canonical_revision::canonical_revision_job_payload(&active_revision);
        let active_fingerprint = serde_json::from_str::<Value>(&active_job_payload).unwrap()
            ["source_fingerprint"]
            .as_str()
            .unwrap()
            .to_owned();
        service
            .store
            .replace_stage1_node_projection(
                namespace,
                "node:resurrected-node",
                NamedProjectionWrite {
                    payload: json!({
                        "document":"current source",
                        "metadata":{},
                        "source_fingerprint":active_fingerprint
                    })
                    .as_object()
                    .unwrap()
                    .clone(),
                    last_authoritative_seq: active.event.seq,
                    last_materialized_seq: active.event.seq,
                    projection_schema_version: 1,
                    materialization_status: "ready".to_owned(),
                },
            )
            .unwrap();
        service
            .store
            .enqueue_index_job(NewIndexJob {
                job_id: "stale-resurrection-delete".to_owned(),
                namespace: namespace.to_owned(),
                entity_kind: "node".to_owned(),
                entity_id: entity_id.to_owned(),
                index_kind: "node_embedding".to_owned(),
                op: "DELETE".to_owned(),
                payload_json: Some(stale_delete_payload.clone()),
                max_retries: 3,
            })
            .unwrap();

        let summary = service
            .run_embedding_projection_jobs_once(1, 30, Some(namespace))
            .await
            .unwrap();
        assert_eq!(summary.claimed, 1);
        assert_eq!(summary.projected, 0);
        assert!(calls.lock().unwrap().is_empty(), "DELETE must not invoke embeddings");
        let jobs = service
            .store
            .list_index_jobs(Some(namespace), None, Some("node"), Some(entity_id), Some("node_embedding"), 10)
            .unwrap();
        assert_eq!(
            jobs.iter()
                .find(|job| job.job_id == "stale-resurrection-delete")
                .unwrap()
                .status,
            "DONE"
        );
        let replacement = jobs
            .iter()
            .find(|job| job.job_id != "stale-resurrection-delete" && job.status == "PENDING")
            .expect("the active revision must be requeued before the stale delete is acknowledged");
        assert_eq!(replacement.op, "UPSERT");
        assert_eq!(replacement.payload_json.as_deref(), Some(active_job_payload.as_str()));
        let fenced = RevisionFencedChromaProjectionStore {
            queue: service.store.clone(),
            graph: service.chroma_graph.as_ref().unwrap().clone(),
        };
        let deleted_fingerprint = serde_json::from_str::<Value>(&stale_delete_payload).unwrap()
            ["source_fingerprint"]
            .as_str()
            .unwrap()
            .to_owned();
        let stale_upsert = kogwistar_store::GraphProjectionWriteStore::upsert_graph_projection_record_with_outcome(
            &fenced,
            namespace,
            "node",
            GraphRecord {
                id: entity_id.to_owned(),
                document: Some("stale tombstoned content".to_owned()),
                metadata: serde_json::from_value(json!({
                    "_kogwistar_stage2_ready":true,
                    "_kogwistar_source_revision":deleted.event.seq,
                    "_kogwistar_source_fingerprint":deleted_fingerprint
                }))
                .unwrap(),
                embedding: Some(vec![0.0, 1.0]),
            },
        )
        .await
        .unwrap();
        assert_eq!(
            stale_upsert,
            kogwistar_store::ProjectionWriteOutcome::Superseded
        );
        let pending = service
            .store
            .list_index_jobs(
                Some(namespace),
                Some("PENDING"),
                Some("node"),
                Some(entity_id),
                Some("node_embedding"),
                10,
            )
            .unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].op, "UPSERT");
        assert_eq!(pending[0].payload_json.as_deref(), Some(active_job_payload.as_str()));

        let active_fingerprint = serde_json::from_str::<Value>(&active_job_payload).unwrap()
            ["source_fingerprint"]
            .as_str()
            .unwrap()
            .to_owned();
        let raced_upsert = kogwistar_store::GraphProjectionWriteStore::upsert_graph_projection_record_with_outcome(
            &fenced,
            namespace,
            "node",
            GraphRecord {
                id: entity_id.to_owned(),
                document: Some("current before Chroma race".to_owned()),
                metadata: serde_json::from_value(json!({
                    "_kogwistar_stage2_ready":true,
                    "_kogwistar_source_revision":active.event.seq,
                    "_kogwistar_source_fingerprint":active_fingerprint
                }))
                .unwrap(),
                embedding: Some(vec![0.6, 0.8]),
            },
        )
        .await
        .unwrap();
        assert_eq!(
            raced_upsert,
            kogwistar_store::ProjectionWriteOutcome::Superseded
        );
        let latest = sqlite_canonical_entity_revision(
            &service.store,
            namespace,
            "node",
            entity_id,
        )
        .unwrap()
        .unwrap();
        let latest_payload =
            kogwistar_engine::canonical_revision::canonical_revision_job_payload(&latest);
        let pending = service
            .store
            .list_index_jobs(
                Some(namespace),
                Some("PENDING"),
                Some("node"),
                Some(entity_id),
                Some("node_embedding"),
                10,
            )
            .unwrap();
        assert_eq!(pending.len(), 1, "race recovery should coalesce into one job");
        assert_eq!(pending[0].payload_json.as_deref(), Some(latest_payload.as_str()));

        service
            .store
            .append_raw_entity_event(
                namespace,
                SqliteNewRawEntityEvent {
                    event_id: "resurrected-node-delete-after-race".to_owned(),
                    entity_kind: "node".to_owned(),
                    entity_id: entity_id.to_owned(),
                    op: "DELETE".to_owned(),
                    payload_json: json!({"id":entity_id}).to_string(),
                },
            )
            .unwrap();
        let deleted_outcome = kogwistar_store::GraphProjectionWriteStore::delete_graph_projection_record_with_outcome(
            &fenced,
            namespace,
            "node",
            entity_id,
        )
        .await
        .unwrap();
        assert_eq!(
            deleted_outcome,
            kogwistar_store::ProjectionWriteOutcome::Applied
        );
        assert!(service
            .store
            .get_stage1_node_projection(namespace, &format!("node:{entity_id}"))
            .unwrap()
            .is_none());
        let _ = stop_chroma.send(());
        let requests = chroma.await.unwrap();
        assert!(!requests.is_empty());
        assert!(
            requests.iter().any(|(line, _)| line.contains("/upsert")),
            "the simulated race must occur during a real Chroma mutation"
        );
        assert!(requests.iter().any(|(line, _)| line.contains("/delete")));

        drop(service);
        let _ = std::fs::remove_file(path);
    }
