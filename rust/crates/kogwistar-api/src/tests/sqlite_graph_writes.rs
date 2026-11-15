#[tokio::test]
async fn sqlite_run_admission_persists_authenticated_owner_binding() {
    let sqlite_path = std::env::temp_dir().join(format!(
        "kogwistar-sqlite-owner-binding-{}.sqlite",
        uuid::Uuid::new_v4().simple()
    ));
    let auth_path = std::env::temp_dir().join(format!(
        "kogwistar-sqlite-owner-binding-auth-{}.sqlite",
        uuid::Uuid::new_v4().simple()
    ));
    let auth = SqliteAuthStore::open(&auth_path).unwrap();
    auth.resolve_external_identity(ResolveExternalIdentity {
        issuer: "test".to_owned(),
        subject: "sqlite-owner".to_owned(),
        email: "sqlite-owner@example.test".to_owned(),
        display_name: None,
        new_user_id: "sqlite-owner".to_owned(),
        default_role: "rw".to_owned(),
        default_ns: "workflow".to_owned(),
    })
    .await
    .unwrap();
    drop(auth);

    let auth_config = AuthConfig {
        auth_db_url: Some(format!("sqlite:///{}", auth_path.display())),
        ..AuthConfig::default()
    };
    let service = SqliteRunApplicationService::open(&sqlite_path)
        .unwrap()
        .with_owner_auth_config(&auth_config)
        .await
        .unwrap();
    let principal = json!({
        "user_id": "sqlite-owner",
        "sub": "sqlite-owner@example.test",
        "role": "rw",
        "ns": ["workflow"],
        "storage_ns": "workflow",
        "capabilities": ["spawn_process", "workflow.run.write", "write_graph"]
    });
    let response = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/workflow/runs".to_owned(),
            body: serde_json::to_vec(&json!({
                "run_id": "sqlite-owner-bound-run",
                "workflow_id": "sqlite-owner-bound-workflow",
                "conversation_id": "sqlite-owner-bound-conversation",
                "user_id": "sqlite-owner"
            }))
            .unwrap(),
            principal: principal.clone(),
        })
        .await;
    assert_eq!(response.status, StatusCode::ACCEPTED.as_u16());
    let run = service
        .store
        .get_server_run("sqlite-owner-bound-run")
        .unwrap()
        .unwrap();
    assert_eq!(run.user_id.as_deref(), Some("sqlite-owner"));
    let created = service
        .store
        .list_server_run_events("sqlite-owner-bound-run", 0, 1)
        .unwrap()
        .into_iter()
        .find(|event| event.event_type == "run.created")
        .unwrap();
    let payload: Value = serde_json::from_str(&created.payload_json).unwrap();
    assert_eq!(payload["runtime_owner_binding"]["version"], 1);
    assert_eq!(payload["runtime_owner_binding"]["user_id"], "sqlite-owner");
    assert!(
        payload["runtime_owner_binding"]["admitted_capabilities"]
            .as_array()
            .unwrap()
            .iter()
            .any(|capability| capability == "write_graph")
    );
    let revalidated = service
        .resolve_inline_native_owner_principal("sqlite-owner-bound-run", "write_graph")
        .await
        .unwrap();
    assert_eq!(
        authenticated_run_owner_id(&revalidated),
        Some("sqlite-owner")
    );
    assert!(
        principal_strings(&revalidated, "capabilities")
            .iter()
            .any(|capability| capability == "write_graph")
    );

    let mismatch = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/workflow/runs".to_owned(),
            body: serde_json::to_vec(&json!({
                "run_id": "sqlite-owner-mismatch-run",
                "workflow_id": "sqlite-owner-bound-workflow",
                "conversation_id": "sqlite-owner-bound-conversation",
                "user_id": "another-user"
            }))
            .unwrap(),
            principal,
        })
        .await;
    assert_eq!(mismatch.status, StatusCode::FORBIDDEN.as_u16());
    assert!(
        service
            .store
            .get_server_run("sqlite-owner-mismatch-run")
            .unwrap()
            .is_none()
    );

    drop(service);
    remove_test_sqlite(&sqlite_path);
    remove_test_sqlite(&auth_path);
}

#[tokio::test]
async fn sqlite_chroma_public_document_upsert_is_scope_isolated_and_durable() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let chroma_setup = tokio::spawn(async move {
        for _ in 0..2 {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut bytes = Vec::new();
            let (header_end, content_length) = loop {
                let mut chunk = [0_u8; 1024];
                let count = stream.read(&mut chunk).await.unwrap();
                assert!(count > 0);
                bytes.extend_from_slice(&chunk[..count]);
                if let Some(header_end) = bytes.windows(4).position(|part| part == b"\r\n\r\n") {
                    let headers = String::from_utf8_lossy(&bytes[..header_end]);
                    let content_length = headers
                        .lines()
                        .find_map(|line| {
                            let (name, value) = line.split_once(':')?;
                            name.eq_ignore_ascii_case("content-length")
                                .then(|| value.trim().parse::<usize>().ok())
                                .flatten()
                        })
                        .unwrap_or(0);
                    if bytes.len() >= header_end + 4 + content_length {
                        break (header_end, content_length);
                    }
                }
            };
            let request: Value =
                serde_json::from_slice(&bytes[header_end + 4..header_end + 4 + content_length])
                    .unwrap();
            let body = json!({
                "id": format!("collection-{}", request["name"].as_str().unwrap()),
                "metadata": {"hnsw:space": "cosine"}
            })
            .to_string();
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream.write_all(response.as_bytes()).await.unwrap();
        }
    });
    let config = |collection: &str| kogwistar_store_chroma::ChromaStoreConfig {
        endpoint: endpoint.clone(),
        collection: collection.to_owned(),
        ..Default::default()
    };
    let graph = kogwistar_store_chroma::ChromaGraphStore::connect_entity_collections(
        config("source-doc-nodes"),
        config("source-doc-edges"),
    )
    .await
    .unwrap();
    chroma_setup.await.unwrap();

    let path = std::env::temp_dir().join(format!(
        "kogwistar-sqlite-chroma-source-document-{}.sqlite",
        uuid::Uuid::new_v4()
    ));
    let service = SqliteRunApplicationService::open(&path)
        .unwrap()
        .with_chroma_graph(graph);
    assert!(service.supports_http_operation("POST", "/api/document"));
    let body = json!({
        "contract_version": 1,
        "doc_id": "source-doc-1",
        "doc_type": "text",
        "insertion_method": "parser-v1",
        "content": "The authoritative source text.",
        "metadata": {"origin": "test"}
    });
    let principal = json!({
        "sub": "alice",
        "user_id": "alice",
        "role": "rw",
        "ns": ["docs"],
        "storage_ns": "docs",
        "workspace": "workspace-a",
        "graph_space": "knowledge",
        "capabilities": ["write_graph"]
    });
    let response = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/document".to_owned(),
            body: serde_json::to_vec(&body).unwrap(),
            principal: principal.clone(),
        })
        .await;
    assert_eq!(response.status, StatusCode::OK.as_u16());
    let scope = GraphScope {
        namespace: "docs".to_owned(),
        workspace_id: Some("workspace-a".to_owned()),
        graph_space: Some("knowledge".to_owned()),
    };
    let stored = service
        .store
        .source_document_projection(&scope, "source-doc-1")
        .unwrap()
        .unwrap();
    assert_eq!(
        stored.document.as_deref(),
        Some("The authoritative source text.")
    );
    assert_eq!(stored.metadata["origin"], "test");
    assert_eq!(stored.metadata["doc_type"], "text");
    assert!(
        service
            .store
            .source_document_projection(
                &GraphScope {
                    graph_space: Some("wisdom".to_owned()),
                    ..scope.clone()
                },
                "source-doc-1",
            )
            .unwrap()
            .is_none()
    );

    let denied = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/document".to_owned(),
            body: serde_json::to_vec(&json!({
                "contract_version": 1,
                "doc_id": "denied-doc",
                "content": "must not persist"
            }))
            .unwrap(),
            principal: json!({"role": "ro", "ns": ["docs"]}),
        })
        .await;
    assert_eq!(denied.status, StatusCode::FORBIDDEN.as_u16());
    assert!(
        service
            .store
            .source_document_projection(&scope, "denied-doc")
            .unwrap()
            .is_none()
    );

    drop(service);
    let reopened = SqliteStore::open(&path).unwrap();
    assert_eq!(
        reopened
            .source_document_projection(&scope, "source-doc-1")
            .unwrap()
            .unwrap()
            .document
            .as_deref(),
        Some("The authoritative source text.")
    );
    drop(reopened);
    remove_test_sqlite(&path);
}

#[tokio::test]
async fn sqlite_chroma_graph_mutation_commits_canonical_events_and_eventual_jobs() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let (stop_chroma, mut stop_chroma_rx) = tokio::sync::oneshot::channel();
    let fake_chroma = tokio::spawn(async move {
        let mut projected = Vec::new();
        let mut fail_first_upsert = true;
        loop {
            let (mut stream, _) = tokio::select! {
                _ = &mut stop_chroma_rx => break,
                connection = listener.accept() => connection.unwrap(),
            };
            let mut bytes = Vec::new();
            let mut chunk = [0_u8; 2048];
            let (header_end, content_length) = loop {
                let count = stream.read(&mut chunk).await.unwrap();
                assert!(count > 0, "Chroma connection closed before request body");
                bytes.extend_from_slice(&chunk[..count]);
                if let Some(header_end) = bytes.windows(4).position(|window| window == b"\r\n\r\n")
                {
                    let headers = String::from_utf8_lossy(&bytes[..header_end]);
                    let content_length = headers
                        .lines()
                        .find_map(|line| {
                            let (name, value) = line.split_once(':')?;
                            name.eq_ignore_ascii_case("content-length")
                                .then(|| value.trim().parse::<usize>().ok())
                                .flatten()
                        })
                        .unwrap_or(0);
                    if bytes.len() >= header_end + 4 + content_length {
                        break (header_end, content_length);
                    }
                }
            };
            let request: Value =
                serde_json::from_slice(&bytes[header_end + 4..header_end + 4 + content_length])
                    .unwrap();
            let request_line = String::from_utf8_lossy(&bytes[..header_end])
                .lines()
                .next()
                .unwrap_or_default()
                .to_owned();
            let (status, response) = if request_line.contains("/collections HTTP") {
                (
                    "200 OK",
                    json!({
                        "id": format!("collection-{}", request["name"].as_str().unwrap()),
                        "metadata": {"hnsw:space":"cosine"}
                    }),
                )
            } else {
                projected.push((request_line, request));
                if fail_first_upsert
                    && projected.last().unwrap().0.contains("/upsert")
                    && projected.last().unwrap().1["ids"][0] == "node-1"
                {
                    fail_first_upsert = false;
                    (
                        "500 Internal Server Error",
                        json!({"error":"injected Chroma projection failure"}),
                    )
                } else {
                    ("200 OK", json!({}))
                }
            };
            let response = response.to_string();
            let wire = format!(
                "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                response.len(),
                response
            );
            stream.write_all(wire.as_bytes()).await.unwrap();
        }
        projected
    });

    let chroma_config = |collection: &str| kogwistar_store_chroma::ChromaStoreConfig {
        endpoint: endpoint.clone(),
        collection: collection.to_owned(),
        ..Default::default()
    };
    let graph = kogwistar_store_chroma::ChromaGraphStore::connect_entity_collections(
        chroma_config("nodes"),
        chroma_config("edges"),
    )
    .await
    .unwrap();
    let sqlite_path = std::env::temp_dir().join(format!(
        "kogwistar-sqlite-chroma-graph-mutation-{}.sqlite",
        uuid::Uuid::new_v4()
    ));
    let metadata_only = SqliteRunApplicationService::open(&sqlite_path).unwrap();
    assert!(!metadata_only.supports_http_operation("POST", INTERNAL_GRAPH_MUTATIONS_ROUTE));
    drop(metadata_only);
    let embedding_calls = Arc::new(Mutex::new(Vec::new()));
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
        calls: embedding_calls.clone(),
        fail: None,
    });
    let service = SqliteRunApplicationService::open(&sqlite_path)
        .unwrap()
        .with_chroma_graph(graph)
        .with_embedding_provider(provider.clone());
    assert!(service.supports_http_operation("POST", INTERNAL_GRAPH_MUTATIONS_ROUTE));

    let mut batch: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/graph-mutation-batch-v1.json"
    )))
    .unwrap();
    batch["mutations"][0]["record"]["metadata"] = json!({
        "workspace_id":"ws", "graph_space":"facts"
    });
    let python_models: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/python-graph-model-payloads-v1.json"
    )))
    .unwrap();
    let mut node_payload = python_models["cases"][0]["payload"].clone();
    node_payload["id"] = json!("node-1");
    node_payload["embedding"] = json!([1.0, 0.0]);
    batch["mutations"][0]["payload"] = node_payload.clone();
    batch["mutations"][0]["model_envelope"] = json!({
        "schema_version": 1,
        "model_class": "node",
        "payload": node_payload,
    });
    let mut edge_mutation = batch["mutations"][0].clone();
    edge_mutation["table"] = json!("gke_edges");
    edge_mutation["entity_kind"] = json!("edge");
    edge_mutation["event_id"] = json!("event-edge");
    edge_mutation["record"]["id"] = json!("edge-1");
    let mut edge_payload = python_models["cases"][1]["payload"].clone();
    edge_payload["id"] = json!("edge-1");
    edge_payload["embedding"] = json!([1.0, 0.0]);
    edge_payload["source_ids"] = json!(["node-1"]);
    edge_mutation["payload"] = edge_payload.clone();
    edge_mutation["model_envelope"] = json!({
        "schema_version": 1,
        "model_class": "edge",
        "payload": edge_payload,
    });
    batch["mutations"]
        .as_array_mut()
        .unwrap()
        .push(edge_mutation);
    let body = serde_json::to_vec(&batch).unwrap();
    let principal = json!({
        "sub":"alice", "user_id":"alice", "role":"rw", "ns":["tenant-a"],
        "storage_ns":"tenant-a", "workspace":"ws", "graph_space":"facts",
        "capabilities":["write_graph"]
    });
    let denied = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: INTERNAL_GRAPH_MUTATIONS_ROUTE.to_owned(),
            body: body.clone(),
            principal: json!({"sub":"alice", "role":"ro", "ns":["tenant-a"], "storage_ns":"tenant-a", "workspace":"ws", "graph_space":"facts"}),
        })
        .await;
    assert_eq!(denied.status, StatusCode::FORBIDDEN.as_u16());
    assert!(
        service
            .store
            .replay_raw_events("tenant-a", 0, 10)
            .unwrap()
            .is_empty()
    );

    for (scope_key, value) in [
        ("namespace", "tenant-b"),
        ("workspace_id", "workspace-b"),
        ("graph_space", "other"),
    ] {
        let mut out_of_scope_batch = batch.clone();
        for mutation in out_of_scope_batch["mutations"].as_array_mut().unwrap() {
            mutation["scope"][scope_key] = json!(value);
            if scope_key != "namespace" {
                let metadata_key = if scope_key == "workspace_id" {
                    "workspace_id"
                } else {
                    "graph_space"
                };
                mutation["record"]["metadata"][metadata_key] = json!(value);
                mutation["payload"]["metadata"][metadata_key] = json!(value);
                mutation["model_envelope"]["payload"]["metadata"][metadata_key] =
                    json!(value);
            }
        }
        let out_of_scope = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: INTERNAL_GRAPH_MUTATIONS_ROUTE.to_owned(),
                body: serde_json::to_vec(&out_of_scope_batch).unwrap(),
                principal: principal.clone(),
            })
            .await;
        assert_eq!(out_of_scope.status, StatusCode::FORBIDDEN.as_u16(), "{scope_key}");
        assert!(service.store.replay_raw_events("tenant-a", 0, 10).unwrap().is_empty());
    }

    let request = || ApiEffectRequest {
        contract_version: 1,
        method: "POST".to_owned(),
        path_and_query: INTERNAL_GRAPH_MUTATIONS_ROUTE.to_owned(),
        body: body.clone(),
        principal: principal.clone(),
    };
    let first = service.execute(request()).await;
    assert_eq!(
        first.status,
        StatusCode::OK.as_u16(),
        "{}",
        String::from_utf8_lossy(&first.body)
    );
    let first_body: Value = serde_json::from_slice(&first.body).unwrap();
    assert_eq!(first_body["projection_consistency"], "eventual");
    assert_eq!(first_body["results"].as_array().unwrap().len(), 2);
    assert_eq!(first_body["results"][0]["inserted"], true);

    let retry = service.execute(request()).await;
    assert_eq!(
        retry.status,
        StatusCode::OK.as_u16(),
        "{}",
        String::from_utf8_lossy(&retry.body)
    );
    let retry_body: Value = serde_json::from_slice(&retry.body).unwrap();
    assert_eq!(retry_body["results"][0]["mutated"], false);
    let canonical_events = service.store.replay_raw_events("tenant-a", 0, 10).unwrap();
    assert_eq!(canonical_events.len(), 2);
    for event in &canonical_events {
        let payload: Value = serde_json::from_str(&event.payload_json).unwrap();
        assert!(payload["metadata"].get("_class_name").is_none());
        assert_eq!(payload["metadata"]["workspace_id"], "ws");
        assert_eq!(payload["metadata"]["graph_space"], "facts");
    }
    let jobs = service
        .store
        .list_index_jobs(Some("tenant-a"), Some("PENDING"), None, None, None, 10)
        .unwrap();
    assert_eq!(jobs.len(), 2);
    assert!(jobs.iter().all(|job| job.index_kind == "node_embedding"));
    assert_eq!(
        jobs.iter()
            .map(|job| job.entity_kind.as_str())
            .collect::<std::collections::BTreeSet<_>>(),
        ["edge", "node"].into_iter().collect()
    );

    // Canonical SQLite events and jobs must survive a service restart before
    // the eventual Chroma projection is attempted.
    drop(service);
    let graph = kogwistar_store_chroma::ChromaGraphStore::connect_entity_collections(
        chroma_config("nodes"),
        chroma_config("edges"),
    )
    .await
    .unwrap();
    let service = SqliteRunApplicationService::open(&sqlite_path)
        .unwrap()
        .with_chroma_graph(graph)
        .with_embedding_provider(provider.clone());
    let recovered_events = service.store.replay_raw_events("tenant-a", 0, 10).unwrap();
    assert_eq!(recovered_events.len(), 2);
    for event in &recovered_events {
        let payload: Value = serde_json::from_str(&event.payload_json).unwrap();
        assert!(payload["metadata"].get("_class_name").is_none());
    }
    let recovered_jobs = service
        .store
        .list_index_jobs(Some("tenant-a"), Some("PENDING"), None, None, None, 10)
        .unwrap();
    assert_eq!(recovered_jobs.len(), 2);
    for job in &recovered_jobs {
        let job_payload: Value = serde_json::from_str(job.payload_json.as_deref().unwrap()).unwrap();
        let expected_class = if job.entity_kind == "node" { "Node" } else { "Edge" };
        assert_eq!(
            job_payload["projection_metadata"]["_class_name"],
            expected_class
        );
    }
    // Simulate lost derived Stage-1 rows: durable queue metadata must restore
    // subtype identity without adding it to canonical event payloads.
    service
        .store
        .clear_stage1_node_projection("tenant-a", "node:node-1")
        .unwrap();
    service
        .store
        .clear_stage1_node_projection("tenant-a", "edge:edge-1")
        .unwrap();

    let failed_projection = service
        .run_embedding_projection_jobs_once(10, 30, Some("tenant-a"))
        .await
        .unwrap();
    assert_eq!(failed_projection.claimed, 2);
    assert_eq!(failed_projection.projected, 1);
    assert_eq!(failed_projection.retried, 1);
    assert_eq!(
        service
            .store
            .replay_raw_events("tenant-a", 0, 10)
            .unwrap()
            .len(),
        2
    );
    let after_failure = service
        .store
        .list_index_jobs(Some("tenant-a"), None, None, None, None, 10)
        .unwrap();
    assert_eq!(
        after_failure
            .iter()
            .filter(|job| job.status == "DONE")
            .count(),
        1
    );
    assert_eq!(
        after_failure
            .iter()
            .filter(|job| job.status == "PENDING")
            .count(),
        1
    );

    // A Chroma failure cannot undo canonical SQLite state. Reopen the service
    // and retry only the durable failed projection job.
    drop(service);
    tokio::time::sleep(std::time::Duration::from_millis(2200)).await;
    let graph = kogwistar_store_chroma::ChromaGraphStore::connect_entity_collections(
        chroma_config("nodes"),
        chroma_config("edges"),
    )
    .await
    .unwrap();
    let service = SqliteRunApplicationService::open(&sqlite_path)
        .unwrap()
        .with_chroma_graph(graph)
        .with_embedding_provider(provider.clone());
    let trigger_connection = rusqlite::Connection::open(&sqlite_path).unwrap();
    trigger_connection
        .execute_batch(
            "CREATE TRIGGER reject_node_projection_ack
             BEFORE UPDATE OF status ON index_jobs
             WHEN NEW.entity_id = 'node-1' AND NEW.status = 'DONE'
             BEGIN SELECT RAISE(ABORT, 'injected projection acknowledgement failure'); END;",
        )
        .unwrap();
    let ack_failure = service
        .run_embedding_projection_jobs_once(10, 1, Some("tenant-a"))
        .await
        .unwrap_err();
    assert!(ack_failure.contains("injected projection acknowledgement failure"));
    let after_ack_failure = service
        .store
        .list_index_jobs(Some("tenant-a"), None, None, None, None, 10)
        .unwrap();
    let node_job = after_ack_failure
        .iter()
        .find(|job| job.entity_id == "node-1")
        .unwrap();
    assert_eq!(node_job.status, "DOING");
    assert!(node_job.accepted_result_json.is_some());
    assert_eq!(
        after_ack_failure
            .iter()
            .find(|job| job.entity_id == "edge-1")
            .unwrap()
            .status,
        "DONE"
    );

    drop(service);
    trigger_connection
        .execute_batch("DROP TRIGGER reject_node_projection_ack")
        .unwrap();
    drop(trigger_connection);
    tokio::time::sleep(std::time::Duration::from_millis(2200)).await;
    let graph = kogwistar_store_chroma::ChromaGraphStore::connect_entity_collections(
        chroma_config("nodes"),
        chroma_config("edges"),
    )
    .await
    .unwrap();
    let service = SqliteRunApplicationService::open(&sqlite_path)
        .unwrap()
        .with_chroma_graph(graph)
        .with_embedding_provider(provider);
    let final_projection = service
        .run_embedding_projection_jobs_once(10, 30, Some("tenant-a"))
        .await
        .unwrap();
    assert_eq!(final_projection.claimed, 1);
    assert_eq!(final_projection.projected, 0);
    assert_eq!(final_projection.reused, 1);
    assert_eq!(
        service
            .store
            .replay_raw_events("tenant-a", 0, 10)
            .unwrap()
            .len(),
        2
    );
    let projection_jobs = service
        .store
        .list_index_jobs(Some("tenant-a"), None, None, None, None, 10)
        .unwrap();
    assert_eq!(projection_jobs.len(), 2);
    assert!(projection_jobs.iter().all(|job| job.status == "DONE"));
    assert_eq!(embedding_calls.lock().unwrap().len(), 1);
    let _ = stop_chroma.send(());
    let chroma_upserts = fake_chroma.await.unwrap();
    assert_eq!(chroma_upserts.len(), 4);
    let node_attempts = chroma_upserts
        .iter()
        .filter(|(_, body)| body["ids"][0] == "node-1")
        .collect::<Vec<_>>();
    let edge_attempts = chroma_upserts
        .iter()
        .filter(|(_, body)| body["ids"][0] == "edge-1")
        .collect::<Vec<_>>();
    assert_eq!(node_attempts.len(), 3);
    assert_eq!(edge_attempts.len(), 1);
    assert_eq!(node_attempts[0].1, node_attempts[1].1);
    assert_eq!(node_attempts[1].1, node_attempts[2].1);
    let node_upsert = chroma_upserts
        .iter()
        .find(|(_, body)| body["ids"][0] == "node-1")
        .unwrap();
    assert!(node_upsert.0.contains("collection-nodes"));
    assert_eq!(node_upsert.1["embeddings"][0], json!([0.6, 0.8]));
    assert_eq!(node_upsert.1["metadatas"][0]["namespace"], "tenant-a");
    assert_eq!(node_upsert.1["metadatas"][0]["workspace_id"], "ws");
    assert_eq!(node_upsert.1["metadatas"][0]["graph_space"], "facts");
    assert_eq!(node_upsert.1["metadatas"][0]["_class_name"], "Node");
    let edge_upsert = chroma_upserts
        .iter()
        .find(|(_, body)| body["ids"][0] == "edge-1")
        .unwrap();
    assert!(edge_upsert.0.contains("collection-edges"));
    assert_eq!(edge_upsert.1["embeddings"][0], json!([0.6, 0.8]));
    assert_eq!(edge_upsert.1["metadatas"][0]["namespace"], "tenant-a");
    assert_eq!(edge_upsert.1["metadatas"][0]["workspace_id"], "ws");
    assert_eq!(edge_upsert.1["metadatas"][0]["graph_space"], "facts");
    assert_eq!(edge_upsert.1["metadatas"][0]["_class_name"], "Edge");

    drop(service);
    remove_test_sqlite(&sqlite_path);
}

#[tokio::test]
async fn sqlite_chroma_vector_query_enforces_source_span_acl() {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let vector_query_count = Arc::new(AtomicUsize::new(0));
    let vector_query_count_for_server = Arc::clone(&vector_query_count);
    let (stop_chroma, mut stop_chroma_rx) = tokio::sync::oneshot::channel();
    let fake_chroma = tokio::spawn(async move {
        loop {
            let (mut stream, _) = tokio::select! {
                _ = &mut stop_chroma_rx => break,
                connection = listener.accept() => connection.unwrap(),
            };
            let mut bytes = Vec::new();
            let mut chunk = [0_u8; 2048];
            let (header_end, content_length) = loop {
                let count = stream.read(&mut chunk).await.unwrap();
                assert!(count > 0, "Chroma connection closed before request body");
                bytes.extend_from_slice(&chunk[..count]);
                if let Some(header_end) = bytes.windows(4).position(|window| window == b"\r\n\r\n")
                {
                    let headers = String::from_utf8_lossy(&bytes[..header_end]);
                    let content_length = headers
                        .lines()
                        .find_map(|line| {
                            let (name, value) = line.split_once(':')?;
                            name.eq_ignore_ascii_case("content-length")
                                .then(|| value.trim().parse::<usize>().ok())
                                .flatten()
                        })
                        .unwrap_or(0);
                    if bytes.len() >= header_end + 4 + content_length {
                        break (header_end, content_length);
                    }
                }
            };
            let request: Value =
                serde_json::from_slice(&bytes[header_end + 4..header_end + 4 + content_length])
                    .unwrap();
            let request_line = String::from_utf8_lossy(&bytes[..header_end])
                .lines()
                .next()
                .unwrap_or_default()
                .to_owned();
            let payload = if request_line.contains("/collections HTTP") {
                json!({
                    "id": format!("collection-{}", request["name"].as_str().unwrap()),
                    "metadata": {"hnsw:space":"cosine"}
                })
            } else if request_line.contains("collection-source-acl-nodes/query") {
                vector_query_count_for_server.fetch_add(1, Ordering::SeqCst);
                assert_eq!(request["n_results"], 1);
                let query_filter = request["where"].to_string();
                assert!(
                    query_filter.contains("\"entity_type\":{\"$ne\":\"acl_record\"}"),
                    "ACL helper exclusion must be pushed into vector top-k: {query_filter}"
                );
                assert!(
                    query_filter.contains("\"lifecycle_status\":{\"$ne\":\"tombstoned\"}"),
                    "tombstones must be excluded before vector top-k: {query_filter}"
                );
                assert!(
                    query_filter.contains("\"lifecycle_status\":{\"$ne\":\"deleted\"}"),
                    "legacy deleted records must be excluded before vector top-k: {query_filter}"
                );
                let revoked_query = request["where"].to_string().contains("node-revoked");
                let (node_id, original) = if revoked_query {
                    (
                        "node-revoked",
                        json!({"entity_type":"node","label":"node-revoked","mentions":[]}),
                    )
                } else {
                    (
                        "node-source-span-private",
                        json!({
                            "entity_type":"node",
                            "label":"node-source-span-private",
                            "mentions":[{"spans":[{}]}]
                        }),
                    )
                };
                json!({
                    "ids":[[node_id]],
                    "documents":[[node_id]],
                    "metadatas":[[{
                        "_kogwistar_namespace":"docs",
                        "_kogwistar_original_metadata":original.to_string()
                    }]],
                    "embeddings":[[[1.0,0.0]]],
                    "distances":[[0.0]]
                })
            } else if request_line.contains("collection-source-acl-edges/query") {
                vector_query_count_for_server.fetch_add(1, Ordering::SeqCst);
                assert_eq!(request["n_results"], 1);
                let original = json!({
                    "entity_type":"edge",
                    "source_ids":["private-endpoint"],
                    "target_ids":["public-endpoint"],
                    "source_edge_ids":[],
                    "target_edge_ids":[],
                    "mentions":[]
                });
                json!({
                    "ids":[["edge-with-private-endpoint"]],
                    "documents":[["edge-with-private-endpoint"]],
                    "metadatas":[[{
                        "_kogwistar_namespace":"docs",
                        "_kogwistar_original_metadata":original.to_string()
                    }]],
                    "embeddings":[[[1.0,0.0]]],
                    "distances":[[0.0]]
                })
            } else if request_line.contains("collection-source-acl-nodes/get") {
                if let Some(ids) = request.get("ids").and_then(Value::as_array) {
                    let nodes = [
                        (
                            "private-endpoint",
                            json!({
                                "entity_type":"node",
                                "label":"private-endpoint",
                                "mentions":[]
                            }),
                        ),
                        (
                            "public-endpoint",
                            json!({
                                "entity_type":"node",
                                "label":"public-endpoint",
                                "mentions":[]
                            }),
                        ),
                    ];
                    let matching = nodes
                        .iter()
                        .filter(|(id, _)| ids.iter().any(|requested| requested == id))
                        .collect::<Vec<_>>();
                    json!({
                        "ids":matching.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
                        "documents":matching.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
                        "metadatas":matching.iter().map(|(_, original)| json!({
                            "_kogwistar_namespace":"docs",
                            "_kogwistar_original_metadata":original.to_string()
                        })).collect::<Vec<_>>(),
                        "embeddings":vec![Value::Null; matching.len()]
                    })
                } else {
                let records = [
                    (
                        "acl-source-node",
                        json!({
                            "entity_type":"acl_record",
                            "acl_truth_graph":"knowledge",
                            "acl_target_entity_id":"node-source-span-private",
                            "acl_target_grain":"node",
                            "acl_version":1,
                            "acl_mode":"public"
                        }),
                    ),
                    (
                        "acl-source-grounding",
                        json!({
                            "entity_type":"acl_record",
                            "acl_truth_graph":"knowledge",
                            "acl_target_entity_id":"node-source-span-private",
                            "acl_target_grain":"grounding",
                            "acl_target_item_id":"node-source-span-private::mention::0",
                            "acl_version":1,
                            "acl_mode":"public"
                        }),
                    ),
                    (
                        "acl-source-span",
                        json!({
                            "entity_type":"acl_record",
                            "acl_truth_graph":"knowledge",
                            "acl_target_entity_id":"node-source-span-private",
                            "acl_target_grain":"span",
                            "acl_target_item_id":"node-source-span-private::mention::0::span::0",
                            "acl_version":1,
                            "acl_mode":"private",
                            "owner_id":"bob"
                        }),
                    ),
                    (
                        "acl-node-revoked",
                        json!({
                            "entity_type":"acl_record",
                            "acl_truth_graph":"knowledge",
                            "acl_target_entity_id":"node-revoked",
                            "acl_target_grain":"node",
                            "acl_version":1,
                            "acl_mode":"public"
                        }),
                    ),
                    (
                        "acl-edge-public",
                        json!({
                            "entity_type":"acl_record",
                            "acl_truth_graph":"knowledge",
                            "acl_target_entity_id":"edge-with-private-endpoint",
                            "acl_target_grain":"edge",
                            "acl_version":1,
                            "acl_mode":"public"
                        }),
                    ),
                    (
                        "acl-private-endpoint",
                        json!({
                            "entity_type":"acl_record",
                            "acl_truth_graph":"knowledge",
                            "acl_target_entity_id":"private-endpoint",
                            "acl_target_grain":"node",
                            "acl_version":1,
                            "acl_mode":"private",
                            "owner_id":"bob"
                        }),
                    ),
                    (
                        "acl-public-endpoint",
                        json!({
                            "entity_type":"acl_record",
                            "acl_truth_graph":"knowledge",
                            "acl_target_entity_id":"public-endpoint",
                            "acl_target_grain":"node",
                            "acl_version":1,
                            "acl_mode":"public"
                        }),
                    ),
                ];
                json!({
                    "ids": records.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
                    "documents": vec![Value::Null; records.len()],
                    "metadatas": records.iter().map(|(_, original)| json!({
                        "_kogwistar_namespace":"docs",
                        "_kogwistar_original_metadata":original.to_string()
                    })).collect::<Vec<_>>(),
                    "embeddings": vec![Value::Null; records.len()]
                })
                }
            } else {
                json!({"ids":[],"documents":[],"metadatas":[],"embeddings":[]})
            };
            let response = payload.to_string();
            let wire = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                response.len(),
                response
            );
            stream.write_all(wire.as_bytes()).await.unwrap();
        }
    });

    let chroma_config = |collection: &str| kogwistar_store_chroma::ChromaStoreConfig {
        endpoint: endpoint.clone(),
        collection: collection.to_owned(),
        ..Default::default()
    };
    let graph = kogwistar_store_chroma::ChromaGraphStore::connect_entity_collections(
        chroma_config("source-acl-nodes"),
        chroma_config("source-acl-edges"),
    )
    .await
    .unwrap();
    let sqlite_path = std::env::temp_dir().join(format!(
        "kogwistar-sqlite-chroma-source-acl-{}.sqlite",
        uuid::Uuid::new_v4().simple()
    ));
    let service = SqliteRunApplicationService::open(&sqlite_path)
        .unwrap()
        .with_chroma_graph(graph);
    let mut canonical_acl_mutations = Vec::new();
    for (id, entity_id, grain, item_id, mode, owner_id) in [
        (
            "acl-source-node",
            "node-source-span-private",
            "node",
            None,
            "public",
            None,
        ),
        (
            "acl-source-grounding",
            "node-source-span-private",
            "grounding",
            Some("node-source-span-private::mention::0"),
            "public",
            None,
        ),
        (
            "acl-source-span",
            "node-source-span-private",
            "span",
            Some("node-source-span-private::mention::0::span::0"),
            "private",
            Some("bob"),
        ),
        (
            "acl-edge-public",
            "edge-with-private-endpoint",
            "edge",
            None,
            "public",
            None,
        ),
        (
            "acl-private-endpoint",
            "private-endpoint",
            "node",
            None,
            "private",
            Some("bob"),
        ),
        (
            "acl-public-endpoint",
            "public-endpoint",
            "node",
            None,
            "public",
            None,
        ),
        (
            "acl-node-revoked",
            "node-revoked",
            "node",
            None,
            "public",
            None,
        ),
    ] {
        let mut metadata = json!({
            "entity_type":"acl_record",
            "acl_truth_graph":"knowledge",
            "acl_target_entity_id":entity_id,
            "acl_target_grain":grain,
            "acl_version":1,
            "acl_mode":mode
        });
        if let Some(item_id) = item_id {
            metadata["acl_target_item_id"] = json!(item_id);
        }
        if let Some(owner_id) = owner_id {
            metadata["owner_id"] = json!(owner_id);
        }
        canonical_acl_mutations.push(json!({
            "scope":{"namespace":"docs","workspace_id":null,"graph_space":null},
            "table":"gke_nodes",
            "entity_kind":"node",
            "event_id":format!("seed-{id}"),
            "op":"ADD",
            "payload":{"id":id,"document":null,"metadata":metadata.clone()},
            "record":{"id":id,"document":null,"metadata":metadata,"embedding":null},
            "embedding_dim":0
        }));
    }
    let seed_acl = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/internal/graph/mutations".to_owned(),
            body: serde_json::to_vec(&json!({
                "contract_version":1,
                "atomic":true,
                "mutations":canonical_acl_mutations,
                "index_jobs":[]
            }))
            .unwrap(),
            principal: json!({
                "sub":"owner",
                "role":"rw",
                "storage_ns":"docs",
                "capabilities":["write_graph"]
            }),
        })
        .await;
    assert_eq!(seed_acl.status, StatusCode::OK.as_u16());
    let response = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/internal/graph/vector-query".to_owned(),
            body: serde_json::to_vec(&json!({
                "entity_kind":"node",
                "query_embeddings":[[1.0,0.0]],
                "metric":"cosine",
                "truth_graph":"knowledge",
                "where":{"label":"node-source-span-private"},
                "n_results":1,
                "include":["distances"]
            }))
            .unwrap(),
            principal: json!({
                "sub":"alice",
                "role":"ro",
                "storage_ns":"docs",
                "capabilities":["read_graph"]
            }),
        })
        .await;
    assert_eq!(response.status, StatusCode::OK.as_u16());
    let result: Value = serde_json::from_slice(&response.body).unwrap();
    assert_eq!(result["ids"], json!([[]]));
    assert_eq!(vector_query_count.load(Ordering::SeqCst), 1);

    let edge_response = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/internal/graph/vector-query".to_owned(),
            body: serde_json::to_vec(&json!({
                "entity_kind":"edge",
                "query_embeddings":[[1.0,0.0]],
                "metric":"cosine",
                "truth_graph":"knowledge",
                "n_results":1,
                "include":["distances"]
            }))
            .unwrap(),
            principal: json!({
                "sub":"alice",
                "role":"ro",
                "storage_ns":"docs",
                "capabilities":["read_graph"]
            }),
        })
        .await;
    assert_eq!(edge_response.status, StatusCode::OK.as_u16());
    let edge_result: Value = serde_json::from_slice(&edge_response.body).unwrap();
    assert_eq!(edge_result["ids"], json!([[]]));
    assert_eq!(vector_query_count.load(Ordering::SeqCst), 2);

    let revoked_query = || ApiEffectRequest {
        contract_version: 1,
        method: "POST".to_owned(),
        path_and_query: "/internal/graph/vector-query".to_owned(),
        body: serde_json::to_vec(&json!({
            "entity_kind":"node",
            "query_embeddings":[[1.0,0.0]],
            "metric":"cosine",
            "truth_graph":"knowledge",
            "where":{"label":"node-revoked"},
            "n_results":1,
            "include":["distances"]
        }))
        .unwrap(),
        principal: json!({
            "sub":"alice",
            "role":"ro",
            "storage_ns":"docs",
            "capabilities":["read_graph"]
        }),
    };
    let public_before_revoke = service.execute(revoked_query()).await;
    assert_eq!(public_before_revoke.status, StatusCode::OK.as_u16());
    let public_before_revoke: Value = serde_json::from_slice(&public_before_revoke.body).unwrap();
    assert_eq!(public_before_revoke["ids"], json!([["node-revoked"]]));
    assert_eq!(vector_query_count.load(Ordering::SeqCst), 3);

    let revoke = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/internal/graph/mutations".to_owned(),
            body: serde_json::to_vec(&json!({
                "contract_version":1,
                "atomic":true,
                "index_jobs":[],
                "mutations":[{
                    "scope":{"namespace":"docs","workspace_id":null,"graph_space":null},
                    "table":"gke_nodes",
                    "entity_kind":"node",
                    "event_id":"revoke-acl-node-revoked-v2",
                    "op":"REPLACE",
                    "payload":{
                        "id":"acl-node-revoked",
                        "document":null,
                        "metadata":{
                            "entity_type":"acl_record",
                            "acl_truth_graph":"knowledge",
                            "acl_target_entity_id":"node-revoked",
                            "acl_target_grain":"node",
                            "acl_version":2,
                            "acl_mode":"private",
                            "owner_id":"bob"
                        }
                    },
                    "record":{
                        "id":"acl-node-revoked",
                        "document":null,
                        "metadata":{
                            "entity_type":"acl_record",
                            "acl_truth_graph":"knowledge",
                            "acl_target_entity_id":"node-revoked",
                            "acl_target_grain":"node",
                            "acl_version":2,
                            "acl_mode":"private",
                            "owner_id":"bob"
                        },
                        "embedding":null
                    },
                    "embedding_dim":0
                }]
            }))
            .unwrap(),
            principal: json!({
                "sub":"owner",
                "role":"rw",
                "storage_ns":"docs",
                "capabilities":["write_graph"]
            }),
        })
        .await;
    assert_eq!(revoke.status, StatusCode::OK.as_u16());

    let hidden_after_revoke = service.execute(revoked_query()).await;
    assert_eq!(hidden_after_revoke.status, StatusCode::OK.as_u16());
    let hidden_after_revoke: Value = serde_json::from_slice(&hidden_after_revoke.body).unwrap();
    assert_eq!(hidden_after_revoke["ids"], json!([[]]));
    assert_eq!(vector_query_count.load(Ordering::SeqCst), 4);

    drop(service);
    let _ = stop_chroma.send(());
    fake_chroma.await.unwrap();
    remove_test_sqlite(&sqlite_path);
}
