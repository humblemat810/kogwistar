    #[test]
    fn sqlite_chroma_stage1_recovery_projects_edge_and_hyperedge_fields() {
        let model_fixture: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/python-graph-model-payloads-v1.json"
        ))
        .unwrap();
        let recovery_fixture: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/stage1-recovery-document-v2.json"
        ))
        .unwrap();
        let mut payload = model_fixture["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|case| case["kind"] == "edge")
            .unwrap()["payload"]
            .clone();
        let entity_id = recovery_fixture["edge_entity_id"].as_str().unwrap();
        payload["id"] = json!(entity_id);
        let revision = kogwistar_engine::canonical_revision::CanonicalEntityRevision {
            namespace: "tenant-a".to_owned(),
            entity_kind: "edge".to_owned(),
            entity_id: entity_id.to_owned(),
            revision: 7,
            state: "active".to_owned(),
            payload,
        };

        let recovered = RevisionFencedChromaProjectionStore::restore_stage1_from_revision(
            &revision,
            "fixture-fingerprint",
            None,
        )
        .unwrap();
        assert_eq!(
            recovered.payload["document"].as_str(),
            recovery_fixture["edge_expected_document"].as_str()
        );
        let metadata = &recovered.payload["metadata"];
        assert_eq!(metadata["namespace"], "tenant-a");
        assert_eq!(metadata["node_endpoint_count"], 2);
        assert_eq!(metadata["edge_endpoint_count"], 0);
        assert_eq!(metadata["source_ids"], "[\"parity-node\"]");
        assert!(metadata.get("references").and_then(Value::as_str).is_some());
        assert_eq!(recovered.projection_schema_version, 2);
    }

    #[tokio::test]
    async fn sqlite_chroma_embedding_worker_projects_revision_fenced_stage1_document() {
        use kogwistar_store::NamedProjectionWrite;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let (stop_chroma, mut stop_chroma_rx) = tokio::sync::oneshot::channel();
        let chroma = tokio::spawn(async move {
            let mut observed = Vec::new();
            let mut applied = std::collections::BTreeMap::new();
            let mut fail_first_upsert_response = true;
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
                let (status, response) = if request_line.contains("/collections HTTP") {
                    (
                        "200 OK",
                        json!({
                        "id": format!("collection-{}", body["name"].as_str().unwrap()),
                        "metadata": {"hnsw:space":"cosine"}
                        }),
                    )
                } else if request_line.contains("/upsert") {
                    if let Some(ids) = body["ids"].as_array() {
                        for (index, id) in ids.iter().filter_map(Value::as_str).enumerate() {
                            applied.insert(
                                id.to_owned(),
                                json!({
                                    "document": body["documents"][index].clone(),
                                    "metadata": body["metadatas"][index].clone(),
                                    "embedding": body["embeddings"][index].clone()
                                }),
                            );
                        }
                    }
                    if fail_first_upsert_response {
                        fail_first_upsert_response = false;
                        // Model a lost acknowledgement after Chroma applied the idempotent upsert.
                        (
                            "500 Internal Server Error",
                            json!({"error":"injected response loss after upsert"}),
                        )
                    } else {
                        ("200 OK", json!({}))
                    }
                } else {
                    ("200 OK", json!({}))
                };
                let response = response.to_string();
                let response = format!(
                    "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                    response.len(),
                    response
                );
                stream.write_all(response.as_bytes()).await.unwrap();
                observed.push((request_line, body));
            }
            (observed, applied)
        });

        let path = std::env::temp_dir().join(format!(
            "kogwistar-sqlite-chroma-embedding-{}.sqlite3",
            uuid::Uuid::new_v4()
        ));
        let calls = Arc::new(Mutex::new(Vec::new()));
        let provider_failure = Arc::new(std::sync::atomic::AtomicBool::new(true));
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
            fail: Some(provider_failure.clone()),
        });
        let service = open_sqlite_application_service(
            &path,
            Some(SqliteChromaGraphConfig {
                endpoint: endpoint.clone(),
                metric: DistanceMetric::Cosine,
                node_collection: "worker-nodes".to_owned(),
                edge_collection: "worker-edges".to_owned(),
            }),
        )
        .await
        .unwrap()
        .with_embedding_provider(provider.clone());

        let namespace = "worker-tenant";
        let recovery_fixture: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/stage1-recovery-document-v2.json"
        ))
        .unwrap();
        let entity_id = recovery_fixture["entity_id"].as_str().unwrap();
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/python-graph-model-payloads-v1.json"
        ))
        .unwrap();
        let mut entity_payload = fixture["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|case| case["kind"] == "node")
            .unwrap()["payload"]
            .clone();
        entity_payload["id"] = json!(entity_id);
        entity_payload["metadata"]["namespace"] = Value::Null;
        let mut expected_document_payload = entity_payload.clone();
        expected_document_payload.as_object_mut().unwrap().remove("embedding");
        expected_document_payload.as_object_mut().unwrap().remove("metadata");
        let document = kogwistar_contracts::canonical_json(&expected_document_payload, false);
        assert_eq!(document, recovery_fixture["expected_document"].as_str().unwrap());
        let appended = service
            .store
            .append_raw_entity_event(
                namespace,
                SqliteNewRawEntityEvent {
                    event_id: "worker-node-add".to_owned(),
                    entity_kind: "node".to_owned(),
                    entity_id: entity_id.to_owned(),
                    op: "ADD".to_owned(),
                    payload_json: entity_payload.to_string(),
                },
            )
            .unwrap();
        let revision = sqlite_canonical_entity_revision(
            &service.store,
            namespace,
            "node",
            entity_id,
        )
        .unwrap()
        .unwrap();
        let job_payload = kogwistar_engine::canonical_revision::canonical_revision_job_payload(
            &revision,
        );
        let fingerprint = serde_json::from_str::<Value>(&job_payload).unwrap()
            ["source_fingerprint"]
            .as_str()
            .unwrap()
            .to_owned();
        service
            .store
            .enqueue_index_job(NewIndexJob {
                job_id: "worker-node-embedding".to_owned(),
                namespace: namespace.to_owned(),
                entity_kind: "node".to_owned(),
                entity_id: entity_id.to_owned(),
                index_kind: "node_embedding".to_owned(),
                op: "UPSERT".to_owned(),
                payload_json: Some(job_payload),
                max_retries: 3,
            })
            .unwrap();
        let second_id = "worker-batch-node";
        let mut second_payload = entity_payload.clone();
        second_payload["id"] = json!(second_id);
        let second_document_payload = {
            let mut value = second_payload.clone();
            value.as_object_mut().unwrap().remove("embedding");
            value.as_object_mut().unwrap().remove("metadata");
            value
        };
        let second_document =
            kogwistar_contracts::canonical_json(&second_document_payload, false);
        service
            .store
            .append_raw_entity_event(
                namespace,
                SqliteNewRawEntityEvent {
                    event_id: "worker-batch-node-add".to_owned(),
                    entity_kind: "node".to_owned(),
                    entity_id: second_id.to_owned(),
                    op: "ADD".to_owned(),
                    payload_json: second_payload.to_string(),
                },
            )
            .unwrap();
        let second_revision = sqlite_canonical_entity_revision(
            &service.store,
            namespace,
            "node",
            second_id,
        )
        .unwrap()
        .unwrap();
        let second_job_payload =
            kogwistar_engine::canonical_revision::canonical_revision_job_payload(
                &second_revision,
            );
        service
            .store
            .enqueue_index_job(NewIndexJob {
                job_id: "worker-batch-node-embedding".to_owned(),
                namespace: namespace.to_owned(),
                entity_kind: "node".to_owned(),
                entity_id: second_id.to_owned(),
                index_kind: "node_embedding".to_owned(),
                op: "UPSERT".to_owned(),
                payload_json: Some(second_job_payload),
                max_retries: 3,
            })
            .unwrap();
        service
            .store
            .enqueue_index_job(NewIndexJob {
                job_id: "worker-node-refs".to_owned(),
                namespace: namespace.to_owned(),
                entity_kind: "node".to_owned(),
                entity_id: entity_id.to_owned(),
                index_kind: "node_refs".to_owned(),
                op: "UPSERT".to_owned(),
                payload_json: None,
                max_retries: 3,
            })
            .unwrap();

        let failed_attempt = service
            .run_embedding_projection_jobs_once(2, 30, Some(namespace))
            .await
            .unwrap();
        assert_eq!(failed_attempt.claimed, 2);
        assert_eq!(failed_attempt.projected, 0);
        assert_eq!(failed_attempt.retried, 2);
        assert!(service
            .store
            .get_stage1_node_projection(namespace, &format!("node:{entity_id}"))
            .unwrap()
            .is_some());
        let failed_jobs = service
            .store
            .list_index_jobs(Some(namespace), Some("PENDING"), None, None, None, 10)
            .unwrap();
        assert_eq!(
            failed_jobs
                .iter()
                .filter(|job| job.index_kind == "node_embedding" && job.retry_count == 1)
                .count(),
            2
        );

        // The SQLite queue and canonical event survive process restart after a
        // provider failure; Chroma remains a derived eventual projection.
        drop(service);
        provider_failure.store(false, std::sync::atomic::Ordering::Relaxed);
        tokio::time::sleep(std::time::Duration::from_millis(1200)).await;
        let service = open_sqlite_application_service(
            &path,
            Some(SqliteChromaGraphConfig {
                endpoint: endpoint.clone(),
                metric: DistanceMetric::Cosine,
                node_collection: "worker-nodes".to_owned(),
                edge_collection: "worker-edges".to_owned(),
            }),
        )
        .await
        .unwrap()
        .with_embedding_provider(provider.clone());
        assert_eq!(
            sqlite_canonical_entity_revision(&service.store, namespace, "node", entity_id)
                .unwrap()
                .unwrap()
                .revision,
            appended.event.seq
        );
        assert_eq!(
            service
                .store
                .list_index_jobs(Some(namespace), Some("PENDING"), None, None, None, 10)
                .unwrap()
                .iter()
                .filter(|job| job.job_id == "worker-node-embedding")
                .count(),
            1
        );

        let summary = service
            .run_embedding_projection_jobs_once(2, 30, Some(namespace))
            .await
            .unwrap();
        assert_eq!(summary.claimed, 2);
        assert_eq!(summary.projected, 1);
        assert_eq!(summary.retried, 1);
        assert!(service
            .store
            .get_stage1_node_projection(namespace, &format!("node:{entity_id}"))
            .unwrap()
            .is_some());
        let accepted_retry = service
            .store
            .list_index_jobs(Some(namespace), Some("PENDING"), None, None, None, 10)
            .unwrap()
            .into_iter()
            .find(|job| job.index_kind == "node_embedding")
            .expect("failed Chroma projection remains retryable");
        assert!(accepted_retry.accepted_result_json.is_some());
        assert!(accepted_retry.accepted_result_sha256.is_some());
        let replay_id = accepted_retry.entity_id.clone();

        // The accepted embedding result is durable before the Chroma write.
        // Restarting replays it without calling the provider again.
        drop(service);
        tokio::time::sleep(std::time::Duration::from_millis(2200)).await;
        let service = open_sqlite_application_service(
            &path,
            Some(SqliteChromaGraphConfig {
                endpoint: endpoint.clone(),
                metric: DistanceMetric::Cosine,
                node_collection: "worker-nodes".to_owned(),
                edge_collection: "worker-edges".to_owned(),
            }),
        )
        .await
        .unwrap()
        .with_embedding_provider(provider.clone());
        let replay = service
            .run_embedding_projection_jobs_once(2, 30, Some(namespace))
            .await
            .unwrap();
        assert_eq!(replay.claimed, 1);
        assert_eq!(replay.projected, 0);
        assert_eq!(replay.reused, 1);
        assert!(service
            .store
            .get_stage1_node_projection(namespace, &format!("node:{entity_id}"))
            .unwrap()
            .is_none());
        assert!(service
            .store
            .get_stage1_node_projection(namespace, &format!("node:{second_id}"))
            .unwrap()
            .is_none());
        let mut batches = calls.lock().unwrap().clone();
        assert_eq!(batches.len(), 2, "one failed and one recovered provider batch");
        let expected_batch = vec![document.to_owned(), second_document.clone()];
        for batch in &mut batches {
            batch.sort();
            assert_eq!(*batch, expected_batch);
        }
        assert_eq!(batches[0], batches[1]);
        let jobs = service
            .store
            .list_index_jobs(Some(namespace), None, None, None, None, 10)
            .unwrap();
        assert_eq!(
            jobs.iter()
                .find(|job| job.job_id == "worker-node-embedding")
                .unwrap()
                .status,
            "DONE"
        );
        assert_eq!(
            jobs.iter()
                .find(|job| job.job_id == "worker-batch-node-embedding")
                .unwrap()
                .status,
            "DONE"
        );
        assert_eq!(
            jobs.iter()
                .find(|job| job.job_id == "worker-node-refs")
                .unwrap()
                .status,
            "PENDING"
        );
        let _ = stop_chroma.send(());
        let (requests, applied) = chroma.await.unwrap();
        let upserts = requests
            .iter()
            .filter(|(line, _)| line.contains("/upsert"))
            .collect::<Vec<_>>();
        assert_eq!(upserts.len(), 3, "failed upsert is replayed after restart");
        for (upsert_path, upsert) in &upserts {
            assert!(upsert_path.contains("worker-nodes"));
            assert_eq!(upsert["embeddings"][0], json!([0.6, 0.8]));
        }
        assert_eq!(upserts[0].1["ids"][0], replay_id);
        assert_eq!(upserts[0].1, upserts[2].1, "retry reuses the accepted action");
        assert_eq!(applied.len(), 2, "same-ID retry leaves one projection per entity");
        let mut projected_ids = upserts
            .iter()
            .filter_map(|(_, body)| body["ids"][0].as_str())
            .collect::<Vec<_>>();
        projected_ids.sort();
        assert_eq!(projected_ids, vec![entity_id, replay_id.as_str(), second_id]);
        let primary_upsert = upserts
            .iter()
            .find(|(_, body)| body["ids"][0] == entity_id)
            .expect("the first source record is projected");
        let metadata: Value = serde_json::from_str(
            primary_upsert.1["metadatas"][0]["_kogwistar_original_metadata"]
                .as_str()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(metadata["namespace"], namespace);
        assert_eq!(metadata["doc_id"], "doc:rust-python-parity");
        assert_eq!(
            metadata["properties"],
            "{\"attributes\":[\"unicode-雪\",\"nested-payload\"]}"
        );
        assert!(metadata.get("redirect_to_id").is_none());
        assert_eq!(metadata["_kogwistar_stage2_ready"], true);
        assert_eq!(metadata["_kogwistar_source_revision"], appended.event.seq);
        assert_eq!(metadata["_kogwistar_source_fingerprint"], fingerprint);

        let stale_id = "stale-stage1-node";
        let stale_appended = service
            .store
            .append_raw_entity_event(
                namespace,
                SqliteNewRawEntityEvent {
                    event_id: "stale-stage1-node-add".to_owned(),
                    entity_kind: "node".to_owned(),
                    entity_id: stale_id.to_owned(),
                    op: "ADD".to_owned(),
                    payload_json: json!({"id":stale_id,"document":"canonical v1"}).to_string(),
                },
            )
            .unwrap();
        let stale_revision = sqlite_canonical_entity_revision(
            &service.store,
            namespace,
            "node",
            stale_id,
        )
        .unwrap()
        .unwrap();
        let stale_job_payload =
            kogwistar_engine::canonical_revision::canonical_revision_job_payload(&stale_revision);
        service
            .store
            .replace_stage1_node_projection(
                namespace,
                "node:stale-stage1-node",
                NamedProjectionWrite {
                    payload: json!({
                        "document":"stale stage one text",
                        "metadata":{},
                        "source_fingerprint":"not-the-canonical-fingerprint"
                    })
                    .as_object()
                    .unwrap()
                    .clone(),
                    last_authoritative_seq: stale_appended.event.seq,
                    last_materialized_seq: stale_appended.event.seq,
                    projection_schema_version: 1,
                    materialization_status: "ready".to_owned(),
                },
            )
            .unwrap();
        service
            .store
            .enqueue_index_job(NewIndexJob {
                job_id: "stale-stage1-embedding".to_owned(),
                namespace: namespace.to_owned(),
                entity_kind: "node".to_owned(),
                entity_id: stale_id.to_owned(),
                index_kind: "node_embedding".to_owned(),
                op: "UPSERT".to_owned(),
                payload_json: Some(stale_job_payload),
                max_retries: 3,
            })
            .unwrap();
        let stale_summary = service
            .run_embedding_projection_jobs_once(1, 30, Some(namespace))
            .await
            .unwrap();
        assert_eq!(stale_summary.claimed, 1);
        assert_eq!(stale_summary.projected, 0);
        assert_eq!(stale_summary.retried, 1);
        let mut provider_batches = calls.lock().unwrap().clone();
        assert_eq!(provider_batches.len(), 2);
        let expected_batch = vec![document.to_owned(), second_document];
        for batch in &mut provider_batches {
            batch.sort();
            assert_eq!(*batch, expected_batch);
        }
        let stale_job = service
            .store
            .list_index_jobs(Some(namespace), Some("PENDING"), None, None, None, 10)
            .unwrap()
            .into_iter()
            .find(|job| job.job_id == "stale-stage1-embedding")
            .expect("mismatched Stage-1 revision remains retryable");
        assert!(stale_job.last_error.as_deref().unwrap_or_default().contains("revisions do not match"));

        drop(service);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    #[ignore = "requires a disposable live Chroma service; set KOGWISTAR_TEST_CHROMA_URL"]
    // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/tests/sqlite_stage1_recovery.rs#sqlite_search_index_events_project_to_live_chroma_and_survive_reopen; python=kogwistar/engine_core/search_index/service.py#SearchIndexService.upsert_entries; fixture=contracts/golden/search-index-entry-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/sqlite_stage1_recovery.rs#sqlite_search_index_events_project_to_live_chroma_and_survive_reopen; python_test=tests/cdc/test_search_index_cdc.py#test_search_index_replay
    async fn sqlite_search_index_events_project_to_live_chroma_and_survive_reopen() {
        use kogwistar_store::{
            EventReadStore, EventWriteStore, GraphEntityKind, GraphRecord, GraphWriteStore,
            NewEntityEvent, NodeIndexStore,
        };

        let endpoint = std::env::var("KOGWISTAR_TEST_CHROMA_URL")
            .expect("ignored live Chroma test requires KOGWISTAR_TEST_CHROMA_URL");
        let suffix = uuid::Uuid::new_v4().simple().to_string();
        let namespace = format!("search-index-{suffix}");
        let root = std::env::temp_dir().join(format!("kogwistar-search-live-{suffix}"));
        std::fs::create_dir_all(&root).unwrap();
        let store_path = root.join("metadata.sqlite3");
        let fts_path = root.join("index.sqlite3");
        let config = kogwistar_store_chroma::ChromaStoreConfig {
            endpoint,
            collection: "nodes_index".to_owned(),
            metric: DistanceMetric::Cosine,
            ..Default::default()
        };
        let provider = Arc::new(FixedEmbeddingProvider {
            profile: kogwistar_engine::embedding::EmbeddingProfile {
                provider: "test".to_owned(),
                model: "fixed-search-index-v1".to_owned(),
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
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/search-index-entry-v1.json"
        ))
        .unwrap();
        let mut request: kogwistar_contracts::SearchIndexUpsertV1 =
            serde_json::from_value(fixture["request"].clone()).unwrap();
        request.index.truncate(1);
        request.index[0].node_id = format!("node-{suffix}");
        request.index[0].canonical_title = format!("UniquePermit-{suffix}");
        request.index[0].keywords = vec![format!("uniquepermit{suffix}")];

        let store = SqliteStore::open(&store_path).unwrap();
        let vector_store = kogwistar_store_chroma::ChromaVectorCollectionStore::connect(config.clone())
            .await
            .unwrap();
        let fts = std::sync::Mutex::new(crate::search_index_fts::open_index(&fts_path).unwrap());
        let admitted = crate::search_index_fts::admit_search_index_entries(
            &store,
            &vector_store,
            &fts,
            provider.as_ref(),
            &namespace,
            "nodes_index",
            &request.index,
        )
        .await
        .expect("project admitted search-index events into Chroma");
        assert_eq!(admitted.entries_admitted, 1);
        assert_eq!(admitted.events_projected, 1);
        assert_eq!(
            EventReadStore::replay_events(&store, &namespace, 0, 10)
                .await
                .unwrap()
                .len(),
            1
        );
        let key = kogwistar_contracts::make_search_index_key(
            &request.index[0].node_id,
            &request.index[0].canonical_title,
            &request.index[0].provision,
        );
        let records = vector_store
            .get_node_index("nodes_index", &[format!("idx:{key}")])
            .await
            .unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].embedding.as_deref(), Some([0.6_f32, 0.8_f32].as_slice()));
        assert_eq!(records[0].metadata["target_node_id"], request.index[0].node_id);
        assert!(crate::search_index_fts::search_fts(
            &fts.lock().unwrap(),
            &format!("uniquepermit{suffix}"),
            5,
        )
        .unwrap()
        .iter()
        .any(|hit| hit.node_id == request.index[0].node_id));

        drop(fts);
        drop(vector_store);
        drop(store);
        let reopened_store = SqliteStore::open(&store_path).unwrap();
        let reopened_vector = kogwistar_store_chroma::ChromaVectorCollectionStore::connect(config)
            .await
            .unwrap();
        let reopened_fts = std::sync::Mutex::new(crate::search_index_fts::open_index(&fts_path).unwrap());
        assert_eq!(
            EventReadStore::replay_events(&reopened_store, &namespace, 0, 10)
                .await
                .unwrap()
                .len(),
            1,
            "SQLite remains the durable source of truth across reopen"
        );
        let hits = crate::search_index_fts::hybrid_search(
            &reopened_fts,
            &reopened_vector,
            "nodes_index",
            provider.as_ref(),
            &format!("uniquepermit{suffix}"),
            5,
        )
        .await
        .unwrap();
        assert!(hits.iter().any(|hit| hit.node_id == request.index[0].node_id));

        let graph_node_collection = format!("kgw-route-n-{suffix}");
        let graph_edge_collection = format!("kgw-route-e-{suffix}");
        let route_service = open_sqlite_application_service(
            &store_path,
            Some(SqliteChromaGraphConfig {
                endpoint: std::env::var("KOGWISTAR_TEST_CHROMA_URL").unwrap(),
                metric: DistanceMetric::Cosine,
                node_collection: graph_node_collection.clone(),
                edge_collection: graph_edge_collection.clone(),
            }),
        )
        .await
        .unwrap()
        .with_search_index_path(&fts_path)
        .unwrap()
        .with_embedding_provider(provider.clone());
        let chroma_endpoint = std::env::var("KOGWISTAR_TEST_CHROMA_URL").unwrap();
        let graph = kogwistar_store_chroma::ChromaGraphStore::connect_entity_collections(
            kogwistar_store_chroma::ChromaStoreConfig {
                endpoint: chroma_endpoint.clone(),
                collection: graph_node_collection,
                metric: DistanceMetric::Cosine,
                ..Default::default()
            },
            kogwistar_store_chroma::ChromaStoreConfig {
                endpoint: chroma_endpoint,
                collection: graph_edge_collection,
                metric: DistanceMetric::Cosine,
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let node_id = request.index[0].node_id.clone();
        graph
            .upsert_graph_record_of_kind(
                &namespace,
                GraphEntityKind::Node,
                GraphRecord {
                    id: node_id.clone(),
                    document: Some("authorized live Chroma node".to_owned()),
                    metadata: serde_json::from_value(json!({
                        "workspace_id":"workspace-live",
                        "graph_space":"facts",
                    }))
                    .unwrap(),
                    embedding: Some(vec![0.6, 0.8]),
                },
            )
            .await
            .unwrap();

        let acl_id = format!("acl-{node_id}");
        reopened_store
            .append_entity_event(
                &namespace,
                NewEntityEvent {
                    event_id: format!("{acl_id}-public-v1"),
                    entity_kind: "node".to_owned(),
                    entity_id: acl_id.clone(),
                    op: "ADD".to_owned(),
                    payload: json!({
                        "id": acl_id,
                        "metadata": {
                            "workspace_id":"workspace-live",
                            "graph_space":"facts",
                            "entity_type":"acl_record",
                            "acl_truth_graph":"knowledge",
                            "acl_target_grain":"node",
                            "acl_target_entity_id":node_id,
                            "acl_version":1,
                            "acl_mode":"public"
                        }
                    }),
                },
            )
            .await
            .unwrap();
        graph
            .upsert_graph_record_of_kind(
                &namespace,
                GraphEntityKind::Node,
                GraphRecord {
                    id: format!("acl-{node_id}"),
                    document: Some("{}".to_owned()),
                    metadata: serde_json::from_value(json!({
                        "workspace_id":"workspace-live",
                        "graph_space":"facts",
                        "entity_type":"acl_record",
                        "acl_truth_graph":"knowledge",
                        "acl_target_grain":"node",
                        "acl_target_entity_id":node_id,
                        "acl_version":1,
                        "acl_mode":"public"
                    }))
                    .unwrap(),
                    embedding: Some(vec![0.6, 0.8]),
                },
            )
            .await
            .unwrap();

        let inserted_node = graph
            .graph_record_of_kind(&namespace, GraphEntityKind::Node, &node_id)
            .await
            .unwrap()
            .expect("inserted live node must be retrievable");
        assert_eq!(inserted_node.metadata["workspace_id"], "workspace-live");

        let auth = AuthConfig {
            mode: "dev".to_owned(),
            algorithm: Some("HS256".to_owned()),
            key: Some("sqlite-live-chroma-search-secret".to_owned()),
            ..AuthConfig::default()
        };
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let token = |capabilities: &[&str]| {
            jsonwebtoken::encode(
                &jsonwebtoken::Header::new(jsonwebtoken::Algorithm::HS256),
                &json!({
                    "sub":"live-reader",
                    "storage_ns":namespace,
                    "ns":[namespace],
                    "workspace_id":"workspace-live",
                    "graph_space":"facts",
                    "role":"ro",
                    "capabilities":capabilities,
                    "exp":now + 300
                }),
                &jsonwebtoken::EncodingKey::from_secret(
                    b"sqlite-live-chroma-search-secret",
                ),
            )
            .unwrap()
        };
        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: Vec::new(),
                implementation: ImplementationSnapshot::default(),
                auth,
            },
            Arc::new(route_service),
        );
        let mut routed_request = request.clone();
        routed_request.index[0].aliases.push("route-updated".to_owned());
        let denied_tokens = [
            jsonwebtoken::encode(
                &jsonwebtoken::Header::new(jsonwebtoken::Algorithm::HS256),
                &json!({
                    "sub":"missing-writer-capability",
                    "storage_ns":namespace,
                    "ns":[namespace],
                    "workspace_id":"workspace-live",
                    "graph_space":"facts",
                    "role":"rw",
                    "capabilities":[],
                    "exp":now + 300
                }),
                &jsonwebtoken::EncodingKey::from_secret(
                    b"sqlite-live-chroma-search-secret",
                ),
            )
            .unwrap(),
            jsonwebtoken::encode(
                &jsonwebtoken::Header::new(jsonwebtoken::Algorithm::HS256),
                &json!({
                    "sub":"foreign-storage-namespace",
                    "storage_ns":format!("foreign-{suffix}"),
                    "ns":[namespace],
                    "workspace_id":"workspace-live",
                    "graph_space":"facts",
                    "role":"rw",
                    "capabilities":["write_graph"],
                    "exp":now + 300
                }),
                &jsonwebtoken::EncodingKey::from_secret(
                    b"sqlite-live-chroma-search-secret",
                ),
            )
            .unwrap(),
        ];
        for denied_token in denied_tokens {
            let denied = app
                .clone()
                .oneshot(
                    axum::http::Request::builder()
                        .method("POST")
                        .uri("/api/add_index_entries")
                        .header("authorization", format!("Bearer {denied_token}"))
                        .header("content-type", "application/json")
                        .body(axum::body::Body::from(
                            serde_json::to_vec(&routed_request).unwrap(),
                        ))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(denied.status(), axum::http::StatusCode::FORBIDDEN);
        }
        let events = EventReadStore::replay_events(&reopened_store, &namespace, 0, 10)
            .await
            .unwrap();
        assert_eq!(
            events
                .iter()
                .filter(|event| event.entity_kind == "search_index")
                .count(),
            1,
            "denied writes must not append search-index events"
        );
        let writer_token = jsonwebtoken::encode(
            &jsonwebtoken::Header::new(jsonwebtoken::Algorithm::HS256),
            &json!({
                "sub":"live-writer",
                "storage_ns":namespace,
                "ns":[namespace],
                "workspace_id":"workspace-live",
                "graph_space":"facts",
                "role":"rw",
                "capabilities":["write_graph"],
                "exp":now + 300
            }),
            &jsonwebtoken::EncodingKey::from_secret(b"sqlite-live-chroma-search-secret"),
        )
        .unwrap();
        for _ in 0..2 {
            let write = app
                .clone()
                .oneshot(
                    axum::http::Request::builder()
                        .method("POST")
                        .uri("/api/add_index_entries")
                        .header("authorization", format!("Bearer {writer_token}"))
                        .header("content-type", "application/json")
                        .body(axum::body::Body::from(
                            serde_json::to_vec(&routed_request).unwrap(),
                        ))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(write.status(), axum::http::StatusCode::OK);
        }
        let replayed = EventReadStore::replay_events(&reopened_store, &namespace, 0, 10)
            .await
            .unwrap();
        assert_eq!(
            replayed
                .iter()
                .filter(|event| event.entity_kind == "search_index")
                .count(),
            2,
            "retrying the same public write is idempotent"
        );
        let routed_key = kogwistar_contracts::make_search_index_key(
            &routed_request.index[0].node_id,
            &routed_request.index[0].canonical_title,
            &routed_request.index[0].provision,
        );
        let routed_vector = reopened_vector
            .get_node_index("nodes_index", &[format!("idx:{routed_key}")])
            .await
            .unwrap();
        assert_eq!(routed_vector.len(), 1);
        assert_eq!(
            serde_json::from_str::<Value>(
                routed_vector[0].metadata["aliases"].as_str().unwrap()
            )
            .unwrap(),
            json!(["approval", "authorization", "route-updated"])
        );
        let encoded_query = url::form_urlencoded::byte_serialize(
            format!("uniquepermit{suffix}").as_bytes(),
        )
        .collect::<String>();
        let response = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .method("GET")
                    .uri(format!(
                        "/api/search_index_hybrid?q={encoded_query}&limit=5&resolve_node=true"
                    ))
                    .header(
                        "authorization",
                        format!("Bearer {}", token(&["read_graph"])),
                    )
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let response: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), 65_536)
                .await
                .unwrap(),
        )
        .unwrap();
        let visible = response["results"]
            .as_array()
            .unwrap()
            .iter()
            .find(|hit| hit["node_id"] == node_id)
            .unwrap_or_else(|| {
                panic!("authorized indexed node missing from live HTTP response: {response}")
            });
        assert_eq!(visible["documents"], "authorized live Chroma node");
        assert_eq!(visible["metadatas"]["workspace_id"], "workspace-live");

        graph
            .upsert_graph_record_of_kind(
                &namespace,
                GraphEntityKind::Node,
                GraphRecord {
                    id: format!("acl-{node_id}"),
                    document: Some("{}".to_owned()),
                    metadata: serde_json::from_value(json!({
                        "workspace_id":"workspace-live",
                        "graph_space":"facts",
                        "entity_type":"acl_record",
                        "acl_truth_graph":"knowledge",
                        "acl_target_grain":"node",
                        "acl_target_entity_id":node_id,
                        "acl_version":2,
                        "acl_mode":"private"
                    }))
                    .unwrap(),
                    embedding: Some(vec![0.6, 0.8]),
                },
            )
            .await
            .unwrap();
        reopened_store
            .append_entity_event(
                &namespace,
                NewEntityEvent {
                    event_id: format!("acl-{node_id}-private-v2"),
                    entity_kind: "node".to_owned(),
                    entity_id: format!("acl-{node_id}"),
                    op: "REPLACE".to_owned(),
                    payload: json!({
                        "id": format!("acl-{node_id}"),
                        "metadata": {
                            "workspace_id":"workspace-live",
                            "graph_space":"facts",
                            "entity_type":"acl_record",
                            "acl_truth_graph":"knowledge",
                            "acl_target_grain":"node",
                            "acl_target_entity_id":node_id,
                            "acl_version":2,
                            "acl_mode":"private"
                        }
                    }),
                },
            )
            .await
            .unwrap();
        let acl_denied = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .method("GET")
                    .uri(format!(
                        "/api/search_index_hybrid?q={encoded_query}&limit=5&resolve_node=true"
                    ))
                    .header(
                        "authorization",
                        format!("Bearer {}", token(&["read_graph"])),
                    )
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(acl_denied.status(), axum::http::StatusCode::OK);
        let acl_denied: Value = serde_json::from_slice(
            &axum::body::to_bytes(acl_denied.into_body(), 65_536)
                .await
                .unwrap(),
        )
        .unwrap();
        assert!(acl_denied["results"].as_array().unwrap().is_empty());

        let denied = app
            .oneshot(
                axum::http::Request::builder()
                    .method("GET")
                    .uri("/api/search_index_hybrid?q=uniquepermit&limit=5")
                    .header("authorization", format!("Bearer {}", token(&[])))
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(denied.status(), axum::http::StatusCode::FORBIDDEN);

        drop(reopened_fts);
        drop(reopened_vector);
        drop(reopened_store);
        let _ = std::fs::remove_dir_all(root);
    }
