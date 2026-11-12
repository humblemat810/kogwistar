    #[tokio::test]
    #[ignore = "requires KOGWISTAR_TEST_PG_DSN; run explicitly against disposable PostgreSQL"]
    async fn postgres_embedding_worker_uses_configured_provider_and_leaves_other_jobs_pending() {
        let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
            .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
        let batch_contract: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/two-stage-embedding-batch-v1.json"
        )))
        .unwrap();
        let expected_embedding = batch_contract["expected_embedding"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap())
            .collect::<Vec<_>>();
        let expected_stored_embedding = expected_embedding
            .iter()
            .map(|value| *value as f32)
            .collect::<Vec<_>>();
        let batch_size = batch_contract["batch_size"].as_u64().unwrap() as usize;
        let embedding_dimension = batch_contract["embedding_dimension"].as_u64().unwrap() as usize;
        let schema = format!("rust_embedding_worker_{}", uuid::Uuid::new_v4().simple());
        let calls = Arc::new(Mutex::new(Vec::new()));
        let provider = Arc::new(FixedEmbeddingProvider {
            profile: kogwistar_engine::embedding::EmbeddingProfile {
                provider: "test".to_owned(),
                model: "fixed-v1".to_owned(),
                dimension: embedding_dimension,
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
            vector: expected_embedding.clone(),
            calls: calls.clone(),
            fail: None,
        });
        let service = PostgresRunApplicationService::from_dsn(&dsn, &schema)
            .unwrap()
            .with_embedding_provider(provider.clone());
        service.ensure_schema().await.unwrap();
        service
            .store
            .create_graph_schema(
                embedding_dimension,
                kogwistar_store_postgres::GraphTableNames::default(),
            )
            .await
            .unwrap();

        let namespace = format!("embedding-worker-{}", uuid::Uuid::new_v4().simple());
        let node_id = "provider-worker-node";
        service
            .store
            .apply_graph_mutation(GraphMutation {
                scope: GraphScope {
                    namespace: namespace.clone(),
                    workspace_id: None,
                    graph_space: None,
                },
                table: "gke_nodes".to_owned(),
                entity_kind: "node".to_owned(),
                event_id: format!("{node_id}-add"),
                op: "ADD".to_owned(),
                payload: json!({"id": node_id}),
                record: GraphRecord {
                    id: node_id.to_owned(),
                    document: Some("canonical text for provider".to_owned()),
                    metadata: Default::default(),
                    embedding: None,
                },
                embedding_dim: embedding_dimension,
            })
            .await
            .unwrap();
        let edge_id = "provider-worker-edge";
        let edge_document = json!({
            "id": edge_id,
            "source_ids": ["source"],
            "target_ids": ["target"]
        })
        .to_string();
        service
            .store
            .apply_graph_mutation(GraphMutation {
                scope: GraphScope {
                    namespace: namespace.clone(),
                    workspace_id: None,
                    graph_space: None,
                },
                table: "gke_edges".to_owned(),
                entity_kind: "edge".to_owned(),
                event_id: format!("{edge_id}-add"),
                op: "ADD".to_owned(),
                payload: json!({"id": edge_id, "source_ids": ["source"], "target_ids": ["target"]}),
                record: GraphRecord {
                    id: edge_id.to_owned(),
                    document: Some(edge_document.clone()),
                    metadata: Default::default(),
                    embedding: None,
                },
                embedding_dim: embedding_dimension,
            })
            .await
            .unwrap();
        for (job_id, entity_kind, entity_id, index_kind) in [
            ("provider-worker-node-embedding", "node", node_id, "node_embedding"),
            ("provider-worker-edge-embedding", "edge", edge_id, "node_embedding"),
            ("provider-worker-refs", "node", node_id, "node_refs"),
        ] {
            service
                .store
                .enqueue_index_job(NewIndexJob {
                    job_id: format!("{job_id}-{}", uuid::Uuid::new_v4().simple()),
                    namespace: namespace.clone(),
                    entity_kind: entity_kind.to_owned(),
                    entity_id: entity_id.to_owned(),
                    index_kind: index_kind.to_owned(),
                    op: "UPSERT".to_owned(),
                    payload_json: None,
                    max_retries: 3,
                })
                .await
                .unwrap();
        }

        let cleanup_dsn = dsn
            .strip_prefix("postgresql+psycopg://")
            .map(|rest| format!("postgresql://{rest}"))
            .or_else(|| {
                dsn.strip_prefix("postgresql+psycopg2://")
                    .map(|rest| format!("postgresql://{rest}"))
            })
            .unwrap_or_else(|| dsn.clone());
        let (admin, connection) = tokio_postgres::connect(&cleanup_dsn, tokio_postgres::NoTls)
            .await
            .unwrap();
        let connection_task = tokio::spawn(connection);
        admin
            .batch_execute(&format!(
                "CREATE FUNCTION \"{schema}\".fail_embedding_projection() RETURNS trigger \
                 LANGUAGE plpgsql AS $$ BEGIN \
                 IF NEW.id = '{node_id}' AND NEW.embedding IS NOT NULL THEN \
                 RAISE EXCEPTION 'injected projection failure'; END IF; \
                 RETURN NEW; END $$; \
                 CREATE TRIGGER fail_embedding_projection BEFORE UPDATE ON \"{schema}\".gke_nodes \
                 FOR EACH ROW EXECUTE FUNCTION \"{schema}\".fail_embedding_projection();"
            ))
            .await
            .unwrap();

        let failed_projection = service
            .run_embedding_projection_jobs_once(4, 30, Some(&namespace))
            .await
            .unwrap();
        assert_eq!(failed_projection.claimed, 2);
        assert_eq!(failed_projection.projected, 1);
        assert_eq!(failed_projection.retried, 1);
        let provider_batches = calls.lock().unwrap().clone();
        assert_eq!(provider_batches.len(), 1);
        assert_eq!(provider_batches[0].len(), batch_size);
        let mut provider_texts = calls
            .lock()
            .unwrap()
            .iter()
            .flatten()
            .cloned()
            .collect::<Vec<_>>();
        provider_texts.sort();
        assert_eq!(
            provider_texts,
            vec![
                "canonical text for provider".to_owned(),
                edge_document.clone()
            ]
        );

        let retryable = service
            .store
            .list_index_jobs(Some(&namespace), Some("PENDING"), None, None, None, 10)
            .await
            .unwrap();
        let embedding_job = retryable
            .iter()
            .find(|job| job.index_kind == "node_embedding")
            .expect("failed embedding projection remains pending");
        assert!(embedding_job.accepted_result_json.is_some());
        assert!(embedding_job.accepted_result_sha256.is_some());
        assert!(retryable.iter().any(|job| job.index_kind == "node_refs"));

        drop(service);
        let service = PostgresRunApplicationService::from_dsn(&dsn, &schema)
            .unwrap()
            .with_embedding_provider(provider);
        service.ensure_schema().await.unwrap();

        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        admin
            .batch_execute(&format!(
                "DROP TRIGGER fail_embedding_projection ON \"{schema}\".gke_nodes; \
                 DROP FUNCTION \"{schema}\".fail_embedding_projection();"
            ))
            .await
            .unwrap();
        let (worker_shutdown, worker_shutdown_rx) = tokio::sync::oneshot::channel();
        let worker = tokio::spawn(run_inline_embedding_worker(
            Arc::new(service.clone()),
            worker_shutdown_rx,
            true,
            std::time::Duration::ZERO,
            String::new(),
        ));
        let jobs = tokio::time::timeout(std::time::Duration::from_secs(10), async {
            loop {
                let jobs = service
                    .store
                    .list_index_jobs(Some(&namespace), None, None, None, None, 10)
                    .await
                    .unwrap();
                if jobs.iter().filter(|job| job.status == "DONE").count() == 2 {
                    break jobs;
                }
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            }
        })
        .await
        .expect("inline worker completes the recovered embedding job");
        let _ = worker_shutdown.send(());
        worker.await.unwrap();
        assert_eq!(calls.lock().unwrap().len(), 1);
        let mut provider_texts = calls
            .lock()
            .unwrap()
            .iter()
            .flatten()
            .cloned()
            .collect::<Vec<_>>();
        provider_texts.sort();
        assert_eq!(
            provider_texts,
            vec![
                "canonical text for provider".to_owned(),
                edge_document.clone()
            ]
        );

        let projection = service
            .store
            .graph_projection_records(GraphProjectionRead {
                scope: GraphScope {
                    namespace: namespace.clone(),
                    workspace_id: None,
                    graph_space: None,
                },
                table: "gke_nodes".to_owned(),
                ids: Some(vec![node_id.to_owned()]),
                metadata: MetadataFilter::default(),
                limit: 1,
            })
            .await
            .unwrap();
        assert_eq!(
            projection[0].embedding,
            Some(expected_stored_embedding.clone())
        );
        let edge_projection = service
            .store
            .graph_projection_records(GraphProjectionRead {
                scope: GraphScope {
                    namespace: namespace.clone(),
                    workspace_id: None,
                    graph_space: None,
                },
                table: "gke_edges".to_owned(),
                ids: Some(vec![edge_id.to_owned()]),
                metadata: MetadataFilter::default(),
                limit: 1,
            })
            .await
            .unwrap();
        assert_eq!(edge_projection[0].embedding, Some(expected_stored_embedding));
        assert_eq!(jobs.iter().filter(|job| job.status == "DONE").count(), 2);
        assert_eq!(jobs.iter().filter(|job| job.status == "PENDING").count(), 1);

        drop(service);
        admin
            .batch_execute(&format!("DROP SCHEMA \"{schema}\" CASCADE"))
            .await
            .unwrap();
        drop(admin);
        connection_task.await.unwrap().unwrap();
    }

    #[test]
    fn rust_server_embedding_configuration_is_opt_in_and_fails_closed() {
        assert!(!parse_inline_worker_flag("KOGWISTAR_EMBEDDING_WORKER_INLINE", None).unwrap());
        assert!(!parse_inline_worker_flag("KOGWISTAR_EMBEDDING_WORKER_INLINE", Some("off"))
            .unwrap());
        assert!(parse_inline_worker_flag("KOGWISTAR_EMBEDDING_WORKER_INLINE", Some(" YES "))
            .unwrap());
        assert!(parse_inline_worker_flag("KOGWISTAR_EMBEDDING_WORKER_INLINE", Some("maybe"))
            .is_err());
        assert!(embedding_provider_from_settings(None, None, None, None)
            .unwrap()
            .is_none());
        assert!(embedding_provider_from_settings(Some("  "), None, None, None)
            .unwrap()
            .is_none());
        assert!(embedding_provider_from_settings(Some("openai"), Some("m"), Some("3"), None)
            .is_err());
        assert!(embedding_provider_from_settings(Some("ollama"), None, Some("3"), None)
            .is_err());
        assert!(embedding_provider_from_settings(Some("ollama"), Some("m"), Some("0"), None)
            .is_err());
        assert!(embedding_provider_from_settings(Some("ollama"), Some("m"), Some("x"), None)
            .is_err());

        let provider = embedding_provider_from_settings(
            Some("Ollama"),
            Some("nomic-embed-text"),
            Some("768"),
            Some("http://ollama:11434"),
        )
        .unwrap()
        .unwrap();
        assert_eq!(provider.profile().provider, "ollama");
        assert_eq!(provider.profile().model, "nomic-embed-text");
        assert_eq!(provider.profile().dimension, 768);
        assert!(embedding_provider_from_settings(
            Some("ollama"),
            Some("nomic-embed-text"),
            Some("768"),
            Some("not-a-url"),
        )
        .is_err());
    }

    #[test]
    fn rust_postgres_embedding_scope_matches_python_physical_scope() {
        let config: tokio_postgres::Config =
            "postgresql://writer:secret@EXAMPLE.com:5544/dbname".parse().unwrap();
        assert_eq!(
            rust_pgvector_embedding_scope(&config, "tenant").unwrap(),
            "pgvector:5eafd1a9e29e59fdabd83f04e36ed6ae"
        );
    }

    #[test]
    fn registered_embedding_profile_matches_by_full_fingerprint() {
        let profile: kogwistar_engine::embedding::EmbeddingProfile = serde_json::from_str(
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../../contracts/golden/embedding-profile-v1.json"
            )),
        )
        .unwrap();
        let payload = json!({
            "profile_schema_version": 1,
            "embedding_profile": profile.as_value(),
            "embedding_fingerprint": profile.fingerprint(),
            "legacy_adopted": false,
        })
        .as_object()
        .cloned()
        .unwrap();
        let projection = NamedProjection {
            namespace: EMBEDDING_PROFILE_NAMESPACE.to_owned(),
            key: "pgvector:fixture".to_owned(),
            payload,
            last_authoritative_seq: 0,
            last_materialized_seq: 0,
            projection_schema_version: 1,
            materialization_status: "bound".to_owned(),
            updated_at_ms: 0,
        };
        validate_embedding_profile_projection(&projection, &profile, "pgvector:fixture").unwrap();

        let mut changed_profile = profile;
        changed_profile.model = "different-model-same-dimension".to_owned();
        assert!(validate_embedding_profile_projection(
            &projection,
            &changed_profile,
            "pgvector:fixture"
        )
        .is_err());
    }
