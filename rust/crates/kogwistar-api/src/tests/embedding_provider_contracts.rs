    #[tokio::test]
    async fn configured_rust_provider_fills_missing_graph_vector_before_atomic_store_call() {
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
        let service = PostgresRunApplicationService::from_dsn(
            "postgresql://test:test@127.0.0.1:1/test",
            "public",
        )
        .unwrap()
        .with_embedding_provider(provider);
        let mut batch = graph_mutation_batch_without_vector();

        service.embed_missing_graph_vectors(&mut batch).await.unwrap();

        assert_eq!(batch.mutations[0].record.embedding, Some(vec![0.6, 0.8]));
        assert_eq!(batch.mutations[0].embedding_dim, 2);
        assert_eq!(calls.lock().unwrap().as_slice(), &[vec!["a source-grounded node".to_owned()]]);
    }

    #[tokio::test]
    async fn provider_does_not_turn_durable_two_stage_batch_into_synchronous_embedding() {
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
        let service = PostgresRunApplicationService::from_dsn(
            "postgresql://test:test@127.0.0.1:1/test",
            "public",
        )
        .unwrap()
        .with_embedding_provider(provider);
        let mut batch = graph_mutation_batch_without_vector();
        batch.index_jobs.push(kogwistar_contracts::GraphIndexJobV1 {
            job_id: "index-node-1".to_owned(),
            namespace: "tenant-a".to_owned(),
            entity_kind: "node".to_owned(),
            entity_id: "node-1".to_owned(),
            index_kind: "node_embedding".to_owned(),
            op: "UPSERT".to_owned(),
            payload_json: None,
            max_retries: 3,
        });

        service.embed_missing_graph_vectors(&mut batch).await.unwrap();

        assert_eq!(batch.mutations[0].record.embedding, None);
        assert!(calls.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn embedding_worker_requires_provider_before_opening_or_claiming_postgres_jobs() {
        let service = PostgresRunApplicationService::from_dsn(
            "postgresql://test:test@127.0.0.1:1/test",
            "public",
        )
        .unwrap();
        let error = service
            .run_embedding_projection_jobs_once(1, 30, Some("tenant-a"))
            .await
            .unwrap_err();
        assert_eq!(error, "Rust embedding provider is not configured");
    }
