    #[tokio::test]
    #[ignore = "requires KOGWISTAR_TEST_PG_DSN; uses isolated schemas and removes them"]
    // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/tests/postgres_index_projection.rs#postgres_search_index_events_project_to_pgvector_and_survive_reopen; python=kogwistar/engine_core/search_index/service.py#SearchIndexService.upsert_entries; fixture=contracts/golden/search-index-entry-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/postgres_index_projection.rs#postgres_search_index_events_project_to_pgvector_and_survive_reopen; python_test=tests/cdc/test_search_index_cdc.py#test_search_index_replay
    async fn postgres_search_index_events_project_to_pgvector_and_survive_reopen() {
        use kogwistar_store::EventReadStore;

        let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
            .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
        let schema = format!("rust_search_index_{}", uuid::Uuid::new_v4().simple());
        let root = std::env::temp_dir().join(format!("kogwistar-search-pg-{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&root).unwrap();
        let fts_path = root.join("index.sqlite3");
        let mut service = PostgresRunApplicationService::from_dsn(&dsn, &schema).unwrap();
        service.ensure_schema().await.unwrap();
        service
            .store
            .create_graph_schema(2, kogwistar_store_postgres::GraphTableNames::default())
            .await
            .unwrap();

        let provider = FixedEmbeddingProvider {
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
        };
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/search-index-entry-v1.json"
        ))
        .unwrap();
        let mut request: kogwistar_contracts::SearchIndexUpsertV1 =
            serde_json::from_value(fixture["request"].clone()).unwrap();
        request.index.truncate(1);
        let suffix = uuid::Uuid::new_v4().simple().to_string();
        let namespace = format!("search-index-{suffix}");
        request.index[0].node_id = format!("node-{suffix}");
        request.index[0].canonical_title = format!("UniquePermit-{suffix}");
        request.index[0].keywords = vec![format!("uniquepermit{suffix}")];
        let key = kogwistar_contracts::make_search_index_key(
            &request.index[0].node_id,
            &request.index[0].canonical_title,
            &request.index[0].provision,
        );

        let fts = std::sync::Mutex::new(crate::search_index_fts::open_index(&fts_path).unwrap());
        let admitted = crate::search_index_fts::admit_search_index_entries(
            &service.store,
            &service.store,
            &fts,
            &provider,
            &namespace,
            "gke_node_index",
            &request.index,
        )
        .await
        .expect("project admitted search-index events to PostgreSQL FTS and pgvector");
        assert_eq!(admitted.entries_admitted, 1);
        assert_eq!(admitted.events_projected, 1);
        assert_eq!(
            EventReadStore::replay_events(&service.store, &namespace, 0, 10)
                .await
                .unwrap()
                .len(),
            1
        );
        let records = service
            .store
            .get_node_index("gke_node_index", &[format!("idx:{key}")])
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
        drop(service);
        service = PostgresRunApplicationService::from_dsn(&dsn, &schema).unwrap();
        let reopened_fts =
            std::sync::Mutex::new(crate::search_index_fts::open_index(&fts_path).unwrap());
        assert_eq!(
            EventReadStore::replay_events(&service.store, &namespace, 0, 10)
                .await
                .unwrap()
                .len(),
            1,
            "PostgreSQL remains the durable source of truth across service reopen"
        );
        let hits = crate::search_index_fts::hybrid_search(
            &reopened_fts,
            &service.store,
            "gke_node_index",
            &provider,
            &format!("uniquepermit{suffix}"),
            5,
        )
        .await
        .unwrap();
        assert!(hits.iter().any(|hit| hit.node_id == request.index[0].node_id));

        drop(reopened_fts);
        drop(service);
        let cleanup_dsn = dsn
            .strip_prefix("postgresql+psycopg://")
            .map(|rest| format!("postgresql://{rest}"))
            .or_else(|| {
                dsn.strip_prefix("postgresql+psycopg2://")
                    .map(|rest| format!("postgresql://{rest}"))
            })
            .unwrap_or(dsn);
        let (client, connection) = tokio_postgres::connect(&cleanup_dsn, tokio_postgres::NoTls)
            .await
            .unwrap();
        let connection_task = tokio::spawn(connection);
        client
            .batch_execute(&format!(
                "DROP SCHEMA IF EXISTS \"{schema}_conversation\" CASCADE; \
                 DROP SCHEMA IF EXISTS \"{schema}_workflow\" CASCADE; \
                 DROP SCHEMA IF EXISTS \"{schema}_wisdom\" CASCADE; \
                 DROP SCHEMA IF EXISTS \"{schema}\" CASCADE"
            ))
            .await
            .unwrap();
        drop(client);
        connection_task.await.unwrap().unwrap();
        let _ = std::fs::remove_dir_all(root);
    }
