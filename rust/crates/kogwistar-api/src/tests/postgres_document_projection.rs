    #[tokio::test]
    #[ignore = "requires KOGWISTAR_TEST_PG_DSN; run explicitly with --include-ignored"]
    async fn postgres_document_projection_route_persists_reopens_and_retries_when_dsn_available() {
        struct FixedEmbeddingProvider {
            profile: kogwistar_engine::embedding::EmbeddingProfile,
            inputs: Arc<Mutex<Vec<String>>>,
            fail_next: Arc<std::sync::atomic::AtomicBool>,
        }
        impl kogwistar_engine::embedding::EmbeddingProvider for FixedEmbeddingProvider {
            fn profile(&self) -> &kogwistar_engine::embedding::EmbeddingProfile {
                &self.profile
            }

            fn embed<'a>(
                &'a self,
                texts: &'a [String],
            ) -> kogwistar_engine::embedding::EmbeddingFuture<'a> {
                Box::pin(async move {
                    if self
                        .fail_next
                        .swap(false, std::sync::atomic::Ordering::SeqCst)
                    {
                        return Err(kogwistar_engine::embedding::EmbeddingError::Request(
                            "synthetic provider failure".to_owned(),
                        ));
                    }
                    self.inputs.lock().unwrap().extend(texts.iter().cloned());
                    Ok(texts.iter().map(|_| vec![0.6, 0.8, 0.0]).collect())
                })
            }
        }

        let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
            .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
        let schema = format!("rust_api_document_{}", uuid::Uuid::new_v4().simple());
        let (mut client, connection) = tokio_postgres::connect(&dsn, tokio_postgres::NoTls)
            .await
            .unwrap();
        tokio::spawn(async move {
            let _ = connection.await;
        });
        let transaction = client.transaction().await.unwrap();
        transaction
            .query_one(
                "SELECT pg_advisory_xact_lock(hashtext('kogwistar.pgvector.extension'))",
                &[],
            )
            .await
            .unwrap();
        transaction
            .batch_execute(&format!(
                r#"CREATE EXTENSION IF NOT EXISTS vector;
                CREATE SCHEMA "{schema}";
                CREATE TABLE "{schema}"."gke_documents" (
                    id TEXT PRIMARY KEY,
                    document TEXT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    embedding vector(3) NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );"#
            ))
            .await
            .unwrap();
        transaction.commit().await.unwrap();

        let embedding_inputs = Arc::new(Mutex::new(Vec::new()));
        let fail_next_embedding = Arc::new(std::sync::atomic::AtomicBool::new(true));
        let service = PostgresRunApplicationService::from_dsn(&dsn, &schema)
            .unwrap()
            .with_embedding_provider(Arc::new(FixedEmbeddingProvider {
                profile: kogwistar_engine::embedding::EmbeddingProfile {
                    provider: "test".to_owned(),
                    model: "document-route-fixture".to_owned(),
                    dimension: 3,
                    similarity_metric: "cosine".to_owned(),
                    endpoint_fingerprint: None,
                    max_sequence_length: None,
                    crop_token_budget: None,
                    tokenizer_fingerprint: None,
                    crop_policy: None,
                    embedding_kind: None,
                    model_revision: None,
                    preprocessing_fingerprint: None,
                    max_image_patches: None,
                },
                inputs: embedding_inputs.clone(),
                fail_next: fail_next_embedding,
            }));
        service.ensure_schema().await.unwrap();
        service
            .store
            .create_graph_schema(
                3,
                kogwistar_store_postgres::GraphTableNames::default(),
            )
            .await
            .unwrap();
        let key = "document-route-test-secret";
        let token = encode(
            &Header::new(Algorithm::HS256),
            &json!({
                "sub": "alice",
                "role": "rw",
                "roles": ["reader", "rw"],
                "ns": "workflow",
                "storage_ns": "docs",
                "workspace": "workspace-a",
                "graph_space": "knowledge",
                "capabilities": ["write_graph"],
                "exp": 4102444800_u64
            }),
            &EncodingKey::from_secret(key.as_bytes()),
        )
        .unwrap();
        let make_app = || {
            router_with_application(
                ApiState {
                    health: snapshot(),
                    required_roles: vec!["reader".to_owned()],
                    implementation: ImplementationSnapshot::default(),
                    auth: AuthConfig {
                        mode: "jwt".to_owned(),
                        algorithm: Some("HS256".to_owned()),
                        key: Some(key.to_owned()),
                        ..AuthConfig::default()
                    },
                },
                Arc::new(service.clone()),
            )
        };
        let send = |app: Router, body: Value| {
            let token = token.clone();
            async move {
                app.oneshot(
                    Request::builder()
                        .method("POST")
                        .uri(PUBLIC_DOCUMENT_UPSERT_ROUTE)
                        .header("content-type", "application/json")
                        .header("authorization", format!("Bearer {token}"))
                        .body(Body::from(serde_json::to_vec(&body).unwrap()))
                        .unwrap(),
                )
                .await
                .unwrap()
            }
        };

        let first_payload: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/document-upsert-v1.json"
        )))
        .unwrap();
        let failed = send(make_app(), first_payload.clone()).await;
        assert_eq!(failed.status(), StatusCode::BAD_GATEWAY);
        let failed_body = to_bytes(failed.into_body(), 4096).await.unwrap();
        let failure: Value = serde_json::from_slice(&failed_body).unwrap();
        assert_eq!(failure["code"], "KOGWISTAR_EMBEDDING_FAILED");
        let unpersisted = service
            .store
            .graph_projection_records(GraphProjectionRead {
                scope: GraphScope {
                    namespace: "docs".to_owned(),
                    workspace_id: Some("workspace-a".to_owned()),
                    graph_space: Some("knowledge".to_owned()),
                },
                table: "gke_documents".to_owned(),
                ids: Some(vec![first_payload["doc_id"].as_str().unwrap().to_owned()]),
                metadata: MetadataFilter::default(),
                limit: 10,
            })
            .await
            .unwrap();
        assert!(unpersisted.is_empty());

        let first = send(make_app(), first_payload.clone()).await;
        let first_status = first.status();
        let first_body = to_bytes(first.into_body(), 4096).await.unwrap();
        assert_eq!(
            first_status,
            StatusCode::OK,
            "document upsert failed: {}",
            String::from_utf8_lossy(&first_body)
        );
        assert_eq!(
            serde_json::from_slice::<Value>(&first_body).unwrap(),
            json!({"status": "ok"})
        );
        let mut retry_payload = first_payload.clone();
        retry_payload["content"] = json!("updated body");
        let retry = send(make_app(), retry_payload).await;
        assert_eq!(retry.status(), StatusCode::OK);

        let reopened = PostgresRunApplicationService::from_dsn(&dsn, &schema).unwrap();
        let rows = reopened
            .store
            .graph_projection_records(kogwistar_store::GraphProjectionRead {
                scope: GraphScope {
                    namespace: "docs".to_owned(),
                    workspace_id: Some("workspace-a".to_owned()),
                    graph_space: Some("knowledge".to_owned()),
                },
                table: "gke_documents".to_owned(),
                ids: Some(vec![first_payload["doc_id"].as_str().unwrap().to_owned()]),
                metadata: MetadataFilter::default(),
                limit: 10,
            })
            .await
            .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].document.as_deref(), Some("updated body"));
        assert_eq!(rows[0].metadata["namespace"], "docs");
        assert_eq!(rows[0].metadata["workspace_id"], "workspace-a");
        assert_eq!(rows[0].metadata["graph_space"], "knowledge");
        assert_eq!(
            rows[0].metadata["insertion_method"],
            first_payload["insertion_method"]
        );
        assert_eq!(rows[0].embedding.as_deref(), Some(&[0.6, 0.8, 0.0][..]));
        assert_eq!(
            *embedding_inputs.lock().unwrap(),
            vec![
                first_payload["content"].as_str().unwrap().to_owned(),
                "updated body".to_owned()
            ]
        );

        let (cleanup, cleanup_connection) = tokio_postgres::connect(&dsn, tokio_postgres::NoTls)
            .await
            .unwrap();
        tokio::spawn(async move {
            let _ = cleanup_connection.await;
        });
        cleanup
            .batch_execute(&format!("DROP SCHEMA \"{schema}\" CASCADE"))
            .await
            .unwrap();
    }
