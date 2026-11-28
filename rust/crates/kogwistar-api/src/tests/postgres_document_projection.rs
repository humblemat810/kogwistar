    #[tokio::test]
    #[ignore = "requires KOGWISTAR_TEST_PG_DSN; run explicitly with --include-ignored"]
    async fn postgres_document_projection_route_persists_reopens_and_retries_when_dsn_available() {
        use crate::change_events::ChangeEventPublisher;

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
        let (change_sender, mut change_events) = tokio::sync::mpsc::unbounded_channel();
        let change_app = Router::new().route(
            "/ingest",
            post(move |Json(event): Json<Value>| async move {
                let _ = change_sender.send(event);
                StatusCode::OK
            }),
        );
        let change_listener = bind_reachable_loopback_listener().await;
        let change_endpoint = format!("http://{}", change_listener.local_addr().unwrap());
        tokio::spawn(async move {
            axum::serve(change_listener, change_app).await.unwrap();
        });

        let mut service = PostgresRunApplicationService::from_dsn(&dsn, &schema)
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
        service.change_events = ChangeEventPublisher::new(Some(change_endpoint), 8);
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
        let http_contract: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/document-upsert-http-v1.json"
        )))
        .unwrap();
        let method = http_contract["method"].as_str().unwrap().to_owned();
        let path = http_contract["path"].as_str().unwrap().to_owned();
        let send = |app: Router, body: Value| {
            let token = token.clone();
            let method = method.clone();
            let path = path.clone();
            async move {
                app.oneshot(
                    Request::builder()
                        .method(method.as_str())
                        .uri(path)
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
        assert!(tokio::time::timeout(
            std::time::Duration::from_millis(100),
            change_events.recv()
        )
        .await
        .is_err(), "embedding failure must not publish a document event");
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
            first_status.as_u16(),
            http_contract["success_status"].as_u64().unwrap() as u16,
            "document upsert failed: {}",
            String::from_utf8_lossy(&first_body)
        );
        assert_eq!(
            serde_json::from_slice::<Value>(&first_body).unwrap(),
            http_contract["success_response"]
        );
        let first_event = tokio::time::timeout(
            std::time::Duration::from_secs(3),
            change_events.recv(),
        )
        .await
        .expect("PostgreSQL upsert event delivery should be bounded")
        .expect("successful PostgreSQL upsert should publish an event");
        assert_eq!(first_event["op"], "doc.upsert");
        assert_eq!(first_event["seq"], 1);
        assert_eq!(first_event["entity"]["id"], first_payload["doc_id"]);
        assert!(first_event["entity"]["url"].is_null());
        assert_eq!(first_event["payload"]["content"], first_payload["content"]);

        let trigger_ddl = format!(
            r#"CREATE FUNCTION "{schema}".reject_test_document_write()
                   RETURNS trigger LANGUAGE plpgsql AS $$
               BEGIN
                   IF NEW.id = 'storage-fail-doc' THEN
                       RAISE EXCEPTION 'synthetic storage failure';
                   END IF;
                   RETURN NEW;
               END;
               $$;
               CREATE TRIGGER reject_test_document_write
                   BEFORE INSERT OR UPDATE ON "{schema}"."gke_documents"
                   FOR EACH ROW EXECUTE FUNCTION "{schema}".reject_test_document_write();"#
        );
        client.batch_execute(&trigger_ddl).await.unwrap();

        let mut storage_failure_payload = first_payload.clone();
        storage_failure_payload["doc_id"] = json!("storage-fail-doc");
        let storage_failure = send(make_app(), storage_failure_payload).await;
        assert_eq!(storage_failure.status(), StatusCode::CONFLICT);
        assert!(tokio::time::timeout(
            std::time::Duration::from_millis(100),
            change_events.recv()
        )
        .await
        .is_err(), "storage failure must not publish a document event");

        let mut retry_payload = first_payload.clone();
        retry_payload["content"] = json!("updated body");
        let retry = send(make_app(), retry_payload).await;
        assert_eq!(retry.status(), StatusCode::OK);
        let retry_event = tokio::time::timeout(
            std::time::Duration::from_secs(3),
            change_events.recv(),
        )
        .await
        .expect("retry event delivery should be bounded")
        .expect("successful retry should publish an event");
        assert_eq!(retry_event["seq"], 2);
        assert_eq!(retry_event["payload"]["content"], "updated body");

        let ocr_content = json!({
            "scan.pdf": [{
                "pdf_page_num": 1,
                "OCR_text_clusters": [{"text": "Hello"}],
                "non_text_objects": []
            }]
        });
        let ocr_payload = json!({
            "doc_id": "ocr-document-route-test",
            "doc_type": "ocr",
            "insertion_method": "ocr-parser",
            "content": serde_json::to_string(&ocr_content).unwrap()
        });
        let ocr_response = send(make_app(), ocr_payload).await;
        let ocr_status = ocr_response.status();
        let ocr_body = to_bytes(ocr_response.into_body(), 4096).await.unwrap();
        assert_eq!(
            ocr_status,
            StatusCode::OK,
            "OCR document upsert failed: {}",
            String::from_utf8_lossy(&ocr_body)
        );
        let ocr_event = tokio::time::timeout(
            std::time::Duration::from_secs(3),
            change_events.recv(),
        )
        .await
        .expect("OCR event delivery should be bounded")
        .expect("successful OCR upsert should publish an event");
        assert_eq!(ocr_event["seq"], 3);
        assert_eq!(ocr_event["entity"]["id"], "ocr-document-route-test");
        assert_eq!(ocr_event["payload"]["type"], "ocr_document");
        assert_eq!(ocr_event["payload"]["source_map"]["p1_c0"]["text"], "Hello");

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
        let ocr_rows = reopened
            .store
            .graph_projection_records(kogwistar_store::GraphProjectionRead {
                scope: GraphScope {
                    namespace: "docs".to_owned(),
                    workspace_id: Some("workspace-a".to_owned()),
                    graph_space: Some("knowledge".to_owned()),
                },
                table: "gke_documents".to_owned(),
                ids: Some(vec!["ocr-document-route-test".to_owned()]),
                metadata: MetadataFilter::default(),
                limit: 10,
            })
            .await
            .unwrap();
        assert_eq!(ocr_rows.len(), 1);
        assert_eq!(
            ocr_rows[0].document.as_deref(),
            Some("{'scan.pdf': [{'pdf_page_num': 1, 'OCR_text_clusters': [{'text': 'Hello', 'id': 'p1_c0'}], 'non_text_objects': []}]}")
        );
        assert_eq!(ocr_rows[0].metadata["doc_type"], "ocr_document");
        assert_eq!(ocr_rows[0].metadata["insertion_method"], "ocr-parser");
        assert_eq!(
            *embedding_inputs.lock().unwrap(),
            vec![
                first_payload["content"].as_str().unwrap().to_owned(),
                first_payload["content"].as_str().unwrap().to_owned(),
                "updated body".to_owned(),
                "{'scan.pdf': [{'pdf_page_num': 1, 'OCR_text_clusters': [{'text': 'Hello', 'id': 'p1_c0'}], 'non_text_objects': []}]}".to_owned()
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
