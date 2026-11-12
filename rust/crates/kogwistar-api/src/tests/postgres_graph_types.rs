#[test]
fn postgres_graph_type_stores_follow_python_schema_partitioning() {
    let service = PostgresRunApplicationService::from_dsn_with_graph_schemas(
        "postgresql://user:password@localhost/kogwistar",
        "kg_knowledge",
        "kg_conversation",
        "kg_workflow",
        "kg_wisdom",
    )
    .expect("valid PostgreSQL configuration should construct graph stores without connecting");

    assert_eq!(
        service.graph_store_for_type("knowledge").schema(),
        "kg_knowledge"
    );
    assert_eq!(
        service.graph_store_for_type("conversation").schema(),
        "kg_conversation"
    );
    assert_eq!(
        service.graph_store_for_type("workflow").schema(),
        "kg_workflow"
    );
    assert_eq!(service.graph_store_for_type("wisdom").schema(), "kg_wisdom");
    assert_eq!(
        service.graph_store_for_type("CONVERSATION").schema(),
        "kg_conversation"
    );

    let legacy = PostgresRunApplicationService::from_dsn_with_wisdom_schema(
        "postgresql://user:password@localhost/kogwistar",
        "legacy_graph",
        "legacy_wisdom",
    )
    .expect("legacy constructor remains supported");
    assert_eq!(legacy.store.schema(), "legacy_graph");
    assert_eq!(
        legacy.conversation_store.schema(),
        "legacy_graph_conversation"
    );
    assert_eq!(legacy.workflow_store.schema(), "legacy_graph_workflow");
    assert_eq!(legacy.wisdom_store.schema(), "legacy_wisdom");
}

#[tokio::test]
#[ignore = "requires KOGWISTAR_TEST_PG_DSN; uses disposable PostgreSQL schemas"]
async fn postgres_kg_load_persisted_matches_document_and_method_lookup_contract() {
    use axum::{body::Body, http::Request};
    use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
    use tower::ServiceExt;

    let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
        .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
    let schema = format!("rust_load_persisted_{}", uuid::Uuid::new_v4().simple());
    let service = PostgresRunApplicationService::from_dsn(&dsn, &schema).unwrap();
    service.ensure_schema().await.unwrap();
    service
        .store
        .create_graph_schema(2, kogwistar_store_postgres::GraphTableNames::default())
        .await
        .unwrap();

    let scope = GraphScope {
        namespace: "docs".to_owned(),
        workspace_id: None,
        graph_space: None,
    };
    let public_acl = |id: &str, grain: &str, target: &str, item: Option<&str>| {
        let mut metadata = serde_json::Map::from_iter([
            ("entity_type".to_owned(), json!("acl_record")),
            ("acl_truth_graph".to_owned(), json!("knowledge")),
            ("acl_target_grain".to_owned(), json!(grain)),
            ("acl_target_entity_id".to_owned(), json!(target)),
            ("acl_version".to_owned(), json!(1)),
            ("acl_mode".to_owned(), json!("public")),
        ]);
        if let Some(item) = item {
            metadata.insert("acl_target_item_id".to_owned(), json!(item));
        }
        let document = serde_json::to_string(&metadata).unwrap();
        GraphMutation {
            scope: scope.clone(),
            table: "gke_nodes".to_owned(),
            entity_kind: "node".to_owned(),
            event_id: format!("{id}-event"),
            op: "ADD".to_owned(),
            payload: json!({"id":id,"metadata":metadata}),
            record: GraphRecord {
                id: id.to_owned(),
                document: Some(document),
                metadata,
                embedding: None,
            },
            embedding_dim: 2,
        }
    };
    let mention_document = |id: &str| {
        json!({
            "id": id,
            "mentions": [{"spans": [{"doc_id":"doc-a","insertion_method":"parser"}]}]
        })
    };
    for id in ["node-a", "node-b"] {
        let document = mention_document(id);
        service
            .store
            .apply_graph_mutation(GraphMutation {
                scope: scope.clone(),
                table: "gke_nodes".to_owned(),
                entity_kind: "node".to_owned(),
                event_id: format!("{id}-event"),
                op: "ADD".to_owned(),
                payload: document.clone(),
                record: GraphRecord {
                    id: id.to_owned(),
                    document: Some(document.to_string()),
                    metadata: serde_json::Map::new(),
                    embedding: None,
                },
                embedding_dim: 2,
            })
            .await
            .unwrap();
        service
            .store
            .apply_graph_mutation(public_acl(&format!("acl-{id}"), "node", id, None))
            .await
            .unwrap();
        let grounding_id = format!("{id}::mention::0");
        let span_id = format!("{grounding_id}::span::0");
        service
            .store
            .apply_graph_mutation(public_acl(
                &format!("acl-{grounding_id}"),
                "grounding",
                id,
                Some(&grounding_id),
            ))
            .await
            .unwrap();
        service
            .store
            .apply_graph_mutation(public_acl(
                &format!("acl-{span_id}"),
                "span",
                id,
                Some(&span_id),
            ))
            .await
            .unwrap();
    }
    let latest = service
        .store
        .latest_entity_events("docs", "node", &["node-a".to_owned()])
        .await
        .unwrap();
    let event = &latest["node-a"];
    let canonical = read_canonical_entity_revision(
        &[EntityEvent {
            namespace: event.namespace.clone(),
            seq: event.seq,
            event_id: event.event_id.clone(),
            entity_kind: event.entity_kind.clone(),
            entity_id: event.entity_id.clone(),
            op: event.op.clone(),
            payload: serde_json::from_str(&event.payload_json).unwrap(),
        }],
        "docs",
        "node",
        "node-a",
    )
    .unwrap()
    .unwrap();
    let fingerprint_payload =
        kogwistar_engine::canonical_revision::canonical_revision_job_payload(&canonical);
    let fingerprint =
        serde_json::from_str::<Value>(&fingerprint_payload).unwrap()["source_fingerprint"]
            .as_str()
            .unwrap()
            .to_owned();
    let staged_document = mention_document("node-a").to_string();
    let staged_metadata = json!({
        "doc_id":"doc-a",
        "entity_type":"node",
        "lifecycle_status":"active",
        "namespace":"docs"
    });
    let staged_metadata_json = staged_metadata.to_string();
    let (stage_setup, stage_connection) = tokio_postgres::connect(&dsn, tokio_postgres::NoTls)
        .await
        .unwrap();
    let stage_task = tokio::spawn(stage_connection);
    stage_setup
        .execute(
            &format!("DELETE FROM \"{schema}\".gke_node_docs WHERE metadata->>'node_id'=$1"),
            &[&"node-a"],
        )
        .await
        .unwrap();
    stage_setup
        .execute(
            &format!(
                "INSERT INTO \"{schema}\".gke_stage1_projections \
                 (namespace,entity_kind,entity_id,document,metadata,source_fingerprint,revision,materialization_status,updated_at_ms) \
                 VALUES ($1,'node',$2,$3,$4::text::jsonb,$5,$6,'pending',0)"
            ),
            &[
                &"docs",
                &"node-a",
                &staged_document,
                &staged_metadata_json,
                &fingerprint,
                &canonical.revision,
            ],
        )
        .await
        .unwrap();
    drop(stage_setup);
    stage_task.await.unwrap().unwrap();
    let edge_document = json!({
        "id":"edge-a",
        "mentions":[{"spans":[{"doc_id":"doc-a","insertion_method":"parser"}]}],
        "source_ids":["node-a"],
        "target_ids":["node-b"],
        "doc_id":"doc-a"
    });
    service
        .store
        .apply_graph_mutation(GraphMutation {
            scope: scope.clone(),
            table: "gke_edges".to_owned(),
            entity_kind: "edge".to_owned(),
            event_id: "edge-a-event".to_owned(),
            op: "ADD".to_owned(),
            payload: edge_document.clone(),
            record: GraphRecord {
                id: "edge-a".to_owned(),
                document: Some(edge_document.to_string()),
                metadata: serde_json::Map::new(),
                embedding: None,
            },
            embedding_dim: 2,
        })
        .await
        .unwrap();
    service
        .store
        .apply_graph_mutation(public_acl("acl-edge-a", "edge", "edge-a", None))
        .await
        .unwrap();
    let edge_grounding = "edge-a::mention::0";
    let edge_span = "edge-a::mention::0::span::0";
    service
        .store
        .apply_graph_mutation(public_acl(
            "acl-edge-grounding",
            "grounding",
            "edge-a",
            Some(edge_grounding),
        ))
        .await
        .unwrap();
    service
        .store
        .apply_graph_mutation(public_acl(
            "acl-edge-span",
            "span",
            "edge-a",
            Some(edge_span),
        ))
        .await
        .unwrap();

    let bearer = encode(
        &Header::new(Algorithm::HS256),
        &json!({
            "sub":"graph-reader","role":"ro","ns":["docs"],"storage_ns":"docs",
            "capabilities":["read_graph"],"exp":4102444800_u64
        }),
        &EncodingKey::from_secret(b"load-persisted-mcp-test"),
    )
    .unwrap();
    let app = router_with_application(
        ApiState {
            health: snapshot(),
            required_roles: Vec::new(),
            implementation: ImplementationSnapshot::default(),
            auth: AuthConfig {
                mode: "jwt".to_owned(),
                algorithm: Some("HS256".to_owned()),
                key: Some("load-persisted-mcp-test".to_owned()),
                ..AuthConfig::default()
            },
        },
        Arc::new(service.clone()),
    );
    assert!(service.supports_mcp_tool_for_principal(
        "root",
        "kg_load_persisted",
        &json!({"sub":"graph-reader","role":"ro","ns":["docs"],"storage_ns":"docs","capabilities":["read_graph"]})
    ));
    assert!(service.supports_mcp_tool_for_principal(
        "root",
        "kg_propose_bruteforce",
        &json!({"sub":"graph-reader","role":"ro","ns":["docs"],"storage_ns":"docs","capabilities":["read_graph"]})
    ));
    assert!(!service.supports_mcp_tool_for_principal(
        "root",
        "kg_propose_bruteforce",
        &json!({"sub":"graph-reader","role":"ro","ns":["docs"],"storage_ns":"docs","capabilities":[]})
    ));
    let call = |id: &str, inp: Value| {
        Request::builder()
            .method("POST")
            .uri("/mcp")
            .header("content-type", "application/json")
            .header("authorization", format!("Bearer {bearer}"))
            .body(Body::from(
                json!({
                    "jsonrpc":"2.0","id":id,"method":"tools/call",
                    "params":{"name":"kg_load_persisted","arguments":{"inp":inp}}
                })
                .to_string(),
            ))
            .unwrap()
    };
    let short_response = app
        .clone()
        .oneshot(call("short", json!({"doc_ids":["doc-a"]})))
        .await
        .unwrap();
    assert_eq!(short_response.status(), StatusCode::OK);
    let short_body = axum::body::to_bytes(short_response.into_body(), 65_536)
        .await
        .unwrap();
    let short_json: Value = serde_json::from_slice(&short_body).unwrap();
    assert!(
        short_json.get("result").is_some(),
        "MCP response: {short_json}"
    );
    let map_key = crate::short_id_store::short_id_map_key_for_bearer(Some(&bearer));
    let decode_ids = |values: &[Value]| {
        values
            .iter()
            .map(|value| {
                crate::short_id_store::canonical_id_for(
                    ".shortids",
                    &map_key,
                    value.as_str().unwrap(),
                )
                .unwrap()
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(
        decode_ids(short_json["result"]["node_ids"].as_array().unwrap()),
        ["node-a", "node-b"]
    );
    assert_eq!(
        decode_ids(short_json["result"]["edge_ids"].as_array().unwrap()),
        ["edge-a"]
    );

    let canonical_response = app
        .clone()
        .oneshot(call(
            "canonical",
            json!({"doc_ids":["doc-a"],"insertion_method":"parser","sid":false}),
        ))
        .await
        .unwrap();
    let canonical_body = axum::body::to_bytes(canonical_response.into_body(), 65_536)
        .await
        .unwrap();
    let canonical_json: Value = serde_json::from_slice(&canonical_body).unwrap();
    assert_eq!(
        canonical_json["result"]["node_ids"],
        json!(["node-a", "node-b"])
    );
    assert_eq!(canonical_json["result"]["edge_ids"], json!(["edge-a"]));

    let proposal_call = |id: &str, input: Value| {
        Request::builder()
            .method("POST")
            .uri("/mcp")
            .header("content-type", "application/json")
            .header("authorization", format!("Bearer {bearer}"))
            .body(Body::from(
                json!({
                    "jsonrpc":"2.0","id":id,"method":"tools/call",
                    "params":{"name":"kg_propose_bruteforce","arguments":{"inp":input}}
                })
                .to_string(),
            ))
            .unwrap()
    };
    let brute_force_response = app
        .clone()
        .oneshot(proposal_call(
            "brute",
            json!({"pair_kind":"node_node","allowed_docs":["doc-a"]}),
        ))
        .await
        .unwrap();
    let brute_force_body = axum::body::to_bytes(brute_force_response.into_body(), 65_536)
        .await
        .unwrap();
    let brute_force_json: Value = serde_json::from_slice(&brute_force_body).unwrap();
    assert!(
        brute_force_json.get("result").is_some(),
        "MCP response: {brute_force_json}"
    );
    assert_eq!(
        brute_force_json["result"]["pairs"],
        json!([{
            "left_id":"node-a","left_kind":"node",
            "right_id":"node-b","right_kind":"node"
        }])
    );
    let mixed_response = app
        .clone()
        .oneshot(proposal_call(
            "mixed",
            json!({"pair_kind":"any_any","anchor_doc_id":"doc-a"}),
        ))
        .await
        .unwrap();
    let mixed_body = axum::body::to_bytes(mixed_response.into_body(), 65_536)
        .await
        .unwrap();
    let mixed_json: Value = serde_json::from_slice(&mixed_body).unwrap();
    assert!(
        mixed_json.get("result").is_some(),
        "MCP response: {mixed_json}"
    );
    assert_eq!(mixed_json["result"]["pairs"].as_array().unwrap().len(), 3);
    let empty_response = app
        .clone()
        .oneshot(proposal_call(
            "empty-doc",
            json!({"pair_kind":"node_node","allowed_docs":["doc-empty"]}),
        ))
        .await
        .unwrap();
    let empty_body = axum::body::to_bytes(empty_response.into_body(), 65_536)
        .await
        .unwrap();
    let empty_json: Value = serde_json::from_slice(&empty_body).unwrap();
    assert_eq!(empty_json["result"]["pairs"], json!([]));

    let (stale_setup, stale_connection) = tokio_postgres::connect(&dsn, tokio_postgres::NoTls)
        .await
        .unwrap();
    let stale_task = tokio::spawn(stale_connection);
    stale_setup
        .execute(
            &format!(
                "UPDATE \"{schema}\".gke_stage1_projections \
                 SET source_fingerprint='stale' WHERE entity_id='node-a'"
            ),
            &[],
        )
        .await
        .unwrap();
    drop(stale_setup);
    stale_task.await.unwrap().unwrap();
    let stale_response = app
        .clone()
        .oneshot(proposal_call(
            "stale-stage1",
            json!({"pair_kind":"node_node","allowed_docs":["doc-a"]}),
        ))
        .await
        .unwrap();
    let stale_body = axum::body::to_bytes(stale_response.into_body(), 65_536)
        .await
        .unwrap();
    let stale_json: Value = serde_json::from_slice(&stale_body).unwrap();
    assert_eq!(stale_json["result"]["pairs"], json!([]));

    drop(app);
    drop(service);
    let (cleanup, connection) = tokio_postgres::connect(&dsn, tokio_postgres::NoTls)
        .await
        .unwrap();
    let cleanup_task = tokio::spawn(connection);
    cleanup
        .batch_execute(&format!("DROP SCHEMA \"{schema}\" CASCADE"))
        .await
        .unwrap();
    drop(cleanup);
    cleanup_task.await.unwrap().unwrap();
}

#[tokio::test]
#[ignore = "requires KOGWISTAR_TEST_PG_DSN; uses disposable PostgreSQL schemas"]
async fn postgres_graph_mutation_batch_uses_configured_graph_type_store() {
    let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
        .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
    let base = format!("rust_graph_role_{}", uuid::Uuid::new_v4().simple());
    let schemas = [
        format!("{base}_knowledge"),
        format!("{base}_conversation"),
        format!("{base}_workflow"),
        format!("{base}_wisdom"),
    ];
    let service = PostgresRunApplicationService::from_dsn_with_graph_schemas(
        &dsn,
        &schemas[0],
        &schemas[1],
        &schemas[2],
        &schemas[3],
    )
    .unwrap()
    .with_graph_role("conversation")
    .unwrap();
    service.ensure_schema().await.unwrap();
    for store in [
        &service.store,
        &service.conversation_store,
        &service.workflow_store,
        &service.wisdom_store,
    ] {
        store
            .create_graph_schema(2, kogwistar_store_postgres::GraphTableNames::default())
            .await
            .unwrap();
    }

    let mut batch: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/graph-mutation-batch-v1.json"
    )))
    .unwrap();
    batch["mutations"][0]["scope"] = json!({
        "namespace": "tenant-a",
        "workspace_id": "ws",
        "graph_space": "facts"
    });
    batch["mutations"][0]["record"]["id"] = json!("conversation-node");
    batch["mutations"][0]["payload"]["id"] = json!("conversation-node");
    batch["mutations"][0]["event_id"] = json!("conversation-node-event");

    let response = ApplicationService::execute(
        &service,
        ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: INTERNAL_GRAPH_MUTATIONS_ROUTE.to_owned(),
            body: serde_json::to_vec(&batch).unwrap(),
            principal: json!({
                "sub": "alice",
                "role": "rw",
                "ns": ["tenant-a"],
                "storage_ns": "tenant-a",
                "workspace_id": "ws",
                "graph_space": "facts",
                "capabilities": ["write_graph"]
            }),
        },
    )
    .await;
    assert_eq!(
        response.status,
        StatusCode::OK.as_u16(),
        "{}",
        String::from_utf8_lossy(&response.body)
    );

    let scope = GraphScope {
        namespace: "tenant-a".to_owned(),
        workspace_id: Some("ws".to_owned()),
        graph_space: Some("facts".to_owned()),
    };
    let read = |id: &str| GraphProjectionRead {
        scope: scope.clone(),
        table: "gke_nodes".to_owned(),
        ids: Some(vec![id.to_owned()]),
        metadata: MetadataFilter::default(),
        limit: 1,
    };
    let conversation_rows = service
        .conversation_store
        .graph_projection_records(read("conversation-node"))
        .await
        .unwrap();
    let knowledge_rows = service
        .store
        .graph_projection_records(read("conversation-node"))
        .await
        .unwrap();
    assert_eq!(conversation_rows.len(), 1);
    assert!(knowledge_rows.is_empty());

    drop(service);
    let (cleanup, connection) = tokio_postgres::connect(&dsn, tokio_postgres::NoTls)
        .await
        .unwrap();
    let cleanup_task = tokio::spawn(connection);
    cleanup
        .batch_execute(
            &schemas
                .iter()
                .map(|schema| format!("DROP SCHEMA \"{schema}\" CASCADE"))
                .collect::<Vec<_>>()
                .join("; "),
        )
        .await
        .unwrap();
    drop(cleanup);
    cleanup_task.await.unwrap().unwrap();
}

#[tokio::test]
#[ignore = "requires KOGWISTAR_TEST_PG_DSN; uses disposable PostgreSQL schemas"]
async fn postgres_embedding_worker_uses_configured_graph_type_store() {
    struct FixedEmbeddingProvider {
        profile: kogwistar_engine::embedding::EmbeddingProfile,
        inputs: Arc<Mutex<Vec<String>>>,
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
                self.inputs.lock().unwrap().extend(texts.iter().cloned());
                Ok(texts.iter().map(|_| vec![0.6, 0.8]).collect())
            })
        }
    }

    let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
        .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
    let base = format!("rust_graph_role_worker_{}", uuid::Uuid::new_v4().simple());
    let schemas = [
        format!("{base}_knowledge"),
        format!("{base}_conversation"),
        format!("{base}_workflow"),
        format!("{base}_wisdom"),
    ];
    let inputs = Arc::new(Mutex::new(Vec::new()));
    let profile = kogwistar_engine::embedding::EmbeddingProfile {
        provider: "test".to_owned(),
        model: "graph-role-worker-test".to_owned(),
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
    };
    let service = PostgresRunApplicationService::from_dsn_with_graph_schemas(
        &dsn,
        &schemas[0],
        &schemas[1],
        &schemas[2],
        &schemas[3],
    )
    .unwrap()
    .with_graph_role("conversation")
    .unwrap()
    .with_embedding_provider(Arc::new(FixedEmbeddingProvider {
        profile,
        inputs: inputs.clone(),
    }));
    service.ensure_graph_type_schemas().await.unwrap();
    for store in [
        &service.store,
        &service.conversation_store,
        &service.workflow_store,
        &service.wisdom_store,
    ] {
        store
            .create_graph_schema(2, kogwistar_store_postgres::GraphTableNames::default())
            .await
            .unwrap();
    }

    let scope = GraphScope {
        namespace: "tenant-a".to_owned(),
        workspace_id: Some("workspace-a".to_owned()),
        graph_space: Some("conversation".to_owned()),
    };
    let applied = service
        .conversation_store
        .apply_graph_mutations_atomic(vec![GraphMutation {
            scope: scope.clone(),
            table: "gke_nodes".to_owned(),
            entity_kind: "node".to_owned(),
            event_id: "conversation-embedding-event".to_owned(),
            op: "ADD".to_owned(),
            payload: json!({"id":"conversation-embedding-node","label":"Conversation source"}),
            record: GraphRecord {
                id: "conversation-embedding-node".to_owned(),
                document: Some("Conversation embedding source".to_owned()),
                metadata: serde_json::Map::from_iter([
                    ("namespace".to_owned(), json!("tenant-a")),
                    ("workspace_id".to_owned(), json!("workspace-a")),
                    ("graph_space".to_owned(), json!("conversation")),
                ]),
                embedding: None,
            },
            embedding_dim: 2,
        }])
        .await
        .unwrap();
    let revision = kogwistar_engine::canonical_revision::read_canonical_entity_revision(
        &[applied[0].event.clone()],
        "tenant-a",
        "node",
        "conversation-embedding-node",
    )
    .unwrap()
    .unwrap();
    service
        .conversation_store
        .enqueue_index_job(NewIndexJob {
            job_id: "conversation-embedding-job".to_owned(),
            namespace: "tenant-a".to_owned(),
            entity_kind: "node".to_owned(),
            entity_id: "conversation-embedding-node".to_owned(),
            index_kind: "node_embedding".to_owned(),
            op: "UPSERT".to_owned(),
            payload_json: Some(
                kogwistar_engine::canonical_revision::canonical_revision_job_payload(&revision),
            ),
            max_retries: 3,
        })
        .await
        .unwrap();

    let summary = service
        .run_embedding_projection_jobs_once(10, 30, Some("tenant-a"))
        .await
        .unwrap();
    let job = service
        .conversation_store
        .list_index_jobs(
            Some("tenant-a"),
            None,
            Some("node"),
            Some("conversation-embedding-node"),
            Some("node_embedding"),
            10,
        )
        .await
        .unwrap();
    assert_eq!(summary.claimed, 1);
    assert_eq!(
        summary.projected, 1,
        "projection summary: {summary:?}; jobs: {job:#?}"
    );
    assert_eq!(summary.retried, 0);
    assert_eq!(
        inputs.lock().unwrap().as_slice(),
        ["Conversation embedding source"]
    );
    assert_eq!(job.len(), 1);
    assert_eq!(job[0].status, "DONE");
    let promoted = service
        .conversation_store
        .graph_projection_records(GraphProjectionRead {
            scope,
            table: "gke_nodes".to_owned(),
            ids: Some(vec!["conversation-embedding-node".to_owned()]),
            metadata: MetadataFilter::default(),
            limit: 1,
        })
        .await
        .unwrap();
    assert_eq!(promoted.len(), 1);
    assert_eq!(
        promoted[0].embedding.as_deref(),
        Some([0.6_f32, 0.8].as_slice())
    );
    let knowledge_jobs = service
        .store
        .list_index_jobs(
            Some("tenant-a"),
            None,
            Some("node"),
            Some("conversation-embedding-node"),
            Some("node_embedding"),
            10,
        )
        .await
        .unwrap();
    assert!(knowledge_jobs.is_empty());

    drop(service);
    let (cleanup, connection) = tokio_postgres::connect(&dsn, tokio_postgres::NoTls)
        .await
        .unwrap();
    let cleanup_task = tokio::spawn(connection);
    cleanup
        .batch_execute(
            &schemas
                .iter()
                .map(|schema| format!("DROP SCHEMA \"{schema}\" CASCADE"))
                .collect::<Vec<_>>()
                .join("; "),
        )
        .await
        .unwrap();
    drop(cleanup);
    cleanup_task.await.unwrap().unwrap();
}
