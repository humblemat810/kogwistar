    #[test]
    fn graph_neighbor_request_is_typed_and_closed_to_unknown_fields() {
        let request: GraphNeighborsRequestV1 =
            serde_json::from_str(r#"{"rid":" n1 ","doc_id":"doc-1"}"#).unwrap();
        assert_eq!(request.rid, " n1 ");
        assert_eq!(request.doc_id.as_deref(), Some("doc-1"));
        assert_eq!(request.truth_graph, "knowledge");
        assert!(
            serde_json::from_str::<GraphNeighborsRequestV1>(r#"{"rid":"n1","write":true}"#)
                .is_err()
        );
        assert!(parse_graph_neighbors_request(br#"{"rid":""}"#).is_err());
    }

    #[test]
    fn graph_vector_query_contract_is_batched_bounded_and_closed() {
        let input = parse_graph_vector_query_request(
            br#"{"entity_kind":"node","query_embeddings":[[0.1,0.2],[0.3,0.4]],"metric":"ip","where":{"kind":"note"},"n_results":5,"include":["metadatas","distances"]}"#,
        )
        .unwrap();
        assert_eq!(input.entity_kind, "node");
        assert_eq!(input.query_embeddings.len(), 2);
        assert_eq!(input.metric, DistanceMetric::InnerProduct);
        assert_eq!(input.n_results, 5);
        assert_eq!(input.where_filter.equals.get("kind"), Some(&json!("note")));
        assert_eq!(input.include, ["metadatas", "distances"]);

        assert!(parse_graph_vector_query_request(
            br#"{"entity_kind":"document","query_embeddings":[[0.1]],"metric":"cosine"}"#
        )
        .is_err());
        assert!(parse_graph_vector_query_request(
            br#"{"entity_kind":"node","query_embeddings":[[]],"metric":"cosine"}"#
        )
        .is_err());
        assert!(parse_graph_vector_query_request(
            br#"{"entity_kind":"node","query_embeddings":[[0.1]],"metric":"cosine","where":{"$gt":2}}"#
        )
        .is_err());
        assert!(parse_graph_vector_query_request(
            br#"{"entity_kind":"node","query_embeddings":[[0.1]],"metric":"cosine","unknown":true}"#
        )
        .is_err());
        assert!(parse_graph_vector_query_request(
            br#"{"entity_kind":"node","query_embeddings":[[0.1]],"metric":"cosine","include":["implementation_authority"]}"#
        )
        .is_err());
    }

    #[tokio::test]
    async fn graph_vector_query_route_authorizes_and_validates_before_storage() {
        let service = PostgresRunApplicationService::from_dsn(
            "postgres://unused:unused@127.0.0.1:1/unused",
            "graph_vector_query_contract",
        )
        .unwrap();
        let denied = service
            .execute_async(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/internal/graph/vector-query".to_owned(),
                body: br#"{"entity_kind":"node","query_embeddings":[[0.1]],"metric":"cosine"}"#.to_vec(),
                principal: json!({"sub":"alice","capabilities":[]}),
            })
            .await;
        assert_eq!(denied.status, StatusCode::FORBIDDEN.as_u16());

        let malformed = service
            .execute_async(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/internal/graph/vector-query".to_owned(),
                body: br#"{"entity_kind":"node","query_embeddings":[[0.1]],"metric":"cosine","extra":true}"#.to_vec(),
                principal: json!({"sub":"alice","capabilities":["read_graph"]}),
            })
            .await;
        assert_eq!(malformed.status, StatusCode::UNPROCESSABLE_ENTITY.as_u16());
    }

    #[tokio::test]
    #[ignore = "requires KOGWISTAR_TEST_PG_DSN; run explicitly against a disposable database"]
    async fn postgres_graph_vector_query_is_scoped_acl_filtered_and_rank_stable_when_dsn_available()
    {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
        use tower::ServiceExt;

        let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
            .expect("ignored test requires KOGWISTAR_TEST_PG_DSN");
        let schema = format!("rust_vector_query_{}", uuid::Uuid::new_v4().simple());
        let service = PostgresRunApplicationService::from_dsn(&dsn, &schema).unwrap();
        assert!(
            service.supports_mcp_tool("root", "propose_vector"),
            "PostgreSQL graph authority implements vector proposals"
        );
        service.ensure_schema().await.unwrap();
        service
            .store
            .create_graph_schema(2, kogwistar_store_postgres::GraphTableNames::default())
            .await
            .unwrap();
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/graph-acl-vector-query-v1.json"
        )))
        .unwrap();
        let scope = GraphScope {
            namespace: fixture["scope"]["namespace"].as_str().unwrap().to_owned(),
            workspace_id: fixture["scope"]["workspace_id"]
                .as_str()
                .map(str::to_owned),
            graph_space: None,
        };
        for (entity_kind, table, entities) in [
            ("node", "gke_nodes", &fixture["nodes"]),
            ("edge", "gke_edges", &fixture["edges"]),
        ] {
            for entity in entities.as_array().unwrap() {
                let entity_id = entity["id"].as_str().unwrap();
                let mut metadata = Map::new();
                metadata.insert("entity_type".to_owned(), json!(entity_kind));
                if entity_kind == "node" {
                    metadata.insert("label".to_owned(), json!(entity_id));
                } else {
                    metadata.insert("source_ids".to_owned(), entity["source_ids"].clone());
                    metadata.insert("target_ids".to_owned(), entity["target_ids"].clone());
                }
                if let Some(lifecycle_status) = entity["lifecycle_status"].as_str() {
                    metadata.insert("lifecycle_status".to_owned(), json!(lifecycle_status));
                }
                if let Some(redirect_to_id) = entity["redirect_to_id"].as_str() {
                    metadata.insert("redirect_to_id".to_owned(), json!(redirect_to_id));
                }
                service
                    .store
                    .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                        scope: scope.clone(),
                        table: table.to_owned(),
                        record: GraphRecord {
                            id: entity_id.to_owned(),
                            document: Some(entity_id.to_owned()),
                            metadata,
                            embedding: Some(
                                serde_json::from_value(entity["embedding"].clone()).unwrap(),
                            ),
                        },
                        embedding_dim: 2,
                    })
                    .await
                    .unwrap();

                let mut acl_metadata = Map::new();
                for (key, value) in [
                    ("entity_type", json!("acl_record")),
                    ("acl_truth_graph", fixture["truth_graph"].clone()),
                    ("acl_target_entity_id", json!(entity_id)),
                    ("acl_target_grain", json!(entity_kind)),
                    ("acl_version", json!(1)),
                    ("acl_mode", entity["mode"].clone()),
                ] {
                    acl_metadata.insert(key.to_owned(), value);
                }
                if let Some(owner_id) = entity["owner_id"].as_str() {
                    acl_metadata.insert("owner_id".to_owned(), json!(owner_id));
                }
                if let Some(security_scope) = entity["security_scope"].as_str() {
                    acl_metadata.insert("security_scope".to_owned(), json!(security_scope));
                }
                service
                    .store
                    .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                        scope: scope.clone(),
                        table: "gke_nodes".to_owned(),
                        record: GraphRecord {
                            id: format!("acl-{entity_id}"),
                            document: None,
                            metadata: acl_metadata,
                            embedding: None,
                        },
                        embedding_dim: 2,
                    })
                    .await
                    .unwrap();
            }
        }

        let foreign_scope = GraphScope {
            workspace_id: Some("workspace-b".to_owned()),
            ..scope.clone()
        };
        service
            .store
            .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                scope: foreign_scope,
                table: "gke_nodes".to_owned(),
                record: GraphRecord {
                    id: "foreign-workspace-nearest".to_owned(),
                    document: Some(
                        json!({"id":"foreign-workspace-nearest"}).to_string(),
                    ),
                    metadata: Map::from_iter([
                        ("entity_type".to_owned(), json!("node")),
                        ("workspace_id".to_owned(), json!("workspace-b")),
                        ("label".to_owned(), json!("foreign-workspace-nearest")),
                    ]),
                    embedding: Some(vec![1.0, 0.0]),
                },
                embedding_dim: 2,
            })
            .await
            .unwrap();
        service
            .store
            .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                scope: scope.clone(),
                table: "gke_nodes".to_owned(),
                record: GraphRecord {
                    id: "acl-foreign-workspace-nearest".to_owned(),
                    document: None,
                    metadata: Map::from_iter([
                        ("entity_type".to_owned(), json!("acl_record")),
                        ("acl_truth_graph".to_owned(), fixture["truth_graph"].clone()),
                        (
                            "acl_target_entity_id".to_owned(),
                            json!("foreign-workspace-nearest"),
                        ),
                        ("acl_target_grain".to_owned(), json!("node")),
                        ("acl_version".to_owned(), json!(1)),
                        ("acl_mode".to_owned(), json!("public")),
                    ]),
                    embedding: None,
                },
                embedding_dim: 2,
            })
            .await
            .unwrap();

        for principal in fixture["principals"].as_array().unwrap() {
            let subject = principal["id"].as_str().unwrap();
            let mut principal_claims = json!({
                "sub": subject,
                "ns": scope.namespace,
                "storage_ns": scope.namespace,
                "workspace_id": scope.workspace_id,
                "capabilities": ["read_graph"]
            });
            if let Some(scope_parts) = principal.get("scope_parts") {
                for key in ["tenant", "workspace", "project"] {
                    if let Some(value) = scope_parts.get(key) {
                        principal_claims[key] = value.clone();
                    }
                }
            }
            for (entity_kind, expected_key) in [("node", "nodes"), ("edge", "edges")] {
                let request_body = json!({
                    "entity_kind": entity_kind,
                    "query_embeddings": fixture["query_embeddings"],
                    "metric": fixture["metric"],
                    "truth_graph": fixture["truth_graph"],
                    "n_results": fixture["n_results"],
                    "include": ["distances"]
                });
                let response = service
                    .execute_async(ApiEffectRequest {
                        contract_version: 1,
                        method: "POST".to_owned(),
                        path_and_query: "/internal/graph/vector-query".to_owned(),
                        body: serde_json::to_vec(&request_body).unwrap(),
                        principal: principal_claims.clone(),
                    })
                    .await;
                assert_eq!(response.status, StatusCode::OK.as_u16());
                let result: Value = serde_json::from_slice(&response.body).unwrap();
                assert_eq!(
                    result["ids"],
                    principal["expected"][expected_key],
                    "unexpected {entity_kind} vector results for {subject}"
                );
                let redirect = &fixture["current_vector_redirect_contract"][entity_kind];
                for row in result["ids"].as_array().unwrap() {
                    let ids = row.as_array().unwrap();
                    assert!(!ids.contains(&redirect["source_id"]));
                    assert!(ids.contains(&redirect["target_id"]));
                }
                let actual_distances = result["distances"].as_array().unwrap();
                let expected_distances = principal["expected"]["distances"][expected_key]
                    .as_array()
                    .unwrap();
                assert_eq!(actual_distances.len(), expected_distances.len());
                for (actual, expected) in actual_distances.iter().zip(expected_distances) {
                    let actual = actual.as_array().unwrap();
                    let expected = expected.as_array().unwrap();
                    assert_eq!(actual.len(), expected.len());
                    for (actual, expected) in actual.iter().zip(expected) {
                        let actual = actual.as_f64().unwrap();
                        let expected = expected.as_f64().unwrap();
                        assert!(
                            (actual - expected).abs() < 1e-5,
                            "unexpected {entity_kind} vector distance for {subject}: {actual} != {expected}"
                        );
                    }
                }
            }
        }

        let proposal_principal = json!({
            "sub": "owner-b",
            "role": "ro",
            "ns": ["docs"],
            "storage_ns": scope.namespace,
            "workspace_id": scope.workspace_id,
            "capabilities": ["read_graph"]
        });
        assert!(service.supports_mcp_tool_for_principal(
            "root",
            "propose_vector",
            &proposal_principal
        ));
        let proposal_response = service
            .execute_async(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: INTERNAL_GRAPH_PROPOSE_VECTOR_ROUTE.to_owned(),
                body: serde_json::to_vec(&json!({
                    "new_node_ids": ["near-private"],
                    "new_edge_ids": [],
                    "top_k": 10,
                    "include_edges": false,
                    "max_distance": 0.35
                }))
                .unwrap(),
                principal: proposal_principal.clone(),
            })
            .await;
        assert_eq!(proposal_response.status, StatusCode::OK.as_u16());
        let proposal: Value = serde_json::from_slice(&proposal_response.body).unwrap();
        assert_eq!(
            proposal["pairs"],
            json!([
                {
                    "left_id": "near-private",
                    "left_kind": "node",
                    "right_id": "visible-scope",
                    "right_kind": "node"
                },
                {
                    "left_id": "near-private",
                    "left_kind": "node",
                    "right_id": "visible-public",
                    "right_kind": "node"
                }
            ])
        );
        let denied_proposal_principal = json!({
            "sub": "owner-b",
            "role": "ro",
            "ns": ["docs"],
            "storage_ns": scope.namespace,
            "workspace_id": scope.workspace_id,
            "capabilities": []
        });
        assert!(!service.supports_mcp_tool_for_principal(
            "root",
            "propose_vector",
            &denied_proposal_principal
        ));
        let denied_proposal_response = service
            .execute_async(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: INTERNAL_GRAPH_PROPOSE_VECTOR_ROUTE.to_owned(),
                body: serde_json::to_vec(&json!({"new_node_ids": ["near-private"]})).unwrap(),
                principal: denied_proposal_principal,
            })
            .await;
        assert_eq!(
            denied_proposal_response.status,
            StatusCode::FORBIDDEN.as_u16()
        );

        let proposal_fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/graph-vector-proposal-acl-v1.json"
        )))
        .unwrap();
        let mcp_scope = GraphScope {
            namespace: "vector-proposal-mcp".to_owned(),
            workspace_id: None,
            graph_space: None,
        };
        for (id, embedding) in [("n1", vec![1.0, 0.0]), ("n2", vec![0.8, 0.6])] {
            service
                .store
                .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                    scope: mcp_scope.clone(),
                    table: "gke_nodes".to_owned(),
                    record: GraphRecord {
                        id: id.to_owned(),
                        document: Some(id.to_owned()),
                        metadata: Map::from_iter([
                            ("entity_type".to_owned(), json!("node")),
                            ("label".to_owned(), json!(id)),
                        ]),
                        embedding: Some(embedding),
                    },
                    embedding_dim: 2,
                })
                .await
                .unwrap();
            service
                .store
                .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                    scope: mcp_scope.clone(),
                    table: "gke_nodes".to_owned(),
                    record: GraphRecord {
                        id: format!("acl-proposal-{id}"),
                        document: None,
                        metadata: Map::from_iter([
                            ("entity_type".to_owned(), json!("acl_record")),
                            ("acl_truth_graph".to_owned(), json!("knowledge")),
                            ("acl_target_entity_id".to_owned(), json!(id)),
                            ("acl_target_grain".to_owned(), json!("node")),
                            ("acl_version".to_owned(), json!(1)),
                            ("acl_mode".to_owned(), json!("public")),
                        ]),
                        embedding: None,
                    },
                    embedding_dim: 2,
                })
                .await
                .unwrap();
        }
        let auth_secret = "postgres-propose-vector-mcp-test-secret";
        let auth = AuthConfig {
            mode: "dev".to_owned(),
            algorithm: Some("HS256".to_owned()),
            key: Some(auth_secret.to_owned()),
            issuer: Some("local".to_owned()),
            ..AuthConfig::default()
        };
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let issue_token = |capabilities: Vec<&str>| {
            encode(
                &Header::new(Algorithm::HS256),
                &json!({
                    "sub": "proposal-reader",
                    "role": "ro",
                    "ns": ["docs"],
                    "storage_ns": mcp_scope.namespace,
                    "capabilities": capabilities,
                    "iss": "local",
                    "exp": now + 300
                }),
                &EncodingKey::from_secret(auth_secret.as_bytes()),
            )
            .unwrap()
        };
        let reader_token = issue_token(vec!["read_graph"]);
        let denied_token = issue_token(Vec::new());
        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: Vec::new(),
                implementation: ImplementationSnapshot::default(),
                auth,
            },
            Arc::new(service.clone()),
        );
        let listed = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .header("authorization", format!("Bearer {reader_token}"))
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":"list","method":"tools/list"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(listed.status(), StatusCode::OK);
        let listed: Value =
            serde_json::from_slice(&to_bytes(listed.into_body(), 65_536).await.unwrap()).unwrap();
        assert!(listed["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .any(|tool| tool["name"] == proposal_fixture["tool"]));

        let call = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .header("authorization", format!("Bearer {reader_token}"))
                    .body(Body::from(
                        json!({
                            "jsonrpc":"2.0",
                            "id":"proposal",
                            "method":"tools/call",
                            "params":{
                                "name":proposal_fixture["tool"],
                                "arguments":proposal_fixture["mcp_arguments"]
                            }
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(call.status(), StatusCode::OK);
        let call: Value =
            serde_json::from_slice(&to_bytes(call.into_body(), 65_536).await.unwrap()).unwrap();
        assert_eq!(
            call["result"]["pairs"],
            proposal_fixture["expected_pairs"]
        );
        assert_eq!(call["result"]["implementation_authority"], "rust");
        assert_eq!(call["result"]["seam_attestation_version"], 1);
        assert_eq!(call["result"]["seam_crossings"], json!([]));

        let denied_call = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .header("authorization", format!("Bearer {denied_token}"))
                    .body(Body::from(
                        json!({
                            "jsonrpc":"2.0",
                            "id":"denied-proposal",
                            "method":"tools/call",
                            "params":{
                                "name":proposal_fixture["tool"],
                                "arguments":proposal_fixture["mcp_arguments"]
                            }
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(denied_call.status(), StatusCode::OK);
        let denied_call: Value = serde_json::from_slice(
            &to_bytes(denied_call.into_body(), 65_536).await.unwrap(),
        )
        .unwrap();
        assert_eq!(denied_call["error"]["code"], -32602);
        assert_eq!(
            denied_call["error"]["message"],
            "Tool not advertised by this application service"
        );

        let revoked_node_id = "visible-public";
        let revoked_acl_metadata = Map::from_iter([
            ("entity_type".to_owned(), json!("acl_record")),
            (
                "acl_truth_graph".to_owned(),
                fixture["truth_graph"].clone(),
            ),
            (
                "acl_target_entity_id".to_owned(),
                json!(revoked_node_id),
            ),
            ("acl_target_grain".to_owned(), json!("node")),
            ("acl_version".to_owned(), json!(2)),
            ("acl_mode".to_owned(), json!("private")),
            ("owner_id".to_owned(), json!("owner-b")),
        ]);
        service
            .store
            .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                scope: scope.clone(),
                table: "gke_nodes".to_owned(),
                record: GraphRecord {
                    id: format!("acl-{revoked_node_id}"),
                    document: None,
                    metadata: revoked_acl_metadata,
                    embedding: None,
                },
                embedding_dim: 0,
            })
            .await
            .unwrap();
        let revoked_response = service
            .execute_async(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/internal/graph/vector-query".to_owned(),
                body: serde_json::to_vec(&json!({
                    "entity_kind": "node",
                    "query_embeddings": [fixture["query_embeddings"][0]],
                    "metric": fixture["metric"],
                    "truth_graph": fixture["truth_graph"],
                    "where": {"label": revoked_node_id},
                    "n_results": 1,
                    "include": ["distances"]
                }))
                .unwrap(),
                principal: json!({
                    "sub": "reader-a",
                    "ns": scope.namespace,
                    "storage_ns": scope.namespace,
                    "workspace_id": scope.workspace_id,
                    "capabilities": ["read_graph"]
                }),
            })
            .await;
        assert_eq!(revoked_response.status, StatusCode::OK.as_u16());
        let revoked_result: Value = serde_json::from_slice(&revoked_response.body).unwrap();
        assert_eq!(revoked_result["ids"], json!([[]]));

        let item_node_id = "node-source-span-private";
        let item_grounding_id = format!("{item_node_id}::mention::0");
        let item_span_id = format!("{item_grounding_id}::span::0");
        service
            .store
            .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                scope: scope.clone(),
                table: "gke_nodes".to_owned(),
                record: GraphRecord {
                    id: item_node_id.to_owned(),
                    document: Some(item_node_id.to_owned()),
                    metadata: Map::from_iter([
                        ("entity_type".to_owned(), json!("node")),
                        ("label".to_owned(), json!(item_node_id)),
                        ("mentions".to_owned(), json!([{"spans": [{}]}])),
                    ]),
                    embedding: Some(vec![1.0, 0.0]),
                },
                embedding_dim: 2,
            })
            .await
            .unwrap();
        for (record_id, grain, target_item_id, mode) in [
            ("acl-source-node", "node", None, "public"),
            (
                "acl-source-grounding",
                "grounding",
                Some(item_grounding_id.as_str()),
                "public",
            ),
            (
                "acl-source-span",
                "span",
                Some(item_span_id.as_str()),
                "private",
            ),
        ] {
            let mut metadata = Map::from_iter([
                ("entity_type".to_owned(), json!("acl_record")),
                ("acl_truth_graph".to_owned(), fixture["truth_graph"].clone()),
                ("acl_target_entity_id".to_owned(), json!(item_node_id)),
                ("acl_target_grain".to_owned(), json!(grain)),
                ("acl_version".to_owned(), json!(1)),
                ("acl_mode".to_owned(), json!(mode)),
            ]);
            if let Some(target_item_id) = target_item_id {
                metadata.insert("acl_target_item_id".to_owned(), json!(target_item_id));
            }
            if mode == "private" {
                metadata.insert("owner_id".to_owned(), json!("owner-b"));
            }
            service
                .store
                .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                    scope: scope.clone(),
                    table: "gke_nodes".to_owned(),
                    record: GraphRecord {
                        id: record_id.to_owned(),
                        document: None,
                        metadata,
                        embedding: None,
                    },
                    embedding_dim: 0,
                })
                .await
                .unwrap();
        }
        let response = service
            .execute_async(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/internal/graph/vector-query".to_owned(),
                body: serde_json::to_vec(&json!({
                    "entity_kind": "node",
                    "query_embeddings": [[1.0, 0.0]],
                    "metric": "cosine",
                    "truth_graph": fixture["truth_graph"],
                    "where": {"label": item_node_id},
                    "n_results": 1,
                    "include": ["distances"]
                }))
                .unwrap(),
                principal: json!({
                    "sub": "reader-a",
                    "ns": scope.namespace,
                    "storage_ns": scope.namespace,
                    "workspace": scope.workspace_id,
                    "capabilities": ["read_graph"]
                }),
            })
            .await;
        assert_eq!(response.status, StatusCode::OK.as_u16());
        let result: Value = serde_json::from_slice(&response.body).unwrap();
        assert_eq!(result["ids"], json!([[]]));

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
            .batch_execute(&format!("DROP SCHEMA \"{schema}\" CASCADE"))
            .await
            .unwrap();
        drop(client);
        connection_task.await.unwrap().unwrap();
    }

    #[tokio::test]
    #[ignore = "requires KOGWISTAR_TEST_PG_DSN; uses a disposable PostgreSQL schema"]
    async fn postgres_graph_neighbors_omit_cross_scope_endpoint() {
        let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
            .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
        let schema = format!("rust_neighbor_scope_{}", uuid::Uuid::new_v4().simple());
        let service = PostgresRunApplicationService::from_dsn(&dsn, &schema).unwrap();
        service.ensure_schema().await.unwrap();
        service
            .store
            .create_graph_schema(2, kogwistar_store_postgres::GraphTableNames::default())
            .await
            .unwrap();

        let namespace = format!("neighbor-scope-{}", uuid::Uuid::new_v4().simple());
        let source_scope = GraphScope {
            namespace: namespace.clone(),
            workspace_id: Some("workspace-a".to_owned()),
            graph_space: Some("facts".to_owned()),
        };
        let foreign_scope = GraphScope {
            workspace_id: Some("workspace-b".to_owned()),
            ..source_scope.clone()
        };
        for (scope, table, id, metadata, document) in [
            (
                source_scope.clone(),
                "gke_nodes",
                "source-a",
                json!({"entity_type":"node","workspace_id":"workspace-a","graph_space":"facts"}),
                json!({"id":"source-a","label":"Source"}),
            ),
            (
                foreign_scope,
                "gke_nodes",
                "target-b",
                json!({"entity_type":"node","workspace_id":"workspace-b","graph_space":"facts"}),
                json!({"id":"target-b","label":"Foreign target"}),
            ),
            (
                source_scope.clone(),
                "gke_edges",
                "edge-a-b",
                json!({"entity_type":"edge","workspace_id":"workspace-a","graph_space":"facts","source_ids":["source-a"],"target_ids":["target-b"]}),
                json!({"id":"edge-a-b","source_ids":["source-a"],"target_ids":["target-b"]}),
            ),
        ] {
            service
                .store
                .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                    scope,
                    table: table.to_owned(),
                    record: GraphRecord {
                        id: id.to_owned(),
                        document: Some(document.to_string()),
                        metadata: metadata.as_object().unwrap().clone(),
                        embedding: None,
                    },
                    embedding_dim: 0,
                })
                .await
                .unwrap();
        }

        let response = service
            .graph_neighbors_effect(&ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/internal/graph/neighbors".to_owned(),
                body: br#"{"rid":"source-a"}"#.to_vec(),
                principal: json!({
                    "sub":"reader-a",
                    "storage_ns":namespace,
                    "workspace":"workspace-a",
                    "workspace_id":"workspace-a",
                    "graph_space":"facts",
                    "capabilities":["read_graph"]
                }),
            })
            .await;
        assert_eq!(response.status, StatusCode::OK.as_u16());
        let result: Value = serde_json::from_slice(&response.body).unwrap();
        assert_eq!(result["nodes"], json!([]));
        assert_eq!(result["edges"], json!([]));

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
                "DROP SCHEMA IF EXISTS \"{schema}_wisdom\" CASCADE; DROP SCHEMA \"{schema}\" CASCADE"
            ))
            .await
            .unwrap();
        drop(client);
        connection_task.await.unwrap().unwrap();
    }

    #[tokio::test]
    #[ignore = "requires KOGWISTAR_TEST_PG_DSN; uses a disposable PostgreSQL schema"]
    async fn postgres_graph_neighbors_enforce_project_and_source_evidence_acl() {
        let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
            .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
        let schema = format!("rust_neighbor_acl_{}", uuid::Uuid::new_v4().simple());
        let service = PostgresRunApplicationService::from_dsn(&dsn, &schema).unwrap();
        service.ensure_schema().await.unwrap();
        service
            .store
            .create_graph_schema(2, kogwistar_store_postgres::GraphTableNames::default())
            .await
            .unwrap();

        let namespace = format!("neighbor-acl-{}", uuid::Uuid::new_v4().simple());
        let scope = GraphScope {
            namespace: namespace.clone(),
            workspace_id: Some("workspace-a".to_owned()),
            graph_space: Some("facts".to_owned()),
        };
        let mut records = Vec::new();
        for (id, mentions) in [
            ("project-root", json!([])),
            ("project-target", json!([])),
            ("evidence-root", json!([])),
            (
                "evidence-target",
                json!([{"spans":[{"doc_id":"restricted-source"}]}]),
            ),
            ("control-root", json!([])),
            ("control-target", json!([])),
            (
                "same-id",
                json!([{"spans":[{"doc_id":"shared-source"}]}]),
            ),
        ] {
            records.push((
                "gke_nodes",
                id,
                json!({
                    "entity_type":"node",
                    "workspace_id":"workspace-a",
                    "graph_space":"facts",
                    "mentions":mentions
                }),
                json!({"id":id,"label":id,"mentions":mentions}),
            ));
        }
        for (id, source_id, target_id, mentions) in [
            ("project-edge", "project-root", "project-target", json!([])),
            ("evidence-edge", "evidence-root", "evidence-target", json!([])),
            ("control-edge", "control-root", "control-target", json!([])),
            ("same-node-link", "control-root", "same-id", json!([])),
            (
                "same-id",
                "control-root",
                "same-id",
                json!([{"spans":[{"doc_id":"shared-source"}]}]),
            ),
        ] {
            records.push((
                "gke_edges",
                id,
                json!({
                    "entity_type":"edge",
                    "workspace_id":"workspace-a",
                    "graph_space":"facts",
                    "mentions":mentions,
                    "source_ids":[source_id],
                    "target_ids":[target_id]
                }),
                json!({"id":id,"source_ids":[source_id],"target_ids":[target_id]}),
            ));
        }
        for (table, id, metadata, document) in records {
            service
                .store
                .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                    scope: scope.clone(),
                    table: table.to_owned(),
                    record: GraphRecord {
                        id: id.to_owned(),
                        document: Some(document.to_string()),
                        metadata: metadata.as_object().unwrap().clone(),
                        embedding: None,
                    },
                    embedding_dim: 0,
                })
                .await
                .unwrap();
        }

        let mut acl_cases = Vec::new();
        for id in [
            "project-root",
            "evidence-root",
            "evidence-target",
            "control-root",
            "control-target",
        ] {
            acl_cases.push((format!("acl-{id}"), "node", id.to_owned(), None, "public", None));
        }
        for id in ["project-edge", "evidence-edge", "control-edge"] {
            acl_cases.push((format!("acl-{id}"), "edge", id.to_owned(), None, "public", None));
        }
        acl_cases.push((
            "acl-same-node-link".to_owned(),
            "edge",
            "same-node-link".to_owned(),
            None,
            "public",
            None,
        ));
        acl_cases.push((
            "acl-project-target".to_owned(),
            "node",
            "project-target".to_owned(),
            None,
            "scope",
            Some("project-b"),
        ));
        acl_cases.push((
            "acl-evidence-grounding".to_owned(),
            "grounding",
            "evidence-target".to_owned(),
            Some("evidence-target::mention::0"),
            "public",
            None,
        ));
        acl_cases.push((
            "acl-evidence-span".to_owned(),
            "span",
            "evidence-target".to_owned(),
            Some("evidence-target::mention::0::span::0"),
            "private",
            Some("another-principal"),
        ));
        acl_cases.extend([
            (
                "acl-same-node".to_owned(),
                "node",
                "same-id".to_owned(),
                None,
                "public",
                None,
            ),
            (
                "acl-same-edge".to_owned(),
                "edge",
                "same-id".to_owned(),
                None,
                "public",
                None,
            ),
            (
                "acl-same-node-grounding".to_owned(),
                "grounding",
                "same-id".to_owned(),
                Some("__kogwistar_acl_usage_v2__:node:same-id::mention::0"),
                "public",
                None,
            ),
            (
                "acl-same-node-span".to_owned(),
                "span",
                "same-id".to_owned(),
                Some("__kogwistar_acl_usage_v2__:node:same-id::mention::0::span::0"),
                "public",
                None,
            ),
            (
                "acl-same-edge-grounding".to_owned(),
                "grounding",
                "same-id".to_owned(),
                Some("__kogwistar_acl_usage_v2__:edge:same-id::mention::0"),
                "public",
                None,
            ),
            (
                "acl-same-edge-span".to_owned(),
                "span",
                "same-id".to_owned(),
                Some("__kogwistar_acl_usage_v2__:edge:same-id::mention::0::span::0"),
                "private",
                Some("another-principal"),
            ),
        ]);
        for (acl_id, grain, entity_id, item_id, mode, security_scope) in acl_cases {
            let mut metadata = json!({
                "entity_type":"acl_record",
                "acl_truth_graph":"knowledge",
                "acl_target_grain":grain,
                "acl_target_entity_id":entity_id,
                "acl_version":1,
                "acl_mode":mode
            });
            if let Some(item_id) = item_id {
                metadata["acl_target_item_id"] = json!(item_id);
            }
            if let Some(security_scope) = security_scope {
                metadata["security_scope"] = json!(security_scope);
            }
            if mode == "private" {
                metadata["owner_id"] = json!("another-principal");
            }
            service
                .store
                .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                    scope: scope.clone(),
                    table: "gke_nodes".to_owned(),
                    record: GraphRecord {
                        id: acl_id,
                        document: None,
                        metadata: metadata.as_object().unwrap().clone(),
                        embedding: None,
                    },
                    embedding_dim: 0,
                })
                .await
                .unwrap();
        }

        let read_neighbors = |root: &str| ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/internal/graph/neighbors".to_owned(),
            body: serde_json::to_vec(&json!({"rid":root,"truth_graph":"knowledge"})).unwrap(),
            principal: json!({
                "sub":"reader-a",
                "storage_ns":namespace,
                "workspace":"workspace-a",
                "workspace_id":"workspace-a",
                "graph_space":"facts",
                "security_scope":"project-a",
                "capabilities":["read_graph"]
            }),
        };
        for root in ["project-root", "evidence-root"] {
            let response = service.graph_neighbors_effect(&read_neighbors(root)).await;
            assert_eq!(response.status, StatusCode::OK.as_u16());
            let result: Value = serde_json::from_slice(&response.body).unwrap();
            assert_eq!(result["nodes"], json!([]), "root={root}: {result}");
            assert_eq!(result["edges"], json!([]), "root={root}: {result}");
        }
        let control = service
            .graph_neighbors_effect(&read_neighbors("control-root"))
            .await;
        assert_eq!(control.status, StatusCode::OK.as_u16());
        let control: Value = serde_json::from_slice(&control.body).unwrap();
        assert_eq!(control["nodes"], json!(["control-target", "same-id"]));
        assert_eq!(control["edges"], json!(["control-edge", "same-node-link"]));

        let mut owner_request = read_neighbors("control-root");
        let owner_principal = json!({
            "sub":"another-principal",
            "storage_ns":namespace,
            "workspace":"workspace-a",
            "workspace_id":"workspace-a",
            "graph_space":"facts",
            "security_scope":"project-a",
            "capabilities":["read_graph"]
        });
        owner_request.principal = owner_principal;
        let same_id_owner = service.graph_neighbors_effect(&owner_request).await;
        assert_eq!(same_id_owner.status, StatusCode::OK.as_u16());
        let same_id_owner: Value = serde_json::from_slice(&same_id_owner.body).unwrap();
        assert_eq!(same_id_owner["nodes"], json!(["control-target", "same-id"]));
        assert_eq!(
            same_id_owner["edges"],
            json!(["control-edge", "same-id", "same-node-link"])
        );

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
                "DROP SCHEMA IF EXISTS \"{schema}_wisdom\" CASCADE; DROP SCHEMA \"{schema}\" CASCADE"
            ))
            .await
            .unwrap();
        drop(client);
        connection_task.await.unwrap().unwrap();
    }

    #[tokio::test]
    #[ignore = "requires KOGWISTAR_TEST_PG_DSN; uses a disposable PostgreSQL schema"]
    async fn postgres_graph_snapshot_omits_conflicting_workspace_node() {
        let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
            .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
        let schema = format!("rust_snapshot_scope_{}", uuid::Uuid::new_v4().simple());
        let service = PostgresRunApplicationService::from_dsn(&dsn, &schema).unwrap();
        service.ensure_schema().await.unwrap();
        service
            .store
            .create_graph_schema(2, kogwistar_store_postgres::GraphTableNames::default())
            .await
            .unwrap();

        let namespace = format!("snapshot-scope-{}", uuid::Uuid::new_v4().simple());
        for (workspace, id) in [
            ("workspace-a", "source-a"),
            ("workspace-b", "foreign-node-b"),
        ] {
            service
                .store
                .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                    scope: GraphScope {
                        namespace: namespace.clone(),
                        workspace_id: Some(workspace.to_owned()),
                        graph_space: Some("facts".to_owned()),
                    },
                    table: "gke_nodes".to_owned(),
                    record: GraphRecord {
                        id: id.to_owned(),
                        document: Some(json!({"id":id,"label":id}).to_string()),
                        metadata: Map::from_iter([
                            ("entity_type".to_owned(), json!("node")),
                            ("workspace_id".to_owned(), json!(workspace)),
                            ("graph_space".to_owned(), json!("facts")),
                        ]),
                        embedding: None,
                    },
                    embedding_dim: 0,
                })
                .await
                .unwrap();
            service
                .store
                .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                    scope: GraphScope {
                        namespace: namespace.clone(),
                        workspace_id: Some(workspace.to_owned()),
                        graph_space: Some("facts".to_owned()),
                    },
                    table: "gke_nodes".to_owned(),
                    record: GraphRecord {
                        id: format!("acl-{id}"),
                        document: None,
                        metadata: Map::from_iter([
                            ("entity_type".to_owned(), json!("acl_record")),
                            ("acl_truth_graph".to_owned(), json!("knowledge")),
                            ("acl_target_grain".to_owned(), json!("node")),
                            ("acl_target_entity_id".to_owned(), json!(id)),
                            ("acl_mode".to_owned(), json!("public")),
                            ("acl_version".to_owned(), json!(1)),
                        ]),
                        embedding: None,
                    },
                    embedding_dim: 0,
                })
                .await
                .unwrap();
        }

        for (table, id, metadata, document) in [
            (
                "gke_edges",
                "edge-to-foreign-node",
                json!({
                    "entity_type":"edge",
                    "workspace_id":"workspace-a",
                    "graph_space":"facts",
                    "source_ids":["source-a"],
                    "target_ids":["foreign-node-b"]
                }),
                json!({
                    "id":"edge-to-foreign-node",
                    "source_ids":["source-a"],
                    "target_ids":["foreign-node-b"]
                }),
            ),
            (
                "gke_nodes",
                "acl-edge-to-foreign-node",
                json!({
                    "entity_type":"acl_record",
                    "acl_truth_graph":"knowledge",
                    "acl_target_grain":"edge",
                    "acl_target_entity_id":"edge-to-foreign-node",
                    "acl_mode":"public",
                    "acl_version":1
                }),
                Value::Null,
            ),
            (
                "gke_edges",
                "edge-malformed-endpoint",
                json!({
                    "entity_type":"edge",
                    "workspace_id":"workspace-a",
                    "graph_space":"facts",
                    "source_ids":"source-a",
                    "target_ids":["source-a"]
                }),
                json!({
                    "id":"edge-malformed-endpoint",
                    "source_ids":"source-a",
                    "target_ids":["source-a"]
                }),
            ),
            (
                "gke_nodes",
                "acl-edge-malformed-endpoint",
                json!({
                    "entity_type":"acl_record",
                    "acl_truth_graph":"knowledge",
                    "acl_target_grain":"edge",
                    "acl_target_entity_id":"edge-malformed-endpoint",
                    "acl_mode":"public",
                    "acl_version":1
                }),
                Value::Null,
            ),
        ] {
            service
                .store
                .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                    scope: GraphScope {
                        namespace: namespace.clone(),
                        workspace_id: Some("workspace-a".to_owned()),
                        graph_space: Some("facts".to_owned()),
                    },
                    table: table.to_owned(),
                    record: GraphRecord {
                        id: id.to_owned(),
                        document: (!document.is_null()).then(|| document.to_string()),
                        metadata: metadata.as_object().unwrap().clone(),
                        embedding: None,
                    },
                    embedding_dim: 0,
                })
                .await
                .unwrap();
        }

        let response = service
            .execute_async(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: INTERNAL_GRAPH_VIZ_D3_ROUTE.to_owned(),
                body: br#"{"graph_type":"knowledge"}"#.to_vec(),
                principal: json!({
                    "sub":"reader-a",
                    "storage_ns":namespace,
                    "workspace":"workspace-a",
                    "workspace_id":"workspace-a",
                    "graph_space":"facts",
                    "capabilities":["read_graph"]
                }),
            })
            .await;
        let response_status = response.status;
        let response_body = response.body;

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
                "DROP SCHEMA IF EXISTS \"{schema}_wisdom\" CASCADE; DROP SCHEMA \"{schema}\" CASCADE"
            ))
            .await
            .unwrap();
        drop(client);
        connection_task.await.unwrap().unwrap();

        assert_eq!(response_status, StatusCode::OK.as_u16());
        let result: Value = serde_json::from_slice(&response_body).unwrap();
        let node_ids = result["nodes"]
            .as_array()
            .expect("D3 snapshot has node array")
            .iter()
            .filter_map(|node| node["id"].as_str())
            .collect::<Vec<_>>();
        assert!(node_ids.contains(&"source-a"), "snapshot: {result}");
        assert!(!node_ids.contains(&"foreign-node-b"), "snapshot: {result}");
        assert!(
            !node_ids.contains(&"edge-to-foreign-node"),
            "snapshot reified an edge with an out-of-scope endpoint: {result}"
        );
        assert!(
            !node_ids.contains(&"edge-malformed-endpoint"),
            "snapshot reified an edge with malformed endpoint data: {result}"
        );
        assert!(
            result["links"].as_array().unwrap().iter().all(|link| {
                link["source"] != "edge-to-foreign-node"
                    && link["target"] != "edge-to-foreign-node"
                    && link["source"] != "edge-malformed-endpoint"
                    && link["target"] != "edge-malformed-endpoint"
            }),
            "snapshot linked an edge with an invalid endpoint: {result}"
        );
    }
