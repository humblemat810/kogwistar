    #[tokio::test]
    #[ignore = "requires KOGWISTAR_TEST_PG_DSN; run explicitly with --include-ignored"]
    async fn postgres_public_mcp_graph_reads_cross_projection_page_boundary() {
        use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};

        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/graph-traversal-v1.json"
        )))
        .unwrap();
        let cases = fixture["cases"]
            .as_array()
            .unwrap()
            .iter()
            .map(|case| (case["name"].as_str().unwrap(), case))
            .collect::<std::collections::BTreeMap<_, _>>();
        let source_id = "z-source";
        let target_id = "z-target";
        let edge_id = "z-edge";
        let map_fixture_id = |id: &str| match id {
            "A" => source_id.to_owned(),
            "B" => target_id.to_owned(),
            "E1" => edge_id.to_owned(),
            other => other.to_owned(),
        };
        let map_fixture_ids = |ids: &Value| {
            ids.as_array()
                .unwrap()
                .iter()
                .map(|id| map_fixture_id(id.as_str().unwrap()))
                .collect::<Vec<_>>()
        };

        let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN").unwrap();
        let schema = format!("rust_mcp_graph_pages_{}", uuid::Uuid::new_v4().simple());
        let service = PostgresRunApplicationService::from_dsn(&dsn, &schema).unwrap();
        service.ensure_schema().await.unwrap();
        service
            .store
            .create_graph_schema(3, kogwistar_store_postgres::GraphTableNames::default())
            .await
            .unwrap();
        let scope = GraphScope {
            namespace: "alpha".to_owned(),
            workspace_id: None,
            graph_space: None,
        };
        for (table, id, document, metadata) in [
            ("gke_nodes", source_id, json!({"label":"Smoking","doc_id":"D1"}), json!({})),
            ("gke_nodes", target_id, json!({"label":"Lung Cancer","doc_id":"D1"}), json!({})),
            ("gke_nodes", "z-third", json!({"label":"Third node","doc_id":"D1"}), json!({})),
            ("gke_nodes", "z-d2-target", json!({"label":"Lung Cancer","doc_id":"D2"}), json!({})),
            ("gke_edges", edge_id, json!({"relation":"causes","source_ids":[source_id],"target_ids":[target_id],"doc_id":"D1"}), json!({})),
            ("gke_edges", "z-d2-edge", json!({"relation":"causes","source_ids":[source_id],"target_ids":["z-d2-target"],"doc_id":"D2"}), json!({})),
            ("gke_nodes", "acl-source", Value::Null, json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"node","acl_target_entity_id":source_id,"acl_version":1,"acl_mode":"public"})),
            ("gke_nodes", "acl-target", Value::Null, json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"node","acl_target_entity_id":target_id,"acl_version":1,"acl_mode":"public"})),
            ("gke_nodes", "acl-third", Value::Null, json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"node","acl_target_entity_id":"z-third","acl_version":1,"acl_mode":"public"})),
            ("gke_nodes", "acl-d2-target", Value::Null, json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"node","acl_target_entity_id":"z-d2-target","acl_version":1,"acl_mode":"public"})),
            ("gke_nodes", "acl-edge", Value::Null, json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"edge","acl_target_entity_id":edge_id,"acl_version":1,"acl_mode":"public"})),
            ("gke_nodes", "acl-d2-edge", Value::Null, json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"edge","acl_target_entity_id":"z-d2-edge","acl_version":1,"acl_mode":"public"})),
            ("gke_nodes", "acl-new-edge", Value::Null, json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"edge","acl_target_entity_id":"z-new-edge","acl_version":1,"acl_mode":"public"})),
        ] {
            service.store.upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                scope: scope.clone(),
                table: table.to_owned(),
                record: GraphRecord {
                    id: id.to_owned(),
                    document: (!document.is_null()).then(|| document.to_string()),
                    metadata: metadata.as_object().unwrap().clone(),
                    embedding: None,
                },
                embedding_dim: 3,
            }).await.unwrap();
        }
        let (mut client, connection) = tokio_postgres::connect(&dsn, tokio_postgres::NoTls).await.unwrap();
        let connection_task = tokio::spawn(connection);
        let (observer, observer_connection) =
            tokio_postgres::connect(&dsn, tokio_postgres::NoTls).await.unwrap();
        let observer_task = tokio::spawn(observer_connection);
        client.execute(
            &format!("INSERT INTO \"{schema}\".gke_nodes(id, metadata) SELECT 'm' || lpad(n::text, 5, '0'), '{{\"namespace\":\"alpha\"}}'::jsonb FROM generate_series(0, 10000) AS n"),
            &[],
        ).await.unwrap();
        client.execute(
            &format!("INSERT INTO \"{schema}\".gke_edges(id, metadata) SELECT 'm' || lpad(n::text, 5, '0'), '{{\"namespace\":\"alpha\"}}'::jsonb FROM generate_series(0, 10000) AS n"),
            &[],
        ).await.unwrap();
        let key = "graph-pages-test-secret";
        let token = encode(
            &Header::new(Algorithm::HS256),
            &json!({"sub":"alice","role":"ro","roles":["reader"],"ns":"docs","storage_ns":"alpha","capabilities":["read_graph"],"exp":4102444800_u64}),
            &EncodingKey::from_secret(key.as_bytes()),
        ).unwrap();
        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: vec!["reader".to_owned()],
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig {mode:"jwt".to_owned(), algorithm:Some("HS256".to_owned()), key:Some(key.to_owned()), ..AuthConfig::default()},
            },
            Arc::new(service.clone()),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move { axum::serve(listener, app).await });
        let http = reqwest::Client::new();
        let preflight: Value = http
            .post(format!("http://{address}/mcp"))
            .bearer_auth(&token)
            .json(&json!({
                "jsonrpc": "2.0",
                "id": "snapshot-preflight",
                "method": "tools/call",
                "params": {"name": "kg_neighbors", "arguments": {"rid": source_id, "doc_id": "D1"}}
            }))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert!(preflight.get("error").is_none(), "{preflight}");

        // Hold the edge table while the reader establishes its repeatable-read
        // snapshot on nodes. Commit a new edge only after its edge-page SELECT
        // is waiting on the lock; that same read must not observe the edge.
        let transaction = client.transaction().await.unwrap();
        transaction
            .batch_execute(&format!("LOCK TABLE \"{schema}\".gke_edges IN ACCESS EXCLUSIVE MODE"))
            .await
            .unwrap();
        transaction
            .execute(
                &format!("INSERT INTO \"{schema}\".gke_edges(id, document, metadata) VALUES ('z-new-edge', $1, '{{\"namespace\":\"alpha\"}}'::jsonb)"),
                &[&json!({"relation":"causes","source_ids":[source_id],"target_ids":["z-third"],"doc_id":"D1"}).to_string()],
            )
            .await
            .unwrap();
        let concurrent_read = {
            let http = http.clone();
            let token = token.clone();
            tokio::spawn(async move {
                http.post(format!("http://{address}/mcp"))
                    .bearer_auth(token)
                    .json(&json!({
                        "jsonrpc": "2.0",
                        "id": "snapshot-before-edge-commit",
                        "method": "tools/call",
                        "params": {"name": "kg_neighbors", "arguments": {"rid": source_id, "doc_id": "D1"}}
                    }))
                    .send()
                    .await
                    .unwrap()
                    .json::<Value>()
                    .await
                    .unwrap()
            })
        };
        let waiting_for_edge_page = tokio::time::timeout(
            std::time::Duration::from_secs(10),
            async {
                loop {
                    let waiting: bool = observer
                        .query_one(
                            "SELECT EXISTS (SELECT 1 FROM pg_stat_activity WHERE datname = current_database() AND state = 'active' AND wait_event_type = 'Lock' AND query LIKE '%SELECT id, document, metadata::TEXT, embedding::TEXT FROM%' AND query LIKE '%gke_edges%')",
                            &[],
                        )
                        .await
                        .unwrap()
                        .get(0);
                    if waiting {
                        break;
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(2)).await;
                }
            },
        )
        .await;
        assert!(waiting_for_edge_page.is_ok(), "reader did not reach the locked edge projection query");
        transaction.commit().await.unwrap();
        let first_read = concurrent_read.await.unwrap();
        assert!(first_read.get("error").is_none(), "{first_read}");
        assert_eq!(first_read["result"]["implementation_authority"], "rust");
        assert_eq!(first_read["result"]["seam_attestation_version"], 1);
        assert_eq!(first_read["result"]["seam_crossings"], json!([]));
        assert!(!first_read["result"]["nodes"].as_array().unwrap().contains(&json!("z-third")));

        let second_read: Value = http
            .post(format!("http://{address}/mcp"))
            .bearer_auth(&token)
            .json(&json!({
                "jsonrpc": "2.0",
                "id": "snapshot-after-edge-commit",
                "method": "tools/call",
                "params": {"name": "kg_neighbors", "arguments": {"rid": source_id, "doc_id": "D1"}}
            }))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert!(second_read["result"]["nodes"].as_array().unwrap().contains(&json!("z-third")));
        assert_eq!(second_read["result"]["implementation_authority"], "rust");
        assert_eq!(second_read["result"]["seam_attestation_version"], 1);
        assert_eq!(second_read["result"]["seam_crossings"], json!([]));
        client
            .execute(
                &format!("DELETE FROM \"{schema}\".gke_edges WHERE id = 'z-new-edge'"),
                &[],
            )
            .await
            .unwrap();

        let neighbor_case = cases["edge_to_nodes"];
        let find_case = cases["find_edges"];
        let path_case = cases["shortest_path"];
        let hop_case = cases["k_hop"];
        for (name, arguments, expected_field, expected) in [
            (
                "kg_neighbors",
                json!({"rid":edge_id,"doc_id":"D1"}),
                "nodes",
                json!(map_fixture_ids(&neighbor_case["expected_neighbors"]["nodes"])),
            ),
            (
                "kg_find_edges",
                json!({"relation":find_case["relation"],"src_label_contains":find_case["source_label_contains"],"tgt_label_contains":find_case["target_label_contains"],"doc_id":find_case["doc_id"]}),
                "edges",
                json!(map_fixture_ids(&find_case["expected_edge_ids"])),
            ),
            (
                "kg_shortest_path",
                json!({"src_id":source_id,"dst_id":target_id,"doc_id":"D1"}),
                "path",
                json!(map_fixture_ids(&path_case["expected_path"])),
            ),
            (
                "kg_k_hop",
                json!({"start_ids":[source_id],"k":hop_case["k"],"doc_id":"D1"}),
                "layers",
                Value::Null,
            ),
        ] {
            let response = http
                .post(format!("http://{address}/mcp"))
                .bearer_auth(&token)
                .json(&json!({
                    "jsonrpc": "2.0",
                    "id": name,
                    "method": "tools/call",
                    "params": {"name": name, "arguments": arguments}
                }))
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let payload: Value = response.json().await.unwrap();
            assert!(payload.get("error").is_none(), "{name}: {payload}");
            assert_eq!(payload["result"]["implementation_authority"], "rust");
            assert_eq!(payload["result"]["seam_attestation_version"], 1);
            assert_eq!(payload["result"]["seam_crossings"], json!([]));
            if name == "kg_k_hop" {
                let expected_layers = hop_case["expected_layers"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|layer| {
                        json!({
                            "nodes": map_fixture_ids(&layer["nodes"]),
                            "edges": map_fixture_ids(&layer["edges"])
                        })
                    })
                    .collect::<Vec<_>>();
                assert_eq!(payload["result"]["layers"], json!(expected_layers));
            } else {
                assert_eq!(payload["result"][expected_field], expected, "{name}");
                if name == "kg_neighbors" {
                    assert_eq!(
                        payload["result"]["edges"],
                        json!(map_fixture_ids(&neighbor_case["expected_neighbors"]["edges"]))
                    );
                }
            }
        }
        server.abort();
        let _ = server.await;
        drop(service);
        client
            .batch_execute(&format!("DROP SCHEMA \"{schema}\" CASCADE"))
            .await
            .unwrap();
        drop(client);
        drop(observer);
        observer_task.await.unwrap().unwrap();
        connection_task.await.unwrap().unwrap();
    }
