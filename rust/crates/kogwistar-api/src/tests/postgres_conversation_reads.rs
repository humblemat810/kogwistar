    #[tokio::test]
    #[ignore = "requires KOGWISTAR_TEST_PG_DSN; run explicitly with --include-ignored"]
    async fn postgres_latest_snapshot_route_projects_authorized_snapshot_when_dsn_available() {
        use sha2::Digest;

        let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
            .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
        let schema = format!("rust_snapshot_route_{}", uuid::Uuid::new_v4().simple());
        let service = PostgresRunApplicationService::from_dsn(&dsn, &schema).unwrap();
        service.ensure_graph_type_schemas().await.unwrap();
        service
            .conversation_store
            .create_graph_schema(3, kogwistar_store_postgres::GraphTableNames::default())
            .await
            .unwrap();
        let scope = GraphScope {
            namespace: "conversation".to_owned(),
            workspace_id: None,
            graph_space: None,
        };
        let expected = serde_json::from_str::<Value>(include_str!(
            "../../../../../contracts/golden/mount-memory-result-v1.json"
        ))
        .unwrap();
        let mut snapshot_metadata = Map::new();
        snapshot_metadata.insert("entity_type".to_owned(), json!("context_snapshot"));
        snapshot_metadata.insert(
            "conversation_id".to_owned(),
            expected["conversation_id"].clone(),
        );
        snapshot_metadata.insert("run_id".to_owned(), json!("run-1"));
        snapshot_metadata.insert("stage".to_owned(), json!("final"));
        service
            .conversation_store
            .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                scope: scope.clone(),
                table: "gke_nodes".to_owned(),
                record: GraphRecord {
                    id: expected["snapshot_node_id"].as_str().unwrap().to_owned(),
                    document: Some(expected.to_string()),
                    metadata: snapshot_metadata,
                    embedding: None,
                },
                embedding_dim: 3,
            })
            .await
            .unwrap();

        let mut private_snapshot_metadata = Map::new();
        private_snapshot_metadata.insert("entity_type".to_owned(), json!("context_snapshot"));
        private_snapshot_metadata.insert(
            "conversation_id".to_owned(),
            expected["conversation_id"].clone(),
        );
        private_snapshot_metadata.insert("run_id".to_owned(), json!("run-private"));
        private_snapshot_metadata.insert("stage".to_owned(), json!("final"));
        service
            .conversation_store
            .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                scope: scope.clone(),
                table: "gke_nodes".to_owned(),
                record: GraphRecord {
                    id: "snapshot-private".to_owned(),
                    document: Some(json!({"summary": "private snapshot"}).to_string()),
                    metadata: private_snapshot_metadata,
                    embedding: None,
                },
                embedding_dim: 3,
            })
            .await
            .unwrap();
        let mut private_acl_metadata = Map::new();
        for (key, value) in [
            ("entity_type", json!("acl_record")),
            ("acl_truth_graph", json!("conversation")),
            ("acl_target_entity_id", json!("snapshot-private")),
            ("acl_target_grain", json!("node")),
            ("acl_version", json!(1)),
            ("acl_mode", json!("private")),
            ("owner_id", json!("bob")),
        ] {
            private_acl_metadata.insert(key.to_owned(), value);
        }
        service
            .conversation_store
            .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                scope: scope.clone(),
                table: "gke_nodes".to_owned(),
                record: GraphRecord {
                    id: "acl-snapshot-private".to_owned(),
                    document: None,
                    metadata: private_acl_metadata,
                    embedding: None,
                },
                embedding_dim: 3,
            })
            .await
            .unwrap();
        let mut acl_metadata = Map::new();
        for (key, value) in [
            ("entity_type", json!("acl_record")),
            ("acl_truth_graph", json!("conversation")),
            ("acl_target_entity_id", json!("snapshot-1")),
            ("acl_target_grain", json!("node")),
            ("acl_version", json!(1)),
            ("acl_mode", json!("public")),
        ] {
            acl_metadata.insert(key.to_owned(), value);
        }
        service
            .conversation_store
            .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                scope: scope.clone(),
                table: "gke_nodes".to_owned(),
                record: GraphRecord {
                    id: "acl-snapshot-1".to_owned(),
                    document: None,
                    metadata: acl_metadata,
                    embedding: None,
                },
                embedding_dim: 3,
            })
            .await
            .unwrap();

        let response = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query:
                    "/api/conversations/conversation-1/snapshots/latest?run_id=run-1&stage=final"
                        .to_owned(),
                body: Vec::new(),
                principal: json!({
                    "sub": "alice",
                    "role": "ro",
                    "ns": "workflow",
                    "storage_ns": "conversation",
                    "capabilities": ["project_view"]
                }),
            })
            .await;
        assert_eq!(response.status, StatusCode::OK.as_u16());
        let payload: Value = serde_json::from_slice(&response.body).unwrap();
        assert_eq!(payload["snapshot_node_id"], "snapshot-1");
        assert_eq!(payload["conversation_id"], "conversation-1");
        assert_eq!(payload["metadata"]["source"], "integration");
        assert_eq!(payload["properties"]["summary"], "latest snapshot");
        assert_eq!(
            payload
                .as_object()
                .unwrap()
                .keys()
                .cloned()
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([
                "conversation_id".to_owned(),
                "metadata".to_owned(),
                "properties".to_owned(),
                "snapshot_node_id".to_owned(),
            ])
        );

        let syscall_response = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/syscall/v1/mount_memory".to_owned(),
                body: serde_json::to_vec(&json!({
                    "version": "v1",
                    "op": "mount_memory",
                    "args": {
                        "conversation_id": expected["conversation_id"],
                        "run_id": "run-1",
                        "stage": "final"
                    }
                }))
                .unwrap(),
                principal: json!({
                    "sub": "alice",
                    "role": "ro",
                    "ns": "conversation",
                    "storage_ns": "conversation",
                    "capabilities": ["project_view"]
                }),
            })
            .await;
        assert_eq!(syscall_response.status, StatusCode::OK.as_u16());
        let syscall_payload: Value = serde_json::from_slice(&syscall_response.body).unwrap();
        assert_eq!(syscall_payload["version"], "v1");
        assert_eq!(syscall_payload["op"], "mount_memory");
        assert_eq!(syscall_payload["status"], "ok");
        assert_eq!(syscall_payload["result"], expected);

        let private_syscall_response = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/syscall/v1/mount_memory".to_owned(),
                body: serde_json::to_vec(&json!({
                    "version": "v1",
                    "op": "mount_memory",
                    "args": {
                        "conversation_id": expected["conversation_id"],
                        "run_id": "run-private",
                        "stage": "final"
                    }
                }))
                .unwrap(),
                principal: json!({
                    "sub": "alice",
                    "role": "ro",
                    "ns": "conversation",
                    "storage_ns": "conversation",
                    "capabilities": ["project_view"]
                }),
            })
            .await;
        assert_eq!(
            private_syscall_response.status,
            StatusCode::NOT_FOUND.as_u16()
        );
        let private_payload: Value =
            serde_json::from_slice(&private_syscall_response.body).unwrap();
        assert_eq!(
            private_payload["code"],
            "KOGWISTAR_CONTEXT_SNAPSHOT_NOT_FOUND"
        );
        assert!(!private_payload.to_string().contains("private snapshot"));

        let invoke_snapshot_response = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/syscall/v1/invoke_tool".to_owned(),
                body: serde_json::to_vec(&json!({
                    "version": "v1",
                    "op": "invoke_tool",
                    "args": {
                        "tool_name": "latest_snapshot",
                        "conversation_id": expected["conversation_id"],
                        "run_id": "run-1",
                        "stage": "final"
                    }
                }))
                .unwrap(),
                principal: json!({
                    "sub": "alice",
                    "role": "rw",
                    "ns": "conversation",
                    "storage_ns": "conversation",
                    "capabilities": ["project_view"]
                }),
            })
            .await;
        assert_eq!(invoke_snapshot_response.status, StatusCode::OK.as_u16());
        let invoke_snapshot: Value =
            serde_json::from_slice(&invoke_snapshot_response.body).unwrap();
        assert_eq!(invoke_snapshot["version"], "v1");
        assert_eq!(invoke_snapshot["op"], "invoke_tool");
        assert_eq!(invoke_snapshot["status"], "ok");
        assert_eq!(
            invoke_snapshot["result"],
            json!({
                "tool_name": "latest_snapshot",
                "conversation_id": expected["conversation_id"],
                "snapshot": expected
            })
        );
        let initial_invoke_digest = format!(
            "{:x}",
            sha2::Sha256::digest(
                kogwistar_contracts::canonical_json(&invoke_snapshot["result"], false).as_bytes(),
            )
        );

        drop(service);
        let reopened = PostgresRunApplicationService::from_dsn(&dsn, &schema).unwrap();
        reopened.ensure_graph_type_schemas().await.unwrap();
        reopened
            .conversation_store
            .create_graph_schema(3, kogwistar_store_postgres::GraphTableNames::default())
            .await
            .unwrap();
        let retried_invoke_snapshot = reopened
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/syscall/v1/invoke_tool".to_owned(),
                body: serde_json::to_vec(&json!({
                    "version": "v1",
                    "op": "invoke_tool",
                    "args": {
                        "tool_name": "latest_snapshot",
                        "conversation_id": expected["conversation_id"],
                        "run_id": "run-1",
                        "stage": "final"
                    }
                }))
                .unwrap(),
                principal: json!({
                    "sub": "alice",
                    "role": "rw",
                    "ns": "conversation",
                    "storage_ns": "conversation",
                    "capabilities": ["project_view"]
                }),
            })
            .await;
        assert_eq!(
            retried_invoke_snapshot.status,
            StatusCode::OK.as_u16()
        );
        let retried_invoke_snapshot: Value =
            serde_json::from_slice(&retried_invoke_snapshot.body).unwrap();
        assert_eq!(retried_invoke_snapshot["op"], "invoke_tool");
        assert_eq!(
            format!(
                "{:x}",
                sha2::Sha256::digest(
                    kogwistar_contracts::canonical_json(
                        &retried_invoke_snapshot["result"],
                        false,
                    )
                    .as_bytes(),
                )
            ),
            initial_invoke_digest,
            "PostgreSQL syscall retry returns the same result after service reopen"
        );

        drop(reopened);
        let (cleanup, connection) = tokio_postgres::connect(&dsn, tokio_postgres::NoTls)
            .await
            .unwrap();
        tokio::spawn(async move {
            let _ = connection.await;
        });
        cleanup
            .batch_execute(&format!(
                "DROP SCHEMA IF EXISTS \"{schema}_conversation\" CASCADE; \
                 DROP SCHEMA IF EXISTS \"{schema}_workflow\" CASCADE; \
                 DROP SCHEMA IF EXISTS \"{schema}_wisdom\" CASCADE; \
                 DROP SCHEMA IF EXISTS \"{schema}\" CASCADE"
            ))
            .await
            .unwrap();
    }

    #[tokio::test]
    #[ignore = "requires KOGWISTAR_TEST_PG_DSN; run explicitly with --include-ignored"]
    async fn postgres_conversation_routes_project_authorized_summary_and_transcript_when_dsn_available()
     {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use tower::ServiceExt;

        let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
            .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
        let schema = format!("rust_conversation_detail_{}", uuid::Uuid::new_v4().simple());
        let service = PostgresRunApplicationService::from_dsn(&dsn, &schema).unwrap();
        service.ensure_graph_type_schemas().await.unwrap();
        service
            .conversation_store
            .create_graph_schema(3, kogwistar_store_postgres::GraphTableNames::default())
            .await
            .unwrap();
        let scope = GraphScope {
            namespace: "conversation".to_owned(),
            workspace_id: None,
            graph_space: None,
        };
        for (id, conversation_id, entity_type, extra) in [
            (
                "start-1",
                "conversation-1",
                "conversation_start",
                json!({"user_id": "alice", "status": "active"}),
            ),
            (
                "turn-1",
                "conversation-1",
                "conversation_turn",
                json!({"turn_index": 1, "role": "user", "summary": "first"}),
            ),
            (
                "turn-2",
                "conversation-1",
                "conversation_turn",
                json!({"turn_index": 2, "role": "assistant", "summary": "second"}),
            ),
            (
                "start-other-user",
                "conversation-other",
                "conversation_start",
                json!({"user_id": "bob", "status": "active"}),
            ),
        ] {
            let mut metadata = Map::new();
            metadata.insert("entity_type".to_owned(), json!(entity_type));
            metadata.insert("conversation_id".to_owned(), json!(conversation_id));
            metadata.extend(extra.as_object().unwrap().clone());
            service
                .conversation_store
                .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                    scope: scope.clone(),
                    table: "gke_nodes".to_owned(),
                    record: GraphRecord {
                        id: id.to_owned(),
                        document: None,
                        metadata,
                        embedding: None,
                    },
                    embedding_dim: 3,
                })
                .await
                .unwrap();
            let mut acl_metadata = Map::new();
            for (key, value) in [
                ("entity_type", json!("acl_record")),
                ("acl_truth_graph", json!("conversation")),
                ("acl_target_entity_id", json!(id)),
                ("acl_target_grain", json!("node")),
                ("acl_version", json!(1)),
                ("acl_mode", json!("public")),
            ] {
                acl_metadata.insert(key.to_owned(), value);
            }
            service
                .conversation_store
                .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                    scope: scope.clone(),
                    table: "gke_nodes".to_owned(),
                    record: GraphRecord {
                        id: format!("acl-{id}"),
                        document: None,
                        metadata: acl_metadata,
                        embedding: None,
                    },
                    embedding_dim: 3,
                })
                .await
                .unwrap();
        }

        let mut private_turn_metadata = Map::new();
        for (key, value) in [
            ("entity_type", json!("conversation_turn")),
            ("conversation_id", json!("conversation-1")),
            ("turn_index", json!(3)),
            ("role", json!("assistant")),
            ("summary", json!("private")),
        ] {
            private_turn_metadata.insert(key.to_owned(), value);
        }
        service
            .conversation_store
            .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                scope: scope.clone(),
                table: "gke_nodes".to_owned(),
                record: GraphRecord {
                    id: "turn-private".to_owned(),
                    document: None,
                    metadata: private_turn_metadata,
                    embedding: None,
                },
                embedding_dim: 3,
            })
            .await
            .unwrap();
        let private_acl = Map::from_iter([
            ("entity_type".to_owned(), json!("acl_record")),
            ("acl_truth_graph".to_owned(), json!("conversation")),
            ("acl_target_entity_id".to_owned(), json!("turn-private")),
            ("acl_target_grain".to_owned(), json!("node")),
            ("acl_version".to_owned(), json!(1)),
            ("acl_mode".to_owned(), json!("private")),
            ("owner_id".to_owned(), json!("bob")),
        ]);
        service
            .conversation_store
            .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                scope: scope.clone(),
                table: "gke_nodes".to_owned(),
                record: GraphRecord {
                    id: "acl-turn-private".to_owned(),
                    document: None,
                    metadata: private_acl,
                    embedding: None,
                },
                embedding_dim: 3,
            })
            .await
            .unwrap();

        for (id, entity_type, summary, status, acl_mode) in [
            (
                "tool-call:1",
                "tool_call",
                "Read adjacent graph nodes",
                "requested",
                "public",
            ),
            (
                "tool-result:1",
                "tool_result",
                "Returned two visible neighbors",
                "completed",
                "public",
            ),
            (
                "tool-call-other",
                "tool_call",
                "Other user's private call",
                "requested",
                "public",
            ),
            (
                "tool-call-private",
                "tool_call",
                "Private call owned by another user",
                "requested",
                "private",
            ),
        ] {
            let conversation_id = if id == "tool-call-other" {
                "conversation-other"
            } else {
                "conversation:demo"
            };
            let mut metadata = Map::new();
            metadata.insert("entity_type".to_owned(), json!(entity_type));
            metadata.insert("conversation_id".to_owned(), json!(conversation_id));
            metadata.insert("tool_name".to_owned(), json!("kg_neighbors"));
            metadata.insert("summary".to_owned(), json!(summary));
            metadata.insert("turn_index".to_owned(), json!(2));
            let mut properties = json!({"status": status});
            if id == "tool-result:1" {
                properties["count"] = json!(2);
            }
            let record = GraphRecord {
                id: id.to_owned(),
                document: Some(json!({"properties": properties}).to_string()),
                metadata,
                embedding: None,
            };
            service
                .conversation_store
                .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                    scope: scope.clone(),
                    table: "gke_nodes".to_owned(),
                    record,
                    embedding_dim: 3,
                })
                .await
                .unwrap();
            let mut acl_metadata = Map::new();
            for (key, value) in [
                ("entity_type", json!("acl_record")),
                ("acl_truth_graph", json!("conversation")),
                ("acl_target_entity_id", json!(id)),
                ("acl_target_grain", json!("node")),
                ("acl_version", json!(1)),
                ("acl_mode", json!(acl_mode)),
            ] {
                acl_metadata.insert(key.to_owned(), value);
            }
            if acl_mode == "private" {
                acl_metadata.insert("owner_id".to_owned(), json!("bob"));
            }
            service
                .conversation_store
                .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                    scope: scope.clone(),
                    table: "gke_nodes".to_owned(),
                    record: GraphRecord {
                        id: format!("acl-{id}"),
                        document: None,
                        metadata: acl_metadata,
                        embedding: None,
                    },
                    embedding_dim: 3,
                })
                .await
                .unwrap();
        }

        let response = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: "/api/conversations/conversation-1".to_owned(),
                body: Vec::new(),
                principal: json!({
                    "sub": "alice",
                    "role": "ro",
                    "ns": "conversation",
                    "storage_ns": "conversation",
                    "capabilities": ["read_graph"]
                }),
            })
            .await;
        assert_eq!(response.status, StatusCode::OK.as_u16());
        let payload: Value = serde_json::from_slice(&response.body).unwrap();
        assert_eq!(payload["conversation_id"], "conversation-1");
        assert_eq!(payload["user_id"], "alice");
        assert_eq!(payload["status"], "active");
        assert_eq!(payload["start_node_id"], "start-1");
        assert_eq!(payload["tail_node_id"], "turn-2");
        assert_eq!(payload["turn_count"], 2);
        assert!(payload.get("schema_version").is_none());
        assert!(payload.get("implementation_authority").is_none());
        assert!(payload.get("seam_attestation_version").is_none());
        assert!(payload.get("seam_crossings").is_none());
        assert_eq!(
            payload
                .as_object()
                .unwrap()
                .keys()
                .cloned()
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([
                "conversation_id".to_owned(),
                "start_node_id".to_owned(),
                "status".to_owned(),
                "tail_node_id".to_owned(),
                "turn_count".to_owned(),
                "user_id".to_owned(),
            ])
        );

        let list_response = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: "/api/conversations".to_owned(),
                body: Vec::new(),
                principal: json!({
                    "sub": "alice",
                    "role": "ro",
                    "ns": "conversation",
                    "storage_ns": "conversation",
                    "capabilities": ["read_graph"]
                }),
            })
            .await;
        assert_eq!(list_response.status, StatusCode::OK.as_u16());
        let list: Value = serde_json::from_slice(&list_response.body).unwrap();
        assert_eq!(
            list,
            json!({
                "conversations": [{
                    "id": "conversation-1",
                    "start_node_id": "start-1",
                    "status": "active",
                    "turn_count": 2
                }]
            })
        );

        let transcript_response = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: "/api/conversations/conversation-1/turns".to_owned(),
                body: Vec::new(),
                principal: json!({
                    "sub": "alice",
                    "role": "ro",
                    "ns": "conversation",
                    "storage_ns": "conversation",
                    "capabilities": ["read_graph"]
                }),
            })
            .await;
        assert_eq!(transcript_response.status, StatusCode::OK.as_u16());
        let transcript: Value = serde_json::from_slice(&transcript_response.body).unwrap();
        assert_eq!(transcript["conversation_id"], "conversation-1");
        assert_eq!(transcript["turns"].as_array().unwrap().len(), 2);
        assert_eq!(transcript["turns"][0]["node_id"], "turn-1");
        assert_eq!(transcript["turns"][0]["turn_index"], 1);
        assert_eq!(transcript["turns"][0]["role"], "user");
        assert_eq!(transcript["turns"][0]["content"], "first");
        assert_eq!(transcript["turns"][0]["entity_type"], "conversation_turn");
        assert_eq!(transcript["turns"][1]["node_id"], "turn-2");
        assert_eq!(transcript["turns"][1]["turn_index"], 2);
        assert_eq!(transcript["turns"][1]["role"], "assistant");
        assert_eq!(transcript["turns"][1]["content"], "second");
        assert_eq!(transcript["turns"][1]["entity_type"], "conversation_turn");
        assert!(transcript.get("schema_version").is_none());
        assert!(transcript.get("implementation_authority").is_none());
        assert!(transcript["turns"].as_array().unwrap().iter().all(|turn| {
            turn.get("schema_version").is_none()
        }));
        assert_eq!(
            transcript
                .as_object()
                .unwrap()
                .keys()
                .cloned()
                .collect::<BTreeSet<_>>(),
            BTreeSet::from(["conversation_id".to_owned(), "turns".to_owned()])
        );

        let audit_response = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: "/api/workflow/tools/audit?conversation_id=conversation%3Ademo&limit=10"
                    .to_owned(),
                body: Vec::new(),
                principal: json!({
                    "sub": "alice",
                    "role": "ro",
                    "ns": "conversation",
                    "storage_ns": "conversation",
                    "capabilities": ["project_view", "read_graph"]
                }),
            })
            .await;
        assert_eq!(audit_response.status, StatusCode::OK.as_u16());
        let audit: Value = serde_json::from_slice(&audit_response.body).unwrap();
        let mut expected_audit: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/tool-audit-v1.json"
        ))
        .unwrap();
        expected_audit.as_object_mut().unwrap().remove("schema_version");
        for item in expected_audit["items"].as_array_mut().unwrap() {
            item.as_object_mut().unwrap().remove("schema_version");
        }
        assert_eq!(audit, expected_audit);
        assert_eq!(audit["conversation_id"], "conversation:demo");
        assert_eq!(audit["items"].as_array().unwrap().len(), 2);
        assert_eq!(audit["items"][0]["node_id"], "tool-call:1");
        assert_eq!(audit["items"][1]["node_id"], "tool-result:1");
        assert_eq!(audit["items"][1]["properties"]["status"], "completed");
        assert!(audit["items"].as_array().unwrap().iter().all(|item| {
            item["node_id"] != "tool-call-private" && item["node_id"] != "tool-call-other"
        }));
        assert!(audit.get("schema_version").is_none());
        assert!(audit["items"][0].get("schema_version").is_none());

        let receive_request = |principal: Value| ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/syscall/v1/receive_message".to_owned(),
                body: serde_json::to_vec(&json!({
                    "version": "v1",
                    "op": "receive_message",
                    "args": {"conversation_id": "conversation-1", "limit": 0}
                }))
                .unwrap(),
                principal
            };
        let missing_id_response = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/syscall/v1/receive_message".to_owned(),
                body: serde_json::to_vec(&json!({
                    "version": "v1",
                    "op": "receive_message",
                    "args": {}
                }))
                .unwrap(),
                principal: json!({
                    "sub": "alice",
                    "role": "ro",
                    "ns": "conversation",
                    "storage_ns": "conversation",
                    "capabilities": ["read_graph"]
                }),
            })
            .await;
        let receive_error_contract: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/syscall-receive-message-missing-id-v1.json"
        ))
        .unwrap();
        assert_eq!(missing_id_response.status, receive_error_contract["http_status"]);
        assert_eq!(
            serde_json::from_slice::<Value>(&missing_id_response.body).unwrap(),
            receive_error_contract["body"]
        );
        let denied_receive = service
            .execute(receive_request(json!({
                "sub": "alice",
                "role": "ro",
                "ns": "conversation",
                "storage_ns": "conversation",
                "capabilities": []
            })))
            .await;
        assert_eq!(denied_receive.status, StatusCode::FORBIDDEN.as_u16());
        let wrong_namespace_receive = service
            .execute(receive_request(json!({
                "sub": "alice",
                "role": "ro",
                "ns": "workflow",
                "storage_ns": "conversation",
                "capabilities": ["read_graph"]
            })))
            .await;
        assert_eq!(
            wrong_namespace_receive.status,
            StatusCode::FORBIDDEN.as_u16()
        );
        let receive_response = service
            .execute(receive_request(json!({
                    "sub": "alice",
                    "role": "ro",
                    "ns": "conversation",
                    "storage_ns": "conversation",
                    "capabilities": ["read_graph"]
                })))
            .await;
        assert_eq!(receive_response.status, StatusCode::OK.as_u16());
        let receive_payload: Value = serde_json::from_slice(&receive_response.body).unwrap();
        let mut expected_receive = serde_json::from_str::<Value>(include_str!(
            "../../../../../contracts/golden/receive-message-result-v1.json"
        ))
        .unwrap();
        // The fixture's conversation ID is illustrative; the response echoes this request ID.
        expected_receive["conversation_id"] = json!("conversation-1");
        assert_eq!(receive_payload["version"], "v1");
        assert_eq!(receive_payload["op"], "receive_message");
        assert_eq!(receive_payload["status"], "ok");
        assert_eq!(receive_payload["result"], expected_receive);

        let invoke_transcript_response = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/syscall/v1/invoke_tool".to_owned(),
                body: serde_json::to_vec(&json!({
                    "version": "v1",
                    "op": "invoke_tool",
                    "args": {
                        "tool_name": "list_transcript",
                        "conversation_id": "conversation-1"
                    }
                }))
                .unwrap(),
                principal: json!({
                    "sub": "alice",
                    "role": "rw",
                    "ns": "conversation",
                    "storage_ns": "conversation",
                    "capabilities": ["read_graph"]
                }),
            })
            .await;
        assert_eq!(invoke_transcript_response.status, StatusCode::OK.as_u16());
        let invoke_transcript: Value =
            serde_json::from_slice(&invoke_transcript_response.body).unwrap();
        assert_eq!(invoke_transcript["version"], "v1");
        assert_eq!(invoke_transcript["op"], "invoke_tool");
        assert_eq!(invoke_transcript["status"], "ok");
        assert_eq!(
            invoke_transcript["result"],
            json!({
                "tool_name": "list_transcript",
                "conversation_id": "conversation-1",
                "messages": expected_receive["messages"]
            })
        );

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let auth = AuthConfig {
            mode: "dev".to_owned(),
            algorithm: Some("HS256".to_owned()),
            key: Some("tool-audit-mcp-secret".to_owned()),
            jwks_json: None,
            issuer: Some("local".to_owned()),
            audience: None,
            oidc_providers_json: None,
            auth_db_url: None,
        };
        let token = encode(
            &Header::new(Algorithm::HS256),
            &json!({
                "sub": "alice",
                "role": "ro",
                "ns": ["workflow"],
                "storage_ns": "conversation",
                "capabilities": ["project_view", "read_graph"],
                "iss": "local",
                "exp": now + 300
            }),
            &EncodingKey::from_secret(b"tool-audit-mcp-secret"),
        )
        .unwrap();
        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: Vec::new(),
                implementation: ImplementationSnapshot::default(),
                auth,
            },
            Arc::new(service.clone()),
        );
        let http_audit = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/api/workflow/tools/audit?conversation_id=conversation%3Ademo&limit=10")
                    .header("authorization", format!("Bearer {token}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(http_audit.status(), StatusCode::OK);
        let http_audit: Value =
            serde_json::from_slice(&to_bytes(http_audit.into_body(), 65_536).await.unwrap())
                .unwrap();
        assert_eq!(http_audit, audit);

        let mcp_audit = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp/workflow")
                    .header("content-type", "application/json")
                    .header("authorization", format!("Bearer {token}"))
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":"audit","method":"tools/call","params":{"name":"workflow.tool_audit","arguments":{"conversation_id":"conversation:demo","limit":10}}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(mcp_audit.status(), StatusCode::OK);
        let mcp_audit: Value =
            serde_json::from_slice(&to_bytes(mcp_audit.into_body(), 65_536).await.unwrap())
                .unwrap();
        assert_eq!(mcp_audit["result"], audit);

        for (id, turn_index, workspace_id, graph_space, summary) in [
            ("scoped-turn-1", 1, "workspace-a", "conversation", "first scoped"),
            ("scoped-turn-2", 2, "workspace-a", "conversation", "second scoped"),
            (
                "foreign-workspace-turn",
                3,
                "workspace-b",
                "conversation",
                "foreign workspace",
            ),
            (
                "foreign-graph-space-turn",
                4,
                "workspace-a",
                "knowledge",
                "foreign graph space",
            ),
        ] {
            let scoped_record = GraphScope {
                namespace: "conversation".to_owned(),
                workspace_id: Some(workspace_id.to_owned()),
                graph_space: Some(graph_space.to_owned()),
            };
            for (record_id, metadata) in [
                (
                    id.to_owned(),
                    Map::from_iter([
                        ("entity_type".to_owned(), json!("conversation_turn")),
                        ("conversation_id".to_owned(), json!("scoped-conversation")),
                        ("turn_index".to_owned(), json!(turn_index)),
                        ("role".to_owned(), json!("assistant")),
                        ("summary".to_owned(), json!(summary)),
                    ]),
                ),
                (
                    format!("acl-{id}"),
                    Map::from_iter([
                        ("entity_type".to_owned(), json!("acl_record")),
                        ("acl_truth_graph".to_owned(), json!("conversation")),
                        ("acl_target_entity_id".to_owned(), json!(id)),
                        ("acl_target_grain".to_owned(), json!("node")),
                        ("acl_version".to_owned(), json!(1)),
                        ("acl_mode".to_owned(), json!("public")),
                    ]),
                ),
            ] {
                service
                    .conversation_store
                    .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                        scope: scoped_record.clone(),
                        table: "gke_nodes".to_owned(),
                        record: GraphRecord {
                            id: record_id,
                            document: None,
                            metadata,
                            embedding: None,
                        },
                        embedding_dim: 3,
                    })
                    .await
                    .unwrap();
            }
        }
        let scoped_receive = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/syscall/v1/receive_message".to_owned(),
                body: serde_json::to_vec(&json!({
                    "version": "v1",
                    "op": "receive_message",
                    "args": {"conversation_id": "scoped-conversation", "limit": 0}
                }))
                .unwrap(),
                principal: json!({
                    "sub": "alice",
                    "role": "ro",
                    "ns": "conversation",
                    "storage_ns": "conversation",
                    "workspace_id": "workspace-a",
                    "graph_space": "conversation",
                    "capabilities": ["read_graph"]
                }),
            })
            .await;
        assert_eq!(scoped_receive.status, StatusCode::OK.as_u16());
        let scoped_receive: Value = serde_json::from_slice(&scoped_receive.body).unwrap();
        assert_eq!(scoped_receive["result"]["messages"].as_array().unwrap().len(), 2);
        assert_eq!(scoped_receive["result"]["messages"][0]["node_id"], "scoped-turn-1");
        assert_eq!(scoped_receive["result"]["messages"][1]["node_id"], "scoped-turn-2");

        let (cleanup, connection) = tokio_postgres::connect(&dsn, tokio_postgres::NoTls)
            .await
            .unwrap();
        tokio::spawn(async move {
            let _ = connection.await;
        });
        cleanup
            .batch_execute(&format!(
                "DROP SCHEMA IF EXISTS \"{schema}_conversation\" CASCADE; \
                 DROP SCHEMA IF EXISTS \"{schema}_workflow\" CASCADE; \
                 DROP SCHEMA IF EXISTS \"{schema}_wisdom\" CASCADE; \
                 DROP SCHEMA IF EXISTS \"{schema}\" CASCADE"
            ))
            .await
            .unwrap();
    }
