    #[tokio::test]
    async fn sqlite_syscall_spawn_and_terminate_use_authoritative_run_service() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-syscall-{nonce}.sqlite3"));
        let service = SqliteRunApplicationService::open(&path).unwrap();
        let view_request = |capabilities: &[&str]| ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/syscall/v1/project_view".to_owned(),
            body: serde_json::to_vec(&json!({
                "version":"v1",
                "op":"project_view",
                "args":{}
            }))
            .unwrap(),
            principal: json!({
                "sub":"viewer",
                "role":"ro",
                "ns":"workflow",
                "capabilities":capabilities,
            }),
        };
        let denied_view = service.execute_sync(view_request(&["project_view"]));
        assert_eq!(denied_view.status, StatusCode::FORBIDDEN.as_u16());
        let view = service.execute_sync(view_request(&["project_view", "read_security_scope"]));
        assert_eq!(view.status, StatusCode::OK.as_u16());
        let view = serde_json::from_slice::<Value>(&view.body).unwrap();
        assert_eq!(view["version"], "v1");
        assert_eq!(view["op"], "project_view");
        assert_eq!(view["status"], "ok");
        assert!(view["result"]["scheduler"].is_object());
        assert!(view["result"]["runs"].is_object());
        let receive_request = |namespace: &str| ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/syscall/v1/receive_message".to_owned(),
            body: serde_json::to_vec(&json!({
                "version":"v1",
                "op":"receive_message",
                "args":{"conversation_id":"conversation-1"}
            }))
            .unwrap(),
            principal: json!({
                "sub":"viewer",
                "role":"ro",
                "ns":namespace,
                "capabilities":["read_graph"],
            }),
        };
        assert_eq!(
            service.execute_sync(receive_request("workflow")).status,
            StatusCode::FORBIDDEN.as_u16()
        );
        assert_eq!(
            service.execute_sync(receive_request("conversation")).status,
            StatusCode::NOT_IMPLEMENTED.as_u16()
        );
        let mount_request = |namespace: &str| ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/syscall/v1/mount_memory".to_owned(),
            body: serde_json::to_vec(&json!({
                "version":"v1",
                "op":"mount_memory",
                "args":{"conversation_id":"conversation-1"}
            }))
            .unwrap(),
            principal: json!({
                "sub":"viewer",
                "role":"ro",
                "ns":namespace,
                "capabilities":["project_view"],
            }),
        };
        assert_eq!(
            service.execute_sync(mount_request("workflow")).status,
            StatusCode::FORBIDDEN.as_u16()
        );
        assert_eq!(
            service.execute_sync(mount_request("conversation")).status,
            StatusCode::NOT_IMPLEMENTED.as_u16()
        );
        let principal = json!({
            "sub":"operator",
            "role":"rw",
            "ns":"workflow",
            "capabilities":["spawn_process","workflow.run.write","workflow.run.read"],
        });
        let spawned = service.execute_sync(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/syscall/v1/spawn_process".to_owned(),
            body: serde_json::to_vec(&json!({
                "version":"v1",
                "op":"spawn_process",
                "args":{
                    "workflow_id":"wf-syscall",
                    "conversation_id":"conv-syscall",
                    "initial_state":{
                        "source":"syscall",
                        "_rt_requested_capabilities":[
                            "workflow.run.write"
                        ],
                        "_rt_effective_capabilities":["service.manage"]
                    },
                },
            }))
            .unwrap(),
            principal: principal.clone(),
        });
        assert_eq!(spawned.status, StatusCode::OK.as_u16());
        let spawned = serde_json::from_slice::<Value>(&spawned.body).unwrap();
        assert_eq!(spawned["version"], "v1");
        assert_eq!(spawned["op"], "spawn_process");
        assert_eq!(spawned["status"], "ok");
        let run_id = spawned["result"]["run_id"].as_str().unwrap();
        assert!(service.store.get_server_run(run_id).unwrap().is_some());
        let lanes = service
            .store
            .list_projected_lane_messages(LaneMessageFilter {
                namespace: Some("workflow".to_owned()),
                inbox_id: Some("workflow-runtime".to_owned()),
                correlation_id: Some(run_id.to_owned()),
                ..LaneMessageFilter::default()
            })
            .unwrap();
        let payload =
            serde_json::from_str::<Value>(lanes[0].payload_json.as_deref().unwrap()).unwrap();
        assert_eq!(
            payload["initial_state"]["_rt_effective_capabilities"],
            json!(["workflow.run.write"])
        );
        assert_eq!(payload["initial_state"]["_rt_delegated_by"], "operator");
        assert_eq!(
            payload["initial_state"]["_rt_delegation"],
            json!({
                "schema_version": 1,
                "delegated_by": "operator",
                "effective_capabilities": ["workflow.run.write"]
            })
        );
        let terminated = service.execute_sync(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/syscall/v1/terminate_process".to_owned(),
            body: serde_json::to_vec(&json!({
                "version":"v1",
                "op":"terminate_process",
                "args":{"run_id":run_id},
            }))
            .unwrap(),
            principal,
        });
        assert_eq!(terminated.status, StatusCode::OK.as_u16());
        let terminated = serde_json::from_slice::<Value>(&terminated.body).unwrap();
        assert_eq!(terminated["result"]["cancel_requested"], true);
        assert!(
            service
                .store
                .get_server_run(run_id)
                .unwrap()
                .unwrap()
                .cancel_requested
        );
        drop(service);
        std::fs::remove_file(path).unwrap();
    }

    // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/sqlite_automation_services.rs#syscall_effect; python=kogwistar/server/syscall_api.py#create_syscall_router; fixture=contracts/golden/run-cancel-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/sqlite_runtime/worker_lifecycle.rs#sqlite_chroma_syscall_spawn_and_terminate_use_authoritative_run_service; python_test=tests/server/test_chat_server_api.py#test_run_lifecycle_syscalls_use_python_backend
    #[tokio::test]
    #[ignore = "requires KOGWISTAR_TEST_CHROMA_URL"]
    async fn sqlite_chroma_syscall_spawn_and_terminate_use_authoritative_run_service() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let endpoint = std::env::var("KOGWISTAR_TEST_CHROMA_URL")
            .expect("KOGWISTAR_TEST_CHROMA_URL must point to live Chroma");
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let suffix = uuid::Uuid::new_v4().simple().to_string();
        let config = |kind: &str| kogwistar_store_chroma::ChromaStoreConfig {
            endpoint: endpoint.clone(),
            collection: format!("lifecycle-{kind}-{suffix}"),
            ..Default::default()
        };
        let graph = kogwistar_store_chroma::ChromaGraphStore::connect_entity_collections(
            config("nodes"),
            config("edges"),
        )
        .await
        .unwrap();
        let path = std::env::temp_dir().join(format!("kogwistar-syscall-lifecycle-{nonce}.sqlite3"));
        let service = SqliteRunApplicationService::open(&path)
            .unwrap()
            .with_chroma_graph(graph);
        let principal = json!({
            "sub":"chroma-lifecycle-owner",
            "role":"rw",
            "ns":"workflow",
            "capabilities":["spawn_process", "workflow.run.write", "workflow.run.read"]
        });
        let spawned = service.execute_sync(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/syscall/v1/spawn_process".to_owned(),
            body: serde_json::to_vec(&json!({
                "version":"v1",
                "op":"spawn_process",
                "args":{
                    "workflow_id":"chroma-lifecycle-workflow",
                    "conversation_id":"chroma-lifecycle-conversation",
                    "initial_state":{"source":"sqlite-chroma"}
                }
            }))
            .unwrap(),
            principal: principal.clone(),
        });
        assert_eq!(spawned.status, StatusCode::OK.as_u16());
        let spawned: Value = serde_json::from_slice(&spawned.body).unwrap();
        let submit_contract: Value = serde_json::from_str(include_str!(
            "../../../../../../contracts/golden/workflow-run-submit-http-v1.json"
        ))
        .unwrap();
        let spawned_result = &spawned["result"];
        assert!(submit_contract["required_common_fields"]
            .as_array()
            .unwrap()
            .iter()
            .all(|field| spawned_result.get(field.as_str().unwrap()).is_some()));
        for (field, value) in submit_contract["expected_common_values"].as_object().unwrap() {
            assert_eq!(&spawned_result[field], value, "field {field}");
        }
        let run_id = spawned["result"]["run_id"].as_str().unwrap();
        assert!(service.store.get_server_run(run_id).unwrap().is_some());

        let denied_termination = service.execute_sync(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/syscall/v1/terminate_process".to_owned(),
            body: serde_json::to_vec(&json!({
                "version":"v1",
                "op":"terminate_process",
                "args":{"run_id":run_id}
            }))
            .unwrap(),
            principal: json!({
                "sub":"chroma-lifecycle-owner", "role":"rw", "ns":"workflow",
                "capabilities":["workflow.run.read"]
            }),
        });
        assert_eq!(denied_termination.status, StatusCode::FORBIDDEN.as_u16());
        assert!(!service
            .store
            .get_server_run(run_id)
            .unwrap()
            .unwrap()
            .cancel_requested);

        let terminated = service.execute_sync(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/syscall/v1/terminate_process".to_owned(),
            body: serde_json::to_vec(&json!({
                "version":"v1",
                "op":"terminate_process",
                "args":{"run_id":run_id}
            }))
            .unwrap(),
            principal: principal.clone(),
        });
        assert_eq!(terminated.status, StatusCode::OK.as_u16());
        let terminated: Value = serde_json::from_slice(&terminated.body).unwrap();
        let cancel_contract: Value = serde_json::from_str(include_str!(
            "../../../../../../contracts/golden/run-cancel-v1.json"
        ))
        .unwrap();
        for field in cancel_contract["required_fields"].as_array().unwrap() {
            let field = field.as_str().unwrap();
            assert!(terminated["result"].get(field).is_some(), "missing {field}");
        }
        assert!(cancel_contract["accepted_run_statuses"]
            .as_array()
            .unwrap()
            .contains(&terminated["result"]["status"]));
        assert_eq!(terminated["result"]["cancel_requested"], true);
        assert!(service
            .store
            .get_server_run(run_id)
            .unwrap()
            .unwrap()
            .cancel_requested);
        let repeated_termination = service.execute_sync(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/syscall/v1/terminate_process".to_owned(),
            body: serde_json::to_vec(&json!({
                "version":"v1",
                "op":"terminate_process",
                "args":{"run_id":run_id}
            }))
            .unwrap(),
            principal: principal.clone(),
        });
        assert_eq!(repeated_termination.status, StatusCode::OK.as_u16());
        let repeated_termination: Value =
            serde_json::from_slice(&repeated_termination.body).unwrap();
        assert!(cancel_contract["accepted_run_statuses"]
            .as_array()
            .unwrap()
            .contains(&repeated_termination["result"]["status"]));
        assert_eq!(repeated_termination["result"]["cancel_requested"], true);
        drop(service);
        std::fs::remove_file(path).unwrap();
    }

    // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/document_graph.rs#request_approval_syscall_effect; python=kogwistar/server/syscall_api.py#create_syscall_router; fixture=contracts/golden/syscall-request-approval-success-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/sqlite_runtime/worker_lifecycle.rs#sqlite_chroma_request_approval_matches_shared_fixture; python_test=tests/server/test_chat_server_api.py#test_request_approval_syscall_uses_python_backend
    #[tokio::test]
    #[ignore = "requires KOGWISTAR_TEST_CHROMA_URL"]
    async fn sqlite_chroma_request_approval_matches_shared_fixture() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let endpoint = std::env::var("KOGWISTAR_TEST_CHROMA_URL")
            .expect("KOGWISTAR_TEST_CHROMA_URL must point to live Chroma");
        let suffix = uuid::Uuid::new_v4().simple().to_string();
        let config = |kind: &str| kogwistar_store_chroma::ChromaStoreConfig {
            endpoint: endpoint.clone(),
            collection: format!("approval-{kind}-{suffix}"),
            ..Default::default()
        };
        let graph = kogwistar_store_chroma::ChromaGraphStore::connect_entity_collections(
            config("nodes"),
            config("edges"),
        )
        .await
        .unwrap();
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-syscall-approval-chroma-{nonce}.sqlite3"));
        let service = SqliteRunApplicationService::open(&path)
            .unwrap()
            .with_chroma_graph(graph);
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../../../../contracts/golden/syscall-request-approval-success-v1.json"
        ))
        .unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let response = service.execute_sync(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/syscall/v1/request_approval".to_owned(),
                body: serde_json::to_vec(&case["request"]).unwrap(),
                principal: json!({
                    "sub":"shared-approval-owner",
                    "role":"rw",
                    "ns":"workflow",
                    "capabilities":["approve_action", "project_view", "read_security_scope"]
                }),
            });
            assert_eq!(response.status, case["http_status"].as_u64().unwrap() as u16);
            let payload: Value = serde_json::from_slice(&response.body).unwrap();
            assert_eq!(payload["version"], case["request"]["version"]);
            assert_eq!(payload["op"], case["request"]["op"]);
            assert_eq!(payload["status"], "ok");
            assert!(payload.get("error").is_some_and(Value::is_null));
            let result = &payload["result"];
            assert_eq!(result["status"], case["result_status"]);
            let entries = result[case["snapshot_field"].as_str().unwrap()]
                [case["entries_field"].as_str().unwrap()]
                .as_array()
                .unwrap();
            assert!(entries.contains(&case["expected_entry"]));
        }
        let unrecognized_fixture: Value = serde_json::from_str(include_str!(
            "../../../../../../contracts/golden/syscall-request-approval-unrecognized-v1.json"
        ))
        .unwrap();
        let unrecognized = service.execute_sync(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/syscall/v1/request_approval".to_owned(),
            body: serde_json::to_vec(&unrecognized_fixture["request"]).unwrap(),
            principal: json!({
                "sub":"shared-approval-owner",
                "role":"rw",
                "ns":"workflow",
                "capabilities":["approve_action", "project_view", "read_security_scope"]
            }),
        });
        assert_eq!(
            unrecognized.status,
            unrecognized_fixture["http_status"].as_u64().unwrap() as u16
        );
        assert_eq!(
            serde_json::from_slice::<Value>(&unrecognized.body).unwrap(),
            unrecognized_fixture["response"]
        );
        let denied = service.execute_sync(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/syscall/v1/request_approval".to_owned(),
            body: serde_json::to_vec(&fixture["cases"][0]["request"]).unwrap(),
            principal: json!({
                "sub":"shared-approval-owner",
                "role":"rw",
                "ns":"workflow",
                "capabilities":["project_view", "read_security_scope"]
            }),
        });
        assert_eq!(denied.status, StatusCode::FORBIDDEN.as_u16());
        drop(service);
        std::fs::remove_file(path).unwrap();
    }

    // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/sqlite_chroma_conversation_reads.rs#sqlite_chroma_mount_memory_effect; python=kogwistar/server/syscall_api.py#mount_memory; fixture=contracts/golden/mount-memory-result-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/sqlite_runtime/worker_lifecycle.rs#sqlite_chroma_mount_memory_matches_shared_fixture_and_acl; python_test=tests/server/test_chat_server_api.py#test_mount_memory_syscall_uses_python_backend
    #[tokio::test]
    #[ignore = "requires KOGWISTAR_TEST_CHROMA_URL"]
    async fn sqlite_chroma_mount_memory_matches_shared_fixture_and_acl() {
        use kogwistar_store::{GraphEntityKind, GraphRecord, GraphWriteStore};

        let endpoint = std::env::var("KOGWISTAR_TEST_CHROMA_URL")
            .expect("KOGWISTAR_TEST_CHROMA_URL must point to live Chroma");
        let suffix = uuid::Uuid::new_v4().simple().to_string();
        let namespace = format!("mount-memory-{suffix}");
        let node_collection = format!("mount-memory-n-{suffix}");
        let edge_collection = format!("mount-memory-e-{suffix}");
        let path = std::env::temp_dir().join(format!("kogwistar-mount-memory-{suffix}.sqlite3"));
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../../../../contracts/golden/mount-memory-result-v1.json"
        ))
        .unwrap();
        let service = open_sqlite_application_service(
            &path,
            Some(SqliteChromaGraphConfig {
                endpoint: endpoint.clone(),
                metric: DistanceMetric::Cosine,
                node_collection: node_collection.clone(),
                edge_collection: edge_collection.clone(),
            }),
        )
        .await
        .unwrap();
        let common = kogwistar_store_chroma::ChromaStoreConfig {
            endpoint,
            metric: DistanceMetric::Cosine,
            ..Default::default()
        };
        let graph = kogwistar_store_chroma::ChromaGraphStore::connect_entity_collections(
            kogwistar_store_chroma::ChromaStoreConfig {
                collection: node_collection,
                ..common.clone()
            },
            kogwistar_store_chroma::ChromaStoreConfig {
                collection: edge_collection,
                ..common
            },
        )
        .await
        .unwrap();
        let snapshot_id = fixture["snapshot_node_id"].as_str().unwrap();
        let conversation_id = fixture["conversation_id"].as_str().unwrap();
        graph
            .upsert_graph_record_of_kind(
                &namespace,
                GraphEntityKind::Node,
                GraphRecord {
                    id: snapshot_id.to_owned(),
                    document: Some(fixture.to_string()),
                    metadata: serde_json::from_value(json!({
                        "entity_type":"context_snapshot",
                        "conversation_id":conversation_id,
                        "run_id":"memory-run",
                        "stage":"draft_answer",
                        "snapshot_seq":fixture["properties"]["snapshot_seq"],
                        "workspace_id":"workspace-a",
                        "graph_space":"conversation",
                        "lifecycle_status":"active"
                    }))
                    .unwrap(),
                    embedding: Some(vec![0.6, 0.8]),
                },
            )
            .await
            .unwrap();
        let acl_id = format!("acl-{snapshot_id}");
        service
            .store
            .append_raw_entity_event(
                &namespace,
                SqliteNewRawEntityEvent {
                    event_id: format!("event-{acl_id}"),
                    entity_kind: "node".to_owned(),
                    entity_id: acl_id.clone(),
                    op: "ADD".to_owned(),
                    payload_json: json!({
                        "id":acl_id,
                        "document":null,
                        "metadata":{
                            "entity_type":"acl_record",
                            "acl_truth_graph":"conversation",
                            "acl_target_grain":"node",
                            "acl_target_entity_id":snapshot_id,
                            "acl_version":1,
                            "acl_mode":"private",
                            "owner_id":"memory-owner",
                            "workspace_id":"workspace-a",
                            "graph_space":"conversation"
                        }
                    })
                    .to_string(),
                },
            )
            .unwrap();

        let request = |username: &str, run_id: &str| ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/syscall/v1/mount_memory".to_owned(),
            body: json!({
                "version":"v1",
                "op":"mount_memory",
                "args":{
                    "conversation_id":conversation_id,
                    "run_id":run_id,
                    "stage":"draft_answer"
                }
            })
            .to_string()
            .into_bytes(),
            principal: json!({
                "sub":username,
                "role":"rw",
                "storage_ns":namespace,
                "workspace_id":"workspace-a",
                "graph_space":"conversation",
                "ns":["conversation"],
                "capabilities":["project_view"]
            }),
        };
        let owner = service.execute(request("memory-owner", "memory-run")).await;
        assert_eq!(owner.status, StatusCode::OK.as_u16());
        let owner: Value = serde_json::from_slice(&owner.body).unwrap();
        assert_eq!(owner["result"]["snapshot_node_id"], fixture["snapshot_node_id"]);
        assert_eq!(owner["result"]["conversation_id"], fixture["conversation_id"]);
        assert_eq!(owner["result"]["metadata"], fixture["metadata"]);
        assert_eq!(owner["result"]["properties"], fixture["properties"]);

        let missing_snapshot = service
            .execute(request("memory-owner", "missing-memory-run"))
            .await;
        assert_eq!(missing_snapshot.status, StatusCode::NOT_FOUND.as_u16());
        assert!(!String::from_utf8_lossy(&missing_snapshot.body).contains(snapshot_id));

        let outsider = service
            .execute(request("unrelated-reader", "memory-run"))
            .await;
        assert_eq!(outsider.status, StatusCode::NOT_FOUND.as_u16());
        assert!(!String::from_utf8_lossy(&outsider.body).contains(snapshot_id));
        drop(graph);
        drop(service);
        let _ = std::fs::remove_file(path);
    }

    // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/sqlite_chroma_write_effects.rs#sqlite_chroma_invoke_builtin_tool_effect; python=kogwistar/server/syscall_api.py#create_syscall_router.dispatch; fixture=contracts/golden/invoke-tool-results-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/sqlite_runtime/worker_lifecycle.rs#sqlite_chroma_invoke_list_transcript_reads_live_graph; python_test=tests/server/test_chat_server_api.py#test_invoke_tool_syscall_uses_python_backend
    #[tokio::test]
    #[ignore = "requires KOGWISTAR_TEST_CHROMA_URL"]
    async fn sqlite_chroma_invoke_list_transcript_reads_live_graph() {
        use kogwistar_store::{GraphEntityKind, GraphRecord, GraphWriteStore};

        let endpoint = std::env::var("KOGWISTAR_TEST_CHROMA_URL")
            .expect("KOGWISTAR_TEST_CHROMA_URL must point to live Chroma");
        let suffix = uuid::Uuid::new_v4().simple().to_string();
        let node_collection = format!("invoke-tool-n-{suffix}");
        let edge_collection = format!("invoke-tool-e-{suffix}");
        let namespace = "conversation".to_owned();
        let path = std::env::temp_dir().join(format!("kogwistar-invoke-tool-{suffix}.sqlite3"));
        let service = open_sqlite_application_service(
            &path,
            Some(SqliteChromaGraphConfig {
                endpoint: endpoint.clone(),
                metric: DistanceMetric::Cosine,
                node_collection: node_collection.clone(),
                edge_collection: edge_collection.clone(),
            }),
        )
        .await
        .unwrap();
        let principal = |capabilities: &[&str]| {
            json!({
                "sub":"list-owner",
                "role":"rw",
                "storage_ns":namespace,
                "graph_space":"conversation",
                "ns":["conversation"],
                "capabilities":capabilities
            })
        };
        let conversation_id = format!("conversation-{suffix}");
        let created = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/syscall/v1/create_conversation".to_owned(),
                body: json!({
                    "version":"v1",
                    "op":"create_conversation",
                    "args":{"conversation_id":conversation_id,"user_id":"list-owner"}
                })
                .to_string()
                .into_bytes(),
                principal: principal(&["spawn_process"]),
            })
            .await;
        assert_eq!(
            created.status,
            StatusCode::OK.as_u16(),
            "{}",
            String::from_utf8_lossy(&created.body)
        );
        let created_payload: Value = serde_json::from_slice(&created.body).unwrap();
        let start_node_id = created_payload["result"]["start_node_id"]
            .as_str()
            .unwrap();
        let graph = kogwistar_store_chroma::ChromaGraphStore::connect_entity_collections(
            kogwistar_store_chroma::ChromaStoreConfig {
                endpoint: endpoint.clone(),
                collection: node_collection,
                metric: DistanceMetric::Cosine,
                ..Default::default()
            },
            kogwistar_store_chroma::ChromaStoreConfig {
                endpoint,
                collection: edge_collection,
                metric: DistanceMetric::Cosine,
                ..Default::default()
            },
        )
        .await
        .unwrap();
        graph
            .upsert_graph_record_of_kind(
                &namespace,
                GraphEntityKind::Node,
                GraphRecord {
                    id: start_node_id.to_owned(),
                    document: Some("conversation start".to_owned()),
                    metadata: serde_json::from_value(json!({
                        "entity_type":"conversation_start",
                        "conversation_id":conversation_id,
                        "user_id":"list-owner",
                        "lifecycle_status":"active",
                        "graph_space":"conversation"
                    }))
                    .unwrap(),
                    embedding: Some(vec![0.6, 0.8]),
                },
            )
            .await
            .unwrap();
        let invoke = |capabilities: &[&str]| ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/syscall/v1/invoke_tool".to_owned(),
            body: json!({
                "version":"v1",
                "op":"invoke_tool",
                "args":{"tool_name":"list_transcript","conversation_id":conversation_id}
            })
            .to_string()
            .into_bytes(),
            principal: principal(capabilities),
        };
        let denied = service
            .execute(invoke(&["invoke_tool"]))
            .await;
        assert_eq!(denied.status, StatusCode::FORBIDDEN.as_u16());

        let error_fixture: Value = serde_json::from_str(include_str!(
            "../../../../../../contracts/golden/syscall-invoke-tool-errors-v1.json"
        ))
        .unwrap();
        let graph_ids_before_errors = graph
            .graph_records_of_kind_bounded(
                &namespace,
                GraphEntityKind::Node,
                &kogwistar_store::MetadataFilter::default(),
                100,
            )
            .await
            .unwrap()
            .into_iter()
            .map(|record| record.id)
            .collect::<std::collections::BTreeSet<_>>();
        for case in error_fixture["cases"].as_array().unwrap() {
            let response = service
                .execute(ApiEffectRequest {
                    contract_version: 1,
                    method: "POST".to_owned(),
                    path_and_query: "/api/syscall/v1/invoke_tool".to_owned(),
                    body: serde_json::to_vec(&case["request"]).unwrap(),
                    principal: principal(&["invoke_tool", "read_graph"]),
                })
                .await;
            assert_eq!(
                response.status,
                case["http_status"].as_u64().unwrap() as u16,
                "{}",
                case["name"]
            );
            assert_eq!(
                serde_json::from_slice::<Value>(&response.body).unwrap(),
                case["response"],
                "{}",
                case["name"]
            );
        }
        let graph_ids_after_errors = graph
            .graph_records_of_kind_bounded(
                &namespace,
                GraphEntityKind::Node,
                &kogwistar_store::MetadataFilter::default(),
                100,
            )
            .await
            .unwrap()
            .into_iter()
            .map(|record| record.id)
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(graph_ids_after_errors, graph_ids_before_errors);

        let response = service
            .execute(invoke(&["invoke_tool", "read_graph"]))
            .await;
        assert_eq!(
            response.status,
            StatusCode::OK.as_u16(),
            "{}",
            String::from_utf8_lossy(&response.body),
        );
        let payload: Value = serde_json::from_slice(&response.body).unwrap();
        let result = &payload["result"];
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../../../../contracts/golden/invoke-tool-results-v1.json"
        ))
        .unwrap();
        for field in fixture["results"]["list_transcript"]["root_fields"]
            .as_array()
            .unwrap()
        {
            assert!(result.get(field.as_str().unwrap()).is_some());
        }
        assert_eq!(result["tool_name"], "list_transcript");
        assert_eq!(result["conversation_id"], conversation_id);
        assert_eq!(result["messages"], json!([]));
        drop(service);
        let _ = std::fs::remove_file(path);
    }

    // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/sqlite_chroma_write_effects.rs#sqlite_chroma_send_message_effect; python=kogwistar/server/syscall_api.py#create_syscall_router.dispatch; fixture=contracts/golden/conversation-create-syscall-result-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/sqlite_runtime/worker_lifecycle.rs#sqlite_chroma_send_message_is_idempotent_and_durable; python_test=tests/server/test_chat_server_api.py#test_send_message_syscall_uses_python_backend
    #[tokio::test]
    #[ignore = "requires KOGWISTAR_TEST_CHROMA_URL"]
    async fn sqlite_chroma_send_message_is_idempotent_and_durable() {
        use kogwistar_engine::embedding::{EmbeddingFuture, EmbeddingProfile, EmbeddingProvider};
        use kogwistar_store::{NamedProjectionWrite, WorkflowDesignSnapshotWrite};
        use kogwistar_store_sqlite::SqliteStore;
        use std::sync::Arc;

        struct TestEmbeddingProvider {
            profile: EmbeddingProfile,
        }
        impl EmbeddingProvider for TestEmbeddingProvider {
            fn profile(&self) -> &EmbeddingProfile {
                &self.profile
            }

            fn embed<'a>(&'a self, texts: &'a [String]) -> EmbeddingFuture<'a> {
                Box::pin(async move { Ok(texts.iter().map(|_| vec![0.6, 0.8]).collect()) })
            }
        }

        let endpoint = std::env::var("KOGWISTAR_TEST_CHROMA_URL")
            .expect("KOGWISTAR_TEST_CHROMA_URL must point to live Chroma");
        let suffix = uuid::Uuid::new_v4().simple().to_string();
        let database = std::env::temp_dir().join(format!("kogwistar-send-message-{suffix}.sqlite3"));
        let auth_database = std::env::temp_dir().join(format!("kogwistar-send-auth-{suffix}.sqlite3"));
        let workflow_id = "agentic_answering.v2";
        let response_contract: Value = serde_json::from_str(include_str!(
            "../../../../../../contracts/golden/send-message-syscall-result-v1.json"
        ))
        .unwrap();
        let workflow_contract: Value = serde_json::from_str(include_str!(
            "../../../../../../contracts/golden/answer-workflow-runtime-contract-v1.json"
        ))
        .unwrap();
        let workflow = &workflow_contract["workflows"][workflow_id];
        let nodes = workflow["required_ops"]
            .as_array()
            .unwrap()
            .iter()
            .map(|operation| {
                let operation = operation.as_str().unwrap();
                json!({
                    "id":operation,
                    "label":operation,
                    "metadata":{
                        "entity_type":"workflow_node",
                        "workflow_id":workflow_id,
                        "wf_op":operation,
                        "wf_version":"v1",
                        "wf_executor_target":"rust-native",
                        "wf_start":operation=="start"
                    }
                })
            })
            .collect::<Vec<_>>();
        let edges = workflow["required_routes"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(index, route)| {
                json!({
                    "id":format!("send-message-route-{index}"),
                    "source_ids":[route["from"]],
                    "target_ids":[route["to"]],
                    "metadata":{
                        "entity_type":"workflow_edge",
                        "workflow_id":workflow_id,
                        "wf_predicate":route["predicate"],
                        "wf_priority":route["priority"],
                        "wf_is_default":route["default"],
                        "wf_multiplicity":route["multiplicity"]
                    }
                })
            })
            .collect::<Vec<_>>();
        let store = SqliteStore::open(&database).unwrap();
        store
            .replace_named_projection(
                "workflow_design",
                workflow_id,
                NamedProjectionWrite {
                    payload: serde_json::Map::from_iter([
                        ("current_version".to_owned(), json!(1)),
                        ("snapshot_schema_version".to_owned(), json!(1)),
                    ]),
                    last_authoritative_seq: 1,
                    last_materialized_seq: 1,
                    projection_schema_version: 1,
                    materialization_status: "ready".to_owned(),
                },
            )
            .unwrap();
        store
            .put_workflow_design_snapshot(
                workflow_id,
                WorkflowDesignSnapshotWrite {
                    version: 1,
                    seq: 1,
                    schema_version: 1,
                    payload_json: serde_json::to_string(&json!({"nodes":nodes,"edges":edges}))
                        .unwrap(),
                },
            )
            .unwrap();
        drop(store);
        let auth_store = SqliteAuthStore::open(&auth_database).unwrap();
        auth_store
            .resolve_external_identity(ResolveExternalIdentity {
                issuer: "test".to_owned(),
                subject: "send-owner".to_owned(),
                email: "send-owner@example.test".to_owned(),
                display_name: None,
                new_user_id: "send-owner".to_owned(),
                default_role: "rw".to_owned(),
                default_ns: "conversation".to_owned(),
            })
            .await
            .unwrap();
        drop(auth_store);
        let auth_config = AuthConfig {
            auth_db_url: Some(format!("sqlite:///{}", auth_database.display())),
            ..AuthConfig::default()
        };
        let provider = Arc::new(TestEmbeddingProvider {
            profile: EmbeddingProfile {
                provider: "test".to_owned(),
                model: "send-message-test".to_owned(),
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
        });
        let database_graph = open_sqlite_application_service(
            &database,
            Some(SqliteChromaGraphConfig {
                endpoint,
                metric: DistanceMetric::Cosine,
                node_collection: format!("send-message-n-{suffix}"),
                edge_collection: format!("send-message-e-{suffix}"),
            }),
        )
        .await
        .unwrap();
        let service = database_graph
            .with_embedding_provider(provider)
            .with_owner_auth_config(&auth_config)
            .await
            .unwrap();
        let principal = json!({
            "sub":"send-owner",
            "user_id":"send-owner",
            "role":"rw",
            "storage_ns":"conversation",
            "workspace_id":"workspace-a",
            "workspace":"workspace-a",
            "graph_space":"conversation",
            "ns":["conversation"],
            "capabilities":["spawn_process","send_message","read_graph","write_graph","workflow.run.write"]
        });
        let conversation_id = format!("send-conversation-{suffix}");
        let create = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/api/syscall/v1/create_conversation".to_owned(),
                body: json!({
                    "version":"v1",
                    "op":"create_conversation",
                    "args":{"conversation_id":conversation_id,"user_id":"send-owner"}
                })
                .to_string()
                .into_bytes(),
                principal: principal.clone(),
            })
            .await;
        assert_eq!(create.status, StatusCode::OK.as_u16(), "{}", String::from_utf8_lossy(&create.body));

        let make_send = |text: &str| ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/syscall/v1/send_message".to_owned(),
            body: json!({
                "version":"v1",
                "op":"send_message",
                "args":{
                    "conversation_id":conversation_id,
                    "text":text,
                    "workflow_id":"agentic_answering.v2",
                    "idempotency_key":"send-once"
                }
            })
            .to_string()
            .into_bytes(),
            principal: principal.clone(),
        };
        let first = service.execute(make_send("hello from Rust Chroma path")).await;
        assert_eq!(first.status, response_contract["http_status"].as_u64().unwrap() as u16, "{}", String::from_utf8_lossy(&first.body));
        let first_payload: Value = serde_json::from_slice(&first.body).unwrap();
        for (key, value) in response_contract["envelope"].as_object().unwrap() {
            assert_eq!(first_payload[key], *value);
        }
        for field in response_contract["common_result_fields"].as_array().unwrap() {
            assert!(first_payload["result"].get(field.as_str().unwrap()).is_some());
        }
        for (key, value) in response_contract["common_result_values"].as_object().unwrap() {
            assert_eq!(first_payload["result"][key], *value);
        }
        let run_id = first_payload["result"]["run_id"].as_str().unwrap();
        let user_turn_id = first_payload["result"]["turn_node_id"].as_str().unwrap();
        let event_count = service.store.replay_raw_events("conversation", 0, 100).unwrap().len();

        let replay = service.execute(make_send("hello from Rust Chroma path")).await;
        assert_eq!(replay.status, response_contract["http_status"].as_u64().unwrap() as u16);
        let replay_payload: Value = serde_json::from_slice(&replay.body).unwrap();
        for (key, value) in response_contract["envelope"].as_object().unwrap() {
            assert_eq!(replay_payload[key], *value);
        }
        for field in response_contract["common_result_fields"].as_array().unwrap() {
            let field = field.as_str().unwrap();
            assert_eq!(replay_payload["result"][field], first_payload["result"][field]);
        }
        assert_eq!(replay_payload["result"]["run_id"], run_id);
        assert_eq!(replay_payload["result"]["turn_node_id"], user_turn_id);
        assert_eq!(replay_payload["result"]["status"], "queued");
        assert_eq!(replay_payload["result"]["idempotent"], true);
        for (field, expected) in response_contract["rust_replay_additive_result"].as_object().unwrap() {
            let expected = if expected.as_str() == Some("user_turn_node_id") {
                replay_payload["result"]["user_turn_node_id"].clone()
            } else {
                expected.clone()
            };
            assert_eq!(replay_payload["result"][field], expected);
        }
        assert_eq!(service.store.replay_raw_events("conversation", 0, 100).unwrap().len(), event_count);

        let conflict = service.execute(make_send("different payload")).await;
        assert_eq!(conflict.status, response_contract["conflict_status"].as_u64().unwrap() as u16);
        assert_eq!(service.store.replay_raw_events("conversation", 0, 100).unwrap().len(), event_count);
        assert_eq!(
            service
                .store
                .get_server_run(run_id)
                .unwrap()
                .unwrap()
                .user_turn_node_id,
            Some(user_turn_id.to_owned())
        );
        assert!(service.store.replay_raw_events("conversation", 0, 100).unwrap().iter().any(|event| {
            event.entity_id == user_turn_id
                && serde_json::from_str::<Value>(&event.payload_json)
                    .ok()
                    .is_some_and(|payload| payload["summary"] == "hello from Rust Chroma path")
        }));

        drop(service);
        let _ = std::fs::remove_file(database);
        let _ = std::fs::remove_file(auth_database);
    }

    fn assert_checkpoint_syscall_reads_recorded_state_and_checks_scope(
        service: &SqliteRunApplicationService,
        contract: &Value,
    ) {
        service
            .store
            .create_server_run(ServerRunCreate {
                run_id: contract["run_id"].as_str().unwrap().to_owned(),
                conversation_id: "checkpoint-conversation".to_owned(),
                workflow_id: contract["workflow_id"].as_str().unwrap().to_owned(),
                user_id: Some("checkpoint-owner".to_owned()),
                user_turn_node_id: "turn-1".to_owned(),
                status: "running".to_owned(),
            })
            .unwrap();
        service
            .store
            .append_server_run_event(
                contract["run_id"].as_str().unwrap(),
                "workflow.recorded_transition.v1",
                serde_json::to_string(&json!({
                    "transition_id":"transition-1",
                    "reduced":{
                        "kind":"recorded_step_success",
                        "state":{
                            "last_step_seq":0,
                            "workflow_id":contract["workflow_id"],
                            "last_node_id":"node-1",
                            "state":contract["state"],
                            "frontier":{"pending":[]}
                        },
                        "state_update":[],
                        "update":null,
                        "result":null,
                        "errors":[],
                        "server_status":"running"
                    }
                }))
                .unwrap(),
            )
            .unwrap();

        let request = |step_seq: Value, capabilities: Value| ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/syscall/v1/checkpoint".to_owned(),
            body: serde_json::to_vec(&json!({
                "version":contract["version"],
                "op":contract["op"],
                "args":{"run_id":contract["run_id"],"step_seq":step_seq}
            }))
            .unwrap(),
            principal: json!({
                "sub":"checkpoint-owner",
                "role":"ro",
                "ns":"workflow",
                "capabilities":capabilities
            }),
        };

        let denied = service.execute_sync(request(
            contract["step_seq_input"].clone(),
            json!([]),
        ));
        assert_eq!(denied.status, StatusCode::FORBIDDEN.as_u16());

        let found = service.execute_sync(request(
            contract["step_seq_input"].clone(),
            json!(["workflow.run.read"]),
        ));
        assert_eq!(found.status, StatusCode::OK.as_u16());
        let found: Value = serde_json::from_slice(&found.body).unwrap();
        assert_eq!(found["op"], contract["op"]);
        assert_eq!(found["result"], contract["expected_result"]);

        let missing = service.execute_sync(request(
            json!(contract["step_seq_expected"].as_i64().unwrap() + 1),
            json!(["workflow.run.read"]),
        ));
        assert_eq!(missing.status, StatusCode::NOT_FOUND.as_u16());
        let missing: Value = serde_json::from_slice(&missing.body).unwrap();
        assert_eq!(missing["code"], "KOGWISTAR_CHECKPOINT_NOT_FOUND");
    }

    #[test]
    fn sqlite_metadata_checkpoint_syscall_reads_recorded_state_and_checks_scope() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let contract: Value = serde_json::from_str(include_str!(
            "../../../../../../contracts/golden/syscall-checkpoint-persisted-v1.json"
        ))
        .unwrap();
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-syscall-checkpoint-{nonce}.sqlite3"));
        let service = SqliteRunApplicationService::open(&path).unwrap();
        assert_checkpoint_syscall_reads_recorded_state_and_checks_scope(&service, &contract);

        drop(service);
        let _ = std::fs::remove_file(path);
    }

    // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/sqlite_automation_services.rs#syscall_effect; python=kogwistar/server/syscall_api.py#create_syscall_router.dispatch; fixture=contracts/golden/syscall-checkpoint-persisted-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/sqlite_runtime/worker_lifecycle.rs#sqlite_chroma_checkpoint_syscall_reads_recorded_state; python_test=tests/server/test_chat_server_api.py#test_checkpoint_syscall_reads_python_backend
    #[tokio::test]
    #[ignore = "requires KOGWISTAR_TEST_CHROMA_URL"]
    async fn sqlite_chroma_checkpoint_syscall_reads_recorded_state() {
        let endpoint = std::env::var("KOGWISTAR_TEST_CHROMA_URL")
            .expect("KOGWISTAR_TEST_CHROMA_URL must point to live Chroma");
        let suffix = uuid::Uuid::new_v4().simple().to_string();
        let path = std::env::temp_dir().join(format!("kogwistar-checkpoint-{suffix}.sqlite3"));
        let service = open_sqlite_application_service(
            &path,
            Some(SqliteChromaGraphConfig {
                endpoint,
                metric: DistanceMetric::Cosine,
                node_collection: format!("checkpoint-n-{suffix}"),
                edge_collection: format!("checkpoint-e-{suffix}"),
            }),
        )
        .await
        .unwrap();
        let contract: Value = serde_json::from_str(include_str!(
            "../../../../../../contracts/golden/syscall-checkpoint-persisted-v1.json"
        ))
        .unwrap();
        assert_checkpoint_syscall_reads_recorded_state_and_checks_scope(&service, &contract);
        drop(service);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    #[ignore = "requires KOGWISTAR_TEST_PG_DSN; run explicitly with --include-ignored"]
    async fn postgres_checkpoint_syscall_reads_recorded_state_and_checks_scope() {
        use kogwistar_store::ServerRunCreate;

        let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
            .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
        let schema = format!("rust_checkpoint_syscall_{}", uuid::Uuid::new_v4().simple());
        let service = PostgresRunApplicationService::from_dsn(&dsn, &schema).unwrap();
        service.ensure_schema().await.unwrap();
        let contract: Value = serde_json::from_str(include_str!(
            "../../../../../../contracts/golden/syscall-checkpoint-persisted-v1.json"
        ))
        .unwrap();
        let run_id = contract["run_id"].as_str().unwrap();
        service
            .store
            .create_server_run(ServerRunCreate {
                run_id: run_id.to_owned(),
                conversation_id: "checkpoint-conversation".to_owned(),
                workflow_id: contract["workflow_id"].as_str().unwrap().to_owned(),
                user_id: Some("checkpoint-owner".to_owned()),
                user_turn_node_id: "turn-1".to_owned(),
                status: "running".to_owned(),
            })
            .await
            .unwrap();
        service
            .store
            .append_server_run_event(
                run_id,
                "workflow.recorded_transition.v1",
                serde_json::to_string(&json!({
                    "transition_id":"transition-1",
                    "reduced":{
                        "kind":"recorded_step_success",
                        "state":{
                            "last_step_seq":0,
                            "workflow_id":contract["workflow_id"],
                            "last_node_id":"node-1",
                            "state":contract["state"],
                            "frontier":{"pending":[]}
                        },
                        "state_update":[],
                        "update":null,
                        "result":null,
                        "errors":[],
                        "server_status":"running"
                    }
                }))
                .unwrap(),
            )
            .await
            .unwrap();

        let request = |capabilities: Value, step_seq: Value| ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/syscall/v1/checkpoint".to_owned(),
            body: serde_json::to_vec(&json!({
                "version":contract["version"],
                "op":contract["op"],
                "args":{"run_id":run_id,"step_seq":step_seq}
            }))
            .unwrap(),
            principal: json!({
                "sub":"checkpoint-owner",
                "role":"ro",
                "ns":"workflow",
                "capabilities":capabilities
            }),
        };

        let denied = service
            .execute(request(json!([]), contract["step_seq_input"].clone()))
            .await;
        assert_eq!(denied.status, StatusCode::FORBIDDEN.as_u16());
        let found = service
            .execute(request(
                json!(["workflow.run.read"]),
                contract["step_seq_input"].clone(),
            ))
            .await;
        assert_eq!(found.status, StatusCode::OK.as_u16());
        let found: Value = serde_json::from_slice(&found.body).unwrap();
        assert_eq!(found["result"], contract["expected_result"]);
        let missing = service
            .execute(request(
                json!(["workflow.run.read"]),
                json!(contract["step_seq_expected"].as_i64().unwrap() + 1),
            ))
            .await;
        assert_eq!(missing.status, StatusCode::NOT_FOUND.as_u16());
        assert_eq!(
            serde_json::from_slice::<Value>(&missing.body).unwrap()["code"],
            "KOGWISTAR_CHECKPOINT_NOT_FOUND"
        );
    }

    #[tokio::test]
    #[ignore = "requires KOGWISTAR_TEST_PG_DSN; run explicitly with --include-ignored"]
    async fn postgres_project_view_syscall_matches_resource_contract_and_authority() {
        let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
            .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
        let schema = format!("rust_project_view_syscall_{}", uuid::Uuid::new_v4().simple());
        let service = PostgresRunApplicationService::from_dsn(&dsn, &schema).unwrap();
        service.ensure_schema().await.unwrap();
        service.ensure_graph_type_schemas().await.unwrap();
        let contract: Value = serde_json::from_str(include_str!(
            "../../../../../../contracts/golden/workflow-resources-http-v1.json"
        ))
        .unwrap();
        let request = |capabilities: Value| ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/syscall/v1/project_view".to_owned(),
            body: serde_json::to_vec(&json!({
                "version":"v1",
                "op":"project_view",
                "args":{}
            }))
            .unwrap(),
            principal: json!({
                "sub":"project-view-reader",
                "role":"ro",
                "ns":"workflow",
                "capabilities":capabilities
            }),
        };

        let denied = service.execute(request(json!([]))).await;
        assert_eq!(denied.status, StatusCode::FORBIDDEN.as_u16());
        let allowed = service
            .execute(request(json!(contract["required_capabilities"])))
            .await;
        assert_eq!(allowed.status, StatusCode::OK.as_u16());
        let body: Value = serde_json::from_slice(&allowed.body).unwrap();
        let result = &body["result"];
        for field in contract["required_fields"].as_array().unwrap() {
            let field = field.as_str().unwrap();
            assert!(result.get(field).is_some(), "missing {field}");
        }
        for (key, fields) in contract["required_object_fields"].as_object().unwrap() {
            for field in fields.as_array().unwrap() {
                let field = field.as_str().unwrap();
                assert!(result[key].get(field).is_some(), "missing {key}.{field}");
            }
        }
    }

    // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/sqlite_automation_services.rs#syscall_effect; python=kogwistar/server/syscall_api.py#create_syscall_router.dispatch; fixture=contracts/golden/workflow-resources-http-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/sqlite_runtime/worker_lifecycle.rs#sqlite_chroma_project_view_matches_resource_fixture_and_authority; python_test=tests/server/test_chat_server_api.py#test_project_view_syscall_uses_python_backend
    #[tokio::test]
    #[ignore = "requires KOGWISTAR_TEST_CHROMA_URL"]
    async fn sqlite_chroma_project_view_matches_resource_fixture_and_authority() {
        let endpoint = std::env::var("KOGWISTAR_TEST_CHROMA_URL")
            .expect("KOGWISTAR_TEST_CHROMA_URL must point to live Chroma");
        let suffix = uuid::Uuid::new_v4().simple().to_string();
        let path = std::env::temp_dir().join(format!("kogwistar-project-view-{suffix}.sqlite3"));
        let service = open_sqlite_application_service(
            &path,
            Some(SqliteChromaGraphConfig {
                endpoint,
                metric: DistanceMetric::Cosine,
                node_collection: format!("project-view-n-{suffix}"),
                edge_collection: format!("project-view-e-{suffix}"),
            }),
        )
        .await
        .unwrap();
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../../../../contracts/golden/workflow-resources-http-v1.json"
        ))
        .unwrap();
        let request = |capabilities: &[&str]| ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/syscall/v1/project_view".to_owned(),
            body: json!({"version":"v1","op":"project_view","args":{}})
                .to_string()
                .into_bytes(),
            principal: json!({
                "sub":"resource-viewer",
                "role":"ro",
                "ns":["workflow"],
                "capabilities":capabilities
            }),
        };
        for denied_capabilities in [&["project_view"][..], &["read_security_scope"][..]] {
            let denied = service.execute_sync(request(denied_capabilities));
            assert_eq!(denied.status, StatusCode::FORBIDDEN.as_u16());
        }
        let response = service.execute_sync(request(&["project_view", "read_security_scope"]));
        assert_eq!(response.status, StatusCode::OK.as_u16());
        let payload: Value = serde_json::from_slice(&response.body).unwrap();
        let result = &payload["result"];
        for field in fixture["required_fields"].as_array().unwrap() {
            assert!(result.get(field.as_str().unwrap()).is_some());
        }
        for (object, required_fields) in fixture["required_object_fields"].as_object().unwrap() {
            for field in required_fields.as_array().unwrap() {
                assert!(result[object].get(field.as_str().unwrap()).is_some());
            }
        }
        drop(service);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn sqlite_metadata_resume_syscall_resumes_persisted_suspended_token() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-syscall-resume-{nonce}.sqlite3"));
        let service = SqliteRunApplicationService::open(&path).unwrap();
        sqlite_resume_syscall_roundtrip(service, path).await;
    }

    #[tokio::test]
    #[ignore = "requires KOGWISTAR_TEST_CHROMA_URL"]
    async fn sqlite_chroma_resume_syscall_resumes_persisted_suspended_token() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let endpoint = std::env::var("KOGWISTAR_TEST_CHROMA_URL")
            .expect("KOGWISTAR_TEST_CHROMA_URL must point to live Chroma");
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let suffix = uuid::Uuid::new_v4().simple().to_string();
        let config = |kind: &str| kogwistar_store_chroma::ChromaStoreConfig {
            endpoint: endpoint.clone(),
            collection: format!("resume-{kind}-{suffix}"),
            ..Default::default()
        };
        let graph = kogwistar_store_chroma::ChromaGraphStore::connect_entity_collections(
            config("nodes"),
            config("edges"),
        )
        .await
        .unwrap();
        let path = std::env::temp_dir().join(format!("kogwistar-syscall-resume-chroma-{nonce}.sqlite3"));
        let service = SqliteRunApplicationService::open(&path)
            .unwrap()
            .with_chroma_graph(graph);
        sqlite_resume_syscall_roundtrip(service, path).await;
    }

    // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/sqlite_service.rs#SqliteRunApplicationService; python=kogwistar/server/chat_service_run_execution.py#_default_resume_runner; fixture=contracts/golden/workflow-run-resume-result-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/sqlite_runtime/worker_lifecycle.rs#sqlite_metadata_resume_syscall_resumes_persisted_suspended_token; python_test=tests/server/test_chat_server_api.py#test_python_resume_syscall_continues_persisted_workflow
    async fn sqlite_resume_syscall_roundtrip(service: SqliteRunApplicationService, path: std::path::PathBuf) {
        use kogwistar_store::RUNTIME_CURRENT_STATE_NAMESPACE;

        let resume_contract: Value = serde_json::from_str(include_str!(
            "../../../../../../contracts/golden/workflow-run-resume-result-v1.json"
        ))
        .unwrap();
        let run_id = "sqlite-resume-syscall-run";
        let workflow_id = "sqlite-resume-syscall-workflow";
        let conversation_id = "sqlite-resume-syscall-conversation";
        let principal = json!({
            "sub":"sqlite-resume-owner",
            "role":"rw",
            "ns":["workflow"],
            "capabilities":["spawn_process", "workflow.run.write"]
        });
        let submitted = service.execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/workflow/runs".to_owned(),
            body: serde_json::to_vec(&json!({
                "run_id":run_id,
                "workflow_id":workflow_id,
                "conversation_id":conversation_id,
                "turn_node_id":"sqlite-resume-turn"
            }))
            .unwrap(),
            principal: principal.clone(),
        }).await;
        assert_eq!(submitted.status, StatusCode::ACCEPTED.as_u16());

        let claimed = service.execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/internal/runtime/claim".to_owned(),
            body: serde_json::to_vec(&json!({
                "claimed_by":"sqlite-test-suspender",
                "limit":1,
                "lease_seconds":60
            }))
            .unwrap(),
            principal: principal.clone(),
        }).await;
        assert_eq!(claimed.status, StatusCode::OK.as_u16());
        let claimed: Value = serde_json::from_slice(&claimed.body).unwrap();
        let work = &claimed["work"][0];
        let suspended = service.execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/internal/runtime/results".to_owned(),
            body: serde_json::to_vec(&json!({
                "handoff": {
                    "message_id":work["message_id"],
                    "claimed_by":work["claimed_by"],
                    "run_id":work["run_id"],
                    "step_id":work["step_id"],
                    "correlation_id":work["correlation_id"]
                },
                "effect": {
                    "contract_version":1,
                    "effect_id":"sqlite-suspend-for-syscall-resume",
                    "status":"suspended",
                    "state_update":[["u", {"before_suspend":true}]],
                    "successors":[],
                    "wait_reason":"approval",
                    "resume_payload":{"question":"continue?"}
                }
            }))
            .unwrap(),
            principal: principal.clone(),
        }).await;
        assert_eq!(
            suspended.status,
            StatusCode::OK.as_u16(),
            "{}",
            String::from_utf8_lossy(&suspended.body)
        );
        let projection = service
            .store
            .get_named_projection(RUNTIME_CURRENT_STATE_NAMESPACE, run_id)
            .unwrap()
            .unwrap();
        let state: RecordedRuntimeState =
            serde_json::from_value(Value::Object(projection.payload)).unwrap();
        let (node_id, _, token_id, _) = state.frontier.suspended[0].clone();

        let resume_service = service.clone();
        let resume_future = tokio::spawn(async move {
            resume_service
                .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/syscall/v1/resume".to_owned(),
            body: serde_json::to_vec(&json!({
                "version":"v1",
                "op":"resume",
                "args":{
                    "run_id":run_id,
                    "workflow_id":workflow_id,
                    "conversation_id":conversation_id,
                    "turn_node_id":"sqlite-resume-turn",
                    "suspended_node_id":node_id,
                    "suspended_token_id":token_id,
                    "client_result":{"approved":true}
                }
            }))
            .unwrap(),
                    principal,
                })
                .await
        });
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let claimed = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/internal/runtime/claim".to_owned(),
                body: serde_json::to_vec(&json!({
                    "claimed_by":"sqlite-test-finisher",
                    "limit":1,
                    "lease_seconds":60
                }))
                .unwrap(),
                principal: json!({
                    "sub":"sqlite-resume-owner",
                    "role":"rw",
                    "ns":["workflow"],
                    "capabilities":["workflow.run.write"]
                }),
            })
            .await;
        assert_eq!(claimed.status, StatusCode::OK.as_u16());
        let claimed: Value = serde_json::from_slice(&claimed.body).unwrap();
        let work = &claimed["work"][0];
        assert_eq!(work["run_id"], run_id);
        let finished = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/internal/runtime/results".to_owned(),
                body: serde_json::to_vec(&json!({
                    "handoff": {
                        "message_id":work["message_id"],
                        "claimed_by":work["claimed_by"],
                        "run_id":work["run_id"],
                        "step_id":work["step_id"],
                        "correlation_id":work["correlation_id"]
                    },
                    "effect": {
                        "contract_version":1,
                        "effect_id":"sqlite-finish-after-syscall-resume",
                        "state_update":[["u", {"after_resume":true}]],
                        "successors":[],
                        "result":{"done":true}
                    }
                }))
                .unwrap(),
                principal: json!({"role":"rw", "ns":["workflow"]}),
            })
            .await;
        assert_eq!(finished.status, StatusCode::OK.as_u16());
        let resumed = resume_future.await.unwrap();
        assert_eq!(
            resumed.status,
            StatusCode::OK.as_u16(),
            "{}",
            String::from_utf8_lossy(&resumed.body)
        );
        let envelope: Value = serde_json::from_slice(&resumed.body).unwrap();
        assert_eq!(envelope["version"], "v1");
        assert_eq!(envelope["op"], "resume");
        assert_eq!(envelope["status"], "ok");
        for field in resume_contract["required_fields"].as_array().unwrap() {
            let field = field.as_str().unwrap();
            assert!(envelope["result"].get(field).is_some(), "missing {field}");
        }
        assert!(resume_contract["completed_statuses"]
            .as_array()
            .unwrap()
            .contains(&envelope["result"]["workflow_status"]));
        assert_eq!(envelope["result"]["workflow_status"], "succeeded");
        assert_eq!(envelope["result"]["final_state"]["after_resume"], true);
        assert_eq!(envelope["result"]["budget"], json!({}));

        drop(service);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn sqlite_metadata_conversation_syscalls_fail_closed() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-syscall-unavailable-{nonce}.sqlite3"));
        let service = SqliteRunApplicationService::open(&path).unwrap();
        let cases = [
            ("create_conversation", "conversation", json!({}), json!([])),
            ("invoke_tool", "conversation", json!({}), json!([])),
            ("mount_memory", "conversation", json!({"conversation_id":"c1"}), json!(["project_view"])),
            ("receive_message", "conversation", json!({"conversation_id":"c1"}), json!(["read_graph"])),
            (
                "send_message",
                "conversation",
                json!({"conversation_id":"c1","content":"hello"}),
                json!([]),
            ),
        ];

        for (operation, namespace, args, capabilities) in cases {
            let response = service.execute_sync(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: format!("/api/syscall/v1/{operation}"),
                body: serde_json::to_vec(&json!({
                    "version":"v1",
                    "op":operation,
                    "args":args,
                }))
                .unwrap(),
                principal: json!({
                    "sub":"metadata-only-test",
                    "role":"rw",
                    "ns":namespace,
                    "capabilities":capabilities,
                }),
            });
            assert_eq!(response.status, StatusCode::NOT_IMPLEMENTED.as_u16(), "{operation}");
            let body: Value = serde_json::from_slice(&response.body).unwrap();
            assert_eq!(body["code"], "KOGWISTAR_API_CAPABILITY_NOT_CUT_OVER", "{operation}");
            assert_eq!(body["path"], format!("/api/syscall/v1/{operation}"), "{operation}");
        }

        assert!(service.store.list_server_runs(None, None, None, 10).unwrap().is_empty());
        drop(service);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn sqlite_request_approval_mutations_require_scope_and_return_snapshot() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-syscall-approval-{nonce}.sqlite3"));
        let service = SqliteRunApplicationService::open(&path).unwrap();
        let request = |action: &str, capabilities: Value| ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/syscall/v1/request_approval".to_owned(),
            body: serde_json::to_vec(&json!({
                "version":"v1",
                "op":"request_approval",
                "args":{
                    "action":action,
                    "approval_action":"request_approval",
                    "subject":"sqlite-approval-target",
                    "capability":"workflow.run.read"
                }
            }))
            .unwrap(),
            principal: json!({
                "sub":"sqlite-approval-owner",
                "role":"rw",
                "ns":"workflow",
                "capabilities":capabilities,
            }),
        };

        for action in ["grant", "revoke"] {
            let denied = service.execute_sync(request(
                action,
                json!(["approve_action", "project_view"]),
            ));
            assert_eq!(denied.status, StatusCode::FORBIDDEN.as_u16(), "{action}");
        }
        let owner = json!("sqlite-approval-owner");
        let state = service
            .capabilities
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let before = capability_snapshot(&owner, &state);
        assert!(!before.to_string().contains("sqlite-approval-target"));
        drop(state);

        let grant = service.execute_sync(request(
            "grant",
            json!(["approve_action", "project_view", "read_security_scope"]),
        ));
        assert_eq!(grant.status, StatusCode::OK.as_u16());
        let grant: Value = serde_json::from_slice(&grant.body).unwrap();
        assert_eq!(grant["result"]["status"], "approved");
        assert!(grant["result"]["approval"]["approvals"]
            .as_array()
            .unwrap()
            .iter()
            .any(|approval| {
                approval["subject"] == "sqlite-approval-target"
                    && approval["action"] == "request_approval"
                    && approval["capabilities"] == json!(["workflow.run.read"])
            }));

        let revoke = service.execute_sync(request(
            "revoke",
            json!(["approve_action", "project_view", "read_security_scope"]),
        ));
        assert_eq!(revoke.status, StatusCode::OK.as_u16());
        let revoke: Value = serde_json::from_slice(&revoke.body).unwrap();
        assert_eq!(revoke["result"]["status"], "revoked");
        assert!(revoke["result"]["revocation"]["revoked"]
            .as_array()
            .unwrap()
            .iter()
            .any(|entry| {
                entry["subject"] == "sqlite-approval-target"
                    && entry["capabilities"] == json!(["workflow.run.read"])
            }));

        drop(service);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn sqlite_worker_suspend_resume_is_durable_and_scheduler_owned() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use std::time::{SystemTime, UNIX_EPOCH};
        use tower::ServiceExt;

        let resume_contract_fixture: Value = serde_json::from_str(include_str!(
            "../../../../../../contracts/golden/run-resume-contract-v1.json"
        ))
        .unwrap();
        let resume_result_fixture: Value = serde_json::from_str(include_str!(
            "../../../../../../contracts/golden/workflow-run-resume-result-v1.json"
        ))
        .unwrap();
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-resume-{nonce}.sqlite3"));
        let reopen = || {
            router_with_application(
                ApiState {
                    health: snapshot(),
                    required_roles: vec!["reader".to_owned()],
                    implementation: ImplementationSnapshot::default(),
                    auth: AuthConfig::default(),
                },
                Arc::new(SqliteRunApplicationService::open(&path).unwrap()),
            )
        };
        let mut app = reopen();
        let call = |app: Router, method: &'static str, uri: String, body: Value| async move {
            let response = app
                .oneshot(
                    Request::builder()
                        .method(method)
                        .uri(uri)
                        .header("content-type", "application/json")
                        .header("x-kogwistar-roles", "reader,rw")
                        .body(Body::from(if body.is_null() {
                            Vec::new()
                        } else {
                            serde_json::to_vec(&body).unwrap()
                        }))
                        .unwrap(),
                )
                .await
                .unwrap();
            let status = response.status();
            let bytes = to_bytes(response.into_body(), 32768).await.unwrap();
            (status, serde_json::from_slice::<Value>(&bytes).unwrap())
        };
        let (status, submitted) = call(
            app.clone(),
            "POST",
            "/api/workflow/runs".to_owned(),
            json!({"workflow_id":"resume-wf","conversation_id":"resume-conv","turn_node_id":"turn-original"}),
        )
        .await;
        assert_eq!(status, StatusCode::ACCEPTED);
        let run_id = submitted["run_id"].as_str().unwrap().to_owned();
        drop(app);
        app = reopen();
        let (_, claimed) = call(
            app.clone(),
            "POST",
            "/internal/runtime/claim".to_owned(),
            json!({"claimed_by":"suspender","limit":1,"lease_seconds":60}),
        )
        .await;
        let work = &claimed["work"][0];
        drop(app);
        app = reopen();
        let suspend = json!({
            "handoff": {
                "message_id": work["message_id"],
                "claimed_by": work["claimed_by"],
                "run_id": work["run_id"],
                "step_id": work["step_id"],
                "correlation_id": work["correlation_id"],
            },
            "effect": {
                "contract_version": 1,
                "effect_id": format!("suspend-{run_id}"),
                "status": "suspended",
                "state_update": [["u", {"before_suspend": true}]],
                "successors": [],
                "wait_reason": "approval",
                "resume_payload": {"question": "continue?"},
            }
        });
        let (status, _) = call(
            app.clone(),
            "POST",
            "/internal/runtime/results".to_owned(),
            suspend,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        drop(app);
        app = reopen();
        let (status, contract) = call(
            app.clone(),
            "GET",
            format!("/api/workflow/runs/{run_id}/resume-contract"),
            Value::Null,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        for field in resume_contract_fixture["required_fields"]
            .as_array()
            .unwrap()
        {
            let field = field.as_str().unwrap();
            assert!(contract.get(field).is_some(), "missing {field}");
        }
        assert_eq!(contract["compatible"], true);
        assert_eq!(contract["checkpoint_schema_version"], 1);
        assert_eq!(
            contract["supported_wait_reasons"],
            resume_contract_fixture["supported_wait_reasons"]
        );
        assert_eq!(contract["resume_options"][0]["suspended_node_id"], "start");
        assert_eq!(contract["wait_reason"], "approval");
        assert_eq!(contract["suspended"][0][0], "start");
        assert_eq!(contract["runtime_routes"], json!([]));
        assert_eq!(contract["node_ops"], json!({}));
        let token_id = contract["suspended"][0][2].as_str().unwrap().to_owned();
        let resume_body = json!({
            "suspended_node_id": "start",
            "suspended_token_id": token_id,
            "client_result": {
                "status": "success",
                "state_update": [["u", {"approved": true}]],
                "successors": [],
                "route_next": [],
                "result": {"workflow_status": "succeeded"}
            },
            "workflow_id": "resume-wf",
            "conversation_id": "resume-conv",
            "turn_node_id": "unused-compatible-field",
        });
        let denied_resume = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!(
                        "/api/workflow/runs/{run_id}/resume?wait_for_completion=true"
                    ))
                    .header("content-type", "application/json")
                    .header("x-kogwistar-roles", "reader")
                    .body(Body::from(serde_json::to_vec(&resume_body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(denied_resume.status(), StatusCode::FORBIDDEN);
        let resume_app = app.clone();
        let resume_uri = format!("/api/runs/{run_id}/resume");
        let resume_future = tokio::spawn(async move {
            call(resume_app, "POST", resume_uri, resume_body).await
        });
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        let (_, claimed) = call(
            app.clone(),
            "POST",
            "/internal/runtime/claim".to_owned(),
            json!({"claimed_by":"finisher","limit":1,"lease_seconds":60}),
        )
        .await;
        let work = &claimed["work"][0];
        assert_eq!(work["step_id"], "start");
        assert_eq!(work["payload"]["turn_node_id"], "turn-original");
        assert_eq!(work["payload"]["resume_effect"]["status"], "success");
        assert_eq!(
            work["payload"]["resume_effect"]["state_update"][0][1]["approved"],
            true
        );
        let finish_result = json!({
            "handoff": {
                "message_id": work["message_id"],
                "claimed_by": work["claimed_by"],
                "run_id": work["run_id"],
                "step_id": work["step_id"],
                "correlation_id": work["correlation_id"],
            },
            "effect": {
                "contract_version": 1,
                "effect_id": format!("finish-{run_id}"),
                "state_update": [["u", {"after_resume": true}]],
                "successors": [],
                "result": {"done": true},
            }
        });
        let (status, first_finish) = call(
            app.clone(),
            "POST",
            "/internal/runtime/results".to_owned(),
            finish_result.clone(),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let (retry_status, retry_finish) = call(
            app.clone(),
            "POST",
            "/internal/runtime/results".to_owned(),
            finish_result,
        )
        .await;
        assert_eq!(retry_status, StatusCode::OK);
        assert_eq!(retry_finish["event_seq"], first_finish["event_seq"]);
        assert_eq!(retry_finish["idempotent"], true);
        let (resume_status, resumed) = resume_future.await.unwrap();
        assert_eq!(resume_status, StatusCode::OK);
        let result_fields = resume_result_fixture["required_fields"]
            .as_array()
            .unwrap()
            .iter()
            .map(|field| field.as_str().unwrap())
            .collect::<std::collections::BTreeSet<_>>();
        let actual_fields = resumed
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(actual_fields, result_fields);
        assert!(resume_result_fixture["completed_statuses"]
            .as_array()
            .unwrap()
            .contains(&resumed["workflow_status"]));
        assert_eq!(resumed["workflow_status"], "succeeded");
        assert_eq!(resumed["final_state"]["before_suspend"], true);
        assert_eq!(resumed["final_state"]["after_resume"], true);
        assert_eq!(resumed["budget"], json!({}));
        drop(app);
        app = reopen();
        let (status, run) = call(
            app,
            "GET",
            format!("/api/workflow/runs/{run_id}"),
            Value::Null,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(run["status"], "succeeded");
        assert_eq!(run["result"]["done"], true);
        std::fs::remove_file(path).unwrap();
    }
