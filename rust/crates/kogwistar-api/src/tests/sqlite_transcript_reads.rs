    #[tokio::test]
    async fn sqlite_chroma_receive_message_returns_scoped_acl_filtered_transcript() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use jsonwebtoken::{EncodingKey, Header, encode};
        use sha2::Digest;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tower::ServiceExt;

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let (stop_chroma, mut stop_chroma_rx) = tokio::sync::oneshot::channel();
        let fake = tokio::spawn(async move {
            let mut requests = Vec::new();
            loop {
                let accepted = tokio::select! {
                    _ = &mut stop_chroma_rx => break,
                    accepted = listener.accept() => accepted.unwrap(),
                };
                let (mut stream, _) = accepted;
                let mut bytes = Vec::new();
                let mut chunk = [0_u8; 4096];
                let (split, length) = loop {
                    let count = stream.read(&mut chunk).await.unwrap();
                    assert!(count > 0);
                    bytes.extend_from_slice(&chunk[..count]);
                    if let Some(split) = bytes.windows(4).position(|part| part == b"\r\n\r\n") {
                        let head = String::from_utf8_lossy(&bytes[..split]);
                        let length = head
                            .lines()
                            .find_map(|line| {
                                let (name, value) = line.split_once(':')?;
                                name.eq_ignore_ascii_case("content-length")
                                    .then(|| value.trim().parse::<usize>().ok())
                                    .flatten()
                            })
                            .unwrap_or(0);
                        if bytes.len() >= split + 4 + length {
                            break (split, length);
                        }
                    }
                };
                let head = String::from_utf8_lossy(&bytes[..split]).to_string();
                let body: Value = serde_json::from_slice(&bytes[split + 4..split + 4 + length])
                    .unwrap();
                let payload = if head.lines().next().unwrap().contains("/collections HTTP") {
                    json!({
                        "id": format!("collection-{}", body["name"].as_str().unwrap()),
                        "metadata": {"hnsw:space": "cosine"}
                    })
                } else if head.lines().next().unwrap().contains("collection-nodes/get") {
                    let nodes = [
                        ("start", json!({"entity_type":"conversation_start","conversation_id":"c1","user_id":"reader-1","status":"active"})),
                        ("turn-2", json!({"entity_type":"conversation_turn","conversation_id":"c1","turn_index":2,"role":"assistant","summary":"second"})),
                        ("turn-1", json!({"entity_type":"conversation_turn","conversation_id":"c1","turn_index":1,"role":"user","summary":"first"})),
                        ("turn-private", json!({"entity_type":"conversation_turn","conversation_id":"c1","turn_index":3,"role":"assistant","summary":"private"})),
                        ("turn-other-workspace", json!({"entity_type":"conversation_turn","conversation_id":"c1","turn_index":4,"role":"assistant","summary":"other workspace","workspace_id":"workspace-b"})),
                        ("turn-other-graph-space", json!({"entity_type":"conversation_turn","conversation_id":"c1","turn_index":5,"role":"assistant","summary":"other graph space","graph_space":"knowledge"})),
                        ("snapshot", json!({"entity_type":"context_snapshot","conversation_id":"c1","run_id":"run-a","stage":"answer","snapshot_seq":7})),
                        ("acl-start", json!({"entity_type":"acl_record","acl_truth_graph":"conversation","acl_target_grain":"node","acl_target_entity_id":"start","acl_version":1,"acl_mode":"public"})),
                        ("acl-turn-1", json!({"entity_type":"acl_record","acl_truth_graph":"conversation","acl_target_grain":"node","acl_target_entity_id":"turn-1","acl_version":1,"acl_mode":"public"})),
                        ("acl-turn-2", json!({"entity_type":"acl_record","acl_truth_graph":"conversation","acl_target_grain":"node","acl_target_entity_id":"turn-2","acl_version":1,"acl_mode":"public"})),
                        ("acl-turn-private", json!({"entity_type":"acl_record","acl_truth_graph":"conversation","acl_target_grain":"node","acl_target_entity_id":"turn-private","acl_version":1,"acl_mode":"private","acl_owner_id":"someone-else"})),
                        ("acl-turn-other-workspace", json!({"entity_type":"acl_record","acl_truth_graph":"conversation","acl_target_grain":"node","acl_target_entity_id":"turn-other-workspace","acl_version":1,"acl_mode":"public","workspace_id":"workspace-b"})),
                        ("acl-turn-other-graph-space", json!({"entity_type":"acl_record","acl_truth_graph":"conversation","acl_target_grain":"node","acl_target_entity_id":"turn-other-graph-space","acl_version":1,"acl_mode":"public","graph_space":"knowledge"})),
                        ("acl-snapshot", json!({"entity_type":"acl_record","acl_truth_graph":"conversation","acl_target_grain":"node","acl_target_entity_id":"snapshot","acl_version":1,"acl_mode":"public"})),
                        ("foreign-start", json!({"entity_type":"conversation_start","conversation_id":"c-foreign","user_id":"other-user","status":"active"})),
                        ("acl-foreign-start", json!({"entity_type":"acl_record","acl_truth_graph":"conversation","acl_target_grain":"node","acl_target_entity_id":"foreign-start","acl_version":1,"acl_mode":"public"})),
                    ];
                    let metadatas = nodes
                        .iter()
                        .map(|(_, metadata)| {
                            let mut original = metadata.as_object().unwrap().clone();
                            let workspace_id = original
                                .entry("workspace_id".to_owned())
                                .or_insert_with(|| json!("workspace-a"))
                                .clone();
                            let graph_space = original
                                .entry("graph_space".to_owned())
                                .or_insert_with(|| json!("conversation"))
                                .clone();
                            json!({
                                "_kogwistar_namespace":"scope-a",
                                "workspace_id":workspace_id,
                                "graph_space":graph_space,
                                "_kogwistar_original_metadata":Value::Object(original).to_string()
                            })
                        })
                        .collect::<Vec<_>>();
                    json!({
                        "ids": nodes.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
                        "documents": nodes.iter().map(|(id, _)| {
                            if *id == "snapshot" {
                                r#"{"metadata":{"summary":"stored summary"},"properties":{"references":["evidence-1"]}}"#.to_owned()
                            } else {
                                "{}".to_owned()
                            }
                        }).collect::<Vec<_>>(),
                        "metadatas": metadatas,
                        "embeddings": nodes.iter().map(|_| Value::Null).collect::<Vec<_>>()
                    })
                } else {
                    json!({"ids": [], "documents": [], "metadatas": [], "embeddings": []})
                };
                let response_body = payload.to_string();
                let response = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                    response_body.len(),
                    response_body
                );
                stream.write_all(response.as_bytes()).await.unwrap();
                requests.push((head, body));
            }
            requests
        });
        let path = std::env::temp_dir().join(format!(
            "kogwistar-sqlite-chroma-transcript-{}.sqlite3",
            uuid::Uuid::new_v4()
        ));
        let service = open_sqlite_application_service(
            &path,
            Some(SqliteChromaGraphConfig {
                endpoint: endpoint.clone(),
                metric: DistanceMetric::Cosine,
                node_collection: "nodes".to_owned(),
                edge_collection: "edges".to_owned(),
            }),
        )
        .await
        .unwrap();
        // ACL decisions are canonical SQLite events; Chroma contains only the graph projection.
        for (target, mode, owner) in [
            ("start", "public", None),
            ("turn-1", "public", None),
            ("turn-2", "public", None),
            ("turn-private", "private", Some("someone-else")),
            ("snapshot", "public", None),
            ("foreign-start", "public", None),
        ] {
            let acl_id = format!("acl-{target}");
            let mut metadata = json!({
                "entity_type":"acl_record",
                "acl_truth_graph":"conversation",
                "acl_target_entity_id":target,
                "acl_target_grain":"node",
                "acl_version":1,
                "acl_mode":mode,
                "workspace_id":"workspace-a",
                "graph_space":"conversation"
            });
            if let Some(owner) = owner {
                metadata["owner_id"] = json!(owner);
            }
            service.store.append_raw_entity_event(
                "scope-a",
                SqliteNewRawEntityEvent {
                    event_id: format!("canonical-{acl_id}"),
                    entity_kind: "node".to_owned(),
                    entity_id: acl_id.clone(),
                    op: "ADD".to_owned(),
                    payload_json: json!({
                        "id":acl_id,
                        "document":null,
                        "metadata":metadata
                    })
                    .to_string(),
                },
            ).unwrap();
        }
        let request = |operation: &str, args: Value, principal: Value| ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: format!("/api/syscall/v1/{operation}"),
            body: json!({"version":"v1","args":args}).to_string().into_bytes(),
            principal,
        };
        let read_request = |path: &str, principal: Value| ApiEffectRequest {
            contract_version: 1,
            method: "GET".to_owned(),
            path_and_query: path.to_owned(),
            body: Vec::new(),
            principal,
        };
        let principal = json!({
            "sub":"reader-1", "storage_ns":"scope-a", "workspace_id":"workspace-a",
            "graph_space":"conversation", "ns":["conversation"], "capabilities":["read_graph"]
        });
        let denied = service
            .execute(request(
                "receive_message",
                json!({"conversation_id":"c1"}),
                json!({"ns":["conversation"],"capabilities":[]}),
            ))
            .await;
        assert_eq!(denied.status, StatusCode::FORBIDDEN.as_u16());
        let wrong_namespace = service
            .execute(request(
                "receive_message",
                json!({"conversation_id":"c1"}),
                json!({
                    "sub":"reader-1",
                    "storage_ns":"scope-a",
                    "workspace_id":"workspace-a",
                    "graph_space":"conversation",
                    "ns":["docs"],
                    "capabilities":["read_graph"]
                }),
            ))
            .await;
        assert_eq!(wrong_namespace.status, StatusCode::FORBIDDEN.as_u16());
        let missing_id = service
            .execute(request("receive_message", json!({}), principal.clone()))
            .await;
        let error_contract: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/syscall-receive-message-missing-id-v1.json"
        ))
        .unwrap();
        assert_eq!(missing_id.status, error_contract["http_status"]);
        assert_eq!(
            serde_json::from_slice::<Value>(&missing_id.body).unwrap(),
            error_contract["body"]
        );

        let response = service
            .execute(request("receive_message", json!({"conversation_id":"c1"}), principal.clone()))
            .await;
        assert_eq!(response.status, StatusCode::OK.as_u16());
        let response: Value = serde_json::from_slice(&response.body).unwrap();
        assert_eq!(response["op"], "receive_message");
        let receive_contract: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/receive-message-result-v1.json"
        ))
        .unwrap();
        assert_eq!(response["result"], receive_contract);
        assert_eq!(response["result"]["conversation_id"], "c1");
        assert_eq!(
            response["result"]["messages"].as_array().map(Vec::len),
            Some(2),
            "unexpected transcript payload: {response}"
        );
        assert_eq!(response["result"]["messages"][0]["node_id"], "turn-1");
        assert_eq!(response["result"]["messages"][1]["node_id"], "turn-2");

        let empty_transcript = service
            .execute(request("receive_message", json!({"conversation_id":"missing"}), principal.clone()))
            .await;
        assert_eq!(empty_transcript.status, StatusCode::OK.as_u16());
        assert_eq!(
            serde_json::from_slice::<Value>(&empty_transcript.body).unwrap()["result"]["messages"],
            json!([])
        );

        let denied_mount = service
            .execute(request(
                "mount_memory",
                json!({"conversation_id":"c1"}),
                json!({"ns":["conversation"],"capabilities":[]}),
            ))
            .await;
        assert_eq!(denied_mount.status, StatusCode::FORBIDDEN.as_u16());

        let mounted = service
            .execute(request(
                "mount_memory",
                json!({"conversation_id":"c1","run_id":"run-a","stage":"answer"}),
                json!({
                    "sub":"reader-1", "storage_ns":"scope-a", "workspace_id":"workspace-a",
                    "graph_space":"conversation", "ns":["conversation"], "capabilities":["project_view"]
                }),
            ))
            .await;
        assert_eq!(mounted.status, StatusCode::OK.as_u16());
        let mounted: Value = serde_json::from_slice(&mounted.body).unwrap();
        assert_eq!(mounted["op"], "mount_memory");
        assert_eq!(mounted["result"]["snapshot_node_id"], "snapshot");
        assert_eq!(mounted["result"]["metadata"]["summary"], "stored summary");
        assert_eq!(mounted["result"]["properties"]["references"], json!(["evidence-1"]));
        assert!(mounted["result"].get("schema_version").is_none());

        let invoked_transcript = service
            .execute(request(
                "invoke_tool",
                json!({"tool_name":"list_transcript","conversation_id":"c1"}),
                json!({
                    "sub":"reader-1", "role":"rw", "storage_ns":"scope-a",
                    "workspace_id":"workspace-a", "graph_space":"conversation",
                    "ns":["conversation"], "capabilities":["invoke_tool","read_graph"]
                }),
            ))
            .await;
        let invoke_contract: Value = serde_json::from_str(include_str!(
            "../../../../../contracts/golden/invoke-tool-results-v1.json"
        ))
        .unwrap();
        assert_eq!(invoke_contract["version"], "v1");
        assert_eq!(invoke_contract["op"], "invoke_tool");
        let assert_invoke_result_shape = |tool_name: &str, result: &Value| {
            let expected = &invoke_contract["results"][tool_name];
            let actual_root = result
                .as_object()
                .unwrap()
                .keys()
                .cloned()
                .collect::<BTreeSet<_>>();
            let expected_root = expected["root_fields"]
                .as_array()
                .unwrap()
                .iter()
                .map(|field| field.as_str().unwrap().to_owned())
                .collect::<BTreeSet<_>>();
            assert_eq!(actual_root, expected_root, "{tool_name} root result fields");
            if let Some(snapshot_fields) = expected["snapshot_fields"].as_array() {
                let actual_snapshot = result["snapshot"]
                    .as_object()
                    .unwrap()
                    .keys()
                    .cloned()
                    .collect::<BTreeSet<_>>();
                let expected_snapshot = snapshot_fields
                    .iter()
                    .map(|field| field.as_str().unwrap().to_owned())
                    .collect::<BTreeSet<_>>();
                assert_eq!(
                    actual_snapshot, expected_snapshot,
                    "{tool_name} nested snapshot fields"
                );
            }
        };
        assert_eq!(invoked_transcript.status, StatusCode::OK.as_u16());
        let invoked_transcript: Value = serde_json::from_slice(&invoked_transcript.body).unwrap();
        assert_invoke_result_shape("list_transcript", &invoked_transcript["result"]);
        assert_eq!(invoked_transcript["result"]["tool_name"], "list_transcript");
        assert_eq!(invoked_transcript["result"]["messages"].as_array().unwrap().len(), 2);
        let initial_transcript_digest = format!("{:x}", sha2::Sha256::digest(
            kogwistar_contracts::canonical_json(&invoked_transcript["result"], false).as_bytes(),
        ));

        let invoked_snapshot = service
            .execute(request(
                "invoke_tool",
                json!({"tool_name":"latest_snapshot","conversation_id":"c1","run_id":"run-a","stage":"answer"}),
                json!({
                    "sub":"reader-1", "role":"rw", "storage_ns":"scope-a",
                    "workspace_id":"workspace-a", "graph_space":"conversation",
                    "ns":["conversation"], "capabilities":["invoke_tool","project_view"]
                }),
            ))
            .await;
        assert_eq!(invoked_snapshot.status, StatusCode::OK.as_u16());
        let invoked_snapshot: Value = serde_json::from_slice(&invoked_snapshot.body).unwrap();
        assert_invoke_result_shape("latest_snapshot", &invoked_snapshot["result"]);
        assert_eq!(invoked_snapshot["result"]["tool_name"], "latest_snapshot");
        assert_eq!(invoked_snapshot["result"]["snapshot"]["snapshot_node_id"], "snapshot");
        let initial_snapshot_digest = format!(
            "{:x}",
            sha2::Sha256::digest(
                kogwistar_contracts::canonical_json(&invoked_snapshot["result"], false)
                    .as_bytes(),
            ),
        );

        let denied_builtin = service
            .execute(request(
                "invoke_tool",
                json!({"tool_name":"capability_snapshot"}),
                json!({"role":"rw","ns":["conversation"],"capabilities":["invoke_tool"]}),
            ))
            .await;
        assert_eq!(denied_builtin.status, StatusCode::FORBIDDEN.as_u16());

        let capability_snapshot = service
            .execute(request(
                "invoke_tool",
                json!({"tool_name":"capability_snapshot"}),
                json!({
                    "sub":"reader-1", "role":"rw", "ns":["conversation"],
                    "capabilities":["invoke_tool","project_view","read_security_scope"]
                }),
            ))
            .await;
        assert_eq!(capability_snapshot.status, StatusCode::OK.as_u16());
        let capability_snapshot: Value = serde_json::from_slice(&capability_snapshot.body).unwrap();
        assert_invoke_result_shape("capability_snapshot", &capability_snapshot["result"]);
        assert_eq!(capability_snapshot["result"]["tool_name"], "capability_snapshot");
        assert_eq!(
            capability_snapshot["result"]["snapshot"]["effective_capabilities"][0],
            "invoke_tool"
        );

        let resource_snapshot = service
            .execute(request(
                "invoke_tool",
                json!({"tool_name":"resource_snapshot"}),
                json!({
                    "sub":"reader-1", "role":"rw", "ns":["conversation"],
                    "capabilities":["invoke_tool","project_view","read_security_scope"]
                }),
            ))
            .await;
        assert_eq!(resource_snapshot.status, StatusCode::OK.as_u16());
        let resource_snapshot: Value = serde_json::from_slice(&resource_snapshot.body).unwrap();
        assert_invoke_result_shape("resource_snapshot", &resource_snapshot["result"]);
        assert_eq!(resource_snapshot["result"]["tool_name"], "resource_snapshot");
        assert_eq!(resource_snapshot["result"]["snapshot"]["services"]["total_services"], 0);

        let syscall_audit = service
            .execute(read_request(
            "/api/syscall/v1/audit?limit=100",
                json!({"role":"ro","ns":["workflow"],"capabilities":["project_view"]}),
            ))
            .await;
        assert_eq!(syscall_audit.status, StatusCode::OK.as_u16());
        let syscall_audit: Value = serde_json::from_slice(&syscall_audit.body).unwrap();
        let events = syscall_audit["events"].as_array().unwrap();
        let audit_for = |operation: &str| {
            events
                .iter()
                .filter(|event| event["op"] == operation)
                .collect::<Vec<_>>()
        };
        let receive_audit = audit_for("receive_message");
        assert_eq!(receive_audit.len(), 5);
        assert_eq!(
            receive_audit
                .iter()
                .filter(|event| event["status"] == "error")
                .count(),
            3
        );
        let mount_audit = audit_for("mount_memory");
        assert_eq!(mount_audit.len(), 2);
        assert_eq!(mount_audit[0]["status"], "error");
        assert_eq!(mount_audit[1]["status"], "ok");
        let invoke_audit = audit_for("invoke_tool");
        assert_eq!(invoke_audit.len(), 5);
        assert_eq!(
            invoke_audit.iter().filter(|event| event["status"] == "error").count(),
            1
        );

        let auth = AuthConfig {
            mode: "dev".to_owned(),
            algorithm: Some("HS256".to_owned()),
            key: Some("sqlite-chroma-transcript-secret".to_owned()),
            ..AuthConfig::default()
        };
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let token = encode(
            &Header::new(jsonwebtoken::Algorithm::HS256),
            &json!({
                "sub":"reader-1", "role":"ro", "storage_ns":"scope-a",
                "workspace_id":"workspace-a", "graph_space":"conversation",
                "ns":["conversation"], "capabilities":["read_graph","project_view"], "exp":now + 300
            }),
            &EncodingKey::from_secret(b"sqlite-chroma-transcript-secret"),
        )
        .unwrap();
        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: vec![],
                implementation: ImplementationSnapshot::default(),
                auth,
            },
            Arc::new(service.clone()),
        );
        let http_response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/syscall/v1/receive_message")
                    .header("content-type", "application/json")
                    .header("authorization", format!("Bearer {token}"))
                    .body(Body::from(r#"{"version":"v1","args":{"conversation_id":"c1"}}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        let http_status = http_response.status();
        let http_body = to_bytes(http_response.into_body(), 65_536).await.unwrap();
        assert_eq!(
            http_status,
            StatusCode::OK,
            "response body: {}",
            String::from_utf8_lossy(&http_body)
        );
        let http_payload: Value = serde_json::from_slice(&http_body).unwrap();
        assert_eq!(http_payload["result"]["messages"][0]["node_id"], "turn-1");
        assert_eq!(http_payload["result"]["messages"][1]["node_id"], "turn-2");
        assert_eq!(http_payload["result"]["messages"].as_array().unwrap().len(), 2);

        let http_mount_memory = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: vec![],
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig {
                    mode: "dev".to_owned(),
                    algorithm: Some("HS256".to_owned()),
                    key: Some("sqlite-chroma-transcript-secret".to_owned()),
                    ..AuthConfig::default()
                },
            },
            Arc::new(service.clone()),
        )
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/syscall/v1/mount_memory")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {token}"))
                .body(Body::from(
                    r#"{"version":"v1","args":{"conversation_id":"c1","run_id":"run-a","stage":"answer"}}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
        let mount_status = http_mount_memory.status();
        let mount_body = to_bytes(http_mount_memory.into_body(), 65_536).await.unwrap();
        assert_eq!(
            mount_status,
            StatusCode::OK,
            "mount_memory response: {}",
            String::from_utf8_lossy(&mount_body)
        );
        let mount_payload: Value = serde_json::from_slice(&mount_body).unwrap();
        assert_eq!(mount_payload["result"]["snapshot_node_id"], "snapshot");
        assert_eq!(mount_payload["result"]["metadata"]["summary"], "stored summary");
        assert!(mount_payload["result"].get("schema_version").is_none());

        let transcript_route = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: vec![],
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig {
                    mode: "dev".to_owned(),
                    algorithm: Some("HS256".to_owned()),
                    key: Some("sqlite-chroma-transcript-secret".to_owned()),
                    ..AuthConfig::default()
                },
            },
            Arc::new(service.clone()),
        )
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/api/conversations/c1/turns")
                .header("authorization", format!("Bearer {token}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
        let transcript_status = transcript_route.status();
        let transcript_body = to_bytes(transcript_route.into_body(), 65_536).await.unwrap();
        assert_eq!(
            transcript_status,
            StatusCode::OK,
            "transcript response: {}",
            String::from_utf8_lossy(&transcript_body)
        );
        let transcript_payload: Value = serde_json::from_slice(&transcript_body).unwrap();
        assert_eq!(transcript_payload["conversation_id"], "c1");
        assert_eq!(transcript_payload["turns"].as_array().unwrap().len(), 2);
        assert_eq!(transcript_payload["turns"][0]["node_id"], "turn-1");
        assert!(transcript_payload.get("schema_version").is_none());
        assert!(transcript_payload["turns"][0].get("schema_version").is_none());
        assert!(service.supports_http_operation("GET", "/api/conversations/c1/turns"));

        assert!(service.supports_mcp_tool("conversation", "conversation.get_transcript"));
        let mcp_app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: vec![],
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig {
                    mode: "dev".to_owned(),
                    algorithm: Some("HS256".to_owned()),
                    key: Some("sqlite-chroma-transcript-secret".to_owned()),
                    ..AuthConfig::default()
                },
            },
            Arc::new(service.clone()),
        );
        let mcp_list = mcp_app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp/conversation")
                    .header("content-type", "application/json")
                    .header("authorization", format!("Bearer {token}"))
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":"list-transcript","method":"tools/list"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let mcp_list_body = to_bytes(mcp_list.into_body(), 65_536).await.unwrap();
        let mcp_list_value: Value = serde_json::from_slice(&mcp_list_body).unwrap();
        assert!(mcp_list_value["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .any(|tool| tool["name"] == "conversation.get_transcript"));

        let mcp_transcript = mcp_app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp/conversation")
                    .header("content-type", "application/json")
                    .header("authorization", format!("Bearer {token}"))
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":"get-transcript","method":"tools/call","params":{"name":"conversation.get_transcript","arguments":{"conversation_id":"c1"}}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(mcp_transcript.status(), StatusCode::OK);
        let mcp_transcript_body = to_bytes(mcp_transcript.into_body(), 65_536).await.unwrap();
        let mcp_transcript_value: Value = serde_json::from_slice(&mcp_transcript_body).unwrap();
        assert_eq!(mcp_transcript_value["result"]["conversation_id"], "c1");
        assert_eq!(mcp_transcript_value["result"]["turns"].as_array().unwrap().len(), 2);
        assert_eq!(mcp_transcript_value["result"]["turns"][0]["node_id"], "turn-1");

        for (uri, expected_path) in [
            ("/api/conversations/c1", "detail"),
            ("/api/conversations", "list"),
        ] {
            let response = router_with_application(
                ApiState {
                    health: snapshot(),
                    required_roles: vec![],
                    implementation: ImplementationSnapshot::default(),
                    auth: AuthConfig {
                        mode: "dev".to_owned(),
                        algorithm: Some("HS256".to_owned()),
                        key: Some("sqlite-chroma-transcript-secret".to_owned()),
                        ..AuthConfig::default()
                    },
                },
                Arc::new(service.clone()),
            )
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri(uri)
                    .header("authorization", format!("Bearer {token}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
            let status = response.status();
            let body = to_bytes(response.into_body(), 65_536).await.unwrap();
            assert_eq!(
                status,
                StatusCode::OK,
                "conversation {expected_path} response: {}",
                String::from_utf8_lossy(&body)
            );
            let payload: Value = serde_json::from_slice(&body).unwrap();
            match expected_path {
                "detail" => {
                    assert_eq!(payload["conversation_id"], "c1");
                    assert_eq!(payload["turn_count"], 2);
                }
                "list" => {
                    assert_eq!(payload["conversations"].as_array().unwrap().len(), 1);
                    assert_eq!(payload["conversations"][0]["id"], "c1");
                    assert_eq!(payload["conversations"][0]["start_node_id"], "start");
                    assert_eq!(payload["conversations"][0]["status"], "active");
                    assert_eq!(payload["conversations"][0]["turn_count"], 2);
                }
                _ => unreachable!(),
            }
        }

        let detail = service
            .execute(read_request("/api/conversations/c1", principal.clone()))
            .await;
        assert_eq!(detail.status, StatusCode::OK.as_u16());
        let detail: Value = serde_json::from_slice(&detail.body).unwrap();
        assert_eq!(detail["conversation_id"], "c1");
        assert_eq!(detail["user_id"], "reader-1");
        assert_eq!(detail["turn_count"], 2);
        assert!(detail.get("schema_version").is_none());

        let conversations = service
            .execute(read_request("/api/conversations", principal.clone()))
            .await;
        assert_eq!(
            conversations.status,
            StatusCode::OK.as_u16(),
            "conversation list response: {}",
            String::from_utf8_lossy(&conversations.body)
        );
        let conversations: Value = serde_json::from_slice(&conversations.body).unwrap();
        assert_eq!(conversations["conversations"].as_array().unwrap().len(), 1);
        assert_eq!(conversations["conversations"][0]["id"], "c1");
        assert_eq!(conversations["conversations"][0]["turn_count"], 2);
        assert!(service.supports_http_operation("GET", "/api/conversations/c1"));
        assert!(service.supports_http_operation("GET", "/api/conversations"));

        let denied_detail = service
            .execute(read_request(
                "/api/conversations/c1",
                json!({"ns":["conversation"],"capabilities":[]}),
            ))
            .await;
        assert_eq!(denied_detail.status, StatusCode::FORBIDDEN.as_u16());
        let denied_list = service
            .execute(read_request(
                "/api/conversations",
                json!({"sub":"reader-1","ns":["docs"],"capabilities":["read_graph"]}),
            ))
            .await;
        assert_eq!(denied_list.status, StatusCode::FORBIDDEN.as_u16());
        let no_identity = service
            .execute(read_request(
                "/api/conversations",
                json!({"ns":["conversation"],"capabilities":["read_graph"]}),
            ))
            .await;
        assert_eq!(no_identity.status, StatusCode::UNAUTHORIZED.as_u16());

        // A read-only syscall retry must keep returning the same canonical result
        // after the SQLite authority is reopened; Chroma remains the graph projection.
        drop(service);
        let service = open_sqlite_application_service(
            &path,
            Some(SqliteChromaGraphConfig {
                endpoint,
                metric: DistanceMetric::Cosine,
                node_collection: "nodes".to_owned(),
                edge_collection: "edges".to_owned(),
            }),
        )
        .await
        .unwrap();
        let retried_transcript = service
            .execute(request(
                "invoke_tool",
                json!({"tool_name":"list_transcript","conversation_id":"c1"}),
                json!({
                    "sub":"reader-1", "role":"rw", "storage_ns":"scope-a",
                    "workspace_id":"workspace-a", "graph_space":"conversation",
                    "ns":["conversation"], "capabilities":["invoke_tool","read_graph"]
                }),
            ))
            .await;
        assert_eq!(retried_transcript.status, StatusCode::OK.as_u16());
        let retried_transcript: Value = serde_json::from_slice(&retried_transcript.body).unwrap();
        assert_eq!(retried_transcript["op"], "invoke_tool");
        assert_eq!(
            format!("{:x}", sha2::Sha256::digest(
                kogwistar_contracts::canonical_json(&retried_transcript["result"], false)
                    .as_bytes(),
            )),
            initial_transcript_digest,
            "the same scoped syscall result survives retry after service restart"
        );
        let retried_snapshot = service
            .execute(request(
                "invoke_tool",
                json!({"tool_name":"latest_snapshot","conversation_id":"c1","run_id":"run-a","stage":"answer"}),
                json!({
                    "sub":"reader-1", "role":"rw", "storage_ns":"scope-a",
                    "workspace_id":"workspace-a", "graph_space":"conversation",
                    "ns":["conversation"], "capabilities":["invoke_tool","project_view"]
                }),
            ))
            .await;
        assert_eq!(retried_snapshot.status, StatusCode::OK.as_u16());
        let retried_snapshot: Value = serde_json::from_slice(&retried_snapshot.body).unwrap();
        assert_invoke_result_shape("latest_snapshot", &retried_snapshot["result"]);
        assert_eq!(
            format!(
                "{:x}",
                sha2::Sha256::digest(
                    kogwistar_contracts::canonical_json(&retried_snapshot["result"], false)
                        .as_bytes(),
                ),
            ),
            initial_snapshot_digest,
            "the same ACL-filtered snapshot syscall result survives service restart"
        );
        let _ = stop_chroma.send(());
        let requests = fake.await.unwrap();
        assert!(requests
            .iter()
            .any(|(head, _)| head.contains("collection-edges/get")));
        assert!(requests.iter().filter(|(head, _)| head.contains("collection-nodes/get")).all(
            |(_, body)| body["where"].to_string().contains("workspace_id")
                && body["where"].to_string().contains("conversation")
        ));
        drop(service);
        let _ = std::fs::remove_file(path);

        let metadata_only_path = std::env::temp_dir().join(format!(
            "kogwistar-sqlite-metadata-only-{}.sqlite3",
            uuid::Uuid::new_v4()
        ));
        let metadata_only = SqliteRunApplicationService::open(&metadata_only_path).unwrap();
        assert!(!metadata_only.supports_mcp_tool("conversation", "conversation.get_transcript"));
        let unsupported = metadata_only
            .execute(request(
                "receive_message",
                json!({"conversation_id":"c1"}),
                json!({
                    "sub":"reader-1", "storage_ns":"scope-a", "ns":["conversation"],
                    "capabilities":["read_graph"]
                }),
            ))
            .await;
        assert_eq!(unsupported.status, StatusCode::NOT_IMPLEMENTED.as_u16());
        assert_eq!(
            serde_json::from_slice::<Value>(&unsupported.body).unwrap()["code"],
            "KOGWISTAR_API_CAPABILITY_NOT_CUT_OVER"
        );
        let unsupported_tool = metadata_only
            .execute(request(
                "invoke_tool",
                json!({"tool_name":"list_transcript","conversation_id":"c1"}),
                json!({
                    "sub":"reader-1", "role":"rw", "storage_ns":"scope-a",
                    "ns":["conversation"], "capabilities":["invoke_tool","read_graph"]
                }),
            ))
            .await;
        assert_eq!(unsupported_tool.status, StatusCode::NOT_IMPLEMENTED.as_u16());
        drop(metadata_only);
        let _ = std::fs::remove_file(metadata_only_path);
    }
