    #[test]
    fn auth_sse_mcp_and_cli_contracts_are_versioned_shapes() {
        assert!(authorize(&ApiAuthRequest {
            roles: vec!["admin".to_owned()],
            required_roles: vec!["admin".to_owned()],
        }));
        assert_eq!(
            sse_frame("run", &json!({"ok": true}), Some("7")),
            "id: 7\nevent: run\ndata: {\"ok\":true}\n\n"
        );
        assert_eq!(mcp_result(json!(1), json!({"ok": true}))["jsonrpc"], "2.0");
        assert_eq!(cli_health(&snapshot()), "ok backend=pg");
    }

    #[tokio::test]
    async fn mcp_session_owner_is_bound_to_configured_oidc_principal_components() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use jsonwebtoken::{EncodingKey, Header, encode};
        use std::time::{SystemTime, UNIX_EPOCH};
        use tower::ServiceExt;

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let secret = "mcp-session-owner-secret";
        let auth = AuthConfig {
            mode: "oidc".to_owned(),
            algorithm: Some("HS256".to_owned()),
            key: Some(secret.to_owned()),
            jwks_json: None,
            issuer: None,
            audience: None,
            oidc_providers_json: None,
            auth_db_url: None,
        };
        let token = |claims: Value| {
            encode(
                &Header::new(Algorithm::HS256),
                &claims,
                &EncodingKey::from_secret(secret.as_bytes()),
            )
            .unwrap()
        };
        let owner_token = token(json!({
            "sub": "shared-user",
            "iss": "issuer-a",
            "client_id": "client-a",
            "role": "reader",
            "exp": now + 300,
        }));
        let other_client_token = token(json!({
            "sub": "shared-user",
            "iss": "issuer-a",
            "client_id": "client-b",
            "role": "reader",
            "exp": now + 300,
        }));
        let other_issuer_token = token(json!({
            "sub": "shared-user",
            "iss": "issuer-b",
            "azp": "client-a",
            "role": "reader",
            "exp": now + 300,
        }));
        let app = router(ApiState {
            health: snapshot(),
            required_roles: vec!["reader".to_owned()],
            implementation: ImplementationSnapshot::default(),
            auth,
        });

        let initialize = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .header("authorization", format!("Bearer {owner_token}"))
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"owner-test","version":"1"}}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(initialize.status(), StatusCode::OK);
        let session_id = initialize.headers()["mcp-session-id"]
            .to_str()
            .unwrap()
            .to_owned();

        for foreign_token in [&other_client_token, &other_issuer_token] {
            let foreign = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/mcp")
                        .header("content-type", "application/json")
                        .header("mcp-protocol-version", "2025-06-18")
                        .header("mcp-session-id", &session_id)
                        .header("authorization", format!("Bearer {foreign_token}"))
                        .body(Body::from(r#"{"jsonrpc":"2.0","id":2,"method":"ping"}"#))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(foreign.status(), StatusCode::NOT_FOUND);
        }

        let owner_ping = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .header("mcp-protocol-version", "2025-06-18")
                    .header("mcp-session-id", session_id)
                    .header("authorization", format!("Bearer {owner_token}"))
                    .body(Body::from(r#"{"jsonrpc":"2.0","id":3,"method":"ping"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(owner_ping.status(), StatusCode::OK);
        let owner_body = to_bytes(owner_ping.into_body(), 4096).await.unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&owner_body).unwrap()["result"],
            json!({})
        );
    }

    #[test]
    fn frozen_mcp_tools_are_unique_and_versioned() {
        let contract: Value = serde_json::from_str(FROZEN_MCP_TOOLS_JSON).unwrap();
        assert_eq!(contract["contract_version"], "1.0.0");
        for (surface, expected) in [("root", 13), ("conversation", 5), ("workflow", 44)] {
            let tools = contract["surfaces"][surface].as_array().unwrap();
            assert_eq!(tools.len(), expected);
            let mut names = tools
                .iter()
                .map(|tool| tool["name"].as_str().unwrap())
                .collect::<Vec<_>>();
            names.sort_unstable();
            names.dedup();
            assert_eq!(names.len(), tools.len());
        }
    }

    #[test]
    fn rust_mcp_advertisement_is_subset_of_implemented_handlers() {
        let contract: Value = serde_json::from_str(FROZEN_MCP_TOOLS_JSON).unwrap();
        let root = contract["surfaces"]["root"].as_array().unwrap();
        let root_supported = root
            .iter()
            .filter(|tool| rust_mcp_tool_supported("root", tool["name"].as_str().unwrap()))
            .count();
        assert_eq!(root_supported, 13);
        assert!(rust_mcp_tool_supported("root", "kg_find_edges"));
        assert!(rust_mcp_tool_supported("root", "kg_load_persisted"));
        assert!(rust_mcp_tool_supported("root", "kg_propose_bruteforce"));
        assert!(rust_mcp_tool_supported("root", "kg_k_hop"));
        assert!(rust_mcp_tool_supported("root", "kg_neighbors"));
        assert!(rust_mcp_tool_supported("root", "propose_vector"));
        assert!(rust_mcp_tool_supported(
            "root",
            "kg_semantic_seed_then_expand_text"
        ));
        assert!(rust_mcp_tool_supported("root", "kg_shortest_path"));
        assert!(rust_mcp_tool_supported("root", "kg_viz_cytoscape_json"));
        assert!(rust_mcp_tool_supported("root", "kg_viz_d3_json"));
        assert!(rust_mcp_tool_supported("root", "wisdom.kg_upsert_graph"));
        assert!(rust_mcp_tool_supported(
            "root",
            "document_id_from_file_name"
        ));
        assert!(rust_mcp_tool_supported(
            "root",
            "wisdom.semantic_seed_then_expand"
        ));
        let conversation = contract["surfaces"]["conversation"].as_array().unwrap();
        let conversation_supported = conversation
            .iter()
            .filter(|tool| rust_mcp_tool_supported("conversation", tool["name"].as_str().unwrap()))
            .count();
        assert_eq!(conversation_supported, 5);
        assert!(rust_mcp_tool_supported("conversation", "conversation.create"));
        assert!(rust_mcp_tool_supported("conversation", "conversation.ask"));
        assert!(rust_mcp_tool_supported(
            "conversation",
            "conversation.get_transcript"
        ));
        let workflow = contract["surfaces"]["workflow"].as_array().unwrap();
        let supported = workflow
            .iter()
            .filter(|tool| rust_mcp_tool_supported("workflow", tool["name"].as_str().unwrap()))
            .count();
        assert_eq!(supported, 44);
        assert!(rust_mcp_tool_supported(
            "workflow",
            "workflow.process_table"
        ));
        assert!(rust_mcp_tool_supported("workflow", "workflow.blocked_runs"));
        assert!(rust_mcp_tool_supported("workflow", "workflow.operator_inbox"));
        assert!(rust_mcp_tool_supported(
            "workflow",
            "workflow.process_timeline"
        ));
        assert!(rust_mcp_tool_supported("workflow", "workflow.run_steps"));
        assert!(rust_mcp_tool_supported("workflow", "workflow.run_evidence"));
        assert!(rust_mcp_tool_supported("workflow", "workflow.run_lineage"));
    }

    #[tokio::test]
    async fn mcp_session_lifecycle_works_over_tcp_transport() {
        let mcp_sessions = Arc::new(McpSessionRegistry::default());
        let app = router_with_application_and_mcp_sessions(
            ApiState {
                health: snapshot(),
                required_roles: Vec::new(),
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig::default(),
            },
            Arc::new(UnavailableApplicationService),
            mcp_sessions.clone(),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .unwrap();
        let endpoint = format!("http://{address}/mcp");
        let accept = "application/json, text/event-stream";
        let initialize = client
            .post(&endpoint)
            .header("accept", accept)
            .header("content-type", "application/json")
            .header("x-kogwistar-roles", "reader")
            .json(&json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "rust-tcp-contract", "version": "1"}
                }
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(initialize.status(), StatusCode::OK);
        assert_eq!(
            initialize.headers()["content-type"],
            "text/event-stream; charset=utf-8"
        );
        let session_id = initialize.headers()["mcp-session-id"]
            .to_str()
            .unwrap()
            .to_owned();
        let initialize_body = initialize.text().await.unwrap();
        let initialize_data = initialize_body
            .lines()
            .find_map(|line| line.strip_prefix("data: "))
            .unwrap();
        assert_eq!(
            serde_json::from_str::<Value>(initialize_data).unwrap()["result"]["protocolVersion"],
            "2025-06-18"
        );

        let initialized = client
            .post(&endpoint)
            .header("accept", accept)
            .header("content-type", "application/json")
            .header("mcp-protocol-version", "2025-06-18")
            .header("mcp-session-id", &session_id)
            .header("x-kogwistar-roles", "reader")
            .json(&json!({"jsonrpc": "2.0", "method": "notifications/initialized"}))
            .send()
            .await
            .unwrap();
        assert_eq!(initialized.status(), StatusCode::ACCEPTED);
        assert_eq!(initialized.headers()["mcp-session-id"], session_id);

        let ping = client
            .post(&endpoint)
            .header("accept", accept)
            .header("content-type", "application/json")
            .header("mcp-protocol-version", "2025-06-18")
            .header("mcp-session-id", &session_id)
            .header("x-kogwistar-roles", "reader")
            .json(&json!({"jsonrpc": "2.0", "id": 2, "method": "ping"}))
            .send()
            .await
            .unwrap();
        assert_eq!(ping.status(), StatusCode::OK);
        assert_eq!(ping.headers()["mcp-session-id"], session_id);
        let ping_body = ping.text().await.unwrap();
        let ping_data = ping_body
            .lines()
            .find_map(|line| line.strip_prefix("data: "))
            .unwrap();
        assert_eq!(
            serde_json::from_str::<Value>(ping_data).unwrap()["result"],
            json!({})
        );

        let session_owner = json!({
            "sub": "anonymous",
            "role": "ro",
            "roles": ["reader"],
            "ns": "workflow"
        });
        mcp_sessions
            .publish(
                &session_id,
                &session_owner,
                Some("2025-06-18"),
                json!({
                    "jsonrpc": "2.0",
                    "method": "notifications/message",
                    "params": {"level": "info", "data": "before stream"}
                }),
            )
            .unwrap();

        let mut get_stream = client
            .get(&endpoint)
            .header("accept", "text/event-stream")
            .header("mcp-protocol-version", "2025-06-18")
            .header("mcp-session-id", &session_id)
            .header("last-event-id", "0")
            .header("x-kogwistar-roles", "reader")
            .send()
            .await
            .unwrap();
        assert_eq!(get_stream.status(), StatusCode::OK);
        assert_eq!(
            get_stream.headers()[header::CONTENT_TYPE],
            "text/event-stream; charset=utf-8"
        );
        let first_frame = get_stream.chunk().await.unwrap().unwrap();
        let first_frame = std::str::from_utf8(&first_frame).unwrap();
        assert!(first_frame.contains("event: message"));
        assert!(first_frame.contains("id: 1"));
        assert!(first_frame.contains("before stream"));

        let duplicate_get = client
            .get(&endpoint)
            .header("accept", "text/event-stream")
            .header("mcp-protocol-version", "2025-06-18")
            .header("mcp-session-id", &session_id)
            .header("x-kogwistar-roles", "reader")
            .send()
            .await
            .unwrap();
        assert_eq!(duplicate_get.status(), StatusCode::CONFLICT);

        mcp_sessions
            .publish(
                &session_id,
                &session_owner,
                Some("2025-06-18"),
                json!({
                    "jsonrpc": "2.0",
                    "method": "notifications/message",
                    "params": {"level": "info", "data": "while connected"}
                }),
            )
            .unwrap();
        let second_frame = get_stream.chunk().await.unwrap().unwrap();
        let second_frame = std::str::from_utf8(&second_frame).unwrap();
        assert!(second_frame.contains("id: 2"));
        assert!(second_frame.contains("while connected"));
        drop(get_stream);

        let mut resumed_get = client
            .get(&endpoint)
            .header("accept", "text/event-stream")
            .header("mcp-protocol-version", "2025-06-18")
            .header("mcp-session-id", &session_id)
            .header("last-event-id", "1")
            .header("x-kogwistar-roles", "reader")
            .send()
            .await
            .unwrap();
        assert_eq!(resumed_get.status(), StatusCode::OK);
        let replayed_frame = resumed_get.chunk().await.unwrap().unwrap();
        let replayed_frame = std::str::from_utf8(&replayed_frame).unwrap();
        assert!(replayed_frame.contains("id: 2"));
        assert!(replayed_frame.contains("while connected"));
        drop(resumed_get);

        let get_stream_without_sse_accept = client
            .get(&endpoint)
            .header("accept", "application/json")
            .header("mcp-protocol-version", "2025-06-18")
            .header("mcp-session-id", &session_id)
            .header("x-kogwistar-roles", "reader")
            .send()
            .await
            .unwrap();
        assert_eq!(
            get_stream_without_sse_accept.status(),
            StatusCode::NOT_ACCEPTABLE
        );

        let terminated = client
            .delete(&endpoint)
            .header("mcp-protocol-version", "2025-06-18")
            .header("mcp-session-id", &session_id)
            .header("x-kogwistar-roles", "reader")
            .send()
            .await
            .unwrap();
        assert_eq!(terminated.status(), StatusCode::OK);

        server.abort();
    }

    #[tokio::test]
    #[ignore = "manual Python MCP SDK smoke; requires KOGWISTAR_PYTHON with mcp/httpx"]
    async fn python_mcp_sdk_receives_rust_server_initiated_notification() {
        use std::process::{Command, Stdio};
        use std::time::Instant;

        let python = std::env::var("KOGWISTAR_PYTHON")
            .expect("set KOGWISTAR_PYTHON to a Python interpreter with mcp and httpx installed");
        let mcp_sessions = Arc::new(McpSessionRegistry::default());
        let app = router_with_application_and_mcp_sessions(
            ApiState {
                health: snapshot(),
                required_roles: Vec::new(),
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig::default(),
            },
            Arc::new(UnavailableApplicationService),
            mcp_sessions.clone(),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}/mcp", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let ready_path = std::env::temp_dir().join(format!(
            "kogwistar-mcp-sdk-session-{}.txt",
            uuid::Uuid::new_v4().simple()
        ));
        let ready_path_literal = serde_json::to_string(&ready_path.to_string_lossy()).unwrap();
        let script = format!(
            r#"
import asyncio
from pathlib import Path
import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

ENDPOINT = {endpoint:?}
READY = Path({ready_path_literal})
received = asyncio.Event()
received_methods = []

async def on_message(message):
    method = getattr(message, "method", None)
    if method == "notifications/message":
        received_methods.append(method)
    if received_methods:
        received.set()

async def record_session(response):
    session_id = response.headers.get("mcp-session-id")
    if session_id:
        READY.write_text(session_id, encoding="utf-8")

async def main():
    async with httpx.AsyncClient(
        headers={{"x-kogwistar-roles": "reader"}},
        event_hooks={{"response": [record_session]}},
        timeout=15,
    ) as client:
        async with streamable_http_client(ENDPOINT, http_client=client) as (read, write):
            async with ClientSession(read, write, message_handler=on_message) as session:
                await session.initialize()
                await asyncio.wait_for(received.wait(), timeout=20)
                assert received_methods == ["notifications/message"], received_methods
                print("python-sdk-server-notification-received")

asyncio.run(main())
"#
        );
        let mut child = Command::new(python)
            .arg("-c")
            .arg(script)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("launch configured Python MCP SDK");
        let deadline = Instant::now() + Duration::from_secs(25);
        let session_id = loop {
            if let Ok(value) = std::fs::read_to_string(&ready_path) {
                break value;
            }
            if child.try_wait().unwrap().is_some() || Instant::now() >= deadline {
                let _ = child.kill();
                let output = child.wait_with_output().unwrap();
                let _ = std::fs::remove_file(&ready_path);
                server.abort();
                panic!(
                    "Python SDK failed before session readiness; stdout={} stderr={}",
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                );
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        };
        let session_owner = json!({
            "sub": "anonymous",
            "role": "ro",
            "roles": ["reader"],
            "ns": "workflow"
        });
        let stream_deadline = Instant::now() + Duration::from_secs(10);
        while !mcp_sessions.has_active_get(&session_id) && Instant::now() < stream_deadline {
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        assert!(
            mcp_sessions.has_active_get(&session_id),
            "Python MCP SDK did not establish the server-initiated GET stream"
        );
        let logging_event_id = mcp_sessions
            .publish(
                &session_id,
                &session_owner,
                Some("2025-11-25"),
                json!({
                    "jsonrpc": "2.0",
                    "method": "notifications/message",
                    "params": {
                        "level": "info",
                        "logger": "rust-interoperability-smoke",
                        "data": "rust-sdk-interoperability-marker"
                    }
                }),
            )
            .unwrap();
        assert_eq!(logging_event_id, "1");
        let child_deadline = Instant::now() + Duration::from_secs(25);
        while child.try_wait().unwrap().is_none() && Instant::now() < child_deadline {
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        if child.try_wait().unwrap().is_none() {
            let _ = child.kill();
        }
        let output = child.wait_with_output().unwrap();
        let _ = std::fs::remove_file(&ready_path);
        server.abort();
        assert!(
            output.status.success(),
            "Python SDK smoke failed; stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(String::from_utf8_lossy(&output.stdout)
            .contains("python-sdk-server-notification-received"));
    }

    #[test]
    fn mcp_outbound_session_events_are_scoped_bounded_and_resumable() {
        let registry = McpSessionRegistry::default();
        let owner = json!({"sub": "client-a"});
        let session_id = registry
            .open(owner.clone(), "2025-11-25".to_owned())
            .unwrap();
        let message = |index| {
            json!({
                "jsonrpc": "2.0",
                "method": "notifications/message",
                "params": {"level": "info", "data": {"index": index}}
            })
        };

        assert_eq!(
            registry.publish(&session_id, &json!({"sub": "client-b"}), None, message(0)),
            Err(McpSessionError::NotFound)
        );
        assert_eq!(
            registry.publish(&session_id, &owner, None, json!({"jsonrpc": "2.0", "id": 1})),
            Err(McpSessionError::InvalidOutboundMessage)
        );
        for unsupported_method in [
            "notifications/progress",
            "notifications/tools/list_changed",
            "notifications/resources/list_changed",
            "notifications/prompts/list_changed",
        ] {
            assert_eq!(
                registry.publish(
                    &session_id,
                    &owner,
                    None,
                    json!({"jsonrpc": "2.0", "method": unsupported_method})
                ),
                Err(McpSessionError::InvalidOutboundMessage),
                "unimplemented MCP notification family must fail closed: {unsupported_method}"
            );
        }
        assert_eq!(
            registry.publish(
                &session_id,
                &owner,
                None,
                json!({
                    "jsonrpc": "2.0",
                    "method": "notifications/message",
                    "params": {"level": "info", "data": "x".repeat(5000)}
                })
            ),
            Err(McpSessionError::OutboundMessageTooLarge)
        );
        for invalid_log in [
            json!({"level": "info"}),
            json!({"level": "bogus", "data": "x"}),
            json!({"level": "info", "data": "x", "logger": 7}),
        ] {
            assert_eq!(
                registry.publish(
                    &session_id,
                    &owner,
                    None,
                    json!({
                        "jsonrpc": "2.0",
                        "method": "notifications/message",
                        "params": invalid_log
                    })
                ),
                Err(McpSessionError::InvalidOutboundMessage)
            );
        }
        assert!(matches!(
            registry.subscribe_get(&session_id, &owner, None, Some("not-a-sequence")),
            Err(McpSessionError::InvalidLastEventId)
        ));

        for index in 0..257 {
            registry
                .publish(&session_id, &owner, None, message(index))
                .unwrap();
        }
        assert!(matches!(
            registry.subscribe_get(&session_id, &owner, None, Some("0")),
            Err(McpSessionError::EventHistoryUnavailable)
        ));
        let resumed = registry
            .subscribe_get(&session_id, &owner, Some("2025-11-25"), Some("256"))
            .unwrap();
        assert_eq!(resumed.replay.len(), 1);
        assert_eq!(resumed.replay[0].event_id(), "257");
        assert!(matches!(
            registry.subscribe_get(&session_id, &owner, None, None),
            Err(McpSessionError::GetStreamAlreadyActive)
        ));
        registry.finish_get(&session_id);

        let final_log = json!({
            "jsonrpc": "2.0",
            "method": "notifications/message",
            "params": {"level": "notice", "data": "tail"}
        });
        assert_eq!(
            registry
                .publish(&session_id, &owner, None, final_log.clone())
                .unwrap(),
            "258"
        );
        let resumed_log = registry
            .subscribe_get(&session_id, &owner, None, Some("257"))
            .unwrap();
        assert_eq!(resumed_log.replay.len(), 1);
        assert_eq!(resumed_log.replay[0].event_id(), "258");
        assert_eq!(resumed_log.replay[0].message(), &final_log);
        registry.finish_get(&session_id);
    }

    #[test]
    fn mcp_session_registry_expires_idle_sessions_but_preserves_active_streams() {
        let registry = McpSessionRegistry::default();
        let owner = json!({"sub": "client-a"});
        let idle_session = registry
            .open(owner.clone(), "2025-11-25".to_owned())
            .unwrap();
        {
            let mut sessions = registry
                .sessions
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            sessions.get_mut(&idle_session).unwrap().last_seen =
                Instant::now() - MCP_SESSION_TTL;
        }
        assert_eq!(
            registry.touch(&idle_session, &owner, None),
            Err(McpSessionError::NotFound)
        );
        assert!(!registry
            .sessions
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .contains_key(&idle_session));

        let active_session = registry
            .open(owner.clone(), "2025-11-25".to_owned())
            .unwrap();
        let _subscription = registry
            .subscribe_get(&active_session, &owner, None, None)
            .unwrap();
        {
            let mut sessions = registry
                .sessions
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            sessions.get_mut(&active_session).unwrap().last_seen =
                Instant::now() - MCP_SESSION_TTL;
        }
        assert_eq!(
            registry.touch(&active_session, &owner, None),
            Ok("2025-11-25".to_owned())
        );
        registry.finish_get(&active_session);
    }

    #[test]
    fn mcp_session_registry_enforces_capacity_and_reclaims_expired_slots() {
        let registry = McpSessionRegistry::default();
        let owner = json!({"sub": "bounded-client"});
        let sessions = (0..MCP_MAX_SESSIONS)
            .map(|_| {
                registry
                    .open(owner.clone(), "2025-11-25".to_owned())
                    .expect("all slots below the configured cap should be admitted")
            })
            .collect::<Vec<_>>();

        assert_eq!(
            registry.open(owner.clone(), "2025-11-25".to_owned()),
            None,
            "an additional live session must be rejected at capacity"
        );
        {
            let mut active = registry
                .sessions
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            active.get_mut(&sessions[0]).unwrap().last_seen = Instant::now() - MCP_SESSION_TTL;
        }
        assert!(registry
            .open(owner, "2025-11-25".to_owned())
            .is_some());
        assert_eq!(
            registry
                .sessions
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .len(),
            MCP_MAX_SESSIONS
        );
    }
