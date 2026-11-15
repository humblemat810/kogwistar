    #[test]
    fn workflow_capability_approve_mcp_admission_matches_python_policy_for_all_stores() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "kogwistar-capability-approve-mcp-{nonce}.sqlite3"
        ));
        let sqlite = SqliteRunApplicationService::open(&path).unwrap();
        let postgres = PostgresRunApplicationService::from_dsn(
            "postgres://unused:unused@127.0.0.1:1/unused",
            "capability_approve_mcp_admission",
        )
        .unwrap();
        let services: [&dyn ApplicationService; 2] = [&sqlite, &postgres];

        for service in services {
            for name in [
                "workflow.capability_approve",
                "workflow.capability_revoke",
            ] {
                assert!(service.supports_mcp_tool_for_principal(
                    "workflow",
                    name,
                    &json!({"role":"rw", "ns":["workflow"], "capabilities":["approve_action", "project_view", "read_security_scope"]})
                ));
                for capabilities in [
                    vec![],
                    vec!["approve_action"],
                    vec!["approve_action", "project_view"],
                    vec!["approve_action", "read_security_scope"],
                ] {
                    assert!(!service.supports_mcp_tool_for_principal(
                        "workflow",
                        name,
                        &json!({"role":"rw", "ns":["workflow"], "capabilities":capabilities})
                    ));
                }
                assert!(!service.supports_mcp_tool_for_principal(
                    "workflow",
                    name,
                    &json!({"role":"ro", "ns":["workflow"], "capabilities":["approve_action", "project_view", "read_security_scope"]})
                ));
                assert!(!service.supports_mcp_tool_for_principal(
                    "workflow",
                    name,
                    &json!({"role":"rw", "ns":["conversation"], "capabilities":["approve_action", "project_view", "read_security_scope"]})
                ));
                assert!(!service.supports_mcp_tool_for_principal(
                    "root",
                    name,
                    &json!({"role":"rw", "ns":["workflow"], "capabilities":["approve_action", "project_view", "read_security_scope"]})
                ));
            }
        }

        drop(sqlite);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn conversation_run_status_mcp_admission_matches_python_policy_for_all_stores() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-status-mcp-{nonce}.sqlite3"));
        let sqlite = SqliteRunApplicationService::open(&path).unwrap();
        let postgres = PostgresRunApplicationService::from_dsn(
            "postgres://unused:unused@127.0.0.1:1/unused",
            "conversation_status_mcp_admission",
        )
        .unwrap();
        let services: [&dyn ApplicationService; 2] = [&sqlite, &postgres];

        for service in services {
            assert!(service.supports_mcp_tool_for_principal(
                "conversation",
                "conversation.run_status",
                &json!({"role":"ro", "ns":["conversation"]})
            ));
            assert!(!service.supports_mcp_tool_for_principal(
                "conversation",
                "conversation.run_status",
                &json!({"role":"rw", "ns":["workflow"]})
            ));
            assert!(service.supports_mcp_tool_for_principal(
                "conversation",
                "conversation.run_status",
                &json!({"role":"anonymous", "ns":["conversation"]})
            ));
            assert!(!service.supports_mcp_tool_for_principal(
                "root",
                "conversation.run_status",
                &json!({"role":"ro", "ns":["conversation"]})
            ));
        }

        drop(sqlite);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn conversation_transcript_mcp_admission_matches_python_policy_for_all_stores() {
        use std::time::{SystemTime, UNIX_EPOCH};
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-transcript-mcp-{nonce}.sqlite3"));

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let chroma_setup = tokio::spawn(async move {
            for _ in 0..2 {
                let (mut stream, _) = listener.accept().await.unwrap();
                let mut bytes = Vec::new();
                let mut chunk = [0_u8; 2048];
                let (header_end, content_length) = loop {
                    let count = stream.read(&mut chunk).await.unwrap();
                    assert!(count > 0, "Chroma connection closed before request body");
                    bytes.extend_from_slice(&chunk[..count]);
                    if let Some(header_end) = bytes
                        .windows(4)
                        .position(|window| window == b"\r\n\r\n")
                    {
                        let headers = String::from_utf8_lossy(&bytes[..header_end]);
                        let content_length = headers
                            .lines()
                            .find_map(|line| {
                                let (name, value) = line.split_once(':')?;
                                name.eq_ignore_ascii_case("content-length")
                                    .then(|| value.trim().parse::<usize>().ok())
                                    .flatten()
                            })
                            .unwrap_or(0);
                        if bytes.len() >= header_end + 4 + content_length {
                            break (header_end, content_length);
                        }
                    }
                };
                let request: Value = serde_json::from_slice(
                    &bytes[header_end + 4..header_end + 4 + content_length],
                )
                .unwrap();
                let response = json!({
                    "id": format!("collection-{}", request["name"].as_str().unwrap()),
                    "metadata": {"hnsw:space": "cosine"}
                })
                .to_string();
                let wire = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    response.len(), response
                );
                stream.write_all(wire.as_bytes()).await.unwrap();
            }
        });
        let chroma_config = |collection: &str| kogwistar_store_chroma::ChromaStoreConfig {
            endpoint: endpoint.clone(),
            collection: collection.to_owned(),
            ..Default::default()
        };
        let graph = kogwistar_store_chroma::ChromaGraphStore::connect_entity_collections(
            chroma_config("transcript-nodes"),
            chroma_config("transcript-edges"),
        )
        .await
        .unwrap();
        chroma_setup.await.unwrap();

        let sqlite = SqliteRunApplicationService::open(&path)
            .unwrap()
            .with_chroma_graph(graph);
        let postgres = PostgresRunApplicationService::from_dsn(
            "postgres://unused:unused@127.0.0.1:1/unused",
            "conversation_transcript_mcp_admission",
        )
        .unwrap();
        let services: [&dyn ApplicationService; 2] = [&sqlite, &postgres];

        for service in services {
            assert!(service.supports_mcp_tool_for_principal(
                "conversation",
                "conversation.get_transcript",
                &json!({"role":"ro", "ns":["conversation"], "capabilities":["read_graph"]})
            ));
            assert!(!service.supports_mcp_tool_for_principal(
                "conversation",
                "conversation.get_transcript",
                &json!({"role":"ro", "ns":["conversation"], "capabilities":[]})
            ));
            assert!(!service.supports_mcp_tool_for_principal(
                "conversation",
                "conversation.get_transcript",
                &json!({"role":"rw", "ns":["conversation"], "capabilities":["write_graph"]})
            ));
            assert!(!service.supports_mcp_tool_for_principal(
                "conversation",
                "conversation.get_transcript",
                &json!({"role":"ro", "ns":["workflow"], "capabilities":["read_graph"]})
            ));
            assert!(!service.supports_mcp_tool_for_principal(
                "root",
                "conversation.get_transcript",
                &json!({"role":"ro", "ns":["conversation"], "capabilities":["read_graph"]})
            ));
            assert!(service.supports_mcp_tool_for_principal(
                "conversation",
                "conversation.get_transcript",
                &json!({"role":"ro", "ns":["conversation"]})
            ));
        }

        drop(sqlite);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn conversation_cancel_mcp_admission_matches_python_policy_for_all_stores() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-cancel-mcp-{nonce}.sqlite3"));
        let sqlite = SqliteRunApplicationService::open(&path).unwrap();
        let postgres = PostgresRunApplicationService::from_dsn(
            "postgres://unused:unused@127.0.0.1:1/unused",
            "conversation_cancel_mcp_admission",
        )
        .unwrap();
        let services: [&dyn ApplicationService; 2] = [&sqlite, &postgres];

        for service in services {
            assert!(service.supports_mcp_tool_for_principal(
                "conversation",
                "conversation.cancel_run",
                &json!({"role":"rw", "ns":["conversation"]})
            ));
            assert!(service.supports_mcp_tool_for_principal(
                "conversation",
                "conversation.cancel_run",
                &json!({"role":"rw", "ns":["*"]})
            ));
            assert!(!service.supports_mcp_tool_for_principal(
                "conversation",
                "conversation.cancel_run",
                &json!({"role":"ro", "ns":["conversation"], "capabilities":["workflow.run.write"]})
            ));
            assert!(!service.supports_mcp_tool_for_principal(
                "conversation",
                "conversation.cancel_run",
                &json!({"role":"rw", "ns":["workflow"], "capabilities":["workflow.run.write"]})
            ));
            assert!(!service.supports_mcp_tool_for_principal(
                "root",
                "conversation.cancel_run",
                &json!({"role":"rw", "ns":["conversation"]})
            ));
        }

        drop(sqlite);
        let _ = std::fs::remove_file(path);
    }
