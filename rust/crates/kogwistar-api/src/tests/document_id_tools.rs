    #[tokio::test]
    async fn document_id_lookup_mcp_admission_matches_python_policy_for_all_stores() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("kogwistar-document-lookup-auth-{nonce}.sqlite3"));
        let sqlite = SqliteRunApplicationService::open(&path).unwrap();
        let postgres = PostgresRunApplicationService::from_dsn(
            "postgres://unused:unused@127.0.0.1:1/unused",
            "document_lookup_mcp_auth",
        )
        .unwrap();
        let services: [&dyn ApplicationService; 2] = [&sqlite, &postgres];

        for service in services {
            let authorized = json!({
                "sub":"document-lookup-authorized",
                "role":"ro",
                "ns":["docs"],
                "storage_ns":"docs",
                "capabilities":["read_graph"]
            });
            assert!(service.supports_mcp_tool_for_principal(
                "root",
                "document_id_from_file_name",
                &authorized
            ));
            // Roles retain their historical implicit read_graph baseline.
            assert!(service.supports_mcp_tool_for_principal(
                "root",
                "document_id_from_file_name",
                &json!({"role":"ro", "ns":["docs"]})
            ));

            for principal in [
                json!({"sub":"no-capability", "role":"ro", "ns":["docs"], "capabilities":[]}),
                json!({"sub":"wrong-capability", "role":"rw", "ns":["docs"], "capabilities":["write_graph"]}),
                json!({"sub":"wrong-scope", "role":"ro", "ns":["conversation"], "capabilities":["read_graph"]}),
            ] {
                assert!(!service.supports_mcp_tool_for_principal(
                    "root",
                    "document_id_from_file_name",
                    &principal
                ));
            }
            assert!(!service.supports_mcp_tool_for_principal(
                "conversation",
                "document_id_from_file_name",
                &authorized
            ));

            let denied = service
                .execute(ApiEffectRequest {
                    contract_version: 1,
                    method: "POST".to_owned(),
                    path_and_query: "/internal/document/lookup".to_owned(),
                    body: br#"{"file_name":"document-id-fixture"}"#.to_vec(),
                    principal: json!({
                        "sub":"no-capability",
                        "role":"ro",
                        "ns":["docs"],
                        "storage_ns":"docs",
                        "capabilities":[]
                    }),
                })
                .await;
            assert_eq!(denied.status, StatusCode::FORBIDDEN.as_u16());
        }

        drop(sqlite);
        let _ = std::fs::remove_file(path);
    }
