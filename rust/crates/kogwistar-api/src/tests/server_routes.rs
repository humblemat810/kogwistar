    #[test]
    fn configured_search_index_fts_sidecar_reopens() {
        let nonce = uuid::Uuid::new_v4();
        let graph_path = std::env::temp_dir().join(format!(
            "kogwistar-search-index-graph-{nonce}.sqlite3"
        ));
        let index_path = std::env::temp_dir().join(format!(
            "kogwistar-search-index-sidecar-{nonce}.sqlite3"
        ));
        let service = SqliteRunApplicationService::open(&graph_path)
            .expect("open SQLite graph service")
            .with_search_index_path(&index_path)
            .expect("configure persistent FTS sidecar");
        assert!(index_path.exists());
        drop(service);

        let reopened = crate::search_index_fts::open_index(&index_path)
            .expect("reopen configured FTS sidecar");
        let table_count: i64 = reopened
            .query_row(
                "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='semantic_index_fts'",
                [],
                |row| row.get(0),
            )
            .expect("query FTS schema");
        assert_eq!(table_count, 1);
        drop(reopened);
        remove_test_sqlite(&index_path);
        remove_test_sqlite(&graph_path);
    }

    fn snapshot() -> HealthSnapshot {
        HealthSnapshot {
            backend: "pg".to_owned(),
            persist_directory: "root".to_owned(),
            conversation_persist_directory: "conversation".to_owned(),
            workflow_persist_directory: "workflow".to_owned(),
            wisdom_persist_directory: "wisdom".to_owned(),
            pg_schema_base: Some("kogwistar".to_owned()),
        }
    }

    #[test]
    fn health_preserves_python_shape() {
        let value = health_response(&snapshot());
        assert_eq!(value["ok"], true);
        assert_eq!(value["backend"], "pg");
        assert_eq!(value["pg_schema_base"], "kogwistar");
    }

    #[test]
    fn latest_snapshot_route_parsing_is_exact_and_preserves_query_values() {
        assert_eq!(
            latest_snapshot_conversation_id(
                "/api/conversations/conversation-1/snapshots/latest?run_id=run-1"
            ),
            Some("conversation-1".to_owned())
        );
        assert_eq!(
            latest_snapshot_conversation_id(
                "/api/conversations/conversation-1/extra/snapshots/latest"
            ),
            None
        );
        assert_eq!(
            latest_snapshot_conversation_id("/api/conversations//snapshots/latest"),
            None
        );
        assert_eq!(
            latest_snapshot_conversation_id(
                "/api/conversations/id%20with%20spaces/snapshots/latest"
            ),
            Some("id with spaces".to_owned())
        );
        assert_eq!(
            latest_snapshot_conversation_id("/api/conversations/id%2Fother/snapshots/latest"),
            None
        );
        assert_eq!(
            context_snapshot_query(
                "/api/conversations/c/snapshots/latest?run_id=run%2F1&stage=answer+final"
            ),
            (Some("run/1".to_owned()), Some("answer final".to_owned()))
        );
        assert_eq!(
            context_snapshot_query("/api/conversations/c/snapshots/latest?run_id="),
            (Some(String::new()), None)
        );
    }

    #[test]
    fn conversation_detail_route_parsing_is_exact_and_decodes_path_id() {
        assert!(is_conversation_list_route("/api/conversations"));
        assert!(is_conversation_list_route("/api/conversations?limit=10"));
        assert!(!is_conversation_list_route("/api/conversations/"));
        assert_eq!(
            conversation_detail_id("/api/conversations/conversation-1?view=summary"),
            Some("conversation-1".to_owned())
        );
        assert_eq!(
            conversation_detail_id("/api/conversations/id%20with%20spaces"),
            Some("id with spaces".to_owned())
        );
        assert_eq!(conversation_detail_id("/api/conversations/"), None);
        assert_eq!(conversation_detail_id("/api/conversations/a/b"), None);
        assert_eq!(
            conversation_detail_id("/api/conversations/id%2Fother"),
            None
        );
        assert_eq!(conversation_detail_id("/api/conversations/c/turns"), None);
        assert_eq!(
            conversation_detail_id("/api/conversations/c/snapshots/latest"),
            None
        );
    }

    #[test]
    fn tool_audit_query_decodes_scope_and_validates_limits() {
        assert_eq!(
            tool_audit_query("/api/workflow/tools/audit?conversation_id=conversation%3A1&limit=12")
                .unwrap(),
            (Some("conversation:1".to_owned()), 12)
        );
        assert_eq!(
            tool_audit_query("/api/workflow/tools/audit").unwrap(),
            (None, 200)
        );
        assert!(tool_audit_query("/api/workflow/tools/audit?limit=-1").is_err());
        assert!(tool_audit_query("/api/workflow/tools/audit?limit=10001").is_err());
        assert!(tool_audit_query("/api/workflow/tools/audit?conversation_id=").is_err());
    }

    #[test]
    fn conversation_transcript_route_parsing_is_exact_and_decodes_path_id() {
        assert_eq!(
            conversation_transcript_id("/api/conversations/conversation-1/turns?limit=10"),
            Some("conversation-1".to_owned())
        );
        assert_eq!(
            conversation_transcript_id("/api/conversations/id%20with%20spaces/turns"),
            Some("id with spaces".to_owned())
        );
        assert_eq!(
            conversation_transcript_id("/api/conversations//turns"),
            None
        );
        assert_eq!(
            conversation_transcript_id("/api/conversations/a/b/turns"),
            None
        );
        assert_eq!(
            conversation_transcript_id("/api/conversations/id%2Fother/turns"),
            None
        );
        assert_eq!(conversation_transcript_id("/api/conversations/c"), None);
        assert_eq!(
            conversation_transcript_id("/api/conversations/c/snapshots/latest"),
            None
        );
    }
