    #[test]
    fn sqlite_native_answer_view_projects_only_the_admitted_conversation() {
        let record = |id: &str, document: Value, metadata: serde_json::Map<String, Value>| {
            GraphRecord {
                id: id.to_owned(),
                document: Some(serde_json::to_string(&document).unwrap()),
                metadata,
                embedding: None,
            }
        };
        let nodes = vec![
            record("turn-c1", json!({
                "entity_type": "conversation_turn",
                "conversation_id": "c1",
                "role": "user",
                "content": "question",
            }), serde_json::Map::new()),
            record("context-c1", json!({
                "entity_type": "conversation_turn",
                "conversation_id": "c1",
                "role": "assistant",
                "content": "answer",
            }), serde_json::Map::new()),
            record("turn-c2", json!({
                "entity_type": "conversation_turn",
                "conversation_id": "c2",
                "role": "user",
                "content": "other question",
            }), serde_json::Map::new()),
        ];
        let edges = vec![
            record("edge-c1", json!({"source": "turn-c1", "target": "context-c1"}),
                serde_json::Map::from_iter([("doc_id".to_owned(), json!("conv:c1"))])),
            record("edge-c2", json!({"source": "turn-c2", "target": "context-c2"}),
                serde_json::Map::from_iter([("doc_id".to_owned(), json!("conv:c2"))])),
        ];

        let view = match sqlite_native_answer_view_from_visible_records(
            &nodes,
            &edges,
            "c1",
            "turn-c1",
        ) {
            Ok(view) => view,
            Err(_) => panic!("valid visible conversation context should be accepted"),
        };
        assert_eq!(view["turn"]["id"], "turn-c1");
        assert_eq!(view["context_nodes"].as_array().unwrap().len(), 2);
        assert_eq!(view["context_edges"].as_array().unwrap().len(), 1);

        let inactive_turn = record("turn-c1", json!({
            "entity_type": "conversation_turn",
            "conversation_id": "c1",
            "role": "user",
            "content": "question",
            "lifecycle_status": "tombstoned",
        }), serde_json::Map::new());
        assert!(sqlite_native_answer_view_from_visible_records(
            &[inactive_turn],
            &[],
            "c1",
            "turn-c1",
        )
        .is_err());
    }

    #[test]
    fn sqlite_native_candidates_exclude_acl_private_and_tombstoned_records() {
        let node = |id: &str, lifecycle: Option<&str>| {
            let mut metadata = serde_json::Map::new();
            metadata.insert("label".to_owned(), json!(id));
            metadata.insert("summary".to_owned(), json!(format!("summary {id}")));
            if let Some(status) = lifecycle {
                metadata.insert("lifecycle_status".to_owned(), json!(status));
            }
            GraphRecord {
                id: id.to_owned(),
                document: Some(format!("document {id}")),
                metadata,
                embedding: None,
            }
        };
        let acl = |id: &str, target: &str, mode: &str| {
            let mut metadata = serde_json::Map::new();
            metadata.insert("entity_type".to_owned(), json!("acl_record"));
            metadata.insert("acl_truth_graph".to_owned(), json!("knowledge"));
            metadata.insert("acl_target_grain".to_owned(), json!("node"));
            metadata.insert("acl_target_entity_id".to_owned(), json!(target));
            metadata.insert("acl_version".to_owned(), json!(1));
            metadata.insert("acl_mode".to_owned(), json!(mode));
            if mode == "private" {
                metadata.insert("acl_owner_id".to_owned(), json!("other-user"));
            }
            GraphRecord {
                id: id.to_owned(),
                document: None,
                metadata,
                embedding: None,
            }
        };
        let candidates = vec![
            node("visible", None),
            node("private", None),
            node("deleted", Some("tombstoned")),
        ];
        let acl_records = vec![
            acl("acl-visible", "visible", "public"),
            acl("acl-private", "private", "private"),
            acl("acl-deleted", "deleted", "public"),
        ];
        let principal = GraphAclPrincipal {
            principal_id: "reader".to_owned(),
            ..Default::default()
        };
        let projected = sqlite_native_candidate_values(
            &candidates,
            &acl_records,
            "knowledge",
            &principal,
        );
        assert_eq!(projected.as_array().unwrap().len(), 1);
        assert_eq!(projected[0]["id"], "visible");
        assert_eq!(projected[0]["doc"], "document visible");
    }
