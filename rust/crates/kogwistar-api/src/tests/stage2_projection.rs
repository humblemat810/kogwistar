    #[test]
    fn sqlite_chroma_stage2_read_fence_requires_matching_active_event_revision() {
        let event = SqliteRawEntityEvent {
            namespace: "tenant-a".to_owned(),
            seq: 7,
            event_id: "node-7".to_owned(),
            entity_kind: "node".to_owned(),
            entity_id: "n1".to_owned(),
            op: "REPLACE".to_owned(),
            payload_json: r#"{"lifecycle_patch":{"lifecycle_status":"active"}}"#.to_owned(),
            created_at: 0,
        };
        let stage2_record = |ready: Value, revision: Option<Value>| {
            let mut metadata = serde_json::Map::new();
            metadata.insert("_kogwistar_stage2_ready".to_owned(), ready);
            if let Some(revision) = revision {
                metadata.insert("_kogwistar_source_revision".to_owned(), revision);
            }
            GraphRecord {
                id: "n1".to_owned(),
                document: Some("node payload".to_owned()),
                metadata,
                embedding: None,
            }
        };

        assert!(SqliteRunApplicationService::stage2_record_is_current(
            &stage2_record(Value::Bool(true), Some(json!(7))),
            Some(&event),
        ));
        assert!(!SqliteRunApplicationService::stage2_record_is_current(
            &stage2_record(Value::Bool(true), Some(json!(6))),
            Some(&event),
        ));
        assert!(!SqliteRunApplicationService::stage2_record_is_current(
            &stage2_record(Value::Bool(true), None),
            Some(&event),
        ));
        assert!(!SqliteRunApplicationService::stage2_record_is_current(
            &stage2_record(Value::Bool(true), Some(json!(7))),
            None,
        ));
        assert!(!SqliteRunApplicationService::stage2_record_is_current(
            &stage2_record(Value::Bool(true), Some(Value::Bool(true))),
            Some(&event),
        ));
        assert!(!SqliteRunApplicationService::stage2_record_is_current(
            &stage2_record(Value::Bool(false), Some(json!(7))),
            Some(&event),
        ));

        let deleted = SqliteRawEntityEvent {
            op: "DELETE".to_owned(),
            ..event.clone()
        };
        assert!(!SqliteRunApplicationService::stage2_record_is_current(
            &stage2_record(Value::Bool(true), Some(json!(7))),
            Some(&deleted),
        ));
        let legacy_single_stage = GraphRecord {
            id: "legacy".to_owned(),
            document: None,
            metadata: serde_json::Map::new(),
            embedding: None,
        };
        assert!(SqliteRunApplicationService::stage2_record_is_current(
            &legacy_single_stage,
            None,
        ));
    }
