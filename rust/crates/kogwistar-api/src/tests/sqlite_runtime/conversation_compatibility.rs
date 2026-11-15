    #[test]
    fn snapshot_syscall_compatibility_drops_rust_only_fields() {
        let expected = serde_json::from_str::<Value>(include_str!(
            "../../../../../../contracts/golden/mount-memory-result-v1.json"
        ))
        .unwrap();
        let mut internal_payload = expected.clone();
        let internal_object = internal_payload.as_object_mut().unwrap();
        internal_object.insert("schema_version".to_owned(), json!(1));
        internal_object.insert("implementation_authority".to_owned(), json!("rust"));
        internal_object.insert("seam_attestation_version".to_owned(), json!(1));
        internal_object.insert("seam_crossings".to_owned(), json!([]));
        let mut response = json_effect(StatusCode::OK, internal_payload);
        strip_snapshot_implementation_fields(&mut response);
        let actual = serde_json::from_slice::<Value>(&response.body).unwrap();
        assert_eq!(actual, expected);

        let typed: ContextSnapshotV1 = serde_json::from_value({
            let mut value = expected.clone();
            value
                .as_object_mut()
                .unwrap()
                .insert("schema_version".to_owned(), json!(1));
            value
        })
        .unwrap();
        assert_eq!(public_context_snapshot_value(typed), expected);
        assert_eq!(
            context_snapshot_syscall_field(json!({}).as_object().unwrap(), "stage"),
            Ok(None)
        );
        assert_eq!(
            context_snapshot_syscall_field(json!({"stage": null}).as_object().unwrap(), "stage"),
            Ok(None)
        );
        assert!(
            context_snapshot_syscall_field(json!({"stage": 1}).as_object().unwrap(), "stage")
                .is_err()
        );

        let mut failure = json_effect(
            StatusCode::NOT_FOUND,
            json!({"error": {"code": "KOGWISTAR_CONTEXT_SNAPSHOT_NOT_FOUND"}}),
        );
        let original_body = failure.body.clone();
        strip_snapshot_implementation_fields(&mut failure);
        assert_eq!(failure.body, original_body);
    }

    #[test]
    fn receive_message_compatibility_matches_shared_wire_fixture() {
        let expected = serde_json::from_str::<Value>(include_str!(
            "../../../../../../contracts/golden/receive-message-result-v1.json"
        ))
        .unwrap();
        let mut internal_payload = expected.clone();
        let internal_object = internal_payload.as_object_mut().unwrap();
        let mut turns = internal_object.remove("messages").unwrap();
        for turn in turns.as_array_mut().unwrap() {
            turn.as_object_mut()
                .unwrap()
                .insert("schema_version".to_owned(), json!(1));
        }
        internal_object.insert("turns".to_owned(), turns);
        internal_object.insert("schema_version".to_owned(), json!(1));
        internal_object.insert("implementation_authority".to_owned(), json!("rust"));
        internal_object.insert("seam_attestation_version".to_owned(), json!(1));
        internal_object.insert("seam_crossings".to_owned(), json!([]));
        let mut response = json_effect(StatusCode::OK, internal_payload);

        legacy_receive_message_payload(&mut response);

        assert_eq!(response.status, StatusCode::OK.as_u16());
        let actual = serde_json::from_slice::<Value>(&response.body).unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn receive_message_invalid_internal_payload_fails_closed() {
        let mut response = json_effect(
            StatusCode::OK,
            json!({
                "schema_version": 1,
                "conversation_id": "conversation-1",
                "turns": [],
                "unexpected": true
            }),
        );

        legacy_receive_message_payload(&mut response);

        assert_eq!(response.status, StatusCode::INTERNAL_SERVER_ERROR.as_u16());
        let error = serde_json::from_slice::<Value>(&response.body).unwrap();
        assert_eq!(error["code"], "KOGWISTAR_INVALID_RESPONSE");
    }
