    #[test]
    fn raw_family_validator_accepts_oversized_python_hierarchy_levels() {
        let fixture = fixture();
        let baseline = fixture["cases"][0]["payload"].clone();
        let huge = "18446744073709551616";

        let top_level_payload = serde_json::to_string(&baseline)
            .expect("serialize baseline payload")
            .replace(
                "\"level_from_root\":0",
                &format!("\"level_from_root\":{huge}"),
            );
        let top_level_wire = format!(
            "{{\"schema_version\":1,\"model_class\":\"node\",\"payload\":{top_level_payload}}}"
        );
        assert!(top_level_wire.contains(huge));
        validate_graph_model_family_from_str(&top_level_wire)
            .expect("large top-level Python integer validates");
        assert!(
            validate_graph_model_family_from_str(&top_level_wire.replace(huge, "2.5")).is_err()
        );
        let negative_huge = format!("-{huge}");
        let negative_top_level = top_level_wire.replace(huge, &negative_huge);
        validate_graph_model_family_from_str(&negative_top_level)
            .expect("large negative top-level Python integer validates");
        let negative_v2 = RawGraphModelEnvelopeV1::from_json(&negative_top_level)
            .expect("negative raw envelope parses")
            .decode_persistent_v2()
            .expect("negative V2 decode retains the exact integer");
        assert_eq!(
            negative_v2
                .level_from_root
                .expect("negative level present")
                .as_decimal_str(),
            negative_huge
        );

        let mut metadata_only = baseline;
        metadata_only
            .as_object_mut()
            .expect("node payload object")
            .remove("level_from_root");
        let metadata_payload = serde_json::to_string(&metadata_only)
            .expect("serialize metadata-only payload")
            .replace(
                "\"level_from_root\":0",
                &format!("\"level_from_root\":{huge}"),
            );
        let metadata_wire = format!(
            "{{\"schema_version\":1,\"model_class\":\"node\",\"payload\":{metadata_payload}}}"
        );
        validate_graph_model_family_from_str(&metadata_wire)
            .expect("large metadata fallback integer validates");
        let negative_metadata = metadata_wire.replace(huge, &negative_huge);
        validate_graph_model_family_from_str(&negative_metadata)
            .expect("large negative metadata fallback integer validates");
    }

    #[test]
    fn raw_graph_envelope_round_trip_preserves_python_big_integer_tokens() {
        let fixture = fixture();
        let baseline = serde_json::to_string(&fixture["cases"][0]["payload"])
            .expect("serialize graph payload");
        let huge = "1606938044258990275541962092341162602522202993782792835301376";
        let payload = baseline.replace(
            "\"level_from_root\":0",
            &format!("\"level_from_root\":{huge}"),
        );
        let wire =
            format!("{{\"schema_version\":1,\"model_class\":\"node\",\"payload\":{payload}}}");

        let envelope = RawGraphModelEnvelopeV1::from_json(&wire)
            .expect("raw envelope accepts valid JSON without materializing payload numbers");
        envelope
            .validate_family_v1()
            .expect("raw envelope validates using the shared family contract");
        let round_trip = envelope.to_json().expect("serialize raw envelope");

        assert!(round_trip.contains(huge));
        assert_eq!(envelope.payload.get(), payload);
    }

    #[test]
    fn raw_graph_envelope_reads_python_integer_without_narrowing_or_losing_precedence() {
        let huge = "1606938044258990275541962092341162602522202993782792835301376";
        let wire = format!(
            "{{\"schema_version\":1,\"model_class\":\"node\",\"payload\":{{\
             \"level_from_root\":{huge},\"metadata\":{{\"level_from_root\":7}}}}}}"
        );
        let envelope = RawGraphModelEnvelopeV1::from_json(&wire).expect("valid raw envelope");
        assert_eq!(
            envelope
                .level_from_root()
                .expect("read exact top-level integer")
                .expect("level is present")
                .as_decimal_str(),
            huge
        );

        let metadata_wire = wire.replace(&format!("\"level_from_root\":{huge},"), "");
        let metadata_envelope =
            RawGraphModelEnvelopeV1::from_json(&metadata_wire).expect("valid raw envelope");
        assert_eq!(
            metadata_envelope
                .level_from_root()
                .expect("read metadata fallback")
                .expect("metadata level is present")
                .as_decimal_str(),
            "7"
        );

        let null_wire = metadata_wire.replace("\"level_from_root\":7", "\"level_from_root\":null");
        let null_envelope =
            RawGraphModelEnvelopeV1::from_json(&null_wire).expect("valid raw envelope");
        assert_eq!(
            null_envelope
                .level_from_root()
                .expect("explicit null is valid"),
            None
        );
    }

    #[test]
    fn persistent_v2_keeps_unbounded_level_and_lossless_payload_without_changing_v1() {
        let fixture = fixture();
        let huge = "1606938044258990275541962092341162602522202993782792835301376";
        let baseline = serde_json::to_string(&fixture["cases"][0]["payload"])
            .expect("serialize graph payload");
        let payload = baseline.replace(
            "\"level_from_root\":0",
            &format!("\"level_from_root\":{huge}"),
        );
        let wire =
            format!("{{\"schema_version\":1,\"model_class\":\"node\",\"payload\":{payload}}}");
        let raw = RawGraphModelEnvelopeV1::from_json(&wire).expect("valid raw envelope");
        let decoded = raw
            .decode_persistent_v2()
            .expect("typed V2 decoder retains the Python integer");

        assert_eq!(
            decoded
                .level_from_root
                .as_ref()
                .expect("level remains available")
                .as_decimal_str(),
            huge
        );
        assert!(decoded.raw_payload.get().contains(huge));
        let PersistentGraphEntityV1::Node(node) = decoded.entity else {
            panic!("node envelope must decode as a node");
        };
        assert_eq!(node.core.level_from_root, None);

        let ordinary_wire =
            format!("{{\"schema_version\":1,\"model_class\":\"node\",\"payload\":{baseline}}}");
        let ordinary = RawGraphModelEnvelopeV1::from_json(&ordinary_wire)
            .expect("ordinary level envelope")
            .decode_persistent_v2()
            .expect("ordinary level remains compatible");
        assert_eq!(
            ordinary
                .level_from_root
                .expect("ordinary level is present")
                .as_decimal_str(),
            "0"
        );
        let PersistentGraphEntityV1::Node(node) = ordinary.entity else {
            panic!("node envelope must decode as a node");
        };
        assert_eq!(node.core.level_from_root, Some(0));
    }

    #[test]
    fn persistent_v2_reads_unbounded_metadata_level_without_narrowing() {
        let fixture = fixture();
        let huge = "1606938044258990275541962092341162602522202993782792835301376";
        let mut payload = fixture["cases"][0]["payload"].clone();
        let object = payload.as_object_mut().expect("fixture payload object");
        object.remove("level_from_root");
        object
            .get_mut("metadata")
            .and_then(Value::as_object_mut)
            .expect("metadata object")
            .insert("level_from_root".to_owned(), Value::Number(7.into()));
        let payload = serde_json::to_string(&payload)
            .expect("serialize graph payload")
            .replace(
                "\"level_from_root\":7",
                &format!("\"level_from_root\":{huge}"),
            );
        let wire =
            format!("{{\"schema_version\":1,\"model_class\":\"node\",\"payload\":{payload}}}");
        let decoded = RawGraphModelEnvelopeV1::from_json(&wire)
            .expect("valid raw envelope")
            .decode_persistent_v2()
            .expect("metadata level remains available");

        assert_eq!(
            decoded
                .level_from_root
                .expect("metadata level is present")
                .as_decimal_str(),
            huge
        );
        assert!(decoded.raw_payload.get().contains(huge));
    }
