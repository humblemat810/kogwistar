    use crate::graph_slice::{
        GRAPH_SLICE_MANIFEST_VERSION, GraphSliceDeclarationV1, GraphSliceFieldV1,
        GraphSliceManifestV1, GraphSliceMode,
    };
    use std::collections::BTreeSet;

    fn fixture() -> Value {
        serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/python-graph-model-payloads-v1.json"
        )))
        .expect("valid Python graph payload fixture")
    }

    fn assert_persistent_v2_validation_matches_v1(model_class: GraphModelClass, payload: &Value) {
        let legacy_result =
            GraphModelEnvelope::new(model_class, payload.clone()).validate_family_v1();
        let wire = serde_json::json!({
            "schema_version": GRAPH_MODEL_ENVELOPE_VERSION,
            "model_class": model_class.wire_name(),
            "payload": payload,
        })
        .to_string();
        let v2_result = RawGraphModelEnvelopeV1::from_json(&wire)
            .and_then(|envelope| envelope.decode_persistent_v2());

        assert_eq!(
            v2_result.is_ok(),
            legacy_result.is_ok(),
            "persistent V2 validation must preserve the V1 family contract for {}: V1={legacy_result:?}, V2={v2_result:?}",
            model_class.wire_name()
        );
    }
