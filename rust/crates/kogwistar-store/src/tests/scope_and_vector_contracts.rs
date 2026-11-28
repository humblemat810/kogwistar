#[test]
fn graph_scope_metadata_rejects_unrequested_workspace_and_graph_space() {
    let scope = GraphScope {
        namespace: "tenant-a".to_owned(),
        workspace_id: None,
        graph_space: None,
    };
    assert!(
        scope
            .validate_metadata_scope(&Map::from_iter([(
                "workspace_id".to_owned(),
                Value::String("workspace-a".to_owned()),
            )]))
            .is_err()
    );
    assert!(
        scope
            .validate_metadata_scope(&Map::from_iter([(
                "graph_space".to_owned(),
                Value::String("conversation".to_owned()),
            )]))
            .is_err()
    );
    assert!(
        scope
            .validate_metadata_scope(&Map::from_iter([("workspace_id".to_owned(), Value::Null,)]))
            .is_ok()
    );
}

#[test]
fn standardized_vector_distance_scores_are_backend_neutral() {
    assert_eq!(
        similarity_from_distance(Some(0.25), DistanceMetric::Cosine),
        Some(0.75)
    );
    assert_eq!(
        similarity_from_distance(Some(3.0), DistanceMetric::L2),
        Some(0.25)
    );
    assert_eq!(
        similarity_from_distance(Some(-0.8), DistanceMetric::InnerProduct),
        Some(0.8)
    );
    assert_eq!(similarity_from_distance(None, DistanceMetric::Cosine), None);
    assert_eq!(
        similarity_from_distance(Some(f64::NAN), DistanceMetric::Cosine),
        None
    );
}

fn edge_record(id: &str, document: Value) -> GraphRecord {
    GraphRecord {
        id: id.to_owned(),
        document: Some(document.to_string()),
        metadata: Map::new(),
        embedding: None,
    }
}

fn node_record(id: &str) -> GraphRecord {
    GraphRecord::new(id)
}

// PYTHON_PARITY_V1: rust=rust/crates/kogwistar-store/src/lib.rs#graph_viz_d3_from_records; python=kogwistar/visualization/graph_viz.py#to_d3_force; fixture=contracts/golden/python-graph-model-payloads-v1.json; rust_test=rust/crates/kogwistar-store/src/lib.rs#graph_visualization_model_payloads_match_python_fixture; python_test=tests/test_visualization.py#test_graph_visualization_payloads_match_shared_python_model_fixture
