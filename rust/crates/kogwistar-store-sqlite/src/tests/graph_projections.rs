// Domain-focused tests; shared fixtures live in the parent test module.
use super::*;

#[test]
fn stage1_node_projection_isolated_crud() {
    let (store, path) = store("stage1-crud");
    let projection = NamedProjectionWrite {
        payload: serde_json::json!({
            "id": "node-1",
            "metadata": {"kind": "person"}
        })
        .as_object()
        .unwrap()
        .clone(),
        last_authoritative_seq: 7,
        last_materialized_seq: 7,
        projection_schema_version: 1,
        materialization_status: "pending".to_owned(),
    };
    store
        .replace_stage1_node_projection("tenant-a", "node-1", projection)
        .unwrap();
    assert_eq!(
        store
            .get_stage1_node_projection("tenant-a", "node-1")
            .unwrap()
            .unwrap()
            .payload["metadata"]["kind"],
        "person"
    );
    assert_eq!(
        store
            .list_stage1_node_projections("tenant-a")
            .unwrap()
            .len(),
        1
    );
    assert!(
        store
            .get_named_projection("tenant-a", "node-1")
            .unwrap()
            .is_none()
    );
    store
        .clear_stage1_node_projection("tenant-a", "node-1")
        .unwrap();
    assert!(
        store
            .get_stage1_node_projection("tenant-a", "node-1")
            .unwrap()
            .is_none()
    );
    drop(store);
    fs::remove_file(path).unwrap();
}

#[test]
fn stage1_conditional_clear_preserves_a_newer_authoritative_revision() {
    let (store, path) = store("stage1-conditional-clear");
    let write = |seq| NamedProjectionWrite {
        payload: serde_json::json!({"revision": seq})
            .as_object()
            .unwrap()
            .clone(),
        last_authoritative_seq: seq,
        last_materialized_seq: seq,
        projection_schema_version: 1,
        materialization_status: "pending".to_owned(),
    };
    store
        .replace_stage1_node_projection("tenant-a", "node-1", write(7))
        .unwrap();
    store
        .replace_stage1_node_projection("tenant-a", "node-1", write(8))
        .unwrap();

    assert!(
        !store
            .clear_stage1_node_projection_if_current("tenant-a", "node-1", 7)
            .unwrap()
    );
    assert_eq!(
        store
            .get_stage1_node_projection("tenant-a", "node-1")
            .unwrap()
            .unwrap()
            .last_authoritative_seq,
        8
    );
    assert!(
        store
            .clear_stage1_node_projection_if_current("tenant-a", "node-1", 8)
            .unwrap()
    );
    assert!(
        store
            .get_stage1_node_projection("tenant-a", "node-1")
            .unwrap()
            .is_none()
    );
    drop(store);
    fs::remove_file(path).unwrap();
}
#[test]
fn named_projection_batch_cas_is_all_or_nothing() {
    let (store, path) = store("named-projection-batch-cas");
    let projection = |value: i64| NamedProjectionWrite {
        payload: serde_json::json!({"value": value})
            .as_object()
            .unwrap()
            .clone(),
        last_authoritative_seq: value,
        last_materialized_seq: value,
        projection_schema_version: 1,
        materialization_status: "ready".to_owned(),
    };

    assert!(
        !store
            .compare_and_swap_named_projections(vec![
                NamedProjectionCas {
                    namespace: "skills".to_owned(),
                    key: "provider".to_owned(),
                    expected_last_authoritative_seq: None,
                    expected_last_materialized_seq: None,
                    projection: projection(1),
                },
                NamedProjectionCas {
                    namespace: "catalog".to_owned(),
                    key: "provider".to_owned(),
                    expected_last_authoritative_seq: Some(99),
                    expected_last_materialized_seq: Some(99),
                    projection: projection(1),
                },
            ])
            .unwrap()
    );
    assert!(
        store
            .get_named_projection("skills", "provider")
            .unwrap()
            .is_none()
    );

    assert!(
        store
            .compare_and_swap_named_projections(vec![
                NamedProjectionCas {
                    namespace: "skills".to_owned(),
                    key: "provider".to_owned(),
                    expected_last_authoritative_seq: None,
                    expected_last_materialized_seq: None,
                    projection: projection(1),
                },
                NamedProjectionCas {
                    namespace: "catalog".to_owned(),
                    key: "provider".to_owned(),
                    expected_last_authoritative_seq: None,
                    expected_last_materialized_seq: None,
                    projection: projection(1),
                },
            ])
            .unwrap()
    );
    assert_eq!(
        store
            .get_named_projection("skills", "provider")
            .unwrap()
            .unwrap()
            .payload["value"],
        1
    );
    assert_eq!(
        store
            .get_named_projection("catalog", "provider")
            .unwrap()
            .unwrap()
            .payload["value"],
        1
    );
    drop(store);
    fs::remove_file(path).unwrap();
}
