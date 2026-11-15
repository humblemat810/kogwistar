use super::*;

fn acl_event(event_id: &str, target: &str, truth_graph: &str, op: &str) -> NewRawEntityEvent {
    NewRawEntityEvent {
        event_id: event_id.to_owned(),
        entity_kind: "node".to_owned(),
        entity_id: format!("acl-{target}"),
        op: op.to_owned(),
        payload_json: serde_json::json!({
            "id": format!("acl-{target}"),
            "metadata": {
                "entity_type": "acl_record",
                "acl_truth_graph": truth_graph,
                "acl_target_entity_id": target,
                "acl_target_grain": "node",
                "acl_mode": "private",
                "tombstoned": false
            },
            "properties": {"shared_with_principals": []}
        })
        .to_string(),
    }
}

fn scope(namespace: &str) -> kogwistar_store::GraphScope {
    kogwistar_store::GraphScope {
        namespace: namespace.to_owned(),
        workspace_id: None,
        graph_space: None,
    }
}

#[test]
fn acl_records_are_read_from_canonical_events_by_scope_and_target() {
    let (store, path) = store("acl-canonical-records");
    store
        .immediate_transaction(|uow| {
            uow.append_raw_entity_event(
                "conversation",
                acl_event("acl-1", "node-1", "conversation", "ADD"),
            )?;
            uow.append_raw_entity_event(
                "conversation",
                acl_event("acl-2", "node-2", "conversation", "ADD"),
            )?;
            uow.append_raw_entity_event(
                "conversation",
                acl_event("acl-3", "node-1", "wisdom", "ADD"),
            )?;
            uow.append_raw_entity_event(
                "other",
                acl_event("acl-4", "node-1", "conversation", "ADD"),
            )?;
            Ok(())
        })
        .unwrap();

    let records = store
        .immediate_transaction(|uow| {
            uow.graph_acl_records_for_entities(
                &scope("conversation"),
                "conversation",
                &["node-1".to_owned()],
            )
        })
        .unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].id, "acl-node-1");
    assert_eq!(records[0].metadata["acl_truth_graph"], "conversation");
    assert_eq!(
        records[0].metadata["shared_with_principals"],
        serde_json::json!([])
    );

    drop(store);
    let _ = fs::remove_file(path);
}

#[test]
fn acl_record_lookup_marks_tombstoned_canonical_events() {
    let (store, path) = store("acl-canonical-tombstone");
    store
        .immediate_transaction(|uow| {
            uow.append_raw_entity_event(
                "conversation",
                acl_event("acl-add", "node-1", "conversation", "ADD"),
            )?;
            uow.append_raw_entity_event(
                "conversation",
                acl_event("acl-delete", "node-1", "conversation", "TOMBSTONE"),
            )?;
            Ok(())
        })
        .unwrap();
    let records = store
        .immediate_transaction(|uow| {
            uow.graph_acl_records_for_entities(
                &scope("conversation"),
                "conversation",
                &["node-1".to_owned()],
            )
        })
        .unwrap();
    assert_eq!(records.len(), 2);
    assert!(!records[0].metadata["tombstoned"].as_bool().unwrap());
    assert!(records[1].metadata["tombstoned"].as_bool().unwrap());

    drop(store);
    let _ = fs::remove_file(path);
}

#[test]
fn acl_record_lookup_respects_explicit_workspace_and_graph_space_scope() {
    let (store, path) = store("acl-canonical-storage-scope");
    let mut matching = acl_event("acl-matching", "node-1", "conversation", "ADD");
    matching.payload_json = serde_json::json!({
        "id": "acl-node-1",
        "metadata": {
            "entity_type": "acl_record",
            "acl_truth_graph": "conversation",
            "acl_target_entity_id": "node-1",
            "acl_target_grain": "node",
            "workspace_id": "workspace-a",
            "graph_space": "dialogue"
        }
    })
    .to_string();
    let mut other_workspace = acl_event("acl-other-workspace", "node-1", "conversation", "ADD");
    other_workspace.payload_json = serde_json::json!({
        "id": "acl-node-1-other",
        "metadata": {
            "entity_type": "acl_record",
            "acl_truth_graph": "conversation",
            "acl_target_entity_id": "node-1",
            "acl_target_grain": "node",
            "workspace_id": "workspace-b",
            "graph_space": "dialogue"
        }
    })
    .to_string();
    store
        .immediate_transaction(|uow| {
            uow.append_raw_entity_event("conversation", matching)?;
            uow.append_raw_entity_event("conversation", other_workspace)?;
            Ok(())
        })
        .unwrap();

    let scoped = kogwistar_store::GraphScope {
        namespace: "conversation".to_owned(),
        workspace_id: Some("workspace-a".to_owned()),
        graph_space: Some("dialogue".to_owned()),
    };
    let records = store
        .immediate_transaction(|uow| {
            uow.graph_acl_records_for_entities(&scoped, "conversation", &["node-1".to_owned()])
        })
        .unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].metadata["workspace_id"], "workspace-a");

    drop(store);
    let _ = fs::remove_file(path);
}
