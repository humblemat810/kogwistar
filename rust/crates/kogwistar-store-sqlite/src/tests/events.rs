// Domain-focused tests; shared fixtures live in the parent test module.
use super::*;

#[test]
fn restored_raw_event_preserves_identity_and_is_idempotent() {
    let (store, path) = store("restored-raw-event");
    let event = RestoredRawEntityEvent {
        seq: 1,
        event_id: "archive-event".to_owned(),
        entity_kind: "node".to_owned(),
        entity_id: "node-1".to_owned(),
        op: "UPSERT".to_owned(),
        payload_json: "{\"value\":1}".to_owned(),
        created_at: 123,
    };
    assert!(
        store
            .append_restored_raw_entity_event("archive", event.clone())
            .unwrap()
            .inserted
    );
    assert!(
        !store
            .append_restored_raw_entity_event("archive", event.clone())
            .unwrap()
            .inserted
    );
    assert_eq!(
        store.replay_raw_events("archive", 0, 10).unwrap()[0].created_at,
        123
    );
    assert!(
            store
                .append_restored_raw_entity_event(
                    "archive",
                    RestoredRawEntityEvent { seq: 3, ..event },
                )
                .is_err()
        );
    drop(store);
    fs::remove_file(path).unwrap();
}
#[test]
fn global_user_and_scoped_sequences_are_isolated_and_current() {
    let (store, path) = store("sequences");
    assert_eq!(store.current_global_seq().unwrap(), 0);
    assert_eq!(store.next_global_seq().unwrap(), 1);
    assert_eq!(store.next_global_seq().unwrap(), 2);
    assert_eq!(store.current_user_seq("alice").unwrap(), 0);
    assert_eq!(store.next_user_seq("alice").unwrap(), 1);
    assert_eq!(store.next_user_seq("alice").unwrap(), 2);
    assert_eq!(store.next_scoped_seq("project-7").unwrap(), 1);
    assert_eq!(store.current_scoped_seq("project-7").unwrap(), 1);
    assert_eq!(store.current_user_seq("alice").unwrap(), 2);
    store.set_scoped_seq("project-7", 11).unwrap();
    assert_eq!(store.next_user_seq("project-7").unwrap(), 12);
    assert!(matches!(
        store.set_user_seq("alice", -1),
        Err(SqliteStoreError::NegativeSequenceValue { value: -1 })
    ));
    drop(store);
    fs::remove_file(path).unwrap();
}
#[test]
fn append_replay_idempotency_and_cross_namespace_collision_follow_contract() {
    let (store, path) = store("events");
    let first = block_on(EventWriteStore::append_entity_event(
        &store,
        "one",
        event("same", "one"),
    ))
    .unwrap();
    let second = block_on(EventWriteStore::append_entity_event(
        &store,
        "one",
        event("second", "two"),
    ))
    .unwrap();
    let retry = block_on(EventWriteStore::append_entity_event(
        &store,
        "one",
        event("same", "changed"),
    ))
    .unwrap();
    assert_eq!((first.event.seq, second.event.seq), (1, 2));
    assert!(first.inserted);
    assert!(!retry.inserted);
    assert_eq!(retry.event.entity_id, "one");
    assert_eq!(
        block_on(EventReadStore::replay_events(&store, "one", 1, 10))
            .unwrap()
            .iter()
            .map(|event| event.seq)
            .collect::<Vec<_>>(),
        [2]
    );
    assert!(matches!(
        store.append_raw_entity_event("two", raw_event("same", "{}")),
        Err(SqliteStoreError::EventIdNamespaceCollision { ref existing_namespace, .. }) if existing_namespace == "one"
    ));
    assert_eq!(
        block_on(EventWriteStore::append_entity_event(
            &store,
            "two",
            event("same", "changed"),
        )),
        Err(StoreError::EventIdNamespaceCollision {
            event_id: "same".to_owned(),
            existing_namespace: "one".to_owned(),
            requested_namespace: "two".to_owned(),
        })
    );
    assert_eq!(store.alloc_event_seq("gap").unwrap(), 1);
    assert_eq!(
        block_on(EventReadStore::latest_event_seq(&store, "gap")).unwrap(),
        0
    );
    drop(store);
    fs::remove_file(path).unwrap();
}
#[test]
fn cursor_trait_is_strict_while_legacy_set_is_permissive() {
    let (store, path) = store("cursors");
    block_on(EventWriteStore::append_entity_event(
        &store,
        "ns",
        event("one", "one"),
    ))
    .unwrap();
    assert_eq!(
        block_on(EventWriteStore::advance_replay_cursor(
            &store, "ns", "sink", 1
        ))
        .unwrap()
        .last_seq,
        1
    );
    assert_eq!(
        block_on(EventWriteStore::advance_replay_cursor(
            &store, "ns", "sink", 0
        )),
        Err(StoreError::CursorRegresses {
            current: 1,
            requested: 0
        })
    );
    assert_eq!(
        block_on(EventWriteStore::advance_replay_cursor(
            &store, "ns", "sink", 2
        )),
        Err(StoreError::CursorOutOfRange {
            cursor: 2,
            latest: 1
        })
    );
    assert_eq!(
        store
            .set_replay_cursor_legacy("ns", "sink", 99)
            .unwrap()
            .last_seq,
        99
    );
    assert_eq!(
        store
            .set_replay_cursor_legacy("ns", "sink", -7)
            .unwrap()
            .last_seq,
        -7
    );
    drop(store);
    fs::remove_file(path).unwrap();
}
#[test]
fn arbitrary_precision_event_payload_survives_reopen_and_typed_replay() {
    let (store, path) = store("arbitrary-precision-event");
    let integer = "1606938044258990275541962092341162602522202993782792835301376";
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../../contracts/golden/python-graph-model-payloads-v1.json"
    ))
    .unwrap();
    let payload = fixture["cases"][0]["canonical_json"]
        .as_str()
        .expect("shared Python node fixture")
        .replace(
            "\"level_from_root\":0",
            &format!("\"level_from_root\":{integer}"),
        );
    store
        .append_raw_entity_event("ns", raw_event("large-level", &payload))
        .unwrap();
    drop(store);

    let reopened = SqliteStore::open(&path).unwrap();
    let raw = reopened.replay_raw_events("ns", 0, 1).unwrap();
    assert_eq!(raw[0].payload_json, payload);

    let replayed = block_on(EventReadStore::replay_events(&reopened, "ns", 0, 1)).unwrap();
    assert_eq!(
        serde_json::to_string(&replayed[0].payload).unwrap(),
        payload
    );
    let envelope_wire =
        format!(r#"{{"schema_version":1,"model_class":"node","payload":{payload}}}"#);
    let decoded = kogwistar_contracts::RawGraphModelEnvelopeV1::from_json(&envelope_wire)
        .unwrap()
        .decode_persistent_v2()
        .expect("replayed Python-shaped node retains its exact typed hierarchy level");
    assert_eq!(
        decoded
            .level_from_root
            .as_ref()
            .map(|level| level.as_decimal_str()),
        Some(integer)
    );
    drop(reopened);
    fs::remove_file(path).unwrap();
}
