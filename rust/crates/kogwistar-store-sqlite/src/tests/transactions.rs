// Domain-focused tests; shared fixtures live in the parent test module.
use super::*;

#[test]
fn immediate_uow_commits_and_rolls_back_sequences_events_and_cursors_together() {
    let (store, path) = store("uow");
    store
        .immediate_transaction(|uow| {
            assert_eq!(uow.next_global_seq()?, 1);
            assert_eq!(
                uow.append_raw_entity_event("ns", raw_event("committed", "{}"))?
                    .event
                    .seq,
                1
            );
            uow.strict_advance_replay_cursor("ns", "sink", 1)?;
            Ok(())
        })
        .unwrap();
    let rollback: SqliteStoreResult<()> = store.immediate_transaction(|uow| {
        assert_eq!(uow.next_global_seq()?, 2);
        assert_eq!(
            uow.append_raw_entity_event("ns", raw_event("rolled-back", "{}"))?
                .event
                .seq,
            2
        );
        uow.strict_advance_replay_cursor("ns", "sink", 2)?;
        Err(SqliteStoreError::TransactionAborted(
            "test rollback".to_owned(),
        ))
    });
    assert!(matches!(
        rollback,
        Err(SqliteStoreError::TransactionAborted(_))
    ));
    assert_eq!(store.current_global_seq().unwrap(), 1);
    assert_eq!(store.latest_retained_event_seq("ns").unwrap(), 1);
    assert_eq!(store.replay_cursor("ns", "sink").unwrap().last_seq, 1);
    assert!(
        store
            .replay_raw_events("ns", 0, 10)
            .unwrap()
            .iter()
            .all(|event| event.event_id != "rolled-back")
    );
    drop(store);
    fs::remove_file(path).unwrap();
}
#[test]
fn external_transaction_joins_operations_and_commits_or_rolls_back() {
    let (store, path) = store("external-uow");
    store.begin_external_transaction().unwrap();
    assert_eq!(store.next_global_seq().unwrap(), 1);
    store
        .append_raw_entity_event("ns", raw_event("commit", "{}"))
        .unwrap();
    store.commit_external_transaction().unwrap();
    assert_eq!(store.current_global_seq().unwrap(), 1);
    assert_eq!(store.replay_raw_events("ns", 0, 10).unwrap().len(), 1);

    store.begin_external_transaction().unwrap();
    assert_eq!(store.next_global_seq().unwrap(), 2);
    store
        .append_raw_entity_event("ns", raw_event("rollback", "{}"))
        .unwrap();
    store.rollback_external_transaction().unwrap();
    assert_eq!(store.current_global_seq().unwrap(), 1);
    assert_eq!(store.replay_raw_events("ns", 0, 10).unwrap().len(), 1);

    drop(store);
    fs::remove_file(path).unwrap();
}
#[test]
fn reopen_is_durable_and_concurrent_global_allocations_are_serialized() {
    let (store, path) = store("durable");
    store.next_global_seq().unwrap();
    store
        .append_raw_entity_event("ns", raw_event("durable", "{\"kept\":true}"))
        .unwrap();
    drop(store);
    let reopened = SqliteStore::open(&path).unwrap();
    assert_eq!(reopened.current_global_seq().unwrap(), 1);
    assert_eq!(
        reopened.replay_raw_events("ns", 0, 1).unwrap()[0].payload_json,
        "{\"kept\":true}"
    );

    let workers = 6;
    let barrier = Arc::new(Barrier::new(workers));
    let mut handles = Vec::new();
    for _ in 0..workers {
        let store = reopened.clone();
        let barrier = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            barrier.wait();
            store.next_global_seq().unwrap()
        }));
    }
    let mut issued = handles
        .into_iter()
        .map(|handle| handle.join().unwrap())
        .collect::<Vec<_>>();
    issued.sort_unstable();
    assert_eq!(issued, [2, 3, 4, 5, 6, 7]);
    drop(reopened);
    fs::remove_file(path).unwrap();
}
