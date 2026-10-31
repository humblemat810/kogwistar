// Domain-focused tests; shared fixtures live in the parent test module.
use super::*;

#[tokio::test]
#[ignore = "requires KOGWISTAR_TEST_PG_DSN; run explicitly with --include-ignored"]
async fn postgres_integration_event_store_core() -> PostgresStoreResult<()> {
    let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
        .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
    let schema = test_schema("core");
    let store = PostgresStore::from_dsn(&dsn, &schema)?;
    store.ensure_schema().await?;

    // Rust-created schema can be queried through Python-compatible DDL.
    let client = store.client().await?;
    let tables = Tables::new(&schema)?;
    let columns = client
            .query_one(
                "SELECT count(*) FROM information_schema.columns WHERE table_schema = $1 AND table_name = 'entity_events'",
                &[&schema],
            )
            .await
            .map_err(backend)?;
    assert_eq!(columns.try_get::<_, i64>(0).map_err(backend)?, 8);
    drop(client);

    let first = store
        .append_raw_entity_event("one", raw("same", "{\"spaced\": [ 1, 2 ]}"))
        .await?;
    assert_eq!(first.event.seq, 1);
    assert_eq!(first.event.payload_json, "{\"spaced\": [ 1, 2 ]}");
    let retry = store
        .append_raw_entity_event("one", raw("same", "{}"))
        .await?;
    assert!(!retry.inserted);
    assert_eq!(retry.event.seq, 1);
    assert!(matches!(
        store.append_raw_entity_event("two", raw("same", "{}")).await,
        Err(PostgresStoreError::EventIdNamespaceCollision { ref existing_namespace, ref requested_namespace, .. })
            if existing_namespace == "one" && requested_namespace == "two"
    ));

    for id in ["two", "three"] {
        store.append_raw_entity_event("one", raw(id, "{}")).await?;
    }
    assert_eq!(store.replay_raw_events("one", 1, 1).await?.len(), 1);
    assert_eq!(store.latest_retained_event_seq("one").await?, 3);
    assert_eq!(
        EventWriteStore::advance_replay_cursor(&store, "one", "sink", 3)
            .await?
            .last_seq,
        3
    );
    assert!(matches!(
        EventWriteStore::advance_replay_cursor(&store, "one", "sink", 2).await,
        Err(StoreError::CursorRegresses {
            current: 3,
            requested: 2
        })
    ));
    assert!(matches!(
        EventWriteStore::advance_replay_cursor(&store, "one", "sink", 4).await,
        Err(StoreError::CursorOutOfRange {
            cursor: 4,
            latest: 3
        })
    ));
    assert_eq!(
        store
            .set_replay_cursor_legacy("one", "sink", 99)
            .await?
            .last_seq,
        99
    );

    let rollback: PostgresStoreResult<()> = store
        .transaction(|uow| {
            Box::pin(async move {
                uow.append_raw_entity_event("rollback", raw("rollback-id", "{}"))
                    .await?;
                uow.set_replay_cursor_legacy("rollback", "sink", 1).await?;
                Err(PostgresStoreError::TransactionAborted("test".to_owned()))
            })
        })
        .await;
    assert!(matches!(
        rollback,
        Err(PostgresStoreError::TransactionAborted(_))
    ));
    assert_eq!(store.latest_retained_event_seq("rollback").await?, 0);
    assert_eq!(store.replay_cursor("rollback", "sink").await?.last_seq, 0);

    // Cancelling after a write drops the native Transaction and must not
    // return an open transaction to the pool or retain partial state.
    let write_reached = Arc::new(Notify::new());
    let task_store = store.clone();
    let task_reached = Arc::clone(&write_reached);
    let cancelled = tokio::spawn(async move {
        task_store
            .transaction(|uow| {
                Box::pin(async move {
                    uow.append_raw_entity_event("cancelled", raw("cancelled-id", "{}"))
                        .await?;
                    task_reached.notify_one();
                    std::future::pending::<()>().await;
                    #[allow(unreachable_code)]
                    Ok(())
                })
            })
            .await
    });
    write_reached.notified().await;
    cancelled.abort();
    assert!(
        cancelled
            .await
            .expect_err("cancelled transaction completed")
            .is_cancelled()
    );
    assert_eq!(store.latest_retained_event_seq("cancelled").await?, 0);

    let concurrent = (0..8).map(|n| {
        let store = store.clone();
        tokio::spawn(async move {
            store
                .append_raw_entity_event("concurrent", raw(&format!("e-{n}"), "{}"))
                .await
                .map(|row| row.event.seq)
        })
    });
    let mut sequences = Vec::new();
    for task in concurrent {
        sequences.push(task.await.expect("task panicked")?);
    }
    sequences.sort_unstable();
    assert_eq!(sequences, (1..=8).collect::<Vec<_>>());

    // Search path deliberately points elsewhere; all store queries remain isolated.
    let client = store.client().await?;
    client
        .batch_execute("SET search_path TO pg_catalog")
        .await
        .map_err(backend)?;
    drop(client);
    assert_eq!(EventReadStore::latest_event_seq(&store, "one").await?, 3);

    let isolated_schema = test_schema("isolated");
    let isolated = PostgresStore::from_dsn(&dsn, &isolated_schema)?;
    isolated.ensure_schema().await?;
    assert_eq!(isolated.latest_retained_event_seq("one").await?, 0);
    isolated
        .append_raw_entity_event("one", raw("isolated-id", "{}"))
        .await?;
    assert_eq!(isolated.latest_retained_event_seq("one").await?, 1);
    assert_eq!(store.latest_retained_event_seq("one").await?, 3);

    let cleaner = store.client().await?;
    cleaner
        .batch_execute(&format!("DROP SCHEMA {} CASCADE", tables.quoted_schema))
        .await
        .map_err(backend)?;
    let isolated_tables = Tables::new(&isolated_schema)?;
    let cleaner = store.client().await?;
    cleaner
        .batch_execute(&format!(
            "DROP SCHEMA {} CASCADE",
            isolated_tables.quoted_schema
        ))
        .await
        .map_err(backend)?;
    Ok(())
}

#[tokio::test]
#[ignore = "requires KOGWISTAR_TEST_PG_DSN; run explicitly with --include-ignored"]
async fn restored_raw_event_preserves_python_archive_envelope_when_dsn_available()
-> PostgresStoreResult<()> {
    let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
        .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
    let schema = test_schema("restored_event");
    let store = PostgresStore::from_dsn(&dsn, &schema)?;
    store.ensure_schema().await?;
    let namespace = format!("archive-{}", std::process::id());
    let event = RestoredRawEntityEvent {
        seq: 1,
        event_id: "archive-event-1".to_owned(),
        entity_kind: "source".to_owned(),
        entity_id: "source-1".to_owned(),
        op: "UPSERT".to_owned(),
        payload_json: "{ \"title\": \"snow \u{96ea}\" }".to_owned(),
        created_at: 1_700_000_000,
    };

    let first = store
        .append_restored_raw_entity_event(&namespace, event.clone())
        .await?;
    let retry = store
        .append_restored_raw_entity_event(&namespace, event.clone())
        .await?;
    assert!(first.inserted);
    assert!(!retry.inserted);
    assert_eq!(retry.event.seq, event.seq);
    assert_eq!(retry.event.payload_json, event.payload_json);
    for conflicting in [
        RestoredRawEntityEvent {
            seq: 2,
            ..event.clone()
        },
        RestoredRawEntityEvent {
            entity_kind: "edge".to_owned(),
            ..event.clone()
        },
        RestoredRawEntityEvent {
            entity_id: "source-2".to_owned(),
            ..event.clone()
        },
        RestoredRawEntityEvent {
            op: "DELETE".to_owned(),
            ..event.clone()
        },
        RestoredRawEntityEvent {
            payload_json: "{\"different\":true}".to_owned(),
            ..event.clone()
        },
        RestoredRawEntityEvent {
            created_at: event.created_at + 1,
            ..event.clone()
        },
    ] {
        assert!(matches!(
            store
                .append_restored_raw_entity_event(&namespace, conflicting)
                .await,
            Err(PostgresStoreError::Backend(message))
                if message.contains("conflicts with stored event")
        ));
    }
    assert!(matches!(
        store
            .append_restored_raw_entity_event("other-archive", event.clone(),)
            .await,
        Err(PostgresStoreError::EventIdNamespaceCollision { .. })
    ));
    assert!(matches!(
        store
            .append_restored_raw_entity_event(
                &namespace,
                RestoredRawEntityEvent { seq: 3, event_id: "gap".to_owned(), ..event },
            )
            .await,
        Err(PostgresStoreError::Backend(message)) if message.contains("must be 2, got 3")
    ));

    let next = store
        .append_raw_entity_event(&namespace, raw("next", "{}"))
        .await?;
    assert_eq!(next.event.seq, 2);
    let rows = store.replay_raw_events(&namespace, 0, 10).await?;
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].payload_json, "{ \"title\": \"snow \u{96ea}\" }");
    assert_eq!(rows[1].event_id, "next");

    let cleaner = store.client().await?;
    let created_at: i64 = cleaner
        .query_one(
            &format!(
                "SELECT EXTRACT(EPOCH FROM created_at)::BIGINT FROM {} WHERE event_id=$1",
                Tables::new(&schema)?.entity_events
            ),
            &[&event.event_id],
        )
        .await
        .map_err(backend)?
        .try_get(0)
        .map_err(backend)?;
    assert_eq!(created_at, event.created_at);
    cleaner
        .batch_execute(&format!(
            "DROP SCHEMA {} CASCADE",
            Tables::new(&schema)?.quoted_schema
        ))
        .await
        .map_err(backend)?;
    Ok(())
}
