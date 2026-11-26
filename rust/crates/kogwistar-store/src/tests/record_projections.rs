#[test]
fn tool_audit_projection_is_scope_filtered_and_canonical() {
    let mut call = GraphRecord::new("tool-call-1");
    call.metadata
        .insert("entity_type".into(), "tool_call".into());
    call.metadata
        .insert("tool_name".into(), "graph.read".into());
    call.metadata
        .insert("conversation_id".into(), "conversation-1".into());
    call.metadata.insert("turn_index".into(), 2.into());
    call.document = Some(
        serde_json::json!({
            "properties": {"status": "completed", "count": 1}
        })
        .to_string(),
    );

    let mut result = GraphRecord::new("tool-result-1");
    result
        .metadata
        .insert("entity_type".into(), "tool_result".into());
    result
        .metadata
        .insert("tool_name".into(), "graph.read".into());
    result
        .metadata
        .insert("conversation_id".into(), "conversation-1".into());
    result.document = Some(
        serde_json::json!({
            "properties": {"status": "completed"}
        })
        .to_string(),
    );

    let mut other = GraphRecord::new("tool-result-other");
    other
        .metadata
        .insert("entity_type".into(), "tool_result".into());
    other
        .metadata
        .insert("tool_name".into(), "graph.read".into());
    other
        .metadata
        .insert("conversation_id".into(), "conversation-2".into());

    let mut tombstoned = GraphRecord::new("tool-call-deleted");
    tombstoned
        .metadata
        .insert("entity_type".into(), "tool_call".into());
    tombstoned
        .metadata
        .insert("tool_name".into(), "secret".into());
    tombstoned
        .metadata
        .insert("conversation_id".into(), "conversation-1".into());
    tombstoned
        .metadata
        .insert("lifecycle_status".into(), "tombstoned".into());

    let items = tool_audit_items_from_visible_records(
        &[result, call, other, tombstoned],
        Some("conversation-1"),
        10,
    )
    .unwrap();
    assert_eq!(items.len(), 2);
    assert_eq!(items[0].node_id, "tool-call-1");
    assert_eq!(items[0].turn_index, Some(2));
    assert_eq!(items[0].properties["status"], "completed");
    assert_eq!(items[1].node_id, "tool-result-1");
    assert!(
        tool_audit_items_from_visible_records(&[GraphRecord::new("ignored")], None, 0)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn tool_audit_projection_fails_closed_for_malformed_visible_records() {
    let mut malformed = GraphRecord::new("tool-call-invalid");
    malformed
        .metadata
        .insert("entity_type".into(), "tool_call".into());
    malformed
        .metadata
        .insert("conversation_id".into(), "conversation-1".into());
    let error = tool_audit_items_from_visible_records(&[malformed], None, 10).unwrap_err();
    assert_eq!(
        error,
        kogwistar_contracts::ToolAuditContractError::EmptyField { field: "tool_name" }
    );
}

#[test]
fn conversation_transcript_projection_is_ordered_and_scope_filtered() {
    let mut assistant = GraphRecord::new("assistant-1");
    assistant
        .metadata
        .insert("entity_type".into(), "assistant_turn".into());
    assistant
        .metadata
        .insert("conversation_id".into(), "conversation-1".into());
    assistant.metadata.insert("turn_index".into(), 1.into());
    assistant.metadata.insert("role".into(), "assistant".into());
    assistant.metadata.insert("summary".into(), "answer".into());

    let mut user = GraphRecord::new("user-0");
    user.metadata
        .insert("entity_type".into(), "conversation_turn".into());
    user.metadata
        .insert("conversation_id".into(), "conversation-1".into());
    user.metadata.insert("turn_index".into(), 0.into());
    user.metadata.insert("role".into(), "user".into());
    user.metadata.insert("summary".into(), "question".into());

    let mut other = user.clone();
    other.id = "other-user".into();
    other
        .metadata
        .insert("conversation_id".into(), "conversation-2".into());

    let transcript = conversation_transcript_from_visible_records(
        &[assistant, other, user],
        "conversation-1",
        10,
    )
    .unwrap();
    assert_eq!(transcript.turns.len(), 2);
    assert_eq!(transcript.turns[0].node_id, "user-0");
    assert_eq!(transcript.turns[1].node_id, "assistant-1");
}

#[test]
fn conversation_transcript_projection_rejects_duplicate_turn_indexes() {
    let mut first = GraphRecord::new("turn-a");
    first.metadata.extend(
        serde_json::json!({
            "entity_type": "conversation_turn",
            "conversation_id": "conversation-1",
            "turn_index": 0,
            "role": "user",
            "summary": "one"
        })
        .as_object()
        .unwrap()
        .clone(),
    );
    let mut second = first.clone();
    second.id = "turn-b".into();
    let error = conversation_transcript_from_visible_records(&[first, second], "conversation-1", 0)
        .unwrap_err();
    assert_eq!(error, ConversationContractError::InvalidTurnOrder);
}

#[test]
fn conversation_summary_projection_derives_owner_status_and_tail() {
    let mut start = GraphRecord::new("conversation-start");
    start.metadata.extend(
        serde_json::json!({
            "entity_type": "conversation_start",
            "conversation_id": "conversation-1",
            "user_id": "alice",
            "status": "active"
        })
        .as_object()
        .unwrap()
        .clone(),
    );
    let mut turn = GraphRecord::new("turn-0");
    turn.metadata.extend(
        serde_json::json!({
            "entity_type": "conversation_turn",
            "conversation_id": "conversation-1",
            "turn_index": 0,
            "role": "user",
            "summary": "question"
        })
        .as_object()
        .unwrap()
        .clone(),
    );
    let summary =
        conversation_summary_from_visible_records(&[start, turn], "conversation-1").unwrap();
    assert_eq!(summary.user_id, "alice");
    assert_eq!(summary.start_node_id, "conversation-start");
    assert_eq!(summary.tail_node_id, "turn-0");
    assert_eq!(summary.turn_count, 1);
}

#[test]
fn conversation_list_projection_is_user_scoped_and_counts_visible_turns() {
    let mut start_alice = GraphRecord::new("start-alice");
    start_alice.metadata.extend(
        serde_json::json!({
            "entity_type": "conversation_start",
            "conversation_id": "conversation-alice",
            "user_id": "alice",
            "status": "active"
        })
        .as_object()
        .unwrap()
        .clone(),
    );
    let mut start_bob = start_alice.clone();
    start_bob.id = "start-bob".to_owned();
    start_bob
        .metadata
        .insert("conversation_id".to_owned(), "conversation-bob".into());
    start_bob
        .metadata
        .insert("user_id".to_owned(), "bob".into());
    let mut first_turn = GraphRecord::new("turn-1");
    first_turn.metadata.extend(
        serde_json::json!({
            "entity_type": "conversation_turn",
            "conversation_id": "conversation-alice",
            "turn_index": 0
        })
        .as_object()
        .unwrap()
        .clone(),
    );
    let mut second_turn = first_turn.clone();
    second_turn.id = "turn-2".to_owned();
    second_turn
        .metadata
        .insert("entity_type".to_owned(), "assistant_turn".into());
    second_turn
        .metadata
        .insert("turn_index".to_owned(), 1.into());

    let records = [start_alice, start_bob, first_turn, second_turn];
    let entries = conversation_list_for_user_from_visible_records(&records, "alice");
    assert_eq!(
        entries,
        vec![ConversationListEntry {
            id: "conversation-alice".to_owned(),
            start_node_id: "start-alice".to_owned(),
            status: "active".to_owned(),
            turn_count: 2,
        }]
    );

    let mut chunked = ConversationListAccumulator::new("alice");
    chunked.extend_visible_records(&records[..1]);
    chunked.extend_visible_records(&records[1..3]);
    chunked.extend_visible_records(&records[3..]);
    assert_eq!(chunked.finish(), entries);
}

#[test]
fn context_snapshot_projection_selects_latest_matching_scope() {
    let mut older = GraphRecord::new("snapshot-old");
    older
        .metadata
        .insert("entity_type".into(), "context_snapshot".into());
    older
        .metadata
        .insert("conversation_id".into(), "conversation-1".into());
    older.metadata.insert("run_id".into(), "run-1".into());
    older.metadata.insert("stage".into(), "answer".into());
    older.metadata.insert("snapshot_seq".into(), 3.into());
    older.document = Some(
        serde_json::json!({
            "metadata": {"stage": "answer"},
            "properties": {"value": "old"}
        })
        .to_string(),
    );

    let mut newer = older.clone();
    newer.id = "snapshot-new".into();
    newer.metadata.insert("snapshot_seq".into(), 4.into());
    newer.document = Some(
        serde_json::json!({
            "metadata": {"stage": "answer"},
            "properties": {"value": "new"}
        })
        .to_string(),
    );

    let snapshot = context_snapshot_from_visible_records(
        &[older, newer],
        "conversation-1",
        Some("run-1"),
        Some("answer"),
    )
    .unwrap();
    assert_eq!(snapshot.snapshot_node_id, "snapshot-new");
    assert_eq!(snapshot.properties["value"], "new");
}

#[test]
fn context_snapshot_projection_matches_python_run_and_attempt_order_fixture() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/context-snapshot-order-v1.json"
    )))
    .expect("valid shared Python context-snapshot ordering fixture");
    let conversation_id = fixture["conversation_id"].as_str().unwrap();
    let run_id = fixture["run_id"].as_str().unwrap();
    let stage = fixture["stage"].as_str().unwrap();
    let records = fixture["candidates"]
        .as_array()
        .unwrap()
        .iter()
        .map(|candidate| {
            let id = candidate["id"].as_str().unwrap().to_owned();
            let metadata = serde_json::json!({
                "entity_type": "context_snapshot",
                "conversation_id": conversation_id,
                "run_id": run_id,
                "stage": stage,
                "run_step_seq": candidate["run_step_seq"],
                "attempt_seq": candidate["attempt_seq"]
            })
            .as_object()
            .unwrap()
            .clone();
            GraphRecord {
                id,
                document: Some(
                    serde_json::json!({
                        "metadata": metadata,
                        "properties": candidate["properties"]
                    })
                    .to_string(),
                ),
                metadata,
                embedding: None,
            }
        })
        .collect::<Vec<_>>();

    let snapshot =
        context_snapshot_from_visible_records(&records, conversation_id, Some(run_id), Some(stage))
            .expect("Python run_step_seq and attempt_seq metadata is orderable");
    assert_eq!(
        snapshot.snapshot_node_id,
        fixture["expected_snapshot_node_id"].as_str().unwrap()
    );
    assert_eq!(snapshot.properties["prompt_messages"], "latest retry");
}

#[test]
fn context_snapshot_projection_rejects_unordered_candidates() {
    let mut record = GraphRecord::new("snapshot-invalid");
    record
        .metadata
        .insert("entity_type".into(), "context_snapshot".into());
    record
        .metadata
        .insert("conversation_id".into(), "conversation-1".into());
    let error =
        context_snapshot_from_visible_records(&[record], "conversation-1", None, None).unwrap_err();
    assert_eq!(error, ConversationContractError::MissingSnapshotOrder);
}
use kogwistar_contracts::{GraphModelClass, GraphModelEnvelope};
