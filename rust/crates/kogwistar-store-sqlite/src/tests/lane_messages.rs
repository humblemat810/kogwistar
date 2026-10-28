// Domain-focused tests; shared fixtures live in the parent test module.
use super::*;

#[test]
fn targeted_lane_claim_does_not_take_earlier_other_run() {
    let (store, path) = store("targeted-runtime-claim");
    store
        .project_lane_message(runtime_worker_lane("run-decoy", "lane-decoy"))
        .unwrap();
    store
        .project_lane_message(runtime_worker_lane("run-target", "lane-target"))
        .unwrap();

    let claimed = store
        .immediate_transaction(|uow| {
            uow.claim_projected_lane_messages_for_run(
                "runtime",
                "python-workers",
                "run-target",
                "target-worker",
                1,
                60,
            )
        })
        .unwrap();

    assert_eq!(claimed.len(), 1);
    assert_eq!(claimed[0].message_id, "lane-target");
    assert_eq!(
        store
            .get_projected_lane_message("lane-decoy")
            .unwrap()
            .unwrap()
            .status,
        "pending"
    );
    drop(store);
    fs::remove_file(path).unwrap();
}
#[test]
fn filtered_lane_claim_is_atomic_for_run_kind_recipient_and_ids() {
    let (store, path) = store("filtered-control-claim");
    store
        .project_lane_message(runtime_worker_lane("run-target", "lane-good"))
        .unwrap();
    let mut wrong_kind = runtime_worker_lane("run-target", "lane-kind");
    wrong_kind.msg_type = "agent.queue".to_owned();
    store.project_lane_message(wrong_kind).unwrap();
    let mut wrong_recipient = runtime_worker_lane("run-target", "lane-recipient");
    wrong_recipient.recipient_id = "other-agent".to_owned();
    store.project_lane_message(wrong_recipient).unwrap();
    let wrong_run = runtime_worker_lane("run-other", "lane-run");
    store.project_lane_message(wrong_run).unwrap();

    let claimed = store
        .claim_projected_lane_messages_filtered(LaneMessageClaimFilter {
            namespace: "runtime".to_owned(),
            inbox_id: "python-workers".to_owned(),
            claimed_by: "target-worker".to_owned(),
            message_ids: Some(vec![
                "lane-good".to_owned(),
                "lane-kind".to_owned(),
                "lane-recipient".to_owned(),
                "lane-run".to_owned(),
            ]),
            run_id: Some("run-target".to_owned()),
            msg_type: Some("workflow.worker.request.v1".to_owned()),
            recipient_id: Some("python-worker".to_owned()),
            limit: 10,
            lease_seconds: 60,
        })
        .unwrap();

    assert_eq!(
        claimed
            .iter()
            .map(|row| row.message_id.as_str())
            .collect::<Vec<_>>(),
        ["lane-good"]
    );
    for message_id in ["lane-kind", "lane-recipient", "lane-run"] {
        assert_eq!(
            store
                .get_projected_lane_message(message_id)
                .unwrap()
                .unwrap()
                .status,
            "pending"
        );
    }
    drop(store);
    fs::remove_file(path).unwrap();
}
#[test]
fn expired_lane_claim_cannot_be_finalized_before_reclaim() {
    let (store, path) = store("expired-lane-claim");
    for (message_id, operation) in [
        ("expired-ack", "ack"),
        ("expired-requeue", "requeue"),
        ("expired-dead-letter", "dead-letter"),
    ] {
        let mut row = runtime_worker_lane("run-target", message_id);
        row.available_at = 0;
        store.project_lane_message(row).unwrap();
        let claimed = store
            .claim_projected_lane_messages_filtered(LaneMessageClaimFilter {
                namespace: "runtime".to_owned(),
                inbox_id: "python-workers".to_owned(),
                claimed_by: "attempt-1".to_owned(),
                message_ids: Some(vec![message_id.to_owned()]),
                run_id: Some("run-target".to_owned()),
                msg_type: Some("workflow.worker.request.v1".to_owned()),
                recipient_id: Some("python-worker".to_owned()),
                limit: 1,
                lease_seconds: -1,
            })
            .unwrap();
        assert_eq!(claimed.len(), 1);
        assert!(claimed[0].lease_until.as_ref().unwrap().as_i64().unwrap() < unix_epoch_seconds());

        match operation {
            "ack" => store
                .ack_projected_lane_message(message_id, "attempt-1")
                .unwrap(),
            "requeue" => store
                .requeue_projected_lane_message(message_id, "attempt-1", None, 0)
                .unwrap(),
            "dead-letter" => store
                .dead_letter_projected_lane_message(message_id, "attempt-1", None)
                .unwrap(),
            _ => unreachable!(),
        }
        let row = store
            .get_projected_lane_message(message_id)
            .unwrap()
            .unwrap();
        assert_eq!(row.status, "claimed", "{operation}");
        assert_eq!(row.claimed_by.as_deref(), Some("attempt-1"), "{operation}");
        assert_eq!(row.retry_count, 0, "{operation}");
    }
    drop(store);
    fs::remove_file(path).unwrap();
}
