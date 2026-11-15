// Domain-focused tests; shared fixtures live in the parent test module.
use super::*;

#[test]
fn expired_claim_token_cannot_mutate_queue_state() {
    let (store, path) = store("expired-index-claim");
    for job_id in ["done", "failed", "retried", "requeued"] {
        store
            .enqueue_index_job(NewIndexJob {
                job_id: job_id.to_owned(),
                namespace: "ns".to_owned(),
                entity_kind: "node".to_owned(),
                entity_id: job_id.to_owned(),
                index_kind: "test".to_owned(),
                op: "UPSERT".to_owned(),
                payload_json: None,
                max_retries: 3,
            })
            .unwrap();
    }
    let claims = store.claim_index_jobs(4, -1, Some("ns")).unwrap();
    let tokens = claims
        .iter()
        .map(|claim| (claim.job_id.as_str(), claim.claim_token.as_deref().unwrap()))
        .collect::<BTreeMap<_, _>>();

    assert!(
        !store
            .mark_index_job_done("done", Some(tokens["done"]))
            .unwrap()
    );
    store
        .mark_index_job_failed("failed", "late", true, Some(tokens["failed"]))
        .unwrap();
    store
        .bump_retry_and_requeue("retried", "late", 0, Some(tokens["retried"]))
        .unwrap();
    store
        .requeue_index_job_at_tail("requeued", "{}".to_owned(), 0, Some(tokens["requeued"]))
        .unwrap();

    let reclaimed = store.claim_index_jobs(4, 30, Some("ns")).unwrap();
    assert_eq!(reclaimed.len(), 4);
    assert!(reclaimed.iter().all(|claim| claim.retry_count == 0));
    assert!(
        reclaimed
            .iter()
            .all(|claim| { claim.claim_token.as_deref() != Some(tokens[claim.job_id.as_str()]) })
    );
    assert!(store.mark_index_job_done("done", None).unwrap());
    drop(store);
    fs::remove_file(path).unwrap();
}
#[test]
fn accepted_index_job_result_is_first_live_claim_wins() {
    let (store, path) = store("accepted-result");
    store
        .enqueue_index_job(NewIndexJob {
            job_id: "candidate-job".to_owned(),
            namespace: "maintenance".to_owned(),
            entity_kind: "maintenance".to_owned(),
            entity_id: "doc-1".to_owned(),
            index_kind: "parse".to_owned(),
            op: "UPSERT".to_owned(),
            payload_json: Some("{}".to_owned()),
            max_retries: 3,
        })
        .unwrap();
    let claim = store
        .claim_index_jobs(1, 30, Some("maintenance"))
        .unwrap()
        .remove(0);
    let accepted = store
        .accept_index_job_result(
            &claim.job_id,
            claim.claim_token.as_deref().unwrap(),
            "{\"winner\":1}",
            "digest-1",
        )
        .unwrap();
    assert_eq!(accepted.status, "accepted");
    let existing = store
        .accept_index_job_result(&claim.job_id, "stale-worker", "{\"winner\":2}", "digest-2")
        .unwrap();
    assert_eq!(existing.status, "existing");
    assert_eq!(existing.result_json.as_deref(), Some("{\"winner\":1}"));
    drop(store);
    fs::remove_file(path).unwrap();
}

#[test]
fn filtered_claim_is_atomic_scoped_and_leaves_other_job_kinds_pending() {
    let (store, path) = store("filtered-index-claim");
    for (job_id, namespace, index_kind) in [
        ("embedding-a", "tenant-a", "node_embedding"),
        ("metadata-a", "tenant-a", "node_refs"),
        ("embedding-b", "tenant-b", "node_embedding"),
    ] {
        store
            .enqueue_index_job(NewIndexJob {
                job_id: job_id.to_owned(),
                namespace: namespace.to_owned(),
                entity_kind: "node".to_owned(),
                entity_id: job_id.to_owned(),
                index_kind: index_kind.to_owned(),
                op: "UPSERT".to_owned(),
                payload_json: None,
                max_retries: 3,
            })
            .unwrap();
    }

    let claims = block_on(IndexJobWriteStore::claim_index_jobs_filtered(
        &store,
        10,
        30,
        IndexJobClaimFilter {
            namespace: Some("tenant-a".to_owned()),
            index_kinds: vec!["node_embedding".to_owned()],
        },
    ))
    .unwrap();
    assert_eq!(
        claims
            .iter()
            .map(|job| job.job_id.as_str())
            .collect::<Vec<_>>(),
        ["embedding-a"]
    );

    let pending = store
        .list_index_jobs(Some("tenant-a"), Some("PENDING"), None, None, None, 10)
        .unwrap();
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].job_id, "metadata-a");

    let unfiltered_other_kind = store.claim_index_jobs(10, 30, Some("tenant-a")).unwrap();
    assert_eq!(unfiltered_other_kind.len(), 1);
    assert_eq!(unfiltered_other_kind[0].job_id, "metadata-a");
    drop(store);
    fs::remove_file(path).unwrap();
}

#[test]
fn filtered_claim_rejects_empty_or_blank_index_kinds() {
    let (store, path) = store("filtered-index-claim-invalid");
    for index_kinds in [Vec::new(), vec!["  ".to_owned()]] {
        assert!(
            store
                .claim_index_jobs_filtered(
                    1,
                    30,
                    IndexJobClaimFilter {
                        namespace: None,
                        index_kinds,
                    },
                )
                .is_err()
        );
    }
    drop(store);
    fs::remove_file(path).unwrap();
}
