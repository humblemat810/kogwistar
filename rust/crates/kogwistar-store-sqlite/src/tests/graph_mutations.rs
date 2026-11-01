use super::*;
use kogwistar_store::{GraphMutation, GraphMutationBatch, GraphRecord, GraphScope};

fn mutation(event_id: &str, id: &str) -> GraphMutation {
    GraphMutation {
        scope: GraphScope {
            namespace: "conversation".to_owned(),
            workspace_id: Some("workspace-a".to_owned()),
            graph_space: Some("conversation".to_owned()),
        },
        table: "gke_nodes".to_owned(),
        entity_kind: "node".to_owned(),
        event_id: event_id.to_owned(),
        op: "ADD".to_owned(),
        payload: serde_json::json!({"id": id, "label": id, "metadata": {"engine_type": "conversation"}}),
        record: GraphRecord {
            id: id.to_owned(),
            document: Some(format!("{{\"id\":\"{id}\"}}")),
            metadata: serde_json::Map::new(),
            embedding: None,
        },
        embedding_dim: 0,
    }
}

#[test]
fn canonical_graph_batch_commits_events_and_projection_jobs_together() {
    let (store, path) = store("canonical-graph-batch");
    let applied = store
        .apply_canonical_graph_mutation_batch(GraphMutationBatch {
            mutations: vec![mutation("event-1", "node-1"), mutation("event-2", "node-2")],
            index_jobs: Vec::new(),
        })
        .unwrap();
    assert_eq!(applied.len(), 2);
    assert!(applied.iter().all(|item| item.inserted && item.mutated));
    let events = store.replay_raw_events("conversation", 0, 10).unwrap();
    assert_eq!(events.len(), 2);
    assert_eq!(
        events[0].payload_json.parse::<serde_json::Value>().unwrap()["metadata"]["namespace"],
        "conversation"
    );
    assert_eq!(
        events[0].payload_json.parse::<serde_json::Value>().unwrap()["metadata"]["workspace_id"],
        "workspace-a"
    );
    assert_eq!(
        events[0].payload_json.parse::<serde_json::Value>().unwrap()["metadata"]["graph_space"],
        "conversation"
    );
    let jobs = store
        .list_index_jobs(
            Some("conversation"),
            Some("PENDING"),
            Some("node"),
            None,
            Some("node_embedding"),
            10,
        )
        .unwrap();
    assert_eq!(jobs.len(), 2);
    assert!(jobs.iter().all(|job| job.payload_json.is_some()));

    let retried = store
        .apply_canonical_graph_mutation_batch(GraphMutationBatch {
            mutations: vec![mutation("event-1", "node-1")],
            index_jobs: Vec::new(),
        })
        .unwrap();
    assert_eq!(retried.len(), 1);
    assert!(!retried[0].inserted && !retried[0].mutated);
    assert_eq!(
        store
            .replay_raw_events("conversation", 0, 10)
            .unwrap()
            .len(),
        2
    );
    assert_eq!(
        store
            .list_index_jobs(
                Some("conversation"),
                Some("PENDING"),
                Some("node"),
                None,
                Some("node_embedding"),
                10,
            )
            .unwrap()
            .len(),
        2
    );
    drop(store);
    let _ = fs::remove_file(path);
}

#[test]
fn canonical_graph_batch_rolls_back_event_when_projection_job_insert_fails() {
    let (store, path) = store("canonical-graph-job-failure");
    drop(store);

    let connection = rusqlite::Connection::open(&path).unwrap();
    connection
        .execute_batch(
            "CREATE TRIGGER reject_canonical_projection_job
             BEFORE INSERT ON index_jobs
             BEGIN
                 SELECT RAISE(ABORT, 'injected projection-job failure');
             END;",
        )
        .unwrap();
    drop(connection);

    let store = SqliteStore::open(&path).unwrap();
    let result = store.apply_canonical_graph_mutation_batch(GraphMutationBatch {
        mutations: vec![mutation(
            "event-before-failed-job",
            "node-before-failed-job",
        )],
        index_jobs: Vec::new(),
    });

    assert!(result.is_err());
    assert!(
        store
            .replay_raw_events("conversation", 0, 10)
            .unwrap()
            .is_empty(),
        "canonical event must roll back with its failed projection intent"
    );
    assert!(
        store
            .list_index_jobs(Some("conversation"), None, None, None, None, 10)
            .unwrap()
            .is_empty()
    );

    drop(store);
    let _ = fs::remove_file(path);
}

#[test]
fn invalid_replayed_event_rolls_back_the_entire_canonical_batch() {
    let (store, path) = store("canonical-graph-batch-rollback");
    let result = store.immediate_transaction(|uow| {
        uow.apply_canonical_graph_mutation_batch(GraphMutationBatch {
            mutations: vec![
                mutation("shared-event", "node-1"),
                mutation("shared-event", "node-2"),
            ],
            index_jobs: Vec::new(),
        })
    });
    assert!(result.is_err());
    assert!(
        store
            .replay_raw_events("conversation", 0, 10)
            .unwrap()
            .is_empty()
    );
    assert!(
        store
            .list_index_jobs(Some("conversation"), Some("PENDING"), None, None, None, 10,)
            .unwrap()
            .is_empty()
    );
    drop(store);
    let _ = fs::remove_file(path);
}

#[test]
fn canonical_graph_batch_rejects_mixed_scopes_before_writing() {
    let (store, path) = store("canonical-graph-batch-scope");
    let mut other = mutation("event-2", "node-2");
    other.scope.namespace = "other".to_owned();
    assert!(
        store
            .apply_canonical_graph_mutation_batch(GraphMutationBatch {
                mutations: vec![mutation("event-1", "node-1"), other],
                index_jobs: Vec::new(),
            })
            .is_err()
    );
    assert!(
        store
            .replay_raw_events("conversation", 0, 10)
            .unwrap()
            .is_empty()
    );
    drop(store);
    let _ = fs::remove_file(path);
}

#[test]
fn canonical_graph_batch_rejects_conflicting_payload_scope_before_writing() {
    let (store, path) = store("canonical-graph-payload-scope-conflict");
    let mut conflicting = mutation("scope-conflict-event", "node-1");
    conflicting.payload["metadata"] = serde_json::json!({"workspace_id":"other-workspace"});
    let result = store.apply_canonical_graph_mutation_batch(GraphMutationBatch {
        mutations: vec![conflicting],
        index_jobs: Vec::new(),
    });
    assert!(result.is_err());
    assert!(
        store
            .replay_raw_events("conversation", 0, 10)
            .unwrap()
            .is_empty()
    );
    drop(store);
    let _ = fs::remove_file(path);
}

#[test]
fn canonical_edge_mutation_uses_shared_embedding_job_kind() {
    let (store, path) = store("canonical-edge-projection-kind");
    let mut edge = mutation("edge-event-1", "edge-1");
    edge.table = "gke_edges".to_owned();
    edge.entity_kind = "edge".to_owned();
    edge.payload = serde_json::json!({
        "id":"edge-1", "relation":"related", "source_ids":["node-1"],
        "target_ids":["node-2"], "metadata":{"engine_type":"conversation"}
    });
    edge.record.document = Some(edge.payload.to_string());

    store
        .apply_canonical_graph_mutation_batch(GraphMutationBatch {
            mutations: vec![edge],
            index_jobs: Vec::new(),
        })
        .unwrap();
    let jobs = store
        .list_index_jobs(Some("conversation"), Some("PENDING"), None, None, None, 10)
        .unwrap();
    assert_eq!(jobs.len(), 1);
    assert_eq!(jobs[0].entity_kind, "edge");
    assert_eq!(jobs[0].index_kind, "node_embedding");

    drop(store);
    let _ = fs::remove_file(path);
}
