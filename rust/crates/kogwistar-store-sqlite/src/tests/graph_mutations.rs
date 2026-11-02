use super::*;
use kogwistar_store::{GraphMutation, GraphMutationBatch, GraphRecord, GraphScope, NewIndexJob};

fn mutation(event_id: &str, id: &str) -> GraphMutation {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../../contracts/golden/python-graph-model-payloads-v1.json"
    ))
    .unwrap();
    let mut payload = fixture["cases"][0]["payload"].clone();
    payload["id"] = serde_json::Value::String(id.to_owned());
    payload["label"] = serde_json::Value::String(id.to_owned());
    payload["metadata"]["workspace_id"] = serde_json::Value::String("workspace-a".to_owned());
    payload["metadata"]["graph_space"] = serde_json::Value::String("conversation".to_owned());
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
        payload: payload.clone(),
        record: GraphRecord {
            id: id.to_owned(),
            document: Some(payload.to_string()),
            metadata: payload["metadata"].as_object().unwrap().clone(),
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
            index_jobs: vec![NewIndexJob {
                job_id: "duplicate-node-embedding".to_owned(),
                namespace: "conversation".to_owned(),
                entity_kind: "node".to_owned(),
                entity_id: "node-1".to_owned(),
                index_kind: "node_embedding".to_owned(),
                op: "UPSERT".to_owned(),
                payload_json: None,
                max_retries: 3,
            }],
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
    for id in ["node-1", "node-2"] {
        let projection = store
            .get_stage1_node_projection("conversation", &format!("node:{id}"))
            .unwrap()
            .expect("canonical active graph writes are immediately visible in Stage-1");
        assert_eq!(
            projection.last_authoritative_seq,
            projection.last_materialized_seq
        );
        assert_eq!(projection.materialization_status, "pending");
        assert_eq!(projection.payload["id"], id);
        assert!(projection.payload["source_fingerprint"].as_str().is_some());
    }

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
             WHEN NEW.entity_id = 'node-2'
             BEGIN
                 SELECT RAISE(ABORT, 'injected projection-job failure');
             END;",
        )
        .unwrap();
    drop(connection);

    let store = SqliteStore::open(&path).unwrap();
    let result = store.apply_canonical_graph_mutation_batch(GraphMutationBatch {
        mutations: vec![
            mutation("event-node-1", "node-1"),
            mutation("event-node-2", "node-2"),
        ],
        index_jobs: Vec::new(),
    });

    assert!(result.is_err());
    assert!(
        store
            .replay_raw_events("conversation", 0, 10)
            .unwrap()
            .is_empty(),
        "all canonical events must roll back when a later projection intent fails"
    );
    assert!(
        store
            .list_index_jobs(Some("conversation"), None, None, None, None, 10)
            .unwrap()
            .is_empty()
    );
    assert!(
        store
            .list_stage1_node_projections("conversation")
            .unwrap()
            .is_empty()
    );

    drop(store);
    let _ = fs::remove_file(path);
}

#[test]
fn canonical_graph_batch_rolls_back_when_stage1_metadata_projection_fails() {
    let (store, path) = store("canonical-graph-stage1-failure");
    drop(store);

    let connection = rusqlite::Connection::open(&path).unwrap();
    connection
        .execute_batch(
            "CREATE TRIGGER reject_stage1_metadata_projection
             BEFORE INSERT ON stage1_node_projections
             WHEN NEW.key = 'node:node-2'
             BEGIN
                 SELECT RAISE(ABORT, 'injected Stage-1 metadata projection failure');
             END;",
        )
        .unwrap();
    drop(connection);

    let store = SqliteStore::open(&path).unwrap();
    let result = store.apply_canonical_graph_mutation_batch(GraphMutationBatch {
        mutations: vec![
            mutation("event-node-1", "node-1"),
            mutation("event-node-2", "node-2"),
        ],
        index_jobs: Vec::new(),
    });

    assert!(result.is_err());
    assert!(
        store
            .replay_raw_events("conversation", 0, 10)
            .unwrap()
            .is_empty(),
        "canonical events must roll back when a later metadata projection fails"
    );
    assert!(
        store
            .list_index_jobs(Some("conversation"), None, None, None, None, 10)
            .unwrap()
            .is_empty(),
        "durable projection jobs must roll back with the canonical events"
    );
    assert!(
        store
            .list_stage1_node_projections("conversation")
            .unwrap()
            .is_empty(),
        "earlier Stage-1 rows must roll back with the failed projection"
    );

    drop(store);
    let _ = fs::remove_file(path);
}

#[test]
fn canonical_graph_batch_rolls_back_when_explicit_index_intent_fails() {
    let (store, path) = store("canonical-graph-explicit-job-failure");
    drop(store);

    let connection = rusqlite::Connection::open(&path).unwrap();
    connection
        .execute_batch(
            "CREATE TRIGGER reject_explicit_index_intent
             BEFORE INSERT ON index_jobs
             WHEN NEW.job_id = 'explicit-job'
             BEGIN
                 SELECT RAISE(ABORT, 'injected explicit index-intent failure');
             END;",
        )
        .unwrap();
    drop(connection);

    let store = SqliteStore::open(&path).unwrap();
    let result = store.apply_canonical_graph_mutation_batch(GraphMutationBatch {
        mutations: vec![mutation("event-node-1", "node-1")],
        index_jobs: vec![NewIndexJob {
            job_id: "explicit-job".to_owned(),
            namespace: "conversation".to_owned(),
            entity_kind: "node".to_owned(),
            entity_id: "node-1".to_owned(),
            index_kind: "sidecar_embedding".to_owned(),
            op: "UPSERT".to_owned(),
            payload_json: Some("{}".to_owned()),
            max_retries: 3,
        }],
    });

    assert!(result.is_err());
    assert!(
        store
            .replay_raw_events("conversation", 0, 10)
            .unwrap()
            .is_empty(),
        "canonical events must roll back when a later explicit intent fails"
    );
    assert!(
        store
            .list_index_jobs(Some("conversation"), None, None, None, None, 10)
            .unwrap()
            .is_empty(),
        "canonical and explicit index intents must roll back together"
    );
    assert!(
        store
            .list_stage1_node_projections("conversation")
            .unwrap()
            .is_empty(),
        "the derived Stage-1 projection must roll back with the event"
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
fn canonical_graph_batch_rejects_payload_scope_when_scope_dimension_is_absent() {
    let (store, path) = store("canonical-graph-payload-extra-scope");
    let mut conflicting = mutation("scope-conflict-event", "node-1");
    conflicting.scope.workspace_id = None;
    conflicting.scope.graph_space = None;
    conflicting.payload["metadata"] = serde_json::json!({"namespace":"conversation"});
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

    let mut record_conflict = mutation("record-scope-conflict-event", "node-2");
    record_conflict.scope.workspace_id = None;
    record_conflict.scope.graph_space = None;
    record_conflict.payload["metadata"] = serde_json::json!({"namespace":"conversation"});
    assert!(
        store
            .apply_canonical_graph_mutation_batch(GraphMutationBatch {
                mutations: vec![record_conflict],
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
fn canonical_edge_mutation_uses_shared_embedding_job_kind() {
    let (store, path) = store("canonical-edge-projection-kind");
    let mut edge = mutation("edge-event-1", "edge-1");
    edge.table = "gke_edges".to_owned();
    edge.entity_kind = "edge".to_owned();
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../../contracts/golden/python-graph-model-payloads-v1.json"
    ))
    .unwrap();
    edge.payload = fixture["cases"][1]["payload"].clone();
    edge.payload["id"] = serde_json::Value::String("edge-1".to_owned());
    edge.payload["relation"] = serde_json::Value::String("related".to_owned());
    edge.payload["metadata"]["workspace_id"] = serde_json::Value::String("workspace-a".to_owned());
    edge.payload["metadata"]["graph_space"] = serde_json::Value::String("conversation".to_owned());
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
    assert!(
        store
            .get_stage1_node_projection("conversation", "edge:edge-1")
            .unwrap()
            .is_some()
    );

    drop(store);
    let _ = fs::remove_file(path);
}

#[test]
fn canonical_tombstone_clears_stage1_projection_in_the_event_transaction() {
    let (store, path) = store("canonical-stage1-tombstone");
    store
        .apply_canonical_graph_mutation_batch(GraphMutationBatch {
            mutations: vec![mutation("node-add", "node-1")],
            index_jobs: Vec::new(),
        })
        .unwrap();
    let mut tombstone = mutation("node-delete", "node-1");
    tombstone.op = "TOMBSTONE".to_owned();
    store
        .apply_canonical_graph_mutation_batch(GraphMutationBatch {
            mutations: vec![tombstone],
            index_jobs: Vec::new(),
        })
        .unwrap();

    assert!(
        store
            .get_stage1_node_projection("conversation", "node:node-1")
            .unwrap()
            .is_none()
    );
    let events = store.replay_raw_events("conversation", 0, 10).unwrap();
    assert_eq!(events.len(), 2);
    drop(store);
    let _ = fs::remove_file(path);
}

#[test]
fn ungrounded_graph_mutation_remains_canonical_but_is_not_stage1_indexed() {
    let (store, path) = store("canonical-stage1-ungrounded");
    let mut ungrounded = mutation("node-ungrounded", "node-1");
    ungrounded.payload["mentions"] = serde_json::json!([]);
    store
        .apply_canonical_graph_mutation_batch(GraphMutationBatch {
            mutations: vec![ungrounded],
            index_jobs: Vec::new(),
        })
        .unwrap();

    assert_eq!(
        store
            .replay_raw_events("conversation", 0, 10)
            .unwrap()
            .len(),
        1
    );
    assert!(
        store
            .get_stage1_node_projection("conversation", "node:node-1")
            .unwrap()
            .is_none()
    );
    drop(store);
    let _ = fs::remove_file(path);
}
