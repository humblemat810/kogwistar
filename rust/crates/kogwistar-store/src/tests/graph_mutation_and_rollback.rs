fn fixture_payload(index: usize) -> Value {
    let fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/python-graph-model-payloads-v1.json"
    )))
    .expect("valid graph model fixture");
    fixture["cases"][index]["payload"].clone()
}

fn typed_mutation(entity_kind: &str, payload: Value) -> GraphMutation {
    let id = payload["id"].as_str().expect("fixture id").to_owned();
    GraphMutation {
        scope: GraphScope {
            namespace: "ns".to_owned(),
            workspace_id: Some("ws".to_owned()),
            graph_space: Some("facts".to_owned()),
        },
        table: if entity_kind == "edge" {
            "gke_edges".to_owned()
        } else {
            "gke_nodes".to_owned()
        },
        entity_kind: entity_kind.to_owned(),
        event_id: "event-1".to_owned(),
        op: "ADD".to_owned(),
        payload: payload.clone(),
        record: GraphRecord {
            id,
            document: None,
            metadata: payload["metadata"].as_object().cloned().unwrap_or_default(),
            embedding: None,
        },
        embedding_dim: 0,
    }
}

#[test]
fn runtime_projection_contract_has_one_stable_shape() {
    let mut payload = Map::new();
    payload.insert("status".to_owned(), Value::String("running".to_owned()));
    let projection = runtime_projection_write(payload, 7, "ready");
    assert_eq!(
        runtime_checkpoint_namespace("conv"),
        "conv:workflow_checkpoint_latest"
    );
    assert_eq!(runtime_status_namespace("conv"), "conv:workflow_run_status");
    assert_eq!(projection.last_authoritative_seq, 7);
    assert_eq!(projection.last_materialized_seq, 7);
    assert_eq!(
        projection.projection_schema_version,
        RUNTIME_PROJECTION_SCHEMA_VERSION
    );
    assert_eq!(projection.materialization_status, "ready");
}

#[test]
fn typed_graph_mutation_accepts_matching_payload_and_identity() {
    let payload = fixture_payload(1);
    let mutation = typed_mutation("edge", payload.clone());
    let envelope = GraphModelEnvelope::new(GraphModelClass::Edge, payload);
    validate_typed_graph_mutation(&mutation, &envelope).expect("valid typed mutation");
}

#[test]
fn typed_graph_mutation_rejects_kind_and_identity_confusion() {
    let edge_payload = fixture_payload(1);
    let edge_mutation = typed_mutation("node", edge_payload.clone());
    let edge_envelope = GraphModelEnvelope::new(GraphModelClass::Edge, edge_payload);
    assert!(matches!(
        validate_typed_graph_mutation(&edge_mutation, &edge_envelope),
        Err(StoreError::InvalidEntityEventPayload { .. })
    ));

    let node_payload = fixture_payload(0);
    let wrong_edge_envelope = GraphModelEnvelope::new(GraphModelClass::Edge, node_payload.clone());
    let wrong_edge_mutation = typed_mutation("node", node_payload);
    let error = validate_typed_graph_mutation(&wrong_edge_mutation, &wrong_edge_envelope)
        .expect_err("malformed edge family must fail before the later kind mismatch");
    assert!(matches!(
        error,
        StoreError::InvalidEntityEventPayload { message } if message.contains("source_ids")
    ));

    let node_payload = fixture_payload(0);
    let mut node_mutation = typed_mutation("node", node_payload.clone());
    node_mutation.record.id = "different-id".to_owned();
    let node_envelope = GraphModelEnvelope::new(GraphModelClass::Node, node_payload);
    assert!(matches!(
        validate_typed_graph_mutation(&node_mutation, &node_envelope),
        Err(StoreError::InvalidEntityEventPayload { .. })
    ));
}

#[test]
fn typed_graph_mutation_cannot_bypass_base_mutation_validation() {
    let payload = fixture_payload(0);
    let envelope = GraphModelEnvelope::new(GraphModelClass::Node, payload.clone());
    let valid = typed_mutation("node", payload);

    let mut invalid = valid.clone();
    invalid.table = "gke_edges".to_owned();
    assert!(validate_typed_graph_mutation(&invalid, &envelope).is_err());

    let mut invalid = valid.clone();
    invalid.scope.namespace.clear();
    assert!(validate_typed_graph_mutation(&invalid, &envelope).is_err());

    let mut invalid = valid.clone();
    invalid.op = "DELETE".to_owned();
    assert!(validate_typed_graph_mutation(&invalid, &envelope).is_err());

    let mut invalid = valid.clone();
    invalid.payload = Value::Array(Vec::new());
    assert!(validate_typed_graph_mutation(&invalid, &envelope).is_err());
}

#[test]
fn graph_projection_target_mapping_is_shared_and_exact() {
    let targets = [
        ("node", "gke_nodes"),
        ("edge", "gke_edges"),
        ("document", "gke_documents"),
        ("domain", "gke_domains"),
    ];
    for (entity_kind, table) in targets {
        assert_eq!(graph_projection_table(entity_kind), Ok(table));
        validate_graph_projection_target(table, entity_kind)
            .expect("matching projection target should be accepted");
        let mutation = GraphMutation {
            scope: GraphScope {
                namespace: "ns".to_owned(),
                workspace_id: None,
                graph_space: None,
            },
            table: table.to_owned(),
            entity_kind: entity_kind.to_owned(),
            event_id: format!("{entity_kind}-event"),
            op: "ADD".to_owned(),
            payload: if matches!(entity_kind, "node" | "edge") {
                serde_json::json!({"id": format!("{entity_kind}-id")})
            } else {
                serde_json::json!({})
            },
            record: GraphRecord::new(format!("{entity_kind}-id")),
            embedding_dim: 0,
        };
        validate_graph_mutation(&mutation).expect("projection mutation should be valid");
        for (_, other_table) in targets {
            if other_table != table {
                assert!(validate_graph_projection_target(other_table, entity_kind).is_err());
            }
        }
    }
    assert!(graph_projection_table("unknown").is_err());
    assert!(validate_graph_lifecycle_target("gke_documents", "document").is_err());
    assert!(validate_graph_lifecycle_target("gke_domains", "domain").is_err());
    assert!(validate_graph_lifecycle_target("gke_nodes", "node").is_ok());
    assert!(validate_graph_lifecycle_target("gke_edges", "edge").is_ok());
}

#[test]
fn direct_graph_mutation_validation_rejects_effect_bypass_inputs() {
    let mut mutation = typed_mutation("node", fixture_payload(0));
    mutation.table = "gke_edges".to_owned();
    assert!(matches!(
        validate_graph_mutation(&mutation),
        Err(StoreError::InvalidEntityEventPayload { .. })
    ));

    mutation.table = "gke_nodes".to_owned();
    mutation.payload = Value::Array(Vec::new());
    assert!(matches!(
        validate_graph_mutation(&mutation),
        Err(StoreError::InvalidEntityEventPayload { .. })
    ));
}

#[test]
fn direct_graph_mutation_validation_rejects_unrequested_scope_dimensions() {
    let mut mutation = typed_mutation("node", fixture_payload(0));
    mutation.scope.workspace_id = None;
    mutation.scope.graph_space = None;
    mutation.record.metadata.insert(
        "workspace_id".to_owned(),
        Value::String("foreign".to_owned()),
    );
    assert!(validate_graph_mutation(&mutation).is_err());

    mutation.record.metadata.remove("workspace_id");
    mutation.payload["metadata"]["graph_space"] = Value::String("foreign".to_owned());
    assert!(validate_graph_mutation(&mutation).is_err());
}

#[test]
fn graph_mutation_replay_requires_identical_effect_identity_and_payload() {
    let mutation = typed_mutation("node", fixture_payload(0));
    let event = EntityEvent {
        namespace: mutation.scope.namespace.clone(),
        seq: 1,
        event_id: mutation.event_id.clone(),
        entity_kind: mutation.entity_kind.clone(),
        entity_id: mutation.record.id.clone(),
        op: mutation.op.clone(),
        payload: mutation.payload.clone(),
    };
    assert_eq!(validate_graph_mutation_replay(&mutation, &event), Ok(()));

    let mut altered = event.clone();
    altered.entity_kind = "edge".to_owned();
    assert!(validate_graph_mutation_replay(&mutation, &altered).is_err());
    let mut altered = event.clone();
    altered.entity_id.push_str("-other");
    assert!(validate_graph_mutation_replay(&mutation, &altered).is_err());
    let mut altered = event.clone();
    altered.op = "REPLACE".to_owned();
    assert!(validate_graph_mutation_replay(&mutation, &altered).is_err());
    let mut altered = event.clone();
    altered.payload["summary"] = Value::String("different".to_owned());
    assert!(validate_graph_mutation_replay(&mutation, &altered).is_err());
    let mut altered = event.clone();
    altered.event_id.push_str("-other");
    assert!(validate_graph_mutation_replay(&mutation, &altered).is_err());
    let mut altered = event;
    altered.namespace.push_str("-other");
    assert!(validate_graph_mutation_replay(&mutation, &altered).is_err());
}

#[test]
fn graph_mutation_batch_validation_rejects_scope_and_index_drift() {
    let first = typed_mutation("node", fixture_payload(0));
    let mut second = typed_mutation("edge", fixture_payload(1));
    second.scope.namespace = "other".to_owned();
    assert!(matches!(
        validate_graph_mutation_batch(&GraphMutationBatch {
            mutations: vec![first.clone(), second],
            index_jobs: Vec::new(),
        }),
        Err(StoreError::InvalidEntityEventPayload { .. })
    ));

    assert!(matches!(
        validate_graph_mutation_batch(&GraphMutationBatch {
            mutations: vec![first],
            index_jobs: vec![NewIndexJob {
                job_id: "job-1".to_owned(),
                namespace: "other".to_owned(),
                entity_kind: "node".to_owned(),
                entity_id: "n-1".to_owned(),
                index_kind: "doc".to_owned(),
                op: "UPSERT".to_owned(),
                payload_json: None,
                max_retries: 0,
            }],
        }),
        Err(StoreError::InvalidEntityEventPayload { .. })
    ));
}

#[test]
fn opaque_payload_metadata_does_not_select_storage_scope() {
    // Python's kg_graph_type is engine configuration, not an entity field.
    // A similarly named payload value must remain opaque to the store.
    for metadata_value in [
        "knowledge",
        "conversation",
        "workflow",
        "wisdom",
        "custom-role",
    ] {
        for (entity_kind, fixture_index) in [("node", 0), ("edge", 1)] {
            let mut mutation = typed_mutation(entity_kind, fixture_payload(fixture_index));
            mutation.payload["metadata"]["engine_type"] = Value::String(metadata_value.into());
            mutation.record.metadata.insert(
                "engine_type".to_owned(),
                Value::String(metadata_value.into()),
            );

            assert_eq!(mutation.scope.graph_space.as_deref(), Some("facts"));
            assert!(
                validate_graph_mutation_batch(&GraphMutationBatch {
                    mutations: vec![mutation],
                    index_jobs: Vec::new(),
                })
                .is_ok(),
                "opaque metadata {metadata_value:?} must not specialize {entity_kind} storage"
            );
        }
    }
}

#[test]
fn graph_mutation_batch_rejects_index_retry_counts_outside_portable_range() {
    let mutation = typed_mutation("node", fixture_payload(0));
    for max_retries in [-1, i32::MAX as i64 + 1] {
        let result = validate_graph_mutation_batch(&GraphMutationBatch {
            mutations: vec![mutation.clone()],
            index_jobs: vec![NewIndexJob {
                job_id: "job-1".to_owned(),
                namespace: mutation.scope.namespace.clone(),
                entity_kind: "node".to_owned(),
                entity_id: mutation.record.id.clone(),
                index_kind: "doc".to_owned(),
                op: "UPSERT".to_owned(),
                payload_json: None,
                max_retries,
            }],
        });
        assert!(matches!(
            result,
            Err(StoreError::InvalidEntityEventPayload { .. })
        ));
    }

    let boundary = validate_graph_mutation_batch(&GraphMutationBatch {
        mutations: vec![mutation.clone()],
        index_jobs: vec![NewIndexJob {
            job_id: "job-boundary".to_owned(),
            namespace: mutation.scope.namespace.clone(),
            entity_kind: "node".to_owned(),
            entity_id: mutation.record.id.clone(),
            index_kind: "doc".to_owned(),
            op: "UPSERT".to_owned(),
            payload_json: None,
            max_retries: i32::MAX as i64,
        }],
    });
    assert!(boundary.is_ok());
}

#[test]
fn graph_mutation_rejects_payload_and_projection_identity_drift() {
    let payload = fixture_payload(0);
    let mut mutation = typed_mutation("node", payload);
    mutation.record.id = "different-node".to_owned();

    assert!(matches!(
        validate_graph_mutation(&mutation),
        Err(StoreError::InvalidEntityEventPayload { .. })
    ));
}

// Document-store capability defaults. Python parity: backend capability
// selection before `RollbackSubsystem.rollback_document` dispatch.

#[derive(Default)]
struct UnsupportedDocumentStore;

impl DocumentStore for UnsupportedDocumentStore {}

fn block_on<T>(future: impl std::future::Future<Output = T>) -> T {
    let waker = std::task::Waker::noop();
    let mut context = std::task::Context::from_waker(waker);
    let mut future = std::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            std::task::Poll::Ready(value) => return value,
            std::task::Poll::Pending => std::thread::yield_now(),
        }
    }
}

#[test]
fn document_rollback_default_fails_closed_without_claiming_support() {
    let store = UnsupportedDocumentStore;
    assert!(!DocumentStore::supports_document_rollback(&store));

    let result = block_on(DocumentStore::rollback_document(
        &store,
        DocumentRollbackRequest {
            scope: GraphScope {
                namespace: "docs".to_owned(),
                workspace_id: None,
                graph_space: None,
            },
            document_id: "source-a".to_owned(),
            operation_id: "rollback-a".to_owned(),
            deleted_at: "2026-11-16T00:00:00Z".to_owned(),
            page_size: 32,
        },
    ));

    assert!(matches!(
        result,
        Err(StoreError::Backend { backend, message })
            if backend == "document" && message.contains("not supported")
    ));
}
