#[test]
fn failed_inline_graph_effect_clears_provider_event_side_effects() {
    let mut effect: RecordedWorkerSuccessEffect = serde_json::from_value(json!({
        "contract_version": 1,
        "effect_id": "effect-1",
        "status": "success",
        "provider_event_acknowledgements": [{
            "message_id": "message-1",
            "claimed_by": "worker-1",
            "event_id": "event-1"
        }],
        "provider_event_publications": [{
            "namespace": "conversation",
            "conversation_id": "conversation-1",
            "event": {
                "event_id": "event-2",
                "trace": {
                    "run_id": "run-1",
                    "turn_id": "turn-1",
                    "workflow_node_id": "node-1",
                    "step_id": "step-1",
                    "provider_id": "provider-1"
                },
                "event": {"kind": "response_started"}
            }
        }]
    }))
    .expect("valid worker effect");

    fail_inline_graph_effect(
        &mut effect,
        "KOGWISTAR_GRAPH_AUTHORIZATION_DENIED",
        "owner authorization failed",
    );

    assert_eq!(
        effect.status,
        kogwistar_runtime::RuntimeWorkerEffectStatus::Failed
    );
    assert!(effect.provider_event_acknowledgements.is_empty());
    assert!(effect.provider_event_publications.is_empty());
    assert_eq!(effect.errors, ["owner authorization failed"]);
}

#[test]
fn inline_graph_mutation_scope_uses_current_owner_scope() {
    let principal = json!({
        "ns": ["docs"],
        "storage_ns": "docs",
        "workspace": "workspace-a",
        "graph_space": "knowledge"
    });
    let mutation: kogwistar_contracts::GraphMutationV1 = serde_json::from_value(json!({
        "scope": {
            "namespace": "docs",
            "workspace_id": "workspace-a",
            "graph_space": "knowledge"
        },
        "table": "gke_nodes",
        "entity_kind": "node",
        "event_id": "event-1",
        "op": "upsert",
        "payload": {},
        "record": {"id": "node-1", "metadata": {}},
        "embedding_dim": 0
    }))
    .expect("valid graph mutation");

    assert!(!inline_graph_mutations_outside_owner_scope(
        &principal,
        std::slice::from_ref(&mutation)
    ));

    let mut cross_scope = mutation;
    cross_scope.scope.workspace_id = Some("workspace-b".to_owned());
    assert!(inline_graph_mutations_outside_owner_scope(
        &principal,
        &[cross_scope]
    ));
}

#[tokio::test]
async fn sqlite_inline_graph_owner_revocation_fails_claim_durably() {
    let store_path = std::env::temp_dir().join(format!(
        "kogwistar-sqlite-inline-owner-{}.sqlite",
        uuid::Uuid::new_v4().simple()
    ));
    let auth_path = std::env::temp_dir().join(format!(
        "kogwistar-sqlite-inline-owner-auth-{}.sqlite",
        uuid::Uuid::new_v4().simple()
    ));
    let auth = SqliteAuthStore::open(&auth_path).unwrap();
    auth.resolve_external_identity(ResolveExternalIdentity {
        issuer: "test".to_owned(),
        subject: "alice".to_owned(),
        email: "alice@example.test".to_owned(),
        display_name: None,
        new_user_id: "alice".to_owned(),
        default_role: "rw".to_owned(),
        default_ns: "alpha".to_owned(),
    })
    .await
    .unwrap();
    drop(auth);

    let mut registry = kogwistar_runtime::NativeOperationRegistry::default();
    registry
        .register_graph("app/inline-owner-test", "v1", |request| {
            let entity_id = format!("inline-node-{}", request.run_id);
            let namespace = if request.run_id == "sqlite-inline-cross-scope" {
                "beta"
            } else {
                "alpha"
            };
            let batch = serde_json::from_value(json!({
                "contract_version": 1,
                "atomic": true,
                "mutations": [{
                    "scope": {"namespace": namespace},
                    "table": "gke_nodes",
                    "entity_kind": "node",
                    "event_id": format!("inline-event-{}", request.run_id),
                    "op": "ADD",
                    "payload": {"id": entity_id},
                    "record": {"id": entity_id, "document": "inline owner test"},
                    "embedding_dim": 0
                }],
                "index_jobs": []
            }))
            .unwrap();
            Ok((
                kogwistar_runtime::NativeOperationOutput {
                    state_update: Vec::new(),
                    result: Some(json!({"inline_graph": true})),
                },
                batch,
            ))
        })
        .unwrap();
    let auth_config = AuthConfig {
        auth_db_url: Some(format!("sqlite:///{}", auth_path.display())),
        ..AuthConfig::default()
    };
    let mut service = SqliteRunApplicationService::open(&store_path)
        .unwrap()
        .with_native_operations(registry)
        .with_owner_auth_config(&auth_config)
        .await
        .unwrap();
    let submit = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/workflow/runs".to_owned(),
            body: serde_json::to_vec(&json!({
                "run_id": "sqlite-inline-owner-revoked",
                "workflow_id": "wf",
                "conversation_id": "conv-inline-owner-revoked",
                "start_node_id": "entry",
                "node_ops": {"entry": "rust-native/app/inline-owner-test"}
            }))
            .unwrap(),
            principal: json!({
                "user_id": "alice",
                "sub": "alice@example.test",
                "role": "rw",
                "ns": "alpha",
                "storage_ns": "alpha",
                "capabilities": ["workflow.run.write", "write_graph"]
            }),
        })
        .await;
    assert_eq!(submit.status, StatusCode::ACCEPTED.as_u16());

    service.owner_auth_store = None;
    assert!(matches!(
        service
            .resolve_inline_native_owner_principal("sqlite-inline-owner-revoked", "write_graph")
            .await,
        Err(InlineOwnerPrincipalError::Unavailable(_))
    ));
    service.owner_auth_store = Some(AuthStore::from_config(&auth_config).await.unwrap());

    tokio::time::sleep(std::time::Duration::from_millis(1100)).await;
    let auth_connection = rusqlite::Connection::open(&auth_path).unwrap();
    let changed = auth_connection
        .execute(
            "UPDATE users SET global_role = 'ro' WHERE user_id = 'alice'",
            [],
        )
        .unwrap();
    assert_eq!(changed, 1);

    assert!(service.execute_inline_native_work_once().await.unwrap());
    let run = service
        .store
        .get_server_run("sqlite-inline-owner-revoked")
        .unwrap()
        .unwrap();
    assert_eq!(run.status, "failed");
    let lanes = service
        .store
        .list_projected_lane_messages(LaneMessageFilter {
            correlation_id: Some("sqlite-inline-owner-revoked".to_owned()),
            limit: 10,
            ..LaneMessageFilter::default()
        })
        .unwrap();
    assert_eq!(lanes.len(), 1);
    assert_eq!(lanes[0].status, "completed");
    assert!(!service.execute_inline_native_work_once().await.unwrap());
    assert!(
        service
            .store
            .replay_raw_events("alpha", 0, 100)
            .unwrap()
            .iter()
            .all(|event| event.event_id != "inline-event-sqlite-inline-owner-revoked")
    );

    let restored = auth_connection
        .execute(
            "UPDATE users SET global_role = 'rw' WHERE user_id = 'alice'",
            [],
        )
        .unwrap();
    assert_eq!(restored, 1);
    let cross_scope_submit = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/workflow/runs".to_owned(),
            body: serde_json::to_vec(&json!({
                "run_id": "sqlite-inline-cross-scope",
                "workflow_id": "wf",
                "conversation_id": "conv-inline-cross-scope",
                "start_node_id": "entry",
                "node_ops": {"entry": "rust-native/app/inline-owner-test"}
            }))
            .unwrap(),
            principal: json!({
                "user_id": "alice",
                "sub": "alice@example.test",
                "role": "rw",
                "ns": "alpha",
                "storage_ns": "alpha",
                "capabilities": ["workflow.run.write", "write_graph"]
            }),
        })
        .await;
    assert_eq!(cross_scope_submit.status, StatusCode::ACCEPTED.as_u16());
    tokio::time::sleep(std::time::Duration::from_millis(1100)).await;

    assert!(service.execute_inline_native_work_once().await.unwrap());
    assert_eq!(
        service
            .store
            .get_server_run("sqlite-inline-cross-scope")
            .unwrap()
            .unwrap()
            .status,
        "failed"
    );
    let cross_scope_lanes = service
        .store
        .list_projected_lane_messages(LaneMessageFilter {
            correlation_id: Some("sqlite-inline-cross-scope".to_owned()),
            limit: 10,
            ..LaneMessageFilter::default()
        })
        .unwrap();
    assert_eq!(cross_scope_lanes.len(), 1);
    assert_eq!(cross_scope_lanes[0].status, "completed");
    assert!(!service.execute_inline_native_work_once().await.unwrap());
    assert!(
        service
            .store
            .replay_raw_events("beta", 0, 100)
            .unwrap()
            .iter()
            .all(|event| event.event_id != "inline-event-sqlite-inline-cross-scope")
    );

    drop(auth_connection);
    drop(service);
    let _ = std::fs::remove_file(store_path);
    let _ = std::fs::remove_file(auth_path);
}

#[test]
fn recorded_step_snapshots_matches_python_fixture() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../../../contracts/golden/workflow-step-trace-context-v1.json"
    ))
    .unwrap();
    let recorded = ServerRunEvent {
        seq: 4,
        run_id: "run-1".to_owned(),
        event_type: "workflow.recorded_transition.v1".to_owned(),
        payload_json: json!({
            "transition_id": "transition-1",
            "reduced": {
                "state": {
                    "last_step_seq": 0,
                    "workflow_id": "workflow-1",
                    "last_node_id": "node-1",
                    "state": {}
                },
                "result": {"ok": true},
                "errors": [],
                "server_status": "running"
            }
        })
        .to_string(),
        created_at_ms: 10,
    };
    let trace = ServerRunEvent {
        seq: 5,
        run_id: "run-1".to_owned(),
        event_type: "workflow.trace.v1".to_owned(),
        payload_json: json!({
            "trace": {
                "event": "runtime.native.step.completed",
                "step_seq": 0,
                "node_id": "node-1",
                "trace_id": fixture["trace_context"]["trace_id"],
                "span_id": fixture["trace_context"]["span_id"],
                "parent_span_id": fixture["trace_context"]["parent_span_id"],
                "attempt": fixture["trace_context"]["attempt"]
            }
        })
        .to_string(),
        created_at_ms: 11,
    };

    let steps = recorded_step_snapshots(vec![recorded.clone(), trace]);
    assert_eq!(steps.len(), 1);
    assert_eq!(
        steps[0].trace_context.as_ref().unwrap(),
        &fixture["trace_context"]
    );
    let legacy_steps = recorded_step_snapshots(vec![recorded]);
    assert!(legacy_steps[0].trace_context.is_none());
    assert!(
        fixture["legacy_step_omits_trace_context"]
            .as_bool()
            .unwrap()
    );
}

// PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/tests/part_08.rs#in_memory_backend_routes_authenticated_graph_write_and_mcp_read; python=kogwistar/server_mcp_with_admin.py#api_viz_d3; fixture=contracts/golden/graph-acl-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/part_08.rs#in_memory_backend_routes_authenticated_graph_write_and_mcp_read; python_test=tests/test_visualization.py#test_visualization_uses_acl_policy_for_selected_python_graph_type
#[tokio::test]
async fn in_memory_backend_routes_authenticated_graph_write_and_mcp_read() {
    let acl_fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/graph-acl-v1.json"
    )))
    .unwrap();
    let isolation = &acl_fixture["graph_type_isolation"];
    let shared_entity_id = isolation["entity_id"].as_str().unwrap();
    let isolation_cases = isolation["cases"].as_array().unwrap();
    let knowledge_policy = isolation_cases
        .iter()
        .find(|case| case["truth_graph"] == "knowledge")
        .unwrap();
    let conversation_policy = isolation_cases
        .iter()
        .find(|case| case["truth_graph"] == "conversation")
        .unwrap();

    let secret = "in-memory-api-contract-secret";
    let auth = AuthConfig {
        mode: "jwt".to_owned(),
        algorithm: Some("HS256".to_owned()),
        key: Some(secret.to_owned()),
        ..AuthConfig::default()
    };
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let token_for_graph_space = |namespace: &str, capabilities: &[&str], graph_space: &str| {
        encode(
            &Header::new(Algorithm::HS256),
            &json!({
                "sub": "writer",
                "role": "rw",
                "ns": [namespace],
                "storage_ns": "tenant-a",
                "workspace": "workspace-a",
                "workspace_id": "workspace-a",
                "graph_space": graph_space,
                "capabilities": capabilities,
                "exp": now + 300
            }),
            &EncodingKey::from_secret(secret.as_bytes()),
        )
        .unwrap()
    };
    let token = |namespace: &str, capabilities: &[&str]| {
        token_for_graph_space(namespace, capabilities, "facts")
    };
    let application = Arc::new(InMemoryGraphApplicationService::default());
    let make_app = || {
        router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: vec![],
                implementation: ImplementationSnapshot::default(),
                auth: auth.clone(),
            },
            application.clone(),
        )
    };
    let app = make_app();
    let scope = json!({
        "namespace": "tenant-a",
        "workspace_id": "workspace-a",
        "graph_space": "facts"
    });
    let mutation = |id: &str, table: &str, kind: &str, metadata: Value, document: Option<&str>| {
        json!({
            "scope": scope.clone(),
            "table": table,
            "entity_kind": kind,
            "event_id": format!("event-{id}"),
            "op": "ADD",
            "payload": {"id": id, "metadata": metadata},
            "record": {
                "id": id,
                "document": document,
                "metadata": metadata,
                "embedding": null
            },
            "embedding_dim": 0
        })
    };
    let mut mutations = vec![
        mutation(
            "n1",
            "gke_nodes",
            "node",
            json!({"label":"one"}),
            Some(r#"{"label":"one"}"#),
        ),
        mutation(
            "n2",
            "gke_nodes",
            "node",
            json!({"label":"two"}),
            Some(r#"{"label":"two"}"#),
        ),
        mutation(
            "e1",
            "gke_edges",
            "edge",
            json!({"source_ids":["n1"],"target_ids":["n2"],"relation":"links"}),
            Some(r#"{"source_ids":["n1"],"target_ids":["n2"],"relation":"links"}"#),
        ),
        mutation(
            "n3",
            "gke_nodes",
            "node",
            json!({"label":"private"}),
            Some("private"),
        ),
        mutation(
            shared_entity_id,
            "gke_nodes",
            "node",
            json!({"label":"same ID in knowledge"}),
            Some("same ID in knowledge"),
        ),
        mutation(
            "acl-n1",
            "gke_nodes",
            "node",
            json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"node","acl_target_entity_id":"n1","acl_version":1,"acl_mode":"public"}),
            None,
        ),
        mutation(
            "acl-shared-knowledge",
            "gke_nodes",
            "node",
            json!({
                "entity_type":"acl_record",
                "acl_truth_graph":"knowledge",
                "acl_target_grain":"node",
                "acl_target_entity_id":shared_entity_id,
                "acl_version":1,
                "acl_mode":knowledge_policy["mode"]
            }),
            None,
        ),
        mutation(
            "acl-n2",
            "gke_nodes",
            "node",
            json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"node","acl_target_entity_id":"n2","acl_version":1,"acl_mode":"public"}),
            None,
        ),
        mutation(
            "acl-e1",
            "gke_nodes",
            "node",
            json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"edge","acl_target_entity_id":"e1","acl_version":1,"acl_mode":"public"}),
            None,
        ),
        mutation(
            "acl-n3",
            "gke_nodes",
            "node",
            json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"node","acl_target_entity_id":"n3","acl_version":1,"acl_mode":"private","acl_owner_id":"another-user"}),
            None,
        ),
    ];
    for (index, embedding) in [
        (0, json!([1.0, 0.0])),
        (1, json!([0.8, 0.2])),
        (3, json!([0.99, 0.01])),
        (5, json!([0.999, 0.001])),
    ] {
        mutations[index]["record"]["embedding"] = embedding;
        mutations[index]["embedding_dim"] = json!(2);
    }
    let body = json!({"contract_version":1,"atomic":true,"mutations":mutations,"index_jobs":[]});

    let denied_write = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/internal/graph/mutations")
                .header("content-type", "application/json")
                .header(
                    "authorization",
                    format!("Bearer {}", token("docs", &["read_graph"])),
                )
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(denied_write.status(), StatusCode::FORBIDDEN);

    let denied_namespace = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/internal/graph/mutations")
                .header("content-type", "application/json")
                .header(
                    "authorization",
                    format!("Bearer {}", token("workflow", &["write_graph"])),
                )
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(denied_namespace.status(), StatusCode::FORBIDDEN);

    let mut cross_workspace_body = body.clone();
    for mutation in cross_workspace_body["mutations"].as_array_mut().unwrap() {
        mutation["scope"]["workspace_id"] = json!("workspace-b");
    }
    let denied_scope = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/internal/graph/mutations")
                .header("content-type", "application/json")
                .header(
                    "authorization",
                    format!("Bearer {}", token("docs", &["write_graph"])),
                )
                .body(Body::from(
                    serde_json::to_vec(&cross_workspace_body).unwrap(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(denied_scope.status(), StatusCode::FORBIDDEN);

    let written = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/internal/graph/mutations")
                .header("content-type", "application/json")
                .header(
                    "authorization",
                    format!("Bearer {}", token("docs", &["write_graph"])),
                )
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(written.status(), StatusCode::OK);
    let written: Value =
        serde_json::from_slice(&to_bytes(written.into_body(), 65_536).await.unwrap()).unwrap();
    assert_eq!(written["atomic"], true);
    assert_eq!(written["results"].as_array().unwrap().len(), 10);

    let cross_kind_fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/graph-neighbors-cross-kind-id-v1.json"
    )))
    .unwrap();
    let parent_edge = &cross_kind_fixture["parent_edge"];
    let cross_kind_id = cross_kind_fixture["expected"]["node_ids"][0]
        .as_str()
        .unwrap();
    let parent_edge_id = cross_kind_fixture["rid"].as_str().unwrap();
    let mut cross_kind_node = mutation(
        cross_kind_id,
        "gke_nodes",
        "node",
        json!({"label":"shared node identity"}),
        Some(r#"{"label":"shared node identity"}"#),
    );
    cross_kind_node["event_id"] = json!("event-cross-kind-node");
    let mut cross_kind_edge = mutation(
        cross_kind_id,
        "gke_edges",
        "edge",
        json!({"source_ids":[],"target_ids":[],"relation":"shared edge identity"}),
        Some(r#"{"source_ids":[],"target_ids":[],"relation":"shared edge identity"}"#),
    );
    cross_kind_edge["event_id"] = json!("event-cross-kind-edge");
    let mut parent_edge_mutation = mutation(
        parent_edge_id,
        "gke_edges",
        "edge",
        json!(parent_edge),
        Some(&parent_edge.to_string()),
    );
    parent_edge_mutation["event_id"] = json!("event-cross-kind-parent-edge");
    let cross_kind_mutations = vec![
        cross_kind_node,
        cross_kind_edge,
        parent_edge_mutation,
        mutation(
            "acl-cross-kind-node",
            "gke_nodes",
            "node",
            json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"node","acl_target_entity_id":cross_kind_id,"acl_version":1,"acl_mode":"public"}),
            None,
        ),
        mutation(
            "acl-cross-kind-edge",
            "gke_nodes",
            "node",
            json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"edge","acl_target_entity_id":cross_kind_id,"acl_version":1,"acl_mode":"public"}),
            None,
        ),
        mutation(
            "acl-cross-kind-parent-edge",
            "gke_nodes",
            "node",
            json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"edge","acl_target_entity_id":parent_edge_id,"acl_version":1,"acl_mode":"public"}),
            None,
        ),
    ];
    let cross_kind_body = json!({
        "contract_version":1,
        "atomic":true,
        "mutations":cross_kind_mutations,
        "index_jobs":[]
    });
    let cross_kind_write = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/internal/graph/mutations")
                .header("content-type", "application/json")
                .header(
                    "authorization",
                    format!("Bearer {}", token("docs", &["write_graph"])),
                )
                .body(Body::from(serde_json::to_vec(&cross_kind_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(cross_kind_write.status(), StatusCode::OK);
    let cross_kind_neighbors = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp")
                .header("content-type", "application/json")
                .header(
                    "authorization",
                    format!("Bearer {}", token("docs", &["read_graph"])),
                )
                .body(Body::from(
                    json!({
                        "jsonrpc":"2.0",
                        "id":"cross-kind-neighbors",
                        "method":"tools/call",
                        "params":{"name":"kg_neighbors","arguments":{
                            "rid":parent_edge_id,
                            "truth_graph":"knowledge"
                        }}
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(cross_kind_neighbors.status(), StatusCode::OK);
    let cross_kind_neighbors: Value = serde_json::from_slice(
        &to_bytes(cross_kind_neighbors.into_body(), 65_536)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(
        cross_kind_neighbors["result"]["implementation_authority"],
        "rust"
    );
    assert_eq!(
        cross_kind_neighbors["result"]["seam_attestation_version"],
        1
    );
    assert_eq!(cross_kind_neighbors["result"]["seam_crossings"], json!([]));
    assert_eq!(
        cross_kind_neighbors["result"]["nodes"],
        cross_kind_fixture["expected"]["node_ids"]
    );
    assert_eq!(
        cross_kind_neighbors["result"]["edges"],
        cross_kind_fixture["expected"]["edge_ids"]
    );

    let conversation_scope = json!({
        "namespace": "tenant-a",
        "workspace_id": "workspace-a",
        "graph_space": "conversation"
    });
    let conversation_mutation = |id: &str, metadata: Value, document: Option<&str>| {
        let mut value = mutation(id, "gke_nodes", "node", metadata.clone(), document);
        value["event_id"] = json!(format!("conversation-event-{id}"));
        value["scope"] = conversation_scope.clone();
        value
    };
    let conversation_body = json!({
        "contract_version": 1,
        "atomic": true,
        "index_jobs": [],
        "mutations": [
            conversation_mutation(
                "conversation-node",
                json!({"label":"conversation only"}),
                Some("conversation only"),
            ),
            conversation_mutation(
                shared_entity_id,
                json!({"label":"same ID in conversation"}),
                Some("same ID in conversation"),
            ),
            conversation_mutation(
                "conversation-acl",
                json!({
                    "entity_type":"acl_record",
                    "acl_truth_graph":"conversation",
                    "acl_target_grain":"node",
                    "acl_target_entity_id":"conversation-node",
                    "acl_version":1,
                    "acl_mode":"public"
                }),
                None,
            ),
            conversation_mutation(
                "conversation-acl-shared",
                json!({
                    "entity_type":"acl_record",
                    "acl_truth_graph":"conversation",
                    "acl_target_grain":"node",
                    "acl_target_entity_id":shared_entity_id,
                    "acl_version":1,
                    "acl_mode":conversation_policy["mode"],
                    "acl_owner_id":conversation_policy["owner_id"]
                }),
                None,
            )
        ]
    });
    let conversation_written = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/internal/graph/mutations")
                .header("content-type", "application/json")
                .header(
                    "authorization",
                    format!(
                        "Bearer {}",
                        token_for_graph_space("docs", &["write_graph"], "conversation")
                    ),
                )
                .body(Body::from(serde_json::to_vec(&conversation_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(conversation_written.status(), StatusCode::OK);

    for case in isolation_cases
        .iter()
        .filter(|case| matches!(case["truth_graph"].as_str(), Some("workflow" | "wisdom")))
    {
        let truth_graph = case["truth_graph"].as_str().unwrap();
        let mut graph_node = mutation(
            shared_entity_id,
            "gke_nodes",
            "node",
            json!({"label":format!("same ID in {truth_graph}")}),
            Some("same ID in selected graph"),
        );
        graph_node["scope"]["graph_space"] = json!(truth_graph);
        graph_node["event_id"] = json!(format!("{truth_graph}-event-node"));
        let mut graph_acl = mutation(
            &format!("{truth_graph}-acl-shared"),
            "gke_nodes",
            "node",
            json!({
                "entity_type":"acl_record",
                "acl_truth_graph":truth_graph,
                "acl_target_grain":"node",
                "acl_target_entity_id":shared_entity_id,
                "acl_version":1,
                "acl_mode":case["mode"]
            }),
            None,
        );
        graph_acl["scope"]["graph_space"] = json!(truth_graph);
        graph_acl["event_id"] = json!(format!("{truth_graph}-event-acl"));
        let graph_body = json!({
            "contract_version":1,
            "atomic":true,
            "index_jobs":[],
            "mutations":[graph_node, graph_acl]
        });
        let written = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/graph/mutations")
                    .header("content-type", "application/json")
                    .header(
                        "authorization",
                        format!(
                            "Bearer {}",
                            token_for_graph_space("docs", &["write_graph"], truth_graph)
                        ),
                    )
                    .body(Body::from(serde_json::to_vec(&graph_body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(written.status(), StatusCode::OK, "graph={truth_graph}");
    }

    let visualization_request = |truth_graph: &str, graph_space: &str| ApiEffectRequest {
        contract_version: 1,
        method: "POST".to_owned(),
        path_and_query: INTERNAL_GRAPH_VIZ_D3_ROUTE.to_owned(),
        body: serde_json::to_vec(&json!({"graph_type":truth_graph})).unwrap(),
        principal: json!({
            "sub":"writer", "role":"rw", "ns":["docs"],
            "storage_ns":"tenant-a", "workspace_id":"workspace-a",
            "graph_space":graph_space, "capabilities":["read_graph"]
        }),
    };
    let conversation_visualization = application
        .graph_read_effect(&visualization_request("conversation", "conversation"))
        .await;
    assert_eq!(conversation_visualization.status, StatusCode::OK.as_u16());
    let conversation_visualization: Value =
        serde_json::from_slice(&conversation_visualization.body).unwrap();
    assert!(
        conversation_visualization["nodes"]
            .to_string()
            .contains("conversation-node"),
        "selected conversation visualization did not return its graph: {conversation_visualization}"
    );
    assert!(
        !conversation_visualization["nodes"]
            .to_string()
            .contains("n1"),
        "conversation visualization leaked knowledge graph data: {conversation_visualization}"
    );
    assert!(
        !conversation_visualization["nodes"]
            .to_string()
            .contains(shared_entity_id),
        "conversation visualization ignored its graph-scoped private ACL: {conversation_visualization}"
    );

    let mismatched_visualization = application
        .graph_read_effect(&visualization_request("conversation", "facts"))
        .await;
    assert_eq!(
        mismatched_visualization.status,
        StatusCode::FORBIDDEN.as_u16()
    );

    for case in isolation_cases {
        let truth_graph = case["truth_graph"].as_str().unwrap();
        let graph_space = match truth_graph {
            "knowledge" => "facts",
            other => other,
        };
        let response = application
            .graph_read_effect(&visualization_request(truth_graph, graph_space))
            .await;
        assert_eq!(
            response.status,
            StatusCode::OK.as_u16(),
            "graph={truth_graph}"
        );
        let payload: Value = serde_json::from_slice(&response.body).unwrap();
        let rendered_nodes = payload["nodes"].to_string();
        let expected_visible = case["expected_visible"].as_bool().unwrap();
        assert_eq!(
            rendered_nodes.contains(shared_entity_id),
            expected_visible,
            "selected graph={truth_graph} used the wrong ACL policy: {payload}"
        );
        if truth_graph == "workflow" || truth_graph == "wisdom" {
            assert!(
                !rendered_nodes.contains("n1") && !rendered_nodes.contains("conversation-node"),
                "selected graph={truth_graph} leaked records from another graph: {payload}"
            );
        }
    }

    let deferred_visualization = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/api/viz/d3.json?graph_type=conversation")
                .header(
                    "authorization",
                    format!(
                        "Bearer {}",
                        token_for_graph_space("docs", &["read_graph"], "conversation")
                    ),
                )
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(deferred_visualization.status(), StatusCode::OK);
    let visualization: Value = serde_json::from_slice(
        &to_bytes(deferred_visualization.into_body(), 65_536)
            .await
            .unwrap(),
    )
    .unwrap();
    assert!(
        visualization["nodes"]
            .to_string()
            .contains("conversation-node"),
        "public visualization failed to preserve graph and insertion-method selection: {visualization}"
    );

    for (path, element_path) in [
        ("/api/viz/d3.json?graph_type=knowledge&mode=reify", "nodes"),
        (
            "/api/viz/cytoscape.json?graph_type=knowledge&mode=reify",
            "elements",
        ),
    ] {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri(path)
                    .header(
                        "authorization",
                        format!("Bearer {}", token("docs", &["read_graph"])),
                    )
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK, "route {path}");
        let visualization: Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 65_536).await.unwrap()).unwrap();
        let ids = visualization[element_path]
            .as_array()
            .unwrap()
            .iter()
            .map(|item| {
                if element_path == "elements" {
                    item["data"]["id"].as_str().unwrap()
                } else {
                    item["id"].as_str().unwrap()
                }
            })
            .collect::<Vec<_>>();
        assert!(ids.contains(&shared_entity_id), "{path}: {ids:?}");
        assert!(ids.iter().any(|id| id.starts_with("n1")), "{path}: {ids:?}");
        assert!(ids.iter().any(|id| id.starts_with("n2")), "{path}: {ids:?}");
        assert!(ids.iter().any(|id| id.starts_with("e1")), "{path}: {ids:?}");
        assert!(
            ids.iter().all(|id| !id.contains("n3")),
            "private graph records must not be exposed by {path}: {ids:?}"
        );
    }

    let bundle = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/viz/d3.bundle?graph_type=knowledge&mode=reify")
                .header(
                    "authorization",
                    format!("Bearer {}", token("docs", &["read_graph"])),
                )
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(bundle.status(), StatusCode::OK);
    assert_eq!(bundle.headers()["content-type"], "text/html; charset=utf-8");
    let bundle = String::from_utf8(
        to_bytes(bundle.into_body(), 1_048_576)
            .await
            .unwrap()
            .to_vec(),
    )
    .unwrap();
    let embedded = bundle
        .split_once("window.__EMBEDDED_DATA__ = ")
        .and_then(|(_, rest)| rest.split_once(';'))
        .map(|(payload, _)| serde_json::from_str::<Value>(payload).unwrap())
        .expect("bundle includes valid embedded graph data");
    let embedded_node_ids = embedded["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|node| node["id"].as_str())
        .collect::<Vec<_>>();
    assert!(embedded_node_ids.contains(&"n1"));
    assert!(embedded_node_ids.contains(&"n2"));
    assert!(embedded_node_ids.contains(&"e1"));
    assert!(!embedded_node_ids.contains(&"n3"));

    let read = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {}", token("docs", &["read_graph"])))
                .body(Body::from(
                    r#"{"jsonrpc":"2.0","id":"neighbors","method":"tools/call","params":{"name":"kg_neighbors","arguments":{"rid":"n1","truth_graph":"knowledge"}}}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(read.status(), StatusCode::OK);
    let read: Value =
        serde_json::from_slice(&to_bytes(read.into_body(), 65_536).await.unwrap()).unwrap();
    assert_eq!(read["result"]["nodes"], json!(["n2"]), "response: {read}");
    assert_eq!(read["result"]["edges"], json!(["e1"]));

    for (tool, arguments, expected) in [
        (
            "kg_find_edges",
            json!({"relation":"links","src_label_contains":"one","tgt_label_contains":"two"}),
            json!({"edges":["e1"]}),
        ),
        (
            "kg_shortest_path",
            json!({"src_id":"n1","dst_id":"n2","max_depth":4}),
            json!({"path":["n1","e1","n2"]}),
        ),
        (
            "kg_k_hop",
            json!({"start_ids":["n1"],"k":1}),
            json!({"layers":[{"nodes":[],"edges":["e1"]}]}),
        ),
    ] {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .header(
                        "authorization",
                        format!("Bearer {}", token("docs", &["read_graph"])),
                    )
                    .body(Body::from(
                        json!({
                            "jsonrpc":"2.0",
                            "id":tool,
                            "method":"tools/call",
                            "params":{"name":tool,"arguments":arguments}
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let response: Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 65_536).await.unwrap()).unwrap();
        assert_eq!(response["result"]["implementation_authority"], "rust");
        assert_eq!(response["result"]["seam_attestation_version"], 1);
        assert_eq!(response["result"]["seam_crossings"], json!([]));
        let mut result = response["result"].clone();
        for field in [
            "implementation_authority",
            "seam_attestation_version",
            "seam_crossings",
        ] {
            result.as_object_mut().unwrap().remove(field);
        }
        assert_eq!(result, expected, "tool={tool}");
    }

    for (tool, key) in [
        ("kg_viz_d3_json", "nodes"),
        ("kg_viz_cytoscape_json", "elements"),
    ] {
        let visualization = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .header(
                        "authorization",
                        format!("Bearer {}", token("docs", &["read_graph"])),
                    )
                    .body(Body::from(format!(
                        "{{\"jsonrpc\":\"2.0\",\"id\":\"{tool}\",\"method\":\"tools/call\",\"params\":{{\"name\":\"{tool}\",\"arguments\":{{\"doc_id\":null,\"mode\":\"reify\"}}}}}}"
                    )))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(visualization.status(), StatusCode::OK);
        let visualization: Value =
            serde_json::from_slice(&to_bytes(visualization.into_body(), 65_536).await.unwrap())
                .unwrap();
        let rows = visualization["result"][key].as_array().unwrap();
        let rendered = rows.iter().map(Value::to_string).collect::<String>();
        assert!(rendered.contains("n1") && rendered.contains("n2"));
        assert!(
            !rendered.contains("n3"),
            "private node leaked: {visualization}"
        );
    }

    let vector_query = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/internal/graph/vector-query")
                .header("content-type", "application/json")
                .header(
                    "authorization",
                    format!("Bearer {}", token("docs", &["read_graph"])),
                )
                .body(Body::from(
                    json!({
                        "entity_kind": "node",
                        "query_embeddings": [[1.0, 0.0], [0.0, 1.0]],
                        "metric": "cosine",
                        "truth_graph": "knowledge",
                        "n_results": 2,
                        "where": {},
                        "include": ["distances"]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(vector_query.status(), StatusCode::OK);
    let vector_query: Value =
        serde_json::from_slice(&to_bytes(vector_query.into_body(), 65_536).await.unwrap()).unwrap();
    assert_eq!(vector_query["ids"], json!([["n1", "n2"], ["n2", "n1"]]));
    assert_eq!(vector_query["distances"][0].as_array().unwrap().len(), 2);
    assert_eq!(vector_query["distances"][1].as_array().unwrap().len(), 2);
    assert!(vector_query.get("documents").is_none());
}

// PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/tests/part_08.rs#postgres_visualization_read_matches_shared_python_selection_and_acl; python=kogwistar/server_mcp_with_admin.py#api_viz_d3; fixture=contracts/golden/visualization-insertion-filter-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/part_08.rs#postgres_visualization_read_matches_shared_python_selection_and_acl; python_test=tests/test_visualization.py#test_visualization_public_routes_forward_shared_fixture_options
#[tokio::test]
#[ignore = "requires KOGWISTAR_TEST_PG_DSN; uses a disposable PostgreSQL schema"]
async fn postgres_visualization_read_matches_shared_python_selection_and_acl() {
    use axum::body::{Body, to_bytes};
    use axum::http::Request;
    use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
    use tower::ServiceExt;

    let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
        .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
    let schema = format!("rust_viz_{}", uuid::Uuid::new_v4().simple());
    let knowledge_schema = format!("{schema}_knowledge");
    let conversation_schema = format!("{schema}_conversation");
    let workflow_schema = format!("{schema}_workflow");
    let wisdom_schema = format!("{schema}_wisdom");
    let service = PostgresRunApplicationService::from_dsn_with_graph_schemas(
        &dsn,
        &knowledge_schema,
        &conversation_schema,
        &workflow_schema,
        &wisdom_schema,
    )
    .unwrap();
    service.ensure_schema().await.unwrap();
    for store in [
        &service.store,
        &service.conversation_store,
        &service.workflow_store,
        &service.wisdom_store,
    ] {
        store
            .create_graph_schema(3, kogwistar_store_postgres::GraphTableNames::default())
            .await
            .unwrap();
    }

    let fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/visualization-insertion-filter-v1.json"
    )))
    .unwrap();
    let scope = GraphScope {
        namespace: "viz-test".to_owned(),
        workspace_id: None,
        graph_space: None,
    };
    let mut entities = fixture["nodes"].as_array().unwrap().clone();
    entities.push(json!({
        "id": "n-private",
        "references": [{"doc_id": "d1", "insertion_method": "parser"}]
    }));
    for entity in &entities {
        let id = entity["id"].as_str().unwrap();
        let metadata = Map::from_iter([
            ("entity_type".to_owned(), json!("node")),
            ("doc_id".to_owned(), fixture["doc_id"].clone()),
            ("lifecycle_status".to_owned(), json!("active")),
        ]);
        service
            .store
            .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                scope: scope.clone(),
                table: "gke_nodes".to_owned(),
                record: GraphRecord {
                    id: id.to_owned(),
                    document: Some(
                        json!({
                            "id": id,
                            "label": id,
                            "references": entity["references"]
                        })
                        .to_string(),
                    ),
                    metadata,
                    embedding: None,
                },
                embedding_dim: 3,
            })
            .await
            .unwrap();
        let acl = if id == "n-private" {
            json!({
                "entity_type": "acl_record",
                "acl_truth_graph": "knowledge",
                "acl_target_grain": "node",
                "acl_target_entity_id": id,
                "acl_mode": "private",
                "owner_id": "bob",
                "acl_version": 1,
                "tombstoned": false
            })
        } else {
            json!({
                "entity_type": "acl_record",
                "acl_truth_graph": "knowledge",
                "acl_target_grain": "node",
                "acl_target_entity_id": id,
                "acl_mode": "public",
                "acl_version": 1,
                "tombstoned": false
            })
        };
        service
            .store
            .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                scope: scope.clone(),
                table: "gke_nodes".to_owned(),
                record: GraphRecord {
                    id: format!("acl|{id}"),
                    document: Some("{}".to_owned()),
                    metadata: acl.as_object().unwrap().clone(),
                    embedding: None,
                },
                embedding_dim: 3,
            })
            .await
            .unwrap();
    }

    service
        .conversation_store
        .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
            scope: scope.clone(),
            table: "gke_nodes".to_owned(),
            record: GraphRecord {
                id: "conversation-only-node".to_owned(),
                document: Some(
                    json!({"id":"conversation-only-node","label":"conversation-only-node"})
                        .to_string(),
                ),
                metadata: Map::from_iter([
                    ("entity_type".to_owned(), json!("node")),
                    ("lifecycle_status".to_owned(), json!("active")),
                ]),
                embedding: None,
            },
            embedding_dim: 3,
        })
        .await
        .unwrap();
    service
        .conversation_store
        .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
            scope: scope.clone(),
            table: "gke_nodes".to_owned(),
            record: GraphRecord {
                id: "acl|conversation-only-node".to_owned(),
                document: Some("{}".to_owned()),
                metadata: Map::from_iter([
                    ("entity_type".to_owned(), json!("acl_record")),
                    ("acl_truth_graph".to_owned(), json!("conversation")),
                    ("acl_target_grain".to_owned(), json!("node")),
                    (
                        "acl_target_entity_id".to_owned(),
                        json!("conversation-only-node"),
                    ),
                    ("acl_mode".to_owned(), json!("public")),
                    ("acl_version".to_owned(), json!(1)),
                    ("tombstoned".to_owned(), json!(false)),
                ]),
                embedding: None,
            },
            embedding_dim: 3,
        })
        .await
        .unwrap();

    let conversation_result = ApplicationService::execute(
        &service,
        ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: INTERNAL_GRAPH_VIZ_D3_ROUTE.to_owned(),
            body: serde_json::to_vec(&json!({"graph_type":"conversation"})).unwrap(),
            principal: json!({
                "sub": "alice",
                "user_id": "alice",
                "storage_ns": "viz-test",
                "ns": ["viz-test"],
                "capabilities": ["read_graph"]
            }),
        },
    )
    .await;
    assert_eq!(conversation_result.status, StatusCode::OK.as_u16());
    let conversation_payload: Value = serde_json::from_slice(&conversation_result.body).unwrap();
    assert!(
        conversation_payload["nodes"]
            .to_string()
            .contains("conversation-only-node")
    );
    assert!(!conversation_payload["nodes"].to_string().contains("n1"));

    let response = ApplicationService::execute(
        &service,
        ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: INTERNAL_GRAPH_VIZ_D3_ROUTE.to_owned(),
            body: serde_json::to_vec(&json!({
                "doc_id": fixture["doc_id"],
                "mode": "reify",
                "insertion_method": fixture["insertion_method"],
                "graph_type": "knowledge"
            }))
            .unwrap(),
            principal: json!({
                "sub": "alice",
                "user_id": "alice",
                "storage_ns": "viz-test",
                "ns": ["viz-test"],
                "capabilities": ["read_graph"]
            }),
        },
    )
    .await;
    assert_eq!(response.status, StatusCode::OK.as_u16());
    let payload: Value = serde_json::from_slice(&response.body).unwrap();
    let mut returned_ids = payload["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|node| node["id"].as_str())
        .collect::<Vec<_>>();
    returned_ids.sort_unstable();
    assert_eq!(
        returned_ids,
        fixture["expected_matches_by_kind"]["node"]
            .as_array()
            .unwrap()
            .iter()
            .map(|id| id.as_str().unwrap())
            .collect::<Vec<_>>()
    );

    let secret = "postgres-visualization-public-route-test-secret";
    let auth = AuthConfig {
        mode: "jwt".to_owned(),
        algorithm: Some("HS256".to_owned()),
        key: Some(secret.to_owned()),
        ..AuthConfig::default()
    };
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let token = |capabilities: &[&str]| {
        encode(
            &Header::new(Algorithm::HS256),
            &json!({
                "sub": "alice",
                "role": "rw",
                "ns": ["viz-test"],
                "storage_ns": "viz-test",
                "capabilities": capabilities,
                "exp": now + 300
            }),
            &EncodingKey::from_secret(secret.as_bytes()),
        )
        .unwrap()
    };
    let app = router_with_application(
        ApiState {
            health: snapshot(),
            required_roles: vec![],
            implementation: ImplementationSnapshot::default(),
            auth,
        },
        Arc::new(service.clone()),
    );
    let query = format!(
        "doc_id={}&mode=reify&insertion_method={}&graph_type=knowledge",
        fixture["doc_id"].as_str().unwrap(),
        fixture["insertion_method"].as_str().unwrap()
    );
    let mut expected_node_ids = fixture["expected_matches_by_kind"]["node"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>();
    expected_node_ids.sort_unstable();
    for (path, result_key) in [
        ("/api/viz/d3.json", "nodes"),
        ("/api/viz/cytoscape.json", "elements"),
    ] {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri(format!("{path}?{query}"))
                    .header(
                        "authorization",
                        format!("Bearer {}", token(&["read_graph"])),
                    )
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK, "{path}");
        let payload: Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 65_536).await.unwrap()).unwrap();
        let ids = payload[result_key]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|item| {
                item.get("id")
                    .and_then(Value::as_str)
                    .or_else(|| item.get("data").and_then(|data| data["id"].as_str()))
            })
            .collect::<Vec<_>>();
        let mut returned_node_ids = ids
            .iter()
            .copied()
            .filter(|id| id.starts_with('n'))
            .collect::<Vec<_>>();
        returned_node_ids.sort_unstable();
        assert_eq!(returned_node_ids, expected_node_ids, "{path}");
        assert!(!ids.contains(&"n-private"), "{path}: {ids:?}");
    }

    let bundle = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(format!("/viz/d3.bundle?{query}"))
                .header(
                    "authorization",
                    format!("Bearer {}", token(&["read_graph"])),
                )
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(bundle.status(), StatusCode::OK);
    assert_eq!(bundle.headers()["content-type"], "text/html; charset=utf-8");
    let bundle = String::from_utf8(
        to_bytes(bundle.into_body(), 1_048_576)
            .await
            .unwrap()
            .to_vec(),
    )
    .unwrap();
    let embedded = bundle
        .split_once("window.__EMBEDDED_DATA__ = ")
        .and_then(|(_, rest)| rest.split_once(';'))
        .map(|(payload, _)| serde_json::from_str::<Value>(payload).unwrap())
        .expect("public bundle contains serialized graph data");
    let bundle_ids = embedded["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|node| node["id"].as_str())
        .collect::<Vec<_>>();
    let mut returned_bundle_node_ids = bundle_ids
        .iter()
        .copied()
        .filter(|id| id.starts_with('n'))
        .collect::<Vec<_>>();
    returned_bundle_node_ids.sort_unstable();
    assert_eq!(returned_bundle_node_ids, expected_node_ids);
    assert!(!bundle_ids.contains(&"n-private"));

    for path in [
        format!("/api/viz/d3.json?{query}"),
        format!("/api/viz/cytoscape.json?{query}"),
        format!("/viz/d3.bundle?{query}"),
    ] {
        let denied = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri(path)
                    .header("authorization", format!("Bearer {}", token(&[])))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(denied.status(), StatusCode::FORBIDDEN);
    }

    drop(service);
    let (cleanup, connection) = tokio_postgres::connect(&dsn, tokio_postgres::NoTls)
        .await
        .unwrap();
    let cleanup_task = tokio::spawn(connection);
    cleanup
        .batch_execute(&format!(
            "DROP SCHEMA \"{knowledge_schema}\" CASCADE; \
             DROP SCHEMA \"{conversation_schema}\" CASCADE; \
             DROP SCHEMA \"{workflow_schema}\" CASCADE; \
             DROP SCHEMA \"{wisdom_schema}\" CASCADE"
        ))
        .await
        .unwrap();
    drop(cleanup);
    cleanup_task.await.unwrap().unwrap();
}

#[tokio::test]
async fn in_memory_vector_query_matches_shared_python_acl_contract() {
    let fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/graph-acl-vector-query-v1.json"
    )))
    .unwrap();
    assert_eq!(fixture["schema_version"], 1);

    let secret = "in-memory-vector-acl-contract-secret";
    let auth = AuthConfig {
        mode: "jwt".to_owned(),
        algorithm: Some("HS256".to_owned()),
        key: Some(secret.to_owned()),
        ..AuthConfig::default()
    };
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let token = |subject: &str, capabilities: &[&str]| {
        encode(
            &Header::new(Algorithm::HS256),
            &json!({
                "sub": subject,
                "role": "rw",
                "ns": ["docs"],
                "storage_ns": fixture["scope"]["namespace"],
                "workspace": fixture["scope"]["workspace_id"],
                "workspace_id": fixture["scope"]["workspace_id"],
                "capabilities": capabilities,
                "exp": now + 300
            }),
            &EncodingKey::from_secret(secret.as_bytes()),
        )
        .unwrap()
    };
    let app = router_with_application(
        ApiState {
            health: snapshot(),
            required_roles: vec![],
            implementation: ImplementationSnapshot::default(),
            auth,
        },
        Arc::new(InMemoryGraphApplicationService::default()),
    );
    let scope = json!({
        "namespace": fixture["scope"]["namespace"],
        "workspace_id": fixture["scope"]["workspace_id"],
        "graph_space": null
    });
    let mut mutations = Vec::new();
    for (kind, table, records) in [
        ("node", "gke_nodes", &fixture["nodes"]),
        ("edge", "gke_edges", &fixture["edges"]),
    ] {
        for spec in records.as_array().unwrap() {
            let id = spec["id"].as_str().unwrap();
            let mut metadata = serde_json::Map::new();
            metadata.insert("entity_type".to_owned(), json!(kind));
            if kind == "node" {
                metadata.insert("label".to_owned(), json!(id));
            } else {
                metadata.insert("source_ids".to_owned(), spec["source_ids"].clone());
                metadata.insert("target_ids".to_owned(), spec["target_ids"].clone());
            }
            mutations.push(json!({
                "scope": scope,
                "table": table,
                "entity_kind": kind,
                "event_id": format!("{kind}-{id}"),
                "op": "ADD",
                "payload": {"id": id},
                "record": {
                    "id": id,
                    "document": id,
                    "metadata": metadata,
                    "embedding": spec["embedding"]
                },
                "embedding_dim": 2
            }));

            let mut acl_metadata = serde_json::Map::new();
            for (key, value) in [
                ("entity_type", json!("acl_record")),
                ("acl_truth_graph", fixture["truth_graph"].clone()),
                ("acl_target_entity_id", json!(id)),
                ("acl_target_grain", json!(kind)),
                ("acl_version", json!(1)),
                ("acl_mode", spec["mode"].clone()),
            ] {
                acl_metadata.insert(key.to_owned(), value);
            }
            if let Some(owner_id) = spec["owner_id"].as_str() {
                acl_metadata.insert("owner_id".to_owned(), json!(owner_id));
            }
            let acl_id = format!("acl-{kind}-{id}");
            mutations.push(json!({
                "scope": scope,
                "table": "gke_nodes",
                "entity_kind": "node",
                "event_id": format!("event-{acl_id}"),
                "op": "ADD",
                "payload": {"id": acl_id},
                "record": {
                    "id": acl_id,
                    "document": null,
                    "metadata": acl_metadata,
                    "embedding": null
                },
                "embedding_dim": 0
            }));
        }
    }
    let write = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/internal/graph/mutations")
                .header("content-type", "application/json")
                .header(
                    "authorization",
                    format!("Bearer {}", token("owner-b", &["write_graph"])),
                )
                .body(Body::from(
                    json!({"contract_version":1,"atomic":true,"mutations":mutations,"index_jobs":[]})
                        .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(write.status(), StatusCode::OK);

    for principal in fixture["principals"].as_array().unwrap() {
        let subject = principal["id"].as_str().unwrap();
        for (kind, expected_key) in [("node", "nodes"), ("edge", "edges")] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/internal/graph/vector-query")
                        .header("content-type", "application/json")
                        .header(
                            "authorization",
                            format!("Bearer {}", token(subject, &["read_graph"])),
                        )
                        .body(Body::from(
                            json!({
                                "entity_kind": kind,
                                "query_embeddings": fixture["query_embeddings"],
                                "metric": fixture["metric"],
                                "truth_graph": fixture["truth_graph"],
                                "n_results": fixture["n_results"],
                                "include": ["distances"]
                            })
                            .to_string(),
                        ))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let result: Value =
                serde_json::from_slice(&to_bytes(response.into_body(), 65_536).await.unwrap())
                    .unwrap();
            assert_eq!(result["implementation_authority"], "rust");
            assert_eq!(result["seam_attestation_version"], 1);
            assert_eq!(result["seam_crossings"], json!([]));
            assert_eq!(
                result["ids"], principal["expected"][expected_key],
                "unexpected {kind} candidates for {subject}"
            );
            let expected_distances = &principal["expected"]["distances"][expected_key];
            let actual_distances = result["distances"].as_array().unwrap();
            assert_eq!(
                actual_distances.len(),
                expected_distances.as_array().unwrap().len()
            );
            for (actual, expected_values) in actual_distances
                .iter()
                .zip(expected_distances.as_array().unwrap())
            {
                let actual = actual.as_array().unwrap();
                let expected_values = expected_values.as_array().unwrap();
                assert_eq!(actual.len(), expected_values.len());
                for (actual, expected) in actual.iter().zip(expected_values) {
                    assert!(
                        (actual.as_f64().unwrap() - expected.as_f64().unwrap()).abs() < 1e-5,
                        "unexpected {kind} vector distance for {subject}"
                    );
                }
            }
        }
    }
}

#[test]
fn postgres_graph_type_stores_follow_python_schema_partitioning() {
    let service = PostgresRunApplicationService::from_dsn_with_graph_schemas(
        "postgresql://user:password@localhost/kogwistar",
        "kg_knowledge",
        "kg_conversation",
        "kg_workflow",
        "kg_wisdom",
    )
    .expect("valid PostgreSQL configuration should construct graph stores without connecting");

    assert_eq!(
        service.graph_store_for_type("knowledge").schema(),
        "kg_knowledge"
    );
    assert_eq!(
        service.graph_store_for_type("conversation").schema(),
        "kg_conversation"
    );
    assert_eq!(
        service.graph_store_for_type("workflow").schema(),
        "kg_workflow"
    );
    assert_eq!(service.graph_store_for_type("wisdom").schema(), "kg_wisdom");
    assert_eq!(
        service.graph_store_for_type("CONVERSATION").schema(),
        "kg_conversation"
    );

    let legacy = PostgresRunApplicationService::from_dsn_with_wisdom_schema(
        "postgresql://user:password@localhost/kogwistar",
        "legacy_graph",
        "legacy_wisdom",
    )
    .expect("legacy constructor remains supported");
    assert_eq!(legacy.store.schema(), "legacy_graph");
    assert_eq!(
        legacy.conversation_store.schema(),
        "legacy_graph_conversation"
    );
    assert_eq!(legacy.workflow_store.schema(), "legacy_graph_workflow");
    assert_eq!(legacy.wisdom_store.schema(), "legacy_wisdom");
}

#[tokio::test]
#[ignore = "requires KOGWISTAR_TEST_PG_DSN; uses disposable PostgreSQL schemas"]
async fn postgres_graph_mutation_batch_uses_configured_graph_type_store() {
    let dsn = std::env::var("KOGWISTAR_TEST_PG_DSN")
        .expect("ignored PostgreSQL test requires KOGWISTAR_TEST_PG_DSN");
    let base = format!("rust_graph_role_{}", uuid::Uuid::new_v4().simple());
    let schemas = [
        format!("{base}_knowledge"),
        format!("{base}_conversation"),
        format!("{base}_workflow"),
        format!("{base}_wisdom"),
    ];
    let service = PostgresRunApplicationService::from_dsn_with_graph_schemas(
        &dsn,
        &schemas[0],
        &schemas[1],
        &schemas[2],
        &schemas[3],
    )
    .unwrap()
    .with_graph_role("conversation")
    .unwrap();
    service.ensure_schema().await.unwrap();
    for store in [
        &service.store,
        &service.conversation_store,
        &service.workflow_store,
        &service.wisdom_store,
    ] {
        store
            .create_graph_schema(2, kogwistar_store_postgres::GraphTableNames::default())
            .await
            .unwrap();
    }

    let mut batch: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/graph-mutation-batch-v1.json"
    )))
    .unwrap();
    batch["mutations"][0]["scope"] = json!({
        "namespace": "tenant-a",
        "workspace_id": "ws",
        "graph_space": "facts"
    });
    batch["mutations"][0]["record"]["id"] = json!("conversation-node");
    batch["mutations"][0]["payload"]["id"] = json!("conversation-node");
    batch["mutations"][0]["event_id"] = json!("conversation-node-event");

    let response = ApplicationService::execute(
        &service,
        ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: INTERNAL_GRAPH_MUTATIONS_ROUTE.to_owned(),
            body: serde_json::to_vec(&batch).unwrap(),
            principal: json!({
                "sub": "alice",
                "role": "rw",
                "ns": ["tenant-a"],
                "storage_ns": "tenant-a",
                "workspace_id": "ws",
                "graph_space": "facts",
                "capabilities": ["write_graph"]
            }),
        },
    )
    .await;
    assert_eq!(
        response.status,
        StatusCode::OK.as_u16(),
        "{}",
        String::from_utf8_lossy(&response.body)
    );

    let scope = GraphScope {
        namespace: "tenant-a".to_owned(),
        workspace_id: Some("ws".to_owned()),
        graph_space: Some("facts".to_owned()),
    };
    let read = |id: &str| GraphProjectionRead {
        scope: scope.clone(),
        table: "gke_nodes".to_owned(),
        ids: Some(vec![id.to_owned()]),
        metadata: MetadataFilter::default(),
        limit: 1,
    };
    let conversation_rows = service
        .conversation_store
        .graph_projection_records(read("conversation-node"))
        .await
        .unwrap();
    let knowledge_rows = service
        .store
        .graph_projection_records(read("conversation-node"))
        .await
        .unwrap();
    assert_eq!(conversation_rows.len(), 1);
    assert!(knowledge_rows.is_empty());

    drop(service);
    let (cleanup, connection) = tokio_postgres::connect(&dsn, tokio_postgres::NoTls)
        .await
        .unwrap();
    let cleanup_task = tokio::spawn(connection);
    cleanup
        .batch_execute(
            &schemas
                .iter()
                .map(|schema| format!("DROP SCHEMA \"{schema}\" CASCADE"))
                .collect::<Vec<_>>()
                .join("; "),
        )
        .await
        .unwrap();
    drop(cleanup);
    cleanup_task.await.unwrap().unwrap();
}

#[tokio::test]
async fn in_memory_mcp_traversal_matches_shared_python_contract() {
    let fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/graph-traversal-v1.json"
    )))
    .unwrap();
    assert_eq!(fixture["schema_version"], 1);

    let secret = "in-memory-traversal-contract-secret";
    let auth = AuthConfig {
        mode: "jwt".to_owned(),
        algorithm: Some("HS256".to_owned()),
        key: Some(secret.to_owned()),
        ..AuthConfig::default()
    };
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let token = encode(
        &Header::new(Algorithm::HS256),
        &json!({
            "sub":"fixture-reader",
            "role":"rw",
            "ns":["docs"],
            "storage_ns":"fixture-space",
            "workspace":"fixture-workspace",
            "workspace_id":"fixture-workspace",
            "graph_space":"fixture-graph",
            "capabilities":["read_graph", "write_graph"],
            "exp":now + 300
        }),
        &EncodingKey::from_secret(secret.as_bytes()),
    )
    .unwrap();
    let app = router_with_application(
        ApiState {
            health: snapshot(),
            required_roles: vec![],
            implementation: ImplementationSnapshot::default(),
            auth,
        },
        Arc::new(InMemoryGraphApplicationService::default()),
    );
    let scope = json!({
        "namespace":"fixture-space",
        "workspace_id":"fixture-workspace",
        "graph_space":"fixture-graph"
    });
    let mut mutations = Vec::new();
    for node_id in fixture["graph"]["nodes"].as_array().unwrap() {
        let id = node_id.as_str().unwrap();
        let label = fixture["graph"]["node_labels"][id].as_str().unwrap();
        mutations.push(json!({
            "scope":scope,
            "table":"gke_nodes",
            "entity_kind":"node",
            "event_id":format!("fixture-node-{id}"),
            "op":"ADD",
            "payload":{"id":id,"metadata":{"label":label,"doc_id":"D1"}},
            "record":{
                "id":id,
                "document":json!({"label":label,"doc_id":"D1"}).to_string(),
                "metadata":{"label":label,"doc_id":"D1"},
                "embedding":null
            },
            "embedding_dim":0
        }));
    }
    for edge in fixture["graph"]["edges"].as_array().unwrap() {
        let id = edge["id"].as_str().unwrap();
        let metadata = json!({
            "relation":edge["relation"],
            "source_ids":edge["source_ids"],
            "target_ids":edge["target_ids"],
            "doc_id":edge["doc_id"]
        });
        mutations.push(json!({
            "scope":scope,
            "table":"gke_edges",
            "entity_kind":"edge",
            "event_id":format!("fixture-edge-{id}"),
            "op":"ADD",
            "payload":{"id":id,"metadata":metadata},
            "record":{"id":id,"document":metadata.to_string(),"metadata":metadata,"embedding":null},
            "embedding_dim":0
        }));
    }
    for (grain, id) in [("node", "A"), ("node", "B"), ("edge", "E1")] {
        let metadata = json!({
            "entity_type":"acl_record",
            "acl_truth_graph":"knowledge",
            "acl_target_grain":grain,
            "acl_target_entity_id":id,
            "acl_version":1,
            "acl_mode":"public"
        });
        mutations.push(json!({
            "scope":scope,
            "table":"gke_nodes",
            "entity_kind":"node",
            "event_id":format!("fixture-acl-{grain}-{id}"),
            "op":"ADD",
            "payload":{"id":format!("acl-{grain}-{id}"),"metadata":metadata},
            "record":{"id":format!("acl-{grain}-{id}"),"document":null,"metadata":metadata,"embedding":null},
            "embedding_dim":0
        }));
    }
    let write = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/internal/graph/mutations")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {token}"))
                .body(Body::from(
                    json!({"contract_version":1,"atomic":true,"mutations":mutations,"index_jobs":[]})
                        .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(write.status(), StatusCode::OK);

    let cases = fixture["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|case| (case["name"].as_str().unwrap(), case))
        .collect::<std::collections::BTreeMap<_, _>>();
    let call = |name: &str, arguments: Value| {
        let app = app.clone();
        let token = token.clone();
        let name = name.to_owned();
        async move {
            let response = app
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/mcp")
                        .header("content-type", "application/json")
                        .header("authorization", format!("Bearer {token}"))
                        .body(Body::from(
                            json!({
                                "jsonrpc":"2.0",
                                "id":name,
                                "method":"tools/call",
                                "params":{"name":name,"arguments":arguments}
                            })
                            .to_string(),
                        ))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let response: Value =
                serde_json::from_slice(&to_bytes(response.into_body(), 65_536).await.unwrap())
                    .unwrap();
            assert!(response.get("error").is_none(), "{name}: {response}");
            assert_eq!(response["result"]["implementation_authority"], "rust");
            assert_eq!(response["result"]["seam_attestation_version"], 1);
            assert_eq!(response["result"]["seam_crossings"], json!([]));
            let mut result = response["result"].clone();
            for field in [
                "implementation_authority",
                "seam_attestation_version",
                "seam_crossings",
            ] {
                result.as_object_mut().unwrap().remove(field);
            }
            result
        }
    };

    let edge_case = cases["edge_to_nodes"];
    assert_eq!(
        call(
            "kg_neighbors",
            json!({"rid":edge_case["rid"],"doc_id":"D1"})
        )
        .await,
        edge_case["expected_neighbors"]
    );
    let path_case = cases["shortest_path"];
    assert_eq!(
        call(
            "kg_shortest_path",
            json!({"src_id":path_case["src_id"],"dst_id":path_case["dst_id"],"doc_id":"D1"})
        )
        .await["path"],
        path_case["expected_path"]
    );
    let hop_case = cases["k_hop"];
    assert_eq!(
        call(
            "kg_k_hop",
            json!({"start_ids":hop_case["start_ids"],"k":hop_case["k"],"doc_id":"D1"})
        )
        .await["layers"],
        hop_case["expected_layers"]
    );
    let find_case = cases["find_edges"];
    assert_eq!(
        call(
            "kg_find_edges",
            json!({
                "relation":find_case["relation"],
                "src_label_contains":find_case["source_label_contains"],
                "tgt_label_contains":find_case["target_label_contains"],
                "doc_id":find_case["doc_id"]
            })
        )
        .await["edges"],
        find_case["expected_edge_ids"]
    );
}

#[tokio::test]
async fn in_memory_mcp_acl_traversal_matches_shared_python_contract() {
    let fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/graph-acl-traversal-v1.json"
    )))
    .unwrap();
    assert_eq!(fixture["schema_version"], 1);

    let secret = "in-memory-acl-traversal-contract-secret";
    let auth = AuthConfig {
        mode: "jwt".to_owned(),
        algorithm: Some("HS256".to_owned()),
        key: Some(secret.to_owned()),
        ..AuthConfig::default()
    };
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let token = |subject: &str| {
        encode(
            &Header::new(Algorithm::HS256),
            &json!({
                "sub":subject,
                "role":"rw",
                "ns":["docs"],
                "storage_ns":"acl-fixture-space",
                "workspace":"acl-fixture-workspace",
                "workspace_id":"acl-fixture-workspace",
                "graph_space":"acl-fixture-graph",
                "capabilities":["read_graph", "write_graph"],
                "exp":now + 300
            }),
            &EncodingKey::from_secret(secret.as_bytes()),
        )
        .unwrap()
    };
    let app = router_with_application(
        ApiState {
            health: snapshot(),
            required_roles: vec![],
            implementation: ImplementationSnapshot::default(),
            auth,
        },
        Arc::new(InMemoryGraphApplicationService::default()),
    );
    let scope = json!({
        "namespace":"acl-fixture-space",
        "workspace_id":"acl-fixture-workspace",
        "graph_space":"acl-fixture-graph"
    });
    let mut mutations = Vec::new();
    let mut nodes = Vec::new();
    for node in fixture["nodes"].as_array().unwrap() {
        let id = node["id"].as_str().unwrap();
        let mut metadata = json!({
            "label": if id == "source-public" { "Public source" } else { "Private target" },
            "doc_id": fixture["doc_id"]
        });
        if let Some(owner_id) = node["owner_id"].as_str() {
            metadata["owner_id"] = json!(owner_id);
        }
        mutations.push(json!({
            "scope":scope,
            "table":"gke_nodes",
            "entity_kind":"node",
            "event_id":format!("acl-fixture-node-{id}"),
            "op":"ADD",
            "payload":{"id":id,"metadata":metadata},
            "record":{
                "id":id,
                "document":metadata.to_string(),
                "metadata":metadata,
                "embedding":null
            },
            "embedding_dim":0
        }));
        let mut acl = json!({
            "entity_type":"acl_record",
            "acl_truth_graph":fixture["truth_graph"],
            "acl_target_grain":"node",
            "acl_target_entity_id":id,
            "acl_version":1,
            "acl_mode":node["mode"]
        });
        if let Some(owner_id) = node["owner_id"].as_str() {
            acl["owner_id"] = json!(owner_id);
        }
        nodes.push((format!("acl-node-{id}"), acl));
    }
    let edge = &fixture["edge"];
    let edge_id = edge["id"].as_str().unwrap();
    let edge_metadata = json!({
        "relation":edge["relation"],
        "source_ids":edge["source_ids"],
        "target_ids":edge["target_ids"],
        "doc_id":fixture["doc_id"]
    });
    mutations.push(json!({
        "scope":scope,
        "table":"gke_edges",
        "entity_kind":"edge",
        "event_id":format!("acl-fixture-edge-{edge_id}"),
        "op":"ADD",
        "payload":{"id":edge_id,"metadata":edge_metadata},
        "record":{
            "id":edge_id,
            "document":edge_metadata.to_string(),
            "metadata":edge_metadata,
            "embedding":null
        },
        "embedding_dim":0
    }));
    let mut edge_acl = json!({
        "entity_type":"acl_record",
        "acl_truth_graph":fixture["truth_graph"],
        "acl_target_grain":"edge",
        "acl_target_entity_id":edge_id,
        "acl_version":1,
        "acl_mode":edge["mode"]
    });
    if let Some(owner_id) = edge["owner_id"].as_str() {
        edge_acl["owner_id"] = json!(owner_id);
    }
    nodes.push((format!("acl-edge-{edge_id}"), edge_acl));
    for (id, metadata) in nodes {
        mutations.push(json!({
            "scope":scope,
            "table":"gke_nodes",
            "entity_kind":"node",
            "event_id":format!("acl-fixture-record-{id}"),
            "op":"ADD",
            "payload":{"id":id,"metadata":metadata},
            "record":{"id":id,"document":null,"metadata":metadata,"embedding":null},
            "embedding_dim":0
        }));
    }
    let write = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/internal/graph/mutations")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {}", token("owner-a")))
                .body(Body::from(
                    json!({"contract_version":1,"atomic":true,"mutations":mutations,"index_jobs":[]})
                        .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(write.status(), StatusCode::OK);

    for principal in fixture["principals"].as_array().unwrap() {
        let subject = principal["id"].as_str().unwrap();
        let principal_token = token(subject);
        let call = |name: &str, arguments: Value| {
            let app = app.clone();
            let token = principal_token.clone();
            let name = name.to_owned();
            async move {
                let response = app
                    .oneshot(
                        Request::builder()
                            .method("POST")
                            .uri("/mcp")
                            .header("content-type", "application/json")
                            .header("authorization", format!("Bearer {token}"))
                            .body(Body::from(
                                json!({
                                    "jsonrpc":"2.0",
                                    "id":name,
                                    "method":"tools/call",
                                    "params":{"name":name,"arguments":arguments}
                                })
                                .to_string(),
                            ))
                            .unwrap(),
                    )
                    .await
                    .unwrap();
                assert_eq!(response.status(), StatusCode::OK);
                let response: Value =
                    serde_json::from_slice(&to_bytes(response.into_body(), 65_536).await.unwrap())
                        .unwrap();
                assert!(response.get("error").is_none(), "{name}: {response}");
                assert_eq!(response["result"]["implementation_authority"], "rust");
                assert_eq!(response["result"]["seam_attestation_version"], 1);
                assert_eq!(response["result"]["seam_crossings"], json!([]));
                let mut result = response["result"].clone();
                for field in [
                    "implementation_authority",
                    "seam_attestation_version",
                    "seam_crossings",
                ] {
                    result.as_object_mut().unwrap().remove(field);
                }
                result
            }
        };
        let expected_neighbors = &principal["expected_neighbors"];
        assert_eq!(
            call(
                "kg_neighbors",
                json!({"rid":"source-public","doc_id":fixture["doc_id"]})
            )
            .await,
            *expected_neighbors,
            "principal={subject}"
        );
        assert_eq!(
            call(
                "kg_k_hop",
                json!({"start_ids":["source-public"],"k":2,"doc_id":fixture["doc_id"]})
            )
            .await["layers"],
            principal["expected_k_hop"],
            "principal={subject}"
        );
        assert_eq!(
            call(
                "kg_shortest_path",
                json!({
                    "src_id":"source-public",
                    "dst_id":"target-private",
                    "max_depth":8,
                    "doc_id":fixture["doc_id"]
                })
            )
            .await["path"],
            principal["expected_shortest_path"],
            "principal={subject}"
        );
        assert_eq!(
            call(
                "kg_find_edges",
                json!({"relation":edge["relation"],"doc_id":fixture["doc_id"]})
            )
            .await["edges"],
            principal["expected_find_edges"],
            "principal={subject}"
        );
    }
}
