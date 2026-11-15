#[tokio::test]
async fn in_memory_create_conversation_is_atomic_and_matches_shared_contract() {
    let service = InMemoryGraphApplicationService::default();
    let path = "/api/syscall/v1/create_conversation";
    assert!(service.supports_http_operation("POST", path));
    let principal = json!({
        "sub":"alice",
        "role":"rw",
        "storage_ns":"conversation",
        "ns":["conversation"],
        "graph_space":"conversation"
    });
    let request = |args: Value| ApiEffectRequest {
        contract_version: 1,
        method: "POST".to_owned(),
        path_and_query: path.to_owned(),
        body: json!({"version":"v1","args":args}).to_string().into_bytes(),
        principal: principal.clone(),
    };
    let body: Value = serde_json::from_str(include_str!(
        "../../../../../contracts/golden/conversation-create-syscall-result-v1.json"
    ))
    .unwrap();
    let args = json!({
        "user_id":"alice",
        "conversation_id":body["conversation_id"],
        "start_node_id":body["start_node_id"]
    });
    let spoofed_owner = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: path.to_owned(),
            body: json!({"version":"v1","args":{
                "user_id":"mallory",
                "conversation_id":"spoofed-conversation",
                "start_node_id":"spoofed-start"
            }})
            .to_string()
            .into_bytes(),
            principal: principal.clone(),
        })
        .await;
    assert_eq!(spoofed_owner.status, StatusCode::FORBIDDEN.as_u16());
    let response = service.execute(request(args.clone())).await;
    assert_eq!(response.status, StatusCode::OK.as_u16());
    let response: Value = serde_json::from_slice(&response.body).unwrap();
    assert_eq!(response["op"], "create_conversation");
    assert_eq!(response["result"], body);

    let scope = GraphScope {
        namespace: "conversation".to_owned(),
        workspace_id: Some(String::new()),
        graph_space: Some("conversation".to_owned()),
    };
    let mut start_filter = MetadataFilter::default();
    start_filter
        .equals
        .insert("entity_type".to_owned(), json!("conversation_start"));
    start_filter
        .equals
        .insert("conversation_id".to_owned(), body["conversation_id"].clone());
    let starts = service
        .store
        .graph_projection_records(GraphProjectionRead {
            scope: scope.clone(),
            table: "gke_nodes".to_owned(),
            ids: None,
            metadata: start_filter,
            limit: 10,
        })
        .await
        .unwrap();
    assert_eq!(starts.len(), 1);
    assert_eq!(starts[0].id, body["start_node_id"]);

    let mut acl_filter = MetadataFilter::default();
    acl_filter
        .equals
        .insert("entity_type".to_owned(), json!("acl_record"));
    acl_filter
        .equals
        .insert("acl_truth_graph".to_owned(), json!("conversation"));
    acl_filter
        .equals
        .insert("acl_target_entity_id".to_owned(), body["start_node_id"].clone());
    let acl_rows = service
        .store
        .graph_projection_records(GraphProjectionRead {
            scope: scope.clone(),
            table: "gke_nodes".to_owned(),
            ids: None,
            metadata: acl_filter,
            limit: 10,
        })
        .await
        .unwrap();
    assert!(!acl_rows.is_empty());
    assert!(acl_rows
        .iter()
        .any(|row| row.metadata.get("owner_id").and_then(Value::as_str) == Some("alice")));
    let jobs = kogwistar_store::IndexJobReadStore::index_jobs(
        &service.store,
        Some("conversation"),
        Some("PENDING"),
        Some("node"),
        Some(body["start_node_id"].as_str().unwrap()),
        Some("node_embedding"),
        10,
    )
    .await
    .unwrap();
    assert_eq!(jobs.len(), 1);

    let duplicate = service.execute(request(args)).await;
    assert_eq!(duplicate.status, StatusCode::CONFLICT.as_u16());
    assert_eq!(
        serde_json::from_slice::<Value>(&duplicate.body).unwrap()["code"],
        "KOGWISTAR_CONVERSATION_CREATE_CONFLICT"
    );
    let start_id_collision = service
        .execute(request(json!({
            "user_id":"alice",
            "conversation_id":"different-conversation",
            "start_node_id":body["start_node_id"]
        })))
        .await;
    assert_eq!(start_id_collision.status, StatusCode::CONFLICT.as_u16());
    let other_workspace_duplicate = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: path.to_owned(),
            body: json!({
                "version":"v1",
                "args":{
                    "user_id":"alice",
                    "conversation_id":body["conversation_id"],
                    "start_node_id":"other-workspace-start"
                }
            })
            .to_string()
            .into_bytes(),
            principal: json!({
                "sub":"alice",
                "role":"rw",
                "storage_ns":"conversation",
                "workspace_id":"workspace-b",
                "ns":["conversation"],
                "graph_space":"conversation"
            }),
        })
        .await;
    assert_eq!(
        other_workspace_duplicate.status,
        StatusCode::CONFLICT.as_u16()
    );

    let concurrent_args = |start_node_id: &str| {
        json!({
            "user_id":"alice",
            "conversation_id":"conversation-race",
            "start_node_id":start_node_id
        })
    };
    let (left, right) = tokio::join!(
        service.execute(request(concurrent_args("conversation-race-start-a"))),
        service.execute(request(concurrent_args("conversation-race-start-b")))
    );
    let statuses = [left.status, right.status];
    assert_eq!(statuses.iter().filter(|status| **status == 200).count(), 1);
    assert_eq!(statuses.iter().filter(|status| **status == 409).count(), 1);
    let mut race_filter = MetadataFilter::default();
    race_filter
        .equals
        .insert("entity_type".to_owned(), json!("conversation_start"));
    race_filter
        .equals
        .insert("conversation_id".to_owned(), json!("conversation-race"));
    let race_starts = service
        .store
        .graph_projection_records(GraphProjectionRead {
            scope,
            table: "gke_nodes".to_owned(),
            ids: None,
            metadata: race_filter,
            limit: 10,
        })
        .await
        .unwrap();
    assert_eq!(race_starts.len(), 1);
}

#[tokio::test]
async fn in_memory_receive_message_matches_shared_python_contract() {
    let service = InMemoryGraphApplicationService::default();
    let path = "/api/syscall/v1/receive_message";
    assert!(service.supports_http_operation("POST", path));

    let scope = GraphScope {
        namespace: "scope-a".to_owned(),
        workspace_id: Some("workspace-a".to_owned()),
        graph_space: Some("conversation".to_owned()),
    };
    let mut mutations = Vec::new();
    let mut add_node = |id: &str, metadata: Value, node_scope: GraphScope| {
        mutations.push(GraphMutation {
            scope: node_scope,
            table: "gke_nodes".to_owned(),
            entity_kind: "node".to_owned(),
            event_id: format!("event-{id}"),
            op: "ADD".to_owned(),
            payload: json!({"id":id,"metadata":metadata}),
            record: GraphRecord {
                id: id.to_owned(),
                document: Some("{}".to_owned()),
                metadata: serde_json::from_value(metadata).unwrap(),
                embedding: None,
            },
            embedding_dim: 0,
        });
    };
    for (id, turn_index, role, summary) in [
        ("turn-1", 1, "user", "first"),
        ("turn-2", 2, "assistant", "second"),
    ] {
        add_node(
            id,
            json!({
                "entity_type":"conversation_turn",
                "conversation_id":"c1",
                "turn_index":turn_index,
                "role":role,
                "summary":summary
            }),
            scope.clone(),
        );
        add_node(
            &format!("acl-{id}"),
            json!({
                "entity_type":"acl_record",
                "acl_truth_graph":"conversation",
                "acl_target_grain":"node",
                "acl_target_entity_id":id,
                "acl_version":1,
                "acl_mode":"public"
            }),
            scope.clone(),
        );
    }
    add_node(
        "turn-private",
        json!({
            "entity_type":"conversation_turn",
            "conversation_id":"c1",
            "turn_index":3,
            "role":"assistant",
            "summary":"private"
        }),
        scope.clone(),
    );
    add_node(
        "acl-turn-private",
        json!({
            "entity_type":"acl_record",
            "acl_truth_graph":"conversation",
            "acl_target_grain":"node",
            "acl_target_entity_id":"turn-private",
            "acl_version":1,
            "acl_mode":"private",
            "owner_id":"someone-else"
        }),
        scope.clone(),
    );
    GraphMutationStore::apply_graph_mutations_atomic(&service.store, mutations)
        .await
        .unwrap();

    let mut out_of_scope_mutations = Vec::new();
    let mut add_out_of_scope_node = |id: &str, metadata: Value, node_scope: GraphScope| {
        out_of_scope_mutations.push(GraphMutation {
            scope: node_scope,
            table: "gke_nodes".to_owned(),
            entity_kind: "node".to_owned(),
            event_id: format!("event-{id}"),
            op: "ADD".to_owned(),
            payload: json!({"id":id,"metadata":metadata}),
            record: GraphRecord {
                id: id.to_owned(),
                document: Some("{}".to_owned()),
                metadata: serde_json::from_value(metadata).unwrap(),
                embedding: None,
            },
            embedding_dim: 0,
        });
    };
    let outside_scope = GraphScope {
        workspace_id: Some("workspace-other".to_owned()),
        ..scope.clone()
    };
    add_out_of_scope_node(
        "turn-outside-scope",
        json!({
            "entity_type":"conversation_turn",
            "conversation_id":"c1",
            "turn_index":4,
            "role":"assistant",
            "summary":"out of scope"
        }),
        outside_scope.clone(),
    );
    add_out_of_scope_node(
        "acl-turn-outside-scope",
        json!({
            "entity_type":"acl_record",
            "acl_truth_graph":"conversation",
            "acl_target_grain":"node",
            "acl_target_entity_id":"turn-outside-scope",
            "acl_version":1,
            "acl_mode":"public"
        }),
        outside_scope,
    );
    GraphMutationStore::apply_graph_mutations_atomic(&service.store, out_of_scope_mutations)
        .await
        .unwrap();

    let request = |args: Value, principal: Value| ApiEffectRequest {
        contract_version: 1,
        method: "POST".to_owned(),
        path_and_query: path.to_owned(),
        body: json!({"version":"v1","args":args}).to_string().into_bytes(),
        principal,
    };
    let principal = json!({
        "sub":"reader-1",
        "role":"ro",
        "storage_ns":"scope-a",
        "workspace_id":"workspace-a",
        "graph_space":"conversation",
        "ns":["conversation"],
        "capabilities":["read_graph"]
    });
    let denied = service
        .execute(request(
            json!({"conversation_id":"c1"}),
            json!({"ns":["conversation"],"capabilities":[]}),
        ))
        .await;
    assert_eq!(denied.status, StatusCode::FORBIDDEN.as_u16());

    let wrong_namespace = service
        .execute(request(
            json!({"conversation_id":"c1"}),
            json!({"ns":["docs"],"capabilities":["read_graph"]}),
        ))
        .await;
    assert_eq!(wrong_namespace.status, StatusCode::FORBIDDEN.as_u16());

    let missing_id = service
        .execute(request(json!({}), principal.clone()))
        .await;
    let missing_id_contract: Value = serde_json::from_str(include_str!(
        "../../../../../contracts/golden/syscall-receive-message-missing-id-v1.json"
    ))
    .unwrap();
    assert_eq!(missing_id.status, missing_id_contract["http_status"]);
    assert_eq!(
        serde_json::from_slice::<Value>(&missing_id.body).unwrap(),
        missing_id_contract["body"]
    );

    let response = service
        .execute(request(json!({"conversation_id":"c1"}), principal))
        .await;
    assert_eq!(response.status, StatusCode::OK.as_u16());
    let response: Value = serde_json::from_slice(&response.body).unwrap();
    let contract: Value = serde_json::from_str(include_str!(
        "../../../../../contracts/golden/receive-message-result-v1.json"
    ))
    .unwrap();
    assert_eq!(response["op"], "receive_message");
    assert_eq!(response["result"], contract);
}
