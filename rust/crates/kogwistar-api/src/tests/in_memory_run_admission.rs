#[tokio::test]
async fn in_memory_native_run_admission_is_authenticated_and_atomic() {
    let service = InMemoryGraphApplicationService::default();
    assert!(service.supports_http_operation("POST", "/api/workflow/runs"));
    assert!(service.supports_http_operation("GET", "/api/workflow/runs/memory-http-run/events"));
    assert!(service.supports_http_operation("POST", "/api/workflow/runs/memory-http-run/resume"));
    assert!(service.supports_mcp_tool_for_principal(
        "workflow",
        "workflow.run_resume",
        &json!({"role":"rw", "ns":["workflow"], "capabilities":["workflow.run.write"]})
    ));
    assert!(service.supports_mcp_tool_for_principal(
        "workflow",
        "workflow.run_resume",
        &json!({"role":"rw", "ns":["workflow"], "capabilities":["approve_action"]})
    ));
    for denied in [
        json!({"role":"ro", "ns":["workflow"], "capabilities":["workflow.run.write"]}),
        json!({"role":"ro", "ns":["workflow"], "capabilities":["workflow.run.write"]}),
        json!({"role":"rw", "ns":["conversation"], "capabilities":["workflow.run.write"]}),
        json!({"role":"rw", "ns":["workflow"], "capabilities":["workflow.run.read"]}),
    ] {
        assert!(!service.supports_mcp_tool_for_principal(
            "workflow",
            "workflow.run_resume",
            &denied
        ));
    }
    let request = |principal: Value, body: Value| ApiEffectRequest {
        contract_version: 1,
        method: "POST".to_owned(),
        path_and_query: "/api/workflow/runs".to_owned(),
        body: serde_json::to_vec(&body).unwrap(),
        principal,
    };
    let body = json!({
        "run_id": "memory-http-run",
        "workflow_id": "memory-http-workflow",
        "conversation_id": "memory-http-conversation",
        "expected_run_owner_id": "owner",
        "initial_state": {
            "seed": 7,
            "_rt_run_owner_id": "forged-owner",
            "_rt_graph_scope": {"storage_namespace": "forged"}
        },
        "node_ops": {"start": "noop"},
        "node_executor_targets": {"start": "rust-native"}
    });
    let admission_contract: Value = serde_json::from_str(include_str!(
        "../../../../../contracts/golden/workflow-run-submit-auth-v1.json"
    ))
    .unwrap();
    let unauthorized = service
        .execute(request(
            json!({"sub":"owner","role":"rw","ns":["workflow"],"capabilities":["workflow.run.read"]}),
            body.clone(),
        ))
        .await;
    assert_eq!(unauthorized.status, StatusCode::FORBIDDEN.as_u16());
    assert!(
        kogwistar_store::ServerRunReadStore::server_run(&service.store, "memory-http-run")
            .await
            .unwrap()
            .is_none()
    );

    let mut wrong_expected_owner = body.clone();
    wrong_expected_owner["run_id"] = json!("memory-wrong-expected-owner");
    wrong_expected_owner["expected_run_owner_id"] = json!("attacker");
    let owner_mismatch = service
        .execute(request(
            json!({
                "sub":"owner",
                "role":"rw",
                "ns":["workflow"],
                "capabilities":["spawn_process", "workflow.run.write", "workflow.run.read"]
            }),
            wrong_expected_owner,
        ))
        .await;
    assert_eq!(owner_mismatch.status, StatusCode::FORBIDDEN.as_u16());
    assert!(
        kogwistar_store::ServerRunReadStore::server_run(
            &service.store,
            "memory-wrong-expected-owner"
        )
        .await
        .unwrap()
        .is_none()
    );

    for missing_capability in admission_contract["required_capabilities"]
        .as_array()
        .unwrap()
    {
        let granted = admission_contract["required_capabilities"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|capability| *capability != missing_capability)
            .cloned()
            .collect::<Vec<_>>();
        let denied_run_id = format!("missing-{}", missing_capability.as_str().unwrap());
        let mut denied_body = body.clone();
        denied_body["run_id"] = json!(denied_run_id);
        let response = service
            .execute(request(
                json!({
                    "sub":"owner",
                    "role":"rw",
                    "ns":["workflow"],
                    "capabilities":granted
                }),
                denied_body,
            ))
            .await;
        assert_eq!(
            response.status,
            admission_contract["denied_status"].as_u64().unwrap() as u16
        );
        assert!(
            kogwistar_store::ServerRunReadStore::server_run(&service.store, &denied_run_id)
                .await
                .unwrap()
                .is_none()
        );
    }

    let principal = json!({
        "sub":"owner",
        "role":"rw",
        "ns":["workflow"],
        "capabilities":["spawn_process", "workflow.run.write", "workflow.run.read"]
    });
    let admitted = service
        .execute(request(principal.clone(), body.clone()))
        .await;
    assert_eq!(admitted.status, StatusCode::ACCEPTED.as_u16());
    let replay = service.execute(request(principal.clone(), body)).await;
    assert_eq!(replay.status, StatusCode::ACCEPTED.as_u16());
    let queued_lane = kogwistar_store::LaneMessageReadStore::projected_lane_message(
        &service.store,
        "lane|runtime-start|memory-http-run",
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(queued_lane.status, "pending");
    let run = kogwistar_store::ServerRunReadStore::server_run(&service.store, "memory-http-run")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(run.user_id.as_deref(), Some("owner"));
    assert_eq!(run.status, "running");
    assert!(service.execute_inline_native_work_once().await.unwrap());
    let completed =
        kogwistar_store::ServerRunReadStore::server_run(&service.store, "memory-http-run")
            .await
            .unwrap()
            .unwrap();
    assert_eq!(completed.status, "succeeded");

    let resume_request = |principal: Value| ApiEffectRequest {
        contract_version: 1,
        method: "POST".to_owned(),
        path_and_query: "/api/workflow/runs/memory-http-run/resume".to_owned(),
        body: serde_json::to_vec(&json!({
            "suspended_node_id": "start",
            "suspended_token_id": "stale-token",
            "client_result": {"approved": true},
            "workflow_id": "memory-http-workflow",
            "conversation_id": "memory-http-conversation"
        }))
        .unwrap(),
        principal,
    };
    let unauthorized_resume = service
        .execute(resume_request(json!({
            "sub":"owner",
            "role":"rw",
            "ns":["workflow"],
            "capabilities":["workflow.run.read"]
        })))
        .await;
    assert_eq!(
        unauthorized_resume.status,
        StatusCode::FORBIDDEN.as_u16()
    );
    let read_only_resume = service
        .execute(resume_request(json!({
            "sub":"owner",
            "role":"ro",
            "ns":["workflow"],
            "capabilities":["workflow.run.write"]
        })))
        .await;
    assert_eq!(read_only_resume.status, StatusCode::FORBIDDEN.as_u16());
    let stale_resume = service
        .execute(resume_request(json!({
            "sub":"owner",
            "role":"rw",
            "ns":["workflow"],
            "capabilities":["workflow.run.write"]
        })))
        .await;
    assert_eq!(stale_resume.status, StatusCode::CONFLICT.as_u16());
    let final_state = kogwistar_store::ProjectionReadStore::named_projection(
        &service.store,
        kogwistar_store::RUNTIME_CURRENT_STATE_NAMESPACE,
        "memory-http-run",
    )
    .await
    .unwrap()
    .unwrap()
    .payload;
    assert_eq!(final_state["state"]["_rt_run_owner_id"], "owner");
    assert_eq!(
        final_state["state"]["_rt_graph_scope"]["storage_namespace"],
        "workflow"
    );

    let mut conflicting_body = json!({
        "run_id": "memory-http-run",
        "workflow_id": "memory-http-workflow",
        "conversation_id": "memory-http-conversation",
        "initial_state": {"seed": 8},
        "node_ops": {"start": "noop"},
        "node_executor_targets": {"start": "rust-native"}
    });
    conflicting_body["initial_state"]["_rt_run_owner_id"] = json!("forged-owner");
    let conflict = service
        .execute(request(principal.clone(), conflicting_body))
        .await;
    assert_eq!(conflict.status, StatusCode::CONFLICT.as_u16());

    let owner_read = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "GET".to_owned(),
            path_and_query: "/api/workflow/runs/memory-http-run/events?after_seq=0".to_owned(),
            body: Vec::new(),
            principal: principal.clone(),
        })
        .await;
    assert_eq!(owner_read.status, StatusCode::OK.as_u16());
    let event_result: Value = serde_json::from_slice(&owner_read.body).unwrap();
    let events = event_result["events"].as_array().unwrap();
    assert!(events.len() >= 3);
    assert!(
        events
            .iter()
            .any(|event| event["event_type"] == "workflow.recorded_transition.v1")
    );

    let other_owner = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "GET".to_owned(),
            path_and_query: "/api/workflow/runs/memory-http-run".to_owned(),
            body: Vec::new(),
            principal: json!({
                "sub":"other",
                "role":"rw",
                "ns":["workflow"],
                "capabilities":["workflow.run.read"]
            }),
        })
        .await;
    assert_eq!(other_owner.status, StatusCode::FORBIDDEN.as_u16());

    let unsupported = service
        .execute(request(
            principal,
            json!({
                "run_id":"memory-http-unsupported",
                "workflow_id":"memory-http-workflow",
                "conversation_id":"memory-http-conversation",
                "node_ops":{"start":"python-only-operation"},
                "node_executor_targets":{"start":"python-worker"}
            }),
        ))
        .await;
    assert_eq!(
        unsupported.status,
        StatusCode::UNPROCESSABLE_ENTITY.as_u16()
    );
    assert!(
        kogwistar_store::ServerRunReadStore::server_run(&service.store, "memory-http-unsupported")
            .await
            .unwrap()
            .is_none()
    );
}

#[tokio::test]
async fn in_memory_run_cancel_matches_shared_contract_and_authority() {
    let service = InMemoryGraphApplicationService::default();
    let route = "/api/workflow/runs/memory-cancel-run/cancel";
    assert!(service.supports_http_operation("POST", route));
    assert!(service.supports_mcp_tool_for_principal(
        "workflow",
        "workflow.run_cancel",
        &json!({"role":"rw", "ns":["workflow"], "capabilities":[]})
    ));
    assert!(!service.supports_mcp_tool_for_principal(
        "workflow",
        "workflow.run_cancel",
        &json!({"role":"ro", "ns":["workflow"], "capabilities":["workflow.run.write"]})
    ));
    assert!(!service.supports_mcp_tool_for_principal(
        "workflow",
        "workflow.run_cancel",
        &json!({"role":"rw", "ns":["conversation"], "capabilities":["workflow.run.write"]})
    ));

    let submit = |principal: Value| ApiEffectRequest {
        contract_version: 1,
        method: "POST".to_owned(),
        path_and_query: "/api/workflow/runs".to_owned(),
        body: serde_json::to_vec(&json!({
            "run_id":"memory-cancel-run",
            "workflow_id":"memory-cancel-workflow",
            "conversation_id":"memory-cancel-conversation",
            "expected_run_owner_id":"alice",
            "initial_state":{"seed":1},
            "node_ops":{"start":"noop"},
            "node_executor_targets":{"start":"rust-native"}
        }))
        .unwrap(),
        principal,
    };
    let principal = json!({
        "sub":"alice",
        "role":"rw",
        "ns":["workflow"],
        "capabilities":["spawn_process", "workflow.run.write", "workflow.run.read"]
    });
    assert_eq!(
        service.execute(submit(principal.clone())).await.status,
        StatusCode::ACCEPTED.as_u16()
    );

    let cancel = |principal: Value| ApiEffectRequest {
        contract_version: 1,
        method: "POST".to_owned(),
        path_and_query: route.to_owned(),
        body: Vec::new(),
        principal,
    };
    let denied = service
        .execute(cancel(json!({
            "sub":"alice", "role":"rw", "ns":["workflow"],
            "capabilities":["workflow.run.read"]
        })))
        .await;
    assert_eq!(denied.status, StatusCode::FORBIDDEN.as_u16());
    assert_eq!(
        kogwistar_store::ServerRunReadStore::server_run(&service.store, "memory-cancel-run")
            .await
            .unwrap()
            .unwrap()
            .status,
        "running",
        "denied cancellation must not mutate the run"
    );

    let contract: Value = serde_json::from_str(include_str!(
        "../../../../../contracts/golden/run-cancel-v1.json"
    ))
    .unwrap();
    let accepted = service.execute(cancel(principal)).await;
    assert_eq!(
        accepted.status,
        contract["accepted_status"].as_u64().unwrap() as u16
    );
    let body: Value = serde_json::from_slice(&accepted.body).unwrap();
    for field in contract["required_fields"].as_array().unwrap() {
        let field = field.as_str().unwrap();
        assert!(body.get(field).is_some(), "missing {field}");
    }
    assert!(contract["accepted_run_statuses"].as_array().unwrap().contains(&body["status"]));
    assert_eq!(body["cancel_requested"], true);
    assert_eq!(body["status"], "cancelled");

    let events_after_cancel = kogwistar_store::ServerRunReadStore::server_run_events(
        &service.store,
        "memory-cancel-run",
        0,
        100,
    )
    .await
    .unwrap();
    let terminal_repeat = service
        .execute(cancel(json!({"sub":"alice", "role":"rw", "ns":["workflow"], "capabilities":[]})))
        .await;
    assert_eq!(terminal_repeat.status, StatusCode::ACCEPTED.as_u16());
    assert_eq!(
        kogwistar_store::ServerRunReadStore::server_run_events(
            &service.store,
            "memory-cancel-run",
            0,
            100,
        )
        .await
        .unwrap()
        .len(),
        events_after_cancel.len(),
        "terminal repeat must not append another transition"
    );
}

#[tokio::test]
async fn in_memory_workflow_syscalls_use_native_run_handlers() {
    let service = InMemoryGraphApplicationService::default();
    let principal = json!({
        "sub":"syscall-owner",
        "role":"rw",
        "ns":["workflow"],
        "capabilities":["spawn_process", "workflow.run.write"]
    });
    let spawn_path = "/api/syscall/v1/spawn_process";
    assert!(service.supports_http_operation("POST", spawn_path));
    let spawn = |principal: Value| ApiEffectRequest {
        contract_version: 1,
        method: "POST".to_owned(),
        path_and_query: spawn_path.to_owned(),
        body: serde_json::to_vec(&json!({
            "version":"syscall-v1",
            "args":{
                "run_id":"memory-syscall-run",
                "workflow_id":"memory-syscall-workflow",
                "conversation_id":"memory-syscall-conversation",
                "expected_run_owner_id":"syscall-owner",
                "node_ops":{"start":"noop"},
                "node_executor_targets":{"start":"rust-native"}
            }
        })).unwrap(),
        principal,
    };
    let denied = service.execute(spawn(json!({
        "sub":"syscall-owner", "role":"rw", "ns":["workflow"],
        "capabilities":["workflow.run.write"]
    }))).await;
    assert_eq!(denied.status, StatusCode::FORBIDDEN.as_u16());
    assert!(kogwistar_store::ServerRunReadStore::server_run(&service.store, "memory-syscall-run")
        .await.unwrap().is_none());

    let spawned = service.execute(spawn(principal.clone())).await;
    assert_eq!(spawned.status, StatusCode::OK.as_u16());
    let spawned: Value = serde_json::from_slice(&spawned.body).unwrap();
    assert_eq!(spawned["version"], "syscall-v1");
    assert_eq!(spawned["op"], "spawn_process");
    assert_eq!(spawned["status"], "ok");
    assert_eq!(spawned["result"]["run_id"], "memory-syscall-run");

    for op in [
        "terminate_process",
        "resume",
        "checkpoint",
        "project_view",
        "request_approval",
    ] {
        assert!(service.supports_http_operation("POST", &format!("/api/syscall/v1/{op}")));
    }
    let denied_checkpoint = service.execute(ApiEffectRequest {
        contract_version: 1,
        method: "POST".to_owned(),
        path_and_query: "/api/syscall/v1/checkpoint".to_owned(),
        body: serde_json::to_vec(&json!({
            "version":"syscall-v1",
            "args":{"run_id":"memory-syscall-run", "step_seq":"0"}
        })).unwrap(),
        principal: json!({
            "sub":"syscall-owner", "role":"ro", "ns":["workflow"],
            "capabilities":[]
        }),
    }).await;
    assert_eq!(denied_checkpoint.status, StatusCode::FORBIDDEN.as_u16());
    let denied_terminate = service.execute(ApiEffectRequest {
        contract_version: 1,
        method: "POST".to_owned(),
        path_and_query: "/api/syscall/v1/terminate_process".to_owned(),
        body: serde_json::to_vec(&json!({
            "version":"syscall-v1",
            "args":{"run_id":"memory-syscall-run"}
        })).unwrap(),
        principal: json!({
            "sub":"syscall-owner", "role":"rw", "ns":["workflow"],
            "capabilities":["workflow.run.read"]
        }),
    }).await;
    assert_eq!(denied_terminate.status, StatusCode::FORBIDDEN.as_u16());
    assert_eq!(kogwistar_store::ServerRunReadStore::server_run(
        &service.store, "memory-syscall-run",
    ).await.unwrap().unwrap().status, "running");
    let terminated = service.execute(ApiEffectRequest {
        contract_version: 1,
        method: "POST".to_owned(),
        path_and_query: "/api/syscall/v1/terminate_process".to_owned(),
        body: serde_json::to_vec(&json!({
            "version":"syscall-v1",
            "args":{"run_id":"memory-syscall-run"}
        })).unwrap(),
        principal,
    }).await;
    assert_eq!(terminated.status, StatusCode::OK.as_u16());
    let terminated: Value = serde_json::from_slice(&terminated.body).unwrap();
    assert_eq!(terminated["op"], "terminate_process");
    assert_eq!(terminated["result"]["status"], "cancelled");

    let checkpoint = service.execute(ApiEffectRequest {
        contract_version: 1,
        method: "POST".to_owned(),
        path_and_query: "/api/syscall/v1/checkpoint".to_owned(),
        body: serde_json::to_vec(&json!({
            "version":"syscall-v1",
            "args":{"run_id":"memory-syscall-run", "step_seq":0}
        })).unwrap(),
        principal: json!({
            "sub":"syscall-owner", "role":"ro", "ns":["workflow"],
            "capabilities":["workflow.run.read"]
        }),
    }).await;
    assert_eq!(checkpoint.status, StatusCode::OK.as_u16());
    let checkpoint: Value = serde_json::from_slice(&checkpoint.body).unwrap();
    assert_eq!(checkpoint["op"], "checkpoint");
    assert_eq!(checkpoint["status"], "ok");
    assert_eq!(checkpoint["result"]["run_id"], "memory-syscall-run");
    assert_eq!(checkpoint["result"]["step_seq"], 0);

    let conversation_scoped_checkpoint = service.execute(ApiEffectRequest {
        contract_version: 1,
        method: "POST".to_owned(),
        path_and_query: "/api/syscall/v1/checkpoint".to_owned(),
        body: serde_json::to_vec(&json!({
            "version":"syscall-v1",
            "args":{"run_id":"memory-syscall-run", "step_seq":0}
        })).unwrap(),
        principal: json!({
            "sub":"syscall-owner", "role":"ro", "ns":["conversation"],
            "capabilities":["workflow.run.read"]
        }),
    }).await;
    assert_eq!(conversation_scoped_checkpoint.status, StatusCode::OK.as_u16());

    let project_view = service.execute(ApiEffectRequest {
        contract_version: 1,
        method: "POST".to_owned(),
        path_and_query: "/api/syscall/v1/project_view".to_owned(),
        body: serde_json::to_vec(&json!({"version":"syscall-v1", "args":{}})).unwrap(),
        principal: json!({
            "sub":"syscall-reader", "role":"ro", "ns":["workflow"],
            "capabilities":["read_security_scope", "project_view"]
        }),
    }).await;
    assert_eq!(project_view.status, StatusCode::OK.as_u16());
    let project_view: Value = serde_json::from_slice(&project_view.body).unwrap();
    assert_eq!(project_view["op"], "project_view");
    assert_eq!(project_view["status"], "ok");
    assert!(project_view["result"]["runs"]["total_runs"].as_u64().unwrap() >= 1);
    let denied_project_view = service.execute(ApiEffectRequest {
        contract_version: 1,
        method: "POST".to_owned(),
        path_and_query: "/api/syscall/v1/project_view".to_owned(),
        body: serde_json::to_vec(&json!({"version":"syscall-v1", "args":{}})).unwrap(),
        principal: json!({
            "sub":"syscall-reader", "role":"ro", "ns":["wisdom"],
            "capabilities":["read_security_scope", "project_view"]
        }),
    }).await;
    assert_eq!(denied_project_view.status, StatusCode::FORBIDDEN.as_u16());

    for (action, capabilities) in [
        ("grant", vec!["approve_action", "project_view"]),
        ("grant", vec!["approve_action", "read_security_scope"]),
        ("revoke", vec!["approve_action", "project_view"]),
        ("revoke", vec!["approve_action", "read_security_scope"]),
    ] {
        let denied_mutation = service.execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/syscall/v1/request_approval".to_owned(),
            body: serde_json::to_vec(&json!({
                "version":"syscall-v1",
                "args":{
                    "action":action,
                    "subject":"unauthorized-target",
                    "capability":"workflow.run.write"
                }
            })).unwrap(),
            principal: json!({
                "sub":"syscall-owner", "role":"rw", "ns":["workflow"],
                "capabilities":capabilities
            }),
        }).await;
        assert_eq!(denied_mutation.status, StatusCode::FORBIDDEN.as_u16());
    }

    let approved = service.execute(ApiEffectRequest {
        contract_version: 1,
        method: "POST".to_owned(),
        path_and_query: "/api/syscall/v1/request_approval".to_owned(),
        body: serde_json::to_vec(&json!({
            "version":"syscall-v1",
            "args":{
                "action":"grant", "approval_action":"request_approval",
                "subject":"syscall-target", "capability":"workflow.run.read"
            }
        })).unwrap(),
        principal: json!({
            "sub":"syscall-owner", "role":"rw", "ns":["workflow"],
            "capabilities":["approve_action", "project_view", "read_security_scope"]
        }),
    }).await;
    assert_eq!(approved.status, StatusCode::OK.as_u16());
    let approved: Value = serde_json::from_slice(&approved.body).unwrap();
    assert_eq!(approved["result"]["status"], "approved");
    assert!(!approved["result"]["approval"]["approvals"]
        .as_array()
        .unwrap()
        .iter()
        .any(|approval| approval["subject"] == "unauthorized-target"));
    assert!(approved["result"]["approval"]["approvals"].as_array().unwrap().iter().any(
        |approval| approval["subject"] == "syscall-target"
            && approval["action"] == "request_approval"
            && approval["capabilities"] == json!(["workflow.run.read"])
    ));

    let revoked = service.execute(ApiEffectRequest {
        contract_version: 1,
        method: "POST".to_owned(),
        path_and_query: "/api/syscall/v1/request_approval".to_owned(),
        body: serde_json::to_vec(&json!({
            "version":"syscall-v1",
            "args":{"action":"revoke", "subject":"syscall-target", "capability":"workflow.run.read"}
        })).unwrap(),
        principal: json!({
            "sub":"syscall-owner", "role":"rw", "ns":["workflow"],
            "capabilities":["approve_action", "project_view", "read_security_scope"]
        }),
    }).await;
    assert_eq!(revoked.status, StatusCode::OK.as_u16());
    let revoked: Value = serde_json::from_slice(&revoked.body).unwrap();
    assert_eq!(revoked["result"]["status"], "revoked");
    assert!(!revoked["result"]["revocation"]["revoked"]
        .as_array()
        .unwrap()
        .iter()
        .any(|entry| entry["subject"] == "unauthorized-target"));
    assert!(revoked["result"]["revocation"]["revoked"].as_array().unwrap().iter().any(
        |entry| entry["subject"] == "syscall-target"
            && entry["capabilities"] == json!(["workflow.run.read"])
    ));

    let approval = service.execute(ApiEffectRequest {
        contract_version: 1,
        method: "POST".to_owned(),
        path_and_query: "/api/syscall/v1/request_approval".to_owned(),
        body: serde_json::to_vec(&json!({
            "version":"syscall-v1", "args":{"action":"deny", "reason":"test"}
        })).unwrap(),
        principal: json!({
            "sub":"syscall-owner", "role":"rw", "ns":["workflow"],
            "capabilities":[]
        }),
    }).await;
    assert_eq!(approval.status, StatusCode::OK.as_u16());
    let approval: Value = serde_json::from_slice(&approval.body).unwrap();
    assert_eq!(approval["op"], "request_approval");
    assert_eq!(approval["status"], "blocked");
    assert_eq!(approval["error"]["reason"], "test");
    let requested_approval = service.execute(ApiEffectRequest {
        contract_version: 1,
        method: "POST".to_owned(),
        path_and_query: "/api/syscall/v1/request_approval".to_owned(),
        body: serde_json::to_vec(&json!({
            "version":"syscall-v1", "args":{"action":"unrecognized"}
        })).unwrap(),
        principal: json!({
            "sub":"syscall-owner", "role":"rw", "ns":["workflow"],
            "capabilities":[]
        }),
    }).await;
    assert_eq!(requested_approval.status, StatusCode::OK.as_u16());
    let requested_approval: Value = serde_json::from_slice(&requested_approval.body).unwrap();
    assert_eq!(requested_approval["status"], "ok");
    assert_eq!(requested_approval["result"]["status"], "requested");
    assert_eq!(requested_approval["result"]["reason"], "approval pending");
    let missing_action_approval = service.execute(ApiEffectRequest {
        contract_version: 1,
        method: "POST".to_owned(),
        path_and_query: "/api/syscall/v1/request_approval".to_owned(),
        body: serde_json::to_vec(&json!({"version":"syscall-v1", "args":{}})).unwrap(),
        principal: json!({
            "sub":"syscall-owner", "role":"rw", "ns":["workflow"],
            "capabilities":[]
        }),
    }).await;
    assert_eq!(missing_action_approval.status, StatusCode::BAD_REQUEST.as_u16());
    let denied_approval = service.execute(ApiEffectRequest {
        contract_version: 1,
        method: "POST".to_owned(),
        path_and_query: "/api/syscall/v1/request_approval".to_owned(),
        body: serde_json::to_vec(&json!({
            "version":"syscall-v1", "args":{"action":"deny"}
        })).unwrap(),
        principal: json!({
            "sub":"syscall-owner", "role":"ro", "ns":["workflow"],
            "capabilities":["approve_action"]
        }),
    }).await;
    assert_eq!(denied_approval.status, StatusCode::FORBIDDEN.as_u16());
}

#[tokio::test]
async fn in_memory_project_view_syscall_is_reachable_through_authenticated_http() {
    use axum::body::{Body, to_bytes};
    use axum::http::Request;
    use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
    use std::sync::Arc;
    use tower::ServiceExt;

    let secret = "memory-project-view-syscall-secret";
    let service = InMemoryGraphApplicationService::default();
    let app = router_with_application(
        ApiState {
            health: snapshot(),
            required_roles: Vec::new(),
            implementation: ImplementationSnapshot::default(),
            auth: AuthConfig {
                mode: "jwt".to_owned(),
                algorithm: Some("HS256".to_owned()),
                key: Some(secret.to_owned()),
                issuer: Some("local".to_owned()),
                ..AuthConfig::default()
            },
        },
        Arc::new(service),
    );
    let token = |capabilities: Vec<&str>| {
        encode(
            &Header::new(Algorithm::HS256),
            &json!({
                "sub":"project-view-reader", "role":"ro", "ns":["workflow"],
                "capabilities":capabilities, "iss":"local", "exp":4102444800_u64
            }),
            &EncodingKey::from_secret(secret.as_bytes()),
        ).unwrap()
    };
    let call = |jwt: String| {
        Request::builder()
            .method("POST")
            .uri("/api/syscall/v1/project_view")
            .header("content-type", "application/json")
            .header("authorization", format!("Bearer {jwt}"))
            .body(Body::from(json!({"version":"syscall-v1", "args":{}}).to_string()))
            .unwrap()
    };
    let denied = app.clone().oneshot(call(token(vec!["project_view"]))).await.unwrap();
    assert_eq!(denied.status(), StatusCode::FORBIDDEN);
    let accepted = app.oneshot(call(token(vec!["read_security_scope", "project_view"]))).await.unwrap();
    assert_eq!(accepted.status(), StatusCode::OK);
    let body: Value = serde_json::from_slice(
        &to_bytes(accepted.into_body(), 65_536).await.unwrap(),
    ).unwrap();
    assert_eq!(body["version"], "syscall-v1");
    assert_eq!(body["op"], "project_view");
    assert_eq!(body["status"], "ok");
    assert_eq!(body["result"]["runs"]["total_runs"], 0);
}

// PYTHON_PARITY_NOTE_V1: MCP transport authorization and dispatch reach the in-memory workflow cancel effect; denied capability preserves run state. rust=rust/crates/kogwistar-api/src/tests/in_memory_run_admission.rs#in_memory_workflow_cancel_mcp_reaches_authorized_store_effect; python=tests/mcp/test_chat_mcp_tools.py#test_mcp_workflow_run_cancel_matches_shared_contract; fixture=contracts/golden/workflow-run-cancel-mcp-v1.json; rust_contract=contracts/golden/run-cancel-v1.json
#[tokio::test]
async fn in_memory_workflow_cancel_mcp_reaches_authorized_store_effect() {
    use axum::body::{Body, to_bytes};
    use axum::http::Request;
    use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
    use std::sync::Arc;
    use tower::ServiceExt;

    let secret = "in-memory-workflow-cancel-mcp-secret";
    let service = InMemoryGraphApplicationService::default();
    let admitted = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/workflow/runs".to_owned(),
            body: serde_json::to_vec(&json!({
                "run_id":"memory-mcp-cancel-run",
                "workflow_id":"memory-mcp-cancel-workflow",
                "conversation_id":"memory-mcp-cancel-conversation",
                "expected_run_owner_id":"alice",
                "initial_state":{"seed":1},
                "node_ops":{"start":"noop"},
                "node_executor_targets":{"start":"rust-native"}
            }))
            .unwrap(),
            principal: json!({
                "sub":"alice", "role":"rw", "ns":["workflow"],
                "capabilities":["spawn_process", "workflow.run.write"]
            }),
        })
        .await;
    assert_eq!(admitted.status, StatusCode::ACCEPTED.as_u16());

    let token = |capabilities: Vec<&str>| {
        encode(
            &Header::new(Algorithm::HS256),
            &json!({
                "sub":"alice", "role":"rw", "ns":["workflow"],
                "capabilities":capabilities, "iss":"local", "exp":4102444800_u64
            }),
            &EncodingKey::from_secret(secret.as_bytes()),
        )
        .unwrap()
    };
    let app = router_with_application(
        ApiState {
            health: snapshot(),
            required_roles: Vec::new(),
            implementation: ImplementationSnapshot::default(),
            auth: AuthConfig {
                mode: "jwt".to_owned(),
                algorithm: Some("HS256".to_owned()),
                key: Some(secret.to_owned()),
                issuer: Some("local".to_owned()),
                ..AuthConfig::default()
            },
        },
        Arc::new(service.clone()),
    );
    let call = |jwt: String| {
        Request::builder()
            .method("POST")
            .uri("/mcp/workflow")
            .header("content-type", "application/json")
            .header("accept", "application/json, text/event-stream")
            .header("authorization", format!("Bearer {jwt}"))
            .body(Body::from(
                json!({
                    "jsonrpc":"2.0", "id":"memory-cancel", "method":"tools/call",
                    "params":{"name":"workflow.run_cancel", "arguments":{"run_id":"memory-mcp-cancel-run"}}
                })
                .to_string(),
            ))
            .unwrap()
    };

    let denied = app
        .clone()
        .oneshot(call(token(vec!["workflow.run.read"])))
        .await
        .unwrap();
    assert_eq!(denied.status(), StatusCode::OK);
    let denied_wire = to_bytes(denied.into_body(), 8192).await.unwrap();
    let denied_data = String::from_utf8(denied_wire.to_vec()).unwrap();
    let denied_message = denied_data
        .lines()
        .find_map(|line| line.strip_prefix("data: "))
        .unwrap();
    assert_eq!(
        serde_json::from_str::<Value>(denied_message).unwrap()["result"]["isError"],
        true
    );
    assert_eq!(
        kogwistar_store::ServerRunReadStore::server_run(&service.store, "memory-mcp-cancel-run")
            .await
            .unwrap()
            .unwrap()
            .status,
        "running"
    );

    let accepted = app
        .oneshot(call(token(vec!["workflow.run.write"])))
        .await
        .unwrap();
    assert_eq!(accepted.status(), StatusCode::OK);
    let accepted_wire = to_bytes(accepted.into_body(), 8192).await.unwrap();
    let accepted_data = String::from_utf8(accepted_wire.to_vec()).unwrap();
    let accepted_message = accepted_data
        .lines()
        .find_map(|line| line.strip_prefix("data: "))
        .unwrap();
    let accepted_value: Value = serde_json::from_str(accepted_message).unwrap();
    assert_ne!(accepted_value["result"]["isError"], true);
    let cancelled = kogwistar_store::ServerRunReadStore::server_run(
        &service.store,
        "memory-mcp-cancel-run",
    )
    .await
    .unwrap()
    .unwrap();
    assert!(cancelled.cancel_requested);
    assert_eq!(cancelled.status, "cancelled");
}

#[tokio::test]
async fn in_memory_native_synchronous_resume_matches_shared_result_contract() {
    use kogwistar_store::{
        LaneMessageClaimFilter, LaneMessageReadStore, LaneMessageWriteStore,
        ProjectionReadStore, RUNTIME_CURRENT_STATE_NAMESPACE,
    };

    let service = InMemoryGraphApplicationService::default();
    let principal = json!({
        "sub":"resume-owner",
        "role":"rw",
        "ns":["workflow"],
        "capabilities":["spawn_process", "workflow.run.write"]
    });
    let submitted = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/workflow/runs".to_owned(),
            body: serde_json::to_vec(&json!({
                "run_id":"memory-resume-run",
                "workflow_id":"memory-resume-workflow",
                "conversation_id":"memory-resume-conversation",
                "initial_state":{"started":true},
                "node_ops":{"start":"noop"},
                "node_executor_targets":{"start":"rust-native"}
            }))
            .unwrap(),
            principal: principal.clone(),
        })
        .await;
    assert_eq!(submitted.status, StatusCode::ACCEPTED.as_u16());

    let start_lane = LaneMessageReadStore::projected_lane_message(
        &service.store,
        "lane|runtime-start|memory-resume-run",
    )
    .await
    .unwrap()
    .unwrap();
    let claimed = LaneMessageWriteStore::claim_projected_lane_messages_filtered(
        &service.store,
        LaneMessageClaimFilter {
            namespace: "workflow".to_owned(),
            inbox_id: "workflow-runtime".to_owned(),
            claimed_by: "test-suspender".to_owned(),
            message_ids: Some(vec![start_lane.message_id]),
            run_id: Some("memory-resume-run".to_owned()),
            msg_type: Some("workflow.step.execute".to_owned()),
            recipient_id: Some("rust-native-worker".to_owned()),
            limit: 1,
            lease_seconds: 60,
        },
    )
    .await
    .unwrap()
    .pop()
    .unwrap();
    let handoff = kogwistar_runtime::RecordedWorkerHandoff {
        message_id: claimed.message_id,
        claimed_by: claimed.claimed_by.unwrap(),
        run_id: "memory-resume-run".to_owned(),
        step_id: claimed.step_id.unwrap(),
        correlation_id: claimed.correlation_id.unwrap(),
    };
    let effect: kogwistar_runtime::RecordedWorkerSuccessEffect = serde_json::from_value(json!({
        "contract_version":1,
        "effect_id":"suspend-memory-resume-run",
        "status":"suspended",
        "wait_reason":"approval",
        "resume_payload":{"question":"continue?"}
    }))
    .unwrap();
    service
        .store
        .apply_claimed_recorded_worker_effect(handoff, effect)
        .unwrap();

    let recorded = ProjectionReadStore::named_projection(
        &service.store,
        RUNTIME_CURRENT_STATE_NAMESPACE,
        "memory-resume-run",
    )
    .await
    .unwrap()
    .unwrap();
    let state: kogwistar_runtime::RecordedRuntimeState =
        serde_json::from_value(Value::Object(recorded.payload)).unwrap();
    let (node_id, _, token_id, _) = state.frontier.suspended[0].clone();

    let resumed = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/workflow/runs/memory-resume-run/resume?wait_for_completion=true"
                .to_owned(),
            body: serde_json::to_vec(&json!({
                "suspended_node_id":node_id,
                "suspended_token_id":token_id,
                "client_result":{"approved":true},
                "workflow_id":"memory-resume-workflow",
                "conversation_id":"memory-resume-conversation"
            }))
            .unwrap(),
            principal,
        })
        .await;
    assert_eq!(resumed.status, StatusCode::OK.as_u16());
    let result: Value = serde_json::from_slice(&resumed.body).unwrap();
    let contract: Value = serde_json::from_str(include_str!(
        "../../../../../contracts/golden/workflow-run-resume-result-v1.json"
    ))
    .unwrap();
    let expected_fields = contract["required_fields"]
        .as_array()
        .unwrap()
        .iter()
        .map(|field| field.as_str().unwrap())
        .collect::<std::collections::BTreeSet<_>>();
    let actual_fields = result
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(actual_fields, expected_fields);
    assert_eq!(result["workflow_status"], "succeeded");
    assert_eq!(result["final_state"]["started"], true);
}

#[tokio::test]
async fn in_memory_resume_syscall_preserves_resume_envelope_and_result() {
    use kogwistar_store::{
        LaneMessageClaimFilter, LaneMessageReadStore, LaneMessageWriteStore, ProjectionReadStore,
        RUNTIME_CURRENT_STATE_NAMESPACE,
    };

    let service = InMemoryGraphApplicationService::default();
    let principal = json!({
        "sub":"resume-syscall-owner", "role":"rw", "ns":["workflow"],
        "capabilities":["spawn_process", "workflow.run.write"]
    });
    let run_id = "memory-resume-syscall-run";
    let workflow_id = "memory-resume-syscall-workflow";
    let conversation_id = "memory-resume-syscall-conversation";
    let submitted = service.execute(ApiEffectRequest {
        contract_version: 1,
        method: "POST".to_owned(),
        path_and_query: "/api/workflow/runs".to_owned(),
        body: serde_json::to_vec(&json!({
            "run_id":run_id, "workflow_id":workflow_id,
            "conversation_id":conversation_id,
            "node_ops":{"start":"noop"},
            "node_executor_targets":{"start":"rust-native"}
        })).unwrap(),
        principal: principal.clone(),
    }).await;
    assert_eq!(submitted.status, StatusCode::ACCEPTED.as_u16());

    let lane = LaneMessageReadStore::projected_lane_message(
        &service.store, &format!("lane|runtime-start|{run_id}"),
    ).await.unwrap().unwrap();
    let claim = LaneMessageWriteStore::claim_projected_lane_messages_filtered(
        &service.store,
        LaneMessageClaimFilter {
            namespace: "workflow".to_owned(), inbox_id: "workflow-runtime".to_owned(),
            claimed_by: "test-suspender".to_owned(), message_ids: Some(vec![lane.message_id]),
            run_id: Some(run_id.to_owned()), msg_type: Some("workflow.step.execute".to_owned()),
            recipient_id: Some("rust-native-worker".to_owned()), limit: 1, lease_seconds: 60,
        },
    ).await.unwrap().pop().unwrap();
    service.store.apply_claimed_recorded_worker_effect(
        kogwistar_runtime::RecordedWorkerHandoff {
            message_id: claim.message_id, claimed_by: claim.claimed_by.unwrap(),
            run_id: run_id.to_owned(), step_id: claim.step_id.unwrap(),
            correlation_id: claim.correlation_id.unwrap(),
        },
        serde_json::from_value(json!({
            "contract_version":1, "effect_id":"suspend-syscall-resume",
            "status":"suspended", "wait_reason":"approval",
            "resume_payload":{"question":"continue?"}
        })).unwrap(),
    ).unwrap();
    let projection = ProjectionReadStore::named_projection(
        &service.store, RUNTIME_CURRENT_STATE_NAMESPACE, run_id,
    ).await.unwrap().unwrap();
    let state: kogwistar_runtime::RecordedRuntimeState =
        serde_json::from_value(Value::Object(projection.payload)).unwrap();
    let (node_id, _, token_id, _) = state.frontier.suspended[0].clone();

    let resumed = service.execute(ApiEffectRequest {
        contract_version: 1, method: "POST".to_owned(),
        path_and_query: "/api/syscall/v1/resume".to_owned(),
        body: serde_json::to_vec(&json!({
            "version":"syscall-v1",
            "args":{
                "run_id":run_id, "workflow_id":workflow_id,
                "conversation_id":conversation_id, "suspended_node_id":node_id,
                "suspended_token_id":token_id, "client_result":{"approved":true}
            }
        })).unwrap(),
        principal,
    }).await;
    assert_eq!(resumed.status, StatusCode::OK.as_u16());
    let envelope: Value = serde_json::from_slice(&resumed.body).unwrap();
    assert_eq!(envelope["version"], "syscall-v1");
    assert_eq!(envelope["op"], "resume");
    assert_eq!(envelope["status"], "ok");
    assert_eq!(envelope["result"]["workflow_status"], "succeeded");
    for field in ["workflow_status", "final_state", "budget"] {
        assert!(envelope["result"].get(field).is_some(), "missing {field}");
    }
}

#[tokio::test]
async fn in_memory_native_admission_uses_authoritative_design_and_rejects_conflicts() {
    use kogwistar_store::{NamedProjectionWrite, WorkflowDesignSnapshotWrite};

    let service = InMemoryGraphApplicationService::default();
    let workflow_id = "memory-authoritative-workflow";
    kogwistar_store::ProjectionWriteStore::replace_named_projection(
        &service.store,
        "workflow_design",
        workflow_id,
        NamedProjectionWrite {
            payload: serde_json::Map::from_iter([
                ("current_version".to_owned(), json!(1)),
                ("snapshot_schema_version".to_owned(), json!(1)),
            ]),
            last_authoritative_seq: 1,
            last_materialized_seq: 1,
            projection_schema_version: 1,
            materialization_status: "ready".to_owned(),
        },
    )
    .await
    .unwrap();
    kogwistar_store::WorkflowDesignHistoryWriteStore::put_workflow_design_snapshot(
        &service.store,
        workflow_id,
        WorkflowDesignSnapshotWrite {
            version: 1,
            seq: 1,
            schema_version: 1,
            payload_json: serde_json::to_string(&json!({
                "nodes": [{
                    "id": "start",
                    "metadata": {
                        "wf_start": true,
                        "wf_terminal": true,
                        "wf_op": "increment-counter",
                        "wf_version": "v1",
                        "wf_executor_target": "rust-native"
                    }
                }],
                "edges": []
            }))
            .unwrap(),
        },
    )
    .await
    .unwrap();

    let principal = json!({
        "sub":"design-owner",
        "role":"rw",
        "ns":["workflow"],
        "capabilities":["spawn_process", "workflow.run.write"]
    });
    let submit = |workflow_id: &str, run_id: &str, node_ops: Value| ApiEffectRequest {
        contract_version: 1,
        method: "POST".to_owned(),
        path_and_query: "/api/workflow/runs".to_owned(),
        body: serde_json::to_vec(&json!({
            "run_id": run_id,
            "workflow_id": workflow_id,
            "conversation_id": "memory-authoritative-conversation",
            "node_ops": node_ops
        }))
        .unwrap(),
        principal: principal.clone(),
    };

    let mut authoritative_request = submit(workflow_id, "memory-authoritative-run", json!({}));
    let mut authoritative_body: Value = serde_json::from_slice(&authoritative_request.body).unwrap();
    authoritative_body["initial_state"]["counter"] = json!(0);
    authoritative_request.body = serde_json::to_vec(&authoritative_body).unwrap();
    let accepted = service.execute(authoritative_request).await;
    assert_eq!(accepted.status, StatusCode::ACCEPTED.as_u16());
    let lane = kogwistar_store::LaneMessageReadStore::projected_lane_message(
        &service.store,
        "lane|runtime-start|memory-authoritative-run",
    )
    .await
    .unwrap()
    .unwrap();
    let payload: Value = serde_json::from_str(lane.payload_json.as_deref().unwrap()).unwrap();
    assert_eq!(payload["op"], "increment-counter");
    assert_eq!(payload["executor_target"], "rust-native");
    assert!(service.execute_inline_native_work_once().await.unwrap());

    let conflict = service
        .execute(submit(
            workflow_id,
            "memory-authoritative-conflict",
            json!({"start": "noop"}),
        ))
        .await;
    assert_eq!(conflict.status, StatusCode::CONFLICT.as_u16());
    assert!(
        kogwistar_store::ServerRunReadStore::server_run(
            &service.store,
            "memory-authoritative-conflict"
        )
        .await
        .unwrap()
        .is_none()
    );

    let multistep_workflow = "memory-multistep-workflow";
    let mut multistep_request = submit(
        multistep_workflow,
        "memory-multistep-run",
        json!({"start": "noop", "finish": "noop"}),
    );
    let mut multistep_body: Value = serde_json::from_slice(&multistep_request.body).unwrap();
    multistep_body["start_node_id"] = json!("start");
    multistep_body["runtime_routes"] = json!([{
        "edge_id": "route",
        "source_node_id": "start",
        "target_node_id": "finish",
        "aliases": [],
        "join_mask": 0,
        "predicate": null,
        "multiplicity": "all",
        "is_default": false,
        "priority": 0,
        "source_fanout": false
    }]);
    multistep_body["node_executor_targets"] = json!({
        "start": "rust-native",
        "finish": "rust-native"
    });
    multistep_request.body = serde_json::to_vec(&multistep_body).unwrap();
    let accepted_multistep = service.execute(multistep_request).await;
    assert_eq!(
        accepted_multistep.status,
        StatusCode::ACCEPTED.as_u16()
    );
    assert!(service.execute_inline_native_work_once().await.unwrap());
    let successor_lanes = kogwistar_store::LaneMessageReadStore::projected_lane_messages(
        &service.store,
        kogwistar_store::LaneMessageFilter {
            namespace: Some("workflow".to_owned()),
            inbox_id: Some("workflow-runtime".to_owned()),
            correlation_id: Some("memory-multistep-run".to_owned()),
            ..kogwistar_store::LaneMessageFilter::default()
        },
    )
    .await
    .unwrap();
    assert!(successor_lanes.iter().any(|lane| {
        lane.step_id.as_deref() == Some("finish") && lane.status == "pending"
    }));
    assert!(service.execute_inline_native_work_once().await.unwrap());
    assert!(!service.execute_inline_native_work_once().await.unwrap());
    let completed = kogwistar_store::ServerRunReadStore::server_run(
        &service.store,
        "memory-multistep-run",
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(completed.status, "succeeded");
    let events = kogwistar_store::ServerRunReadStore::server_run_events(
        &service.store,
        "memory-multistep-run",
        0,
        20,
    )
    .await
    .unwrap();
    assert_eq!(
        events
            .iter()
            .filter(|event| event.event_type == "workflow.recorded_transition.v1")
            .count(),
        4,
    );
}

#[tokio::test]
async fn in_memory_run_owner_revalidation_rejects_revoked_inactive_and_changed_scope() {
    let auth_path = std::env::temp_dir().join(format!(
        "kogwistar-memory-owner-auth-{}.sqlite",
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
    let auth_config = AuthConfig {
        auth_db_url: Some(format!("sqlite:///{}", auth_path.display())),
        ..AuthConfig::default()
    };
    let provider_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let provider_address = provider_listener.local_addr().unwrap();
    let service = InMemoryGraphApplicationService::default()
        .with_owner_auth_config(&auth_config)
        .await
        .unwrap()
        .with_chat_completion_provider(
            crate::native_answer_generation::OpenAiCompatibleChatConfig::new(
                format!("http://{provider_address}/v1/chat/completions"),
                "memory-event-model",
                None,
            )
            .unwrap()
            .with_streaming(true),
        );
    let principal = json!({
        "user_id":"alice",
        "sub":"alice",
        "role":"rw",
        "ns":["workflow", "alpha"],
        "storage_ns":"alpha",
        "capabilities":["spawn_process", "workflow.run.write", "write_graph"]
    });
    let admitted = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/api/workflow/runs".to_owned(),
            body: serde_json::to_vec(&json!({
                "run_id":"memory-owner-revalidation",
                "workflow_id":"memory-owner-workflow",
                "conversation_id":"memory-owner-conversation",
                "node_ops":{"start":"rust-native/agent.event.request"},
                "node_executor_targets":{"start":"rust-native"},
                "initial_state":{"provider_request":{
                    "contract_version":"v1", "model":"memory-event-model",
                    "instructions":"fixture only", "input":"not sent after revocation"
                }}
            }))
            .unwrap(),
            principal,
        })
        .await;
    assert_eq!(admitted.status, StatusCode::ACCEPTED.as_u16());
    assert!(
        service
            .resolve_inline_native_owner_principal("memory-owner-revalidation", "write_graph")
            .await
            .is_ok()
    );

    let auth_connection = rusqlite::Connection::open(&auth_path).unwrap();
    auth_connection
        .execute(
            "UPDATE users SET global_role = 'ro' WHERE user_id = 'alice'",
            [],
        )
        .unwrap();
    assert!(service.execute_inline_native_work_once().await.is_err());
    assert!(
        tokio::time::timeout(
            std::time::Duration::from_millis(100),
            provider_listener.accept()
        )
        .await
        .is_err()
    );
    let provider_messages = kogwistar_store::LaneMessageReadStore::projected_lane_messages(
        &service.store,
        LaneMessageFilter {
            inbox_id: Some("agent-provider-events".to_owned()),
            conversation_id: Some("memory-owner-conversation".to_owned()),
            limit: 10,
            ..LaneMessageFilter::default()
        },
    )
    .await
    .unwrap();
    assert!(provider_messages.is_empty());
    let run_events = kogwistar_store::ServerRunReadStore::server_run_events(
        &service.store,
        "memory-owner-revalidation",
        0,
        100,
    )
    .await
    .unwrap();
    assert!(
        !run_events
            .iter()
            .any(|event| event.event_type == "workflow.provider_event.v1")
    );
    assert!(matches!(
        service
            .resolve_inline_native_owner_principal("memory-owner-revalidation", "write_graph")
            .await,
        Err(InlineOwnerPrincipalError::Denied(_))
    ));

    auth_connection
        .execute(
            "UPDATE users SET global_role = 'rw', global_ns = 'beta' WHERE user_id = 'alice'",
            [],
        )
        .unwrap();
    assert!(matches!(
        service
            .resolve_inline_native_owner_principal("memory-owner-revalidation", "write_graph")
            .await,
        Err(InlineOwnerPrincipalError::Denied(_))
    ));

    auth_connection
        .execute(
            "UPDATE users SET global_ns = 'alpha', is_active = 0 WHERE user_id = 'alice'",
            [],
        )
        .unwrap();
    assert!(matches!(
        service
            .resolve_inline_native_owner_principal("memory-owner-revalidation", "write_graph")
            .await,
        Err(InlineOwnerPrincipalError::Denied(_))
    ));
    drop(auth_connection);
    drop(service);
    let _ = std::fs::remove_file(auth_path);
}
