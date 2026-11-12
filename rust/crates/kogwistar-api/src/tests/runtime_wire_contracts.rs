    #[test]
    fn runtime_start_tracks_only_reachable_join_obligations() {
        assert_eq!(runtime_join_outstanding(4, 0b0101), vec![1, 0, 1, 0]);
    }

    #[test]
    fn python_runtime_wire_vectors_round_trip_through_rust_dtos() {
        let vectors: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/adr015-runtime-wire-v1.json"
        )))
        .unwrap();
        let submit: RuntimeSubmitRun = serde_json::from_value(vectors["submit"].clone()).unwrap();
        let claim: RuntimeClaimRequest = serde_json::from_value(vectors["claim"].clone()).unwrap();
        let result: RuntimeWorkerResultRequest =
            serde_json::from_value(vectors["result_effect"].clone()).unwrap();
        let transition_result: RuntimeWorkerResultRequest =
            serde_json::from_value(vectors["result_transition"].clone()).unwrap();
        let mut legacy_worker_result = vectors["result_effect"].clone();
        legacy_worker_result
            .as_object_mut()
            .unwrap()
            .remove("transition");
        let _: RuntimeWorkerResultRequest = serde_json::from_value(legacy_worker_result).unwrap();
        let resume: RuntimeResumeRequest =
            serde_json::from_value(vectors["resume"].clone()).unwrap();
        let sqlite_work: RuntimeClaimedWork =
            serde_json::from_value(vectors["claimed_work_sqlite"].clone()).unwrap();
        let postgres_work: RuntimeClaimedWork =
            serde_json::from_value(vectors["claimed_work_postgres"].clone()).unwrap();
        let step_execute: kogwistar_runtime::RuntimeStepExecutePayload =
            serde_json::from_value(vectors["step_execute"].clone()).unwrap();
        assert_eq!(serde_json::to_value(submit).unwrap(), vectors["submit"]);
        assert_eq!(serde_json::to_value(claim).unwrap(), vectors["claim"]);
        assert_eq!(
            serde_json::to_value(result).unwrap(),
            vectors["result_effect"]
        );
        assert_eq!(
            serde_json::to_value(transition_result).unwrap(),
            vectors["result_transition"]
        );
        assert_eq!(serde_json::to_value(resume).unwrap(), vectors["resume"]);
        assert_eq!(
            serde_json::to_value(sqlite_work).unwrap(),
            vectors["claimed_work_sqlite"]
        );
        assert_eq!(
            serde_json::to_value(postgres_work).unwrap(),
            vectors["claimed_work_postgres"]
        );
        assert_eq!(
            serde_json::to_value(step_execute).unwrap(),
            vectors["step_execute"]
        );
    }

    #[test]
    fn native_claim_attempt_is_internal_and_matches_trace_fixture() {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/runtime-trace-context-v1.json"
        )))
        .unwrap();
        let retry_attempts = fixture["retry_attempts"].as_array().unwrap();
        let context = &fixture["context"];
        let payload = json!({
            "contract_version": 1,
            "kind": "workflow.step.execute",
            "worker_protocol": "sync-v1",
            "run_id": context["run_id"],
            "workflow_id": "workflow",
            "conversation_id": "conversation",
            "turn_node_id": null,
            "node_id": context["node_id"],
            "op": "noop",
            "executor_target": kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET,
            "op_version": "v1",
            "runtime_routes": [],
            "join_mask": 0,
            "token_id": context["token_id"],
            "parent_token_id": null,
            "step_seq": context["step_seq"],
            "expected_event_seq": 0,
            "state": {},
            "resume_effect": null
        });
        let lane = ProjectedLaneMessage {
            message_id: "message".to_owned(),
            namespace: "namespace".to_owned(),
            purpose: "runtime".to_owned(),
            inbox_id: "inbox".to_owned(),
            conversation_id: "conversation".to_owned(),
            recipient_id: "worker".to_owned(),
            sender_id: "scheduler".to_owned(),
            msg_type: "workflow.step.execute".to_owned(),
            status: "claimed".to_owned(),
            seq: 1,
            conversation_seq: 1,
            claimed_by: Some("worker".to_owned()),
            lease_until: None,
            retry_count: 1,
            created_at: 0,
            available_at: 0,
            run_id: Some("run-42".to_owned()),
            step_id: Some("parse".to_owned()),
            correlation_id: Some("correlation".to_owned()),
            payload_json: None,
            error_json: None,
            prev_message_id: None,
            next_message_id: None,
            inbox_tail_message_id: None,
            conversation_tail_message_id: None,
        };
        let python_retry_lane = ProjectedLaneMessage {
            recipient_id: "python-worker-env-test".to_owned(),
            ..lane.clone()
        };
        let python_retry_work = runtime_claimed_work(
            python_retry_lane,
            "python-process-1",
            "run-42".to_owned(),
            json!({"kind":"workflow.step.execute"}),
            0,
        );
        assert_eq!(
            python_retry_work.payload["attempt"],
            retry_attempts[1]["attempt"]
        );
        let python_retry_wire = serde_json::to_value(&python_retry_work).unwrap();
        assert!(python_retry_wire.get("attempt_number").is_none());
        assert_eq!(
            python_retry_wire["payload"]["attempt"],
            retry_attempts[1]["attempt"]
        );

        let first_python_lane = ProjectedLaneMessage {
            recipient_id: "python-worker".to_owned(),
            retry_count: 0,
            ..lane.clone()
        };
        let first_python_work = runtime_claimed_work(
            first_python_lane,
            "python-process-1",
            "run-42".to_owned(),
            json!({"kind":"workflow.step.execute"}),
            0,
        );
        assert!(first_python_work.payload.get("attempt").is_none());
        assert_eq!(retry_attempts[0]["attempt"], 1);

        for retry_case in fixture["retry_attempts"].as_array().unwrap() {
            let retry_lane = ProjectedLaneMessage {
                retry_count: retry_case["retry_count"].as_i64().unwrap(),
                ..lane.clone()
            };
            let work = runtime_claimed_work(
                retry_lane,
                "rust-native-worker",
                "run-42".to_owned(),
                payload.clone(),
                0,
            );
            let worker_wire = serde_json::to_value(&work).unwrap();
            assert!(worker_wire.get("attempt_number").is_none());

            let (handoff, request) = native_request_for_claimed_work(&work).unwrap();
            assert_eq!(handoff.message_id, "message");
            assert_eq!(handoff.run_id, "run-42");
            assert_eq!(handoff.step_id, "parse");
            assert_eq!(handoff.correlation_id, "correlation");
            assert_eq!(
                request.ephemeral_context["_trace_attempt"],
                retry_case["attempt"]
            );
            let effect = kogwistar_runtime::execute_native_step(&request).unwrap();
            assert_eq!(effect.trace_events[0]["event"], "runtime.native.step.completed");
            for (field, expected) in [
                ("run_id", json!("run-42")),
                ("workflow_id", json!("workflow")),
                ("conversation_id", json!("conversation")),
                ("node_id", json!("parse")),
                ("token_id", json!("token-7")),
                ("step_seq", json!(3)),
            ] {
                assert_eq!(effect.trace_events[0][field], expected, "field {field}");
            }
            for (field, expected) in retry_case["expected"].as_object().unwrap() {
                assert_eq!(&effect.trace_events[0][field], expected, "field {field}");
            }
            let failed = failed_native_result(handoff, &request, "TEST_FAILURE", "test failure");
            assert_eq!(failed.handoff.message_id, "message");
            assert_eq!(failed.handoff.run_id, "run-42");
            assert_eq!(failed.handoff.step_id, "parse");
            assert_eq!(failed.handoff.correlation_id, "correlation");
            let failed_effect = failed.effect.as_ref().unwrap();
            assert_eq!(
                failed_effect.status,
                kogwistar_runtime::RuntimeWorkerEffectStatus::Failed
            );
            assert_eq!(failed_effect.errors, ["test failure"]);
            assert_eq!(
                failed_effect.trace_events[0]["event"],
                "runtime.native.step.failed"
            );
            for (field, expected) in [
                ("run_id", json!("run-42")),
                ("workflow_id", json!("workflow")),
                ("conversation_id", json!("conversation")),
                ("node_id", json!("parse")),
                ("token_id", json!("token-7")),
                ("step_seq", json!(3)),
            ] {
                assert_eq!(
                    failed_effect.trace_events[0][field],
                    expected,
                    "failed trace field {field}"
                );
            }
            for (field, expected) in retry_case["expected"].as_object().unwrap() {
                assert_eq!(
                    &failed_effect.trace_events[0][field],
                    expected,
                    "failed trace field {field}"
                );
            }
        }
    }

    #[test]
    fn runtime_wire_dtos_reject_unknown_root_and_nested_fields() {
        let vectors: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/adr015-runtime-wire-v1.json"
        )))
        .unwrap();
        macro_rules! assert_unknown_field_rejected {
            ($dto:ty, $value:expr) => {{
                let mut value = $value;
                value["unexpected"] = json!(true);
                let error = serde_json::from_value::<$dto>(value).unwrap_err();
                assert!(error.to_string().contains("unknown field"));
            }};
        }

        assert_unknown_field_rejected!(RuntimeSubmitRun, vectors["submit"].clone());
        assert_unknown_field_rejected!(RuntimeClaimRequest, vectors["claim"].clone());
        assert_unknown_field_rejected!(
            RuntimeWorkerResultRequest,
            vectors["result_effect"].clone()
        );
        assert_unknown_field_rejected!(RuntimeResumeRequest, vectors["resume"].clone());
        assert_unknown_field_rejected!(RuntimeClaimedWork, vectors["claimed_work_sqlite"].clone());
        assert_unknown_field_rejected!(
            kogwistar_runtime::RuntimeStepExecutePayload,
            vectors["step_execute"].clone()
        );

        let mut nested_route = vectors["submit"].clone();
        nested_route["runtime_routes"][0]["unexpected"] = json!(true);
        let error = serde_json::from_value::<RuntimeSubmitRun>(nested_route).unwrap_err();
        assert!(error.to_string().contains("unknown field"));
    }

    #[test]
    fn runtime_start_builder_preserves_all_python_wire_identity() {
        let vectors: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/adr015-runtime-wire-v1.json"
        )))
        .unwrap();
        let mut payload = vectors["submit"].clone();
        payload["expected_event_seq"] = json!(7);
        let transition = runtime_start_transition(&payload, "run-1");
        assert_eq!(transition.run_id, "run-1");
        assert_eq!(transition.workflow_id, "wf-1");
        assert_eq!(transition.conversation_id, "conv-1");
        assert_eq!(transition.user_id.as_deref(), Some("user-1"));
        assert_eq!(transition.user_turn_node_id, None);
        assert_eq!(transition.node_id.as_deref(), Some("entry"));
        assert_eq!(transition.token_id.as_deref(), Some("run-1"));
        assert_eq!(transition.expected_event_seq, 7);
        let frontier = transition.frontier.unwrap();
        assert_eq!(frontier.pending[0].0, "entry");
        assert_eq!(frontier.pending[0].1, 1);
        assert_eq!(frontier.join_node_ids, vec!["join"]);
        assert_eq!(frontier.join_outstanding, vec![1]);
    }

    #[test]
    fn worker_protocol_tracks_runtime_kind_and_service_payload_state() {
        assert_eq!(worker_protocol_for_runtime_kind("sync"), "sync-v1");
        assert_eq!(worker_protocol_for_runtime_kind("async"), "async-v2");
        assert_eq!(worker_protocol_for_runtime_kind("unknown"), "sync-v1");
        let async_attestation = runtime_implementation_attestation("async");
        assert_eq!(async_attestation["schema_version"], 1);
        assert_eq!(async_attestation["scheduler"]["language"], "rust");
        assert_eq!(
            async_attestation["scheduler"]["version"],
            env!("CARGO_PKG_VERSION")
        );
        assert_eq!(async_attestation["worker_contract"]["protocol"], "async-v2");

        let mut sync_attestation = runtime_implementation_attestation("sync");
        sync_attestation["scheduler"]
            .as_object_mut()
            .unwrap()
            .remove("version");
        let shared_attestation: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/runtime-implementation-attestation-v1.json"
        )))
        .unwrap();
        assert_eq!(sync_attestation, shared_attestation);

        let mut state = serde_json::Map::new();
        state.insert(
            "_rt_worker_protocol".to_owned(),
            Value::String(worker_protocol_for_runtime_kind("async").to_owned()),
        );
        let recorded = kogwistar_runtime::RecordedRuntimeState {
            contract_version: kogwistar_runtime::RECORDED_RUNTIME_CONTRACT_VERSION,
            run_id: "run".to_owned(),
            workflow_id: "workflow".to_owned(),
            conversation_id: "conversation".to_owned(),
            user_turn_node_id: None,
            state,
            static_routes: Vec::new(),
            frontier: kogwistar_runtime::RuntimeFrontier {
                pending: Vec::new(),
                suspended: Vec::new(),
                join_node_ids: Vec::new(),
                join_outstanding: Vec::new(),
                join_waiters: BTreeMap::new(),
            },
            status: kogwistar_runtime::RecordedRunStatus::Running,
            last_step_seq: 0,
            wait_reason: None,
            resume_payload: None,
            last_node_id: None,
            last_token_id: None,
            last_parent_token_id: None,
        };
        let payload = kogwistar_runtime::RuntimeStepExecutePayload::from_recorded_state(
            &recorded,
            kogwistar_runtime::RuntimeStepExecuteRequest {
                node_id: "node".to_owned(),
                op: json!("op"),
                join_mask: 0,
                token_id: "token".to_owned(),
                parent_token_id: None,
                step_seq: 1,
                expected_event_seq: 1,
                resume_effect: None,
            },
        );
        assert_eq!(payload.worker_protocol, "async-v2");
    }
    struct FixedEmbeddingProvider {
        profile: kogwistar_engine::embedding::EmbeddingProfile,
        vector: Vec<f64>,
        calls: Arc<Mutex<Vec<Vec<String>>>>,
        fail: Option<Arc<std::sync::atomic::AtomicBool>>,
    }

    impl kogwistar_engine::embedding::EmbeddingProvider for FixedEmbeddingProvider {
        fn profile(&self) -> &kogwistar_engine::embedding::EmbeddingProfile {
            &self.profile
        }

        fn embed<'a>(
            &'a self,
            texts: &'a [String],
        ) -> kogwistar_engine::embedding::EmbeddingFuture<'a> {
            Box::pin(async move {
                self.calls.lock().unwrap().push(texts.to_vec());
                if self
                    .fail
                    .as_ref()
                    .is_some_and(|fail| fail.load(std::sync::atomic::Ordering::Relaxed))
                {
                    return Err(kogwistar_engine::embedding::EmbeddingError::Request(
                        "test provider failure".to_owned(),
                    ));
                }
                Ok(texts.iter().map(|_| self.vector.clone()).collect())
            })
        }
    }

    fn graph_mutation_batch_without_vector() -> GraphMutationBatchInput {
        let mut body: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/graph-mutation-batch-v1.json"
        )))
        .unwrap();
        body["mutations"][0]["record"]["document"] = json!("a source-grounded node");
        body["mutations"][0]["record"]["embedding"] = Value::Null;
        body["mutations"][0]["embedding_dim"] = json!(0);
        serde_json::from_value(body).unwrap()
    }
