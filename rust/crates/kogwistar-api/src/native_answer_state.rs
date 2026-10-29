use kogwistar_runtime::{
    NativeOperationOutput, NativeOperationRegistrationError, NativeOperationRegistry,
    NativeRoutedOperationOutput, RecordedStateUpdate, RuntimeNativeExecutionError,
    RuntimeNativeStepRequest, RuntimeSuccessor,
};
use serde_json::{Map, Value, json};

pub(crate) fn register(
    registry: &mut NativeOperationRegistry,
) -> Result<(), NativeOperationRegistrationError> {
    registry.register_routed("aa_maybe_iterate", "v1", execute_maybe_iterate)
}

fn optional_integer(state: &Map<String, Value>, field: &str) -> i64 {
    state.get(field).and_then(Value::as_i64).unwrap_or_default()
}

fn selected_target(
    request: &RuntimeNativeStepRequest,
    should_iterate: bool,
) -> Result<Vec<RuntimeSuccessor>, RuntimeNativeExecutionError> {
    if request.runtime_routes.is_empty() {
        return Ok(Vec::new());
    }
    let matching = request
        .runtime_routes
        .iter()
        .filter(|route| route.source_node_id == request.node_id)
        .filter(|route| {
            if should_iterate {
                route.predicate.as_deref() == Some("aa_should_iterate")
            } else {
                route.is_default
                    && route
                        .predicate
                        .as_deref()
                        .is_none_or(|predicate| predicate == "always")
            }
        })
        .map(|route| RuntimeSuccessor {
            node_id: route.target_node_id.clone(),
            join_mask: route.join_mask,
        })
        .collect::<Vec<_>>();
    if matching.len() != 1 {
        return Err(RuntimeNativeExecutionError::InvalidState {
            field: "runtime_routes",
            reason: "expected exactly one frozen route for the iteration decision",
        });
    }
    Ok(matching)
}

fn execute_maybe_iterate(
    request: &RuntimeNativeStepRequest,
) -> Result<NativeRoutedOperationOutput, RuntimeNativeExecutionError> {
    let state = &request.state;
    let evaluation = state.get("evaluation").and_then(Value::as_object);
    let iter_idx = optional_integer(state, "iter_idx");
    let max_iter = state
        .get("max_iter")
        .and_then(Value::as_i64)
        .filter(|value| *value != 0)
        .unwrap_or(1);
    let needs_more = evaluation
        .and_then(|value| value.get("needs_more_info"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let sufficient = evaluation
        .and_then(|value| value.get("is_sufficient"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let next_iter = iter_idx.checked_add(1);
    let should_iterate = needs_more && !sufficient && next_iter.is_some_and(|next| next < max_iter);
    let recorded_iter = if should_iterate {
        next_iter.expect("iteration guard checked the increment")
    } else {
        iter_idx
    };
    let log_entry = if should_iterate {
        "aa_maybe_iterate:True"
    } else {
        "aa_maybe_iterate:False"
    };
    let mut update = Map::new();
    update.insert("should_iterate".to_owned(), json!(should_iterate));
    update.insert("iter_idx".to_owned(), json!(recorded_iter));

    let mut op_log = state
        .get("op_log")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    op_log.push(json!(log_entry));
    let mut final_state = state.clone();
    final_state.extend(update.clone());
    final_state.insert("op_log".to_owned(), Value::Array(op_log));

    Ok(NativeRoutedOperationOutput {
        output: NativeOperationOutput {
            state_update: vec![
                RecordedStateUpdate::Pair(("u".to_owned(), update)),
                RecordedStateUpdate::Pair((
                    "a".to_owned(),
                    Map::from_iter([("op_log".to_owned(), Value::String(log_entry.to_owned()))]),
                )),
            ],
            result: Some(json!({
                "workflow_status": "succeeded",
                "final_state": final_state,
                "conversation_node_id": null
            })),
        },
        successors: selected_target(request, should_iterate)?,
    })
}

#[cfg(test)]
mod tests {
    use kogwistar_runtime::{
        RUST_NATIVE_EXECUTOR_TARGET, RuntimeNativeStepRequest, RuntimeStaticRoute,
        execute_native_step_with_registry,
    };
    use serde_json::{Map, Value, json};

    fn request(state: Value) -> RuntimeNativeStepRequest {
        RuntimeNativeStepRequest {
            executor_target: RUST_NATIVE_EXECUTOR_TARGET.to_owned(),
            operation: "aa_maybe_iterate".to_owned(),
            op_version: "v1".to_owned(),
            run_id: "run-1".to_owned(),
            workflow_id: "agentic_answering.v2".to_owned(),
            node_id: "maybe-iterate".to_owned(),
            conversation_id: "conv-1".to_owned(),
            turn_node_id: None,
            token_id: "token-1".to_owned(),
            parent_token_id: None,
            step_seq: 7,
            state: serde_json::from_value::<Map<String, Value>>(state).unwrap(),
            runtime_routes: vec![
                RuntimeStaticRoute {
                    edge_id: "iterate-again".to_owned(),
                    source_node_id: "maybe-iterate".to_owned(),
                    target_node_id: "retrieve".to_owned(),
                    aliases: Vec::new(),
                    join_mask: 0,
                    predicate: Some("aa_should_iterate".to_owned()),
                    multiplicity: "one".to_owned(),
                    is_default: false,
                    priority: 0,
                    source_fanout: false,
                },
                RuntimeStaticRoute {
                    edge_id: "persist-answer".to_owned(),
                    source_node_id: "maybe-iterate".to_owned(),
                    target_node_id: "persist".to_owned(),
                    aliases: Vec::new(),
                    join_mask: 0,
                    predicate: None,
                    multiplicity: "one".to_owned(),
                    is_default: true,
                    priority: 100,
                    source_fanout: false,
                },
            ],
        }
    }

    #[test]
    fn maybe_iterate_matches_python_loop_predicate_and_advances_only_when_routed() {
        let registry = crate::native_answer_persist::registry().unwrap();
        let effect = execute_native_step_with_registry(
            &request(json!({
                "evaluation": {"needs_more_info": true, "is_sufficient": false},
                "iter_idx": 0,
                "max_iter": 2,
                "op_log": ["aa_evaluate_answer"]
            })),
            &registry,
        )
        .unwrap();
        assert!(effect.graph_mutation_batch_v2.is_none());
        let result = effect.result.unwrap();
        assert_eq!(effect.successors[0].node_id, "retrieve");
        assert_eq!(result["final_state"]["should_iterate"], true);
        assert_eq!(result["final_state"]["iter_idx"], 1);
        assert_eq!(
            result["final_state"]["op_log"],
            json!(["aa_evaluate_answer", "aa_maybe_iterate:True"])
        );
    }

    #[test]
    fn maybe_iterate_stops_for_sufficient_or_exhausted_runs_and_uses_python_defaults() {
        let registry = crate::native_answer_persist::registry().unwrap();
        for state in [
            json!({
                "evaluation": {"needs_more_info": true, "is_sufficient": true},
                "iter_idx": 0,
                "max_iter": 2
            }),
            json!({
                "evaluation": {"needs_more_info": true, "is_sufficient": false},
                "iter_idx": 1,
                "max_iter": 2
            }),
            json!({"evaluation": {"needs_more_info": true}}),
        ] {
            let effect = execute_native_step_with_registry(&request(state), &registry).unwrap();
            assert_eq!(effect.successors[0].node_id, "persist");
            let result = effect.result.unwrap();
            assert_eq!(result["final_state"]["should_iterate"], false);
        }
    }

    #[test]
    fn maybe_iterate_fails_closed_when_frozen_routes_do_not_match_its_decision() {
        let registry = crate::native_answer_persist::registry().unwrap();
        let mut request = request(json!({
            "evaluation": {"needs_more_info": true, "is_sufficient": false},
            "iter_idx": 0,
            "max_iter": 2
        }));
        request.runtime_routes.remove(0);
        assert!(execute_native_step_with_registry(&request, &registry).is_err());
    }
}
