use kogwistar_contracts::{GraphDefaultAclIntentV1, GraphMutationBatchV2};
use kogwistar_runtime::{
    NativeOperationRegistrationError, NativeOperationRegistry, RecordedStateUpdate,
    RuntimeNativeExecutionError, RuntimeNativeStepRequest,
};
use serde_json::{Map, Value, json};

pub(crate) fn register(
    registry: &mut NativeOperationRegistry,
) -> Result<(), NativeOperationRegistrationError> {
    registry.register_graph_v2("aa_prepare", "v1", execute_prepare)
}

fn invalid(field: &'static str, reason: &'static str) -> RuntimeNativeExecutionError {
    RuntimeNativeExecutionError::InvalidState { field, reason }
}

fn required_string<'a>(
    state: &'a Map<String, Value>,
    field: &'static str,
) -> Result<&'a str, RuntimeNativeExecutionError> {
    state
        .get(field)
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| invalid(field, "expected a non-empty string"))
}

fn optional_integer(state: &Map<String, Value>, field: &str) -> i64 {
    state.get(field).and_then(Value::as_i64).unwrap_or_default()
}

fn execute_prepare(
    request: &RuntimeNativeStepRequest,
) -> Result<
    (
        kogwistar_runtime::NativeOperationOutput,
        Option<GraphMutationBatchV2>,
    ),
    RuntimeNativeExecutionError,
> {
    let state = &request.state;
    let conversation_id = required_string(state, "conversation_id")?;
    if conversation_id != request.conversation_id {
        return Err(invalid(
            "conversation_id",
            "must match the admitted runtime conversation",
        ));
    }
    let run_id = state
        .get("run_id")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .unwrap_or(&request.run_id);
    let run_node_id = crate::native_answer_persist::stable_conversation_id(
        "ptr",
        &[
            ("scope", &format!("conv:{conversation_id}")),
            ("pointer_kind", "agent_run"),
            ("target_kind", "run"),
            ("target_id", run_id),
        ],
    );

    let run_step_seq = optional_integer(state, "run_step_seq");
    let attempt_seq = optional_integer(state, "attempt_seq");
    let iter_idx = optional_integer(state, "iter_idx");
    let max_candidates = state
        .get("max_candidates")
        .and_then(Value::as_i64)
        .filter(|value| *value != 0)
        .unwrap_or(20);
    let materialize_depth = state
        .get("materialize_depth")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .unwrap_or("shallow");

    let mut prepare_update = Map::new();
    prepare_update.insert("run_id".to_owned(), json!(run_id));
    prepare_update.insert("run_node_id".to_owned(), json!(run_node_id));
    prepare_update.insert("run_step_seq".to_owned(), json!(run_step_seq));
    prepare_update.insert("attempt_seq".to_owned(), json!(attempt_seq));
    prepare_update.insert("iter_idx".to_owned(), json!(iter_idx));
    prepare_update.insert("max_candidates".to_owned(), json!(max_candidates));
    prepare_update.insert("materialize_depth".to_owned(), json!(materialize_depth));
    if let Some(summary) = state
        .get("prev_turn_meta_summary")
        .filter(|value| value.is_object())
    {
        prepare_update.insert("prev_turn_meta_summary".to_owned(), summary.clone());
    }

    let span = crate::native_answer_persist::span_payload();
    let node_metadata = json!({
        "level_from_root": 0,
        "entity_type": "agent_run",
        "in_conversation_chain": false,
        "lifecycle_status": "active",
        "redirect_to_id": null
    });
    let agent_name = state
        .get("agent_class_name")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .unwrap_or("AgenticAnsweringAgent");
    let node_payload = json!({
        "id": run_node_id,
        "label": format!("Agent {agent_name} Run {run_id}"),
        "type": "entity",
        "summary": format!("Agent run anchor {run_id}"),
        "domain_id": null,
        "canonical_entity_id": null,
        "properties": {"run_id": run_id, "entity_type": "agent_run"},
        "mentions": [{"spans": [span], "multimodal_spans": []}],
        "embedding": null,
        "doc_id": null,
        "metadata": node_metadata,
        "level_from_root": 0,
        "role": "system",
        "turn_index": null,
        "conversation_id": conversation_id,
        "user_id": null
    });

    let scope = state
        .get("_rt_graph_scope")
        .filter(|value| value.is_object())
        .ok_or_else(|| invalid("_rt_graph_scope", "trusted graph scope is missing"))?;
    let namespace = scope
        .get("storage_namespace")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| invalid("_rt_graph_scope", "trusted storage namespace is missing"))?;
    let graph_scope = json!({
        "storage_namespace": namespace,
        "workspace": scope.get("workspace").cloned().unwrap_or(Value::Null),
        "graph_space": scope.get("graph_space").cloned().unwrap_or(Value::Null)
    });
    let node_id = node_payload["id"]
        .as_str()
        .expect("native run node ID is constructed as a string");
    let document_fields = crate::native_answer_persist::model_document_fields(
        &node_payload,
        &["embedding", "metadata"],
    );
    let mut metadata = node_metadata.as_object().cloned().unwrap_or_default();
    metadata.insert("_class_name".to_owned(), json!("ConversationNode"));
    let (mutation, jobs) = crate::native_memory_pin::graph_mutation(
        request,
        &graph_scope,
        node_id,
        format!(
            "native-aa-prepare|{}|{}|{}",
            request.run_id, request.node_id, node_id
        ),
        "node",
        "conversation_node",
        node_payload.clone(),
        document_fields,
        metadata,
    )?;
    let jobs = jobs
        .into_iter()
        .filter(|job| job.get("index_kind").and_then(Value::as_str) != Some("node_embedding"))
        .collect();
    let batch = GraphMutationBatchV2 {
        contract_version: 2,
        atomic: true,
        mutations: vec![mutation],
        index_jobs: serde_json::from_value(Value::Array(jobs))
            .map_err(|_| invalid("aa_prepare", "index job intent is invalid"))?,
        default_acl: GraphDefaultAclIntentV1 {
            truth_graph: "conversation".to_owned(),
        },
    };
    batch
        .validate()
        .map_err(|_| invalid("aa_prepare", "atomic graph batch is invalid"))?;

    let op_log = state
        .get("op_log")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut updated_op_log = op_log;
    updated_op_log.push(json!("aa_prepare"));
    let mut final_state = state.clone();
    final_state.extend(prepare_update.clone());
    final_state.insert("op_log".to_owned(), Value::Array(updated_op_log.clone()));
    let output = kogwistar_runtime::NativeOperationOutput {
        state_update: vec![
            RecordedStateUpdate::Pair(("u".to_owned(), prepare_update)),
            RecordedStateUpdate::Pair((
                "a".to_owned(),
                Map::from_iter([("op_log".to_owned(), Value::String("aa_prepare".to_owned()))]),
            )),
        ],
        result: Some(json!({
            "workflow_status": "succeeded",
            "final_state": final_state,
            "conversation_node_id": run_node_id
        })),
    };
    Ok((output, Some(batch)))
}

#[cfg(test)]
mod tests {
    use super::execute_prepare;
    use kogwistar_runtime::{
        RUST_NATIVE_EXECUTOR_TARGET, RuntimeNativeStepRequest, execute_native_step_with_registry,
    };
    use serde_json::{Map, Value, json};

    fn request(state: Value) -> RuntimeNativeStepRequest {
        RuntimeNativeStepRequest {
            executor_target: RUST_NATIVE_EXECUTOR_TARGET.to_owned(),
            operation: "aa_prepare".to_owned(),
            op_version: "v1".to_owned(),
            run_id: "runtime-run".to_owned(),
            workflow_id: "agentic_answering.v2".to_owned(),
            node_id: "prepare".to_owned(),
            conversation_id: "conv-1".to_owned(),
            turn_node_id: None,
            token_id: "token-1".to_owned(),
            parent_token_id: None,
            step_seq: 1,
            state: serde_json::from_value::<Map<String, Value>>(state).unwrap(),
            runtime_routes: Vec::new(),
        }
    }

    fn state() -> Value {
        json!({
            "conversation_id": "conv-1",
            "run_id": "run-1",
            "run_step_seq": 2,
            "attempt_seq": 1,
            "iter_idx": 3,
            "max_candidates": 12,
            "materialize_depth": "deep",
            "prev_turn_meta_summary": {
                "prev_node_char_distance_from_last_summary": 90,
                "prev_node_distance_from_last_summary": 4,
                "tail_turn_index": 2
            },
            "_rt_graph_scope": {
                "storage_namespace": "ns-a",
                "workspace": "ws-a",
                "graph_space": "conversation"
            }
        })
    }

    #[test]
    fn prepare_initializes_state_and_emits_python_compatible_run_anchor() {
        let request = request(state());
        let registry = crate::native_answer_persist::registry().unwrap();
        let effect = execute_native_step_with_registry(&request, &registry).unwrap();
        let batch = effect.graph_mutation_batch_v2.unwrap();
        batch.validate().unwrap();
        assert_eq!(batch.mutations.len(), 1);
        assert_eq!(batch.mutations[0].scope.namespace, "ns-a");
        assert_eq!(
            batch.mutations[0].payload["metadata"]["entity_type"],
            "agent_run"
        );
        assert_eq!(batch.mutations[0].payload["role"], "system");
        assert_eq!(batch.index_jobs.len(), 2);
        let result = effect.result.unwrap();
        assert_eq!(result["conversation_node_id"], batch.mutations[0].record.id);
        assert_eq!(result["final_state"]["run_id"], "run-1");
        assert_eq!(result["final_state"]["max_candidates"], 12);
        assert_eq!(result["final_state"]["materialize_depth"], "deep");
        assert_eq!(result["final_state"]["op_log"][0], "aa_prepare");
        assert_eq!(
            result["final_state"]["run_node_id"],
            "ptr:he6soktrn72ewrish2ra"
        );
    }

    #[test]
    fn prepare_defaults_policy_and_fails_closed_on_scope_drift() {
        let mut minimal = state();
        for field in [
            "run_id",
            "run_step_seq",
            "attempt_seq",
            "iter_idx",
            "max_candidates",
            "materialize_depth",
            "prev_turn_meta_summary",
        ] {
            minimal.as_object_mut().unwrap().remove(field);
        }
        let runtime_request = request(minimal.clone());
        let registry = crate::native_answer_persist::registry().unwrap();
        let effect = execute_native_step_with_registry(&runtime_request, &registry).unwrap();
        let result = effect.result.unwrap();
        assert_eq!(result["final_state"]["run_id"], "runtime-run");
        assert_eq!(result["final_state"]["max_candidates"], 20);
        assert_eq!(result["final_state"]["materialize_depth"], "shallow");

        minimal["conversation_id"] = json!("another-conversation");
        assert!(execute_prepare(&request(minimal)).is_err());
    }
}
