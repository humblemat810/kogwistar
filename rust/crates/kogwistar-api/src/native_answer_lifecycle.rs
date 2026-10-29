use kogwistar_runtime::{
    NativeOperationOutput, NativeOperationRegistrationError, NativeOperationRegistry,
    RecordedStateUpdate, RuntimeNativeStepRequest,
};
use serde_json::{Map, Value, json};

pub(crate) fn register(
    registry: &mut NativeOperationRegistry,
) -> Result<(), NativeOperationRegistrationError> {
    registry.register("start", "v1", execute_start)?;
    registry.register("end", "v1", execute_end)
}

fn execute_start(
    request: &RuntimeNativeStepRequest,
) -> Result<NativeOperationOutput, kogwistar_runtime::RuntimeNativeExecutionError> {
    execute_lifecycle_step(
        request,
        "start",
        Map::from_iter([
            ("started".to_owned(), Value::Bool(true)),
            ("turn_index".to_owned(), Value::from(0)),
        ]),
    )
}

fn execute_end(
    request: &RuntimeNativeStepRequest,
) -> Result<NativeOperationOutput, kogwistar_runtime::RuntimeNativeExecutionError> {
    execute_lifecycle_step(
        request,
        "end",
        Map::from_iter([("done".to_owned(), Value::Bool(true))]),
    )
}

fn execute_lifecycle_step(
    request: &RuntimeNativeStepRequest,
    operation: &str,
    mut update: Map<String, Value>,
) -> Result<NativeOperationOutput, kogwistar_runtime::RuntimeNativeExecutionError> {
    let mut op_log = request
        .state
        .get("op_log")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    op_log.push(json!(operation));
    update.insert("op_log".to_owned(), Value::Array(op_log));

    let mut final_state = request.state.clone();
    final_state.extend(update.clone());
    Ok(NativeOperationOutput {
        state_update: vec![RecordedStateUpdate::Pair(("u".to_owned(), update))],
        result: Some(json!({
            "workflow_status": "succeeded",
            "final_state": final_state,
            "conversation_node_id": null
        })),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn request(operation: &str) -> RuntimeNativeStepRequest {
        RuntimeNativeStepRequest {
            executor_target: kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET.to_owned(),
            operation: operation.to_owned(),
            op_version: "v1".to_owned(),
            run_id: "run-1".to_owned(),
            workflow_id: "agentic_answering.v2".to_owned(),
            node_id: operation.to_owned(),
            conversation_id: "conv-1".to_owned(),
            turn_node_id: None,
            token_id: "token-1".to_owned(),
            parent_token_id: None,
            step_seq: 0,
            state: serde_json::from_value(json!({"op_log": ["prior"]})).unwrap(),
            runtime_routes: Vec::new(),
            ephemeral_context: Map::new(),
        }
    }

    #[test]
    fn lifecycle_handlers_match_python_state_contract() {
        let mut registry = NativeOperationRegistry::default();
        register(&mut registry).unwrap();

        let start =
            kogwistar_runtime::execute_native_step_with_registry(&request("start"), &registry)
                .unwrap();
        let started = start.result.unwrap()["final_state"].clone();
        assert_eq!(started["started"], true);
        assert_eq!(started["turn_index"], 0);
        assert_eq!(started["op_log"], json!(["prior", "start"]));

        let end = kogwistar_runtime::execute_native_step_with_registry(&request("end"), &registry)
            .unwrap();
        let ended = end.result.unwrap()["final_state"].clone();
        assert_eq!(ended["done"], true);
        assert_eq!(ended["op_log"], json!(["prior", "end"]));
    }
}
