use kogwistar_runtime::{
    NativeOperationOutput, NativeOperationRegistrationError, NativeOperationRegistry,
    RecordedStateUpdate, RuntimeNativeExecutionError, RuntimeNativeStepRequest,
};
use serde_json::{Map, Value, json};

const AUTHORIZED_VIEW_KEY: &str = "authorized_answer_view";
const SYSTEM_PROMPT: &str = "You are a helpful assistant. Answer the user using the conversation and any provided evidence.";

pub(crate) fn register(
    registry: &mut NativeOperationRegistry,
) -> Result<(), NativeOperationRegistrationError> {
    registry.register(
        "aa_get_view_and_question",
        "v1",
        execute_get_view_and_question,
    )
}

fn invalid(field: &'static str, reason: &'static str) -> RuntimeNativeExecutionError {
    RuntimeNativeExecutionError::InvalidState { field, reason }
}

fn execute_get_view_and_question(
    request: &RuntimeNativeStepRequest,
) -> Result<NativeOperationOutput, RuntimeNativeExecutionError> {
    let view = request
        .ephemeral_context
        .get(AUTHORIZED_VIEW_KEY)
        .and_then(Value::as_object)
        .ok_or_else(|| {
            invalid(
                AUTHORIZED_VIEW_KEY,
                "host-authorized current-turn context is required",
            )
        })?;
    let turn = view
        .get("turn")
        .and_then(Value::as_object)
        .ok_or_else(|| invalid(AUTHORIZED_VIEW_KEY, "authorized user turn is missing"))?;
    let turn_id = turn
        .get("id")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| invalid(AUTHORIZED_VIEW_KEY, "authorized turn ID is missing"))?;
    let expected_turn_id = request
        .turn_node_id
        .as_deref()
        .or_else(|| request.state.get("turn_node_id").and_then(Value::as_str))
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| invalid("turn_node_id", "admitted user-turn ID is required"))?;
    if turn_id != expected_turn_id {
        return Err(invalid(
            AUTHORIZED_VIEW_KEY,
            "authorized turn does not match the admitted user turn",
        ));
    }
    let document = turn
        .get("document")
        .and_then(Value::as_str)
        .and_then(|raw| serde_json::from_str::<Value>(raw).ok())
        .ok_or_else(|| invalid(AUTHORIZED_VIEW_KEY, "authorized turn document is malformed"))?;
    let metadata = turn
        .get("metadata")
        .and_then(Value::as_object)
        .ok_or_else(|| invalid(AUTHORIZED_VIEW_KEY, "authorized turn metadata is missing"))?;
    let document_metadata = document.get("metadata").and_then(Value::as_object);
    let value = |key: &str| {
        document
            .get(key)
            .or_else(|| document_metadata.and_then(|metadata| metadata.get(key)))
            .or_else(|| metadata.get(key))
    };
    if value("entity_type").and_then(Value::as_str) != Some("conversation_turn")
        || value("conversation_id").and_then(Value::as_str)
            != Some(request.conversation_id.as_str())
        || value("role").and_then(Value::as_str) != Some("user")
        || matches!(
            value("lifecycle_status").and_then(Value::as_str),
            Some("tombstoned" | "deleted")
        )
    {
        return Err(invalid(
            AUTHORIZED_VIEW_KEY,
            "turn must be an active user turn in the admitted conversation",
        ));
    }
    let question = document
        .get("summary")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| invalid(AUTHORIZED_VIEW_KEY, "user turn has no question text"))?;

    let mut update = Map::new();
    update.insert("system_prompt".to_owned(), json!(SYSTEM_PROMPT));
    update.insert("question".to_owned(), json!(question));
    let mut final_state = request.state.clone();
    final_state.extend(update.clone());
    let mut op_log = request
        .state
        .get("op_log")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    op_log.push(json!("aa_get_view_and_question"));
    final_state.insert("op_log".to_owned(), Value::Array(op_log));
    let mut auxiliary = Map::new();
    auxiliary.insert("op_log".to_owned(), json!("aa_get_view_and_question"));
    Ok(NativeOperationOutput {
        state_update: vec![
            RecordedStateUpdate::Pair(("u".to_owned(), update)),
            RecordedStateUpdate::Pair(("a".to_owned(), auxiliary)),
        ],
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
    use kogwistar_runtime::{RUST_NATIVE_EXECUTOR_TARGET, execute_native_step_with_registry};

    fn request(turn: Value) -> RuntimeNativeStepRequest {
        let mut ephemeral_context = Map::new();
        ephemeral_context.insert(AUTHORIZED_VIEW_KEY.to_owned(), json!({"turn": turn}));
        RuntimeNativeStepRequest {
            executor_target: RUST_NATIVE_EXECUTOR_TARGET.to_owned(),
            operation: "aa_get_view_and_question".to_owned(),
            op_version: "v1".to_owned(),
            run_id: "run-1".to_owned(),
            workflow_id: "agentic_answering.v2".to_owned(),
            node_id: "view-1".to_owned(),
            conversation_id: "conv-1".to_owned(),
            turn_node_id: Some("turn-1".to_owned()),
            token_id: "token-1".to_owned(),
            parent_token_id: None,
            step_seq: 1,
            state: serde_json::from_value(json!({
                "conversation_id": "conv-1",
                "turn_node_id": "turn-1",
                "op_log": ["aa_prepare"]
            }))
            .unwrap(),
            runtime_routes: Vec::new(),
            ephemeral_context,
        }
    }

    fn user_turn() -> Value {
        json!({
            "id": "turn-1",
            "document": json!({
                "summary": "What is the Rust-native path?",
                "metadata": {
                    "entity_type": "conversation_turn",
                    "conversation_id": "conv-1",
                    "role": "user",
                    "lifecycle_status": "active"
                }
            })
            .to_string(),
            "metadata": {}
        })
    }

    #[test]
    fn projects_python_compatible_question_and_system_prompt() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../../contracts/golden/answer-view-operation-v1.json"
        ))
        .unwrap();
        let request = request(user_turn());
        let mut registry = NativeOperationRegistry::default();
        register(&mut registry).unwrap();
        let effect = execute_native_step_with_registry(&request, &registry).unwrap();
        assert_eq!(effect.state_update.len(), 2);
        assert_eq!(
            effect.result.as_ref().unwrap()["final_state"]["question"],
            fixture["expected"]["question"]
        );
        assert_eq!(
            effect.result.as_ref().unwrap()["final_state"]["system_prompt"],
            fixture["expected"]["system_prompt"]
        );
        assert_eq!(
            effect.result.as_ref().unwrap()["final_state"]["op_log"],
            fixture["expected"]["op_log"]
        );
        assert_eq!(SYSTEM_PROMPT, fixture["expected"]["system_prompt"]);
    }

    #[test]
    fn fails_closed_for_mismatched_turn_scope_or_non_user_turn() {
        let mut mismatched = user_turn();
        mismatched["id"] = json!("other-turn");
        assert!(execute_get_view_and_question(&request(mismatched)).is_err());

        let mut wrong_role = user_turn();
        wrong_role["document"] = json!(
            json!({
                "summary": "not the user's prompt",
                "metadata": {
                    "entity_type": "conversation_turn",
                    "conversation_id": "conv-1",
                    "role": "assistant",
                    "lifecycle_status": "active"
                }
            })
            .to_string()
        );
        assert!(execute_get_view_and_question(&request(wrong_role)).is_err());

        let mut wrong_conversation = user_turn();
        wrong_conversation["document"] = json!(
            json!({
                "summary": "cross-scope prompt",
                "metadata": {
                    "entity_type": "conversation_turn",
                    "conversation_id": "other-conversation",
                    "role": "user",
                    "lifecycle_status": "active"
                }
            })
            .to_string()
        );
        assert!(execute_get_view_and_question(&request(wrong_conversation)).is_err());
    }

    #[test]
    fn host_authorized_context_is_required_and_inactive_turns_are_rejected() {
        let mut req = request(user_turn());
        req.ephemeral_context.clear();
        let mut registry = NativeOperationRegistry::default();
        register(&mut registry).unwrap();
        assert!(execute_native_step_with_registry(&req, &registry).is_err());

        let mut inactive = user_turn();
        inactive["document"] = json!(
            json!({
                "summary": "deleted question",
                "metadata": {
                    "entity_type": "conversation_turn",
                    "conversation_id": "conv-1",
                    "role": "user",
                    "lifecycle_status": "tombstoned"
                }
            })
            .to_string()
        );
        assert!(execute_get_view_and_question(&request(inactive)).is_err());
    }
}
