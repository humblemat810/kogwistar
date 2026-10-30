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

    let mut turns = view
        .get("context_nodes")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_else(|| vec![Value::Object(turn.clone())])
        .into_iter()
        .filter_map(|record| {
            let id = record.get("id")?.as_str()?.to_owned();
            let document = record
                .get("document")?
                .as_str()
                .and_then(|raw| serde_json::from_str::<Value>(raw).ok())?;
            let metadata = document
                .get("metadata")
                .or_else(|| record.get("metadata"))?;
            let field = |key: &str| document.get(key).or_else(|| metadata.get(key));
            if field("entity_type").and_then(Value::as_str) != Some("conversation_turn")
                || field("conversation_id").and_then(Value::as_str)
                    != Some(request.conversation_id.as_str())
                || matches!(
                    field("lifecycle_status").and_then(Value::as_str),
                    Some("tombstoned" | "deleted")
                )
            {
                return None;
            }
            let role = field("role")
                .and_then(Value::as_str)
                .filter(|role| matches!(*role, "user" | "assistant" | "tool"))?;
            let content = document.get("summary")?.as_str()?.to_owned();
            let turn_index = field("turn_index").and_then(Value::as_i64).unwrap_or(-1);
            Some((turn_index, id, role.to_owned(), content))
        })
        .collect::<Vec<_>>();
    turns.sort_by(|left, right| (left.0, &left.1).cmp(&(right.0, &right.1)));
    if turns.iter().all(|(_, id, _, _)| id != turn_id) {
        return Err(invalid(
            AUTHORIZED_VIEW_KEY,
            "authorized conversation context omitted the admitted user turn",
        ));
    }
    let system_message = json!({
        "role": "system",
        "content": SYSTEM_PROMPT,
        "node_id": null,
        "source": "system"
    });
    let mut remaining_budget = 6000usize.saturating_sub((SYSTEM_PROMPT.chars().count() / 4).max(1));
    let mut selected_turns = Vec::new();
    for turn in turns.into_iter().rev().take(8) {
        let cost = (turn.3.chars().count() / 4).max(1);
        if cost <= remaining_budget {
            remaining_budget -= cost;
            selected_turns.push(turn);
        }
    }
    selected_turns.reverse();
    let mut context_messages = vec![system_message];
    context_messages.extend(
        selected_turns
            .into_iter()
            .map(|(turn_index, id, role, content)| {
                json!({
                    "role": role,
                    "content": content,
                    "node_id": id,
                    "source": "history_turn",
                    "turn_index": turn_index
                })
            }),
    );
    let context_tokens_used = context_messages
        .iter()
        .filter_map(|message| message.get("content").and_then(Value::as_str))
        .map(|content| (content.chars().count() / 4).max(1))
        .sum::<usize>();
    let context_node_ids = context_messages
        .iter()
        .filter_map(|message| message.get("node_id").and_then(Value::as_str))
        .map(str::to_owned)
        .collect::<Vec<_>>();

    let mut update = Map::new();
    update.insert("system_prompt".to_owned(), json!(SYSTEM_PROMPT));
    update.insert("question".to_owned(), json!(question));
    update.insert(
        "conversation_context_messages".to_owned(),
        Value::Array(context_messages),
    );
    update.insert(
        "conversation_context_node_ids".to_owned(),
        json!(context_node_ids),
    );
    update.insert("context_budget_tokens".to_owned(), json!(6000));
    update.insert("context_tokens_used".to_owned(), json!(context_tokens_used));
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

    #[test]
    fn projects_last_eight_authorized_turns_in_chronological_order() {
        let make_turn = |index: usize| {
            let id = format!("turn-{index}");
            json!({
                "id": id,
                "document": json!({
                    "summary": format!("message {index}"),
                    "role": if index == 9 || index % 2 == 0 { "user" } else { "assistant" },
                    "turn_index": index,
                    "metadata": {
                        "entity_type": "conversation_turn",
                        "conversation_id": "conv-1",
                        "lifecycle_status": "active"
                    }
                }).to_string(),
                "metadata": {}
            })
        };
        let turns = (0..10).map(make_turn).collect::<Vec<_>>();
        let mut req = request(turns[9].clone());
        req.turn_node_id = Some("turn-9".to_owned());
        req.state["turn_node_id"] = json!("turn-9");
        req.ephemeral_context.insert(
            AUTHORIZED_VIEW_KEY.to_owned(),
            json!({"turn": turns[9], "context_nodes": turns}),
        );

        let result = execute_get_view_and_question(&req).unwrap();
        let final_state = &result.result.unwrap()["final_state"];
        let messages = final_state["conversation_context_messages"]
            .as_array()
            .unwrap();
        assert_eq!(messages.len(), 9);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[1]["node_id"], "turn-2");
        assert_eq!(messages[1]["content"], "message 2");
        assert_eq!(messages[8]["node_id"], "turn-9");
        assert_eq!(messages[8]["content"], "message 9");
        assert_eq!(
            final_state["conversation_context_node_ids"],
            json!([
                "turn-2", "turn-3", "turn-4", "turn-5", "turn-6", "turn-7", "turn-8", "turn-9"
            ])
        );
        assert!(final_state["context_tokens_used"].as_u64().unwrap() <= 6000);
    }
}
