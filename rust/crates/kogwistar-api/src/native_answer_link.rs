use kogwistar_contracts::{GraphDefaultAclIntentV1, GraphMutationBatchV2, stable_id};
use kogwistar_runtime::{
    NativeOperationOutput, NativeOperationRegistrationError, NativeOperationRegistry,
    RecordedStateUpdate, RuntimeNativeExecutionError, RuntimeNativeStepRequest,
};
use serde_json::{Map, Value, json};

const AUTHORIZED_TURNS_KEY: &str = "authorized_turn_records";

pub(crate) fn register(
    registry: &mut NativeOperationRegistry,
) -> Result<(), NativeOperationRegistrationError> {
    registry.register_graph_v2("link_assistant_turn", "v1", execute_link_assistant_turn)
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

fn python_repr(value: &str) -> String {
    let quote = if value.contains('\'') && !value.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut result = String::with_capacity(value.len() + 2);
    result.push(quote);
    for ch in value.chars() {
        match ch {
            '\\' => result.push_str("\\\\"),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            ch if ch == quote => {
                result.push('\\');
                result.push(ch);
            }
            ch if ch.is_control() => result.push_str(&format!("\\x{:02x}", ch as u32)),
            ch => result.push(ch),
        }
    }
    result.push(quote);
    result
}

fn python_list_repr(values: &[&str]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(|value| python_repr(value))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn response_node_id(state: &Map<String, Value>) -> Option<&str> {
    state
        .get("answer")
        .and_then(Value::as_object)
        .and_then(|answer| {
            answer
                .get("response_node_id")
                .or_else(|| answer.get("assistant_turn_node_id"))
        })
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .or_else(|| {
            state
                .get("agentic_answering_result")
                .and_then(Value::as_object)
                .and_then(|result| {
                    result
                        .get("assistant_turn_node_id")
                        .or_else(|| result.get("response_node_id"))
                })
                .and_then(Value::as_str)
                .filter(|value| !value.trim().is_empty())
        })
}

fn execute_link_assistant_turn(
    request: &RuntimeNativeStepRequest,
) -> Result<(NativeOperationOutput, Option<GraphMutationBatchV2>), RuntimeNativeExecutionError> {
    let state = &request.state;
    let conversation_id = required_string(state, "conversation_id")?;
    if conversation_id != request.conversation_id {
        return Err(invalid(
            "conversation_id",
            "must match the admitted runtime conversation",
        ));
    }
    let user_turn_id = required_string(state, "turn_node_id")?;
    let user_id = required_string(state, "user_id")?;
    let run_owner_id = required_string(state, "_rt_run_owner_id")?;
    if user_id != run_owner_id {
        return Err(invalid(
            "user_id",
            "must match the durable runtime run owner",
        ));
    }

    let Some(response_id) = response_node_id(state) else {
        let mut final_state = state.clone();
        let mut op_log = state
            .get("op_log")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        op_log.push(json!("link_assistant_turn:skipped"));
        final_state.insert("op_log".to_owned(), Value::Array(op_log));
        return Ok((
            NativeOperationOutput {
                state_update: vec![RecordedStateUpdate::Pair((
                    "a".to_owned(),
                    Map::from_iter([("op_log".to_owned(), json!("link_assistant_turn:skipped"))]),
                ))],
                result: Some(json!({
                    "workflow_status": "succeeded",
                    "final_state": final_state,
                    "conversation_node_id": null
                })),
            },
            None,
        ));
    };
    validate_authorized_turns(request, conversation_id, user_turn_id, response_id)?;

    let turn_index = state
        .get("turn_index")
        .and_then(Value::as_i64)
        .unwrap_or_default();
    let source_ids = python_list_repr(&[user_turn_id]);
    let target_ids = python_list_repr(&[response_id]);
    let edge_id = stable_id(
        "conversation.edge",
        &[
            user_id.to_owned(),
            conversation_id.to_owned(),
            "next_turn".to_owned(),
            turn_index.to_string(),
            source_ids,
            target_ids,
            "[]".to_owned(),
            "[]".to_owned(),
            "conversation_edge".to_owned(),
        ],
    )
    .to_string();
    let previous = state
        .get("prev_turn_meta_summary")
        .and_then(Value::as_object);
    let char_distance = previous
        .and_then(|summary| summary.get("prev_node_char_distance_from_last_summary"))
        .and_then(Value::as_i64)
        .unwrap_or_default();
    let turn_distance = previous
        .and_then(|summary| summary.get("prev_node_distance_from_last_summary"))
        .and_then(Value::as_i64)
        .unwrap_or_default();
    let tail_turn_index = previous
        .and_then(|summary| summary.get("tail_turn_index"))
        .and_then(Value::as_i64)
        .unwrap_or_default();
    let span = state
        .get("self_span")
        .filter(|value| value.is_object())
        .cloned()
        .unwrap_or_else(crate::native_answer_persist::span_payload);
    let payload = json!({
        "id": edge_id,
        "source_ids": [user_turn_id],
        "target_ids": [response_id],
        "relation": "next_turn",
        "label": "next_turn",
        "type": "relationship",
        "summary": "Sequential flow",
        "doc_id": format!("conv:{conversation_id}"),
        "mentions": [{"spans": [span], "multimodal_spans": []}],
        "domain_id": null,
        "canonical_entity_id": null,
        "properties": {"entity_type": "conversation_edge"},
        "embedding": null,
        "metadata": {
            "relation": "next_turn",
            "target_id": response_id,
            "char_distance_from_last_summary": char_distance,
            "turn_distance_from_last_summary": turn_distance,
            "tail_turn_index": tail_turn_index,
            "causal_type": "chain",
            "lifecycle_status": "active",
            "redirect_to_id": null
        },
        "source_edge_ids": [],
        "target_edge_ids": []
    });
    let scope = state
        .get("_rt_graph_scope")
        .filter(|value| value.is_object())
        .ok_or_else(|| invalid("_rt_graph_scope", "trusted graph scope is missing"))?;
    let graph_scope = json!({
        "storage_namespace": scope.get("storage_namespace").cloned().unwrap_or(Value::Null),
        "workspace": scope.get("workspace").cloned().unwrap_or(Value::Null),
        "graph_space": scope.get("graph_space").cloned().unwrap_or(Value::Null)
    });
    let document = crate::native_answer_persist::model_document_fields(&payload, &[]);
    let mut metadata = payload["metadata"].as_object().cloned().unwrap_or_default();
    metadata.insert("_class_name".to_owned(), json!("ConversationEdge"));
    let (mutation, jobs) = crate::native_memory_pin::graph_mutation(
        request,
        &graph_scope,
        &edge_id,
        format!(
            "native-link-assistant|{}|{}|{}",
            request.run_id, request.node_id, edge_id
        ),
        "edge",
        "conversation_edge",
        payload,
        document,
        metadata,
    )?;
    let jobs = jobs
        .into_iter()
        .filter(|job| job.get("index_kind").and_then(Value::as_str) != Some("node_embedding"))
        .collect::<Vec<_>>();
    let batch = GraphMutationBatchV2 {
        contract_version: 2,
        atomic: true,
        mutations: vec![mutation],
        index_jobs: serde_json::from_value(Value::Array(jobs))
            .map_err(|_| invalid("link_assistant_turn", "index job intent is invalid"))?,
        default_acl: GraphDefaultAclIntentV1 {
            truth_graph: "conversation".to_owned(),
        },
    };
    batch
        .validate()
        .map_err(|_| invalid("link_assistant_turn", "atomic graph batch is invalid"))?;

    let mut update = Map::new();
    update.insert("assistant_turn_edge_id".to_owned(), json!(edge_id));
    update.insert("turn_index".to_owned(), json!(turn_index));
    let mut final_state = state.clone();
    final_state.extend(update.clone());
    let mut op_log = state
        .get("op_log")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    op_log.push(json!("link_assistant_turn"));
    final_state.insert("op_log".to_owned(), Value::Array(op_log));
    let output = NativeOperationOutput {
        state_update: vec![
            RecordedStateUpdate::Pair(("u".to_owned(), update)),
            RecordedStateUpdate::Pair((
                "a".to_owned(),
                Map::from_iter([("op_log".to_owned(), json!("link_assistant_turn"))]),
            )),
        ],
        result: Some(json!({
            "workflow_status": "succeeded",
            "final_state": final_state,
            "conversation_node_id": edge_id
        })),
    };
    Ok((output, Some(batch)))
}

fn validate_authorized_turns(
    request: &RuntimeNativeStepRequest,
    conversation_id: &str,
    user_turn_id: &str,
    response_id: &str,
) -> Result<(), RuntimeNativeExecutionError> {
    let records = request
        .ephemeral_context
        .get(AUTHORIZED_TURNS_KEY)
        .filter(|value| value.is_object())
        .ok_or_else(|| {
            invalid(
                AUTHORIZED_TURNS_KEY,
                "host-authorized turn reads are required",
            )
        })?;
    let nodes = records
        .get("nodes")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid(AUTHORIZED_TURNS_KEY, "authorized turn records are missing"))?;
    if nodes.len() != 2 {
        return Err(invalid(
            AUTHORIZED_TURNS_KEY,
            "both visible conversation turns are required",
        ));
    }
    for (id, expected_type) in [
        (user_turn_id, "conversation_turn"),
        (response_id, "assistant_turn"),
    ] {
        let record = nodes
            .iter()
            .find(|record| record.get("id").and_then(Value::as_str) == Some(id))
            .ok_or_else(|| invalid(AUTHORIZED_TURNS_KEY, "a linked turn is not visible"))?;
        let metadata = record
            .get("metadata")
            .and_then(Value::as_object)
            .ok_or_else(|| invalid(AUTHORIZED_TURNS_KEY, "turn metadata is missing"))?;
        let actual_type = metadata.get("entity_type").and_then(Value::as_str);
        let valid_type = actual_type == Some(expected_type)
            || (expected_type == "assistant_turn" && actual_type == Some("conversation_turn"));
        if metadata.get("conversation_id").and_then(Value::as_str) != Some(conversation_id)
            || !valid_type
            || matches!(
                metadata.get("lifecycle_status").and_then(Value::as_str),
                Some("tombstoned" | "deleted")
            )
        {
            return Err(invalid(
                AUTHORIZED_TURNS_KEY,
                "turn scope, type, or lifecycle is invalid",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{AUTHORIZED_TURNS_KEY, execute_link_assistant_turn, python_list_repr};
    use kogwistar_runtime::{RUST_NATIVE_EXECUTOR_TARGET, RuntimeNativeStepRequest};
    use serde_json::{Map, Value, json};

    fn request(state: Value) -> RuntimeNativeStepRequest {
        RuntimeNativeStepRequest {
            executor_target: RUST_NATIVE_EXECUTOR_TARGET.to_owned(),
            operation: "link_assistant_turn".to_owned(),
            op_version: "v1".to_owned(),
            run_id: "answer-run".to_owned(),
            workflow_id: "agentic_answering.v2".to_owned(),
            node_id: "link".to_owned(),
            conversation_id: "conversation-1".to_owned(),
            turn_node_id: Some("user-turn".to_owned()),
            token_id: "token-1".to_owned(),
            parent_token_id: None,
            step_seq: 3,
            state: serde_json::from_value::<Map<String, Value>>(state).unwrap(),
            runtime_routes: Vec::new(),
            ephemeral_context: Map::from_iter([(
                AUTHORIZED_TURNS_KEY.to_owned(),
                json!({"nodes": [
                    {"id":"user-turn", "metadata":{"conversation_id":"conversation-1", "entity_type":"conversation_turn"}},
                    {"id":"assistant-turn", "metadata":{"conversation_id":"conversation-1", "entity_type":"assistant_turn"}}
                ], "edges": []}),
            )]),
        }
    }

    #[test]
    fn edge_identity_list_format_matches_python_repr_for_ids() {
        assert_eq!(
            python_list_repr(&["user-turn", "assistant-turn"]),
            "['user-turn', 'assistant-turn']"
        );
        assert_eq!(python_list_repr(&["has'quote"]), "[\"has'quote\"]");
    }

    #[test]
    fn links_assistant_turn_as_atomic_conversation_graph_effect() {
        let request = request(json!({
            "conversation_id": "conversation-1",
            "turn_node_id": "user-turn",
            "user_id": "alice",
            "_rt_run_owner_id": "alice",
            "_rt_graph_scope": {"storage_namespace": "ns-a"},
            "turn_index": 4,
            "answer": {"response_node_id": "assistant-turn"},
            "prev_turn_meta_summary": {
                "prev_node_char_distance_from_last_summary": 20,
                "prev_node_distance_from_last_summary": 3,
                "tail_turn_index": 4
            }
        }));
        let (output, batch) = execute_link_assistant_turn(&request).unwrap();
        let batch = batch.unwrap();
        assert!(batch.atomic);
        assert_eq!(batch.default_acl.truth_graph, "conversation");
        assert_eq!(batch.mutations.len(), 1);
        assert_eq!(batch.mutations[0].entity_kind, "edge");
        assert_eq!(batch.mutations[0].payload["relation"], "next_turn");
        assert_eq!(
            batch.mutations[0].payload["source_ids"],
            json!(["user-turn"])
        );
        assert_eq!(
            batch.mutations[0].payload["target_ids"],
            json!(["assistant-turn"])
        );
        assert_eq!(
            batch.mutations[0].record.id, "be2eb030-c937-5648-afd9-a7eef6c8d33c",
            "native identity must match Python's ConversationEdge stable ID"
        );
        assert_eq!(batch.index_jobs.len(), 2);
        assert_eq!(
            output.result.unwrap()["final_state"]["assistant_turn_edge_id"],
            batch.mutations[0].record.id
        );
    }

    #[test]
    fn rejects_link_state_from_a_different_owner_or_turn() {
        let wrong_owner = request(json!({
            "conversation_id": "conversation-1", "turn_node_id": "user-turn",
            "user_id": "mallory", "_rt_run_owner_id": "alice",
            "answer": {"response_node_id": "assistant-turn"}
        }));
        assert!(execute_link_assistant_turn(&wrong_owner).is_err());

        let wrong_turn = request(json!({
            "conversation_id": "conversation-1", "turn_node_id": "other-turn",
            "user_id": "alice", "_rt_run_owner_id": "alice",
            "answer": {"response_node_id": "assistant-turn"}
        }));
        assert!(execute_link_assistant_turn(&wrong_turn).is_err());
    }
}
