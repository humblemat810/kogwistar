use std::collections::BTreeMap;

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

#[derive(Clone)]
struct ContextItem {
    kind: &'static str,
    role: String,
    text: String,
    node_id: Option<String>,
    edge_ids: Vec<String>,
    pointer_ids: Vec<String>,
    priority: i64,
    pinned: bool,
    max_tokens: Option<usize>,
    source: &'static str,
    turn_index: i64,
}

fn record_document(record: &Value) -> Option<Value> {
    record
        .get("document")?
        .as_str()
        .and_then(|raw| serde_json::from_str::<Value>(raw).ok())
}

fn record_field<'a>(record: &'a Value, document: &'a Value, key: &str) -> Option<&'a Value> {
    document
        .get(key)
        .or_else(|| {
            document
                .get("metadata")
                .and_then(|metadata| metadata.get(key))
        })
        .or_else(|| {
            record
                .get("metadata")
                .and_then(|metadata| metadata.get(key))
        })
}

fn record_is_active(record: &Value, document: &Value) -> bool {
    !matches!(
        record_field(record, document, "lifecycle_status").and_then(Value::as_str),
        Some("tombstoned" | "deleted")
    )
}

fn id_list(value: Option<&Value>) -> Vec<String> {
    let values = match value {
        Some(Value::Array(values)) => Some(values.clone()),
        Some(Value::String(encoded)) => serde_json::from_str::<Vec<Value>>(encoded).ok(),
        _ => None,
    };
    values
        .unwrap_or_default()
        .into_iter()
        .filter_map(|value| value.as_str().map(str::to_owned))
        .collect()
}

fn approximate_tokens(text: &str) -> usize {
    (text.chars().count() / 4).max(1)
}

fn conversation_context_projection(
    view: &Map<String, Value>,
    conversation_id: &str,
    admitted_turn_id: &str,
) -> Result<Value, RuntimeNativeExecutionError> {
    let node_records = view
        .get("context_nodes")
        .and_then(Value::as_array)
        .cloned()
        .or_else(|| view.get("turn").cloned().map(|turn| vec![turn]))
        .unwrap_or_default();
    let mut nodes = BTreeMap::<String, (Value, Value)>::new();
    for record in node_records {
        let Some(id) = record.get("id").and_then(Value::as_str) else {
            continue;
        };
        let Some(document) = record_document(&record) else {
            continue;
        };
        if record_field(&record, &document, "conversation_id").and_then(Value::as_str)
            != Some(conversation_id)
            || !record_is_active(&record, &document)
        {
            continue;
        }
        nodes.insert(id.to_owned(), (record, document));
    }

    let mut turns = nodes
        .iter()
        .filter_map(|(id, (record, document))| {
            if record_field(record, document, "entity_type").and_then(Value::as_str)
                != Some("conversation_turn")
            {
                return None;
            }
            let role = record_field(record, document, "role")
                .and_then(Value::as_str)
                .filter(|role| matches!(*role, "user" | "assistant" | "tool"))?;
            let text = document.get("summary")?.as_str()?.to_owned();
            let turn_index = record_field(record, document, "turn_index")
                .and_then(Value::as_i64)
                .unwrap_or(-1);
            Some((turn_index, id.clone(), role.to_owned(), text))
        })
        .collect::<Vec<_>>();
    turns.sort_by(|left, right| (left.0, &left.1).cmp(&(right.0, &right.1)));
    if turns.iter().all(|(_, id, _, _)| id != admitted_turn_id) {
        return Err(invalid(
            AUTHORIZED_VIEW_KEY,
            "authorized conversation context omitted the admitted user turn",
        ));
    }
    let tail_turns = turns.into_iter().rev().take(8).collect::<Vec<_>>();
    let tail_turn_ids = tail_turns
        .iter()
        .map(|(_, id, _, _)| id.clone())
        .collect::<std::collections::BTreeSet<_>>();

    let edge_records = view
        .get("context_edges")
        .and_then(Value::as_array)
        .into_iter()
        .flatten();
    let mut memory_context_ids = std::collections::BTreeSet::new();
    let mut pointer_ids = std::collections::BTreeSet::new();
    let mut memory_edge_ids = std::collections::BTreeSet::new();
    let mut pointer_edge_ids = std::collections::BTreeSet::new();
    for record in edge_records {
        let Some(document) = record_document(record) else {
            continue;
        };
        if !record_is_active(record, &document) {
            continue;
        }
        let relation = record_field(record, &document, "relation")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let sources = id_list(record_field(record, &document, "source_ids"));
        if !sources.first().is_some_and(|id| tail_turn_ids.contains(id)) {
            continue;
        }
        let targets = id_list(record_field(record, &document, "target_ids"));
        let Some(edge_id) = record.get("id").and_then(Value::as_str) else {
            continue;
        };
        match relation {
            "has_memory_context" => {
                memory_context_ids.extend(targets);
                memory_edge_ids.insert(edge_id.to_owned());
            }
            "references" => {
                pointer_ids.extend(targets);
                pointer_edge_ids.insert(edge_id.to_owned());
            }
            _ => {}
        }
    }

    let mut items = vec![ContextItem {
        kind: "system_prompt",
        role: "system".to_owned(),
        text: SYSTEM_PROMPT.to_owned(),
        node_id: None,
        edge_ids: Vec::new(),
        pointer_ids: Vec::new(),
        priority: 0,
        pinned: true,
        max_tokens: Some(900),
        source: "system",
        turn_index: -1,
    }];
    let summary = nodes
        .iter()
        .filter(|(_, (_, document))| {
            document.get("type").and_then(Value::as_str) == Some("memory_summary")
        })
        .max_by_key(|(_, (_, document))| {
            document
                .get("turn_index")
                .and_then(Value::as_i64)
                .unwrap_or(-1)
        });
    if let Some((id, (_, document))) = summary {
        items.push(ContextItem {
            kind: "head_summary",
            role: "system".to_owned(),
            text: document
                .get("summary")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            node_id: Some(id.clone()),
            edge_ids: Vec::new(),
            pointer_ids: Vec::new(),
            priority: 20,
            pinned: true,
            max_tokens: Some(800),
            source: "summary",
            turn_index: -1,
        });
    }
    for id in memory_context_ids {
        let Some((_, document)) = nodes.get(&id) else {
            continue;
        };
        if record_field(&nodes[&id].0, document, "entity_type").and_then(Value::as_str)
            != Some("memory_context")
        {
            continue;
        }
        items.push(ContextItem {
            kind: "memory_context",
            role: "system".to_owned(),
            text: document
                .get("summary")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            node_id: Some(id),
            edge_ids: memory_edge_ids.iter().cloned().collect(),
            pointer_ids: Vec::new(),
            priority: 30,
            pinned: true,
            max_tokens: Some(600),
            source: "memory_pinned",
            turn_index: -1,
        });
    }
    for id in pointer_ids {
        let Some((_, document)) = nodes.get(&id) else {
            continue;
        };
        if record_field(&nodes[&id].0, document, "entity_type").and_then(Value::as_str)
            != Some("knowledge_reference")
        {
            continue;
        }
        let properties = document.get("properties").and_then(Value::as_object);
        let refers_to = properties
            .and_then(|properties| properties.get("refers_to_id"))
            .and_then(Value::as_str);
        let label = document
            .get("label")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let summary = document
            .get("summary")
            .and_then(Value::as_str)
            .unwrap_or_default();
        items.push(ContextItem {
            kind: "pinned_kg_ref",
            role: "system".to_owned(),
            text: format!("{label}\n{summary}").trim().to_owned(),
            node_id: Some(id),
            edge_ids: pointer_edge_ids.iter().cloned().collect(),
            pointer_ids: refers_to.map(str::to_owned).into_iter().collect(),
            priority: 40,
            pinned: true,
            max_tokens: Some(500),
            source: "kg_ref",
            turn_index: -1,
        });
    }
    let ordered_tail = tail_turns.into_iter().rev().collect::<Vec<_>>();
    let tail_count = ordered_tail.len();
    for (position, (turn_index, id, role, text)) in ordered_tail.into_iter().enumerate() {
        items.push(ContextItem {
            kind: "tail_turn",
            role,
            text,
            node_id: Some(id),
            edge_ids: Vec::new(),
            pointer_ids: Vec::new(),
            priority: 50 + (tail_count - position - 1) as i64,
            pinned: false,
            max_tokens: None,
            source: "history_turn",
            turn_index,
        });
    }

    let mut pinned = items
        .iter()
        .filter(|item| item.pinned)
        .cloned()
        .collect::<Vec<_>>();
    pinned.sort_by(|left, right| {
        (
            left.priority,
            left.kind,
            left.node_id.as_deref().unwrap_or(""),
        )
            .cmp(&(
                right.priority,
                right.kind,
                right.node_id.as_deref().unwrap_or(""),
            ))
    });
    let mut tail = items
        .into_iter()
        .filter(|item| !item.pinned)
        .collect::<Vec<_>>();
    tail.sort_by_key(|item| (item.priority, item.turn_index, item.node_id.clone()));
    let mut used = 0usize;
    let budget = 6000usize;
    let mut kept = Vec::new();
    let mut dropped = Vec::new();
    for item in pinned.into_iter().chain(tail.iter().cloned()) {
        let original_cost = approximate_tokens(&item.text);
        if used + original_cost <= budget {
            used += original_cost;
            kept.push(item);
            continue;
        }
        if let Some(max_tokens) = item.max_tokens.filter(|limit| *limit < original_cost) {
            let mut compressed = item.clone();
            compressed.text = item
                .text
                .chars()
                .take(max_tokens.saturating_mul(4))
                .collect();
            let compressed_cost = approximate_tokens(&compressed.text);
            if used + compressed_cost <= budget {
                used += compressed_cost;
                kept.push(compressed);
                dropped.push(json!({"kind":item.kind,"node_id":item.node_id,"reason":"compressed","token_cost":original_cost}));
                continue;
            }
        }
        dropped.push(json!({"kind":item.kind,"node_id":item.node_id,"reason":"over_budget","token_cost":original_cost}));
    }
    let mut turns_kept = kept
        .iter()
        .filter(|item| item.kind == "tail_turn")
        .cloned()
        .collect::<Vec<_>>();
    turns_kept.sort_by_key(|item| (item.turn_index, item.node_id.clone()));
    kept.retain(|item| item.kind != "tail_turn");
    kept.extend(turns_kept);

    let mut system_parts = Vec::new();
    let mut summaries = Vec::new();
    let mut memories = Vec::new();
    let mut references = Vec::new();
    let mut messages = Vec::new();
    for item in &kept {
        match item.kind {
            "system_prompt" if !item.text.trim().is_empty() => {
                system_parts.push(item.text.trim().to_owned())
            }
            "head_summary" if !item.text.trim().is_empty() => summaries.push(item.text.trim()),
            "memory_context" if !item.text.trim().is_empty() => memories.push(item.text.trim()),
            "pinned_kg_ref" if !item.text.trim().is_empty() => references.push(item.text.trim()),
            "tail_turn" => messages.push(json!({
                "role": item.role,
                "content": item.text,
                "node_id": item.node_id,
                "source": item.source
            })),
            _ => {}
        }
    }
    let mut prelude = Vec::new();
    if !system_parts.is_empty() {
        prelude.push(system_parts.join("\n\n"));
    }
    if !summaries.is_empty() {
        prelude.push(format!("Conversation summary:\n{}", summaries.join("\n\n")));
    }
    if !memories.is_empty() {
        prelude.push(format!("Memory context:\n{}", memories.join("\n\n")));
    }
    if !references.is_empty() {
        let bullets = references
            .iter()
            .map(|reference| format!("- {}", reference.replace('\n', " | ")))
            .collect::<Vec<_>>()
            .join("\n");
        prelude.push(format!("Pinned knowledge references:\n{bullets}"));
    }
    if !prelude.is_empty() {
        messages.insert(
            0,
            json!({"role":"system","content":prelude.join("\n\n").trim(),"node_id":null,"source":"system"}),
        );
    }
    if messages.is_empty() {
        messages.push(json!({"role":"system","content":"","node_id":null,"source":"system"}));
    }
    let node_ids = kept
        .iter()
        .filter_map(|item| item.node_id.clone())
        .collect::<std::collections::BTreeSet<_>>();
    let edge_ids = kept
        .iter()
        .flat_map(|item| item.edge_ids.iter().cloned())
        .collect::<std::collections::BTreeSet<_>>();
    let refs = kept
        .iter()
        .flat_map(|item| item.pointer_ids.iter().cloned())
        .collect::<std::collections::BTreeSet<_>>();
    let included_node_ids = node_ids.into_iter().collect::<Vec<_>>();
    Ok(json!({
        "messages": messages,
        "included_node_ids": included_node_ids,
        "included_edge_ids": edge_ids,
        "included_pointer_ids": refs,
        "tokens_used": used,
        "token_budget": budget,
        "dropped": dropped
    }))
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

    let context_projection =
        conversation_context_projection(view, &request.conversation_id, turn_id)?;

    let mut update = Map::new();
    update.insert("system_prompt".to_owned(), json!(SYSTEM_PROMPT));
    update.insert("question".to_owned(), json!(question));
    update.insert(
        "conversation_context_messages".to_owned(),
        context_projection["messages"].clone(),
    );
    update.insert(
        "conversation_context_node_ids".to_owned(),
        context_projection["included_node_ids"].clone(),
    );
    update.insert(
        "conversation_context_edge_ids".to_owned(),
        context_projection["included_edge_ids"].clone(),
    );
    update.insert(
        "conversation_context_pointer_ids".to_owned(),
        context_projection["included_pointer_ids"].clone(),
    );
    update.insert(
        "conversation_context_dropped".to_owned(),
        context_projection["dropped"].clone(),
    );
    update.insert(
        "context_budget_tokens".to_owned(),
        context_projection["token_budget"].clone(),
    );
    update.insert(
        "context_tokens_used".to_owned(),
        context_projection["tokens_used"].clone(),
    );
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
        let mut context_nodes = turns.clone();
        context_nodes.extend([
            json!({
                "id": "memory-summary",
                "document": json!({
                    "type": "memory_summary",
                    "summary": "Conversation summary text",
                    "turn_index": 1,
                    "metadata": {
                        "entity_type": "memory_summary",
                        "conversation_id": "conv-1",
                        "lifecycle_status": "active"
                    }
                }).to_string(),
                "metadata": {}
            }),
            json!({
                "id": "memory-context",
                "document": json!({
                    "summary": "Pinned memory detail",
                    "metadata": {
                        "entity_type": "memory_context",
                        "conversation_id": "conv-1",
                        "lifecycle_status": "active"
                    }
                }).to_string(),
                "metadata": {}
            }),
            json!({
                "id": "knowledge-pointer",
                "document": json!({
                    "label": "Architecture reference",
                    "summary": "A pinned graph reference",
                    "properties": {"refers_to_id": "kg-node-7"},
                    "metadata": {
                        "entity_type": "knowledge_reference",
                        "conversation_id": "conv-1",
                        "lifecycle_status": "active"
                    }
                }).to_string(),
                "metadata": {}
            }),
        ]);
        let context_edges = json!([
            {
                "id": "memory-edge",
                "document": json!({"relation":"has_memory_context","source_ids":["turn-9"],"target_ids":["memory-context"]}).to_string(),
                "metadata": {}
            },
            {
                "id": "reference-edge",
                "document": json!({"relation":"references","source_ids":["turn-9"],"target_ids":["knowledge-pointer"]}).to_string(),
                "metadata": {}
            }
        ]);
        let mut req = request(turns[9].clone());
        req.turn_node_id = Some("turn-9".to_owned());
        req.state["turn_node_id"] = json!("turn-9");
        req.ephemeral_context.insert(
            AUTHORIZED_VIEW_KEY.to_owned(),
            json!({
                "turn": turns[9],
                "context_nodes": context_nodes,
                "context_edges": context_edges
            }),
        );

        let result = execute_get_view_and_question(&req).unwrap();
        let final_state = &result.result.unwrap()["final_state"];
        let messages = final_state["conversation_context_messages"]
            .as_array()
            .unwrap();
        assert_eq!(messages.len(), 9);
        assert_eq!(messages[0]["role"], "system");
        assert!(
            messages[0]["content"]
                .as_str()
                .unwrap()
                .contains("Conversation summary:\nConversation summary text")
        );
        assert!(
            messages[0]["content"]
                .as_str()
                .unwrap()
                .contains("Memory context:\nPinned memory detail")
        );
        assert!(messages[0]["content"].as_str().unwrap().contains(
            "Pinned knowledge references:\n- Architecture reference | A pinned graph reference"
        ));
        assert_eq!(messages[1]["node_id"], "turn-2");
        assert_eq!(messages[1]["content"], "message 2");
        assert_eq!(messages[8]["node_id"], "turn-9");
        assert_eq!(messages[8]["content"], "message 9");
        for id in [
            "memory-summary",
            "memory-context",
            "knowledge-pointer",
            "turn-2",
            "turn-3",
            "turn-4",
            "turn-5",
            "turn-6",
            "turn-7",
            "turn-8",
            "turn-9",
        ] {
            assert!(
                final_state["conversation_context_node_ids"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|value| value == id)
            );
        }
        assert_eq!(
            final_state["conversation_context_edge_ids"],
            json!(["memory-edge", "reference-edge"])
        );
        assert_eq!(
            final_state["conversation_context_pointer_ids"],
            json!(["kg-node-7"])
        );
        assert!(final_state["context_tokens_used"].as_u64().unwrap() <= 6000);
    }
}
