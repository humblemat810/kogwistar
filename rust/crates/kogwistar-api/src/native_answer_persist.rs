use kogwistar_contracts::{GraphDefaultAclIntentV1, GraphMutationBatchV2};
use kogwistar_runtime::{
    NativeOperationOutput, NativeOperationRegistrationError, NativeOperationRegistry,
    RecordedStateUpdate, RuntimeNativeExecutionError, RuntimeNativeStepRequest,
};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

pub(crate) fn registry() -> Result<NativeOperationRegistry, NativeOperationRegistrationError> {
    let mut registry = crate::native_memory_pin::registry()?;
    crate::native_answer_prepare::register(&mut registry)?;
    crate::native_answer_state::register(&mut registry)?;
    crate::native_answer_materialize::register(&mut registry)?;
    registry.register_graph_v2("aa_persist_response", "v1", execute_persist_response)?;
    Ok(registry)
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

fn python_json_string(value: &str) -> String {
    let mut out = String::from("\"");
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if ch <= '\u{1f}' => {
                out.push_str(&format!("\\u{:04x}", ch as u32));
            }
            ch if ch.is_ascii() => out.push(ch),
            ch => {
                let mut utf16 = [0; 2];
                for unit in ch.encode_utf16(&mut utf16).iter() {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
        }
    }
    out.push('"');
    out
}

pub(crate) fn stable_conversation_id(prefix: &str, fields: &[(&str, &str)]) -> String {
    let mut fields = fields.to_vec();
    fields.sort_by_key(|(key, _)| *key);
    let encoded = fields
        .iter()
        .map(|(key, value)| format!("{}:{}", python_json_string(key), python_json_string(value)))
        .collect::<Vec<_>>()
        .join(",");
    let digest = Sha256::digest(format!("{{{encoded}}}").as_bytes());
    format!("{prefix}:{}", base32_no_pad(&digest[..12]))
}

fn base32_no_pad(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 32] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";
    let bit_len = bytes.len() * 8;
    let mut output = String::with_capacity(bit_len.div_ceil(5));
    for chunk_start in (0..bit_len).step_by(5) {
        let mut value = 0_u8;
        for offset in 0..5 {
            value <<= 1;
            let bit = chunk_start + offset;
            if bit < bit_len {
                value |= (bytes[bit / 8] >> (7 - (bit % 8))) & 1;
            }
        }
        output.push(ALPHABET[value as usize] as char);
    }
    output.make_ascii_lowercase();
    output
}

fn checked_add(
    value: i64,
    delta: i64,
    field: &'static str,
) -> Result<i64, RuntimeNativeExecutionError> {
    value
        .checked_add(delta)
        .ok_or_else(|| invalid(field, "integer value exceeds supported range"))
}

fn object_or_empty<'a>(state: &'a Map<String, Value>, field: &str) -> Map<String, Value> {
    state
        .get(field)
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default()
}

fn string_list(state: &Map<String, Value>, field: &str) -> Vec<Value> {
    state
        .get(field)
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
}

pub(crate) fn span_payload() -> Value {
    json!({
        "collection_page_url": "conversation/_conv:_dummy",
        "document_page_url": "conversation/_conv:_dummy",
        "doc_id": "_conv:_dummy",
        "insertion_method": "system",
        "page_number": 1,
        "start_char": 0,
        "end_char": 1,
        "excerpt": "",
        "context_before": "",
        "context_after": "",
        "source_cluster_id": null,
        "verification": {
            "method": "system",
            "is_verified": true,
            "score": 1.0,
            "notes": ""
        }
    })
}

pub(crate) fn model_document_fields(payload: &Value, excluded: &[&str]) -> Map<String, Value> {
    payload
        .as_object()
        .into_iter()
        .flatten()
        .filter(|(key, _)| !excluded.contains(&key.as_str()))
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect()
}

fn execute_persist_response(
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
    let run_node_id = required_string(state, "run_node_id")?;
    let scope = state
        .get("_rt_graph_scope")
        .filter(|value| value.is_object())
        .ok_or_else(|| invalid("_rt_graph_scope", "trusted graph scope is missing"))?;
    let namespace = scope
        .get("storage_namespace")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| invalid("_rt_graph_scope", "trusted storage namespace is missing"))?;

    let answer = object_or_empty(state, "answer");
    let assistant_text = answer
        .get("text")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let used_node_ids = string_list(state, "used_node_ids");
    let projected_pointer_ids = string_list(state, "projected_pointer_ids");
    let evaluation = state
        .get("evaluation")
        .filter(|value| value.is_object())
        .cloned()
        .unwrap_or_else(|| json!({}));
    let previous = state
        .get("prev_turn_meta_summary")
        .and_then(Value::as_object);
    let previous_chars = previous
        .and_then(|value| value.get("prev_node_char_distance_from_last_summary"))
        .and_then(Value::as_i64)
        .unwrap_or_default();
    let previous_distance = previous
        .and_then(|value| value.get("prev_node_distance_from_last_summary"))
        .and_then(Value::as_i64)
        .unwrap_or_default();
    let previous_tail = previous
        .and_then(|value| value.get("tail_turn_index"))
        .and_then(Value::as_i64)
        .unwrap_or_default();
    let node_turn_index = checked_add(previous_tail, 1, "prev_turn_meta_summary")?;
    let final_tail = checked_add(node_turn_index, 1, "prev_turn_meta_summary")?;
    let final_chars = checked_add(
        previous_chars,
        i64::try_from(assistant_text.chars().count())
            .map_err(|_| invalid("answer.text", "text length exceeds supported range"))?,
        "prev_turn_meta_summary",
    )?;
    let final_distance = checked_add(previous_distance, 1, "prev_turn_meta_summary")?;

    let response_node_id = stable_conversation_id(
        "ptr",
        &[
            ("scope", &format!("conv:{conversation_id}")),
            ("pointer_kind", "turn"),
            ("target_kind", "assistant"),
            ("target_id", run_node_id),
        ],
    );
    let response_edge_id = stable_conversation_id(
        "e",
        &[
            ("scope", &format!("conv:{conversation_id}")),
            ("rel", "generated"),
            ("src", run_node_id),
            ("dst", &response_node_id),
        ],
    );
    let span = span_payload();
    let node_metadata = json!({
        "level_from_root": 0,
        "entity_type": "assistant_turn",
        "in_conversation_chain": true,
        "in_ui_chain": true,
        "lifecycle_status": "active",
        "redirect_to_id": null,
        "role": "assistant",
        "turn_index": node_turn_index,
        "conversation_id": conversation_id
    });
    let node_payload = json!({
        "id": response_node_id,
        "label": "Assistant turn",
        "type": "entity",
        "summary": assistant_text,
        "domain_id": null,
        "canonical_entity_id": null,
        "properties": {"content": assistant_text, "entity_type": "assistant_turn"},
        "mentions": [{"spans": [span], "multimodal_spans": []}],
        "embedding": null,
        "doc_id": null,
        "metadata": node_metadata,
        "level_from_root": 0,
        "role": "assistant",
        "turn_index": node_turn_index,
        "conversation_id": conversation_id,
        "user_id": null
    });
    let edge_metadata = json!({
        "char_distance_from_last_summary": final_chars,
        "turn_distance_from_last_summary": final_distance,
        "lifecycle_status": "active",
        "redirect_to_id": null
    });
    let edge_payload = json!({
        "id": response_edge_id,
        "source_ids": [run_node_id],
        "target_ids": [response_node_id],
        "relation": "generated",
        "label": "generated",
        "type": "relationship",
        "summary": "Agent run generated assistant response",
        "doc_id": format!("conv:{conversation_id}"),
        "mentions": [{"spans": [span], "multimodal_spans": []}],
        "domain_id": null,
        "canonical_entity_id": null,
        "properties": {"entity_type": "conversation_edge", "used_node_ids": used_node_ids},
        "embedding": null,
        "metadata": edge_metadata,
        "source_edge_ids": [],
        "target_edge_ids": []
    });
    let graph_scope = json!({
        "storage_namespace": namespace,
        "workspace": scope.get("workspace").cloned().unwrap_or(Value::Null),
        "graph_space": scope.get("graph_space").cloned().unwrap_or(Value::Null)
    });
    let node_document = model_document_fields(&node_payload, &["embedding", "metadata"]);
    let mut node_meta = node_metadata.as_object().cloned().unwrap_or_default();
    node_meta.insert("_class_name".to_owned(), json!("ConversationNode"));
    let (node_mutation, mut jobs) = crate::native_memory_pin::graph_mutation(
        request,
        &graph_scope,
        &response_node_id,
        format!(
            "native-aa-persist|{}|{}|node|{}",
            request.run_id, request.node_id, response_node_id
        ),
        "node",
        "conversation_node",
        node_payload.clone(),
        node_document,
        node_meta,
    )?;
    let edge_document = model_document_fields(&edge_payload, &[]);
    let mut edge_meta = edge_metadata.as_object().cloned().unwrap_or_default();
    edge_meta.insert("_class_name".to_owned(), json!("ConversationEdge"));
    let (edge_mutation, edge_jobs) = crate::native_memory_pin::graph_mutation(
        request,
        &graph_scope,
        &response_edge_id,
        format!(
            "native-aa-persist|{}|{}|edge|{}",
            request.run_id, request.node_id, response_edge_id
        ),
        "edge",
        "conversation_edge",
        edge_payload,
        edge_document,
        edge_meta,
    )?;
    jobs.extend(edge_jobs);

    let mut summary = Map::new();
    summary.insert(
        "prev_node_char_distance_from_last_summary".to_owned(),
        json!(final_chars),
    );
    summary.insert(
        "prev_node_distance_from_last_summary".to_owned(),
        json!(final_distance),
    );
    summary.insert("tail_turn_index".to_owned(), json!(final_tail));
    let mut result_payload = Map::new();
    result_payload.insert("run_node_id".to_owned(), json!(run_node_id));
    result_payload.insert("assistant_turn_node_id".to_owned(), json!(response_node_id));
    result_payload.insert("assistant_text".to_owned(), json!(assistant_text));
    result_payload.insert("used_node_ids".to_owned(), Value::Array(used_node_ids));
    result_payload.insert(
        "projected_pointer_ids".to_owned(),
        Value::Array(projected_pointer_ids),
    );
    result_payload.insert("claim_citations".to_owned(), Value::Object(answer));
    result_payload.insert("evaluation".to_owned(), evaluation);
    let mut turn_node_dump = node_payload.clone();
    if let Some(spans) = turn_node_dump
        .get_mut("mentions")
        .and_then(Value::as_array_mut)
        .and_then(|mentions| mentions.first_mut())
        .and_then(|grounding| grounding.get_mut("spans"))
        .and_then(Value::as_array_mut)
    {
        for span in spans {
            if let Some(span) = span.as_object_mut() {
                span.remove("insertion_method");
                span.remove("verification");
            }
        }
    }
    result_payload.insert("turn_node_dump".to_owned(), turn_node_dump);
    let mut update = Map::new();
    update.insert(
        "agentic_answering_result".to_owned(),
        Value::Object(result_payload),
    );
    update.insert("prev_turn_meta_summary".to_owned(), Value::Object(summary));
    let mut final_state = state.clone();
    final_state.extend(update.clone());
    let output = NativeOperationOutput {
        state_update: vec![RecordedStateUpdate::Pair(("u".to_owned(), update))],
        result: Some(json!({
            "workflow_status": "succeeded",
            "final_state": final_state,
            "conversation_node_id": response_node_id
        })),
    };
    let batch = GraphMutationBatchV2 {
        contract_version: 2,
        atomic: true,
        mutations: vec![node_mutation, edge_mutation],
        index_jobs: serde_json::from_value(Value::Array(jobs))
            .map_err(|_| invalid("answer", "index job intent is invalid"))?,
        default_acl: GraphDefaultAclIntentV1 {
            truth_graph: "conversation".to_owned(),
        },
    };
    batch
        .validate()
        .map_err(|_| invalid("answer", "atomic graph batch is invalid"))?;
    Ok((output, Some(batch)))
}

#[cfg(test)]
mod tests {
    use super::{registry, stable_conversation_id};
    use kogwistar_runtime::{
        RUST_NATIVE_EXECUTOR_TARGET, RuntimeNativeStepRequest, execute_native_step_with_registry,
    };
    use serde_json::{Map, Value, json};

    fn request(state: Value) -> RuntimeNativeStepRequest {
        RuntimeNativeStepRequest {
            executor_target: RUST_NATIVE_EXECUTOR_TARGET.to_owned(),
            operation: "aa_persist_response".to_owned(),
            op_version: "v1".to_owned(),
            run_id: "run-1".to_owned(),
            workflow_id: "agentic_answering.v2".to_owned(),
            node_id: "persist-response".to_owned(),
            conversation_id: "conv-1".to_owned(),
            turn_node_id: None,
            token_id: "token-1".to_owned(),
            parent_token_id: None,
            step_seq: 5,
            state: serde_json::from_value::<Map<String, Value>>(state).unwrap(),
            runtime_routes: Vec::new(),
            ephemeral_context: Map::new(),
        }
    }

    fn state() -> Value {
        json!({
            "conversation_id": "conv-1",
            "run_node_id": "run-1",
            "_rt_graph_scope": {"storage_namespace": "ns-a", "workspace": "ws-a", "graph_space": "conversation"},
            "answer": {"text": "Hello 🌍", "claim_citations": [{"claim": "c1"}]},
            "used_node_ids": ["knowledge-1"],
            "projected_pointer_ids": ["pointer-1"],
            "evaluation": {"is_sufficient": true},
            "prev_turn_meta_summary": {
                "prev_node_char_distance_from_last_summary": 9,
                "prev_node_distance_from_last_summary": 3,
                "tail_turn_index": 4
            }
        })
    }

    #[test]
    fn python_deterministic_assistant_ids_are_reproduced() {
        assert_eq!(
            stable_conversation_id(
                "ptr",
                &[
                    ("scope", "conv:c1"),
                    ("pointer_kind", "turn"),
                    ("target_kind", "assistant"),
                    ("target_id", "run-1")
                ]
            ),
            "ptr:53wuwyr4c325wx4kjm5q"
        );
        assert_eq!(
            stable_conversation_id(
                "e",
                &[
                    ("scope", "conv:c1"),
                    ("rel", "generated"),
                    ("src", "run-1"),
                    ("dst", "ptr:test")
                ]
            ),
            "e:bzhnpddezcglhu5mmmqa"
        );
        assert_eq!(
            stable_conversation_id(
                "ptr",
                &[
                    ("scope", "conv:雪"),
                    ("pointer_kind", "turn"),
                    ("target_kind", "assistant"),
                    ("target_id", "run-🌱")
                ]
            ),
            "ptr:h4tikop7xaxo45vyejsq"
        );
        assert_eq!(
            stable_conversation_id(
                "e",
                &[
                    ("scope", "conv:雪"),
                    ("rel", "generated"),
                    ("src", "run-🌱"),
                    ("dst", "ptr:h4tikop7xaxo45vyejsq")
                ]
            ),
            "e:yuovcwzvqbo6ebgbnuzq"
        );
    }

    #[test]
    fn response_persistence_returns_python_compatible_state_and_atomic_graph_intent() {
        let registry = registry().unwrap();
        let effect = execute_native_step_with_registry(&request(state()), &registry).unwrap();
        let batch = effect.graph_mutation_batch_v2.unwrap();
        batch.validate().unwrap();
        assert_eq!(batch.default_acl.truth_graph, "conversation");
        assert_eq!(batch.mutations.len(), 2);
        assert_eq!(batch.mutations[0].entity_kind, "node");
        assert_eq!(batch.mutations[1].entity_kind, "edge");
        assert_eq!(batch.mutations[0].scope.namespace, "ns-a");
        assert_eq!(batch.index_jobs.len(), 6);
        assert!(
            batch
                .index_jobs
                .iter()
                .any(|job| job.index_kind == "node_embedding")
        );
        let result = effect.result.unwrap();
        assert_eq!(
            result["final_state"]["prev_turn_meta_summary"]["prev_node_char_distance_from_last_summary"],
            16
        );
        assert_eq!(
            result["final_state"]["prev_turn_meta_summary"]["tail_turn_index"],
            6
        );
        assert_eq!(
            result["final_state"]["agentic_answering_result"]["assistant_text"],
            "Hello 🌍"
        );
        assert!(result["final_state"]["agentic_answering_result"]["turn_node_dump"]["mentions"][0]["spans"][0]
            .get("verification")
            .is_none());
        assert_eq!(
            batch.mutations[0].payload["mentions"][0]["spans"][0]["verification"]["method"],
            "system"
        );
        assert_eq!(result["conversation_node_id"], batch.mutations[0].record.id);
        assert_eq!(
            batch.mutations[0]
                .model_envelope
                .as_ref()
                .unwrap()
                .model_class,
            kogwistar_contracts::GraphModelClass::ConversationNode
        );
    }

    #[test]
    fn response_persistence_rejects_mismatched_conversation_and_missing_scope() {
        let registry = registry().unwrap();
        let mut mismatched = state();
        mismatched["conversation_id"] = json!("other-conversation");
        assert!(execute_native_step_with_registry(&request(mismatched), &registry).is_err());
        let mut no_scope = state();
        no_scope.as_object_mut().unwrap().remove("_rt_graph_scope");
        assert!(execute_native_step_with_registry(&request(no_scope), &registry).is_err());
    }
}
