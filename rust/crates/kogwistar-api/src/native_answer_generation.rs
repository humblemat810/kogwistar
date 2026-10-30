use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

use kogwistar_contracts::{GraphDefaultAclIntentV1, GraphMutationBatchV2, stable_id};
use kogwistar_runtime::{
    NativeOperationOutput, NativeOperationRegistrationError, NativeOperationRegistry,
    RecordedStateUpdate, RuntimeNativeExecutionError, RuntimeNativeStepRequest,
};
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const MAX_REQUEST_BYTES: usize = 256 * 1024;
const MAX_RESPONSE_BYTES: usize = 1024 * 1024;

#[derive(Clone)]
pub(crate) struct OpenAiCompatibleChatConfig {
    endpoint: String,
    model: String,
    api_key: Option<String>,
    timeout: Duration,
}

impl OpenAiCompatibleChatConfig {
    pub(crate) fn model(&self) -> &str {
        &self.model
    }
}

pub(crate) fn register(
    registry: &mut NativeOperationRegistry,
) -> Result<(), NativeOperationRegistrationError> {
    registry.register(
        "aa_generate_answer_with_citations",
        "v1",
        execute_generate_answer,
    )?;
    registry.register(
        "aa_validate_or_repair_citations",
        "v1",
        execute_validate_or_repair_citations,
    )
}

pub(crate) fn answer_needs_citation_repair(
    request: &RuntimeNativeStepRequest,
) -> Result<bool, RuntimeNativeExecutionError> {
    let evidence_pack = request
        .state
        .get("_rt")
        .and_then(Value::as_object)
        .and_then(|runtime| runtime.get("evidence_pack"))
        .ok_or(RuntimeNativeExecutionError::InvalidState {
            field: "_rt.evidence_pack",
            reason: "materialized evidence pack is required",
        })?;
    let used_node_ids = selected_node_ids(request)?;
    let answer: AnswerWithCitations =
        serde_json::from_value(request.state.get("answer").cloned().ok_or(
            RuntimeNativeExecutionError::InvalidState {
                field: "answer",
                reason: "generated answer is required",
            },
        )?)
        .map_err(|_| RuntimeNativeExecutionError::InvalidState {
            field: "answer",
            reason: "answer does not match the structured answer contract",
        })?;
    Ok(!answer_citations_are_valid(
        &answer,
        evidence_pack,
        &used_node_ids,
    ))
}

fn execute_validate_or_repair_citations(
    request: &RuntimeNativeStepRequest,
) -> Result<NativeOperationOutput, RuntimeNativeExecutionError> {
    let needs_repair = answer_needs_citation_repair(request)?;
    let original: AnswerWithCitations =
        serde_json::from_value(request.state.get("answer").cloned().ok_or(
            RuntimeNativeExecutionError::InvalidState {
                field: "answer",
                reason: "generated answer is required",
            },
        )?)
        .map_err(|_| RuntimeNativeExecutionError::InvalidState {
            field: "answer",
            reason: "answer does not match the structured answer contract",
        })?;
    let repaired = if needs_repair {
        request
            .ephemeral_context
            .get("repaired_answer")
            .cloned()
            .and_then(|value| serde_json::from_value::<AnswerWithCitations>(value).ok())
    } else {
        None
    };
    let evidence_pack = request
        .state
        .get("_rt")
        .and_then(Value::as_object)
        .and_then(|runtime| runtime.get("evidence_pack"))
        .ok_or(RuntimeNativeExecutionError::InvalidState {
            field: "_rt.evidence_pack",
            reason: "materialized evidence pack is required",
        })?;
    let selected = selected_node_ids(request)?;
    let repaired_is_valid = repaired
        .as_ref()
        .is_some_and(|answer| answer_citations_are_valid(answer, evidence_pack, &selected));
    let answer = if let Some(repaired) = repaired.filter(|_| repaired_is_valid) {
        serde_json::to_value(repaired)
    } else if needs_repair {
        let mut stripped = original;
        for claim in &mut stripped.claims {
            claim.citations.clear();
        }
        stripped.reasoning.push_str(" | citation_repair_failed");
        serde_json::to_value(stripped)
    } else {
        serde_json::to_value(original)
    }
    .map_err(|_| RuntimeNativeExecutionError::InvalidState {
        field: "answer",
        reason: "answer could not be serialized",
    })?;
    let mut runtime = request
        .state
        .get("_rt")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    if needs_repair {
        if let Some(model) = request
            .ephemeral_context
            .get("provider_model")
            .and_then(Value::as_str)
        {
            runtime.insert(
                "answer_repair_provider".to_owned(),
                json!({"protocol": "openai_compatible_chat_completions", "model": model}),
            );
        }
    }
    let mut op_log = request
        .state
        .get("op_log")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    op_log.push(json!("aa_validate_or_repair_citations"));
    let update = serde_json::Map::from_iter([
        ("answer".to_owned(), answer.clone()),
        ("_rt".to_owned(), Value::Object(runtime)),
    ]);
    let auxiliary = serde_json::Map::from_iter([(
        "op_log".to_owned(),
        json!("aa_validate_or_repair_citations"),
    )]);
    let mut final_state = request.state.clone();
    final_state.extend(update.clone());
    final_state.insert("op_log".to_owned(), Value::Array(op_log));
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

fn selected_node_ids(
    request: &RuntimeNativeStepRequest,
) -> Result<BTreeSet<String>, RuntimeNativeExecutionError> {
    request
        .state
        .get("used_node_ids")
        .and_then(Value::as_array)
        .ok_or(RuntimeNativeExecutionError::InvalidState {
            field: "used_node_ids",
            reason: "selected evidence IDs are required",
        })?
        .iter()
        .map(|id| {
            id.as_str()
                .map(str::to_owned)
                .ok_or(RuntimeNativeExecutionError::InvalidState {
                    field: "used_node_ids",
                    reason: "selected evidence IDs must be strings",
                })
        })
        .collect()
}

fn answer_citations_are_valid(
    answer: &AnswerWithCitations,
    evidence_pack: &Value,
    used_node_ids: &BTreeSet<String>,
) -> bool {
    for claim in &answer.claims {
        for citation in &claim.citations {
            if !used_node_ids.contains(&citation.source_node_id) {
                return false;
            }
            let Some(node) = evidence_pack
                .get("nodes")
                .and_then(Value::as_array)
                .and_then(|nodes| {
                    nodes.iter().find(|node| {
                        node.get("node_id").and_then(Value::as_str)
                            == Some(citation.source_node_id.as_str())
                    })
                })
            else {
                return false;
            };
            let Some(span) = node
                .get("mentions")
                .and_then(Value::as_array)
                .and_then(|mentions| mentions.get(citation.mention_index))
                .and_then(|mention| mention.get("spans"))
                .and_then(Value::as_array)
                .and_then(|spans| spans.get(citation.span_index))
            else {
                return false;
            };
            if span
                .get("excerpt")
                .and_then(Value::as_str)
                .map_or(true, |excerpt| excerpt.trim().is_empty())
            {
                return false;
            }
        }
    }
    true
}

fn execute_generate_answer(
    request: &RuntimeNativeStepRequest,
) -> Result<NativeOperationOutput, RuntimeNativeExecutionError> {
    let evidence_pack = request
        .state
        .get("_rt")
        .and_then(Value::as_object)
        .and_then(|runtime| runtime.get("evidence_pack"))
        .and_then(Value::as_object)
        .ok_or(RuntimeNativeExecutionError::InvalidState {
            field: "_rt.evidence_pack",
            reason: "materialized evidence pack is required",
        })?;
    let selected = request
        .state
        .get("used_node_ids")
        .and_then(Value::as_array)
        .ok_or(RuntimeNativeExecutionError::InvalidState {
            field: "used_node_ids",
            reason: "selected evidence IDs are required",
        })?;
    let selected = selected
        .iter()
        .map(|id| {
            id.as_str()
                .ok_or(RuntimeNativeExecutionError::InvalidState {
                    field: "used_node_ids",
                    reason: "selected evidence IDs must be strings",
                })
        })
        .collect::<Result<BTreeSet<_>, _>>()?;
    let nodes = evidence_pack.get("nodes").and_then(Value::as_array).ok_or(
        RuntimeNativeExecutionError::InvalidState {
            field: "_rt.evidence_pack.nodes",
            reason: "materialized evidence nodes are required",
        },
    )?;
    let available = nodes
        .iter()
        .filter_map(|node| node.get("node_id").and_then(Value::as_str))
        .collect::<BTreeSet<_>>();
    let answer: AnswerWithCitations = serde_json::from_value(
        request
            .ephemeral_context
            .get("provider_answer")
            .cloned()
            .ok_or(RuntimeNativeExecutionError::InvalidState {
                field: "provider_answer",
                reason: "Rust-hosted structured provider output is required",
            })?,
    )
    .map_err(|_| RuntimeNativeExecutionError::InvalidState {
        field: "provider_answer",
        reason: "structured answer does not match the answer contract",
    })?;
    for claim in &answer.claims {
        for citation in &claim.citations {
            if !selected.contains(citation.source_node_id.as_str())
                || !available.contains(citation.source_node_id.as_str())
            {
                return Err(RuntimeNativeExecutionError::InvalidState {
                    field: "provider_answer.claims.citations.source_node_id",
                    reason: "citation must refer to a selected evidence node",
                });
            }
        }
    }
    let answer =
        serde_json::to_value(&answer).map_err(|_| RuntimeNativeExecutionError::InvalidState {
            field: "provider_answer",
            reason: "structured answer could not be serialized",
        })?;
    let provider_model = request
        .ephemeral_context
        .get("provider_model")
        .and_then(Value::as_str)
        .filter(|model| !model.trim().is_empty())
        .ok_or(RuntimeNativeExecutionError::InvalidState {
            field: "provider_model",
            reason: "configured provider model metadata is required",
        })?;
    let mut op_log = request
        .state
        .get("op_log")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    op_log.push(json!("aa_generate_answer_with_citations"));
    let mut runtime = request
        .state
        .get("_rt")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    runtime.insert(
        "answer_provider".to_owned(),
        json!({"protocol": "openai_compatible_chat_completions", "model": provider_model}),
    );
    let update = serde_json::Map::from_iter([
        ("answer".to_owned(), answer.clone()),
        ("_rt".to_owned(), Value::Object(runtime)),
    ]);
    let auxiliary = serde_json::Map::from_iter([(
        "op_log".to_owned(),
        json!("aa_generate_answer_with_citations"),
    )]);
    let mut final_state = request.state.clone();
    final_state.extend(update.clone());
    final_state.insert("op_log".to_owned(), Value::Array(op_log));
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

impl OpenAiCompatibleChatConfig {
    pub(crate) fn new(
        endpoint: impl Into<String>,
        model: impl Into<String>,
        api_key: Option<String>,
    ) -> Result<Self, String> {
        let endpoint = endpoint.into();
        let model = model.into();
        let parsed = url::Url::parse(&endpoint).map_err(|_| "chat endpoint is invalid")?;
        if !matches!(parsed.scheme(), "http" | "https") || parsed.host_str().is_none() {
            return Err("chat endpoint must be an absolute HTTP(S) URL".to_owned());
        }
        if model.trim().is_empty() || model.len() > 256 {
            return Err("chat model must contain 1 to 256 characters".to_owned());
        }
        if api_key.as_ref().is_some_and(|key| key.len() > 8192) {
            return Err("chat API key exceeds the configured limit".to_owned());
        }
        Ok(Self {
            endpoint,
            model,
            api_key,
            timeout: Duration::from_secs(90),
        })
    }

    pub(crate) async fn answer(
        &self,
        system_prompt: &str,
        question: &str,
        evidence_pack: &Value,
        used_node_ids: &[String],
    ) -> Result<Value, String> {
        let (user_prompt, aliases) = answer_user_prompt(question, evidence_pack)?;
        for _ in 0..3 {
            if let Some(answer) = self.complete_answer(system_prompt, &user_prompt).await? {
                return restore_answer_aliases(answer, &aliases, used_node_ids);
            }
        }
        Err(
            "structured chat provider did not return a valid answer object after bounded retries"
                .to_owned(),
        )
    }

    pub(crate) async fn repair(
        &self,
        system_prompt: &str,
        question: &str,
        evidence_pack: &Value,
        used_node_ids: &[String],
        original_answer: &Value,
    ) -> Result<Value, String> {
        let (evidence_text, aliases) = evidence_text_and_aliases(evidence_pack)?;
        let original: AnswerWithCitations = serde_json::from_value(original_answer.clone())
            .map_err(|_| "answer to repair does not match the answer contract".to_owned())?;
        let user_prompt = format!(
            concat!(
                "Repair the answer's citations using only the supplied evidence. Preserve its ",
                "answer text and claims where possible. The reasoning field must be a brief ",
                "rationale summary, not private chain-of-thought. Every citation must use a node ",
                "alias and exact mention/span index from the evidence.\n\nQuestion:\n{}",
                "\n\nEvidence:\n{}\n\nAnswer to repair:\n{}"
            ),
            question, evidence_text, original.text
        );
        let selected = used_node_ids.iter().cloned().collect::<BTreeSet<_>>();
        for _ in 0..3 {
            if let Some(repaired) = self.complete_answer(system_prompt, &user_prompt).await? {
                let Ok(restored) = restore_answer_aliases(repaired, &aliases, used_node_ids) else {
                    continue;
                };
                let Ok(restored) = serde_json::from_value::<AnswerWithCitations>(restored) else {
                    continue;
                };
                if answer_citations_are_valid(&restored, evidence_pack, &selected) {
                    return serde_json::to_value(restored)
                        .map_err(|_| "could not encode repaired answer".to_owned());
                }
            }
        }
        let mut stripped = original;
        for claim in &mut stripped.claims {
            claim.citations.clear();
        }
        stripped.reasoning.push_str(" | citation_repair_failed");
        serde_json::to_value(stripped)
            .map_err(|_| "could not encode answer after failed citation repair".to_owned())
    }

    async fn complete_answer(
        &self,
        system_prompt: &str,
        user_prompt: &str,
    ) -> Result<Option<AnswerWithCitations>, String> {
        let request_body = json!({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "AnswerWithCitations",
                    "strict": true,
                    "schema": answer_schema()
                }
            }
        });
        let request_bytes = serde_json::to_vec(&request_body)
            .map_err(|_| "could not encode structured chat request".to_owned())?;
        if request_bytes.len() > MAX_REQUEST_BYTES {
            return Err("structured chat request exceeds the configured size limit".to_owned());
        }
        let client = reqwest::Client::builder()
            .timeout(self.timeout)
            .build()
            .map_err(|_| "could not initialize structured chat client".to_owned())?;
        let mut request = client
            .post(&self.endpoint)
            .header(CONTENT_TYPE, "application/json")
            .body(request_bytes);
        if let Some(api_key) = &self.api_key {
            request = request.header(AUTHORIZATION, format!("Bearer {api_key}"));
        }
        let response = request
            .send()
            .await
            .map_err(|_| "structured chat provider request failed".to_owned())?;
        let status = response.status();
        if !status.is_success() {
            return Err(format!("structured chat provider returned HTTP {status}"));
        }
        if response
            .content_length()
            .is_some_and(|length| length > MAX_RESPONSE_BYTES as u64)
        {
            return Err(
                "structured chat provider response exceeds the configured size limit".to_owned(),
            );
        }
        let mut response = response;
        let mut response_bytes = Vec::new();
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|_| "could not read structured chat provider response".to_owned())?
        {
            if response_bytes.len().saturating_add(chunk.len()) > MAX_RESPONSE_BYTES {
                return Err(
                    "structured chat provider response exceeds the configured size limit"
                        .to_owned(),
                );
            }
            response_bytes.extend_from_slice(&chunk);
        }
        let Ok(response) = serde_json::from_slice::<ChatCompletionResponse>(&response_bytes) else {
            return Ok(None);
        };
        let Some(content) = response
            .choices
            .into_iter()
            .next()
            .and_then(|choice| choice.message.content)
        else {
            return Ok(None);
        };
        Ok(serde_json::from_str(&content).ok())
    }
}

fn answer_prompt_messages(
    system_prompt: &str,
    question: &str,
    evidence_pack: &Value,
) -> Result<(Vec<Value>, BTreeMap<String, String>), String> {
    let (user_prompt, aliases) = answer_user_prompt(question, evidence_pack)?;
    Ok((
        vec![
            json!({"role": "system", "content": system_prompt}),
            json!({"role": "user", "content": user_prompt}),
        ],
        aliases,
    ))
}

fn answer_user_prompt(
    question: &str,
    evidence_pack: &Value,
) -> Result<(String, BTreeMap<String, String>), String> {
    let (evidence_text, aliases) = evidence_text_and_aliases(evidence_pack)?;
    let user_prompt = format!(
        concat!(
            "Answer the question using only the supplied evidence for factual claims. ",
            "The reasoning field must be a brief rationale summary, not private chain-of-thought. ",
            "Return the required JSON object. Each citation must reference an exact ",
            "node alias and mention/span index from the evidence.\n\nQuestion:\n",
            "{}\n\nEvidence:\n{}"
        ),
        question, evidence_text
    );
    Ok((user_prompt, aliases))
}

pub(crate) fn answer_context_snapshot_batch(
    request: &RuntimeNativeStepRequest,
    model_name: &str,
) -> Result<GraphMutationBatchV2, String> {
    let state = &request.state;
    let string_field = |field: &'static str| {
        state
            .get(field)
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| format!("native answer snapshot requires {field}"))
    };
    let conversation_id = string_field("conversation_id")?;
    if conversation_id != request.conversation_id {
        return Err("native answer snapshot conversation does not match admitted run".to_owned());
    }
    let run_id = state
        .get("run_id")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .unwrap_or(&request.run_id);
    let question = string_field("question")?;
    let system_prompt = string_field("system_prompt")?;
    let evidence_pack = state
        .get("_rt")
        .and_then(Value::as_object)
        .and_then(|runtime| runtime.get("evidence_pack"))
        .ok_or_else(|| "native answer snapshot requires materialized evidence".to_owned())?;
    let (messages, _) = answer_prompt_messages(system_prompt, question, evidence_pack)?;
    let used_node_ids = state
        .get("used_node_ids")
        .and_then(Value::as_array)
        .ok_or_else(|| "native answer snapshot requires selected evidence IDs".to_owned())?
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| "native answer snapshot evidence IDs must be strings".to_owned())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let conversation_context_node_ids = state
        .get("conversation_context_node_ids")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| "native answer context node IDs must be strings".to_owned())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut dependency_node_ids = Vec::new();
    let mut seen_dependency_node_ids = BTreeSet::new();
    for node_id in conversation_context_node_ids
        .iter()
        .chain(used_node_ids.iter())
    {
        if seen_dependency_node_ids.insert(node_id.as_str()) {
            dependency_node_ids.push(node_id);
        }
    }
    let run_step_seq = state
        .get("run_step_seq")
        .and_then(Value::as_u64)
        .unwrap_or_else(|| u64::try_from(request.step_seq).unwrap_or_default());
    let attempt_seq = state
        .get("attempt_seq")
        .and_then(Value::as_u64)
        .unwrap_or_default();
    let tail_turn_index = state
        .get("prev_turn_meta_summary")
        .and_then(|summary| summary.get("tail_turn_index"))
        .and_then(Value::as_u64)
        .unwrap_or_default();
    let evidence_pack_digest = state
        .get("evidence_pack_digest")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let evidence_pack_hash = evidence_pack_digest
        .get("evidence_pack_hash")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let conversation_context_messages = state
        .get("conversation_context_messages")
        .cloned()
        .unwrap_or_else(|| json!([]));
    let conversation_context_node_ids = json!(conversation_context_node_ids);
    let llm_input_payload = json!({"system_prompt": system_prompt, "question": question});
    let extra_hash_payload = json!({
        "question": question,
        "used_node_ids": used_node_ids,
        "evidence_pack_hash": evidence_pack_hash
    });
    let rendered_payload = json!({
        "messages": messages,
        "conversation_context_messages": conversation_context_messages,
        "conversation_context_node_ids": conversation_context_node_ids,
        "llm_input_payload": llm_input_payload,
        "evidence_pack_digest": evidence_pack_digest,
        "extra_hash_payload": extra_hash_payload
    });
    let rendered_context_hash = format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&rendered_payload).map_err(|_| {
            "native answer snapshot could not serialize rendered context".to_owned()
        })?)
    );
    let char_count = messages
        .iter()
        .filter_map(|message| message.get("content").and_then(Value::as_str))
        .map(|content| content.chars().count())
        .sum::<usize>();
    let budget_tokens = state
        .get("context_budget_tokens")
        .and_then(Value::as_u64)
        .unwrap_or_default();
    let snapshot_id = stable_id(
        "conversation.context_snapshot",
        &[
            conversation_id.to_owned(),
            run_id.to_owned(),
            "generate_answer_with_citations".to_owned(),
            run_step_seq.to_string(),
            attempt_seq.to_string(),
        ],
    )
    .to_string();
    let prompt_messages = serde_json::to_string(&messages)
        .map_err(|_| "native answer snapshot could not serialize prompt messages".to_owned())?;
    let conversation_context_messages = serde_json::to_string(&conversation_context_messages)
        .map_err(|_| {
            "native answer snapshot could not serialize conversation context".to_owned()
        })?;
    let conversation_context_node_ids = serde_json::to_string(&conversation_context_node_ids)
        .map_err(|_| {
            "native answer snapshot could not serialize conversation context IDs".to_owned()
        })?;
    let llm_input_payload = serde_json::to_string(&llm_input_payload)
        .map_err(|_| "native answer snapshot could not serialize provider input".to_owned())?;
    let evidence_pack_digest = serde_json::to_string(&evidence_pack_digest)
        .map_err(|_| "native answer snapshot could not serialize evidence digest".to_owned())?;
    let properties = json!({
        "entity_type": "context_snapshot",
        "prompt_messages": prompt_messages,
        "conversation_context_messages": conversation_context_messages,
        "conversation_context_node_ids": conversation_context_node_ids,
        "llm_input_payload": llm_input_payload,
        "evidence_pack_digest": evidence_pack_digest
    });
    let metadata = json!({
        "entity_type": "context_snapshot",
        "level_from_root": 0,
        "in_conversation_chain": false,
        "in_ui_chain": false,
        "run_id": run_id,
        "run_step_seq": run_step_seq,
        "attempt_seq": attempt_seq,
        "stage": "generate_answer_with_citations",
        "model_name": model_name,
        "budget_tokens": budget_tokens,
        "tail_turn_index": tail_turn_index,
        "used_node_ids": used_node_ids,
        "rendered_context_hash": rendered_context_hash,
        "cost.char_count": char_count,
        "cost.token_count": Value::Null
    });
    let scope = state
        .get("_rt_graph_scope")
        .filter(|value| value.is_object())
        .ok_or_else(|| "native answer snapshot trusted graph scope is missing".to_owned())?;
    let graph_scope = json!({
        "storage_namespace": scope.get("storage_namespace").cloned().unwrap_or(Value::Null),
        "workspace": scope.get("workspace").cloned().unwrap_or(Value::Null),
        "graph_space": scope.get("graph_space").cloned().unwrap_or(Value::Null)
    });
    let span = crate::native_answer_persist::span_payload();
    let node_payload = json!({
        "id": snapshot_id,
        "label": "Context Snapshot",
        "type": "entity",
        "summary": "",
        "domain_id": null,
        "canonical_entity_id": null,
        "properties": properties,
        "mentions": [{"spans": [span], "multimodal_spans": []}],
        "embedding": null,
        "doc_id": format!("conv:{conversation_id}"),
        "metadata": metadata,
        "level_from_root": 0,
        "role": "system",
        "turn_index": null,
        "conversation_id": conversation_id,
        "user_id": null
    });
    let node_id = snapshot_id.as_str();
    let node_document = crate::native_answer_persist::model_document_fields(
        &node_payload,
        &["embedding", "metadata"],
    );
    let mut node_metadata = metadata.as_object().cloned().unwrap_or_default();
    node_metadata.insert("_class_name".to_owned(), json!("ConversationNode"));
    let (node_mutation, _) = crate::native_memory_pin::graph_mutation(
        request,
        &graph_scope,
        node_id,
        format!("native-context-snapshot|{run_id}|{run_step_seq}|{attempt_seq}|node"),
        "node",
        "conversation_node",
        node_payload,
        node_document,
        node_metadata,
    )
    .map_err(|_| "native answer snapshot node mutation is invalid".to_owned())?;
    let mut mutations = vec![node_mutation];
    for (ordinal, target_id) in dependency_node_ids.iter().enumerate() {
        let ordinal_text = ordinal.to_string();
        let edge_id = stable_id(
            "conversation.edge",
            &[
                format!("conv:{conversation_id}"),
                "depends_on".to_owned(),
                snapshot_id.clone(),
                target_id.to_string(),
                ordinal_text.clone(),
            ],
        )
        .to_string();
        let edge_payload = json!({
            "id": edge_id,
            "source_ids": [snapshot_id],
                "target_ids": [target_id],
            "relation": "depends_on",
            "label": format!("depends_on:{snapshot_id}->{target_id}"),
            "type": "relationship",
            "summary": format!("Context snapshot {snapshot_id} depends on node {target_id}"),
            "doc_id": format!("conv:{conversation_id}"),
            "mentions": [{"spans": [crate::native_answer_persist::span_payload()], "multimodal_spans": []}],
            "domain_id": null,
            "canonical_entity_id": null,
            "properties": {"entity_type": "conversation_edge", "ordinal": ordinal},
            "embedding": null,
            "metadata": {
                "entity_type": "conversation_edge",
                "ordinal": ordinal,
                "run_id": run_id,
                "run_step_seq": run_step_seq,
                "attempt_seq": attempt_seq,
                "tail_turn_index": tail_turn_index
            },
            "source_edge_ids": [],
            "target_edge_ids": []
        });
        let edge_id = edge_id.as_str();
        let edge_document = crate::native_answer_persist::model_document_fields(&edge_payload, &[]);
        let mut edge_metadata = edge_payload["metadata"]
            .as_object()
            .cloned()
            .unwrap_or_default();
        edge_metadata.insert("_class_name".to_owned(), json!("ConversationEdge"));
        let (edge_mutation, _) = crate::native_memory_pin::graph_mutation(
            request,
            &graph_scope,
            edge_id,
            format!("native-context-snapshot|{run_id}|{run_step_seq}|{attempt_seq}|edge|{ordinal}"),
            "edge",
            "conversation_edge",
            edge_payload,
            edge_document,
            edge_metadata,
        )
        .map_err(|_| "native answer snapshot dependency mutation is invalid".to_owned())?;
        mutations.push(edge_mutation);
    }
    let batch = GraphMutationBatchV2 {
        contract_version: 2,
        atomic: true,
        mutations,
        index_jobs: Vec::new(),
        default_acl: GraphDefaultAclIntentV1 {
            truth_graph: "conversation".to_owned(),
        },
    };
    batch
        .validate()
        .map_err(|_| "native answer snapshot graph batch is invalid".to_owned())?;
    Ok(batch)
}

fn restore_answer_aliases(
    mut answer: AnswerWithCitations,
    aliases: &BTreeMap<String, String>,
    used_node_ids: &[String],
) -> Result<Value, String> {
    let allowed_nodes: BTreeSet<&str> = used_node_ids.iter().map(String::as_str).collect();
    for claim in &mut answer.claims {
        for citation in &mut claim.citations {
            let canonical_id = aliases
                .get(&citation.source_node_id)
                .ok_or_else(|| "answer citation used an unknown node alias".to_owned())?;
            if !allowed_nodes.contains(canonical_id.as_str()) {
                return Err(
                    "answer citation references a node outside selected evidence".to_owned(),
                );
            }
            citation.source_node_id = canonical_id.clone();
        }
    }
    serde_json::to_value(answer).map_err(|_| "could not encode structured answer".to_owned())
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct AnswerWithCitations {
    text: String,
    reasoning: String,
    #[serde(default)]
    claims: Vec<ClaimWithCitations>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ClaimWithCitations {
    claim: String,
    #[serde(default)]
    citations: Vec<SpanRef>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SpanRef {
    source_node_id: String,
    mention_index: usize,
    span_index: usize,
}

#[derive(Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatMessage,
}

#[derive(Deserialize)]
struct ChatMessage {
    content: Option<String>,
}

fn answer_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["text", "reasoning", "claims"],
        "properties": {
            "text": {"type": "string"},
            "reasoning": {"type": "string"},
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["claim", "citations"],
                    "properties": {
                        "claim": {"type": "string"},
                        "citations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": false,
                                "required": ["source_node_id", "mention_index", "span_index"],
                                "properties": {
                                    "source_node_id": {"type": "string"},
                                    "mention_index": {"type": "integer", "minimum": 0},
                                    "span_index": {"type": "integer", "minimum": 0}
                                }
                            }
                        }
                    }
                }
            }
        }
    })
}

fn evidence_text_and_aliases(
    evidence_pack: &Value,
) -> Result<(String, BTreeMap<String, String>), String> {
    let nodes = evidence_pack
        .get("nodes")
        .and_then(Value::as_array)
        .ok_or_else(|| "evidence pack must contain nodes".to_owned())?;
    let edges = evidence_pack
        .get("edges")
        .and_then(Value::as_array)
        .ok_or_else(|| "evidence pack must contain edges".to_owned())?;
    let mut node_ids = nodes
        .iter()
        .filter_map(|node| node.get("node_id").and_then(Value::as_str))
        .map(str::to_owned)
        .collect::<Vec<_>>();
    node_ids.sort();
    node_ids.dedup();
    let aliases = node_ids
        .iter()
        .enumerate()
        .map(|(index, id)| (format!("N{}", index + 1), id.clone()))
        .collect::<BTreeMap<_, _>>();
    let id_to_alias = aliases
        .iter()
        .map(|(alias, id)| (id.as_str(), alias.as_str()))
        .collect::<BTreeMap<_, _>>();
    let mut edge_ids = edges
        .iter()
        .filter_map(|edge| edge.get("id").and_then(Value::as_str))
        .map(str::to_owned)
        .collect::<Vec<_>>();
    edge_ids.sort();
    edge_ids.dedup();
    let edge_aliases = edge_ids
        .iter()
        .enumerate()
        .map(|(index, id)| (id.as_str(), format!("E{}", index + 1)))
        .collect::<BTreeMap<_, _>>();

    let mut lines = Vec::new();
    for node in nodes {
        let id = node
            .get("node_id")
            .and_then(Value::as_str)
            .ok_or_else(|| "evidence node is missing node_id".to_owned())?;
        let alias = id_to_alias
            .get(id)
            .ok_or_else(|| "evidence node could not be aliased".to_owned())?;
        let label = node
            .get("label")
            .and_then(Value::as_str)
            .unwrap_or_default();
        lines.push(format!("NODE {alias} | {label}"));
        if let Some(mentions) = node.get("mentions").and_then(Value::as_array) {
            for (mention_index, mention) in mentions.iter().enumerate() {
                if let Some(spans) = mention.get("spans").and_then(Value::as_array) {
                    for (span_index, span) in spans.iter().enumerate() {
                        let excerpt = span
                            .get("excerpt")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .replace('\n', " ");
                        if !excerpt.trim().is_empty() {
                            let excerpt: String = excerpt.chars().take(240).collect();
                            lines.push(format!("  M{mention_index} S{span_index}: {excerpt}"));
                        }
                    }
                }
            }
        }
    }
    for edge in edges {
        let id = edge
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| "evidence edge is missing id".to_owned())?;
        let relation = edge
            .get("relation")
            .and_then(Value::as_str)
            .unwrap_or("related");
        let summary = edge
            .get("summary")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .replace('\n', " ");
        let summary: String = summary.chars().take(240).collect();
        let alias = edge_aliases
            .get(id)
            .ok_or_else(|| "evidence edge could not be aliased".to_owned())?;
        lines.push(format!("EDGE {alias} | {relation}: {summary}"));
    }
    Ok((lines.join("\n"), aliases))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aliases_are_deterministic_and_keep_canonical_ids_out_of_evidence_text() {
        let pack = json!({
            "nodes": [
                {"node_id": "node-z-secret", "label": "Z", "mentions": []},
                {"node_id": "node-a-secret", "label": "A", "mentions": []}
            ],
            "edges": []
        });
        let (first, aliases) = evidence_text_and_aliases(&pack).unwrap();
        let (second, _) = evidence_text_and_aliases(&pack).unwrap();
        assert_eq!(first, second);
        assert!(first.contains("NODE N1 | A"));
        assert!(first.contains("NODE N2 | Z"));
        assert!(!first.contains("node-a-secret"));
        assert_eq!(aliases.get("N1").map(String::as_str), Some("node-a-secret"));
    }

    #[test]
    fn provider_answer_restores_known_alias_and_rejects_unknown_alias() {
        let answer: AnswerWithCitations = serde_json::from_value(json!({
            "text": "Answer",
            "reasoning": "Evidence supports this.",
            "claims": [{"claim": "fact", "citations": [{
                "source_node_id": "N1", "mention_index": 0, "span_index": 0
            }]}]
        }))
        .unwrap();
        let aliases = BTreeMap::from([("N1".to_owned(), "canonical-node".to_owned())]);
        let allowed = vec!["canonical-node".to_owned()];
        let restored = restore_answer_aliases(answer.clone(), &aliases, &allowed).unwrap();
        assert_eq!(
            restored["claims"][0]["citations"][0]["source_node_id"],
            "canonical-node"
        );
        let mut unknown = answer;
        unknown.claims[0].citations[0].source_node_id = "N99".to_owned();
        assert!(restore_answer_aliases(unknown, &aliases, &allowed).is_err());
        assert!(
            serde_json::from_value::<AnswerWithCitations>(json!({
                "text": "x", "reasoning": "y", "claims": [], "extra": true
            }))
            .is_err()
        );
    }

    #[test]
    fn configuration_rejects_non_http_endpoint_and_blank_model() {
        assert!(OpenAiCompatibleChatConfig::new("file:///tmp/x", "model", None).is_err());
        assert!(OpenAiCompatibleChatConfig::new("http://localhost:1", " ", None).is_err());
    }

    #[test]
    fn native_answer_snapshot_records_the_exact_aliased_provider_messages() {
        let evidence = json!({
            "nodes": [{
                "node_id": "canonical-node-secret",
                "label": "Label",
                "mentions": [{"spans": [{"excerpt": "Verified evidence text"}]}]
            }],
            "edges": []
        });
        let (messages, _) =
            answer_prompt_messages("system instructions", "question", &evidence).unwrap();
        let request = RuntimeNativeStepRequest {
            executor_target: kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET.to_owned(),
            operation: "aa_generate_answer_with_citations".to_owned(),
            op_version: "v1".to_owned(),
            run_id: "run-snapshot".to_owned(),
            workflow_id: "agentic_answering.v2".to_owned(),
            node_id: "answer".to_owned(),
            conversation_id: "conv-snapshot".to_owned(),
            turn_node_id: None,
            token_id: "token-snapshot".to_owned(),
            parent_token_id: None,
            step_seq: 6,
            state: serde_json::from_value(json!({
                "run_id": "run-snapshot",
                "conversation_id": "conv-snapshot",
                "run_step_seq": 12,
                "attempt_seq": 2,
                "question": "question",
                "system_prompt": "system instructions",
                "used_node_ids": ["canonical-node-secret"],
                "conversation_context_messages": [
                    {"role":"system","content":"context system","source":"system","node_id":null},
                    {"role":"user","content":"prior turn","source":"history_turn","node_id":"prior-turn"}
                ],
                "conversation_context_node_ids": ["prior-turn", "canonical-node-secret"],
                "evidence_pack_digest": {
                    "node_ids": ["canonical-node-secret"],
                    "edge_ids": [],
                    "evidence_pack_hash": "fixture-evidence-hash"
                },
                "_rt_graph_scope": {
                    "storage_namespace": "alpha",
                    "workspace": "workspace-a",
                    "graph_space": "space-a"
                },
                "_rt": {"evidence_pack": evidence}
            }))
            .unwrap(),
            runtime_routes: Vec::new(),
            ephemeral_context: Default::default(),
        };
        let batch = answer_context_snapshot_batch(&request, "fixture-model").unwrap();
        assert!(batch.validate().is_ok());
        assert_eq!(batch.mutations.len(), 3);
        let snapshot = &batch.mutations[0].payload;
        let properties = &snapshot["properties"];
        let saved_messages: Value =
            serde_json::from_str(properties["prompt_messages"].as_str().unwrap()).unwrap();
        assert_eq!(saved_messages, json!(messages));
        let saved_conversation_context: Value = serde_json::from_str(
            properties["conversation_context_messages"]
                .as_str()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(saved_conversation_context[1]["node_id"], "prior-turn");
        assert_eq!(
            serde_json::from_str::<Value>(
                properties["conversation_context_node_ids"]
                    .as_str()
                    .unwrap()
            )
            .unwrap(),
            json!(["prior-turn", "canonical-node-secret"])
        );
        assert!(
            saved_messages[1]["content"]
                .as_str()
                .unwrap()
                .contains("NODE N1 | Label")
        );
        assert!(
            !saved_messages[1]["content"]
                .as_str()
                .unwrap()
                .contains("canonical-node-secret")
        );
        assert_eq!(batch.mutations[1].payload["relation"], "depends_on");
        assert_eq!(
            batch.mutations[1].payload["target_ids"],
            json!(["prior-turn"])
        );
        assert_eq!(batch.mutations[2].payload["relation"], "depends_on");
        assert_eq!(
            batch.mutations[2].payload["target_ids"],
            json!(["canonical-node-secret"])
        );
        assert_eq!(batch.default_acl.truth_graph, "conversation");
    }

    #[tokio::test]
    async fn provider_uses_json_schema_and_restores_aliases_from_local_http_fixture() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
            .await
            .unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = Vec::new();
            let mut buffer = [0_u8; 4096];
            let header_end = loop {
                let count = stream.read(&mut buffer).await.unwrap();
                assert_ne!(count, 0);
                request.extend_from_slice(&buffer[..count]);
                if let Some(index) = request.windows(4).position(|window| window == b"\r\n\r\n") {
                    break index + 4;
                }
            };
            let headers = String::from_utf8_lossy(&request[..header_end]).to_ascii_lowercase();
            assert!(headers.contains("authorization: bearer fixture-key"));
            let content_length = headers
                .lines()
                .find_map(|line| line.strip_prefix("content-length: "))
                .and_then(|value| value.trim().parse::<usize>().ok())
                .unwrap();
            while request.len() < header_end + content_length {
                let count = stream.read(&mut buffer).await.unwrap();
                assert_ne!(count, 0);
                request.extend_from_slice(&buffer[..count]);
            }
            let body: Value =
                serde_json::from_slice(&request[header_end..header_end + content_length]).unwrap();
            assert_eq!(body["response_format"]["type"], "json_schema");
            let prompt = body["messages"][1]["content"].as_str().unwrap();
            assert!(prompt.contains("NODE N1 | Label"));
            assert!(!prompt.contains("canonical-node-secret"));
            let answer = json!({
                "text": "Answer",
                "reasoning": "The span supports it.",
                "claims": [{"claim": "Fact", "citations": [{
                    "source_node_id": "N1", "mention_index": 0, "span_index": 0
                }]}]
            });
            let response = json!({
                "choices": [{"message": {"content": answer.to_string()}}]
            })
            .to_string();
            let wire = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                response.len(),
                response
            );
            stream.write_all(wire.as_bytes()).await.unwrap();
        });
        let provider = OpenAiCompatibleChatConfig::new(
            format!("http://{address}/v1/chat/completions"),
            "fixture-model",
            Some("fixture-key".to_owned()),
        )
        .unwrap();
        let result = provider
            .answer(
                "system",
                "question",
                &json!({
                    "nodes": [{"node_id": "canonical-node-secret", "label": "Label", "mentions": [{"spans": [{"excerpt": "Evidence"}]}]}],
                    "edges": []
                }),
                &["canonical-node-secret".to_owned()],
            )
            .await
            .unwrap();
        assert_eq!(
            result["claims"][0]["citations"][0]["source_node_id"],
            "canonical-node-secret"
        );
        server.await.unwrap();
    }

    #[test]
    fn native_answer_operation_rejects_citations_outside_selected_evidence() {
        let mut registry = NativeOperationRegistry::default();
        register(&mut registry).unwrap();
        let mut request = RuntimeNativeStepRequest {
            executor_target: kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET.to_owned(),
            operation: "aa_generate_answer_with_citations".to_owned(),
            op_version: "v1".to_owned(),
            run_id: "run-1".to_owned(),
            workflow_id: "agentic_answering.v2".to_owned(),
            node_id: "answer".to_owned(),
            conversation_id: "conv-1".to_owned(),
            turn_node_id: None,
            token_id: "token-1".to_owned(),
            parent_token_id: None,
            step_seq: 1,
            state: serde_json::from_value(json!({
                "used_node_ids": ["allowed-node"],
                "_rt": {"evidence_pack": {"nodes": [{"node_id": "allowed-node"}]}}
            }))
            .unwrap(),
            runtime_routes: Vec::new(),
            ephemeral_context: serde_json::from_value(json!({
                "provider_model": "fixture-model",
                "provider_answer": {
                    "text": "Answer", "reasoning": "Reasoning", "claims": [{
                        "claim": "Fact", "citations": [{
                            "source_node_id": "other-node", "mention_index": 0, "span_index": 0
                        }]
                    }]
                }
            }))
            .unwrap(),
        };
        assert!(kogwistar_runtime::execute_native_step_with_registry(&request, &registry).is_err());
        request.ephemeral_context.insert(
            "provider_answer".to_owned(),
            json!({
                "text": "Answer", "reasoning": "Reasoning", "claims": [{
                    "claim": "Fact", "citations": [{
                        "source_node_id": "allowed-node", "mention_index": 0, "span_index": 0
                    }]
                }]
            }),
        );
        let success =
            kogwistar_runtime::execute_native_step_with_registry(&request, &registry).unwrap();
        assert_eq!(
            success.result.unwrap()["final_state"]["answer"]["text"],
            "Answer"
        );
    }

    #[test]
    fn native_citation_repair_revalidates_repaired_span_coordinates() {
        let mut registry = NativeOperationRegistry::default();
        register(&mut registry).unwrap();
        let mut request = RuntimeNativeStepRequest {
            executor_target: kogwistar_runtime::RUST_NATIVE_EXECUTOR_TARGET.to_owned(),
            operation: "aa_validate_or_repair_citations".to_owned(),
            op_version: "v1".to_owned(),
            run_id: "run-1".to_owned(),
            workflow_id: "agentic_answering.v2".to_owned(),
            node_id: "repair".to_owned(),
            conversation_id: "conv-1".to_owned(),
            turn_node_id: None,
            token_id: "token-1".to_owned(),
            parent_token_id: None,
            step_seq: 1,
            state: serde_json::from_value(json!({
                "used_node_ids": ["allowed-node"],
                "answer": {
                    "text": "Answer", "reasoning": "", "claims": [{
                        "claim": "Fact", "citations": [{
                            "source_node_id": "allowed-node", "mention_index": 8, "span_index": 0
                        }]
                    }]
                },
                "_rt": {"evidence_pack": {"nodes": [{
                    "node_id": "allowed-node",
                    "mentions": [{"spans": [{"excerpt": "Grounded"}]}]
                }]}}
            }))
            .unwrap(),
            runtime_routes: Vec::new(),
            ephemeral_context: serde_json::from_value(json!({
                "repaired_answer": {
                    "text": "Answer", "reasoning": "", "claims": [{
                        "claim": "Fact", "citations": [{
                            "source_node_id": "allowed-node", "mention_index": 4, "span_index": 0
                        }]
                    }]
                }
            }))
            .unwrap(),
        };
        let rejected =
            kogwistar_runtime::execute_native_step_with_registry(&request, &registry).unwrap();
        let rejected_answer = &rejected.result.unwrap()["final_state"]["answer"];
        assert_eq!(rejected_answer["text"], "Answer");
        assert_eq!(rejected_answer["claims"][0]["citations"], json!([]));
        assert!(
            rejected_answer["reasoning"]
                .as_str()
                .unwrap()
                .contains("citation_repair_failed")
        );

        request.ephemeral_context.insert(
            "repaired_answer".to_owned(),
            json!({
                "text": "Answer", "reasoning": "", "claims": [{
                    "claim": "Fact", "citations": [{
                        "source_node_id": "allowed-node", "mention_index": 0, "span_index": 0
                    }]
                }]
            }),
        );
        let accepted =
            kogwistar_runtime::execute_native_step_with_registry(&request, &registry).unwrap();
        assert_eq!(
            accepted.result.unwrap()["final_state"]["answer"]["claims"][0]["citations"][0]["source_node_id"],
            "allowed-node"
        );
    }
}
