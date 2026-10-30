use kogwistar_contracts::{GraphMutationBatchV2, GraphMutationV1};
use kogwistar_runtime::{
    NativeOperationOutput, NativeOperationRegistrationError, NativeOperationRegistry,
    RecordedStateUpdate, RuntimeNativeExecutionError, RuntimeNativeStepRequest,
};
use serde_json::{Map, Value, json};

pub(crate) fn registry() -> Result<NativeOperationRegistry, NativeOperationRegistrationError> {
    let mut registry = NativeOperationRegistry::default();
    registry.register_graph_v2("memory_pin", "v1", execute_memory_pin)?;
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

fn optional_integer(state: &Map<String, Value>, field: &str) -> i64 {
    state.get(field).and_then(Value::as_i64).unwrap_or_default()
}

fn apply_update(
    request: &RuntimeNativeStepRequest,
    update: Map<String, Value>,
) -> Result<NativeOperationOutput, RuntimeNativeExecutionError> {
    let mut final_state = request.state.clone();
    final_state.extend(update.clone());
    Ok(NativeOperationOutput {
        state_update: vec![RecordedStateUpdate::Pair(("u".to_owned(), update))],
        result: Some(json!({
            "workflow_status": "succeeded",
            "final_state": final_state,
        })),
    })
}

pub(crate) fn metadata_projection(
    metadata: &Map<String, Value>,
    document_fields: &Map<String, Value>,
    doc_id: Option<&str>,
    model_class: &str,
) -> Map<String, Value> {
    let mut projected = metadata.clone();
    projected
        .entry("_class_name".to_owned())
        .or_insert_with(|| Value::String(model_class.to_owned()));
    if let Some(doc_id) = doc_id {
        projected.insert("doc_id".to_owned(), Value::String(doc_id.to_owned()));
    }
    for field in ["doc_id", "label", "type", "summary"] {
        if field != "doc_id"
            && let Some(value) = document_fields.get(field)
        {
            projected.insert(field.to_owned(), value.clone());
        }
    }
    for field in ["properties", "mentions", "source_ids", "target_ids"] {
        if let Some(value) = document_fields.get(field) {
            if value.is_null() {
                continue;
            }
            let encoded = serde_json::to_string(value).unwrap_or_default();
            projected.insert(field.to_owned(), Value::String(encoded));
        }
    }
    projected
}

// Keep the canonical graph fields explicit at this serialization boundary.
#[allow(clippy::too_many_arguments)]
pub(crate) fn graph_mutation(
    request: &RuntimeNativeStepRequest,
    scope: &Value,
    id: &str,
    event_id: String,
    entity_kind: &str,
    model_class: &str,
    payload: Value,
    document_fields: Map<String, Value>,
    metadata: Map<String, Value>,
) -> Result<(GraphMutationV1, Vec<Value>), RuntimeNativeExecutionError> {
    let namespace = scope
        .get("storage_namespace")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| invalid("_rt_graph_scope", "trusted storage namespace is missing"))?;
    let workspace_id = scope
        .get("workspace")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_owned);
    let graph_space = scope
        .get("graph_space")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_owned);
    let table = if entity_kind == "node" {
        "gke_nodes"
    } else {
        "gke_edges"
    };
    let document = serde_json::to_string(&document_fields)
        .map_err(|_| invalid("memory", "canonical graph document could not be serialized"))?;
    let doc_id = payload.get("doc_id").and_then(Value::as_str);
    let projected_metadata = metadata_projection(&metadata, &document_fields, doc_id, model_class);
    let envelope = json!({
        "schema_version": 1,
        "model_class": model_class,
        "payload": payload,
    });
    let mutation = serde_json::from_value::<GraphMutationV1>(json!({
        "scope": {
            "namespace": namespace,
            "workspace_id": workspace_id,
            "graph_space": graph_space,
        },
        "table": table,
        "entity_kind": entity_kind,
        "event_id": event_id,
        "op": "ADD",
        "payload": envelope["payload"],
        "record": {
            "id": id,
            "document": document,
            "metadata": projected_metadata,
        },
        "embedding_dim": 0,
        "model_envelope": envelope,
    }))
    .map_err(|_| invalid("memory", "canonical graph mutation is invalid"))?;

    let base = format!(
        "native-memory-pin|{}|{}|{}|{}|{}",
        request.run_id, request.node_id, request.token_id, request.step_seq, id
    );
    let kinds: &[&str] = if entity_kind == "node" {
        &["node_docs", "node_refs", "node_embedding"]
    } else {
        &["edge_refs", "edge_endpoints", "node_embedding"]
    };
    let jobs = kinds
        .iter()
        .map(|index_kind| {
            json!({
                "job_id": format!("{base}|{index_kind}"),
                "namespace": namespace,
                "entity_kind": entity_kind,
                "entity_id": id,
                "index_kind": index_kind,
                "op": "UPSERT",
                "payload_json": null,
                "max_retries": 3,
            })
        })
        .collect();
    Ok((mutation, jobs))
}

fn execute_memory_pin(
    request: &RuntimeNativeStepRequest,
) -> Result<(NativeOperationOutput, Option<GraphMutationBatchV2>), RuntimeNativeExecutionError> {
    let state = &request.state;
    let memory = state.get("memory").and_then(Value::as_object);
    let selected = memory
        .and_then(|memory| memory.get("selected"))
        .and_then(Value::as_object);
    let text = memory
        .and_then(|memory| memory.get("memory_context_text"))
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty());

    let Some(selected) = selected.filter(|_| text.is_some()) else {
        let mut update = Map::new();
        update.insert(
            "memory_pin".to_owned(),
            json!({"memory_context_node_id": null, "pinned_edge_ids": []}),
        );
        return Ok((apply_update(request, update)?, None));
    };

    let text = text.expect("selected and non-empty context were checked");
    let conversation_id = required_string(state, "conversation_id")?;
    if conversation_id != request.conversation_id {
        return Err(invalid(
            "conversation_id",
            "must match the admitted runtime conversation",
        ));
    }
    let owner_id = required_string(state, "_rt_run_owner_id")?;
    let turn_node_id = required_string(state, "turn_node_id")?;
    if request
        .turn_node_id
        .as_deref()
        .is_some_and(|admitted| admitted != turn_node_id)
    {
        return Err(invalid(
            "turn_node_id",
            "must match the admitted runtime turn",
        ));
    }
    let memory_id = required_string(state, "mem_id")?;
    let turn_index = state
        .get("turn_index")
        .and_then(Value::as_i64)
        .ok_or_else(|| invalid("turn_index", "expected an integer"))?;
    let span = state
        .get("self_span")
        .filter(|value| value.is_object())
        .cloned()
        .ok_or_else(|| invalid("self_span", "expected a span object"))?;
    let scope = state
        .get("_rt_graph_scope")
        .filter(|value| value.is_object())
        .ok_or_else(|| invalid("_rt_graph_scope", "trusted graph scope is missing"))?;
    let nodes = selected
        .get("nodes")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("memory.selected.nodes", "expected an array"))?;
    let edges = selected
        .get("edges")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("memory.selected.edges", "expected an array"))?;
    let source_node_ids = nodes
        .iter()
        .map(|item| {
            item.get("id")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .map(str::to_owned)
                .ok_or_else(|| invalid("memory.selected.nodes", "source node lacks an ID"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let source_edge_ids = edges
        .iter()
        .map(|item| {
            item.get("id")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .map(str::to_owned)
                .ok_or_else(|| invalid("memory.selected.edges", "source edge lacks an ID"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if source_node_ids
        .iter()
        .collect::<std::collections::BTreeSet<_>>()
        .len()
        != source_node_ids.len()
        || source_edge_ids
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != source_edge_ids.len()
    {
        return Err(invalid(
            "memory.selected",
            "selected source IDs must be unique within each entity kind",
        ));
    }

    let previous = state
        .get("prev_turn_meta_summary")
        .and_then(Value::as_object);
    let char_delta = i64::try_from(text.chars().count())
        .map_err(|_| invalid("memory_context_text", "text length exceeds supported range"))?;
    let previous_chars = previous
        .map(|value| optional_integer(value, "prev_node_char_distance_from_last_summary"))
        .unwrap_or_default();
    let previous_distance = previous
        .map(|value| optional_integer(value, "prev_node_distance_from_last_summary"))
        .unwrap_or_default();
    let tail_turn_index = previous
        .map(|value| optional_integer(value, "tail_turn_index"))
        .unwrap_or_default();
    let next_chars = previous_chars
        .checked_add(char_delta)
        .ok_or_else(|| invalid("prev_turn_meta_summary", "character distance overflow"))?;
    let next_distance = previous_distance
        .checked_add(1)
        .ok_or_else(|| invalid("prev_turn_meta_summary", "turn distance overflow"))?;

    let metadata = json!({
        "entity_type": "memory_context",
        "type": "entity",
        "level_from_root": 0,
        "user_id": owner_id,
        "conversation_id": conversation_id,
        "turn_index": turn_index,
        "role": "system",
        "in_conversation_chain": false,
        "visibility": "private",
        "security_scope": scope["security_scope"],
        "owner_security_scope": scope["security_scope"],
        "owner_agent_id": owner_id,
        "agent_id": owner_id,
        "shared_scope": false,
        "lifecycle_status": "active",
        "redirect_to_id": null,
    });
    let mut node_payload = json!({
        "id": memory_id,
        "label": format!("Memory context (turn {turn_index})"),
        "type": "entity",
        "doc_id": memory_id,
        "summary": text,
        "role": "system",
        "turn_index": turn_index,
        "conversation_id": conversation_id,
        "user_id": owner_id,
        "mentions": [{"spans": [span], "multimodal_spans": []}],
        "properties": {
            "user_id": owner_id,
            "source_memory_nodes_ids": source_node_ids,
            "source_memory_edges_ids": source_edge_ids,
        },
        "metadata": metadata,
        "level_from_root": 0,
        "domain_id": null,
        "canonical_entity_id": null,
        "embedding": null,
    });
    let node_doc = json!({
        "label": format!("Memory context (turn {turn_index})"),
        "type": "entity",
        "summary": text,
        "domain_id": null,
        "canonical_entity_id": null,
        "properties": node_payload["properties"],
        "mentions": node_payload["mentions"],
    });
    let node_doc_fields = node_doc
        .as_object()
        .cloned()
        .ok_or_else(|| invalid("memory", "node projection is invalid"))?;
    let node_meta = metadata
        .as_object()
        .cloned()
        .ok_or_else(|| invalid("memory", "node metadata is invalid"))?;
    let mut mutations = Vec::new();
    let mut index_jobs = Vec::new();
    let (mutation, jobs) = graph_mutation(
        request,
        scope,
        memory_id,
        format!(
            "native-memory-pin|{}|{}|node|{}",
            request.run_id, request.node_id, memory_id
        ),
        "node",
        "conversation_node",
        node_payload.take(),
        node_doc_fields,
        node_meta,
    )?;
    mutations.push(mutation);
    index_jobs.extend(jobs);

    let mut pinned_edge_ids = Vec::new();
    let mut append_edge = |id: String,
                           source_ids: Vec<String>,
                           target_ids: Vec<String>,
                           target_edge_ids: Vec<String>,
                           relation: &str,
                           summary: &str,
                           char_distance: i64,
                           turn_distance: i64,
                           include_in_result: bool|
     -> Result<(), RuntimeNativeExecutionError> {
        let edge_metadata = json!({
            "char_distance_from_last_summary": char_distance,
            "turn_distance_from_last_summary": turn_distance,
            "tail_turn_index": tail_turn_index,
            "visibility": "private",
            "security_scope": scope["security_scope"],
            "owner_security_scope": scope["security_scope"],
            "owner_agent_id": owner_id,
            "agent_id": owner_id,
            "shared_scope": false,
            "lifecycle_status": "active",
            "redirect_to_id": null,
        });
        let payload = json!({
            "id": id,
            "label": relation,
            "type": "relationship",
            "summary": summary,
            "domain_id": null,
            "canonical_entity_id": null,
            "properties": {"entity_type": "conversation_edge"},
            "mentions": [{"spans": [span], "multimodal_spans": []}],
            "source_ids": source_ids,
            "target_ids": target_ids,
            "relation": relation,
            "source_edge_ids": [],
            "target_edge_ids": target_edge_ids,
            "embedding": null,
            "doc_id": format!("conv:{conversation_id}"),
            "metadata": edge_metadata,
            "level_from_root": null,
        });
        let mut document_fields = Map::new();
        for field in [
            "label",
            "type",
            "summary",
            "domain_id",
            "canonical_entity_id",
            "properties",
            "mentions",
            "source_ids",
            "target_ids",
            "relation",
            "source_edge_ids",
            "target_edge_ids",
        ] {
            document_fields.insert(field.to_owned(), payload[field].clone());
        }
        let edge_meta = edge_metadata
            .as_object()
            .cloned()
            .ok_or_else(|| invalid("memory", "edge metadata is invalid"))?;
        let payload_id = payload["id"]
            .as_str()
            .ok_or_else(|| invalid("memory", "edge identity is missing"))?
            .to_owned();
        let (mutation, jobs) = graph_mutation(
            request,
            scope,
            &payload_id,
            format!(
                "native-memory-pin|{}|{}|edge|{}",
                request.run_id, request.node_id, payload_id
            ),
            "edge",
            "conversation_edge",
            payload,
            document_fields,
            edge_meta,
        )?;
        if include_in_result {
            pinned_edge_ids.push(id);
        }
        mutations.push(mutation);
        index_jobs.extend(jobs);
        Ok(())
    };

    append_edge(
        format!("{turn_node_id}::hm::{memory_id}"),
        vec![turn_node_id.to_owned()],
        vec![memory_id.to_owned()],
        Vec::new(),
        "has_memory_context",
        "This turn uses long-memory context",
        previous_chars,
        previous_distance,
        true,
    )?;
    for source_id in &source_node_ids {
        append_edge(
            format!("{memory_id}::n::{source_id}"),
            vec![memory_id.to_owned()],
            vec![source_id.clone()],
            Vec::new(),
            "summarizes",
            "Memory context summarizes prior conversation artifact",
            next_chars,
            next_distance,
            false,
        )?;
    }
    for source_id in &source_edge_ids {
        let edge_id = format!("{memory_id}::e::{source_id}");
        append_edge(
            edge_id,
            vec![memory_id.to_owned()],
            Vec::new(),
            vec![source_id.clone()],
            "summarizes",
            "Memory context summarizes prior conversation artifact",
            next_chars,
            next_distance,
            true,
        )?;
    }

    let mut updated_summary = Map::new();
    updated_summary.insert(
        "prev_node_char_distance_from_last_summary".to_owned(),
        json!(next_chars),
    );
    updated_summary.insert(
        "prev_node_distance_from_last_summary".to_owned(),
        json!(next_distance),
    );
    updated_summary.insert("tail_turn_index".to_owned(), json!(tail_turn_index));
    let mut update = Map::new();
    update.insert(
        "memory_pin".to_owned(),
        json!({
            "memory_context_node_id": memory_id,
            "pinned_edge_ids": pinned_edge_ids,
        }),
    );
    update.insert(
        "prev_turn_meta_summary".to_owned(),
        Value::Object(updated_summary),
    );
    let output = apply_update(request, update)?;
    let batch = GraphMutationBatchV2 {
        contract_version: 2,
        atomic: true,
        mutations,
        index_jobs: serde_json::from_value(Value::Array(index_jobs))
            .map_err(|_| invalid("memory", "index job intent is invalid"))?,
        default_acl: kogwistar_contracts::GraphDefaultAclIntentV1 {
            truth_graph: "conversation".to_owned(),
        },
    };
    Ok((output, Some(batch)))
}

#[cfg(test)]
mod tests {
    use super::registry;
    use kogwistar_runtime::{
        RUST_NATIVE_EXECUTOR_TARGET, RecordedStateUpdate, RuntimeNativeStepRequest,
        execute_native_step_with_registry,
    };
    use serde_json::{Map, Value, json};

    fn request(state: Value) -> RuntimeNativeStepRequest {
        RuntimeNativeStepRequest {
            executor_target: RUST_NATIVE_EXECUTOR_TARGET.to_owned(),
            operation: "memory_pin".to_owned(),
            op_version: "v1".to_owned(),
            run_id: "run-1".to_owned(),
            workflow_id: "conversation".to_owned(),
            node_id: "pin-memory".to_owned(),
            conversation_id: "conversation-1".to_owned(),
            turn_node_id: Some("turn-7".to_owned()),
            token_id: "token-1".to_owned(),
            parent_token_id: None,
            step_seq: 4,
            state: serde_json::from_value::<Map<String, Value>>(state).unwrap(),
            runtime_routes: Vec::new(),
            ephemeral_context: Map::new(),
        }
    }

    fn selected_state() -> Value {
        json!({
            "conversation_id": "conversation-1",
            "user_id": "spoofed-user-is-not-used",
            "turn_node_id": "turn-7",
            "turn_index": 7,
            "mem_id": "memory-7",
            "self_span": {
                "collection_page_url": "conversation/1",
                "document_page_url": "conversation/1",
                "doc_id": "conversation-1",
                "insertion_method": "conversation",
                "start_char": 0,
                "end_char": 3,
                "excerpt": "ask"
            },
            "_rt_run_owner_id": "alice",
            "_rt_graph_scope": {
                "storage_namespace": "alpha",
                "security_scope": "tenant-a",
                "workspace": "workspace-a",
                "graph_space": "conversation"
            },
            "prev_turn_meta_summary": {
                "prev_node_char_distance_from_last_summary": 2,
                "prev_node_distance_from_last_summary": 1,
                "tail_turn_index": 6
            },
            "memory": {
                "memory_context_text": "雪a",
                "selected": {
                    "nodes": [{"id": "source-node"}],
                    "edges": [{"id": "source-edge"}]
                }
            }
        })
    }

    #[test]
    fn native_memory_pin_emits_atomic_acl_graph_batch_and_unicode_state_delta() {
        let registry = registry().unwrap();
        let request = request(selected_state());
        let effect = execute_native_step_with_registry(&request, &registry).unwrap();
        let batch = effect.graph_mutation_batch_v2.unwrap();
        batch.validate().unwrap();
        assert_eq!(batch.default_acl.truth_graph, "conversation");
        assert_eq!(batch.mutations.len(), 4);
        assert_eq!(batch.index_jobs.len(), 12);
        assert_eq!(batch.mutations[0].scope.namespace, "alpha");
        assert_eq!(
            batch.mutations[0].scope.workspace_id.as_deref(),
            Some("workspace-a")
        );
        assert_eq!(
            batch.mutations[0].scope.graph_space.as_deref(),
            Some("conversation")
        );
        assert_eq!(batch.mutations[0].record.id, "memory-7");
        assert_eq!(batch.mutations[0].record.metadata["user_id"], "alice");
        assert_eq!(batch.mutations[0].record.metadata["doc_id"], "memory-7");
        for mutation in &batch.mutations {
            let envelope = mutation.model_envelope.as_ref().unwrap();
            envelope.validate_family_v1().unwrap();
            assert_eq!(
                mutation.record.metadata["doc_id"],
                envelope.payload["doc_id"]
            );
        }
        assert!(
            batch.mutations[0]
                .record
                .document
                .as_deref()
                .unwrap()
                .contains("雪a")
        );
        assert!(
            batch
                .index_jobs
                .iter()
                .any(|job| job.index_kind == "node_embedding")
        );
        let RecordedStateUpdate::Pair((operation, update)) = &effect.state_update[0] else {
            panic!("memory pin returns the shared tuple-form state update");
        };
        assert_eq!(operation, "u");
        assert_eq!(
            update["prev_turn_meta_summary"]["prev_node_char_distance_from_last_summary"], 4,
            "Python counts Unicode code points, not UTF-8 bytes"
        );
        assert_eq!(
            effect.result.as_ref().unwrap()["final_state"]["user_id"],
            "spoofed-user-is-not-used"
        );
    }

    #[test]
    fn native_memory_pin_without_selection_is_state_only() {
        let registry = registry().unwrap();
        let mut state = selected_state();
        state["memory"]["selected"] = Value::Null;
        let effect = execute_native_step_with_registry(&request(state), &registry).unwrap();
        assert!(effect.graph_mutation_batch_v2.is_none());
        assert_eq!(
            effect.result.as_ref().unwrap()["final_state"]["memory_pin"]["memory_context_node_id"],
            Value::Null
        );
    }

    #[test]
    fn native_memory_pin_requires_server_scope_and_rejects_duplicate_sources() {
        let registry = registry().unwrap();
        let mut missing_scope = selected_state();
        missing_scope
            .as_object_mut()
            .unwrap()
            .remove("_rt_graph_scope");
        assert!(execute_native_step_with_registry(&request(missing_scope), &registry).is_err());

        let mut duplicate = selected_state();
        duplicate["memory"]["selected"]["nodes"] = json!([
            {"id": "source-node"}, {"id": "source-node"}
        ]);
        assert!(execute_native_step_with_registry(&request(duplicate), &registry).is_err());
    }
}
