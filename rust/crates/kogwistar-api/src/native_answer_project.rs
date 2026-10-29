use kogwistar_contracts::{GraphDefaultAclIntentV1, GraphMutationBatchV2};
use kogwistar_runtime::{
    NativeOperationOutput, NativeOperationRegistrationError, NativeOperationRegistry,
    RecordedStateUpdate, RuntimeNativeExecutionError, RuntimeNativeStepRequest,
};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

const AUTHORIZED_RECORDS_KEY: &str = "authorized_pointer_records";
const MAX_USED_NODES: usize = 100;

pub(crate) fn register(
    registry: &mut NativeOperationRegistry,
) -> Result<(), NativeOperationRegistrationError> {
    registry.register_graph_v2("aa_project_pointers", "v1", execute_project_pointers)
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

fn string_ids(state: &Map<String, Value>) -> Result<Vec<String>, RuntimeNativeExecutionError> {
    let Some(value) = state.get("used_node_ids") else {
        return Ok(Vec::new());
    };
    let values = value
        .as_array()
        .ok_or_else(|| invalid("used_node_ids", "expected an array of node IDs"))?;
    if values.len() > MAX_USED_NODES {
        return Err(invalid(
            "used_node_ids",
            "selected node count exceeds the native limit",
        ));
    }
    values
        .iter()
        .map(|value| {
            value
                .as_str()
                .filter(|id| !id.trim().is_empty())
                .map(str::to_owned)
                .ok_or_else(|| invalid("used_node_ids", "expected non-empty string IDs"))
        })
        .collect()
}

fn pointer_id(scope: &str, target_id: &str) -> String {
    crate::native_answer_persist::stable_conversation_id(
        "ptr",
        &[
            ("scope", scope),
            ("pointer_kind", "kg_node"),
            ("target_kind", "node"),
            ("target_id", target_id),
        ],
    )
}

fn usage_edge_id(scope: &str, run_node_id: &str, pointer_node_id: &str) -> String {
    crate::native_answer_persist::stable_conversation_id(
        "e",
        &[
            ("scope", scope),
            ("rel", "used_evidence"),
            ("src", run_node_id),
            ("dst", pointer_node_id),
        ],
    )
}

fn record_map(
    records: &Value,
    key: &str,
) -> Result<BTreeMap<String, Map<String, Value>>, RuntimeNativeExecutionError> {
    let values = records
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| invalid(AUTHORIZED_RECORDS_KEY, "authorized records are missing"))?;
    let mut result = BTreeMap::new();
    for value in values {
        let record = value
            .as_object()
            .ok_or_else(|| invalid(AUTHORIZED_RECORDS_KEY, "record must be an object"))?;
        let id = record
            .get("id")
            .and_then(Value::as_str)
            .filter(|id| !id.trim().is_empty())
            .ok_or_else(|| invalid(AUTHORIZED_RECORDS_KEY, "record ID is missing"))?;
        let metadata = record
            .get("metadata")
            .and_then(Value::as_object)
            .cloned()
            .ok_or_else(|| invalid(AUTHORIZED_RECORDS_KEY, "record metadata is missing"))?;
        if result.insert(id.to_owned(), metadata).is_some() {
            return Err(invalid(AUTHORIZED_RECORDS_KEY, "duplicate record ID"));
        }
    }
    Ok(result)
}

fn snapshot_hash(value: &Value) -> Result<String, RuntimeNativeExecutionError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|_| invalid("authorized_pointer_records", "snapshot is not serializable"))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn record_document_payload(records: &Value, collection: &str, id: &str) -> Option<Value> {
    let record = records
        .get(collection)?
        .as_array()?
        .iter()
        .find(|record| record.get("id").and_then(Value::as_str) == Some(id))?;
    serde_json::from_str(record.get("document")?.as_str()?).ok()
}

fn execute_project_pointers(
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
    let owner = required_string(state, "user_id")?;
    if required_string(state, "_rt_run_owner_id")? != owner {
        return Err(invalid("user_id", "must match the durable run owner"));
    }
    let run_node_id = required_string(state, "run_node_id")?;
    let ids = string_ids(state)?;
    let records = request
        .ephemeral_context
        .get(AUTHORIZED_RECORDS_KEY)
        .filter(|value| value.is_object())
        .ok_or_else(|| {
            invalid(
                AUTHORIZED_RECORDS_KEY,
                "host-authorized knowledge and conversation reads are required",
            )
        })?;
    let knowledge_nodes = record_map(records, "knowledge_nodes")?;
    let conversation_nodes = record_map(records, "conversation_nodes")?;
    let conversation_edges = record_map(records, "conversation_edges")?;
    let selected_ids = ids.iter().cloned().collect::<BTreeSet<_>>();
    if knowledge_nodes.keys().cloned().collect::<BTreeSet<_>>() != selected_ids {
        return Err(invalid(
            AUTHORIZED_RECORDS_KEY,
            "every selected source node must have a current authorized record",
        ));
    }

    let scope_id = format!("conv:{conversation_id}");
    let expected_run_node = conversation_nodes.get(run_node_id).ok_or_else(|| {
        invalid(
            AUTHORIZED_RECORDS_KEY,
            "the active conversation run anchor must be visible",
        )
    })?;
    if expected_run_node.get("entity_type").and_then(Value::as_str) != Some("agent_run")
        || expected_run_node
            .get("conversation_id")
            .and_then(Value::as_str)
            != Some(conversation_id)
    {
        return Err(invalid(
            AUTHORIZED_RECORDS_KEY,
            "run anchor type or conversation scope is invalid",
        ));
    }

    let mut projected_ids = Vec::with_capacity(ids.len());
    let mut mutations = Vec::new();
    let mut index_jobs = Vec::new();
    let mut seen = BTreeSet::new();
    let mut previous = state
        .get("prev_turn_meta_summary")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let mut char_distance = previous
        .get("prev_node_char_distance_from_last_summary")
        .and_then(Value::as_i64)
        .unwrap_or_default();
    let mut node_distance = previous
        .get("prev_node_distance_from_last_summary")
        .and_then(Value::as_i64)
        .unwrap_or_default();
    let tail_turn_index = previous
        .get("tail_turn_index")
        .and_then(Value::as_i64)
        .unwrap_or_default();
    let graph_scope = state
        .get("_rt_graph_scope")
        .filter(|value| value.is_object())
        .ok_or_else(|| invalid("_rt_graph_scope", "trusted graph scope is missing"))?;
    let graph_scope = json!({
        "storage_namespace": graph_scope.get("storage_namespace").cloned().unwrap_or(Value::Null),
        "workspace": graph_scope.get("workspace").cloned().unwrap_or(Value::Null),
        "graph_space": graph_scope.get("graph_space").cloned().unwrap_or(Value::Null)
    });

    for target_id in &ids {
        let pid = pointer_id(&scope_id, target_id);
        projected_ids.push(pid.clone());
        if !seen.insert(target_id.clone()) {
            continue;
        }
        let source_metadata = &knowledge_nodes[target_id];
        if matches!(
            source_metadata
                .get("lifecycle_status")
                .and_then(Value::as_str),
            Some("tombstoned" | "deleted")
        ) {
            return Err(invalid(
                AUTHORIZED_RECORDS_KEY,
                "selected source node is inactive",
            ));
        }
        let pointer_metadata = conversation_nodes.get(&pid);
        if let Some(metadata) = pointer_metadata {
            if metadata.get("entity_type").and_then(Value::as_str) != Some("knowledge_reference")
                || metadata.get("conversation_id").and_then(Value::as_str) != Some(conversation_id)
                || metadata.get("target_id").and_then(Value::as_str) != Some(target_id)
                || matches!(
                    metadata.get("lifecycle_status").and_then(Value::as_str),
                    Some("tombstoned" | "deleted")
                )
            {
                return Err(invalid(
                    AUTHORIZED_RECORDS_KEY,
                    "existing pointer has a conflicting identity or is inactive",
                ));
            }
        } else {
            let label = source_metadata
                .get("label")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .unwrap_or(target_id);
            let summary = source_metadata
                .get("summary")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let snapshot = json!({
                "entity_id": target_id,
                "label": source_metadata.get("label").cloned().unwrap_or(Value::Null),
                "summary": source_metadata.get("summary").cloned().unwrap_or(Value::Null),
                "type": source_metadata.get("type").cloned().unwrap_or(Value::Null),
                "canonical_entity_id": source_metadata.get("canonical_entity_id").cloned().unwrap_or(Value::Null)
            });
            let snapshot_hash = snapshot_hash(&snapshot)?;
            let properties = json!({
                "is_pointer": true,
                "pointer_mode": "live",
                "pointer_kind": "kg_node",
                "target_namespace": "kg",
                "target_kind": "node",
                "target_id": target_id,
                "refers_to_collection": "nodes",
                "refers_to_id": target_id,
                "snapshot_hash": snapshot_hash,
                "entity_type": "knowledge_reference"
            });
            let metadata = json!({
                "entity_type": "knowledge_reference",
                "is_pointer": true,
                "pointer_mode": "live",
                "pointer_kind": "kg_node",
                "target_namespace": "kg",
                "target_kind": "node",
                "target_id": target_id,
                "refers_to_collection": "nodes",
                "refers_to_id": target_id,
                "level_from_root": 0,
                "in_conversation_chain": false,
                "conversation_id": conversation_id,
                "role": "system",
                "lifecycle_status": "active",
                "redirect_to_id": null
            });
            let span = crate::native_answer_persist::span_payload();
            let payload = json!({
                "id": pid,
                "label": format!("Ref {label}"),
                "type": "reference_pointer",
                "summary": summary,
                "doc_id": null,
                "conversation_id": conversation_id,
                "role": "system",
                "turn_index": null,
                "properties": properties,
                "metadata": metadata,
                "mentions": [{"spans": [span], "multimodal_spans": []}],
                "domain_id": null,
                "canonical_entity_id": null,
                "embedding": null
            });
            char_distance = char_distance
                .checked_add(summary.chars().count() as i64)
                .ok_or_else(|| invalid("prev_turn_meta_summary", "character distance overflow"))?;
            node_distance = node_distance
                .checked_add(1)
                .ok_or_else(|| invalid("prev_turn_meta_summary", "node distance overflow"))?;
            let mut document = crate::native_answer_persist::model_document_fields(
                &payload,
                &["embedding", "metadata"],
            );
            document.remove("id");
            let mut metadata = metadata
                .as_object()
                .cloned()
                .ok_or_else(|| invalid("metadata", "pointer metadata is invalid"))?;
            metadata.insert("_class_name".to_owned(), json!("ConversationNode"));
            let (mutation, mut jobs) = crate::native_memory_pin::graph_mutation(
                request,
                &graph_scope,
                &pid,
                format!(
                    "native-aa-project|{}|{}|node|{}",
                    request.run_id, request.node_id, pid
                ),
                "node",
                "conversation_node",
                payload,
                document,
                metadata,
            )?;
            jobs.retain(|job| {
                job.get("index_kind").and_then(Value::as_str) != Some("node_embedding")
            });
            mutations.push(mutation);
            index_jobs.extend(jobs);
        }

        let eid = usage_edge_id(&scope_id, run_node_id, &pid);
        if conversation_edges.contains_key(&eid) {
            let edge_document = record_document_payload(records, "conversation_edges", &eid)
                .ok_or_else(|| {
                    invalid(
                        AUTHORIZED_RECORDS_KEY,
                        "existing usage edge document is missing or malformed",
                    )
                })?;
            if edge_document.get("relation").and_then(Value::as_str) != Some("used_evidence")
                || edge_document.get("source_ids") != Some(&json!([run_node_id]))
                || edge_document.get("target_ids") != Some(&json!([pid]))
            {
                return Err(invalid(
                    AUTHORIZED_RECORDS_KEY,
                    "existing usage edge has conflicting relation or endpoints",
                ));
            }
            continue;
        }
        let edge_metadata = json!({
            "char_distance_from_last_summary": char_distance,
            "turn_distance_from_last_summary": node_distance,
            "tail_turn_index": tail_turn_index,
            "lifecycle_status": "active",
            "redirect_to_id": null
        });
        let edge_payload = json!({
            "id": eid,
            "source_ids": [run_node_id],
            "target_ids": [pid],
            "relation": "used_evidence",
            "label": "used_evidence",
            "type": "relationship",
            "summary": "Agent used this evidence",
            "doc_id": scope_id,
            "mentions": [{"spans": [crate::native_answer_persist::span_payload()], "multimodal_spans": []}],
            "domain_id": null,
            "canonical_entity_id": null,
            "properties": {"entity_type": "conversation_edge"},
            "metadata": edge_metadata,
            "embedding": null,
            "source_edge_ids": [],
            "target_edge_ids": []
        });
        let document =
            crate::native_answer_persist::model_document_fields(&edge_payload, &["embedding"]);
        let mut metadata = edge_metadata
            .as_object()
            .cloned()
            .ok_or_else(|| invalid("metadata", "usage-edge metadata is invalid"))?;
        metadata.insert("_class_name".to_owned(), json!("ConversationEdge"));
        let (mutation, mut jobs) = crate::native_memory_pin::graph_mutation(
            request,
            &graph_scope,
            &eid,
            format!(
                "native-aa-project|{}|{}|edge|{}",
                request.run_id, request.node_id, eid
            ),
            "edge",
            "conversation_edge",
            edge_payload,
            document,
            metadata,
        )?;
        jobs.retain(|job| job.get("index_kind").and_then(Value::as_str) != Some("node_embedding"));
        mutations.push(mutation);
        index_jobs.extend(jobs);
    }

    previous.insert(
        "prev_node_char_distance_from_last_summary".to_owned(),
        json!(char_distance),
    );
    previous.insert(
        "prev_node_distance_from_last_summary".to_owned(),
        json!(node_distance),
    );
    previous
        .entry("tail_turn_index".to_owned())
        .or_insert(json!(tail_turn_index));
    let mut update = Map::new();
    update.insert("projected_pointer_ids".to_owned(), json!(projected_ids));
    update.insert("prev_turn_meta_summary".to_owned(), Value::Object(previous));
    let mut final_state = state.clone();
    final_state.extend(update.clone());
    let mut op_log = state
        .get("op_log")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let log_entry = format!("aa_project_pointers:{}", ids.len());
    op_log.push(json!(log_entry));
    final_state.insert("op_log".to_owned(), Value::Array(op_log));
    let mut auxiliary = Map::new();
    auxiliary.insert("op_log".to_owned(), json!(log_entry));
    let output = NativeOperationOutput {
        state_update: vec![
            RecordedStateUpdate::Pair(("u".to_owned(), update)),
            RecordedStateUpdate::Pair(("a".to_owned(), auxiliary)),
        ],
        result: Some(json!({
            "workflow_status": "succeeded",
            "final_state": final_state,
            "conversation_node_id": null
        })),
    };
    let batch = if mutations.is_empty() {
        None
    } else {
        let batch = GraphMutationBatchV2 {
            contract_version: 2,
            atomic: true,
            mutations,
            index_jobs: serde_json::from_value(Value::Array(index_jobs))
                .map_err(|_| invalid("aa_project_pointers", "index-job intent is invalid"))?,
            default_acl: GraphDefaultAclIntentV1 {
                truth_graph: "conversation".to_owned(),
            },
        };
        batch
            .validate()
            .map_err(|_| invalid("aa_project_pointers", "atomic graph batch is invalid"))?;
        Some(batch)
    };
    Ok((output, batch))
}

#[cfg(test)]
mod tests {
    use super::*;
    use kogwistar_runtime::{RUST_NATIVE_EXECUTOR_TARGET, RuntimeNativeStepRequest};

    fn request(state: Value) -> RuntimeNativeStepRequest {
        let records = json!({
            "knowledge_nodes": [
                {"id":"kg-1", "metadata":{"label":"One", "summary":"Source summary", "type":"entity", "canonical_entity_id":null, "lifecycle_status":"active"}},
                {"id":"kg-2", "metadata":{"label":"Two", "summary":"Other", "type":"entity", "canonical_entity_id":"canon-2", "lifecycle_status":"active"}}
            ],
            "conversation_nodes": [
                {"id":"run-1", "metadata":{"entity_type":"agent_run", "conversation_id":"conv-1"}}
            ],
            "conversation_edges": []
        });
        RuntimeNativeStepRequest {
            executor_target: RUST_NATIVE_EXECUTOR_TARGET.to_owned(),
            operation: "aa_project_pointers".to_owned(),
            op_version: "v1".to_owned(),
            run_id: "answer-run".to_owned(),
            workflow_id: "agentic_answering.v2".to_owned(),
            node_id: "project".to_owned(),
            conversation_id: "conv-1".to_owned(),
            turn_node_id: None,
            token_id: "token-1".to_owned(),
            parent_token_id: None,
            step_seq: 4,
            state: serde_json::from_value(state).unwrap(),
            runtime_routes: Vec::new(),
            ephemeral_context: Map::from_iter([(AUTHORIZED_RECORDS_KEY.to_owned(), records)]),
        }
    }

    fn state(ids: Value) -> Value {
        json!({
            "conversation_id":"conv-1", "user_id":"alice", "_rt_run_owner_id":"alice",
            "run_node_id":"run-1", "used_node_ids":ids,
            "_rt_graph_scope":{"storage_namespace":"ns-a"},
            "prev_turn_meta_summary":{"prev_node_char_distance_from_last_summary":2,"prev_node_distance_from_last_summary":1,"tail_turn_index":3}
        })
    }

    #[test]
    fn projects_unique_pointers_and_usage_edges_as_one_atomic_batch() {
        assert_eq!(
            pointer_id("conv:conv-1", "kg-1"),
            "ptr:mwzywyej5irufzk5gyvq"
        );
        assert_eq!(
            usage_edge_id("conv:conv-1", "run-1", "ptr:mwzywyej5irufzk5gyvq"),
            "e:3gfnmgmsa7zieg5lb4xq"
        );
        let request = request(state(json!(["kg-1", "kg-2", "kg-1"])));
        let (output, batch) = execute_project_pointers(&request).unwrap();
        let batch = batch.unwrap();
        assert!(batch.atomic);
        assert_eq!(batch.default_acl.truth_graph, "conversation");
        assert_eq!(batch.mutations.len(), 4);
        assert_eq!(batch.index_jobs.len(), 8);
        assert!(
            batch
                .index_jobs
                .iter()
                .all(|job| job.index_kind != "node_embedding")
        );
        assert_eq!(
            output.result.as_ref().unwrap()["final_state"]["projected_pointer_ids"],
            json!([
                pointer_id("conv:conv-1", "kg-1"),
                pointer_id("conv:conv-1", "kg-2"),
                pointer_id("conv:conv-1", "kg-1")
            ])
        );
        let final_state = output.result.unwrap()["final_state"].clone();
        assert_eq!(
            final_state["prev_turn_meta_summary"],
            json!({"prev_node_char_distance_from_last_summary":21,"prev_node_distance_from_last_summary":3,"tail_turn_index":3})
        );
    }

    #[test]
    fn existing_pointer_is_not_rewritten_but_missing_usage_edge_is_repaired() {
        let mut request = request(state(json!(["kg-1"])));
        request.ephemeral_context[AUTHORIZED_RECORDS_KEY]["knowledge_nodes"] = json!([
            {"id":"kg-1", "metadata":{"label":"One", "summary":"Source summary", "type":"entity", "canonical_entity_id":null, "lifecycle_status":"active"}}
        ]);
        let pid = pointer_id("conv:conv-1", "kg-1");
        request.ephemeral_context[AUTHORIZED_RECORDS_KEY]["conversation_nodes"] = json!([
            {"id":"run-1", "metadata":{"entity_type":"agent_run", "conversation_id":"conv-1"}},
            {"id":pid, "metadata":{"entity_type":"knowledge_reference", "conversation_id":"conv-1", "target_id":"kg-1", "lifecycle_status":"active"}}
        ]);
        let (_, batch) = execute_project_pointers(&request).unwrap();
        let batch = batch.unwrap();
        assert_eq!(batch.mutations.len(), 1);
        assert_eq!(batch.mutations[0].entity_kind, "edge");
    }

    #[test]
    fn existing_pointer_and_usage_edge_produce_no_graph_mutations() {
        let mut request = request(state(json!(["kg-1"])));
        let pid = pointer_id("conv:conv-1", "kg-1");
        let eid = usage_edge_id("conv:conv-1", "run-1", &pid);
        request.ephemeral_context[AUTHORIZED_RECORDS_KEY]["knowledge_nodes"] = json!([
            {"id":"kg-1", "metadata":{"label":"One", "summary":"Source summary", "type":"entity", "canonical_entity_id":null, "lifecycle_status":"active"}}
        ]);
        request.ephemeral_context[AUTHORIZED_RECORDS_KEY]["conversation_nodes"] = json!([
            {"id":"run-1", "metadata":{"entity_type":"agent_run", "conversation_id":"conv-1"}},
            {"id":pid, "metadata":{"entity_type":"knowledge_reference", "conversation_id":"conv-1", "target_id":"kg-1", "lifecycle_status":"active"}}
        ]);
        request.ephemeral_context[AUTHORIZED_RECORDS_KEY]["conversation_edges"] = json!([
            {"id":eid, "metadata":{}, "document":serde_json::to_string(&json!({
                "relation":"used_evidence", "source_ids":["run-1"], "target_ids":[pid]
            })).unwrap()}
        ]);
        let (_, batch) = execute_project_pointers(&request).unwrap();
        assert!(batch.is_none());
    }

    #[test]
    fn rejects_unowned_or_unreadable_projection_context() {
        let mut wrong_owner = state(json!(["kg-1"]));
        wrong_owner["_rt_run_owner_id"] = json!("mallory");
        assert!(execute_project_pointers(&request(wrong_owner)).is_err());

        let mut missing_source = request(state(json!(["kg-404"])));
        assert!(execute_project_pointers(&missing_source).is_err());
        missing_source.ephemeral_context[AUTHORIZED_RECORDS_KEY]["knowledge_nodes"] = json!([]);
        assert!(execute_project_pointers(&missing_source).is_err());
    }
}
