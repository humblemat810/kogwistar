use kogwistar_contracts::{DurableProviderEventPublicationV1, DurableProviderEventV1};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ProviderEventStepSnapshot {
    pub step_seq: i64,
    pub workflow_id: String,
    pub workflow_node_id: String,
    pub status: String,
    pub result_json: String,
    pub event_seq: i64,
    pub transition_id: String,
}

pub(crate) fn provider_event_step_snapshots(
    events: Vec<ServerRunEvent>,
    run_id: &str,
    conversation_id: &str,
    workflow_id: &str,
    requested_step_seqs: &[i64],
) -> Result<BTreeMap<i64, ProviderEventStepSnapshot>, String> {
    let requested = requested_step_seqs.iter().copied().collect::<BTreeSet<_>>();
    let mut snapshots = BTreeMap::new();
    for event in events {
        if event.event_type != "workflow.recorded_transition.v1" {
            return Err("workflow-step evidence came from a non-transition event".to_owned());
        }
        let payload: Value = serde_json::from_str(&event.payload_json)
            .map_err(|_| "workflow-step evidence has malformed transition JSON".to_owned())?;
        let reduced = payload
            .get("reduced")
            .ok_or_else(|| "workflow-step evidence has no reduced transition".to_owned())?;
        let state = reduced
            .get("state")
            .ok_or_else(|| "workflow-step evidence has no runtime state".to_owned())?;
        let step_seq = state
            .get("last_step_seq")
            .and_then(Value::as_i64)
            .ok_or_else(|| "workflow-step evidence has no valid step sequence".to_owned())?;
        if !requested.contains(&step_seq) {
            continue;
        }
        if state.get("run_id").and_then(Value::as_str) != Some(run_id)
            || state.get("conversation_id").and_then(Value::as_str) != Some(conversation_id)
            || state.get("workflow_id").and_then(Value::as_str) != Some(workflow_id)
        {
            return Err("workflow-step evidence is outside the active run scope".to_owned());
        }
        let workflow_node_id = state
            .get("last_node_id")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| "workflow-step evidence has no workflow node identity".to_owned())?
            .to_owned();
        let errors = reduced
            .get("errors")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let result_json = serde_json::to_string(
            reduced.get("result").unwrap_or(&Value::Null),
        )
        .map_err(|_| "workflow-step result could not be serialized".to_owned())?;
        let transition_id = payload
            .get("transition_id")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| "workflow-step evidence has no transition identity".to_owned())?
            .to_owned();
        let snapshot = ProviderEventStepSnapshot {
            step_seq,
            workflow_id: workflow_id.to_owned(),
            workflow_node_id,
            status: if errors.is_empty() { "success" } else { "failure" }.to_owned(),
            result_json,
            event_seq: event.seq,
            transition_id,
        };
        if let Some(previous) = snapshots.insert(step_seq, snapshot.clone())
            && previous != snapshot
        {
            return Err("workflow step sequence resolves to conflicting transitions".to_owned());
        }
    }
    if requested.iter().any(|step_seq| !snapshots.contains_key(step_seq)) {
        return Err("a referenced provider-event workflow step is not durable".to_owned());
    }
    Ok(snapshots)
}

pub(crate) fn provider_event_step_link_mutations(
    event_id: &str,
    run_id: &str,
    conversation_id: &str,
    producer_step_seq: Option<i64>,
    handler_step_seq: i64,
    snapshots: &BTreeMap<i64, ProviderEventStepSnapshot>,
    scope: &GraphScope,
) -> Vec<GraphMutation> {
    let mut mutations = Vec::new();
    for snapshot in snapshots.values() {
        let id = format!("wf_step|{run_id}|{}", snapshot.step_seq);
        let summary = format!(
            "step={}, op={}, status={}",
            snapshot.step_seq, snapshot.workflow_node_id, snapshot.status
        );
        let metadata = json!({
            "entity_type": "workflow_step_exec",
            "run_id": run_id,
            "conversation_id": conversation_id,
            "workflow_id": snapshot.workflow_id,
            "workflow_node_id": snapshot.workflow_node_id,
            "step_seq": snapshot.step_seq,
            "op": snapshot.workflow_node_id,
            "status": snapshot.status,
            "duration_ms": 0,
            "result_json": snapshot.result_json,
            "event_seq": snapshot.event_seq,
            "transition_id": snapshot.transition_id,
            "namespace": scope.namespace,
            "workspace_id": scope.workspace_id,
            "graph_space": scope.graph_space,
            "in_conversation_chain": false
        });
        mutations.push(provider_event_node_mutation(
            scope,
            id.clone(),
            format!("WF step {}: {}", snapshot.step_seq, snapshot.workflow_node_id),
            summary,
            metadata.as_object().cloned().unwrap_or_default(),
            None,
        ));
    }

    let message_id = format!("msg:{event_id}");
    let mut links = Vec::new();
    if let Some(producer_step_seq) = producer_step_seq {
        links.push(("emitted_by_step", producer_step_seq));
    }
    links.push(("handled_by_step", handler_step_seq));
    for (relation, step_seq) in links {
        let step_id = format!("wf_step|{run_id}|{step_seq}");
        let edge_id = stable_id(
            "lane_message_step_link",
            &[
                message_id.clone(),
                relation.to_owned(),
                step_id.clone(),
            ],
        )
        .to_string();
        let metadata = json!({
            "entity_type": "conversation_edge",
            "relation": relation,
            "conversation_id": conversation_id,
            "run_id": run_id,
            "source_id": [message_id],
            "target_id": [step_id],
            "char_distance_from_last_summary": 0,
            "turn_distance_from_last_summary": 0,
            "namespace": scope.namespace,
            "workspace_id": scope.workspace_id,
            "graph_space": scope.graph_space,
            "in_conversation_chain": false
        });
        let payload = json!({
            "id": edge_id,
            "label": relation,
            "type": "relationship",
            "summary": relation,
            "doc_id": edge_id,
            "relation": relation,
            "source_ids": [message_id],
            "target_ids": [step_id],
            "source_edge_ids": [],
            "target_edge_ids": [],
            "metadata": metadata,
            "properties": {},
            "embedding": null
        });
        mutations.push(GraphMutation {
            scope: scope.clone(),
            table: "gke_edges".to_owned(),
            entity_kind: "edge".to_owned(),
            event_id: stable_id(
                "agent_provider_event_graph_mutation",
                &["edge".to_owned(), edge_id.clone()],
            )
            .to_string(),
            op: "ADD".to_owned(),
            record: GraphRecord {
                id: edge_id,
                document: Some(payload.to_string()),
                metadata: payload["metadata"].as_object().cloned().unwrap_or_default(),
                embedding: None,
            },
            payload,
            embedding_dim: 0,
        });
    }
    mutations
}

fn provider_event_graph_mutations(
    publication: &DurableProviderEventPublicationV1,
    scope: &GraphScope,
    security_scope: &str,
    created_at: i64,
) -> Vec<GraphMutation> {
    let event = &publication.event;
    let message_id = format!("msg:{}", event.event_id);
    let inbox_id = "agent-provider-events";
    let sender_id = "agent-runtime";
    let recipient_id = "agent-event-consumer";
    let conversation_anchor = stable_id(
        "lane_message_conversation",
        &[publication.conversation_id.clone()],
    )
    .to_string();
    let inbox_anchor = stable_id("lane_message_inbox", &[inbox_id.to_owned()]).to_string();
    let sender_anchor = stable_id("lane_message_anchor", &[sender_id.to_owned()]).to_string();
    let recipient_anchor =
        stable_id("lane_message_anchor", &[recipient_id.to_owned()]).to_string();
    let mut mutations = Vec::new();

    for (id, label, summary, metadata) in [
        (
            conversation_anchor.clone(),
            format!("lane_conversation:{}", publication.conversation_id),
            format!(
                "Lane-messaging conversation anchor for {}",
                publication.conversation_id
            ),
            json!({
                "artifact_kind": "lane_conversation",
                "conversation_id": publication.conversation_id,
                "kind": "lane_conversation",
                "in_conversation_chain": false
            }),
        ),
        (
            inbox_anchor.clone(),
            format!("lane_inbox:{inbox_id}"),
            format!("Lane inbox anchor for {inbox_id}"),
            json!({
                "artifact_kind": "lane_inbox",
                "inbox_id": inbox_id,
                "kind": "lane_inbox",
                "in_conversation_chain": false
            }),
        ),
        (
            sender_anchor.clone(),
            format!("lane_anchor:{sender_id}"),
            format!("Lane sender anchor for {sender_id}"),
            json!({
                "artifact_kind": "lane_anchor",
                "anchor_id": sender_id,
                "kind": "lane_anchor",
                "in_conversation_chain": false
            }),
        ),
        (
            recipient_anchor.clone(),
            format!("lane_anchor:{recipient_id}"),
            format!("Lane recipient anchor for {recipient_id}"),
            json!({
                "artifact_kind": "lane_anchor",
                "anchor_id": recipient_id,
                "kind": "lane_anchor",
                "in_conversation_chain": false
            }),
        ),
    ] {
        mutations.push(provider_event_node_mutation(
            scope,
            id,
            label,
            summary,
            metadata.as_object().cloned().unwrap_or_default(),
            None,
        ));
    }

    let event_payload = serde_json::to_string(event).expect("provider event is serializable");
    let message_summary = format!(
        "Lane message agent.provider_event from {sender_id} to {recipient_id}"
    );
    let mut message_metadata = json!({
        "artifact_kind": "lane_message",
        "conversation_id": publication.conversation_id,
        "inbox_id": inbox_id,
        "sender_id": sender_id,
        "recipient_id": recipient_id,
        "msg_type": "agent.provider_event",
        "status": "pending",
        "priority": 0,
        "correlation_id": event.event_id,
        "reply_to_message_id": null,
        "run_id": event.trace.run_id,
        "step_id": event.trace.step_id,
        "security_scope": security_scope,
        "shared_scope": false,
        "shared_inbox": false,
        "visibility": "private",
        "purpose": "agent_event",
        "payload_json": event_payload,
        "idempotency_key": event.event_id,
        "created_at": created_at,
        "updated_at": created_at,
        "completed_at": null,
        "namespace": publication.namespace,
        "kind": "lane_message"
    });
    mutations.push(provider_event_node_mutation(
        scope,
        message_id.clone(),
        "lane_message:agent.provider_event".to_owned(),
        message_summary,
        message_metadata.as_object_mut().cloned().unwrap_or_default(),
        None,
    ));

    for (relation, target_id) in [
        ("in_conversation", conversation_anchor),
        ("in_inbox", inbox_anchor),
        ("sent_by", sender_anchor),
        ("sent_to", recipient_anchor),
    ] {
        let edge_id = stable_id(
            "lane_message_edge",
            &[message_id.clone(), relation.to_owned(), target_id.clone()],
        )
        .to_string();
        let payload = json!({
            "id": edge_id,
            "label": format!("lane_message_edge:{relation}"),
            "type": "relationship",
            "summary": format!("Lane message semantic edge {relation}"),
            "doc_id": edge_id,
            "relation": relation,
            "source_ids": [message_id],
            "target_ids": [target_id],
            "source_edge_ids": [],
            "target_edge_ids": [],
            "metadata": {
                "artifact_kind": "lane_message_edge",
                "relation_kind": relation,
                "conversation_id": publication.conversation_id
            },
            "properties": {},
            "embedding": null
        });
        let metadata = payload["metadata"].as_object().cloned().unwrap_or_default();
        mutations.push(GraphMutation {
            scope: scope.clone(),
            table: "gke_edges".to_owned(),
            entity_kind: "edge".to_owned(),
            event_id: stable_id(
                "agent_provider_event_graph_mutation",
                &[event.event_id.clone(), "edge".to_owned(), edge_id.clone()],
            )
            .to_string(),
            op: "ADD".to_owned(),
            payload: payload.clone(),
            record: GraphRecord {
                id: edge_id,
                document: Some(payload.to_string()),
                metadata,
                embedding: None,
            },
            embedding_dim: 0,
        });
    }
    mutations
}

fn provider_event_node_mutation(
    scope: &GraphScope,
    id: String,
    label: String,
    summary: String,
    metadata: Map<String, Value>,
    document: Option<String>,
) -> GraphMutation {
    let metadata_value = Value::Object(metadata.clone());
    let payload = json!({
        "id": id,
        "label": label,
        "type": "entity",
        "summary": summary,
        "doc_id": id,
        "metadata": metadata_value,
        "properties": {},
        "embedding": null
    });
    GraphMutation {
        scope: scope.clone(),
        table: "gke_nodes".to_owned(),
        entity_kind: "node".to_owned(),
        event_id: stable_id(
            "agent_provider_event_graph_mutation",
            &["node".to_owned(), id.clone()],
        )
        .to_string(),
        op: "ADD".to_owned(),
        payload,
        record: GraphRecord {
            id,
            document: document.or(Some(summary)),
            metadata,
            embedding: None,
        },
        embedding_dim: 0,
    }
}

fn provider_event_ack_graph_mutations(
    event_id: &str,
    run_id: &str,
    conversation_id: &str,
    workflow_id: &str,
    workflow_node_id: &str,
    step_seq: i64,
    effect_id: &str,
    scope: &GraphScope,
) -> Vec<GraphMutation> {
    let step_node_id = format!("wf_step|{run_id}|{step_seq}");
    let step_metadata = json!({
        "entity_type": "workflow_step_exec",
        "doc_id": step_node_id,
        "run_id": run_id,
        "conversation_id": conversation_id,
        "workflow_id": workflow_id,
        "workflow_node_id": workflow_node_id,
        "step_seq": step_seq,
        "op": workflow_node_id,
        "status": "ok",
        "duration_ms": 0,
        "result_json": json!({"event_id": event_id, "handled": true}).to_string(),
        "effect_id": effect_id,
        "namespace": scope.namespace,
        "workspace_id": scope.workspace_id,
        "graph_space": scope.graph_space,
        "in_conversation_chain": false
    });
    let mut mutations = vec![provider_event_node_mutation(
        scope,
        step_node_id.clone(),
        format!("WF step {step_seq}: {workflow_node_id}"),
        format!("step={step_seq}, op={workflow_node_id}, status=ok"),
        step_metadata.as_object().cloned().unwrap_or_default(),
        None,
    )];

    let message_id = format!("msg:{event_id}");
    let relation = "acknowledged_by_step";
    let edge_id = stable_id(
        "lane_message_step_link",
        &[message_id.clone(), relation.to_owned(), step_node_id.clone()],
    )
    .to_string();
    let metadata = json!({
        "entity_type": "conversation_edge",
        "relation": relation,
        "conversation_id": conversation_id,
        "run_id": run_id,
        "event_id": event_id,
        "step_seq": step_seq,
        "workflow_id": workflow_id,
        "workflow_node_id": workflow_node_id,
        "source_id": [message_id],
        "target_id": [step_node_id],
        "char_distance_from_last_summary": 0,
        "turn_distance_from_last_summary": 0,
        "effect_id": effect_id,
        "namespace": scope.namespace,
        "workspace_id": scope.workspace_id,
        "graph_space": scope.graph_space,
        "in_conversation_chain": false
    });
    let payload = json!({
        "id": edge_id,
        "label": relation,
        "type": "relationship",
        "summary": relation,
        "doc_id": edge_id,
        "relation": relation,
        "source_ids": [message_id],
        "target_ids": [step_node_id],
        "source_edge_ids": [],
        "target_edge_ids": [],
        "metadata": metadata,
        "properties": {},
        "embedding": null
    });
    mutations.push(GraphMutation {
        scope: scope.clone(),
        table: "gke_edges".to_owned(),
        entity_kind: "edge".to_owned(),
        event_id: stable_id(
            "agent_provider_event_handled_graph_mutation",
            &[event_id.to_owned(), "edge".to_owned(), edge_id.clone()],
        )
        .to_string(),
        op: "ADD".to_owned(),
        record: GraphRecord {
            id: edge_id,
            document: Some(payload.to_string()),
            metadata: payload["metadata"].as_object().cloned().unwrap_or_default(),
            embedding: None,
        },
        payload,
        embedding_dim: 0,
    });
    mutations
}

fn provider_event_payloads_match(left: &str, right: &str) -> bool {
    let (Ok(mut left), Ok(mut right)) = (
        serde_json::from_str::<Value>(left),
        serde_json::from_str::<Value>(right),
    ) else {
        return false;
    };
    for value in [&mut left, &mut right] {
        let Some(trace) = value.get_mut("trace").and_then(Value::as_object_mut) else {
            return false;
        };
        for key in [
            "producer_step_seq",
            "producer_attempt",
            "trace_id",
            "span_id",
            "parent_span_id",
        ] {
            trace.remove(key);
        }
    }
    left == right
}

impl SqliteRunApplicationService {
    async fn claim_native_provider_event_context(
        &self,
        request: &kogwistar_runtime::RuntimeNativeStepRequest,
        principal: &Value,
    ) -> Result<Option<Value>, String> {
        let security_scope = security_scope_value(principal);
        let scope = GraphScope {
            namespace: security_scope["storage_namespace"]
                .as_str()
                .ok_or_else(|| "provider-event owner has no storage namespace".to_owned())?
                .to_owned(),
            workspace_id: principal
                .get("workspace_id")
                .or_else(|| principal.get("workspace"))
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .map(str::to_owned),
            graph_space: Some("conversation".to_owned()),
        };
        let owner_id = authenticated_run_owner_id(principal)
            .ok_or_else(|| "provider-event receive requires an authenticated run owner".to_owned())?
            .to_owned();
        let acl_principal = graph_acl_principal_from_claims(
            principal,
            security_scope["security_scope"].as_str(),
        );
        let request = request.clone();
        let claim_owner = format!("rust-native-agent-event:{}", uuid::Uuid::new_v4());
        let outcome = self.store.immediate_transaction(|uow| {
            let run = uow
                .get_server_run(&request.run_id)?
                .ok_or_else(|| SqliteStoreError::RecordedRuntimeConflict(
                    "provider-event run is missing".to_owned(),
                ))?;
            if run.user_id.as_deref() != Some(owner_id.as_str())
                || run.conversation_id != request.conversation_id
            {
                return Err(SqliteStoreError::RecordedRuntimeConflict(
                    "provider-event receive run owner or conversation changed".to_owned(),
                ));
            }
            let candidate_lanes = uow.list_projected_lane_messages(&LaneMessageFilter {
                namespace: Some(scope.namespace.clone()),
                purpose: Some("agent_event".to_owned()),
                inbox_id: Some("agent-provider-events".to_owned()),
                conversation_id: Some(request.conversation_id.clone()),
                msg_type: Some("agent.provider_event".to_owned()),
                recipient_id: Some("agent-event-consumer".to_owned()),
                limit: 20_001,
                ..LaneMessageFilter::default()
            })?;
            if candidate_lanes.len() > 20_000 {
                return Err(SqliteStoreError::TransactionAborted(
                    "provider-event receive exceeded its 20000-candidate safety bound".to_owned(),
                ));
            }
            let matching_lanes = candidate_lanes
                .into_iter()
                .filter(|lane| lane.run_id.as_deref() == Some(request.run_id.as_str()))
                .collect::<Vec<_>>();
            let message_ids = matching_lanes
                .iter()
                .map(|lane| lane.message_id.clone())
                .collect::<Vec<_>>();
            if message_ids.is_empty() {
                return Ok(None);
            }
            let acl_records = uow.graph_acl_records_for_entities(
                &scope,
                "conversation",
                &message_ids,
            )?;
            let mut acl_index = GraphAclIndex::default();
            acl_index.extend_from_records(&acl_records, "conversation");
            let candidate_nodes = matching_lanes
                .iter()
                .map(|lane| GraphRecord {
                    id: lane.message_id.clone(),
                    document: None,
                    metadata: BTreeMap::from([
                        ("entity_type".to_owned(), json!("lane_message")),
                        ("mentions".to_owned(), json!([])),
                    ])
                    .into_iter()
                    .collect(),
                    embedding: None,
                })
                .collect::<Vec<_>>();
            let (visible_nodes, _) = graph_records_visible_by_acl_with_index(
                &candidate_nodes,
                &[],
                &acl_index,
                &acl_principal,
            );
            let visible_ids = visible_nodes
                .into_iter()
                .map(|node| node.id)
                .collect::<Vec<_>>();
            if visible_ids.is_empty() {
                return Ok(None);
            }
            let mut claimed = uow.claim_projected_lane_messages_filtered(LaneMessageClaimFilter {
                namespace: scope.namespace.clone(),
                inbox_id: "agent-provider-events".to_owned(),
                claimed_by: claim_owner.clone(),
                message_ids: Some(visible_ids),
                run_id: Some(request.run_id.clone()),
                msg_type: Some("agent.provider_event".to_owned()),
                recipient_id: Some("agent-event-consumer".to_owned()),
                limit: 1,
                lease_seconds: 60,
            })?;
            let Some(lane) = claimed.pop() else {
                return Ok(None);
            };
            let event: kogwistar_contracts::DurableProviderEventV1 = serde_json::from_str(
                lane.payload_json.as_deref().unwrap_or(""),
            )
            .map_err(|_| {
                SqliteStoreError::RecordedRuntimeConflict(
                    "claimed provider event payload is invalid".to_owned(),
                )
            })?;
            if lane.namespace != scope.namespace
                || lane.purpose != "agent_event"
                || lane.inbox_id != "agent-provider-events"
                || lane.conversation_id != request.conversation_id
                || lane.recipient_id != "agent-event-consumer"
                || lane.sender_id != "agent-runtime"
                || lane.msg_type != "agent.provider_event"
                || lane.status != "claimed"
                || lane.claimed_by.as_deref() != Some(claim_owner.as_str())
                || lane.run_id.as_deref() != Some(request.run_id.as_str())
                || lane.correlation_id.as_deref() != Some(event.event_id.as_str())
                || lane.message_id != format!("msg:{}", event.event_id)
                || event.trace.run_id != request.run_id
            {
                return Err(SqliteStoreError::RecordedRuntimeConflict(
                    "claimed provider event does not match its authorized run and lane".to_owned(),
                ));
            }
            Ok(Some(json!({
                "event": event,
                "claim": {
                    "message_id": lane.message_id,
                    "claimed_by": claim_owner,
                    "event_id": event.event_id,
                }
            })))
        });
        outcome.map_err(|_| "provider-event claim failed closed".to_owned())
    }
}

impl PostgresRunApplicationService {
    async fn claim_native_provider_event_context(
        &self,
        request: &kogwistar_runtime::RuntimeNativeStepRequest,
        principal: &Value,
    ) -> Result<Option<Value>, String> {
        let security_scope = security_scope_value(principal);
        let scope = GraphScope {
            namespace: security_scope["storage_namespace"]
                .as_str()
                .ok_or_else(|| "provider-event owner has no storage namespace".to_owned())?
                .to_owned(),
            workspace_id: principal
                .get("workspace_id")
                .or_else(|| principal.get("workspace"))
                .and_then(Value::as_str)
                .map(str::to_owned),
            graph_space: Some("conversation".to_owned()),
        };
        let acl_principal = graph_acl_principal_from_claims(
            principal,
            security_scope["security_scope"].as_str(),
        );
        let conversation_id = request.conversation_id.clone();
        let run_id = request.run_id.clone();
        let claim_owner = format!("rust-native-agent-event:{}", uuid::Uuid::new_v4());
        self.store
            .transaction(move |uow| {
                Box::pin(async move {
                    let metadata = BTreeMap::from([
                        ("conversation_id".to_owned(), json!(conversation_id)),
                        ("inbox_id".to_owned(), json!("agent-provider-events")),
                        ("msg_type".to_owned(), json!("agent.provider_event")),
                        ("recipient_id".to_owned(), json!("agent-event-consumer")),
                        ("run_id".to_owned(), json!(run_id)),
                    ]);
                    let event_nodes = uow
                        .graph_projection_records(GraphProjectionRead {
                            scope: scope.clone(),
                            table: "gke_nodes".to_owned(),
                            ids: None,
                            metadata: MetadataFilter { equals: metadata },
                            limit: 20_001,
                        })
                        .await?;
                    if event_nodes.len() > 20_000 {
                        return Err(
                            kogwistar_store_postgres::PostgresStoreError::Backend(
                                "provider-event receive exceeded its 20000-candidate safety bound"
                                    .to_owned(),
                            ),
                        );
                    }
                    let event_ids = event_nodes
                        .iter()
                        .map(|record| record.id.clone())
                        .collect::<Vec<_>>();
                    if event_ids.is_empty() {
                        return Ok(None);
                    }
                    let mut acl_index = GraphAclIndex::default();
                    for ids in event_ids.chunks(10_000) {
                        let acl_records = uow
                            .graph_acl_records_for_entities_for_update(
                                &scope,
                                "conversation",
                                ids,
                            )
                            .await?;
                        acl_index.extend_from_records(&acl_records, "conversation");
                    }
                    let (visible_nodes, _) = graph_records_visible_by_acl_with_index(
                        &event_nodes,
                        &[],
                        &acl_index,
                        &acl_principal,
                    );
                    let visible_ids = visible_nodes
                        .iter()
                        .map(|record| record.id.clone())
                        .collect::<Vec<_>>();
                    if visible_ids.is_empty() {
                        return Ok(None);
                    }
                    let claimed = uow
                        .claim_projected_lane_messages_filtered(LaneMessageClaimFilter {
                            namespace: scope.namespace.clone(),
                            inbox_id: "agent-provider-events".to_owned(),
                            claimed_by: claim_owner.clone(),
                            message_ids: Some(visible_ids),
                            run_id: Some(run_id.clone()),
                            msg_type: Some("agent.provider_event".to_owned()),
                            recipient_id: Some("agent-event-consumer".to_owned()),
                            limit: 1,
                            lease_seconds: 60,
                        })
                        .await?;
                    let Some(lane) = claimed.into_iter().next() else {
                        return Ok(None);
                    };
                    let event: kogwistar_contracts::DurableProviderEventV1 =
                        serde_json::from_str(lane.payload_json.as_deref().unwrap_or(""))
                            .map_err(|_| {
                                kogwistar_store_postgres::PostgresStoreError::RecordedRuntimeConflict(
                                    "claimed provider event payload is invalid".to_owned(),
                                )
                            })?;
                    if lane.namespace != scope.namespace
                        || lane.purpose != "agent_event"
                        || lane.inbox_id != "agent-provider-events"
                        || lane.conversation_id != conversation_id
                        || lane.recipient_id != "agent-event-consumer"
                        || lane.sender_id != "agent-runtime"
                        || lane.msg_type != "agent.provider_event"
                        || lane.status != "claimed"
                        || lane.claimed_by.as_deref() != Some(claim_owner.as_str())
                        || lane.run_id.as_deref() != Some(run_id.as_str())
                        || lane.correlation_id.as_deref() != Some(event.event_id.as_str())
                        || lane.message_id != format!("msg:{}", event.event_id)
                        || event.trace.run_id != run_id
                    {
                        return Err(
                            kogwistar_store_postgres::PostgresStoreError::RecordedRuntimeConflict(
                                "claimed provider event does not match its authorized run and lane".to_owned(),
                            ),
                        );
                    }
                    Ok(Some(json!({
                        "event": event,
                        "claim": {
                            "message_id": lane.message_id,
                            "claimed_by": claim_owner,
                            "event_id": event.event_id,
                        }
                    })))
                })
            })
            .await
            .map_err(|_| "provider-event claim failed closed".to_owned())
    }

    async fn persist_native_provider_event(
        &self,
        event: DurableProviderEventV1,
        conversation_id: String,
        trace_payload: Value,
    ) -> Result<(), String> {
        let run_id = event.trace.run_id.clone();
        let principal = self
            .resolve_inline_native_owner_principal(&run_id, "write_graph")
            .await?;
        if !capability_allowed_for_principal(
            &principal,
            &self.capabilities,
            "write_graph.provider_event",
            &["write_graph"],
        ) || !principal_strings(&principal, "capabilities")
            .iter()
            .any(|capability| capability == "write_graph")
        {
            return Err("provider event publication requires the run owner's current write_graph capability".to_owned());
        }
        if !self
            .store
            .graph_authority_capability()
            .supports_atomic_index_jobs()
        {
            return Err("provider event publication requires atomic graph authority".to_owned());
        }
        let security_scope = security_scope_value(&principal);
        let scope = GraphScope {
            namespace: security_scope["storage_namespace"]
                .as_str()
                .ok_or_else(|| "provider event owner has no storage namespace".to_owned())?
                .to_owned(),
            workspace_id: security_scope["workspace"]
                .as_str()
                .filter(|value| !value.is_empty())
                .map(str::to_owned),
            graph_space: Some("conversation".to_owned()),
        };
        let publication = DurableProviderEventPublicationV1 {
            namespace: scope.namespace.clone(),
            conversation_id: conversation_id.clone(),
            event,
        };
        let mutations = provider_event_graph_mutations(
            &publication,
            &scope,
            security_scope["security_scope"].as_str().unwrap_or_default(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|_| "system clock is before Unix epoch".to_owned())?
                .as_secs() as i64,
        );
        validate_provider_event_publications(
            std::slice::from_ref(&publication),
            &mutations,
            &run_id,
        )?;
        if mutations
            .iter()
            .any(|mutation| !graph_mutation_scope_allowed(&mutation.scope, &security_scope))
        {
            return Err("provider event publication is outside the run owner's admitted scope".to_owned());
        }

        let owner_id = authenticated_run_owner_id(&principal)
            .ok_or_else(|| "provider event publication requires an authenticated run owner".to_owned())?
            .to_owned();
        let admission = self
            .store
            .list_server_run_events(&run_id, 0, 1)
            .await
            .map_err(|_| "provider event run admission binding could not be read".to_owned())?
            .into_iter()
            .find(|event| event.event_type == "run.created")
            .ok_or_else(|| "provider event run admission binding is missing".to_owned())?;
        let admission_payload: Value = serde_json::from_str(&admission.payload_json)
            .map_err(|_| "provider event run admission binding is invalid".to_owned())?;
        let binding: RuntimeOwnerBindingV1 = serde_json::from_value(
            admission_payload
                .get("runtime_owner_binding")
                .cloned()
                .ok_or_else(|| "provider event trusted owner binding is missing".to_owned())?,
        )
        .map_err(|_| "provider event trusted owner binding is invalid".to_owned())?;
        if binding.user_id != owner_id {
            return Err("provider event trusted owner binding does not match the run owner".to_owned());
        }
        let message_id = format!("msg:{}", publication.event.event_id);
        let event_payload = serde_json::to_string(&publication.event)
            .map_err(|_| "provider event could not be serialized".to_owned())?;
        let event_type = "workflow.provider_event.v1".to_owned();
        let run_event_payload = serde_json::to_string(&trace_payload)
            .map_err(|_| "provider event trace could not be serialized".to_owned())?;
        let capabilities = self.capabilities.clone();
        let auth_store = self.owner_auth_store.clone().ok_or_else(|| {
            "provider event owner revalidation is not configured".to_owned()
        })?;
        let publication_for_commit = publication.clone();
        let commit_owner_id = owner_id.clone();
        self.store
            .transaction(move |uow| {
                let mutations = mutations.clone();
                let publication = publication_for_commit.clone();
                let run_id = run_id.clone();
                let conversation_id = conversation_id.clone();
                let binding = binding.clone();
                let capabilities = capabilities.clone();
                let auth_store = auth_store.clone();
                let scope = scope.clone();
                let owner_id = commit_owner_id.clone();
                let message_id = message_id.clone();
                let event_payload = event_payload.clone();
                let event_type = event_type.clone();
                let run_event_payload = run_event_payload.clone();
                Box::pin(async move {
                    uow.lock_entity_identity(&scope.namespace, "node", &message_id)
                        .await?;
                    let run = uow.get_server_run(&run_id).await?;
                    if !run.as_ref().is_some_and(|run| {
                        run.user_id.as_deref() == Some(owner_id.as_str())
                            && run.conversation_id == conversation_id
                    }) {
                        return Err(kogwistar_store_postgres::PostgresStoreError::RecordedRuntimeConflict(
                            "provider event owner or conversation changed before commit".to_owned(),
                        ));
                    }
                    if binding.user_id != owner_id {
                        return Err(kogwistar_store_postgres::PostgresStoreError::RecordedRuntimeConflict(
                            "provider event trusted owner binding changed".to_owned(),
                        ));
                    }
                    let current_user = auth_store
                        .auth_user(&owner_id)
                        .await
                        .map_err(|_| kogwistar_store_postgres::PostgresStoreError::RecordedRuntimeConflict(
                            "provider event run owner could not be revalidated".to_owned(),
                        ))?
                        .filter(|user| user.is_active)
                        .ok_or_else(|| kogwistar_store_postgres::PostgresStoreError::RecordedRuntimeConflict(
                            "provider event run owner is missing or inactive".to_owned(),
                        ))?;
                    let current_principal = revalidated_runtime_owner_principal(
                        &binding,
                        &current_user,
                        &capabilities,
                        "write_graph",
                    )
                    .map_err(|_| kogwistar_store_postgres::PostgresStoreError::RecordedRuntimeConflict(
                        "provider event run owner no longer has write_graph authority".to_owned(),
                    ))?;
                    let current_acl_owner_id = authenticated_run_owner_id(&current_principal)
                        .ok_or_else(|| kogwistar_store_postgres::PostgresStoreError::RecordedRuntimeConflict(
                            "provider event run owner identity is invalid".to_owned(),
                        ))?
                        .to_ascii_lowercase();
                    let current_acl_scope = current_principal["security_scope"]
                        .as_str()
                        .map(str::trim)
                        .filter(|value| !value.is_empty())
                        .map(str::to_ascii_lowercase);
                    if !capability_allowed_for_principal(
                        &current_principal,
                        &capabilities,
                        "write_graph.provider_event.commit",
                        &["write_graph"],
                    ) || !principal_strings(&current_principal, "capabilities")
                        .iter()
                        .any(|capability| capability == "write_graph")
                    {
                        return Err(kogwistar_store_postgres::PostgresStoreError::RecordedRuntimeConflict(
                            "provider event write_graph capability was revoked before commit".to_owned(),
                        ));
                    }
                    let current_scope = security_scope_value(&current_principal);
                    if mutations.iter().any(|mutation| {
                        !graph_mutation_scope_allowed(&mutation.scope, &current_scope)
                    }) {
                        return Err(kogwistar_store_postgres::PostgresStoreError::RecordedRuntimeConflict(
                            "provider event scope changed before commit".to_owned(),
                        ));
                    }
                    if let Some(existing) = uow.get_projected_lane_message(&message_id).await? {
                        let matches = existing.namespace == publication.namespace
                            && existing.purpose == "agent_event"
                            && existing.inbox_id == "agent-provider-events"
                            && existing.conversation_id == publication.conversation_id
                            && existing.recipient_id == "agent-event-consumer"
                            && existing.sender_id == "agent-runtime"
                            && existing.msg_type == "agent.provider_event"
                            && existing.run_id.as_deref() == Some(run_id.as_str())
                            && existing.step_id.as_deref()
                                == Some(publication.event.trace.step_id.as_str())
                            && existing.correlation_id.as_deref()
                                == Some(publication.event.event_id.as_str())
                            && existing.payload_json.as_deref().is_some_and(|existing_payload| {
                                provider_event_payloads_match(existing_payload, &event_payload)
                            });
                        if !matches {
                            return Err(kogwistar_store_postgres::PostgresStoreError::RecordedRuntimeConflict(
                                "provider event lane projection conflicts with the existing event".to_owned(),
                            ));
                        }
                        return Ok(());
                    }

                    let message_mutation = mutations
                        .iter()
                        .find(|mutation| mutation.entity_kind == "node" && mutation.record.id == message_id)
                        .cloned()
                        .ok_or_else(|| kogwistar_store_postgres::PostgresStoreError::RecordedRuntimeConflict(
                            "provider event message node is missing".to_owned(),
                        ))?;
                    let (acl_mutations, acl_jobs) = resolve_default_acl_graph_effects(
                        uow,
                        &scope,
                        std::slice::from_ref(&message_mutation),
                        "conversation",
                        &current_acl_owner_id,
                        current_acl_scope.as_deref(),
                        &publication.event.event_id,
                    )
                    .await?;
                    for mutation in mutations {
                        uow.apply_graph_mutation(mutation).await?;
                    }
                    for mutation in acl_mutations {
                        uow.apply_graph_mutation(mutation).await?;
                    }
                    for job in acl_jobs {
                        uow.enqueue_index_job(job).await?;
                    }
                    let now = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map_err(|_| kogwistar_store_postgres::PostgresStoreError::Backend(
                            "system clock is before Unix epoch".to_owned(),
                        ))?
                        .as_secs() as i64;
                    uow.project_lane_message(NewProjectedLaneMessage {
                        message_id,
                        namespace: publication.namespace,
                        purpose: "agent_event".to_owned(),
                        inbox_id: "agent-provider-events".to_owned(),
                        conversation_id: publication.conversation_id,
                        recipient_id: "agent-event-consumer".to_owned(),
                        sender_id: "agent-runtime".to_owned(),
                        msg_type: "agent.provider_event".to_owned(),
                        status: "pending".to_owned(),
                        created_at: now,
                        available_at: now,
                        run_id: Some(run_id.clone()),
                        step_id: Some(publication.event.trace.step_id.clone()),
                        correlation_id: Some(publication.event.event_id.clone()),
                        payload_json: Some(event_payload),
                        error_json: None,
                    })
                    .await?;
                    uow.append_server_run_event(&run_id, &event_type, run_event_payload)
                        .await?;
                    Ok(())
                })
            })
            .await
            .map_err(|_| "provider event publication transaction failed".to_owned())
    }
}
