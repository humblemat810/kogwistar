use super::{
    InMemoryStore, NamedProjection, ServerRun, ServerRunEvent, State, now_ms, now_seconds,
};
use kogwistar_runtime::{
    PersistedRecordedTransition, RECORDED_RUNTIME_CONTRACT_VERSION, RecordedRunStatus,
    RecordedRuntimeState, RecordedRuntimeTransition, RecordedTransitionKind,
    RecordedTransitionResult, RecordedWorkerHandoff, RecordedWorkerSuccessEffect,
    RuntimeWorkerEffectStatus, RuntimeWorkerSuccessEffect,
    authoritative_runtime_successors_for_result, frontier_after_worker_success,
    frontier_after_worker_suspend, reduce_recorded_transition, transition_digest,
    worker_effect_digest,
};
use kogwistar_store::{
    EntityEvent, GraphMutation, GraphMutationBatch, NewProjectedLaneMessage, ProjectedLaneMessage,
    RUNTIME_CURRENT_STATE_NAMESPACE, ServerRunCreate, StoreError, StoreResult,
    runtime_checkpoint_namespace, runtime_projection_write, runtime_status_namespace,
    timestamp_i64, validate_graph_mutation_batch, validate_graph_mutation_replay,
};
use serde_json::Value;

const RECORDED_RUNTIME_EVENT_TYPE: &str = "workflow.recorded_transition.v1";

// PYTHON_PARITY_V1: rust=rust/crates/kogwistar-store-memory/src/recorded_runtime.rs#persist_provider_event_publication; python=kogwistar/engine_core/in_memory_meta.py#append_server_run_event,kogwistar/messaging/service.py#project_lane_message; fixture=contracts/golden/agent-durable-provider-event-v1.json; rust_test=recorded_runtime::tests::provider_event_publication_commits_graph_lane_and_trace_atomically_and_idempotently; python_test=tests/agent/test_provider_event_cross_language_contract.py; note=Rust memory backend atomically publishes graph projection, inbox record, and trace under one process-local lock; Python components remain independently callable.
/// One provider event and every graph/runtime projection derived from it.
/// The in-memory backend commits this bundle under one process-local lock.
#[derive(Clone, Debug, PartialEq)]
pub struct ProviderEventPublicationCommit {
    pub run_id: String,
    pub owner_id: String,
    pub conversation_id: String,
    pub namespace: String,
    pub provider_event_json: String,
    pub graph_mutations: Vec<GraphMutation>,
    pub lane: NewProjectedLaneMessage,
    pub run_event_type: String,
    pub run_event_payload_json: String,
}

impl InMemoryStore {
    /// Atomically publishes a provider event to graph, inbox, and run trace.
    /// This is process-local atomicity only; InMemoryStore is not durable.
    pub fn persist_provider_event_publication(
        &self,
        publication: ProviderEventPublicationCommit,
    ) -> StoreResult<bool> {
        let ProviderEventPublicationCommit {
            run_id,
            owner_id,
            conversation_id,
            namespace,
            provider_event_json,
            graph_mutations,
            lane,
            run_event_type,
            run_event_payload_json,
        } = publication;
        let batch = GraphMutationBatch {
            mutations: graph_mutations,
            index_jobs: Vec::new(),
        };
        let provider_event: kogwistar_contracts::DurableProviderEventV1 =
            serde_json::from_str(&provider_event_json).map_err(json_error)?;
        provider_event.event.validate().map_err(runtime_error)?;
        let event_id = provider_event.event_id;
        validate_graph_mutation_batch(&batch)?;
        if run_id.trim().is_empty()
            || owner_id.trim().is_empty()
            || conversation_id.trim().is_empty()
            || namespace.trim().is_empty()
            || event_id.trim().is_empty()
            || run_event_type != "workflow.provider_event.v1"
            || batch.mutations.iter().any(|mutation| {
                mutation.scope.namespace != namespace
                    || mutation.scope.graph_space.as_deref() != Some("conversation")
            })
            || lane.message_id != format!("msg:{event_id}")
            || lane.namespace != namespace
            || lane.purpose != "agent_event"
            || lane.inbox_id != "agent-provider-events"
            || lane.conversation_id != conversation_id
            || lane.recipient_id != "agent-event-consumer"
            || lane.sender_id != "agent-runtime"
            || lane.msg_type != "agent.provider_event"
            || lane.status != "pending"
            || lane.run_id.as_deref() != Some(run_id.as_str())
            || lane.correlation_id.as_deref() != Some(event_id.as_str())
            || provider_event.trace.run_id != run_id
            || provider_event.trace.turn_id.trim().is_empty()
            || provider_event.trace.workflow_node_id.trim().is_empty()
            || provider_event.trace.step_id.trim().is_empty()
            || provider_event.trace.provider_id.trim().is_empty()
            || provider_event.trace.attempt <= 0
            || lane.payload_json.as_deref() != Some(provider_event_json.as_str())
        {
            return Err(runtime_conflict(
                "provider-event publication fields do not match its graph, lane, or run scope",
            ));
        }
        let run_trace: Value = serde_json::from_str(&run_event_payload_json).map_err(json_error)?;
        if run_trace.get("event_id").and_then(Value::as_str) != Some(event_id.as_str()) {
            return Err(runtime_conflict(
                "provider-event run trace must identify the published event",
            ));
        }
        let now = now_ms();
        let mut state = self.state.write().expect("in-memory store lock poisoned");
        let run = state
            .server_runs
            .get(&run_id)
            .ok_or_else(|| runtime_conflict("provider-event run does not exist"))?;
        if run.user_id.as_deref() != Some(owner_id.as_str())
            || run.conversation_id != conversation_id
            || run.cancel_requested
        {
            return Err(runtime_conflict(
                "provider-event owner, conversation, or run state changed",
            ));
        }

        let existing_lane = state.lane_messages.get(&lane.message_id);
        let existing_run_event = state.server_run_events.get(&run_id).and_then(|events| {
            events.iter().find(|event| {
                event.event_type == run_event_type
                    && serde_json::from_str::<Value>(&event.payload_json)
                        .ok()
                        .is_some_and(|payload| payload["event_id"] == event_id)
            })
        });
        if existing_lane.is_some() || existing_run_event.is_some() {
            let lane_matches = existing_lane.is_some_and(|stored| {
                stored.namespace == lane.namespace
                    && stored.purpose == lane.purpose
                    && stored.inbox_id == lane.inbox_id
                    && stored.conversation_id == lane.conversation_id
                    && stored.recipient_id == lane.recipient_id
                    && stored.sender_id == lane.sender_id
                    && stored.msg_type == lane.msg_type
                    && stored.run_id == lane.run_id
                    && stored.step_id == lane.step_id
                    && stored.correlation_id == lane.correlation_id
                    && stored.payload_json == lane.payload_json
            });
            let trace_matches = existing_run_event
                .is_some_and(|stored| stored.payload_json == run_event_payload_json);
            if !lane_matches || !trace_matches {
                return Err(runtime_conflict(
                    "provider-event replay conflicts with stored lane or trace",
                ));
            }
            for mutation in &batch.mutations {
                let event = state
                    .namespaces
                    .get(&namespace)
                    .and_then(|scope| {
                        scope
                            .event_seq_by_id
                            .get(&mutation.event_id)
                            .and_then(|seq| scope.events.iter().find(|event| event.seq == *seq))
                    })
                    .ok_or_else(|| {
                        runtime_conflict("provider-event replay is missing graph projection")
                    })?;
                validate_graph_mutation_replay(mutation, event)?;
            }
            return Ok(false);
        }
        if state.server_run_events.get(&run_id).is_some_and(|events| {
            events.iter().any(|event| {
                event.event_type == run_event_type
                    && serde_json::from_str::<Value>(&event.payload_json)
                        .ok()
                        .is_some_and(|payload| payload["event_id"] == event_id)
            })
        }) {
            return Err(runtime_conflict("provider-event run trace already exists"));
        }
        for mutation in &batch.mutations {
            if state
                .namespaces
                .get(&namespace)
                .is_some_and(|scope| scope.event_seq_by_id.contains_key(&mutation.event_id))
            {
                return Err(runtime_conflict(
                    "provider-event graph event ID is already in use",
                ));
            }
        }

        for mutation in batch.mutations {
            let scope = InMemoryStore::scope_mut(&mut state, &namespace);
            let seq = scope.next_event_seq + 1;
            let event = EntityEvent {
                namespace: namespace.clone(),
                seq,
                event_id: mutation.event_id.clone(),
                entity_kind: mutation.entity_kind,
                entity_id: mutation.record.id.clone(),
                op: mutation.op,
                payload: mutation.payload,
            };
            scope.next_event_seq = seq;
            scope.event_seq_by_id.insert(event.event_id.clone(), seq);
            scope.events.push(event);
            let projection = state.graph_projections.entry(mutation.scope).or_default();
            let key = (mutation.table, mutation.record.id.clone());
            if !projection.records.contains_key(&key) {
                projection.record_order.push(key.clone());
            }
            projection.records.insert(key, mutation.record);
        }
        insert_initial_lane(&mut state, lane);
        let events = state.server_run_events.entry(run_id.clone()).or_default();
        events.push(ServerRunEvent {
            seq: events.last().map_or(1, |event| event.seq + 1),
            run_id,
            event_type: run_event_type,
            payload_json: run_event_payload_json,
            created_at_ms: now,
        });
        Ok(true)
    }
}

// PYTHON_REFERENCE_V1: rust=rust/crates/kogwistar-store-memory/src/recorded_runtime.rs#apply_recorded_runtime_transition; python=kogwistar/engine_core/in_memory_meta.py#append_server_run_event+update_server_run; rust_test=rust/crates/kogwistar-store-memory/src/recorded_runtime.rs#recorded_transition_is_atomic_replayable_and_projected_in_memory; python_test=tests/core/test_recorded_transition_identity.py#test_metadata_stores_reject_duplicate_recorded_transition_ids; note=Rust adds an atomic reducer transaction; Python remains primitive append/update and rejects duplicate IDs.
// PYTHON_REFERENCE_V1: rust=rust/crates/kogwistar-store-memory/src/recorded_runtime.rs#admit_server_run_with_start; python=kogwistar/engine_core/in_memory_meta.py#create_server_run+append_server_run_event,kogwistar/messaging/service.py#project_lane_message; rust_test=rust/crates/kogwistar-store-memory/src/recorded_runtime.rs#runtime_admission_atomically_records_start_and_first_lane; python_test=tests/core/test_in_memory_meta_store.py#test_in_memory_meta_retry_and_run_cancel_contract,tests/core/test_lane_message_meta_store_contract.py#test_lane_message_projection_contract; note=Rust admission is one process-local atomic write; Python's in-memory primitives remain separately callable.
impl InMemoryStore {
    /// Atomically admits a run, records its admission event and
    /// start transition, and queues its first worker step under one writer lock.
    pub fn admit_server_run_with_start(
        &self,
        run: ServerRunCreate,
        admission_payload: Value,
        transition: RecordedRuntimeTransition,
        lane: NewProjectedLaneMessage,
    ) -> StoreResult<RecordedTransitionResult> {
        validate_runtime_admission(&run, &transition, &lane)?;
        validate_runtime_admission_payload(&run, &admission_payload)?;
        let admission_payload_json =
            serde_json::to_string(&admission_payload).map_err(json_error)?;
        let transition_digest = transition_digest(&transition).map_err(runtime_error)?;
        let now = now_ms();

        let make_run = |created_at_ms| ServerRun {
            run_id: run.run_id.clone(),
            conversation_id: run.conversation_id.clone(),
            workflow_id: run.workflow_id.clone(),
            user_id: run.user_id.clone(),
            user_turn_node_id: Some(run.user_turn_node_id.clone()),
            assistant_turn_node_id: None,
            status: "queued".to_owned(),
            cancel_requested: false,
            result_json: None,
            error_json: None,
            created_at_ms,
            updated_at_ms: created_at_ms,
            started_at_ms: None,
            finished_at_ms: None,
        };

        // Preflight reducer and serialization before touching shared state.
        let mut preflight = State::default();
        preflight
            .server_runs
            .insert(run.run_id.clone(), make_run(now));
        preflight.server_run_events.insert(
            run.run_id.clone(),
            vec![ServerRunEvent {
                seq: 1,
                run_id: run.run_id.clone(),
                event_type: "run.created".to_owned(),
                payload_json: admission_payload_json.clone(),
                created_at_ms: now,
            }],
        );
        apply_transition(&mut preflight, transition.clone(), None, false)?;

        let mut state = self.state.write().expect("in-memory store lock poisoned");
        if let Some(existing) = state.server_runs.get(&run.run_id) {
            let replay = recorded_transition_by_id(&state, &transition.transition_id)?;
            let admission_event_matches = state
                .server_run_events
                .get(&run.run_id)
                .and_then(|events| events.first())
                .is_some_and(|event| {
                    event.event_type == "run.created"
                        && event.payload_json == admission_payload_json
                });
            let lane_matches = state
                .lane_messages
                .get(&lane.message_id)
                .is_some_and(|stored| {
                    stored.namespace == lane.namespace
                        && stored.purpose
                            == if lane.purpose.is_empty() {
                                "user_visible"
                            } else {
                                lane.purpose.as_str()
                            }
                        && stored.inbox_id == lane.inbox_id
                        && stored.conversation_id == lane.conversation_id
                        && stored.recipient_id == lane.recipient_id
                        && stored.sender_id == lane.sender_id
                        && stored.msg_type == lane.msg_type
                        && stored.run_id.as_deref() == Some(run.run_id.as_str())
                        && stored.payload_json == lane.payload_json
                        && stored.step_id == lane.step_id
                        && stored.correlation_id == lane.correlation_id
                        && stored.created_at == lane.created_at
                        && stored.available_at == lane.available_at
                });
            if existing.conversation_id != run.conversation_id
                || existing.workflow_id != run.workflow_id
                || existing.user_id != run.user_id
                || existing.user_turn_node_id.as_deref() != Some(&run.user_turn_node_id)
                || !admission_event_matches
                || !lane_matches
                || replay
                    .as_ref()
                    .is_none_or(|(stored, _)| stored.request_digest != transition_digest)
            {
                return Err(runtime_conflict(
                    "runtime admission replay conflicts with existing run, transition, or lane",
                ));
            }
            return apply_transition(&mut state, transition, None, false);
        }
        if state.lane_messages.contains_key(&lane.message_id) {
            return Err(runtime_conflict(
                "runtime admission lane message ID already exists",
            ));
        }
        if state.server_run_events.contains_key(&run.run_id)
            || recorded_transition_by_id(&state, &transition.transition_id)?.is_some()
        {
            return Err(runtime_conflict(
                "runtime admission found orphaned run events or a reused transition ID",
            ));
        }

        state.server_runs.insert(run.run_id.clone(), make_run(now));
        state.server_run_events.insert(
            run.run_id.clone(),
            vec![ServerRunEvent {
                seq: 1,
                run_id: run.run_id.clone(),
                event_type: "run.created".to_owned(),
                payload_json: admission_payload_json,
                created_at_ms: now,
            }],
        );
        let started = apply_transition(&mut state, transition, None, false)?;
        insert_initial_lane(&mut state, lane);
        Ok(started)
    }

    /// Atomically records a Rust/Python-compatible runtime transition in the
    /// process-local store. One write lock is the transaction boundary; no
    /// resolver, provider, or graph callback runs while that lock is held.
    pub fn apply_recorded_runtime_transition(
        &self,
        transition: RecordedRuntimeTransition,
        abort_after_writes: bool,
    ) -> StoreResult<RecordedTransitionResult> {
        let mut state = self.state.write().expect("in-memory store lock poisoned");
        apply_transition(&mut state, transition, None, abort_after_writes)
    }

    /// Applies one claimed worker effect and acknowledges its lane in the same
    /// write-lock transaction. Retries require the original handoff and digest.
    pub fn apply_claimed_recorded_worker_effect(
        &self,
        handoff: RecordedWorkerHandoff,
        effect: RecordedWorkerSuccessEffect,
    ) -> StoreResult<RecordedTransitionResult> {
        validate_handoff(&handoff)?;
        if effect.contract_version != RECORDED_RUNTIME_CONTRACT_VERSION
            || effect.effect_id.is_empty()
        {
            return Err(runtime_conflict(
                "worker effect contract_version/effect_id is invalid",
            ));
        }
        if effect.graph_mutation_batch.is_some()
            || effect.graph_mutation_batch_v2.is_some()
            || !effect.provider_event_publications.is_empty()
            || !effect.provider_event_acknowledgements.is_empty()
        {
            return Err(runtime_conflict(
                "in-memory worker result does not support graph or provider-event effects",
            ));
        }
        let effect_digest = worker_effect_digest(&effect).map_err(runtime_error)?;
        let trace_events = effect.trace_events.clone();
        let usage = effect.usage.clone();
        let mut state = self.state.write().expect("in-memory store lock poisoned");
        let lane = state
            .lane_messages
            .get(&handoff.message_id)
            .cloned()
            .ok_or_else(|| runtime_conflict("worker lane message does not exist"))?;
        if lane.run_id.as_deref() != Some(handoff.run_id.as_str())
            || lane.step_id.as_deref() != Some(handoff.step_id.as_str())
            || lane.correlation_id.as_deref() != Some(handoff.correlation_id.as_str())
        {
            return Err(runtime_conflict(
                "worker lane run/step/correlation identity differs",
            ));
        }
        if state
            .server_runs
            .get(&handoff.run_id)
            .is_some_and(|run| run.cancel_requested)
        {
            return Err(runtime_conflict(
                "worker lane belongs to a cancel-requested run",
            ));
        }
        if lane.status == "completed" {
            if let Some((persisted, event_seq)) =
                recorded_transition_by_id(&state, &effect.effect_id)?
                && persisted.worker_handoff.as_ref().is_some_and(|stored| {
                    stored == &handoff || stored.message_id == handoff.message_id
                })
            {
                if persisted.worker_effect_digest.as_deref() != Some(effect_digest.as_str()) {
                    return Err(runtime_conflict(
                        "worker handoff retried with different effect",
                    ));
                }
                if let Some((completed, completed_seq)) =
                    recorded_transition_by_id(&state, &format!("complete-{}", effect.effect_id))?
                    && completed.reduced.state.run_id == handoff.run_id
                {
                    return Ok(completed.result(completed_seq, true));
                }
                return Ok(persisted.result(event_seq, true));
            }
            return Err(runtime_conflict(
                "completed worker lane has no matching recorded effect",
            ));
        }
        if lane.status != "claimed"
            || lane.claimed_by.as_deref() != Some(handoff.claimed_by.as_str())
            || lane
                .lease_until
                .as_ref()
                .is_none_or(|lease_until| timestamp_i64(lease_until) < now_seconds())
        {
            return Err(runtime_conflict(
                "worker lane is not actively claimed by this handoff",
            ));
        }

        let payload: Value = serde_json::from_str(lane.payload_json.as_deref().unwrap_or("{}"))
            .map_err(json_error)?;
        let workflow_id = payload["workflow_id"].as_str().unwrap_or_default();
        let conversation_id = payload["conversation_id"].as_str().unwrap_or_default();
        let token_id = payload["token_id"]
            .as_str()
            .unwrap_or(handoff.run_id.as_str());
        let parent_token_id = payload["parent_token_id"].as_str();
        let current = recorded_state_for(&state, &handoff.run_id, workflow_id, conversation_id)?
            .ok_or_else(|| runtime_conflict("worker lane has no recorded runtime state"))?;
        if current
            .frontier
            .pending
            .first()
            .is_none_or(|(node, _, token, parent)| {
                node != &handoff.step_id
                    || token != token_id
                    || parent.as_deref() != parent_token_id
            })
        {
            return Err(runtime_conflict(
                "worker result is not next in canonical frontier order",
            ));
        }
        let step_seq = current.last_step_seq.saturating_add(1);
        let successors = match effect.status {
            RuntimeWorkerEffectStatus::Success => authoritative_runtime_successors_for_result(
                &current.static_routes,
                &handoff.step_id,
                &effect.successors,
                &effect.route_next,
                false,
            ),
            RuntimeWorkerEffectStatus::Suspended => Ok(Vec::new()),
            RuntimeWorkerEffectStatus::Failed => authoritative_runtime_successors_for_result(
                &current.static_routes,
                &handoff.step_id,
                &effect.successors,
                &effect.route_next,
                true,
            ),
        }
        .map_err(runtime_error)?;
        let terminal_failure =
            effect.status == RuntimeWorkerEffectStatus::Failed && successors.is_empty();
        let frontier = match effect.status {
            RuntimeWorkerEffectStatus::Success => frontier_after_worker_success(
                &current.frontier,
                &handoff.step_id,
                token_id,
                parent_token_id,
                step_seq,
                &RuntimeWorkerSuccessEffect { successors },
            ),
            RuntimeWorkerEffectStatus::Suspended => frontier_after_worker_suspend(
                &current.frontier,
                &handoff.step_id,
                token_id,
                parent_token_id,
            ),
            RuntimeWorkerEffectStatus::Failed if terminal_failure => {
                Ok(kogwistar_runtime::RuntimeFrontier::default())
            }
            RuntimeWorkerEffectStatus::Failed => frontier_after_worker_success(
                &current.frontier,
                &handoff.step_id,
                token_id,
                parent_token_id,
                step_seq,
                &RuntimeWorkerSuccessEffect { successors },
            ),
        }
        .map_err(runtime_error)?;
        let transition = RecordedRuntimeTransition {
            contract_version: RECORDED_RUNTIME_CONTRACT_VERSION,
            transition_id: effect.effect_id.clone(),
            expected_event_seq: state
                .server_run_events
                .get(&handoff.run_id)
                .and_then(|events| events.last())
                .map_or(0, |event| event.seq),
            kind: match effect.status {
                RuntimeWorkerEffectStatus::Success => RecordedTransitionKind::RecordedStepSuccess,
                RuntimeWorkerEffectStatus::Suspended => RecordedTransitionKind::Suspend,
                RuntimeWorkerEffectStatus::Failed if terminal_failure => {
                    RecordedTransitionKind::Fail
                }
                RuntimeWorkerEffectStatus::Failed => RecordedTransitionKind::RecordedStepSuccess,
            },
            run_id: handoff.run_id.clone(),
            workflow_id: workflow_id.to_owned(),
            conversation_id: conversation_id.to_owned(),
            user_id: None,
            user_turn_node_id: None,
            step_seq,
            node_id: Some(handoff.step_id.clone()),
            token_id: Some(token_id.to_owned()),
            parent_token_id: parent_token_id.map(str::to_owned),
            initial_state: None,
            state_update: effect.state_update,
            update: effect.update,
            state_schema: effect.state_schema,
            frontier: Some(frontier),
            result: effect.result.clone(),
            wait_reason: effect.wait_reason,
            resume_payload: effect.resume_payload,
            errors: effect.errors,
        };
        let mut telemetry_events = Vec::new();
        if let Some(usage) = usage {
            telemetry_events.push((
                "workflow.usage.v1",
                serde_json::to_string(&serde_json::json!({
                    "run_id": handoff.run_id,
                    "step_id": handoff.step_id,
                    "token_id": token_id,
                    "effect_id": effect.effect_id,
                    "usage": usage,
                }))
                .map_err(json_error)?,
            ));
        }
        for (trace_index, trace) in trace_events.into_iter().enumerate() {
            telemetry_events.push((
                "workflow.trace.v1",
                serde_json::to_string(&serde_json::json!({
                    "run_id": handoff.run_id,
                    "step_id": handoff.step_id,
                    "token_id": token_id,
                    "effect_id": effect.effect_id,
                    "trace_index": trace_index,
                    "trace": trace,
                }))
                .map_err(json_error)?,
            ));
        }
        let applied = apply_transition(
            &mut state,
            transition,
            Some((handoff.clone(), effect_digest)),
            false,
        )?;
        let current_frontier = &applied.reduced.state.frontier;
        let unfinished = !current_frontier.pending.is_empty()
            || !current_frontier.suspended.is_empty()
            || current_frontier
                .join_outstanding
                .iter()
                .any(|value| *value != 0)
            || current_frontier
                .join_waiters
                .values()
                .any(|waiters| !waiters.is_empty());
        let result = if unfinished || applied.reduced.state.status.is_terminal() {
            applied
        } else {
            let completed_state = &applied.reduced.state;
            let expected_event_seq = state
                .server_run_events
                .get(&handoff.run_id)
                .and_then(|events| events.last())
                .map_or(0, |event| event.seq);
            apply_transition(
                &mut state,
                RecordedRuntimeTransition {
                    contract_version: RECORDED_RUNTIME_CONTRACT_VERSION,
                    transition_id: format!("complete-{}", effect.effect_id),
                    expected_event_seq,
                    kind: RecordedTransitionKind::Complete,
                    run_id: handoff.run_id.clone(),
                    workflow_id: completed_state.workflow_id.clone(),
                    conversation_id: completed_state.conversation_id.clone(),
                    user_id: None,
                    user_turn_node_id: None,
                    step_seq: completed_state.last_step_seq,
                    node_id: completed_state.last_node_id.clone(),
                    token_id: completed_state.last_token_id.clone(),
                    parent_token_id: completed_state.last_parent_token_id.clone(),
                    initial_state: None,
                    state_update: Vec::new(),
                    update: None,
                    state_schema: Default::default(),
                    frontier: Some(kogwistar_runtime::RuntimeFrontier::default()),
                    result: effect.result.clone(),
                    wait_reason: None,
                    resume_payload: None,
                    errors: Vec::new(),
                },
                None,
                false,
            )?
        };
        let mut next_seq = result.event_seq;
        for (event_type, payload_json) in telemetry_events {
            next_seq += 1;
            state
                .server_run_events
                .entry(handoff.run_id.clone())
                .or_default()
                .push(ServerRunEvent {
                    seq: next_seq,
                    run_id: handoff.run_id.clone(),
                    event_type: event_type.to_owned(),
                    payload_json,
                    created_at_ms: now_ms(),
                });
        }
        let lane = state
            .lane_messages
            .get_mut(&handoff.message_id)
            .expect("validated worker lane remains present under write lock");
        lane.status = "completed".to_owned();
        lane.claimed_by = None;
        lane.lease_until = None;
        Ok(result)
    }
}

fn validate_runtime_admission(
    run: &ServerRunCreate,
    transition: &RecordedRuntimeTransition,
    lane: &NewProjectedLaneMessage,
) -> StoreResult<()> {
    if run.run_id.trim().is_empty()
        || run.workflow_id.trim().is_empty()
        || run.conversation_id.trim().is_empty()
        || run.user_turn_node_id.trim().is_empty()
        || run.status != "queued"
        || transition.kind != RecordedTransitionKind::Start
        || transition.run_id != run.run_id
        || transition.workflow_id != run.workflow_id
        || transition.conversation_id != run.conversation_id
        || transition.user_id != run.user_id
        || transition.user_turn_node_id.as_deref() != Some(&run.user_turn_node_id)
        || transition.expected_event_seq != 1
        || lane.message_id.trim().is_empty()
        || lane.run_id.as_deref() != Some(run.run_id.as_str())
        || lane.conversation_id != run.conversation_id
        || lane.msg_type != "workflow.step.execute"
        || lane.status != "pending"
        || lane.correlation_id.as_deref() != Some(run.run_id.as_str())
        || lane.step_id.as_deref() != transition.node_id.as_deref()
    {
        return Err(runtime_conflict(
            "runtime admission identity, start transition, or initial lane is invalid",
        ));
    }
    super::namespace(&lane.namespace)?;
    let lane_payload: Value =
        serde_json::from_str(lane.payload_json.as_deref().unwrap_or("{}")).map_err(json_error)?;
    if lane_payload["workflow_id"].as_str() != Some(run.workflow_id.as_str())
        || lane_payload["conversation_id"].as_str() != Some(run.conversation_id.as_str())
        || lane_payload["expected_event_seq"].as_i64() != Some(2)
    {
        return Err(runtime_conflict(
            "runtime admission lane payload does not match the started run sequence",
        ));
    }
    if lane.namespace.trim().is_empty()
        || lane.inbox_id.trim().is_empty()
        || lane.step_id.as_deref().is_none_or(str::is_empty)
    {
        return Err(runtime_conflict(
            "runtime admission initial lane is missing its namespace, inbox, or step",
        ));
    }
    Ok(())
}

fn validate_runtime_admission_payload(run: &ServerRunCreate, payload: &Value) -> StoreResult<()> {
    if payload["run_id"].as_str() != Some(run.run_id.as_str())
        || payload["run_kind"].as_str() != Some("workflow_runtime")
        || payload["conversation_id"].as_str() != Some(run.conversation_id.as_str())
        || payload["workflow_id"].as_str() != Some(run.workflow_id.as_str())
        || payload["status"].as_str() != Some("queued")
        || payload["turn_node_id"].as_str() != Some(run.user_turn_node_id.as_str())
    {
        return Err(runtime_conflict(
            "runtime admission event does not match the admitted run identity",
        ));
    }
    Ok(())
}

fn insert_initial_lane(state: &mut State, row: NewProjectedLaneMessage) {
    let inbox_tail = state
        .lane_messages
        .values()
        .filter(|stored| stored.namespace == row.namespace && stored.inbox_id == row.inbox_id)
        .max_by_key(|stored| (stored.seq, stored.created_at))
        .cloned();
    let conversation_tail = state
        .lane_messages
        .values()
        .filter(|stored| {
            stored.namespace == row.namespace && stored.conversation_id == row.conversation_id
        })
        .max_by_key(|stored| (stored.conversation_seq, stored.created_at))
        .cloned();
    let seq = inbox_tail.as_ref().map_or(1, |stored| stored.seq + 1);
    let conversation_seq = conversation_tail
        .as_ref()
        .map_or(1, |stored| stored.conversation_seq + 1);
    let message_id = row.message_id.clone();
    state.lane_messages.insert(
        message_id.clone(),
        ProjectedLaneMessage {
            message_id: message_id.clone(),
            namespace: row.namespace,
            purpose: if row.purpose.is_empty() {
                "user_visible".to_owned()
            } else {
                row.purpose
            },
            inbox_id: row.inbox_id,
            conversation_id: row.conversation_id,
            recipient_id: row.recipient_id,
            sender_id: row.sender_id,
            msg_type: row.msg_type,
            status: row.status,
            seq,
            conversation_seq,
            claimed_by: None,
            lease_until: None,
            retry_count: 0,
            created_at: row.created_at,
            available_at: row.available_at,
            run_id: row.run_id,
            step_id: row.step_id,
            correlation_id: row.correlation_id,
            payload_json: row.payload_json,
            error_json: row.error_json,
            prev_message_id: inbox_tail.map(|stored| stored.message_id),
            next_message_id: None,
            inbox_tail_message_id: Some(message_id.clone()),
            conversation_tail_message_id: Some(message_id),
        },
    );
}

fn apply_transition(
    state: &mut State,
    transition: RecordedRuntimeTransition,
    worker: Option<(RecordedWorkerHandoff, String)>,
    abort_after_writes: bool,
) -> StoreResult<RecordedTransitionResult> {
    let request_digest = transition_digest(&transition).map_err(runtime_error)?;

    if let Some((persisted, event_seq)) =
        recorded_transition_by_id(state, &transition.transition_id)?
    {
        if persisted.request_digest != request_digest
            || match &worker {
                Some((handoff, digest)) => {
                    persisted.worker_handoff.as_ref() != Some(handoff)
                        || persisted.worker_effect_digest.as_deref() != Some(digest.as_str())
                }
                None => persisted.worker_handoff.is_some(),
            }
        {
            return Err(runtime_conflict(format!(
                "transition_id {:?} reused with different payload",
                transition.transition_id
            )));
        }
        return Ok(persisted.result(event_seq, true));
    }

    let prior_run = state.server_runs.get(&transition.run_id).cloned();
    if let Some(run) = &prior_run
        && (run.workflow_id != transition.workflow_id
            || run.conversation_id != transition.conversation_id)
    {
        return Err(runtime_conflict(format!(
            "run identity differs for {:?}",
            transition.run_id
        )));
    }

    let current = recorded_state(state, &transition)?;
    let current_event_seq = state
        .server_run_events
        .get(&transition.run_id)
        .and_then(|events| events.last())
        .map_or(0, |event| event.seq);
    if current_event_seq != transition.expected_event_seq {
        return Err(runtime_conflict(format!(
            "expected_event_seq {}, current {}",
            transition.expected_event_seq, current_event_seq
        )));
    }

    let reduced =
        reduce_recorded_transition(current.as_ref(), &transition).map_err(runtime_error)?;
    let new_user_turn_node_id = if current.is_none() && prior_run.is_none() {
        Some(
            transition
                .user_turn_node_id
                .clone()
                .ok_or_else(|| runtime_conflict("start transition requires user_turn_node_id"))?,
        )
    } else {
        None
    };
    let event_seq = current_event_seq + 1;
    let persisted = PersistedRecordedTransition {
        contract_version: RECORDED_RUNTIME_CONTRACT_VERSION,
        transition_id: transition.transition_id.clone(),
        request_digest,
        worker_handoff: worker.as_ref().map(|(handoff, _)| handoff.clone()),
        worker_effect_digest: worker.map(|(_, digest)| digest),
        reduced: reduced.clone(),
    };
    let payload_json = serde_json::to_string(&persisted).map_err(json_error)?;
    let result_json = reduced
        .result
        .as_ref()
        .map(serde_json::to_string)
        .transpose()
        .map_err(json_error)?;
    let error_json = if reduced.errors.is_empty() {
        prior_run.as_ref().and_then(|run| run.error_json.clone())
    } else {
        Some(serde_json::to_string(&reduced.errors).map_err(json_error)?)
    };
    let state_payload = serde_json::to_value(&reduced.state)
        .map_err(json_error)?
        .as_object()
        .cloned()
        .ok_or_else(|| runtime_conflict("runtime state projection is not an object"))?;
    let server_status = reduced.server_status.clone();
    let status_materialization = match reduced.state.status {
        RecordedRunStatus::Completed => "completed",
        RecordedRunStatus::Failed => "failed",
        RecordedRunStatus::Cancelled => "cancelled",
        RecordedRunStatus::Suspended => "suspended",
        RecordedRunStatus::Running => "running",
    };
    let assistant_turn_node_id = (reduced.state.status == RecordedRunStatus::Completed)
        .then(|| {
            reduced
                .state
                .state
                .get("agentic_answering_result")
                .and_then(|result| result.get("assistant_turn_node_id"))
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .flatten()
        .or_else(|| {
            prior_run
                .as_ref()
                .and_then(|run| run.assistant_turn_node_id.clone())
        });
    let now = now_ms();

    if abort_after_writes {
        return Err(runtime_conflict(
            "in-memory runtime transition aborted before atomic commit",
        ));
    }

    if let Some(user_turn_node_id) = new_user_turn_node_id {
        state.server_runs.insert(
            transition.run_id.clone(),
            ServerRun {
                run_id: transition.run_id.clone(),
                conversation_id: transition.conversation_id.clone(),
                workflow_id: transition.workflow_id.clone(),
                user_id: transition.user_id.clone(),
                user_turn_node_id: Some(user_turn_node_id),
                assistant_turn_node_id: None,
                status: server_status.clone(),
                cancel_requested: false,
                result_json: None,
                error_json: None,
                created_at_ms: now,
                updated_at_ms: now,
                started_at_ms: None,
                finished_at_ms: None,
            },
        );
    }

    state
        .server_run_events
        .entry(transition.run_id.clone())
        .or_default()
        .push(ServerRunEvent {
            seq: event_seq,
            run_id: transition.run_id.clone(),
            event_type: RECORDED_RUNTIME_EVENT_TYPE.to_owned(),
            payload_json,
            created_at_ms: now,
        });

    put_projection(
        state,
        RUNTIME_CURRENT_STATE_NAMESPACE,
        &transition.run_id,
        runtime_projection_write(state_payload, event_seq, "ready"),
        now,
    );
    put_projection(
        state,
        &runtime_checkpoint_namespace(&transition.conversation_id),
        &transition.run_id,
        runtime_projection_write(reduced.checkpoint.clone(), event_seq, server_status.clone()),
        now,
    );
    put_projection(
        state,
        &runtime_status_namespace(&transition.conversation_id),
        &transition.run_id,
        runtime_projection_write(
            reduced.run_status.clone(),
            event_seq,
            status_materialization,
        ),
        now,
    );

    let run = state
        .server_runs
        .entry(transition.run_id.clone())
        .or_insert_with(|| {
            prior_run
                .clone()
                .expect("existing run checked before reduction")
        });
    run.status = server_status;
    run.assistant_turn_node_id = assistant_turn_node_id;
    run.result_json =
        result_json.or_else(|| prior_run.as_ref().and_then(|row| row.result_json.clone()));
    run.error_json = error_json;
    run.started_at_ms = prior_run
        .as_ref()
        .and_then(|row| row.started_at_ms)
        .or_else(|| (transition.kind == RecordedTransitionKind::Start).then_some(now));
    run.finished_at_ms = if reduced.state.status.is_terminal() {
        Some(now)
    } else {
        prior_run.as_ref().and_then(|row| row.finished_at_ms)
    };
    run.cancel_requested = reduced.state.status == RecordedRunStatus::Cancelled;
    run.updated_at_ms = now;

    Ok(persisted.result(event_seq, false))
}

fn recorded_transition_by_id(
    state: &State,
    transition_id: &str,
) -> StoreResult<Option<(PersistedRecordedTransition, i64)>> {
    for events in state.server_run_events.values() {
        for event in events {
            if event.event_type != RECORDED_RUNTIME_EVENT_TYPE {
                continue;
            }
            let Ok(persisted) =
                serde_json::from_str::<PersistedRecordedTransition>(&event.payload_json)
            else {
                continue;
            };
            if persisted.transition_id == transition_id {
                return Ok(Some((persisted, event.seq)));
            }
        }
    }
    Ok(None)
}

fn recorded_state(
    state: &State,
    transition: &RecordedRuntimeTransition,
) -> StoreResult<Option<RecordedRuntimeState>> {
    recorded_state_for(
        state,
        &transition.run_id,
        &transition.workflow_id,
        &transition.conversation_id,
    )
}

fn recorded_state_for(
    state: &State,
    run_id: &str,
    workflow_id: &str,
    conversation_id: &str,
) -> StoreResult<Option<RecordedRuntimeState>> {
    let mut latest = None;
    if let Some(events) = state.server_run_events.get(run_id) {
        for event in events {
            if event.event_type != RECORDED_RUNTIME_EVENT_TYPE {
                continue;
            }
            let Ok(persisted) =
                serde_json::from_str::<PersistedRecordedTransition>(&event.payload_json)
            else {
                continue;
            };
            if persisted.contract_version != RECORDED_RUNTIME_CONTRACT_VERSION {
                continue;
            }
            let runtime = persisted.reduced.state;
            if runtime.run_id == run_id
                && runtime.workflow_id == workflow_id
                && runtime.conversation_id == conversation_id
            {
                latest = Some(runtime);
            }
        }
    }
    Ok(latest)
}

fn validate_handoff(handoff: &RecordedWorkerHandoff) -> StoreResult<()> {
    for (field, value) in [
        ("message_id", handoff.message_id.as_str()),
        ("claimed_by", handoff.claimed_by.as_str()),
        ("run_id", handoff.run_id.as_str()),
        ("step_id", handoff.step_id.as_str()),
        ("correlation_id", handoff.correlation_id.as_str()),
    ] {
        if value.is_empty() {
            return Err(runtime_conflict(format!(
                "worker handoff {field} must not be empty"
            )));
        }
    }
    Ok(())
}

fn put_projection(
    state: &mut State,
    namespace: &str,
    key: &str,
    projection: kogwistar_store::NamedProjectionWrite,
    updated_at_ms: i64,
) {
    InMemoryStore::scope_mut(state, namespace)
        .projections
        .insert(
            key.to_owned(),
            NamedProjection {
                namespace: namespace.to_owned(),
                key: key.to_owned(),
                payload: projection.payload,
                last_authoritative_seq: projection.last_authoritative_seq,
                last_materialized_seq: projection.last_materialized_seq,
                projection_schema_version: projection.projection_schema_version,
                materialization_status: projection.materialization_status,
                updated_at_ms,
            },
        );
}

fn runtime_error(error: impl std::fmt::Display) -> StoreError {
    runtime_conflict(error.to_string())
}

fn json_error(error: serde_json::Error) -> StoreError {
    runtime_conflict(format!(
        "cannot encode recorded runtime transition: {error}"
    ))
}

fn runtime_conflict(message: impl Into<String>) -> StoreError {
    StoreError::Backend {
        backend: "in-memory runtime".to_owned(),
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kogwistar_store::{
        GraphRecord, GraphScope, LaneMessageClaimFilter, LaneMessageReadStore,
        LaneMessageWriteStore, NewProjectedLaneMessage, ProjectionReadStore, ServerRunReadStore,
    };
    use serde_json::json;
    use std::future::Future;
    use std::pin::pin;
    use std::task::{Context, Poll, Waker};

    fn block_on<T>(future: impl Future<Output = T>) -> T {
        let waker = Waker::noop();
        let mut context = Context::from_waker(waker);
        let mut future = pin!(future);
        loop {
            match future.as_mut().poll(&mut context) {
                Poll::Ready(value) => return value,
                Poll::Pending => std::thread::yield_now(),
            }
        }
    }

    fn start(transition_id: &str) -> RecordedRuntimeTransition {
        serde_json::from_value(json!({
            "contract_version": 1,
            "transition_id": transition_id,
            "expected_event_seq": 0,
            "kind": "start",
            "run_id": "memory-run",
            "workflow_id": "memory-workflow",
            "conversation_id": "memory-conversation",
            "user_id": "user-1",
            "user_turn_node_id": "turn-1",
            "step_seq": 0,
            "initial_state": {"seed": 1},
            "frontier": {"pending": [], "suspended": [], "join_node_ids": [], "join_outstanding": [], "join_waiters": {}}
        }))
        .unwrap()
    }

    fn pending_start() -> RecordedRuntimeTransition {
        let mut transition = start("start-worker-run");
        transition.frontier = Some(
            serde_json::from_value(json!({
                "pending": [["noop", 0, "token-1", null]],
                "suspended": [],
                "join_node_ids": [],
                "join_outstanding": [],
                "join_waiters": {}
            }))
            .unwrap(),
        );
        transition
    }

    fn noop_effect() -> RecordedWorkerSuccessEffect {
        serde_json::from_value(json!({
            "contract_version": 1,
            "effect_id": "native-noop-effect",
            "status": "success",
            "state_update": [],
            "successors": [],
            "route_next": [],
            "result": {"workflow_status": "succeeded", "final_state": {"seed": 1}},
            "trace_events": [{"event": "runtime.native.step.completed", "operation": "noop"}]
        }))
        .unwrap()
    }

    fn worker_lane() -> NewProjectedLaneMessage {
        NewProjectedLaneMessage {
            message_id: "lane-noop-1".to_owned(),
            namespace: "workflow".to_owned(),
            purpose: "system".to_owned(),
            inbox_id: "workflow-runtime".to_owned(),
            conversation_id: "memory-conversation".to_owned(),
            recipient_id: "rust-native-worker".to_owned(),
            sender_id: "rust-scheduler".to_owned(),
            msg_type: "workflow.step.execute".to_owned(),
            status: "pending".to_owned(),
            created_at: now_seconds(),
            available_at: now_seconds(),
            run_id: Some("memory-run".to_owned()),
            step_id: Some("noop".to_owned()),
            correlation_id: Some("memory-run".to_owned()),
            payload_json: Some(
                json!({
                    "workflow_id": "memory-workflow",
                    "conversation_id": "memory-conversation",
                "token_id": "token-1",
                "expected_event_seq": 2
                })
                .to_string(),
            ),
            error_json: None,
        }
    }

    fn runtime_admission_payload() -> Value {
        json!({
            "run_id": "memory-run",
            "run_kind": "workflow_runtime",
            "conversation_id": "memory-conversation",
            "workflow_id": "memory-workflow",
            "status": "queued",
            "turn_node_id": "turn-1"
        })
    }

    fn handoff(claimed_by: &str) -> RecordedWorkerHandoff {
        RecordedWorkerHandoff {
            message_id: "lane-noop-1".to_owned(),
            claimed_by: claimed_by.to_owned(),
            run_id: "memory-run".to_owned(),
            step_id: "noop".to_owned(),
            correlation_id: "memory-run".to_owned(),
        }
    }

    fn provider_publication(event_id: &str) -> ProviderEventPublicationCommit {
        let namespace = "docs".to_owned();
        let conversation_id = "memory-conversation".to_owned();
        let scope = GraphScope {
            namespace: namespace.clone(),
            workspace_id: Some("workspace-1".to_owned()),
            graph_space: Some("conversation".to_owned()),
        };
        let mut metadata = serde_json::Map::new();
        metadata.insert("namespace".to_owned(), json!(namespace));
        metadata.insert("workspace_id".to_owned(), json!("workspace-1"));
        metadata.insert("graph_space".to_owned(), json!("conversation"));
        let trace_payload = json!({"event_id": event_id, "trace_id": "trace-1"}).to_string();
        let provider_event_json = json!({
            "event_id": event_id,
            "trace": {
                "run_id": "memory-run",
                "turn_id": "turn-1",
                "workflow_node_id": "provider-request",
                "step_id": "provider-request",
                "provider_id": "test-provider"
            },
            "event": {"kind": "response_started"}
        })
        .to_string();
        ProviderEventPublicationCommit {
            run_id: "memory-run".to_owned(),
            owner_id: "user-1".to_owned(),
            conversation_id: conversation_id.clone(),
            namespace: namespace.clone(),
            provider_event_json: provider_event_json.clone(),
            graph_mutations: vec![GraphMutation {
                scope,
                table: "gke_nodes".to_owned(),
                entity_kind: "node".to_owned(),
                event_id: format!("graph-{event_id}"),
                op: "ADD".to_owned(),
                payload: json!({"id": format!("event-node-{event_id}")}),
                record: GraphRecord {
                    id: format!("event-node-{event_id}"),
                    document: None,
                    metadata,
                    embedding: None,
                },
                embedding_dim: 0,
            }],
            lane: NewProjectedLaneMessage {
                message_id: format!("msg:{event_id}"),
                namespace,
                purpose: "agent_event".to_owned(),
                inbox_id: "agent-provider-events".to_owned(),
                conversation_id,
                recipient_id: "agent-event-consumer".to_owned(),
                sender_id: "agent-runtime".to_owned(),
                msg_type: "agent.provider_event".to_owned(),
                status: "pending".to_owned(),
                created_at: now_seconds(),
                available_at: now_seconds(),
                run_id: Some("memory-run".to_owned()),
                step_id: Some("provider-request".to_owned()),
                correlation_id: Some(event_id.to_owned()),
                payload_json: Some(provider_event_json),
                error_json: None,
            },
            run_event_type: "workflow.provider_event.v1".to_owned(),
            run_event_payload_json: trace_payload,
        }
    }

    fn admitted_store() -> InMemoryStore {
        let store = InMemoryStore::new();
        let run = ServerRunCreate {
            run_id: "memory-run".to_owned(),
            conversation_id: "memory-conversation".to_owned(),
            workflow_id: "memory-workflow".to_owned(),
            user_id: Some("user-1".to_owned()),
            user_turn_node_id: "turn-1".to_owned(),
            status: "queued".to_owned(),
        };
        let mut transition = pending_start();
        transition.expected_event_seq = 1;
        transition.node_id = Some("noop".to_owned());
        store
            .admit_server_run_with_start(
                run,
                runtime_admission_payload(),
                transition,
                worker_lane(),
            )
            .unwrap();
        store
    }

    #[test]
    fn recorded_transition_is_atomic_replayable_and_projected_in_memory() {
        let store = InMemoryStore::new();
        let transition = start("start-1");
        let aborted = store
            .apply_recorded_runtime_transition(transition.clone(), true)
            .expect_err("injected abort must not commit staged writes");
        assert!(aborted.to_string().contains("aborted before atomic commit"));
        assert!(block_on(store.server_run("memory-run")).unwrap().is_none());
        assert!(
            block_on(store.server_run_events("memory-run", 0, 10))
                .unwrap()
                .is_empty()
        );
        assert!(
            block_on(store.named_projection("workflow_runtime_current_state", "memory-run"))
                .unwrap()
                .is_none()
        );

        let first = store
            .apply_recorded_runtime_transition(transition.clone(), false)
            .unwrap();
        assert_eq!(first.event_seq, 1);
        assert!(!first.idempotent);
        let replay = store
            .apply_recorded_runtime_transition(transition, false)
            .unwrap();
        assert!(replay.idempotent);
        assert_eq!(replay.event_seq, first.event_seq);
        assert_eq!(replay.reduced.state_digest, first.reduced.state_digest);

        let stale = store
            .apply_recorded_runtime_transition(start("stale-start"), false)
            .expect_err("stale event sequence must fail closed");
        assert!(
            stale
                .to_string()
                .contains("expected_event_seq 0, current 1")
        );

        let run = block_on(store.server_run("memory-run"))
            .unwrap()
            .expect("accepted start creates run registry row");
        assert_eq!(run.status, "running");
        let events = block_on(store.server_run_events("memory-run", 0, 10)).unwrap();
        assert_eq!(
            events.len(),
            1,
            "exact replay must not append another event"
        );
        let current =
            block_on(store.named_projection(RUNTIME_CURRENT_STATE_NAMESPACE, "memory-run"))
                .unwrap()
                .expect("accepted transition materializes current state");
        assert_eq!(current.payload["state"]["seed"], 1);
        assert_eq!(current.last_authoritative_seq, 1);
    }

    #[test]
    fn runtime_admission_atomically_records_start_and_first_lane() {
        let store = InMemoryStore::new();
        let run = ServerRunCreate {
            run_id: "memory-run".to_owned(),
            conversation_id: "memory-conversation".to_owned(),
            workflow_id: "memory-workflow".to_owned(),
            user_id: Some("user-1".to_owned()),
            user_turn_node_id: "turn-1".to_owned(),
            status: "queued".to_owned(),
        };
        let mut transition = pending_start();
        transition.expected_event_seq = 1;
        transition.node_id = Some("noop".to_owned());
        let lane = worker_lane();

        let mut invalid_transition = transition.clone();
        invalid_transition.expected_event_seq = 0;
        let failure = store
            .admit_server_run_with_start(
                run.clone(),
                runtime_admission_payload(),
                invalid_transition,
                lane.clone(),
            )
            .expect_err("invalid start must fail before any admission write");
        assert!(failure.to_string().contains("runtime admission"));
        assert!(block_on(store.server_run("memory-run")).unwrap().is_none());
        assert!(
            block_on(store.projected_lane_message("lane-noop-1"))
                .unwrap()
                .is_none()
        );

        let mut invalid_payload = runtime_admission_payload();
        invalid_payload["conversation_id"] = json!("other-conversation");
        assert!(
            store
                .admit_server_run_with_start(
                    run.clone(),
                    invalid_payload,
                    transition.clone(),
                    lane.clone(),
                )
                .is_err()
        );
        assert!(block_on(store.server_run("memory-run")).unwrap().is_none());

        let admitted = store
            .admit_server_run_with_start(
                run.clone(),
                runtime_admission_payload(),
                transition.clone(),
                lane.clone(),
            )
            .unwrap();
        assert_eq!(admitted.event_seq, 2);
        assert!(!admitted.idempotent);
        assert_eq!(
            block_on(store.server_run("memory-run"))
                .unwrap()
                .unwrap()
                .status,
            "running"
        );
        let events = block_on(store.server_run_events("memory-run", 0, 10)).unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].event_type, "run.created");
        assert_eq!(events[1].event_type, RECORDED_RUNTIME_EVENT_TYPE);
        let queued = block_on(store.projected_lane_message("lane-noop-1"))
            .unwrap()
            .unwrap();
        assert_eq!(queued.status, "pending");
        assert_eq!(queued.run_id.as_deref(), Some("memory-run"));

        let mut conflicting_lane = lane.clone();
        conflicting_lane.recipient_id = "other-worker".to_owned();
        assert!(
            store
                .admit_server_run_with_start(
                    run.clone(),
                    runtime_admission_payload(),
                    transition.clone(),
                    conflicting_lane,
                )
                .is_err()
        );
        let mut changed_payload = runtime_admission_payload();
        changed_payload["changed"] = json!(true);
        assert!(
            store
                .admit_server_run_with_start(
                    run.clone(),
                    changed_payload,
                    transition.clone(),
                    lane.clone(),
                )
                .is_err()
        );

        let replay = store
            .admit_server_run_with_start(run, runtime_admission_payload(), transition, lane)
            .unwrap();
        assert!(replay.idempotent);
        assert_eq!(
            block_on(store.server_run_events("memory-run", 0, 10))
                .unwrap()
                .len(),
            2
        );
    }

    #[test]
    fn provider_event_publication_commits_graph_lane_and_trace_atomically_and_idempotently() {
        let store = admitted_store();
        let publication = provider_publication("provider-event-1");
        assert!(
            store
                .persist_provider_event_publication(publication.clone())
                .unwrap()
        );
        assert!(
            !store
                .persist_provider_event_publication(publication.clone())
                .unwrap()
        );
        assert_eq!(
            block_on(store.projected_lane_message("msg:provider-event-1"))
                .unwrap()
                .unwrap()
                .status,
            "pending"
        );
        assert_eq!(
            block_on(store.server_run_events("memory-run", 0, 10))
                .unwrap()
                .iter()
                .filter(|event| event.event_type == "workflow.provider_event.v1")
                .count(),
            1
        );
        assert!(
            store
                .state
                .read()
                .unwrap()
                .graph_projections
                .values()
                .any(|projection| projection.records.contains_key(&(
                    "gke_nodes".to_owned(),
                    "event-node-provider-event-1".to_owned()
                )))
        );

        let mut conflicting_replay = publication;
        conflicting_replay.run_event_payload_json =
            json!({"event_id": "provider-event-1", "trace_id": "changed"}).to_string();
        assert!(
            store
                .persist_provider_event_publication(conflicting_replay)
                .is_err()
        );
        assert_eq!(
            block_on(store.server_run_events("memory-run", 0, 10))
                .unwrap()
                .iter()
                .filter(|event| event.event_type == "workflow.provider_event.v1")
                .count(),
            1
        );
    }

    #[test]
    fn invalid_provider_event_publication_leaves_no_partial_projection() {
        let store = admitted_store();
        let mut publication = provider_publication("provider-event-invalid");
        publication.graph_mutations[0].scope.namespace = "other-namespace".to_owned();
        assert!(
            store
                .persist_provider_event_publication(publication)
                .is_err()
        );
        assert!(
            block_on(store.projected_lane_message("msg:provider-event-invalid"))
                .unwrap()
                .is_none()
        );
        assert_eq!(
            block_on(store.server_run_events("memory-run", 0, 10))
                .unwrap()
                .len(),
            2
        );

        let mut mismatched_trace = provider_publication("provider-event-trace-mismatch");
        mismatched_trace.run_event_payload_json = json!({
            "event_id": "different-event"
        })
        .to_string();
        assert!(
            store
                .persist_provider_event_publication(mismatched_trace)
                .is_err()
        );
        assert!(
            block_on(store.projected_lane_message("msg:provider-event-trace-mismatch"))
                .unwrap()
                .is_none()
        );
        assert_eq!(
            block_on(store.server_run_events("memory-run", 0, 10))
                .unwrap()
                .len(),
            2
        );
    }

    #[test]
    fn claimed_noop_effect_and_lane_ack_are_atomic_and_idempotent() {
        let store = InMemoryStore::new();
        store
            .apply_recorded_runtime_transition(pending_start(), false)
            .unwrap();
        block_on(store.project_lane_message(worker_lane())).unwrap();
        let claimed = block_on(store.claim_projected_lane_messages_filtered(
            LaneMessageClaimFilter {
                namespace: "workflow".to_owned(),
                inbox_id: "workflow-runtime".to_owned(),
                claimed_by: "rust-native-worker#claim-1".to_owned(),
                message_ids: Some(vec!["lane-noop-1".to_owned()]),
                run_id: Some("memory-run".to_owned()),
                msg_type: Some("workflow.step.execute".to_owned()),
                recipient_id: Some("rust-native-worker".to_owned()),
                limit: 1,
                lease_seconds: 60,
            },
        ))
        .unwrap();
        assert_eq!(claimed.len(), 1);

        let denied = store
            .apply_claimed_recorded_worker_effect(
                handoff("rust-native-worker#stale"),
                noop_effect(),
            )
            .expect_err("a different claimant must not commit an effect");
        assert!(denied.to_string().contains("not actively claimed"));
        assert_eq!(
            block_on(store.projected_lane_message("lane-noop-1"))
                .unwrap()
                .unwrap()
                .status,
            "claimed"
        );

        let effect = noop_effect();
        let applied = store
            .apply_claimed_recorded_worker_effect(
                handoff("rust-native-worker#claim-1"),
                effect.clone(),
            )
            .unwrap();
        assert!(!applied.idempotent);
        assert_eq!(applied.reduced.state.status, RecordedRunStatus::Completed);
        assert_eq!(applied.reduced.state.state["seed"], 1);
        let lane = block_on(store.projected_lane_message("lane-noop-1"))
            .unwrap()
            .unwrap();
        assert_eq!(lane.status, "completed");
        assert!(lane.claimed_by.is_none());

        let replay = store
            .apply_claimed_recorded_worker_effect(handoff("rust-native-worker#claim-1"), effect)
            .unwrap();
        assert!(replay.idempotent);
        assert_eq!(replay.event_seq, applied.event_seq);
        let events = block_on(store.server_run_events("memory-run", 0, 10)).unwrap();
        assert_eq!(events.len(), 4, "start, worker step, completion, and trace");
        assert_eq!(events[3].event_type, "workflow.trace.v1");
        assert_eq!(
            block_on(store.server_run("memory-run"))
                .unwrap()
                .unwrap()
                .status,
            "succeeded"
        );

        let mut changed = noop_effect();
        changed.result = Some(json!({"different": true}));
        assert!(
            store
                .apply_claimed_recorded_worker_effect(
                    handoff("rust-native-worker#claim-1"),
                    changed
                )
                .is_err()
        );
        assert_eq!(
            block_on(store.server_run_events("memory-run", 0, 10))
                .unwrap()
                .len(),
            4
        );
    }
}
