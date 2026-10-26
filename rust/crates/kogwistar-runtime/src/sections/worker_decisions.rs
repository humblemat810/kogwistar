/// Worker-visible routing facts.  Workers may report selected successors, but
/// may not replace the durable scheduler frontier wholesale.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeWorkerSuccessEffect {
    #[serde(default)]
    pub successors: Vec<RuntimeSuccessor>,
}

/// Successful Python callback output. Scheduler/run/token identity stays in
/// the claimed lane and is therefore absent from this worker-owned payload.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RecordedWorkerSuccessEffect {
    pub contract_version: u32,
    pub effect_id: String,
    #[serde(default)]
    pub status: RuntimeWorkerEffectStatus,
    #[serde(default)]
    pub state_update: Vec<RecordedStateUpdate>,
    #[serde(default)]
    pub update: Option<Map<String, Value>>,
    #[serde(default)]
    pub state_schema: Map<String, Value>,
    #[serde(default)]
    pub successors: Vec<RuntimeSuccessor>,
    #[serde(default)]
    pub route_next: Vec<String>,
    #[serde(default)]
    pub result: Option<Value>,
    #[serde(default)]
    pub errors: Vec<String>,
    #[serde(default)]
    pub wait_reason: Option<String>,
    #[serde(default)]
    pub resume_payload: Option<Value>,
    #[serde(default)]
    pub usage: Option<Value>,
    #[serde(default)]
    pub trace_events: Vec<Value>,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeWorkerEffectStatus {
    #[default]
    Success,
    Suspended,
    Failed,
}

pub fn worker_effect_digest(
    effect: &RecordedWorkerSuccessEffect,
) -> Result<String, RecordedRuntimeError> {
    digest_value(&serde_json::to_value(effect)?)
}

/// Fold one successful worker effect into the Rust-owned durable frontier.
/// The executing token must exist in `pending`; all unrelated tokens, joins,
/// suspended work, and waiters remain authoritative and untouched.
pub fn frontier_after_worker_success(
    frontier: &RuntimeFrontier,
    node_id: &str,
    token_id: &str,
    parent_token_id: Option<&str>,
    step_seq: i64,
    effect: &RuntimeWorkerSuccessEffect,
) -> Result<RuntimeFrontier, RecordedRuntimeError> {
    let Some((index, (_, current_join_mask, _, _))) =
        frontier
            .pending
            .iter()
            .enumerate()
            .find(|(_, (node, _, token, parent))| {
                node == node_id && token == token_id && parent.as_deref() == parent_token_id
            })
    else {
        return Err(RecordedRuntimeError::TokenNotInFrontier);
    };
    let plan = plan_runtime_successors(&RuntimeSuccessorPlanRequest {
        token_id: token_id.to_owned(),
        parent_token_id: parent_token_id.map(str::to_owned),
        step_seq,
        current_join_mask: *current_join_mask,
        join_outstanding: frontier.join_outstanding.clone(),
        successors: effect.successors.clone(),
    });
    let mut next = frontier.clone();
    next.pending.remove(index);
    next.join_outstanding = plan.join_outstanding;
    for token in plan.tokens {
        let Some(join_index) = next
            .join_node_ids
            .iter()
            .position(|join_node_id| join_node_id == &token.node_id)
        else {
            next.pending.push((
                token.node_id,
                token.join_mask,
                token.token_id,
                token.parent_token_id,
            ));
            continue;
        };
        let arrival = apply_runtime_join_arrival(&RuntimeJoinArrivalRequest {
            join_index,
            join_outstanding: next.join_outstanding.clone(),
            waiters: next
                .join_waiters
                .get(&token.node_id)
                .into_iter()
                .flatten()
                .map(|(join_mask, token_id, parent_token_id)| RuntimeJoinWaiter {
                    join_mask: *join_mask,
                    token_id: token_id.clone(),
                    parent_token_id: parent_token_id.clone(),
                })
                .collect(),
            arrival: RuntimeJoinWaiter {
                join_mask: token.join_mask,
                token_id: token.token_id,
                parent_token_id: token.parent_token_id,
            },
            merge: true,
        });
        next.join_outstanding = arrival.join_outstanding;
        next.join_waiters.insert(
            token.node_id.clone(),
            arrival
                .waiters
                .into_iter()
                .map(|waiter| (waiter.join_mask, waiter.token_id, waiter.parent_token_id))
                .collect(),
        );
        if let Some(released) = arrival.released {
            next.pending.push((
                token.node_id,
                released.join_mask,
                released.token_id,
                released.parent_token_id,
            ));
        }
    }
    next.normalize()
}

pub fn frontier_after_worker_suspend(
    frontier: &RuntimeFrontier,
    node_id: &str,
    token_id: &str,
    parent_token_id: Option<&str>,
) -> Result<RuntimeFrontier, RecordedRuntimeError> {
    let Some(index) = frontier
        .pending
        .iter()
        .position(|(node, _, token, parent)| {
            node == node_id && token == token_id && parent.as_deref() == parent_token_id
        })
    else {
        return Err(RecordedRuntimeError::TokenNotInFrontier);
    };
    let mut next = frontier.clone();
    let token = next.pending.remove(index);
    next.suspended.push(token);
    next.normalize()
}

pub fn frontier_after_worker_resume(
    frontier: &RuntimeFrontier,
    node_id: &str,
    token_id: &str,
    parent_token_id: Option<&str>,
) -> Result<RuntimeFrontier, RecordedRuntimeError> {
    let Some(index) = frontier
        .suspended
        .iter()
        .position(|(node, _, token, parent)| {
            node == node_id && token == token_id && parent.as_deref() == parent_token_id
        })
    else {
        return Err(RecordedRuntimeError::TokenNotInFrontier);
    };
    let mut next = frontier.clone();
    let token = next.suspended.remove(index);
    next.pending.push(token);
    next.normalize()
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeJoinWaiter {
    pub join_mask: i64,
    pub token_id: String,
    #[serde(default)]
    pub parent_token_id: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeJoinArrivalRequest {
    pub join_index: usize,
    pub join_outstanding: Vec<i64>,
    #[serde(default)]
    pub waiters: Vec<RuntimeJoinWaiter>,
    pub arrival: RuntimeJoinWaiter,
    #[serde(default)]
    pub merge: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeJoinArrival {
    pub join_outstanding: Vec<i64>,
    pub waiters: Vec<RuntimeJoinWaiter>,
    #[serde(default)]
    pub released: Option<RuntimeJoinWaiter>,
    pub collapsed_work_items: usize,
    pub outstanding: i64,
}

pub fn apply_runtime_join_arrival(request: &RuntimeJoinArrivalRequest) -> RuntimeJoinArrival {
    let mut join_outstanding = request.join_outstanding.clone();
    let join_bit = if request.join_index < i64::BITS as usize {
        1_i64 << request.join_index
    } else {
        0
    };
    let mut arrival = request.arrival.clone();
    if join_bit != 0 && arrival.join_mask & join_bit != 0 {
        adjust_join_outstanding(&mut join_outstanding, join_bit, -1);
        arrival.join_mask &= !join_bit;
    }
    let mut waiters = request.waiters.clone();
    waiters.push(arrival.clone());
    let outstanding = join_outstanding
        .get(request.join_index)
        .copied()
        .unwrap_or(0);
    if !request.merge || outstanding != 0 {
        return RuntimeJoinArrival {
            join_outstanding,
            waiters,
            released: (!request.merge).then_some(arrival),
            collapsed_work_items: 0,
            outstanding,
        };
    }

    let Some(first) = waiters.first().cloned() else {
        return RuntimeJoinArrival {
            join_outstanding,
            waiters,
            released: None,
            collapsed_work_items: 0,
            outstanding,
        };
    };
    let mut merged_mask = first.join_mask;
    for waiter in &waiters {
        adjust_join_outstanding(&mut join_outstanding, waiter.join_mask, -1);
    }
    for waiter in waiters.iter().skip(1) {
        merged_mask &= waiter.join_mask;
    }
    adjust_join_outstanding(&mut join_outstanding, merged_mask, 1);
    RuntimeJoinArrival {
        join_outstanding,
        waiters: Vec::new(),
        released: Some(RuntimeJoinWaiter {
            join_mask: merged_mask,
            ..first
        }),
        collapsed_work_items: waiters.len().saturating_sub(1),
        outstanding,
    }
}

pub fn apply_runtime_join_arrival_from_str(
    payload_json: &str,
) -> Result<String, serde_json::Error> {
    let request: RuntimeJoinArrivalRequest = serde_json::from_str(payload_json)?;
    serde_json::to_string(&apply_runtime_join_arrival(&request))
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeRetryDecisionRequest {
    pub retry_budget: u64,
    pub attempt_number: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeRetryDecision {
    pub retry_budget: u64,
    pub attempt_number: u64,
    pub should_retry: bool,
    pub exhausted: bool,
    pub next_attempt_number: Option<u64>,
}

pub fn decide_runtime_retry(request: &RuntimeRetryDecisionRequest) -> RuntimeRetryDecision {
    let retry_budget = request.retry_budget.max(1);
    let should_retry = request.attempt_number < retry_budget;
    RuntimeRetryDecision {
        retry_budget,
        attempt_number: request.attempt_number,
        should_retry,
        exhausted: !should_retry,
        next_attempt_number: should_retry.then(|| request.attempt_number.saturating_add(1)),
    }
}

pub fn decide_runtime_retry_from_str(payload_json: &str) -> Result<String, serde_json::Error> {
    let request: RuntimeRetryDecisionRequest = serde_json::from_str(payload_json)?;
    serde_json::to_string(&decide_runtime_retry(&request))
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeNestedInvocationPlanRequest {
    pub parent_run_id: String,
    pub workflow_id: String,
    #[serde(default)]
    pub result_state_key: Option<String>,
    #[serde(default)]
    pub run_id: Option<String>,
    #[serde(default)]
    pub invocation_key: Option<String>,
    pub parent_conversation_id: String,
    #[serde(default)]
    pub conversation_id: Option<String>,
    pub parent_turn_node_id: String,
    #[serde(default)]
    pub turn_node_id: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeNestedInvocationPlan {
    pub child_run_id: String,
    pub conversation_id: String,
    pub turn_node_id: String,
    pub result_state_key: String,
}

pub fn plan_runtime_nested_invocation(
    request: &RuntimeNestedInvocationPlanRequest,
) -> RuntimeNestedInvocationPlan {
    let result_state_key = request
        .result_state_key
        .clone()
        .unwrap_or_else(|| format!("workflow_result::{}", request.workflow_id));
    let effective_turn_node_id = request
        .turn_node_id
        .clone()
        .unwrap_or_else(|| request.parent_turn_node_id.clone());
    let child_run_id = request.run_id.clone().unwrap_or_else(|| {
        kogwistar_contracts::stable_id(
            "workflow.child_run",
            &[
                request.parent_run_id.clone(),
                request.workflow_id.clone(),
                request.result_state_key.clone().unwrap_or_default(),
                effective_turn_node_id.clone(),
                request.invocation_key.clone().unwrap_or_default(),
            ],
        )
        .to_string()
    });
    RuntimeNestedInvocationPlan {
        child_run_id,
        conversation_id: request
            .conversation_id
            .clone()
            .unwrap_or_else(|| request.parent_conversation_id.clone()),
        turn_node_id: effective_turn_node_id,
        result_state_key,
    }
}

pub fn plan_runtime_nested_invocation_from_str(
    payload_json: &str,
) -> Result<String, serde_json::Error> {
    let request: RuntimeNestedInvocationPlanRequest = serde_json::from_str(payload_json)?;
    serde_json::to_string(&plan_runtime_nested_invocation(&request))
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeDispatchDecisionRequest {
    pub max_workers: i64,
    pub inflight: usize,
    pub pending: usize,
    #[serde(default)]
    pub cancelling: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeDispatchDecision {
    pub worker_limit: usize,
    pub launch_capacity: usize,
    pub should_launch: bool,
    pub should_drain: bool,
    pub cancellation_complete: bool,
}

pub fn decide_runtime_dispatch(
    request: &RuntimeDispatchDecisionRequest,
) -> RuntimeDispatchDecision {
    let worker_limit = usize::try_from(request.max_workers.max(1)).unwrap_or(usize::MAX);
    let launch_capacity = if request.cancelling {
        0
    } else {
        worker_limit
            .saturating_sub(request.inflight)
            .min(request.pending)
    };
    RuntimeDispatchDecision {
        worker_limit,
        launch_capacity,
        should_launch: launch_capacity > 0,
        should_drain: request.cancelling && request.inflight > 0,
        cancellation_complete: request.cancelling && request.inflight == 0 && request.pending == 0,
    }
}

pub fn decide_runtime_dispatch_from_str(payload_json: &str) -> Result<String, serde_json::Error> {
    let request: RuntimeDispatchDecisionRequest = serde_json::from_str(payload_json)?;
    serde_json::to_string(&decide_runtime_dispatch(&request))
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeBudgetSuspendRequest {
    pub token_budget: i64,
    pub token_used: i64,
    pub time_budget_ms: i64,
    pub time_used_ms: i64,
    pub rate_limit: i64,
    pub rate_used: i64,
    pub step_budget: i64,
    pub step_used: i64,
    pub call_budget: i64,
    pub call_used: i64,
    pub cost_budget: f64,
    pub cost_used: f64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeBudgetSuspendDecision {
    pub should_suspend: bool,
    #[serde(default)]
    pub reason: Option<String>,
}

pub fn decide_runtime_budget_suspend(
    request: &RuntimeBudgetSuspendRequest,
) -> RuntimeBudgetSuspendDecision {
    let reason = [
        (
            request.token_budget > 0 && request.token_used >= request.token_budget,
            "token",
        ),
        (
            request.time_budget_ms > 0 && request.time_used_ms >= request.time_budget_ms,
            "time",
        ),
        (
            request.rate_limit > 0 && request.rate_used >= request.rate_limit,
            "rate",
        ),
        (
            request.step_budget > 0 && request.step_used >= request.step_budget,
            "step",
        ),
        (
            request.call_budget > 0 && request.call_used >= request.call_budget,
            "call",
        ),
        (
            request.cost_budget > 0.0 && request.cost_used >= request.cost_budget,
            "cost",
        ),
    ]
    .into_iter()
    .find_map(|(exhausted, reason)| exhausted.then(|| reason.to_owned()));
    RuntimeBudgetSuspendDecision {
        should_suspend: reason.is_some(),
        reason,
    }
}

pub fn decide_runtime_budget_suspend_from_str(
    payload_json: &str,
) -> Result<String, serde_json::Error> {
    let request: RuntimeBudgetSuspendRequest = serde_json::from_str(payload_json)?;
    serde_json::to_string(&decide_runtime_budget_suspend(&request))
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeSchedulerToken {
    pub node_id: String,
    pub join_mask: i64,
    pub token_id: String,
    #[serde(default)]
    pub parent_token_id: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeSchedulerTickRequest {
    #[serde(default)]
    pub pending: Vec<RuntimeSchedulerToken>,
    pub inflight: usize,
    pub max_workers: i64,
    #[serde(default)]
    pub cancelling: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeSchedulerTick {
    pub dispatch: Vec<RuntimeSchedulerToken>,
    pub pending: Vec<RuntimeSchedulerToken>,
    pub should_drain: bool,
    pub cancellation_complete: bool,
}

pub fn tick_runtime_scheduler(request: &RuntimeSchedulerTickRequest) -> RuntimeSchedulerTick {
    let decision = decide_runtime_dispatch(&RuntimeDispatchDecisionRequest {
        max_workers: request.max_workers,
        inflight: request.inflight,
        pending: request.pending.len(),
        cancelling: request.cancelling,
    });
    let mut pending = request.pending.clone();
    let dispatch_count = decision.launch_capacity.min(pending.len());
    let dispatch = pending.drain(..dispatch_count).collect();
    RuntimeSchedulerTick {
        dispatch,
        pending,
        should_drain: decision.should_drain,
        cancellation_complete: decision.cancellation_complete,
    }
}

pub fn tick_runtime_scheduler_from_str(payload_json: &str) -> Result<String, serde_json::Error> {
    let request: RuntimeSchedulerTickRequest = serde_json::from_str(payload_json)?;
    serde_json::to_string(&tick_runtime_scheduler(&request))
}
