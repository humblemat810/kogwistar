//! Recorded-only workflow transition reducer for ADR-015 Phase 4.
//!
//! This crate deliberately has no resolver, provider, graph, lane, or Python
//! dependency.  A caller supplies a result that was already recorded by the
//! authoritative worker.  The reducer validates and folds only durable state.

use kogwistar_contracts::canonical_json;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

pub const RECORDED_RUNTIME_CONTRACT_VERSION: u32 = 1;

#[derive(Debug, Error)]
pub enum RecordedRuntimeError {
    #[error("unsupported recorded runtime contract version {got}; expected {expected}")]
    UnsupportedContractVersion { got: u32, expected: u32 },
    #[error("recorded runtime transition id must not be empty")]
    EmptyTransitionId,
    #[error("recorded runtime {field} must not be empty")]
    EmptyIdentifier { field: &'static str },
    #[error("recorded runtime step sequence must not be negative")]
    NegativeStepSequence,
    #[error("recorded runtime transition {kind:?} requires {field}")]
    MissingField {
        kind: RecordedTransitionKind,
        field: &'static str,
    },
    #[error("recorded runtime transition start requires no prior runtime state")]
    StartAlreadyExists,
    #[error("recorded runtime transition {kind:?} requires prior runtime state")]
    MissingPriorState { kind: RecordedTransitionKind },
    #[error("recorded runtime {field} mismatch: expected {expected:?}, got {got:?}")]
    IdentityMismatch {
        field: &'static str,
        expected: String,
        got: String,
    },
    #[error("recorded runtime transition {kind:?} is illegal from status {status:?}")]
    IllegalStatusTransition {
        kind: RecordedTransitionKind,
        status: RecordedRunStatus,
    },
    #[error("recorded runtime terminal state {status:?} is immutable")]
    TerminalImmutable { status: RecordedRunStatus },
    #[error("recorded runtime step sequence regresses: current {current}, requested {requested}")]
    StepSequenceRegresses { current: i64, requested: i64 },
    #[error(
        "recorded runtime step sequence must advance: current {current}, requested {requested}"
    )]
    StepSequenceDoesNotAdvance { current: i64, requested: i64 },
    #[error("recorded runtime state_update and update cannot both be supplied")]
    StateUpdateConflict,
    #[error("recorded runtime state update {mode:?} is unsupported")]
    UnsupportedStateUpdate { mode: String },
    #[error("recorded runtime state field {field:?} must be an array for {operation}")]
    StateFieldNotArray {
        field: String,
        operation: &'static str,
    },
    #[error("recorded runtime state update value for {field:?} cannot be extended")]
    StateValueNotExtendable { field: String },
    #[error("recorded runtime frontier {field} is invalid: {detail}")]
    InvalidFrontier { field: &'static str, detail: String },
    #[error("recorded runtime transition token/node identity is not present in required frontier")]
    TokenNotInFrontier,
    #[error("recorded runtime worker successor selection violates authoritative graph: {0}")]
    InvalidSuccessorSelection(String),
    #[error("recorded runtime terminal transition must not retain unfinished token or join work")]
    TerminalFrontierNotEmpty,
    #[error("cannot encode recorded runtime JSON: {0}")]
    Json(#[from] serde_json::Error),
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeRouteEdge {
    pub edge_id: String,
    #[serde(default)]
    pub target_ids: Vec<String>,
    #[serde(default)]
    pub aliases: Vec<String>,
    #[serde(default)]
    pub predicate: Option<String>,
    #[serde(default = "default_route_multiplicity")]
    pub multiplicity: String,
    #[serde(default)]
    pub is_default: bool,
    #[serde(default = "default_route_priority")]
    pub priority: i64,
    #[serde(default)]
    pub predicate_result: Option<bool>,
    #[serde(default)]
    pub base_result: Option<bool>,
}

fn default_route_multiplicity() -> String {
    "one".to_owned()
}

fn default_route_priority() -> i64 {
    100
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeRouteRequest {
    #[serde(default)]
    pub edges: Vec<RuntimeRouteEdge>,
    #[serde(default)]
    pub explicit_next: Vec<String>,
    #[serde(default)]
    pub fanout: bool,
    #[serde(default)]
    pub failure_only: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeRouteDecision {
    pub next_node_ids: Vec<String>,
    pub selected_edge_indices: Vec<usize>,
    pub evaluated: Vec<(String, bool)>,
    pub selected: Vec<(String, String, String)>,
}

fn route_first_target(edge: &RuntimeRouteEdge) -> Option<&str> {
    edge.target_ids.first().map(String::as_str)
}

fn route_stop_on_first(edge: &RuntimeRouteEdge, fanout: bool) -> bool {
    !fanout && edge.multiplicity != "many"
}

pub fn select_runtime_route(request: &RuntimeRouteRequest) -> RuntimeRouteDecision {
    let mut evaluated = Vec::new();
    let mut selected = Vec::new();

    if !request.explicit_next.is_empty() {
        let mut next_node_ids = Vec::new();
        let mut selected_edge_indices = Vec::new();
        for alias in &request.explicit_next {
            let matched = request.edges.iter().enumerate().find_map(|(index, edge)| {
                let target = route_first_target(edge)?;
                edge.aliases
                    .iter()
                    .any(|candidate| candidate == alias)
                    .then_some((index, edge, target))
            });
            evaluated.push((format!("_route_next:{alias}"), matched.is_some()));
            if let Some((index, edge, target)) = matched {
                selected.push((
                    edge.edge_id.clone(),
                    target.to_owned(),
                    "explicit".to_owned(),
                ));
                selected_edge_indices.push(index);
                next_node_ids.push(target.to_owned());
            }
        }
        if !next_node_ids.is_empty() {
            return RuntimeRouteDecision {
                next_node_ids,
                selected_edge_indices,
                evaluated,
                selected,
            };
        }
    }

    let mut matched = Vec::new();
    for (index, edge) in request.edges.iter().enumerate() {
        let Some(predicate) = edge.predicate.as_deref() else {
            continue;
        };
        let Some(target) = route_first_target(edge) else {
            continue;
        };
        let ok = edge.predicate_result.unwrap_or(false);
        evaluated.push((format!("{}:{predicate}", edge.edge_id), ok));
        if ok {
            matched.push((index, edge, target));
            selected.push((
                edge.edge_id.clone(),
                target.to_owned(),
                "predicate".to_owned(),
            ));
        }
    }
    matched.sort_by(|left, right| right.1.priority.cmp(&left.1.priority));

    let mut candidate_indices = Vec::new();
    let mut candidate_ids = Vec::new();
    for (index, edge, target) in matched {
        if route_stop_on_first(edge, request.fanout) {
            if candidate_ids.is_empty() {
                candidate_indices.push(index);
                candidate_ids.push(target.to_owned());
            }
            return RuntimeRouteDecision {
                next_node_ids: candidate_ids,
                selected_edge_indices: candidate_indices,
                evaluated,
                selected,
            };
        }
        candidate_indices.push(index);
        candidate_ids.push(target.to_owned());
    }
    if !candidate_ids.is_empty() {
        return RuntimeRouteDecision {
            next_node_ids: candidate_ids,
            selected_edge_indices: candidate_indices,
            evaluated,
            selected,
        };
    }
    if request.failure_only {
        return RuntimeRouteDecision {
            next_node_ids: Vec::new(),
            selected_edge_indices: Vec::new(),
            evaluated,
            selected,
        };
    }

    let mut base_matches = Vec::new();
    for (index, edge) in request.edges.iter().enumerate() {
        if edge.predicate.is_some() {
            continue;
        }
        let Some(target) = route_first_target(edge) else {
            continue;
        };
        let ok = edge.base_result.unwrap_or(false);
        evaluated.push((format!("{}:<base>", edge.edge_id), ok));
        if ok {
            base_matches.push((index, edge, target));
            selected.push((edge.edge_id.clone(), target.to_owned(), "base".to_owned()));
            if route_stop_on_first(edge, request.fanout) {
                return RuntimeRouteDecision {
                    next_node_ids: vec![target.to_owned()],
                    selected_edge_indices: vec![index],
                    evaluated,
                    selected,
                };
            }
        }
    }
    if !base_matches.is_empty() {
        let allow_many = request.fanout
            || base_matches
                .iter()
                .any(|(_, edge, _)| edge.multiplicity == "many");
        let count = if allow_many { base_matches.len() } else { 1 };
        return RuntimeRouteDecision {
            next_node_ids: base_matches
                .iter()
                .take(count)
                .map(|(_, _, target)| (*target).to_owned())
                .collect(),
            selected_edge_indices: base_matches
                .iter()
                .take(count)
                .map(|(index, _, _)| *index)
                .collect(),
            evaluated,
            selected,
        };
    }

    for (index, edge) in request.edges.iter().enumerate() {
        if !edge.is_default || edge.target_ids.is_empty() {
            continue;
        }
        let count = if request.fanout {
            edge.target_ids.len()
        } else {
            1
        };
        let next_node_ids = edge.target_ids.iter().take(count).cloned().collect();
        selected.push((
            edge.edge_id.clone(),
            edge.target_ids[0].clone(),
            "default".to_owned(),
        ));
        return RuntimeRouteDecision {
            next_node_ids,
            selected_edge_indices: vec![index],
            evaluated,
            selected,
        };
    }

    RuntimeRouteDecision {
        next_node_ids: Vec::new(),
        selected_edge_indices: Vec::new(),
        evaluated,
        selected,
    }
}

pub fn select_runtime_route_from_str(payload_json: &str) -> Result<String, serde_json::Error> {
    let request: RuntimeRouteRequest = serde_json::from_str(payload_json)?;
    serde_json::to_string(&select_runtime_route(&request))
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeSuccessor {
    pub node_id: String,
    pub join_mask: i64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeStaticRoute {
    pub source_node_id: String,
    pub target_node_id: String,
    #[serde(default)]
    pub join_mask: i64,
    #[serde(default)]
    pub predicate: Option<String>,
    #[serde(default = "default_route_multiplicity")]
    pub multiplicity: String,
    #[serde(default)]
    pub source_fanout: bool,
}

/// Resolve one worker result against the frozen workflow graph.
///
/// Predicate functions remain Python worker code during the strangler window,
/// so a predicate-bearing source accepts the worker's selected successors only
/// after validating them against the exact frozen target and join mask. Static
/// sources remain scheduler-owned and ignore worker-reported routing entirely.
pub fn authoritative_runtime_successors(
    static_routes: &[RuntimeStaticRoute],
    source_node_id: &str,
    worker_selected: &[RuntimeSuccessor],
) -> Result<Vec<RuntimeSuccessor>, RecordedRuntimeError> {
    if static_routes.is_empty() {
        return Ok(worker_selected.to_vec());
    }
    let outgoing = static_routes
        .iter()
        .filter(|route| route.source_node_id == source_node_id)
        .collect::<Vec<_>>();
    if !outgoing.iter().any(|route| route.predicate.is_some()) {
        return Ok(outgoing
            .into_iter()
            .map(|route| RuntimeSuccessor {
                node_id: route.target_node_id.clone(),
                join_mask: route.join_mask,
            })
            .collect());
    }

    let mut selected_routes = Vec::with_capacity(worker_selected.len());
    for successor in worker_selected {
        let Some(route) = outgoing.iter().find(|route| {
            route.target_node_id == successor.node_id && route.join_mask == successor.join_mask
        }) else {
            return Err(RecordedRuntimeError::InvalidSuccessorSelection(format!(
                "source {source_node_id:?} has no route to {:?} with join mask {}",
                successor.node_id, successor.join_mask
            )));
        };
        if selected_routes
            .iter()
            .any(|selected: &&RuntimeStaticRoute| selected.target_node_id == route.target_node_id)
        {
            return Err(RecordedRuntimeError::InvalidSuccessorSelection(format!(
                "source {source_node_id:?} selected target {:?} more than once",
                route.target_node_id
            )));
        }
        selected_routes.push(*route);
    }
    if selected_routes.len() > 1
        && !selected_routes.iter().any(|route| route.source_fanout)
        && !selected_routes
            .iter()
            .all(|route| route.multiplicity == "many")
    {
        return Err(RecordedRuntimeError::InvalidSuccessorSelection(format!(
            "source {source_node_id:?} selected multiple non-fanout routes"
        )));
    }
    Ok(worker_selected.to_vec())
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeSuccessorPlanRequest {
    pub token_id: String,
    #[serde(default)]
    pub parent_token_id: Option<String>,
    pub step_seq: i64,
    pub current_join_mask: i64,
    pub join_outstanding: Vec<i64>,
    #[serde(default)]
    pub successors: Vec<RuntimeSuccessor>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimePlannedToken {
    pub node_id: String,
    pub join_mask: i64,
    pub token_id: String,
    #[serde(default)]
    pub parent_token_id: Option<String>,
    pub spawned: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeSuccessorPlan {
    pub tokens: Vec<RuntimePlannedToken>,
    pub join_outstanding: Vec<i64>,
}

fn adjust_join_outstanding(values: &mut [i64], mask: i64, delta: i64) {
    if mask <= 0 {
        return;
    }
    for (index, value) in values.iter_mut().enumerate() {
        if index < i64::BITS as usize && mask & (1_i64 << index) != 0 {
            *value = (*value + delta).max(0);
        }
    }
}

pub fn plan_runtime_successors(request: &RuntimeSuccessorPlanRequest) -> RuntimeSuccessorPlan {
    let mut join_outstanding = request.join_outstanding.clone();
    let mut tokens = Vec::with_capacity(request.successors.len());
    for (index, successor) in request.successors.iter().enumerate() {
        if index == 0 {
            let leaving = request.current_join_mask & !successor.join_mask;
            let gained = successor.join_mask & !request.current_join_mask;
            adjust_join_outstanding(&mut join_outstanding, leaving, -1);
            adjust_join_outstanding(&mut join_outstanding, gained, 1);
            tokens.push(RuntimePlannedToken {
                node_id: successor.node_id.clone(),
                join_mask: successor.join_mask,
                token_id: request.token_id.clone(),
                parent_token_id: request.parent_token_id.clone(),
                spawned: false,
            });
        } else {
            adjust_join_outstanding(&mut join_outstanding, successor.join_mask, 1);
            let token_key = format!(
                "{}/{}:{index}:{}",
                request.token_id, request.step_seq, successor.node_id
            );
            tokens.push(RuntimePlannedToken {
                node_id: successor.node_id.clone(),
                join_mask: successor.join_mask,
                token_id: kogwistar_contracts::stable_id("token_id", &[token_key])
                    .simple()
                    .to_string(),
                parent_token_id: Some(request.token_id.clone()),
                spawned: true,
            });
        }
    }
    RuntimeSuccessorPlan {
        tokens,
        join_outstanding,
    }
}

pub fn plan_runtime_successors_from_str(payload_json: &str) -> Result<String, serde_json::Error> {
    let request: RuntimeSuccessorPlanRequest = serde_json::from_str(payload_json)?;
    serde_json::to_string(&plan_runtime_successors(&request))
}

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

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RecordedTransitionKind {
    Start,
    RecordedStepSuccess,
    Suspend,
    ResumeResult,
    Cancel,
    Complete,
    Fail,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RecordedRunStatus {
    Running,
    Suspended,
    Completed,
    Failed,
    Cancelled,
}

impl RecordedRunStatus {
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }

    pub fn server_status(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Suspended => "suspended",
            // Server-run registry keeps its older success spelling while the
            // runtime projection exposes Python's workflow "completed".
            Self::Completed => "succeeded",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }
}

/// Python state updates arrive as either `[mode, payload]` pairs or the
/// equivalent object.  Accepting both keeps this private ABI JSON-only while
/// preserving the public runtime's established pair shape.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum RecordedStateUpdate {
    Pair((String, Map<String, Value>)),
    Object {
        mode: String,
        payload: Map<String, Value>,
    },
}

impl RecordedStateUpdate {
    fn parts(&self) -> (&str, &Map<String, Value>) {
        match self {
            Self::Pair((mode, payload)) => (mode, payload),
            Self::Object { mode, payload } => (mode, payload),
        }
    }
}

/// Restart frontier. Tuple encoding is intentional: Python checkpoints encode
/// token and waiter tuples as JSON arrays in exactly this field order.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeFrontier {
    #[serde(default)]
    pub pending: Vec<(String, i64, String, Option<String>)>,
    #[serde(default)]
    pub suspended: Vec<(String, i64, String, Option<String>)>,
    #[serde(default)]
    pub join_node_ids: Vec<String>,
    #[serde(default)]
    pub join_outstanding: Vec<i64>,
    #[serde(default)]
    pub join_waiters: BTreeMap<String, Vec<(i64, String, Option<String>)>>,
}

impl RuntimeFrontier {
    fn normalize(mut self) -> Result<Self, RecordedRuntimeError> {
        let mut joins = BTreeSet::new();
        for join in &self.join_node_ids {
            require_nonempty(join, "frontier.join_node_ids")?;
            if !joins.insert(join.clone()) {
                return Err(RecordedRuntimeError::InvalidFrontier {
                    field: "join_node_ids",
                    detail: format!("duplicate join node {join:?}"),
                });
            }
        }
        if self.join_outstanding.len() != self.join_node_ids.len() {
            return Err(RecordedRuntimeError::InvalidFrontier {
                field: "join_outstanding",
                detail: "length must equal join_node_ids length".to_owned(),
            });
        }
        if self.join_outstanding.iter().any(|value| *value < 0) {
            return Err(RecordedRuntimeError::InvalidFrontier {
                field: "join_outstanding",
                detail: "values must be non-negative".to_owned(),
            });
        }
        for key in self.join_waiters.keys() {
            if !joins.contains(key) {
                return Err(RecordedRuntimeError::InvalidFrontier {
                    field: "join_waiters",
                    detail: format!("unknown join node {key:?}"),
                });
            }
        }
        for join in &self.join_node_ids {
            let waiters = self.join_waiters.entry(join.clone()).or_default();
            for (mask, token_id, parent_token_id) in waiters.iter() {
                if *mask < 0 {
                    return Err(RecordedRuntimeError::InvalidFrontier {
                        field: "join_waiters",
                        detail: "mask must be non-negative".to_owned(),
                    });
                }
                require_nonempty(token_id, "frontier.join_waiters.token_id")?;
                if parent_token_id.as_deref().is_some_and(str::is_empty) {
                    return Err(RecordedRuntimeError::InvalidFrontier {
                        field: "join_waiters",
                        detail: "parent_token_id must not be empty".to_owned(),
                    });
                }
            }
            waiters.sort();
        }

        let mut token_ids = BTreeSet::new();
        for (node_id, mask, token_id, parent_token_id) in
            self.pending.iter().chain(self.suspended.iter())
        {
            require_nonempty(node_id, "frontier.token.node_id")?;
            require_nonempty(token_id, "frontier.token.token_id")?;
            if *mask < 0 {
                return Err(RecordedRuntimeError::InvalidFrontier {
                    field: "pending/suspended",
                    detail: "mask must be non-negative".to_owned(),
                });
            }
            if parent_token_id.as_deref().is_some_and(str::is_empty) {
                return Err(RecordedRuntimeError::InvalidFrontier {
                    field: "pending/suspended",
                    detail: "parent_token_id must not be empty".to_owned(),
                });
            }
            if !token_ids.insert(token_id.clone()) {
                return Err(RecordedRuntimeError::InvalidFrontier {
                    field: "pending/suspended",
                    detail: format!("token {token_id:?} appears more than once"),
                });
            }
        }
        self.pending.sort();
        self.suspended.sort();
        Ok(self)
    }

    fn contains_suspended(
        &self,
        node_id: &str,
        token_id: &str,
        parent_token_id: Option<&str>,
    ) -> bool {
        self.suspended.iter().any(|(node, _, token, parent)| {
            node == node_id && token == token_id && parent.as_deref() == parent_token_id
        })
    }

    fn contains_pending(
        &self,
        node_id: &str,
        token_id: &str,
        parent_token_id: Option<&str>,
    ) -> bool {
        self.pending.iter().any(|(node, _, token, parent)| {
            node == node_id && token == token_id && parent.as_deref() == parent_token_id
        })
    }

    fn has_unfinished_work(&self) -> bool {
        !self.pending.is_empty()
            || !self.suspended.is_empty()
            || self.join_outstanding.iter().any(|count| *count != 0)
            || self
                .join_waiters
                .values()
                .any(|waiters| !waiters.is_empty())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RecordedRuntimeTransition {
    pub contract_version: u32,
    pub transition_id: String,
    /// Last accepted server-run-event sequence for this run.  SQLite checks it
    /// inside the same `BEGIN IMMEDIATE` transaction that appends the event.
    pub expected_event_seq: i64,
    pub kind: RecordedTransitionKind,
    pub run_id: String,
    pub workflow_id: String,
    pub conversation_id: String,
    #[serde(default)]
    pub user_id: Option<String>,
    #[serde(default)]
    pub user_turn_node_id: Option<String>,
    /// Python step sequence. Start's projection may use zero, while durable
    /// reducer state begins at -1 so the first recorded step can be zero.
    pub step_seq: i64,
    #[serde(default)]
    pub node_id: Option<String>,
    #[serde(default)]
    pub token_id: Option<String>,
    #[serde(default)]
    pub parent_token_id: Option<String>,
    #[serde(default)]
    pub initial_state: Option<Map<String, Value>>,
    #[serde(default)]
    pub state_update: Vec<RecordedStateUpdate>,
    #[serde(default)]
    pub update: Option<Map<String, Value>>,
    #[serde(default)]
    pub state_schema: Map<String, Value>,
    #[serde(default)]
    pub frontier: Option<RuntimeFrontier>,
    #[serde(default)]
    pub result: Option<Value>,
    #[serde(default)]
    pub wait_reason: Option<String>,
    #[serde(default)]
    pub resume_payload: Option<Value>,
    #[serde(default)]
    pub errors: Vec<String>,
}

/// Immutable identity of a Python worker result handoff.  The worker result
/// itself remains a `RecordedRuntimeTransition`; this value binds it to the
/// durable lane request which was claimed before the result was submitted.
///
/// This is deliberately data-only.  It cannot carry a callback, resolver, or
/// provider handle across the runtime/store boundary.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RecordedWorkerHandoff {
    pub message_id: String,
    pub claimed_by: String,
    pub run_id: String,
    pub step_id: String,
    pub correlation_id: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RecordedRuntimeState {
    pub contract_version: u32,
    pub run_id: String,
    pub workflow_id: String,
    pub conversation_id: String,
    pub status: RecordedRunStatus,
    pub last_step_seq: i64,
    pub state: Map<String, Value>,
    pub frontier: RuntimeFrontier,
    #[serde(default)]
    pub static_routes: Vec<RuntimeStaticRoute>,
    #[serde(default)]
    pub wait_reason: Option<String>,
    #[serde(default)]
    pub resume_payload: Option<Value>,
    #[serde(default)]
    pub last_node_id: Option<String>,
    #[serde(default)]
    pub last_token_id: Option<String>,
    #[serde(default)]
    pub last_parent_token_id: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReducedRecordedTransition {
    pub contract_version: u32,
    pub transition_id: String,
    pub transition_digest: String,
    pub state_digest: String,
    pub checkpoint_schema_version: u32,
    /// Canonical JSON checkpoint payload, intentionally excluding `_deps`.
    pub state_json: String,
    pub state: RecordedRuntimeState,
    pub server_status: String,
    pub checkpoint: Map<String, Value>,
    pub run_status: Map<String, Value>,
    #[serde(default)]
    pub result: Option<Value>,
    #[serde(default)]
    pub errors: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RecordedTransitionResult {
    #[serde(flatten)]
    pub reduced: ReducedRecordedTransition,
    pub event_seq: i64,
    #[serde(default)]
    pub idempotent: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PersistedRecordedTransition {
    pub contract_version: u32,
    pub transition_id: String,
    pub request_digest: String,
    /// Absent for the original recorded-only runtime operation.  Present for
    /// worker handoffs so exact retries are tied to one durable lane request.
    #[serde(default)]
    pub worker_handoff: Option<RecordedWorkerHandoff>,
    /// Present for the restricted worker-effect endpoint. Older recorded
    /// transition events deserialize with this absent.
    #[serde(default)]
    pub worker_effect_digest: Option<String>,
    pub reduced: ReducedRecordedTransition,
}

impl PersistedRecordedTransition {
    pub fn result(&self, event_seq: i64, idempotent: bool) -> RecordedTransitionResult {
        RecordedTransitionResult {
            reduced: self.reduced.clone(),
            event_seq,
            idempotent,
        }
    }
}

pub fn transition_digest(
    transition: &RecordedRuntimeTransition,
) -> Result<String, RecordedRuntimeError> {
    digest_value(&serde_json::to_value(transition)?)
}

pub fn state_digest(state: &RecordedRuntimeState) -> Result<String, RecordedRuntimeError> {
    digest_value(&serde_json::to_value(state)?)
}

pub fn canonical_json_value(value: &Value) -> String {
    canonical_json(value, false)
}

fn digest_value(value: &Value) -> Result<String, RecordedRuntimeError> {
    let canonical = canonical_json_value(value);
    Ok(hex::encode(Sha256::digest(canonical.as_bytes())))
}

pub fn reduce_recorded_transition(
    current: Option<&RecordedRuntimeState>,
    transition: &RecordedRuntimeTransition,
) -> Result<ReducedRecordedTransition, RecordedRuntimeError> {
    validate_transition_basics(transition)?;
    let transition_digest = transition_digest(transition)?;

    let (
        prior_status,
        prior_step_seq,
        mut state,
        prior_frontier,
        static_routes,
        wait_reason,
        resume_payload,
    ) = match current {
        None => {
            if transition.kind != RecordedTransitionKind::Start {
                return Err(RecordedRuntimeError::MissingPriorState {
                    kind: transition.kind,
                });
            }
            let mut initial = transition.initial_state.clone().unwrap_or_default();
            let static_routes = initial
                .remove("_rt_routes")
                .map(serde_json::from_value)
                .transpose()?
                .unwrap_or_default();
            (
                None,
                -1,
                initial,
                RuntimeFrontier::default(),
                static_routes,
                None,
                None,
            )
        }
        Some(current) => {
            if transition.kind == RecordedTransitionKind::Start {
                return Err(RecordedRuntimeError::StartAlreadyExists);
            }
            validate_identity(current, transition)?;
            if current.status.is_terminal() {
                return Err(RecordedRuntimeError::TerminalImmutable {
                    status: current.status,
                });
            }
            if transition.initial_state.is_some() {
                return Err(RecordedRuntimeError::IllegalStatusTransition {
                    kind: transition.kind,
                    status: current.status,
                });
            }
            (
                Some(current.status),
                current.last_step_seq,
                current.state.clone(),
                current.frontier.clone(),
                current.static_routes.clone(),
                current.wait_reason.clone(),
                current.resume_payload.clone(),
            )
        }
    };

    let next_status = legal_next_status(prior_status, transition.kind)?;
    validate_step_sequence(prior_step_seq, transition)?;
    require_transition_identity(transition)?;
    apply_state_updates(&mut state, transition)?;

    let mut next_frontier = match (&transition.frontier, transition.kind) {
        (Some(frontier), _) => frontier.clone().normalize()?,
        (None, RecordedTransitionKind::Start) => RuntimeFrontier::default(),
        (None, kind) if is_terminal_transition(kind) => RuntimeFrontier::default(),
        (None, _) => prior_frontier.clone().normalize()?,
    };

    if is_terminal_transition(transition.kind) && next_frontier.has_unfinished_work() {
        return Err(RecordedRuntimeError::TerminalFrontierNotEmpty);
    }
    if transition.kind == RecordedTransitionKind::Suspend {
        let node_id = required_string(transition.node_id.as_deref(), transition.kind, "node_id")?;
        let token_id =
            required_string(transition.token_id.as_deref(), transition.kind, "token_id")?;
        if !prior_frontier.contains_pending(
            node_id,
            token_id,
            transition.parent_token_id.as_deref(),
        ) || !next_frontier.contains_suspended(
            node_id,
            token_id,
            transition.parent_token_id.as_deref(),
        ) {
            return Err(RecordedRuntimeError::TokenNotInFrontier);
        }
    }
    if transition.kind == RecordedTransitionKind::RecordedStepSuccess {
        let node_id = required_string(transition.node_id.as_deref(), transition.kind, "node_id")?;
        let token_id =
            required_string(transition.token_id.as_deref(), transition.kind, "token_id")?;
        if !prior_frontier.contains_pending(
            node_id,
            token_id,
            transition.parent_token_id.as_deref(),
        ) {
            return Err(RecordedRuntimeError::TokenNotInFrontier);
        }
    }
    if transition.kind == RecordedTransitionKind::ResumeResult {
        let node_id = required_string(transition.node_id.as_deref(), transition.kind, "node_id")?;
        let token_id =
            required_string(transition.token_id.as_deref(), transition.kind, "token_id")?;
        if !prior_frontier.contains_suspended(
            node_id,
            token_id,
            transition.parent_token_id.as_deref(),
        ) || next_frontier.contains_suspended(
            node_id,
            token_id,
            transition.parent_token_id.as_deref(),
        ) {
            return Err(RecordedRuntimeError::TokenNotInFrontier);
        }
    }

    let mut next_wait_reason = wait_reason;
    let mut next_resume_payload = resume_payload;
    if transition.kind == RecordedTransitionKind::Suspend {
        next_wait_reason = transition.wait_reason.clone();
        next_resume_payload = transition.resume_payload.clone();
    } else if transition.kind == RecordedTransitionKind::ResumeResult {
        next_wait_reason = None;
        next_resume_payload = transition.resume_payload.clone();
    }
    checkpoint_state(&mut state, &next_frontier, next_wait_reason.as_deref())?;

    let next_state = RecordedRuntimeState {
        contract_version: RECORDED_RUNTIME_CONTRACT_VERSION,
        run_id: transition.run_id.clone(),
        workflow_id: transition.workflow_id.clone(),
        conversation_id: transition.conversation_id.clone(),
        status: next_status,
        last_step_seq: match transition.kind {
            RecordedTransitionKind::Start => -1,
            _ => transition.step_seq,
        },
        state,
        frontier: std::mem::take(&mut next_frontier),
        static_routes,
        wait_reason: next_wait_reason,
        resume_payload: next_resume_payload,
        last_node_id: transition.node_id.clone(),
        last_token_id: transition.token_id.clone(),
        last_parent_token_id: transition.parent_token_id.clone(),
    };
    let checkpoint_step_seq = transition.step_seq;
    let checkpoint_node_id = format!("wf_ckpt|{}|{checkpoint_step_seq}", transition.run_id);
    let terminal_node_id = match next_status {
        RecordedRunStatus::Completed => format!("wf_completed|{}", transition.run_id),
        RecordedRunStatus::Failed => format!("wf_failed|{}", transition.run_id),
        RecordedRunStatus::Cancelled => format!("wf_cancelled|{}", transition.run_id),
        _ => checkpoint_node_id.clone(),
    };
    let checkpoint = map(json!({
        "entity_type": "workflow_checkpoint",
        "run_id": transition.run_id,
        "workflow_id": transition.workflow_id,
        "conversation_id": transition.conversation_id,
        "step_seq": checkpoint_step_seq,
        "node_id": checkpoint_node_id,
    }))?;
    let run_status = map(json!({
        "run_id": transition.run_id,
        "workflow_id": transition.workflow_id,
        "conversation_id": transition.conversation_id,
        "status": next_status,
        "terminal": next_status.is_terminal(),
        "terminal_node_id": terminal_node_id,
        "accepted_step_seq": checkpoint_step_seq,
    }))?;

    Ok(ReducedRecordedTransition {
        contract_version: RECORDED_RUNTIME_CONTRACT_VERSION,
        transition_id: transition.transition_id.clone(),
        transition_digest,
        state_digest: state_digest(&next_state)?,
        checkpoint_schema_version: 1,
        state_json: canonical_json_value(&Value::Object(next_state.state.clone())),
        state: next_state,
        server_status: next_status.server_status().to_owned(),
        checkpoint,
        run_status,
        result: transition.result.clone(),
        errors: transition.errors.clone(),
    })
}

fn validate_transition_basics(
    transition: &RecordedRuntimeTransition,
) -> Result<(), RecordedRuntimeError> {
    if transition.contract_version != RECORDED_RUNTIME_CONTRACT_VERSION {
        return Err(RecordedRuntimeError::UnsupportedContractVersion {
            got: transition.contract_version,
            expected: RECORDED_RUNTIME_CONTRACT_VERSION,
        });
    }
    require_nonempty(&transition.transition_id, "transition_id")?;
    require_nonempty(&transition.run_id, "run_id")?;
    require_nonempty(&transition.workflow_id, "workflow_id")?;
    require_nonempty(&transition.conversation_id, "conversation_id")?;
    if transition.expected_event_seq < 0 || transition.step_seq < 0 {
        return Err(RecordedRuntimeError::NegativeStepSequence);
    }
    if !transition.state_update.is_empty() && transition.update.is_some() {
        return Err(RecordedRuntimeError::StateUpdateConflict);
    }
    Ok(())
}

fn validate_identity(
    current: &RecordedRuntimeState,
    transition: &RecordedRuntimeTransition,
) -> Result<(), RecordedRuntimeError> {
    for (field, expected, got) in [
        ("run_id", &current.run_id, &transition.run_id),
        ("workflow_id", &current.workflow_id, &transition.workflow_id),
        (
            "conversation_id",
            &current.conversation_id,
            &transition.conversation_id,
        ),
    ] {
        if expected != got {
            return Err(RecordedRuntimeError::IdentityMismatch {
                field,
                expected: expected.clone(),
                got: got.clone(),
            });
        }
    }
    Ok(())
}

fn legal_next_status(
    current: Option<RecordedRunStatus>,
    kind: RecordedTransitionKind,
) -> Result<RecordedRunStatus, RecordedRuntimeError> {
    match (current, kind) {
        (None, RecordedTransitionKind::Start) => Ok(RecordedRunStatus::Running),
        (Some(RecordedRunStatus::Running), RecordedTransitionKind::RecordedStepSuccess) => {
            Ok(RecordedRunStatus::Running)
        }
        (Some(RecordedRunStatus::Running), RecordedTransitionKind::Suspend) => {
            Ok(RecordedRunStatus::Suspended)
        }
        (Some(RecordedRunStatus::Suspended), RecordedTransitionKind::ResumeResult) => {
            Ok(RecordedRunStatus::Running)
        }
        (
            Some(RecordedRunStatus::Running | RecordedRunStatus::Suspended),
            RecordedTransitionKind::Cancel,
        ) => Ok(RecordedRunStatus::Cancelled),
        (Some(RecordedRunStatus::Running), RecordedTransitionKind::Complete) => {
            Ok(RecordedRunStatus::Completed)
        }
        (
            Some(RecordedRunStatus::Running | RecordedRunStatus::Suspended),
            RecordedTransitionKind::Fail,
        ) => Ok(RecordedRunStatus::Failed),
        (Some(status), _kind) if status.is_terminal() => {
            Err(RecordedRuntimeError::TerminalImmutable { status })
        }
        (Some(status), kind) => Err(RecordedRuntimeError::IllegalStatusTransition { kind, status }),
        (None, kind) => Err(RecordedRuntimeError::MissingPriorState { kind }),
    }
}

fn validate_step_sequence(
    current: i64,
    transition: &RecordedRuntimeTransition,
) -> Result<(), RecordedRuntimeError> {
    if transition.kind == RecordedTransitionKind::Start {
        return Ok(());
    }
    if transition.step_seq < current {
        return Err(RecordedRuntimeError::StepSequenceRegresses {
            current,
            requested: transition.step_seq,
        });
    }
    if !is_terminal_transition(transition.kind) && transition.step_seq <= current {
        return Err(RecordedRuntimeError::StepSequenceDoesNotAdvance {
            current,
            requested: transition.step_seq,
        });
    }
    Ok(())
}

fn is_terminal_transition(kind: RecordedTransitionKind) -> bool {
    matches!(
        kind,
        RecordedTransitionKind::Cancel
            | RecordedTransitionKind::Complete
            | RecordedTransitionKind::Fail
    )
}

fn require_transition_identity(
    transition: &RecordedRuntimeTransition,
) -> Result<(), RecordedRuntimeError> {
    if matches!(
        transition.kind,
        RecordedTransitionKind::RecordedStepSuccess
            | RecordedTransitionKind::Suspend
            | RecordedTransitionKind::ResumeResult
    ) {
        let _ = required_string(transition.node_id.as_deref(), transition.kind, "node_id")?;
        let _ = required_string(transition.token_id.as_deref(), transition.kind, "token_id")?;
    }
    if transition.node_id.as_deref().is_some_and(str::is_empty) {
        return Err(RecordedRuntimeError::MissingField {
            kind: transition.kind,
            field: "node_id",
        });
    }
    if transition.token_id.as_deref().is_some_and(str::is_empty) {
        return Err(RecordedRuntimeError::MissingField {
            kind: transition.kind,
            field: "token_id",
        });
    }
    if transition
        .parent_token_id
        .as_deref()
        .is_some_and(str::is_empty)
    {
        return Err(RecordedRuntimeError::MissingField {
            kind: transition.kind,
            field: "parent_token_id",
        });
    }
    Ok(())
}

fn required_string<'a>(
    value: Option<&'a str>,
    kind: RecordedTransitionKind,
    field: &'static str,
) -> Result<&'a str, RecordedRuntimeError> {
    match value.filter(|value| !value.is_empty()) {
        Some(value) => Ok(value),
        None => Err(RecordedRuntimeError::MissingField { kind, field }),
    }
}

fn require_nonempty(value: &str, field: &'static str) -> Result<(), RecordedRuntimeError> {
    if value.is_empty() {
        Err(RecordedRuntimeError::EmptyIdentifier { field })
    } else {
        Ok(())
    }
}

fn apply_state_updates(
    state: &mut Map<String, Value>,
    transition: &RecordedRuntimeTransition,
) -> Result<(), RecordedRuntimeError> {
    for update in &transition.state_update {
        let (mode, payload) = update.parts();
        match mode {
            "u" => {
                for (key, value) in payload {
                    state.insert(key.clone(), value.clone());
                }
            }
            "a" => {
                for (key, value) in payload {
                    let list = state
                        .entry(key.clone())
                        .or_insert_with(|| Value::Array(Vec::new()));
                    let Value::Array(list) = list else {
                        return Err(RecordedRuntimeError::StateFieldNotArray {
                            field: key.clone(),
                            operation: "append",
                        });
                    };
                    list.push(value.clone());
                }
            }
            "e" => {
                for (key, value) in payload {
                    extend_state_array(state, key, value)?;
                }
            }
            other => {
                return Err(RecordedRuntimeError::UnsupportedStateUpdate {
                    mode: other.to_owned(),
                });
            }
        }
    }
    if let Some(update) = &transition.update {
        for (key, value) in update {
            if transition.state_schema.get(key).and_then(Value::as_str) == Some("a") {
                extend_state_array(state, key, value)?;
            } else {
                state.insert(key.clone(), value.clone());
            }
        }
    }
    Ok(())
}

fn extend_state_array(
    state: &mut Map<String, Value>,
    key: &str,
    value: &Value,
) -> Result<(), RecordedRuntimeError> {
    let list = state
        .entry(key.to_owned())
        .or_insert_with(|| Value::Array(Vec::new()));
    let Value::Array(list) = list else {
        return Err(RecordedRuntimeError::StateFieldNotArray {
            field: key.to_owned(),
            operation: "extend",
        });
    };
    match value {
        Value::Array(values) => list.extend(values.iter().cloned()),
        Value::String(value) => list.extend(
            value
                .chars()
                .map(|character| Value::String(character.to_string())),
        ),
        Value::Object(values) => list.extend(values.keys().cloned().map(Value::String)),
        _ => {
            return Err(RecordedRuntimeError::StateValueNotExtendable {
                field: key.to_owned(),
            });
        }
    }
    Ok(())
}

fn checkpoint_state(
    state: &mut Map<String, Value>,
    frontier: &RuntimeFrontier,
    wait_reason: Option<&str>,
) -> Result<(), RecordedRuntimeError> {
    // These are process-local dependency injection keys in Python.  They must
    // never cross a durable checkpoint boundary.
    state.remove("_deps");
    state.remove("dream_deps");
    state.insert("_rt_join".to_owned(), serde_json::to_value(frontier)?);
    if let Some(wait_reason) = wait_reason {
        state.insert(
            "wait_reason".to_owned(),
            Value::String(wait_reason.to_owned()),
        );
    }
    Ok(())
}

fn map(value: Value) -> Result<Map<String, Value>, RecordedRuntimeError> {
    match value {
        Value::Object(value) => Ok(value),
        _ => unreachable!("json macro emitted an object"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn transition(
        kind: RecordedTransitionKind,
        id: &str,
        expected_event_seq: i64,
        step_seq: i64,
    ) -> RecordedRuntimeTransition {
        RecordedRuntimeTransition {
            contract_version: RECORDED_RUNTIME_CONTRACT_VERSION,
            transition_id: id.to_owned(),
            expected_event_seq,
            kind,
            run_id: "run".to_owned(),
            workflow_id: "workflow".to_owned(),
            conversation_id: "conversation".to_owned(),
            user_id: Some("user".to_owned()),
            user_turn_node_id: Some("turn".to_owned()),
            step_seq,
            node_id: Some("node".to_owned()),
            token_id: Some("token".to_owned()),
            parent_token_id: None,
            initial_state: None,
            state_update: Vec::new(),
            update: None,
            state_schema: Map::new(),
            frontier: None,
            result: None,
            wait_reason: None,
            resume_payload: None,
            errors: Vec::new(),
        }
    }

    fn route_edge(
        edge_id: &str,
        target: &str,
        predicate: Option<&str>,
        result: Option<bool>,
        priority: i64,
    ) -> RuntimeRouteEdge {
        RuntimeRouteEdge {
            edge_id: edge_id.to_owned(),
            target_ids: vec![target.to_owned()],
            aliases: vec![
                target.to_owned(),
                target.split('|').next_back().unwrap().to_owned(),
            ],
            predicate: predicate.map(ToOwned::to_owned),
            multiplicity: "one".to_owned(),
            is_default: false,
            priority,
            predicate_result: result,
            base_result: result,
        }
    }

    #[test]
    fn route_selection_preserves_explicit_predicate_fanout_failure_and_default_rules() {
        let edges = vec![
            route_edge("low", "wf|low", Some("p"), Some(true), 1),
            route_edge("high", "wf|high", Some("p"), Some(true), 9),
        ];
        let explicit = select_runtime_route(&RuntimeRouteRequest {
            edges: edges.clone(),
            explicit_next: vec!["low".to_owned()],
            fanout: false,
            failure_only: false,
        });
        assert_eq!(explicit.next_node_ids, vec!["wf|low"]);
        assert_eq!(explicit.selected_edge_indices, vec![0]);

        let priority = select_runtime_route(&RuntimeRouteRequest {
            edges: edges.clone(),
            explicit_next: vec![],
            fanout: false,
            failure_only: false,
        });
        assert_eq!(priority.next_node_ids, vec!["wf|high"]);
        assert_eq!(priority.selected_edge_indices, vec![1]);

        let fanout = select_runtime_route(&RuntimeRouteRequest {
            edges,
            explicit_next: vec![],
            fanout: true,
            failure_only: false,
        });
        assert_eq!(fanout.next_node_ids, vec!["wf|high", "wf|low"]);
        assert_eq!(fanout.selected_edge_indices, vec![1, 0]);

        let failure = select_runtime_route(&RuntimeRouteRequest {
            edges: vec![route_edge("base", "base", None, Some(true), 1)],
            explicit_next: vec![],
            fanout: false,
            failure_only: true,
        });
        assert!(failure.next_node_ids.is_empty());

        let mut default = route_edge("default", "a", None, Some(false), 1);
        default.target_ids.push("b".to_owned());
        default.is_default = true;
        let default = select_runtime_route(&RuntimeRouteRequest {
            edges: vec![default],
            explicit_next: vec![],
            fanout: true,
            failure_only: false,
        });
        assert_eq!(default.next_node_ids, vec!["a", "b"]);
    }

    #[test]
    fn authoritative_successors_delegate_only_predicate_choice_with_graph_validation() {
        let route = |target: &str, predicate: Option<&str>, multiplicity: &str, fanout: bool| {
            RuntimeStaticRoute {
                source_node_id: "gate".to_owned(),
                target_node_id: target.to_owned(),
                join_mask: if target == "join" { 1 } else { 0 },
                predicate: predicate.map(str::to_owned),
                multiplicity: multiplicity.to_owned(),
                source_fanout: fanout,
            }
        };
        let static_routes = vec![
            route("left", None, "many", true),
            route("right", None, "many", true),
        ];
        assert_eq!(
            authoritative_runtime_successors(
                &static_routes,
                "gate",
                &[RuntimeSuccessor {
                    node_id: "forged".to_owned(),
                    join_mask: 0,
                }],
            )
            .unwrap()
            .into_iter()
            .map(|successor| successor.node_id)
            .collect::<Vec<_>>(),
            vec!["left", "right"]
        );

        let predicate_routes = vec![
            route("left", Some("if_true"), "one", false),
            route("join", Some("if_false"), "one", false),
        ];
        let selected = vec![RuntimeSuccessor {
            node_id: "join".to_owned(),
            join_mask: 1,
        }];
        assert_eq!(
            authoritative_runtime_successors(&predicate_routes, "gate", &selected).unwrap()[0]
                .node_id,
            "join"
        );
        assert!(
            authoritative_runtime_successors(
                &predicate_routes,
                "gate",
                &[RuntimeSuccessor {
                    node_id: "join".to_owned(),
                    join_mask: 0,
                }],
            )
            .is_err()
        );
        assert!(
            authoritative_runtime_successors(
                &predicate_routes,
                "gate",
                &[
                    RuntimeSuccessor {
                        node_id: "left".to_owned(),
                        join_mask: 0,
                    },
                    selected[0].clone(),
                ],
            )
            .is_err()
        );
    }

    #[test]
    fn successor_plan_preserves_continuation_spawns_children_and_updates_join_counts() {
        let plan = plan_runtime_successors(&RuntimeSuccessorPlanRequest {
            token_id: "parent".to_owned(),
            parent_token_id: Some("root".to_owned()),
            step_seq: 7,
            current_join_mask: 0b0011,
            join_outstanding: vec![1, 1, 0],
            successors: vec![
                RuntimeSuccessor {
                    node_id: "left".to_owned(),
                    join_mask: 0b0101,
                },
                RuntimeSuccessor {
                    node_id: "right".to_owned(),
                    join_mask: 0b0110,
                },
            ],
        });
        assert_eq!(plan.join_outstanding, vec![1, 1, 2]);
        assert_eq!(plan.tokens[0].token_id, "parent");
        assert_eq!(plan.tokens[0].parent_token_id.as_deref(), Some("root"));
        assert!(!plan.tokens[0].spawned);
        assert_eq!(plan.tokens[1].parent_token_id.as_deref(), Some("parent"));
        assert!(plan.tokens[1].spawned);
        assert_eq!(
            plan.tokens[1].token_id,
            kogwistar_contracts::stable_id("token_id", &["parent/7:1:right".to_owned()])
                .simple()
                .to_string()
        );
    }

    #[test]
    fn join_arrival_waits_then_merges_masks_and_obligations() {
        let first = apply_runtime_join_arrival(&RuntimeJoinArrivalRequest {
            join_index: 0,
            join_outstanding: vec![2, 2],
            waiters: vec![],
            arrival: RuntimeJoinWaiter {
                join_mask: 0b11,
                token_id: "one".to_owned(),
                parent_token_id: None,
            },
            merge: true,
        });
        assert_eq!(first.outstanding, 1);
        assert!(first.released.is_none());
        assert_eq!(first.waiters[0].join_mask, 0b10);

        let second = apply_runtime_join_arrival(&RuntimeJoinArrivalRequest {
            join_index: 0,
            join_outstanding: first.join_outstanding,
            waiters: first.waiters,
            arrival: RuntimeJoinWaiter {
                join_mask: 0b11,
                token_id: "two".to_owned(),
                parent_token_id: Some("one".to_owned()),
            },
            merge: true,
        });
        assert_eq!(second.outstanding, 0);
        assert_eq!(second.collapsed_work_items, 1);
        assert!(second.waiters.is_empty());
        assert_eq!(second.released.unwrap().join_mask, 0b10);
        assert_eq!(second.join_outstanding, vec![0, 1]);
    }

    #[test]
    fn retry_decision_normalizes_budget_and_marks_exhaustion() {
        assert_eq!(
            decide_runtime_retry(&RuntimeRetryDecisionRequest {
                retry_budget: 0,
                attempt_number: 1,
            }),
            RuntimeRetryDecision {
                retry_budget: 1,
                attempt_number: 1,
                should_retry: false,
                exhausted: true,
                next_attempt_number: None,
            }
        );
        assert_eq!(
            decide_runtime_retry(&RuntimeRetryDecisionRequest {
                retry_budget: 3,
                attempt_number: 2,
            })
            .next_attempt_number,
            Some(3)
        );
    }

    #[test]
    fn nested_invocation_plan_is_deterministic_and_honors_overrides() {
        let request = RuntimeNestedInvocationPlanRequest {
            parent_run_id: "parent".to_owned(),
            workflow_id: "child".to_owned(),
            result_state_key: None,
            run_id: None,
            parent_conversation_id: "conversation".to_owned(),
            conversation_id: Some("child-conversation".to_owned()),
            parent_turn_node_id: "turn".to_owned(),
            turn_node_id: None,
        };
        let first = plan_runtime_nested_invocation(&request);
        assert_eq!(first, plan_runtime_nested_invocation(&request));
        assert_eq!(first.conversation_id, "child-conversation");
        assert_eq!(first.turn_node_id, "turn");
        assert_eq!(first.result_state_key, "workflow_result::child");
    }

    #[test]
    fn dispatch_decision_bounds_launches_and_stops_on_cancel() {
        assert_eq!(
            decide_runtime_dispatch(&RuntimeDispatchDecisionRequest {
                max_workers: 4,
                inflight: 3,
                pending: 7,
                cancelling: false,
            })
            .launch_capacity,
            1
        );
        let cancelling = decide_runtime_dispatch(&RuntimeDispatchDecisionRequest {
            max_workers: 0,
            inflight: 1,
            pending: 0,
            cancelling: true,
        });
        assert_eq!(cancelling.worker_limit, 1);
        assert_eq!(cancelling.launch_capacity, 0);
        assert!(cancelling.should_drain);
        assert!(!cancelling.cancellation_complete);
    }

    #[test]
    fn budget_suspend_decision_includes_cost_limit() {
        let decision = decide_runtime_budget_suspend(&RuntimeBudgetSuspendRequest {
            token_budget: 0,
            token_used: 0,
            time_budget_ms: 0,
            time_used_ms: 0,
            rate_limit: 0,
            rate_used: 0,
            step_budget: 0,
            step_used: 0,
            call_budget: 0,
            call_used: 0,
            cost_budget: 2.5,
            cost_used: 2.5,
        });
        assert!(decision.should_suspend);
        assert_eq!(decision.reason.as_deref(), Some("cost"));
    }

    #[test]
    fn scheduler_tick_owns_fifo_dispatch_and_cancel_stop() {
        let token = |node: &str| RuntimeSchedulerToken {
            node_id: node.to_owned(),
            join_mask: 0,
            token_id: format!("token-{node}"),
            parent_token_id: None,
        };
        let tick = tick_runtime_scheduler(&RuntimeSchedulerTickRequest {
            pending: vec![token("a"), token("b"), token("c")],
            inflight: 1,
            max_workers: 3,
            cancelling: false,
        });
        assert_eq!(tick.dispatch, vec![token("a"), token("b")]);
        assert_eq!(tick.pending, vec![token("c")]);

        let cancelled = tick_runtime_scheduler(&RuntimeSchedulerTickRequest {
            pending: vec![token("a")],
            inflight: 1,
            max_workers: 3,
            cancelling: true,
        });
        assert!(cancelled.dispatch.is_empty());
        assert_eq!(cancelled.pending, vec![token("a")]);
        assert!(cancelled.should_drain);
    }

    #[test]
    fn reducer_canonicalizes_frontier_and_drops_dependencies() {
        let mut start = transition(RecordedTransitionKind::Start, "start", 0, 0);
        start.initial_state = Some(Map::from_iter([
            ("_deps".to_owned(), json!({"not": "durable"})),
            ("value".to_owned(), json!(1)),
        ]));
        start.frontier = Some(RuntimeFrontier {
            pending: vec![
                ("z".to_owned(), 0, "z-token".to_owned(), None),
                ("a".to_owned(), 0, "a-token".to_owned(), None),
            ],
            ..RuntimeFrontier::default()
        });
        let started = reduce_recorded_transition(None, &start).unwrap();
        assert_eq!(started.state.last_step_seq, -1);
        assert_eq!(started.state.frontier.pending[0].0, "a");
        assert!(!started.state.state.contains_key("_deps"));
        assert_eq!(started.checkpoint["node_id"], "wf_ckpt|run|0");
        assert_eq!(started.run_status["status"], "running");
    }

    #[test]
    fn terminal_transition_may_share_last_accepted_step_only() {
        let mut start = transition(RecordedTransitionKind::Start, "start", 0, 0);
        start.frontier = Some(RuntimeFrontier {
            pending: vec![("node".to_owned(), 0, "token".to_owned(), None)],
            ..RuntimeFrontier::default()
        });
        let started = reduce_recorded_transition(None, &start).unwrap();
        let step = transition(RecordedTransitionKind::RecordedStepSuccess, "step", 1, 0);
        let stepped = reduce_recorded_transition(Some(&started.state), &step).unwrap();
        let complete = transition(RecordedTransitionKind::Complete, "complete", 2, 0);
        let completed = reduce_recorded_transition(Some(&stepped.state), &complete).unwrap();
        assert_eq!(completed.state.status, RecordedRunStatus::Completed);
        assert_eq!(completed.server_status, "succeeded");
        assert_eq!(completed.run_status["status"], "completed");
        assert!(reduce_recorded_transition(Some(&completed.state), &complete).is_err());
    }

    #[test]
    fn terminal_transition_rejects_unfinished_join_work() {
        let mut start = transition(RecordedTransitionKind::Start, "start", 0, 0);
        start.frontier = Some(RuntimeFrontier {
            pending: vec![("node".to_owned(), 1, "token".to_owned(), None)],
            join_node_ids: vec!["join".to_owned()],
            join_outstanding: vec![1],
            join_waiters: BTreeMap::from_iter([(
                "join".to_owned(),
                vec![(1, "waiter".to_owned(), None)],
            )]),
            ..RuntimeFrontier::default()
        });
        let started = reduce_recorded_transition(None, &start).unwrap();
        let mut complete = transition(RecordedTransitionKind::Complete, "complete", 1, 0);
        complete.frontier = Some(RuntimeFrontier {
            join_node_ids: vec!["join".to_owned()],
            join_outstanding: vec![1],
            join_waiters: BTreeMap::from_iter([(
                "join".to_owned(),
                vec![(1, "waiter".to_owned(), None)],
            )]),
            ..RuntimeFrontier::default()
        });

        assert!(matches!(
            reduce_recorded_transition(Some(&started.state), &complete),
            Err(RecordedRuntimeError::TerminalFrontierNotEmpty)
        ));
    }

    #[test]
    fn worker_success_effect_preserves_authoritative_frontier_and_plans_children() {
        let frontier = RuntimeFrontier {
            pending: vec![
                ("start".to_owned(), 1, "token".to_owned(), None),
                ("other".to_owned(), 0, "other-token".to_owned(), None),
            ],
            suspended: vec![("sleep".to_owned(), 0, "sleep-token".to_owned(), None)],
            join_node_ids: vec!["join".to_owned()],
            join_outstanding: vec![1],
            join_waiters: BTreeMap::from_iter([("join".to_owned(), Vec::new())]),
        };
        let next = frontier_after_worker_success(
            &frontier,
            "start",
            "token",
            None,
            4,
            &RuntimeWorkerSuccessEffect {
                successors: vec![
                    RuntimeSuccessor {
                        node_id: "left".to_owned(),
                        join_mask: 1,
                    },
                    RuntimeSuccessor {
                        node_id: "right".to_owned(),
                        join_mask: 1,
                    },
                ],
            },
        )
        .unwrap();

        assert!(next.pending.iter().any(|token| token.0 == "other"));
        assert!(
            next.pending
                .iter()
                .any(|token| token.0 == "left" && token.2 == "token")
        );
        let right = next
            .pending
            .iter()
            .find(|token| token.0 == "right")
            .unwrap();
        assert_eq!(right.3.as_deref(), Some("token"));
        assert_ne!(right.2, "token");
        assert_eq!(next.suspended, frontier.suspended);
        assert_eq!(next.join_outstanding, vec![2]);
        assert!(matches!(
            frontier_after_worker_success(
                &frontier,
                "forged",
                "token",
                None,
                4,
                &RuntimeWorkerSuccessEffect { successors: vec![] },
            ),
            Err(RecordedRuntimeError::TokenNotInFrontier)
        ));
    }

    #[test]
    fn worker_success_effect_holds_join_arrivals_until_barrier_releases_once() {
        let root = RuntimeFrontier {
            pending: vec![("start".to_owned(), 1, "root".to_owned(), None)],
            join_node_ids: vec!["join".to_owned()],
            join_outstanding: vec![1],
            join_waiters: BTreeMap::from_iter([("join".to_owned(), Vec::new())]),
            ..RuntimeFrontier::default()
        };
        let branches = frontier_after_worker_success(
            &root,
            "start",
            "root",
            None,
            0,
            &RuntimeWorkerSuccessEffect {
                successors: vec![
                    RuntimeSuccessor {
                        node_id: "left".to_owned(),
                        join_mask: 1,
                    },
                    RuntimeSuccessor {
                        node_id: "right".to_owned(),
                        join_mask: 1,
                    },
                ],
            },
        )
        .unwrap();
        assert_eq!(branches.join_outstanding, vec![2]);
        let left = branches
            .pending
            .iter()
            .find(|token| token.0 == "left")
            .unwrap()
            .clone();
        let after_left = frontier_after_worker_success(
            &branches,
            &left.0,
            &left.2,
            left.3.as_deref(),
            1,
            &RuntimeWorkerSuccessEffect {
                successors: vec![RuntimeSuccessor {
                    node_id: "join".to_owned(),
                    join_mask: 1,
                }],
            },
        )
        .unwrap();
        assert_eq!(after_left.join_outstanding, vec![1]);
        assert!(!after_left.pending.iter().any(|token| token.0 == "join"));
        assert_eq!(after_left.join_waiters["join"].len(), 1);

        let right = after_left
            .pending
            .iter()
            .find(|token| token.0 == "right")
            .unwrap()
            .clone();
        let released = frontier_after_worker_success(
            &after_left,
            &right.0,
            &right.2,
            right.3.as_deref(),
            2,
            &RuntimeWorkerSuccessEffect {
                successors: vec![RuntimeSuccessor {
                    node_id: "join".to_owned(),
                    join_mask: 1,
                }],
            },
        )
        .unwrap();
        assert_eq!(released.join_outstanding, vec![0]);
        assert!(released.join_waiters["join"].is_empty());
        assert_eq!(
            released
                .pending
                .iter()
                .filter(|token| token.0 == "join")
                .count(),
            1
        );
    }

    #[test]
    fn suspend_resume_moves_only_the_owned_token_and_clears_wait_reason() {
        let mut start = transition(RecordedTransitionKind::Start, "start", 0, 0);
        start.frontier = Some(RuntimeFrontier {
            pending: vec![("node".to_owned(), 0, "token".to_owned(), None)],
            ..RuntimeFrontier::default()
        });
        let started = reduce_recorded_transition(None, &start).unwrap();
        let suspended_frontier =
            frontier_after_worker_suspend(&started.state.frontier, "node", "token", None).unwrap();
        let mut suspend = transition(RecordedTransitionKind::Suspend, "suspend", 1, 0);
        suspend.frontier = Some(suspended_frontier.clone());
        suspend.wait_reason = Some("approval".to_owned());
        suspend.resume_payload = Some(json!({"request": 1}));
        let suspended = reduce_recorded_transition(Some(&started.state), &suspend).unwrap();
        assert_eq!(suspended.state.status, RecordedRunStatus::Suspended);
        assert_eq!(suspended.state.wait_reason.as_deref(), Some("approval"));

        let mut resume = transition(RecordedTransitionKind::ResumeResult, "resume", 2, 1);
        resume.frontier = Some(
            frontier_after_worker_resume(&suspended.state.frontier, "node", "token", None).unwrap(),
        );
        resume.resume_payload = Some(json!({"approved": true}));
        let resumed = reduce_recorded_transition(Some(&suspended.state), &resume).unwrap();
        assert_eq!(resumed.state.status, RecordedRunStatus::Running);
        assert_eq!(resumed.state.wait_reason, None);
        assert_eq!(
            resumed.state.resume_payload,
            Some(json!({"approved": true}))
        );
        assert_eq!(
            resumed.state.frontier.pending,
            started.state.frontier.pending
        );
        assert!(resumed.state.frontier.suspended.is_empty());
    }
}
