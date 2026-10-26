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

fn default_legacy_source_fanout() -> bool {
    // Old snapshots represented static fanout as multiple routes. Rich
    // snapshots always serialize this field, so only legacy payloads use it.
    true
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
    matched.sort_by_key(|right| std::cmp::Reverse(right.1.priority));

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
    #[serde(default)]
    pub edge_id: String,
    pub source_node_id: String,
    pub target_node_id: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    #[serde(default)]
    pub join_mask: i64,
    #[serde(default)]
    pub predicate: Option<String>,
    #[serde(default = "default_route_multiplicity")]
    pub multiplicity: String,
    #[serde(default)]
    pub is_default: bool,
    #[serde(default = "default_route_priority")]
    pub priority: i64,
    #[serde(default = "default_legacy_source_fanout")]
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
    authoritative_runtime_successors_for_result(
        static_routes,
        source_node_id,
        worker_selected,
        &[],
        false,
    )
}

pub fn authoritative_runtime_successors_for_result(
    static_routes: &[RuntimeStaticRoute],
    source_node_id: &str,
    worker_selected: &[RuntimeSuccessor],
    explicit_next: &[String],
    failure_only: bool,
) -> Result<Vec<RuntimeSuccessor>, RecordedRuntimeError> {
    if static_routes.is_empty() {
        return Ok(worker_selected.to_vec());
    }
    let mut outgoing = static_routes
        .iter()
        .filter(|route| route.source_node_id == source_node_id)
        .collect::<Vec<_>>();
    outgoing.sort_by_key(|route| route.priority);
    if outgoing.is_empty() {
        if worker_selected.is_empty() && explicit_next.is_empty() {
            return Ok(Vec::new());
        }
        return Err(RecordedRuntimeError::InvalidSuccessorSelection(format!(
            "source {source_node_id:?} has no outgoing frozen routes"
        )));
    }

    let delegates_predicates =
        explicit_next.is_empty() && outgoing.iter().any(|route| route.predicate.is_some());
    if delegates_predicates {
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
                .any(|selected: &&RuntimeStaticRoute| {
                    selected.target_node_id == route.target_node_id
                })
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
    }

    let edges = outgoing
        .iter()
        .map(|route| RuntimeRouteEdge {
            edge_id: if route.edge_id.is_empty() {
                format!("{}->{}", route.source_node_id, route.target_node_id)
            } else {
                route.edge_id.clone()
            },
            target_ids: vec![route.target_node_id.clone()],
            aliases: if route.aliases.is_empty() {
                vec![
                    route.target_node_id.clone(),
                    route
                        .target_node_id
                        .split('|')
                        .next_back()
                        .unwrap_or(&route.target_node_id)
                        .to_owned(),
                ]
            } else {
                route.aliases.clone()
            },
            predicate: route.predicate.clone(),
            multiplicity: route.multiplicity.clone(),
            is_default: route.is_default,
            priority: route.priority,
            predicate_result: route.predicate.as_ref().map(|_| {
                worker_selected.iter().any(|successor| {
                    successor.node_id == route.target_node_id
                        && successor.join_mask == route.join_mask
                })
            }),
            base_result: route.predicate.is_none().then_some(true),
        })
        .collect();
    let decision = select_runtime_route(&RuntimeRouteRequest {
        edges,
        explicit_next: explicit_next.to_vec(),
        fanout: outgoing.iter().any(|route| route.source_fanout),
        failure_only,
    });
    Ok(decision
        .selected_edge_indices
        .into_iter()
        .map(|index| RuntimeSuccessor {
            node_id: outgoing[index].target_node_id.clone(),
            join_mask: outgoing[index].join_mask,
        })
        .collect())
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
