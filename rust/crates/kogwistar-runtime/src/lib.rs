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
    #[error("recorded runtime terminal transition must not retain unfinished token or join work")]
    TerminalFrontierNotEmpty,
    #[error("cannot encode recorded runtime JSON: {0}")]
    Json(#[from] serde_json::Error),
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

    let (prior_status, prior_step_seq, mut state, prior_frontier, wait_reason, resume_payload) =
        match current {
            None => {
                if transition.kind != RecordedTransitionKind::Start {
                    return Err(RecordedRuntimeError::MissingPriorState {
                        kind: transition.kind,
                    });
                }
                let initial = transition.initial_state.clone().unwrap_or_default();
                (None, -1, initial, RuntimeFrontier::default(), None, None)
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
}
