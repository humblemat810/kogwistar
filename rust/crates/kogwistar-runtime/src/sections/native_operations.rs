pub const RECORDED_RUNTIME_CONTRACT_VERSION: u32 = 1;
pub const WORKER_PROTOCOL_SYNC_V1: &str = "sync-v1";
pub const WORKER_PROTOCOL_ASYNC_V2: &str = "async-v2";
pub const RUST_NATIVE_EXECUTOR_TARGET: &str = "rust-native";
pub const PYTHON_EXECUTOR_TARGET: &str = "python-worker";
pub const RUST_NATIVE_OPERATION_PREFIX: &str = "rust-native/";
pub const RUST_NATIVE_OPERATION_SCHEMA_VERSION: u32 = 1;
// Compatibility export retained for callers that used the runtime constant.
pub const RUST_NATIVE_SEAM_ATTESTATION_VERSION: u32 = NATIVE_SEAM_ATTESTATION_VERSION;

pub fn native_operation_name(operation: &str) -> Option<&str> {
    operation
        .strip_prefix(RUST_NATIVE_OPERATION_PREFIX)
        .filter(|name| !name.trim().is_empty())
}

/// Resolve a registered native operation under explicit or legacy routing.
/// Missing targets preserve the historical `rust-native/<operation>` form.
pub fn native_operation_for_target<'a>(
    operation: &'a str,
    executor_target: Option<&str>,
) -> Option<&'a str> {
    match executor_target {
        Some(RUST_NATIVE_EXECUTOR_TARGET) => {
            let name = if operation.starts_with(RUST_NATIVE_OPERATION_PREFIX) {
                native_operation_name(operation)?
            } else {
                operation
            };
            (!name.trim().is_empty()).then_some(name)
        }
        Some(PYTHON_EXECUTOR_TARGET) => None,
        Some(_) => None,
        None => native_operation_name(operation),
    }
}

fn default_operation_version() -> String {
    "v1".to_owned()
}

fn is_default_operation_version(version: &str) -> bool {
    version == "v1"
}

fn is_no_environment_id(value: &Option<String>) -> bool {
    value.is_none()
}

/// Scheduler-owned payload for one Python worker step.  Keeping this wire DTO
/// beside the reducer prevents SQLite, PostgreSQL, and API schedulers from
/// independently hand-copying its fields.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeStepExecutePayload {
    pub contract_version: u32,
    pub kind: String,
    /// Versioned Python callback execution protocol. This is lane metadata,
    /// not reducer state: both protocols submit the same restricted durable
    /// worker-effect DTO. Keeping it in every claimed payload prevents a
    /// sync-v1 worker from accidentally accepting an async-v2 continuation.
    #[serde(default = "default_worker_protocol")]
    pub worker_protocol: String,
    pub run_id: String,
    pub workflow_id: String,
    pub conversation_id: String,
    #[serde(default)]
    pub turn_node_id: Option<String>,
    pub node_id: String,
    pub op: Value,
    /// Explicit executor identity for opt-in native operations. Legacy
    /// Python-worker payloads omit this field and retain their old contract.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub executor_target: Option<String>,
    /// Version of this registered workflow operation, independent of the
    /// worker transport protocol. Missing legacy values mean v1.
    #[serde(
        default = "default_operation_version",
        skip_serializing_if = "is_default_operation_version"
    )]
    pub op_version: String,
    /// Optional worker environment required by this operation. Legacy work
    /// without this field remains environment-agnostic.
    #[serde(default, skip_serializing_if = "is_no_environment_id")]
    pub op_environment_id: Option<String>,
    #[serde(default)]
    pub runtime_routes: Vec<RuntimeStaticRoute>,
    pub join_mask: i64,
    pub token_id: String,
    #[serde(default)]
    pub parent_token_id: Option<String>,
    pub step_seq: i64,
    pub expected_event_seq: i64,
    #[serde(default)]
    pub state: Map<String, Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resume_effect: Option<Value>,
}

#[derive(Clone, Debug)]
pub struct RuntimeStepExecuteRequest {
    pub node_id: String,
    pub op: Value,
    pub join_mask: i64,
    pub token_id: String,
    pub parent_token_id: Option<String>,
    pub step_seq: i64,
    pub expected_event_seq: i64,
    pub resume_effect: Option<Value>,
}

/// The deliberately small first native execution slice.  Native operations
/// are explicit and never inferred from an arbitrary Python operation name.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeNativeStepRequest {
    pub executor_target: String,
    pub operation: String,
    /// Version of the registered native operation. Legacy native payloads
    /// omit this field and retain the v1 operation contract.
    #[serde(default = "default_operation_version")]
    pub op_version: String,
    pub run_id: String,
    pub workflow_id: String,
    pub node_id: String,
    pub conversation_id: String,
    #[serde(default)]
    pub turn_node_id: Option<String>,
    pub token_id: String,
    #[serde(default)]
    pub parent_token_id: Option<String>,
    pub step_seq: i64,
    #[serde(default)]
    pub state: Map<String, Value>,
    #[serde(default)]
    pub runtime_routes: Vec<RuntimeStaticRoute>,
}

#[derive(Debug, Error)]
pub enum RuntimeNativeExecutionError {
    #[error("native executor target must be {expected:?}, got {got:?}")]
    WrongExecutorTarget { expected: String, got: String },
    #[error("native operation must be non-empty")]
    EmptyOperation,
    #[error("native operation {0:?} is not registered")]
    UnsupportedOperation(String),
    #[error("native operation version for {operation:?} is unsupported: {version:?}")]
    UnsupportedOperationVersion { operation: String, version: String },
    #[error("native step identifier {field} must be non-empty")]
    EmptyIdentifier { field: &'static str },
    #[error("native step sequence must not be negative")]
    NegativeStepSequence,
    #[error("native operation state field {field:?} is invalid: {reason}")]
    InvalidState {
        field: &'static str,
        reason: &'static str,
    },
    #[error("native operation {operation:?} failed: {reason}")]
    OperationFailed { operation: String, reason: String },
    #[error("native routing failed: {0}")]
    Routing(#[from] RecordedRuntimeError),
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum NativeOperationRegistrationError {
    #[error("native operation name and version must be non-empty and whitespace-free")]
    InvalidIdentity,
    #[error("native operation {operation:?} version {version:?} is already registered")]
    Duplicate { operation: String, version: String },
}

/// Typed output of one application-owned Rust operation. The runtime still
/// owns routing, effect identity, trace attestation, and durable reduction.
#[derive(Clone, Debug)]
pub struct NativeOperationOutput {
    pub state_update: Vec<RecordedStateUpdate>,
    pub result: Option<Value>,
}

type NativeOperationHandler = dyn Fn(&RuntimeNativeStepRequest) -> Result<NativeOperationOutput, RuntimeNativeExecutionError>
    + Send
    + Sync;

/// Versioned registry for trusted Rust operation implementations. Registration
/// is explicit application composition; it is not a plugin loader or a way for
/// workflow input to install executable code.
#[derive(Clone)]
pub struct NativeOperationRegistry {
    handlers: BTreeMap<String, BTreeMap<String, Arc<NativeOperationHandler>>>,
}

impl Default for NativeOperationRegistry {
    fn default() -> Self {
        let mut registry = Self {
            handlers: BTreeMap::new(),
        };
        registry.insert_builtin("noop", native_noop);
        registry.insert_builtin("increment-counter", native_increment_counter);
        registry.insert_builtin("emit-result", native_emit_result);
        registry
    }
}

impl NativeOperationRegistry {
    fn insert_builtin(
        &mut self,
        operation: &str,
        handler: impl Fn(
            &RuntimeNativeStepRequest,
        ) -> Result<NativeOperationOutput, RuntimeNativeExecutionError>
        + Send
        + Sync
        + 'static,
    ) {
        self.handlers
            .entry(operation.to_owned())
            .or_default()
            .insert("v1".to_owned(), Arc::new(handler));
    }

    /// Add a trusted, application-owned synchronous handler. Handlers must be
    /// bounded, deterministic, and free of external side effects: delivery may
    /// retry before the runtime durably reduces the returned effect.
    pub fn register<F>(
        &mut self,
        operation: impl Into<String>,
        version: impl Into<String>,
        handler: F,
    ) -> Result<(), NativeOperationRegistrationError>
    where
        F: Fn(
                &RuntimeNativeStepRequest,
            ) -> Result<NativeOperationOutput, RuntimeNativeExecutionError>
            + Send
            + Sync
            + 'static,
    {
        let operation = operation.into();
        let version = version.into();
        if operation.is_empty()
            || operation.trim() != operation
            || operation.starts_with(RUST_NATIVE_OPERATION_PREFIX)
            || operation.chars().any(char::is_whitespace)
            || version.is_empty()
            || version.trim() != version
            || version.chars().any(char::is_whitespace)
        {
            return Err(NativeOperationRegistrationError::InvalidIdentity);
        }
        let versions = self.handlers.entry(operation.clone()).or_default();
        if versions.contains_key(&version) {
            return Err(NativeOperationRegistrationError::Duplicate { operation, version });
        }
        versions.insert(version, Arc::new(handler));
        Ok(())
    }

    fn handler(&self, operation: &str, version: &str) -> Option<&Arc<NativeOperationHandler>> {
        self.handlers.get(operation)?.get(version)
    }

    fn contains_operation(&self, operation: &str) -> bool {
        self.handlers.contains_key(operation)
    }

    pub fn supports(&self, operation: &str, version: &str) -> bool {
        self.handler(operation, version).is_some()
    }
}

fn native_noop(
    request: &RuntimeNativeStepRequest,
) -> Result<NativeOperationOutput, RuntimeNativeExecutionError> {
    Ok(NativeOperationOutput {
        state_update: Vec::new(),
        result: Some(json!({
            "workflow_status": "succeeded",
            "final_state": request.state,
        })),
    })
}

fn native_increment_counter(
    request: &RuntimeNativeStepRequest,
) -> Result<NativeOperationOutput, RuntimeNativeExecutionError> {
    let current = request.state.get("counter").and_then(Value::as_i64).ok_or(
        RuntimeNativeExecutionError::InvalidState {
            field: "counter",
            reason: "expected a signed integer",
        },
    )?;
    let next = current
        .checked_add(1)
        .ok_or(RuntimeNativeExecutionError::InvalidState {
            field: "counter",
            reason: "integer overflow",
        })?;
    let mut values = Map::new();
    values.insert("counter".to_owned(), json!(next));
    let mut final_state = request.state.clone();
    final_state.insert("counter".to_owned(), json!(next));
    Ok(NativeOperationOutput {
        state_update: vec![RecordedStateUpdate::Pair(("u".to_owned(), values))],
        result: Some(json!({
            "workflow_status": "succeeded",
            "final_state": final_state,
        })),
    })
}

fn native_emit_result(
    request: &RuntimeNativeStepRequest,
) -> Result<NativeOperationOutput, RuntimeNativeExecutionError> {
    Ok(NativeOperationOutput {
        state_update: Vec::new(),
        result: Some(json!({
            "executor": RUST_NATIVE_EXECUTOR_TARGET,
            "operation": request.operation,
            "schema_version": RUST_NATIVE_OPERATION_SCHEMA_VERSION,
        })),
    })
}

/// Execute with the built-in native operation registry.
pub fn execute_native_step(
    request: &RuntimeNativeStepRequest,
) -> Result<RecordedWorkerSuccessEffect, RuntimeNativeExecutionError> {
    static BUILTINS: OnceLock<NativeOperationRegistry> = OnceLock::new();
    execute_native_step_with_registry(
        request,
        BUILTINS.get_or_init(NativeOperationRegistry::default),
    )
}

/// Execute one explicitly registered Rust operation and emit the shared durable
/// worker-effect contract. Unregistered operations never fall through to Python.
pub fn execute_native_step_with_registry(
    request: &RuntimeNativeStepRequest,
    registry: &NativeOperationRegistry,
) -> Result<RecordedWorkerSuccessEffect, RuntimeNativeExecutionError> {
    if request.executor_target != RUST_NATIVE_EXECUTOR_TARGET {
        return Err(RuntimeNativeExecutionError::WrongExecutorTarget {
            expected: RUST_NATIVE_EXECUTOR_TARGET.to_owned(),
            got: request.executor_target.clone(),
        });
    }
    if request.operation.trim().is_empty() {
        return Err(RuntimeNativeExecutionError::EmptyOperation);
    }
    for (field, value) in [
        ("run_id", request.run_id.as_str()),
        ("workflow_id", request.workflow_id.as_str()),
        ("node_id", request.node_id.as_str()),
        ("conversation_id", request.conversation_id.as_str()),
        ("token_id", request.token_id.as_str()),
    ] {
        if value.trim().is_empty() {
            return Err(RuntimeNativeExecutionError::EmptyIdentifier { field });
        }
    }
    if request.step_seq < 0 {
        return Err(RuntimeNativeExecutionError::NegativeStepSequence);
    }
    let Some(handler) = registry.handler(&request.operation, &request.op_version) else {
        return Err(if registry.contains_operation(&request.operation) {
            RuntimeNativeExecutionError::UnsupportedOperationVersion {
                operation: request.operation.clone(),
                version: request.op_version.clone(),
            }
        } else {
            RuntimeNativeExecutionError::UnsupportedOperation(request.operation.clone())
        });
    };
    let output = handler(request)?;
    validate_standard_result_state(request, &output)?;
    let successors =
        authoritative_runtime_successors(&request.runtime_routes, &request.node_id, &[])?;
    Ok(RecordedWorkerSuccessEffect {
        contract_version: RECORDED_RUNTIME_CONTRACT_VERSION,
        effect_id: format!(
            "native-step|{}|{}|{}|{}",
            request.run_id, request.node_id, request.token_id, request.step_seq
        ),
        status: RuntimeWorkerEffectStatus::Success,
        state_update: output.state_update,
        update: None,
        state_schema: Map::new(),
        successors,
        route_next: Vec::new(),
        result: output.result,
        errors: Vec::new(),
        wait_reason: None,
        resume_payload: None,
        usage: None,
        trace_events: vec![{
            let mut trace = json!({
                "schema_version": RUST_NATIVE_OPERATION_SCHEMA_VERSION,
                "event": "runtime.native.step.completed",
                "executor_target": RUST_NATIVE_EXECUTOR_TARGET,
                "operation": request.operation,
                "op_version": request.op_version,
                "run_id": request.run_id,
                "workflow_id": request.workflow_id,
                "conversation_id": request.conversation_id,
                "turn_node_id": request.turn_node_id,
                "node_id": request.node_id,
                "token_id": request.token_id,
                "parent_token_id": request.parent_token_id,
                "step_seq": request.step_seq,
            });
            attach_rust_native_executor_attestation(&mut trace);
            trace
        }],
    })
}

fn validate_standard_result_state(
    request: &RuntimeNativeStepRequest,
    output: &NativeOperationOutput,
) -> Result<(), RuntimeNativeExecutionError> {
    let Some(result) = output.result.as_ref().and_then(Value::as_object) else {
        return Ok(());
    };
    if !result.contains_key("workflow_status") {
        return Ok(());
    }
    let final_state = result.get("final_state").ok_or_else(|| {
        RuntimeNativeExecutionError::OperationFailed {
            operation: request.operation.clone(),
            reason: "standard workflow result is missing final_state".to_owned(),
        }
    })?;
    if result
        .get("workflow_status")
        .and_then(Value::as_str)
        != Some("succeeded")
    {
        return Err(RuntimeNativeExecutionError::OperationFailed {
            operation: request.operation.clone(),
            reason: "successful native effect must report workflow_status=succeeded".to_owned(),
        });
    }

    let state_update = serde_json::to_value(&output.state_update).map_err(|error| {
        RuntimeNativeExecutionError::OperationFailed {
            operation: request.operation.clone(),
            reason: format!("native operation returned an invalid state update: {error}"),
        }
    })?;
    let expected_state = kogwistar_contracts::apply_state_update_values(
        Value::Object(request.state.clone()),
        &state_update,
        None,
        None,
    )
    .map_err(|error| RuntimeNativeExecutionError::OperationFailed {
        operation: request.operation.clone(),
        reason: format!("native operation returned an invalid state update: {error}"),
    })?;
    if final_state != &expected_state {
        return Err(RuntimeNativeExecutionError::OperationFailed {
            operation: request.operation.clone(),
            reason: "result.final_state does not match the operation state_update".to_owned(),
        });
    }
    Ok(())
}

impl RuntimeStepExecutePayload {
    pub fn from_recorded_state(
        state: &RecordedRuntimeState,
        request: RuntimeStepExecuteRequest,
    ) -> Self {
        let executor_target = state
            .state
            .get("_rt_node_executor_targets")
            .and_then(Value::as_object)
            .and_then(|targets| targets.get(&request.node_id))
            .and_then(Value::as_str)
            .map(str::to_owned)
            .or_else(|| {
                request
                    .op
                    .as_str()
                    .filter(|operation| native_operation_name(operation).is_some())
                    .map(|_| RUST_NATIVE_EXECUTOR_TARGET.to_owned())
            });
        Self {
            contract_version: RECORDED_RUNTIME_CONTRACT_VERSION,
            kind: "workflow.step.execute".to_owned(),
            worker_protocol: worker_protocol_from_state(&state.state).to_owned(),
            run_id: state.run_id.clone(),
            workflow_id: state.workflow_id.clone(),
            conversation_id: state.conversation_id.clone(),
            turn_node_id: state.user_turn_node_id.clone(),
            node_id: request.node_id.clone(),
            op: request.op,
            executor_target,
            op_version: state
                .state
                .get("_rt_node_versions")
                .and_then(Value::as_object)
                .and_then(|versions| versions.get(&request.node_id))
                .and_then(Value::as_str)
                .unwrap_or("v1")
                .to_owned(),
            op_environment_id: state
                .state
                .get("_rt_node_environment_ids")
                .and_then(Value::as_object)
                .and_then(|ids| ids.get(&request.node_id))
                .and_then(Value::as_str)
                .map(str::to_owned),
            runtime_routes: state.static_routes.clone(),
            join_mask: request.join_mask,
            token_id: request.token_id,
            parent_token_id: request.parent_token_id,
            step_seq: request.step_seq,
            expected_event_seq: request.expected_event_seq,
            state: state.state.clone(),
            resume_effect: request.resume_effect,
        }
    }
}

fn default_worker_protocol() -> String {
    WORKER_PROTOCOL_SYNC_V1.to_owned()
}

fn worker_protocol_from_state(state: &Map<String, Value>) -> &str {
    match state
        .get("_rt_worker_protocol")
        .and_then(Value::as_str)
        .map(str::trim)
    {
        Some(WORKER_PROTOCOL_ASYNC_V2) => WORKER_PROTOCOL_ASYNC_V2,
        _ => WORKER_PROTOCOL_SYNC_V1,
    }
}
