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

fn validate_native_successors(
    request: &RuntimeNativeStepRequest,
    successors: &[RuntimeSuccessor],
) -> Result<(), RuntimeNativeExecutionError> {
    let mut seen = std::collections::BTreeSet::new();
    for successor in successors {
        if !seen.insert((successor.node_id.as_str(), successor.join_mask)) {
            return Err(RuntimeNativeExecutionError::Routing(
                RecordedRuntimeError::InvalidSuccessorSelection(format!(
                    "native operation selected successor {:?} with join mask {} more than once",
                    successor.node_id, successor.join_mask
                )),
            ));
        }
        if !request.runtime_routes.iter().any(|route| {
            route.source_node_id == request.node_id
                && route.target_node_id == successor.node_id
                && route.join_mask == successor.join_mask
        }) {
            return Err(RuntimeNativeExecutionError::Routing(
                RecordedRuntimeError::InvalidSuccessorSelection(format!(
                    "native operation selected successor {:?} with join mask {} outside the frozen routes",
                    successor.node_id, successor.join_mask
                )),
            ));
        }
    }
    Ok(())
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
    /// Optional run-pinned worker compatibility identity. Kept as JSON here so
    /// invalid persisted values reach the Python adapter and fail closed rather
    /// than being silently dropped. This is not proof of the running binary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub required_worker_identity: Option<Value>,
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
    /// Host-authorized, process-local inputs for native operations. This is
    /// deliberately excluded from serialization and durable workflow state.
    #[serde(skip)]
    pub ephemeral_context: Map<String, Value>,
}

pub fn native_trace_context_fields(request: &RuntimeNativeStepRequest) -> Map<String, Value> {
    let attempt = request
        .ephemeral_context
        .get("_trace_attempt")
        .and_then(Value::as_u64)
        .filter(|attempt| *attempt > 0)
        .unwrap_or(1);
    Map::from_iter([
        ("trace_id".to_owned(), Value::String(request.run_id.clone())),
        (
            "span_id".to_owned(),
            Value::String(format!(
                "stp:{}:{}:{}:{}:{}",
                request.run_id, request.token_id, request.step_seq, attempt, request.node_id
            )),
        ),
        (
            "parent_span_id".to_owned(),
            Value::String(format!("tok:{}:{}", request.run_id, request.token_id)),
        ),
        ("attempt".to_owned(), Value::from(attempt)),
    ])
}

// PYTHON_PARITY_V1: rust=rust/crates/kogwistar-runtime/src/sections/native_operations.rs#attach_native_trace_context; python=kogwistar/runtime/telemetry.py#TraceContext.as_fields; fixture=contracts/golden/runtime-trace-context-v1.json; rust_test=rust/crates/kogwistar-runtime/src/tests.rs#native_trace_context_matches_shared_python_fixture; python_test=tests/runtime/test_trace_context_contract.py#test_trace_context_defaults_match_shared_rust_fixture
fn attach_native_trace_context(trace: &mut Value, request: &RuntimeNativeStepRequest) {
    if let Some(fields) = trace.as_object_mut() {
        fields.extend(native_trace_context_fields(request));
    }
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

/// Versioned native-operation result that can request durable host effects.
/// The host remains responsible for authorization and transactional commit.
#[derive(Clone, Debug)]
pub struct NativeOperationEffectOutput {
    pub output: NativeOperationOutput,
    pub provider_event_acknowledgements: Vec<ProviderEventAcknowledgementV1>,
}

/// Native state result plus typed dynamic successors. The runtime validates
/// every target and join mask against the frozen workflow routes before recording it.
#[derive(Clone, Debug)]
pub struct NativeRoutedOperationOutput {
    pub output: NativeOperationOutput,
    pub successors: Vec<RuntimeSuccessor>,
}

/// Routed native result that also requests durable host effects. This is used
/// when routing and its effect (for example event acknowledgement) must commit
/// as one worker result.
#[derive(Clone, Debug)]
pub struct NativeRoutedOperationEffectOutput {
    pub output: NativeOperationOutput,
    pub successors: Vec<RuntimeSuccessor>,
    pub provider_event_acknowledgements: Vec<ProviderEventAcknowledgementV1>,
}

type NativeOperationHandler = dyn Fn(&RuntimeNativeStepRequest) -> Result<NativeOperationOutput, RuntimeNativeExecutionError>
    + Send
    + Sync;

type NativeOperationEffectHandler = dyn Fn(&RuntimeNativeStepRequest) -> Result<NativeOperationEffectOutput, RuntimeNativeExecutionError>
    + Send
    + Sync;

type NativeRoutedOperationHandler = dyn Fn(
        &RuntimeNativeStepRequest,
    ) -> Result<NativeRoutedOperationOutput, RuntimeNativeExecutionError>
    + Send
    + Sync;

type NativeRoutedOperationEffectHandler = dyn Fn(
        &RuntimeNativeStepRequest,
    ) -> Result<NativeRoutedOperationEffectOutput, RuntimeNativeExecutionError>
    + Send
    + Sync;

type NativeGraphOperationHandler = dyn Fn(
        &RuntimeNativeStepRequest,
    ) -> Result<
        (NativeOperationOutput, kogwistar_contracts::GraphMutationBatchV1),
        RuntimeNativeExecutionError,
    >
    + Send
    + Sync;

type NativeGraphV2OperationHandler = dyn Fn(
        &RuntimeNativeStepRequest,
    ) -> Result<
        (
            NativeOperationOutput,
            Option<kogwistar_contracts::GraphMutationBatchV2>,
        ),
        RuntimeNativeExecutionError,
    >
    + Send
    + Sync;

#[derive(Clone)]
enum RegisteredNativeOperation {
    State(Arc<NativeOperationHandler>),
    StateWithEffects(Arc<NativeOperationEffectHandler>),
    RoutedState(Arc<NativeRoutedOperationHandler>),
    RoutedStateWithEffects(Arc<NativeRoutedOperationEffectHandler>),
    Graph(Arc<NativeGraphOperationHandler>),
    GraphV2(Arc<NativeGraphV2OperationHandler>),
}

/// Versioned registry for trusted Rust operation implementations. Registration
/// is explicit application composition; it is not a plugin loader or a way for
/// workflow input to install executable code.
#[derive(Clone)]
pub struct NativeOperationRegistry {
    handlers: BTreeMap<String, BTreeMap<String, RegisteredNativeOperation>>,
}

impl Default for NativeOperationRegistry {
    fn default() -> Self {
        let mut registry = Self {
            handlers: BTreeMap::new(),
        };
        registry.insert_builtin("noop", native_noop);
        registry.insert_builtin("increment-counter", native_increment_counter);
        registry.insert_builtin("emit-result", native_emit_result);
        registry.insert_builtin("agent.event.request", native_provider_event_request);
        registry.insert_builtin_routed("agent.event.receive", native_provider_event_receive);
        registry.insert_builtin_routed_with_effects(
            "agent.event.ack",
            native_provider_event_ack,
        );
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
            .insert(
                "v1".to_owned(),
                RegisteredNativeOperation::State(Arc::new(handler)),
            );
    }

    fn insert_builtin_routed(
        &mut self,
        operation: &str,
        handler: impl Fn(
            &RuntimeNativeStepRequest,
        ) -> Result<NativeRoutedOperationOutput, RuntimeNativeExecutionError>
        + Send
        + Sync
        + 'static,
    ) {
        self.handlers
            .entry(operation.to_owned())
            .or_default()
            .insert(
                "v1".to_owned(),
                RegisteredNativeOperation::RoutedState(Arc::new(handler)),
            );
    }

    fn insert_builtin_routed_with_effects(
        &mut self,
        operation: &str,
        handler: impl Fn(
            &RuntimeNativeStepRequest,
        ) -> Result<NativeRoutedOperationEffectOutput, RuntimeNativeExecutionError>
        + Send
        + Sync
        + 'static,
    ) {
        self.handlers
            .entry(operation.to_owned())
            .or_default()
            .insert(
                "v1".to_owned(),
                RegisteredNativeOperation::RoutedStateWithEffects(Arc::new(handler)),
            );
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
        self.register_handler(
            operation.into(),
            version.into(),
            RegisteredNativeOperation::State(Arc::new(handler)),
        )
    }

    /// Register the advisory hook node used by the graph-native event workflow.
    ///
    /// The callback is application-owned and synchronous like other native
    /// operations. Its output is confined to bounded annotations; it cannot
    /// select routes, produce graph effects, or change authority fields.
    pub fn register_event_advisory_hook<F>(
        &mut self,
        hook: F,
    ) -> Result<(), NativeOperationRegistrationError>
    where
        F: Fn(
                &RuntimeNativeStepRequest,
                &kogwistar_contracts::ProviderEventV1,
                &str,
            ) -> Result<Value, String>
            + Send
            + Sync
            + 'static,
    {
        self.register("agent.event.hook", "v1", move |request| {
            native_event_advisory_hook(request, &hook)
        })
    }

    /// Register a trusted state operation that returns typed durable effects.
    /// Effects are requests only; the API validates and commits them together
    /// with the runtime transition.
    pub fn register_with_effects<F>(
        &mut self,
        operation: impl Into<String>,
        version: impl Into<String>,
        handler: F,
    ) -> Result<(), NativeOperationRegistrationError>
    where
        F: Fn(
                &RuntimeNativeStepRequest,
            ) -> Result<NativeOperationEffectOutput, RuntimeNativeExecutionError>
            + Send
            + Sync
            + 'static,
    {
        self.register_handler(
            operation.into(),
            version.into(),
            RegisteredNativeOperation::StateWithEffects(Arc::new(handler)),
        )
    }

    /// Register a trusted state handler that selects successors from the
    /// frozen route set. The selected targets are revalidated by the runtime;
    /// handlers cannot create or alter workflow routes.
    pub fn register_routed<F>(
        &mut self,
        operation: impl Into<String>,
        version: impl Into<String>,
        handler: F,
    ) -> Result<(), NativeOperationRegistrationError>
    where
        F: Fn(
                &RuntimeNativeStepRequest,
            ) -> Result<NativeRoutedOperationOutput, RuntimeNativeExecutionError>
            + Send
            + Sync
            + 'static,
    {
        self.register_handler(
            operation.into(),
            version.into(),
            RegisteredNativeOperation::RoutedState(Arc::new(handler)),
        )
    }

    /// Register a trusted routed operation whose durable host effects and
    /// successor selection must be committed together.
    pub fn register_routed_with_effects<F>(
        &mut self,
        operation: impl Into<String>,
        version: impl Into<String>,
        handler: F,
    ) -> Result<(), NativeOperationRegistrationError>
    where
        F: Fn(
                &RuntimeNativeStepRequest,
            ) -> Result<NativeRoutedOperationEffectOutput, RuntimeNativeExecutionError>
            + Send
            + Sync
            + 'static,
    {
        self.register_handler(
            operation.into(),
            version.into(),
            RegisteredNativeOperation::RoutedStateWithEffects(Arc::new(handler)),
        )
    }

    /// Register a trusted pure handler that proposes a graph batch. The host
    /// must authorize and commit the returned batch with the worker effect.
    pub fn register_graph<F>(
        &mut self,
        operation: impl Into<String>,
        version: impl Into<String>,
        handler: F,
    ) -> Result<(), NativeOperationRegistrationError>
    where
        F: Fn(
                &RuntimeNativeStepRequest,
            ) -> Result<
                (NativeOperationOutput, kogwistar_contracts::GraphMutationBatchV1),
                RuntimeNativeExecutionError,
            > + Send
            + Sync
            + 'static,
    {
        self.register_handler(
            operation.into(),
            version.into(),
            RegisteredNativeOperation::Graph(Arc::new(handler)),
        )
    }

    /// Register a trusted native handler that requests host-derived default
    /// ACL records. The API transaction owns ACL versioning and persistence.
    /// A state-only result may return `None` when the operation has no graph
    /// effects; it must not manufacture a no-op graph mutation.
    pub fn register_graph_v2<F>(
        &mut self,
        operation: impl Into<String>,
        version: impl Into<String>,
        handler: F,
    ) -> Result<(), NativeOperationRegistrationError>
    where
        F: Fn(
                &RuntimeNativeStepRequest,
            ) -> Result<
                (
                    NativeOperationOutput,
                    Option<kogwistar_contracts::GraphMutationBatchV2>,
                ),
                RuntimeNativeExecutionError,
            > + Send
            + Sync
            + 'static,
    {
        self.register_handler(
            operation.into(),
            version.into(),
            RegisteredNativeOperation::GraphV2(Arc::new(handler)),
        )
    }

    fn register_handler(
        &mut self,
        operation: String,
        version: String,
        handler: RegisteredNativeOperation,
    ) -> Result<(), NativeOperationRegistrationError> {
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
        versions.insert(version, handler);
        Ok(())
    }

    fn handler(&self, operation: &str, version: &str) -> Option<&RegisteredNativeOperation> {
        self.handlers.get(operation)?.get(version)
    }

    fn contains_operation(&self, operation: &str) -> bool {
        self.handlers.contains_key(operation)
    }

    pub fn supports(&self, operation: &str, version: &str) -> bool {
        self.handler(operation, version).is_some()
    }

    /// Return registered operation identities in deterministic order.
    pub fn registered_operations(&self) -> Vec<(String, String)> {
        self.handlers
            .iter()
            .flat_map(|(operation, versions)| {
                versions
                    .keys()
                    .map(move |version| (operation.clone(), version.clone()))
            })
            .collect()
    }
}

const MAX_EVENT_ADVISORY_ANNOTATION_CHARS: usize = 6_000;

fn native_event_advisory_hook<F>(
    request: &RuntimeNativeStepRequest,
    hook: &F,
) -> Result<NativeOperationOutput, RuntimeNativeExecutionError>
where
    F: Fn(
            &RuntimeNativeStepRequest,
            &kogwistar_contracts::ProviderEventV1,
            &str,
        ) -> Result<Value, String>
        + Send
        + Sync
        + 'static,
{
    let current = request
        .state
        .get("agent_event_current")
        .and_then(Value::as_object)
        .ok_or_else(|| RuntimeNativeExecutionError::OperationFailed {
            operation: request.operation.clone(),
            reason: "current provider event is missing".to_owned(),
        })?;
    let event_id = current
        .get("event_id")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| RuntimeNativeExecutionError::OperationFailed {
            operation: request.operation.clone(),
            reason: "current provider event ID is missing".to_owned(),
        })?;
    let event_value = current
        .get("event")
        .cloned()
        .ok_or_else(|| RuntimeNativeExecutionError::OperationFailed {
            operation: request.operation.clone(),
            reason: "current provider event payload is missing".to_owned(),
        })?;
    let event: kogwistar_contracts::ProviderEventV1 =
        serde_json::from_value(event_value).map_err(|_| {
            RuntimeNativeExecutionError::OperationFailed {
                operation: request.operation.clone(),
                reason: "current provider event payload is invalid".to_owned(),
            }
        })?;
    if request.state.get("agent_event_route").and_then(Value::as_str) != Some("advisory")
        || event_route_for_claim(request, &event)?.0 != "advisory"
    {
        return Err(RuntimeNativeExecutionError::OperationFailed {
            operation: request.operation.clone(),
            reason: "current provider event is not eligible for advisory handling".to_owned(),
        });
    }

    let key_material = format!("advisory-hook\0{}\0{event_id}", request.run_id);
    let idempotency_key = format!(
        "agent-advisory-hook:v1:{}",
        hex::encode(Sha256::digest(key_material.as_bytes()))
    );
    let raw_annotations = hook(request, &event, &idempotency_key).map_err(|_| {
        RuntimeNativeExecutionError::OperationFailed {
            operation: request.operation.clone(),
            reason: "advisory hook callback failed".to_owned(),
        }
    })?;
    let annotations = bounded_event_advisory_annotations(raw_annotations);
    let hook_count = request
        .state
        .get("agent_event_advisory_hook_count")
        .and_then(Value::as_i64)
        .filter(|count| *count >= 0)
        .unwrap_or(0);
    let response_hook_count = request
        .state
        .get("agent_event_advisory_response_hook_count")
        .and_then(Value::as_i64)
        .filter(|count| *count >= 0)
        .unwrap_or(0);
    let mut update = Map::new();
    update.insert("agent_event_hook_annotations".to_owned(), annotations);
    if let Some(provider_request) =
        event_advisory_provider_request_update(&request.state, update.get("agent_event_hook_annotations").expect("inserted annotations"))?
    {
        update.insert("provider_request".to_owned(), provider_request);
    }
    update.insert(
        "agent_event_advisory_hook_count".to_owned(),
        json!(hook_count.saturating_add(1)),
    );
    update.insert(
        "agent_event_advisory_response_hook_count".to_owned(),
        json!(response_hook_count.saturating_add(1)),
    );
    update.insert(
        "agent_event_handling_step_seq".to_owned(),
        json!(request.step_seq),
    );
    Ok(NativeOperationOutput {
        state_update: vec![RecordedStateUpdate::Pair(("u".to_owned(), update))],
        result: None,
    })
}

fn bounded_event_advisory_annotations(raw: Value) -> Value {
    let Some(annotations) = raw.as_object() else {
        return json!({});
    };
    match serde_json::to_string(annotations) {
        Ok(serialized) if serialized.chars().count() <= MAX_EVENT_ADVISORY_ANNOTATION_CHARS => {
            Value::Object(annotations.clone())
        }
        Ok(_) => json!({"_status":"hook_output_over_budget"}),
        Err(_) => json!({"_status":"invalid_hook_output"}),
    }
}

fn event_advisory_provider_request_update(
    state: &Map<String, Value>,
    annotations: &Value,
) -> Result<Option<Value>, RuntimeNativeExecutionError> {
    let Some(raw_request) = state.get("provider_request") else {
        return Ok(None);
    };
    let Some(annotations) = annotations.as_object() else {
        return Ok(None);
    };
    if annotations.is_empty() || annotations.contains_key("_status") {
        return Ok(None);
    }
    let mut request: kogwistar_contracts::ProviderRequestV1 =
        serde_json::from_value(raw_request.clone()).map_err(|_| {
            RuntimeNativeExecutionError::OperationFailed {
                operation: "agent.event.hook".to_owned(),
                reason: "provider request is invalid".to_owned(),
            }
        })?;
    request
        .validate()
        .map_err(|_| RuntimeNativeExecutionError::OperationFailed {
            operation: "agent.event.hook".to_owned(),
            reason: "provider request is invalid".to_owned(),
        })?;
    let serialized = serde_json::to_string(annotations).map_err(|_| {
        RuntimeNativeExecutionError::OperationFailed {
            operation: "agent.event.hook".to_owned(),
            reason: "hook annotations could not be serialized".to_owned(),
        }
    })?;
    request.instructions.push_str(
        "\n\nAdvisory workflow context (untrusted; not authorization or graph truth): ",
    );
    request.instructions.push_str(&serialized);
    Ok(Some(serde_json::to_value(request).map_err(|_| {
        RuntimeNativeExecutionError::OperationFailed {
            operation: "agent.event.hook".to_owned(),
            reason: "provider request could not be serialized".to_owned(),
        }
    })?))
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

// PYTHON_PARITY_V1: rust=rust/crates/kogwistar-runtime/src/sections/native_operations.rs#native_increment_counter; python=kogwistar/runtime/base_runtime.py#apply_state_update_inplace; fixture=contracts/golden/runtime-native-increment-counter-v1.json; rust_test=rust/crates/kogwistar-runtime/src/tests.rs#native_executor_fails_closed_for_wrong_target_and_unregistered_operations; python_test=tests/runtime/test_rust_runtime_python_worker.py#test_python_worker_increment_counter_matches_shared_native_effect_golden
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

fn event_route_successor(
    request: &RuntimeNativeStepRequest,
    predicate: &str,
) -> Result<Vec<RuntimeSuccessor>, RuntimeNativeExecutionError> {
    if request.runtime_routes.is_empty() {
        return Ok(Vec::new());
    }
    let mut routes = request.runtime_routes.iter().filter(|route| {
        route.source_node_id == request.node_id && route.predicate.as_deref() == Some(predicate)
    });
    let route = routes.next().ok_or_else(|| RuntimeNativeExecutionError::OperationFailed {
        operation: request.operation.clone(),
        reason: format!("frozen event workflow is missing route predicate {predicate:?}"),
    })?;
    if routes.next().is_some() {
        return Err(RuntimeNativeExecutionError::OperationFailed {
            operation: request.operation.clone(),
            reason: format!("frozen event workflow has ambiguous route predicate {predicate:?}"),
        });
    }
    Ok(vec![RuntimeSuccessor {
        node_id: route.target_node_id.clone(),
        join_mask: route.join_mask,
    }])
}

fn native_provider_event_request(
    request: &RuntimeNativeStepRequest,
) -> Result<NativeOperationOutput, RuntimeNativeExecutionError> {
    let result = request
        .ephemeral_context
        .get("provider_event_request_result")
        .and_then(Value::as_object)
        .ok_or_else(|| RuntimeNativeExecutionError::OperationFailed {
            operation: request.operation.clone(),
            reason: "host provider-event request result is unavailable".to_owned(),
        })?;
    let sequence = |field: &'static str| {
        result
            .get(field)
            .and_then(Value::as_i64)
            .filter(|value| *value >= 0)
            .ok_or(RuntimeNativeExecutionError::InvalidState {
                field: "provider_event_request_result",
                reason: "expected non-negative event counters",
            })
    };
    let provider_sequence = sequence("provider_sequence")?;
    let client_sse_sequence = sequence("client_sse_sequence")?;
    let semantic_event_count = sequence("semantic_event_count")?;
    if semantic_event_count == 0 {
        return Err(RuntimeNativeExecutionError::InvalidState {
            field: "provider_event_request_result.semantic_event_count",
            reason: "provider request must publish at least one semantic event",
        });
    }

    let mut update = Map::new();
    update.insert("agent_event_route".to_owned(), json!("receive"));
    update.insert("agent_event_provider_sequence".to_owned(), json!(provider_sequence));
    update.insert(
        "agent_event_client_sse_sequence".to_owned(),
        json!(client_sse_sequence),
    );
    update.insert("agent_event_advisory_response_hook_count".to_owned(), json!(0));
    let mut final_state = request.state.clone();
    final_state.extend(update.clone());
    Ok(NativeOperationOutput {
        state_update: vec![RecordedStateUpdate::Pair(("u".to_owned(), update))],
        result: Some(json!({
            "workflow_status": "succeeded",
            "final_state": final_state,
            "semantic_event_count": semantic_event_count,
        })),
    })
}

fn event_route_for_claim(
    request: &RuntimeNativeStepRequest,
    event: &kogwistar_contracts::ProviderEventV1,
) -> Result<(&'static str, &'static str), RuntimeNativeExecutionError> {
    let hook_count = request
        .state
        .get("agent_event_advisory_hook_count")
        .and_then(Value::as_i64)
        .unwrap_or(0)
        .max(0);
    let max_hooks = request
        .state
        .get("agent_event_max_advisory_hooks_per_run")
        .map(|value| {
            value.as_i64().filter(|count| *count >= 0).ok_or_else(|| {
                RuntimeNativeExecutionError::InvalidState {
                    field: "agent_event_max_advisory_hooks_per_run",
                    reason: "expected a non-negative integer",
                }
            })
        })
        .transpose()?
        .unwrap_or(8);
    let advisory_field_names = request
        .state
        .get("agent_event_advisory_field_names")
        .map(|value| {
            let fields = value.as_array().ok_or(RuntimeNativeExecutionError::InvalidState {
                field: "agent_event_advisory_field_names",
                reason: "expected an array of strings",
            })?;
            fields
                .iter()
                .map(|field| {
                    field.as_str().ok_or(RuntimeNativeExecutionError::InvalidState {
                        field: "agent_event_advisory_field_names",
                        reason: "expected an array of strings",
                    })
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?;
    let advisory_field = match event {
        kogwistar_contracts::ProviderEventV1::FieldCompleted { field_name, .. } => {
            advisory_field_names
                .as_ref()
                .is_some_and(|fields| fields.iter().any(|field| *field == field_name))
        }
        kogwistar_contracts::ProviderEventV1::ReasoningSummary { .. } => request
            .state
            .get("agent_event_allow_reasoning_summaries")
            .map(|value| value.as_bool().ok_or(RuntimeNativeExecutionError::InvalidState {
                field: "agent_event_allow_reasoning_summaries",
                reason: "expected a boolean",
            }))
            .transpose()?
            .unwrap_or(false),
        _ => false,
    };
    if advisory_field && hook_count < max_hooks {
        return Ok(("advisory", "event_is_advisory"));
    }
    match event {
        kogwistar_contracts::ProviderEventV1::ToolCallCompleted { .. } => {
            Ok(("tool", "event_is_tool_call"))
        }
        kogwistar_contracts::ProviderEventV1::ResponseCompleted { .. } => {
            Ok(("final", "event_is_response_complete"))
        }
        kogwistar_contracts::ProviderEventV1::ResponseFailed { .. } => {
            Ok(("failed", "event_is_response_failed"))
        }
        _ => Ok(("no_action", "event_requires_no_action")),
    }
}

fn native_provider_event_receive(
    request: &RuntimeNativeStepRequest,
) -> Result<NativeRoutedOperationOutput, RuntimeNativeExecutionError> {
    let event = request
        .ephemeral_context
        .get("authorized_provider_event")
        .ok_or_else(|| RuntimeNativeExecutionError::OperationFailed {
            operation: request.operation.clone(),
            reason: "host-authorized provider event is unavailable".to_owned(),
        })?;
    let claim = request
        .ephemeral_context
        .get("authorized_provider_event_claim")
        .ok_or_else(|| RuntimeNativeExecutionError::OperationFailed {
            operation: request.operation.clone(),
            reason: "host-authorized provider event claim is unavailable".to_owned(),
        })?;
    let (typed_event, _) = validated_provider_event_claim(request, event, claim)?;
    let (route, predicate) = event_route_for_claim(request, &typed_event.event)?;
    let mut update = Map::new();
    update.insert("agent_event_current".to_owned(), event.clone());
    update.insert("agent_event_claim".to_owned(), claim.clone());
    update.insert("agent_event_route".to_owned(), json!(route));
    update.insert(
        "agent_event_handling_step_seq".to_owned(),
        json!(request.step_seq),
    );
    let mut final_state = request.state.clone();
    final_state.extend(update.clone());
    Ok(NativeRoutedOperationOutput {
        output: NativeOperationOutput {
            state_update: vec![RecordedStateUpdate::Pair(("u".to_owned(), update))],
            result: Some(json!({
                "workflow_status": "succeeded",
                "final_state": final_state,
                "event_id": typed_event.event_id,
            })),
        },
        successors: event_route_successor(request, predicate)?,
    })
}

fn native_provider_event_ack(
    request: &RuntimeNativeStepRequest,
) -> Result<NativeRoutedOperationEffectOutput, RuntimeNativeExecutionError> {
    let event = request
        .state
        .get("agent_event_current")
        .ok_or_else(|| RuntimeNativeExecutionError::OperationFailed {
            operation: request.operation.clone(),
            reason: "durable provider event is unavailable in workflow state".to_owned(),
        })?;
    let claim = request
        .state
        .get("agent_event_claim")
        .ok_or_else(|| RuntimeNativeExecutionError::OperationFailed {
            operation: request.operation.clone(),
            reason: "durable provider event claim is unavailable in workflow state".to_owned(),
        })?;
    let (typed_event, acknowledgement) = validated_provider_event_claim(request, event, claim)?;
    let next_route = match request
        .state
        .get("agent_event_route")
        .and_then(Value::as_str)
        .unwrap_or("no_action")
    {
        "advisory" | "no_action" | "tool" => ("receive", "event_stream_has_more"),
        "resume" => ("request", "event_turn_requires_model_resume"),
        "final" => ("done", "event_run_succeeded"),
        "failed" => ("failed_terminal", "event_run_failed"),
        _ => ("failed_terminal", "event_run_failed"),
    };
    let mut update = Map::new();
    update.insert("agent_event_route".to_owned(), json!(next_route.0));
    update.insert("agent_event_claim".to_owned(), Value::Null);
    let mut final_state = request.state.clone();
    final_state.extend(update.clone());
    Ok(NativeRoutedOperationEffectOutput {
        output: NativeOperationOutput {
            state_update: vec![RecordedStateUpdate::Pair(("u".to_owned(), update))],
            result: Some(json!({
                "workflow_status": "succeeded",
                "final_state": final_state,
                "event_id": typed_event.event_id,
            })),
        },
        successors: event_route_successor(request, next_route.1)?,
        provider_event_acknowledgements: vec![acknowledgement],
    })
}

fn validated_provider_event_claim(
    request: &RuntimeNativeStepRequest,
    event: &Value,
    claim: &Value,
) -> Result<
    (
        kogwistar_contracts::DurableProviderEventV1,
        ProviderEventAcknowledgementV1,
    ),
    RuntimeNativeExecutionError,
> {
    let typed_event = serde_json::from_value::<kogwistar_contracts::DurableProviderEventV1>(
        event.clone(),
    )
    .map_err(|_| RuntimeNativeExecutionError::OperationFailed {
        operation: request.operation.clone(),
        reason: "provider event does not match its versioned contract".to_owned(),
    })?;
    let event_id = claim
        .get("event_id")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty());
    let message_id = claim
        .get("message_id")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty());
    let claimed_by = claim
        .get("claimed_by")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty());
    if event_id != Some(typed_event.event_id.as_str())
        || message_id != event_id.map(|id| format!("msg:{id}")).as_deref()
        || claimed_by.is_none()
        || typed_event.trace.run_id != request.run_id
    {
        return Err(RuntimeNativeExecutionError::OperationFailed {
            operation: request.operation.clone(),
            reason: "provider event and claim identities do not match this run".to_owned(),
        });
    }
    Ok((
        typed_event,
        ProviderEventAcknowledgementV1 {
            message_id: message_id.expect("checked non-empty above").to_owned(),
            claimed_by: claimed_by.expect("checked non-empty above").to_owned(),
            event_id: event_id.expect("checked non-empty above").to_owned(),
        },
    ))
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
    let handler = native_handler_for_request(request, registry)?;
    let (
        output,
        graph_mutation_batch,
        graph_mutation_batch_v2,
        worker_selected,
        provider_event_acknowledgements,
    ) = match handler {
        RegisteredNativeOperation::State(handler) => {
            (handler(request)?, None, None, Vec::new(), Vec::new())
        }
        RegisteredNativeOperation::StateWithEffects(handler) => {
            let effect = handler(request)?;
            (
                effect.output,
                None,
                None,
                Vec::new(),
                effect.provider_event_acknowledgements,
            )
        }
        RegisteredNativeOperation::RoutedState(handler) => {
            let routed = handler(request)?;
            (routed.output, None, None, routed.successors, Vec::new())
        }
        RegisteredNativeOperation::RoutedStateWithEffects(handler) => {
            let routed = handler(request)?;
            (
                routed.output,
                None,
                None,
                routed.successors,
                routed.provider_event_acknowledgements,
            )
        }
        RegisteredNativeOperation::Graph(handler) => {
            let (output, batch) = handler(request)?;
            batch
                .validate()
                .map_err(|error| RuntimeNativeExecutionError::OperationFailed {
                    operation: request.operation.clone(),
                    reason: format!("invalid graph mutation batch: {error}"),
                })?;
            (output, Some(batch), None, Vec::new(), Vec::new())
        }
        RegisteredNativeOperation::GraphV2(handler) => {
            let (output, batch) = handler(request)?;
            if let Some(batch) = &batch {
                batch
                    .validate()
                    .map_err(|error| RuntimeNativeExecutionError::OperationFailed {
                        operation: request.operation.clone(),
                        reason: format!("invalid ACL-aware graph mutation batch: {error}"),
                    })?;
            }
            (output, None, batch, Vec::new(), Vec::new())
        }
    };
    validate_native_output_state(request, &output)?;
    validate_native_successors(request, &worker_selected)?;
    let successors = authoritative_runtime_successors_for_result(
        &request.runtime_routes,
        &request.node_id,
        &worker_selected,
        &[],
        false,
    )?;
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
            attach_native_trace_context(&mut trace, request);
            attach_rust_native_executor_attestation(&mut trace);
            trace
        }],
        graph_mutation_batch,
        graph_mutation_batch_v2,
        provider_event_publications: Vec::new(),
        provider_event_acknowledgements,
    })
}

/// Validate the request envelope and operation registration without invoking
/// the handler, for services that perform persistence or provider work first.
pub fn validate_native_step_registration(
    request: &RuntimeNativeStepRequest,
    registry: &NativeOperationRegistry,
) -> Result<(), RuntimeNativeExecutionError> {
    native_handler_for_request(request, registry).map(|_| ())
}

fn native_handler_for_request<'a>(
    request: &RuntimeNativeStepRequest,
    registry: &'a NativeOperationRegistry,
) -> Result<&'a RegisteredNativeOperation, RuntimeNativeExecutionError> {
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
    Ok(handler)
}

fn validate_native_output_state(
    request: &RuntimeNativeStepRequest,
    output: &NativeOperationOutput,
) -> Result<(), RuntimeNativeExecutionError> {
    let expected_state = if output.state_update.is_empty() {
        None
    } else {
        Some(
            kogwistar_contracts::apply_state_update_pairs(
                Value::Object(request.state.clone()),
                output.state_update.iter().map(RecordedStateUpdate::parts),
            )
            .map_err(|error| RuntimeNativeExecutionError::OperationFailed {
                operation: request.operation.clone(),
                reason: format!("native operation returned an invalid state update: {error}"),
            })?,
        )
    };

    if let Some(result) = output.result.as_ref().and_then(Value::as_object)
        && result.contains_key("workflow_status")
    {
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
        let expected_state = expected_state
            .unwrap_or_else(|| Value::Object(request.state.clone()));
        if final_state != &expected_state {
            return Err(RuntimeNativeExecutionError::OperationFailed {
                operation: request.operation.clone(),
                reason: "result.final_state does not match the operation state_update".to_owned(),
            });
        }
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
            required_worker_identity: state
                .state
                .get("_rt_node_worker_identities")
                .and_then(Value::as_object)
                .and_then(|identities| identities.get(&request.node_id))
                .cloned(),
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
