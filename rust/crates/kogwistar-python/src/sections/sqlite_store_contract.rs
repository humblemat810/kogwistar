// Durable SQLite ABI request/operation contracts. Python parity: kogwistar.engine_core.engine_sqlite.
/// Durable SQLite Phase-3 ABI. This accepts a database path, never a snapshot.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SqliteStoreRequest {
    path: String,
    #[serde(default)]
    context_id: Option<String>,
    #[serde(default)]
    transaction_id: Option<String>,
    #[serde(default)]
    reuse_session: bool,
    operation: SqliteStoreOperation,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NamedProjectionCasOperation {
    namespace: String,
    key: String,
    payload: Map<String, Value>,
    expected_last_authoritative_seq: Option<i64>,
    expected_last_materialized_seq: Option<i64>,
    last_authoritative_seq: i64,
    last_materialized_seq: i64,
    projection_schema_version: i64,
    materialization_status: String,
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum SqliteStoreOperation {
    OpenInit,
    Close,
    BeginTransaction,
    CommitTransaction,
    RollbackTransaction,
    CurrentGlobalSeq,
    NextGlobalSeq,
    CurrentUserSeq {
        user_id: String,
    },
    NextUserSeq {
        user_id: String,
    },
    SetUserSeq {
        user_id: String,
        value: i64,
    },
    CurrentScopedSeq {
        scope_id: String,
    },
    NextScopedSeq {
        scope_id: String,
    },
    SetScopedSeq {
        scope_id: String,
        value: i64,
    },
    AllocEventSeq {
        namespace: String,
    },
    RawAppend {
        namespace: String,
        event_id: String,
        entity_kind: String,
        entity_id: String,
        op: String,
        payload_json: String,
    },
    RawRestore {
        namespace: String,
        seq: i64,
        event_id: String,
        entity_kind: String,
        entity_id: String,
        op: String,
        payload_json: String,
        created_at: i64,
    },
    ExclusiveRawReplay {
        namespace: String,
        after_seq: i64,
        limit: usize,
    },
    LatestRetainedEventSeq {
        namespace: String,
    },
    PruneEntityEventsAfter {
        namespace: String,
        to_seq: i64,
    },
    CursorGet {
        namespace: String,
        consumer: String,
    },
    LegacyCursorSet {
        namespace: String,
        consumer: String,
        last_seq: i64,
    },
    StrictCursorAdvance {
        namespace: String,
        consumer: String,
        last_seq: i64,
    },
    RecoverEntityProjection {
        namespace: String,
        consumer: String,
        projection_namespace: String,
        projection_key: String,
        batch_limit: usize,
        #[serde(default)]
        abort_after_projection: bool,
    },
    RebuildEntityProjection {
        namespace: String,
        consumer: String,
        projection_namespace: String,
        projection_key: String,
        #[serde(default)]
        abort_after_projection: bool,
    },
    GetNamedProjection {
        namespace: String,
        key: String,
    },
    ListNamedProjections {
        namespace: String,
    },
    ReplaceNamedProjection {
        namespace: String,
        key: String,
        payload: Map<String, Value>,
        last_authoritative_seq: i64,
        last_materialized_seq: i64,
        projection_schema_version: i64,
        materialization_status: String,
    },
    CompareAndSwapNamedProjection {
        namespace: String,
        key: String,
        payload: Map<String, Value>,
        expected_last_authoritative_seq: Option<i64>,
        expected_last_materialized_seq: Option<i64>,
        last_authoritative_seq: i64,
        last_materialized_seq: i64,
        projection_schema_version: i64,
        materialization_status: String,
    },
    CompareAndSwapNamedProjections {
        updates: Vec<NamedProjectionCasOperation>,
    },
    ClearNamedProjection {
        namespace: String,
        key: String,
    },
    ClearProjectionNamespace {
        namespace: String,
    },
    GetStage1NodeProjection {
        namespace: String,
        key: String,
    },
    ListStage1NodeProjections {
        namespace: String,
    },
    ReplaceStage1NodeProjection {
        namespace: String,
        key: String,
        payload: Map<String, Value>,
        last_authoritative_seq: i64,
        last_materialized_seq: i64,
        projection_schema_version: i64,
        materialization_status: String,
    },
    ClearStage1NodeProjection {
        namespace: String,
        key: String,
    },
    ClearStage1NodeProjectionIfCurrent {
        namespace: String,
        key: String,
        expected_last_authoritative_seq: i64,
    },
    PutWorkflowDesignSnapshot {
        workflow_id: String,
        version: i64,
        seq: i64,
        payload_json: String,
        schema_version: i64,
    },
    GetWorkflowDesignSnapshot {
        workflow_id: String,
        max_version: i64,
        schema_version: i64,
    },
    ClearWorkflowDesignSnapshots {
        workflow_id: String,
    },
    PutWorkflowDesignDelta {
        workflow_id: String,
        version: i64,
        prev_version: i64,
        target_seq: i64,
        forward_json: String,
        inverse_json: String,
        schema_version: i64,
    },
    GetWorkflowDesignDelta {
        workflow_id: String,
        version: i64,
        schema_version: i64,
    },
    ClearWorkflowDesignDeltas {
        workflow_id: String,
    },
    CreateServerRun {
        run_id: String,
        conversation_id: String,
        workflow_id: String,
        #[serde(default)]
        user_id: Option<String>,
        user_turn_node_id: String,
        #[serde(default = "default_run_status")]
        status: String,
    },
    GetServerRun {
        run_id: String,
    },
    ListServerRuns {
        #[serde(default)]
        status: Option<String>,
        #[serde(default)]
        workflow_id: Option<String>,
        #[serde(default)]
        conversation_id: Option<String>,
        #[serde(default = "default_run_limit")]
        limit: usize,
    },
    AppendServerRunEvent {
        run_id: String,
        event_type: String,
        payload_json: String,
    },
    ListServerRunEvents {
        run_id: String,
        #[serde(default)]
        after_seq: i64,
        #[serde(default = "default_run_event_limit")]
        limit: usize,
    },
    UpdateServerRun {
        run_id: String,
        status: String,
        assistant_turn_node_id: Option<String>,
        result_json: Option<String>,
        error_json: Option<String>,
        started_at_ms: Option<i64>,
        finished_at_ms: Option<i64>,
        #[serde(default)]
        cancel_requested: Option<bool>,
    },
    RequestServerRunCancel {
        run_id: String,
    },
    /// ADR-015 Phase-4 recorded-only durable runtime transition.  The result
    /// is already supplied by a worker; this ABI cannot dispatch work.
    ApplyRecordedRuntimeTransition {
        transition: Box<RecordedRuntimeTransition>,
        #[serde(default)]
        abort_after_writes: bool,
    },
    /// Complete one claimed Python-worker request and record its result in one
    /// Rust-owned SQLite transaction. No callback executes inside the UoW.
    ApplyClaimedRecordedRuntimeTransition {
        handoff: RecordedWorkerHandoff,
        transition: Box<RecordedRuntimeTransition>,
        #[serde(default)]
        abort_after_writes: bool,
    },
    /// Reopen/read recorded runtime state.  It never resumes or dispatches.
    ReadRecordedRuntimeState {
        run_id: String,
        workflow_id: String,
        conversation_id: String,
    },
    GetIndexAppliedFingerprint {
        #[serde(default = "default_namespace")]
        namespace: String,
        coalesce_key: String,
    },
    SetIndexAppliedFingerprint {
        #[serde(default = "default_namespace")]
        namespace: String,
        coalesce_key: String,
        #[serde(default)]
        applied_fingerprint: Option<String>,
        #[serde(default)]
        last_job_id: Option<String>,
    },
    EnqueueIndexJob {
        job_id: String,
        #[serde(default = "default_namespace")]
        namespace: String,
        entity_kind: String,
        entity_id: String,
        index_kind: String,
        op: String,
        #[serde(default)]
        payload_json: Option<String>,
        #[serde(default = "default_max_retries")]
        max_retries: i64,
    },
    ClaimIndexJobs {
        #[serde(default = "default_claim_limit")]
        limit: usize,
        #[serde(default = "default_lease_seconds")]
        lease_seconds: i64,
        #[serde(default = "default_optional_namespace")]
        namespace: Option<String>,
    },
    MarkIndexJobDone {
        job_id: String,
        #[serde(default)]
        claim_token: Option<String>,
    },
    AcceptIndexJobResult {
        job_id: String,
        claim_token: String,
        result_json: String,
        result_sha256: String,
    },
    GetIndexJobResult {
        job_id: String,
    },
    MarkIndexJobFailed {
        job_id: String,
        error: String,
        #[serde(rename = "final", default = "default_true")]
        final_: bool,
        #[serde(default)]
        claim_token: Option<String>,
    },
    BumpRetryAndRequeue {
        job_id: String,
        error: String,
        next_run_at_seconds: i64,
        #[serde(default)]
        claim_token: Option<String>,
    },
    RenewIndexJobLease {
        job_id: String,
        claim_token: String,
        lease_seconds: i64,
    },
    RequeueIndexJobAtTail {
        job_id: String,
        payload_json: String,
        #[serde(default)]
        delay_seconds: i64,
        #[serde(default)]
        claim_token: Option<String>,
    },
    ListIndexJobs {
        #[serde(default = "default_optional_namespace")]
        namespace: Option<String>,
        #[serde(default)]
        status: Option<String>,
        #[serde(default)]
        entity_kind: Option<String>,
        #[serde(default)]
        entity_id: Option<String>,
        #[serde(default)]
        index_kind: Option<String>,
        #[serde(default = "default_list_limit")]
        limit: usize,
    },
    ProjectLaneMessage {
        message_id: String,
        #[serde(default = "default_namespace")]
        namespace: String,
        #[serde(default = "default_lane_purpose")]
        purpose: String,
        inbox_id: String,
        conversation_id: String,
        recipient_id: String,
        sender_id: String,
        msg_type: String,
        status: String,
        created_at: i64,
        available_at: i64,
        #[serde(default)]
        run_id: Option<String>,
        #[serde(default)]
        step_id: Option<String>,
        #[serde(default)]
        correlation_id: Option<String>,
        #[serde(default)]
        payload_json: Option<String>,
        #[serde(default)]
        error_json: Option<String>,
    },
    GetProjectedLaneMessage {
        message_id: String,
    },
    UpdateProjectedLaneMessageStatus {
        message_id: String,
        status: String,
        #[serde(default)]
        error_json: Option<String>,
    },
    UpdateProjectedLaneMessageLinks {
        message_id: String,
        #[serde(default)]
        prev_message_id: Option<String>,
        #[serde(default)]
        next_message_id: Option<String>,
        #[serde(default)]
        inbox_tail_message_id: Option<String>,
        #[serde(default)]
        conversation_tail_message_id: Option<String>,
    },
    ListProjectedLaneMessages {
        #[serde(default = "default_optional_namespace")]
        namespace: Option<String>,
        #[serde(default)]
        purpose: Option<String>,
        #[serde(default)]
        inbox_id: Option<String>,
        #[serde(default)]
        conversation_id: Option<String>,
        #[serde(default)]
        run_id: Option<String>,
        #[serde(default)]
        status: Option<String>,
        #[serde(default)]
        msg_type: Option<String>,
        #[serde(default)]
        sender_id: Option<String>,
        #[serde(default)]
        recipient_id: Option<String>,
        #[serde(default)]
        correlation_id: Option<String>,
        #[serde(default)]
        reply_to_message_id: Option<String>,
        #[serde(default)]
        created_at_gte: Option<i64>,
        #[serde(default)]
        created_at_lte: Option<i64>,
        #[serde(default)]
        available_at_gte: Option<i64>,
        #[serde(default)]
        available_at_lte: Option<i64>,
        #[serde(default = "default_list_limit")]
        limit: usize,
        #[serde(default)]
        newest_first: bool,
    },
    ClearProjectedLaneMessages {
        namespace: String,
    },
    ClaimProjectedLaneMessages {
        #[serde(default = "default_namespace")]
        namespace: String,
        inbox_id: String,
        claimed_by: String,
        #[serde(default = "default_claim_limit")]
        limit: usize,
        #[serde(default = "default_lease_seconds")]
        lease_seconds: i64,
        #[serde(default)]
        message_ids: Option<Vec<String>>,
        #[serde(default)]
        run_id: Option<String>,
        #[serde(default)]
        msg_type: Option<String>,
        #[serde(default)]
        recipient_id: Option<String>,
    },
    AckProjectedLaneMessage {
        message_id: String,
        claimed_by: String,
    },
    RequeueProjectedLaneMessage {
        message_id: String,
        claimed_by: String,
        #[serde(default)]
        error_json: Option<String>,
        #[serde(default)]
        delay_seconds: i64,
    },
    DeadLetterProjectedLaneMessage {
        message_id: String,
        claimed_by: String,
        #[serde(default)]
        error_json: Option<String>,
    },
    Batch {
        operations: Vec<SqliteStoreOperation>,
        #[serde(default)]
        abort: bool,
    },
}
