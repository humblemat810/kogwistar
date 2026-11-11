// PostgreSQL ABI DTOs and graph-mutation translation helpers. Python parity: kogwistar.engine_core.engine_postgres.
/// Durable PostgreSQL Phase-3 ABI. Managed facades reuse a process runtime and
/// an explicitly closeable native store pool. Unmanaged calls remain ephemeral.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PostgresStoreRequest {
    dsn: String,
    schema: String,
    #[serde(default)]
    context_id: Option<String>,
    #[serde(default)]
    transaction_id: Option<String>,
    operation: PostgresStoreOperation,
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum PostgresStoreOperation {
    CloseStore,
    BeginTransaction,
    CommitTransaction,
    RollbackTransaction,
    LockEntityIdentity {
        namespace: String,
        entity_kind: String,
        entity_id: String,
    },
    EnsureSchema,
    GraphAuthorityCapability,
    NextGlobalSeq,
    CurrentGlobalSeq,
    NextUserSeq {
        user_id: String,
    },
    CurrentUserSeq {
        user_id: String,
    },
    SetUserSeq {
        user_id: String,
        value: i64,
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
    CreateGraphSchema {
        embedding_dim: usize,
        #[serde(default)]
        nodes_table: Option<String>,
        #[serde(default)]
        edges_table: Option<String>,
        #[serde(default)]
        documents_table: Option<String>,
        #[serde(default)]
        domains_table: Option<String>,
        #[serde(default)]
        node_index_table: Option<String>,
    },
    NodeIndexUpsert {
        #[serde(default = "default_node_index_table")]
        table: String,
        records: Vec<GraphMutationRecord>,
    },
    NodeIndexGet {
        #[serde(default = "default_node_index_table")]
        table: String,
        ids: Vec<String>,
    },
    NodeIndexQuery {
        #[serde(default = "default_node_index_table")]
        table: String,
        embedding: Vec<f32>,
        #[serde(default = "default_list_limit")]
        limit: usize,
        #[serde(default)]
        metadata: BTreeMap<String, Value>,
        #[serde(default)]
        metric: SnapshotDistanceMetric,
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
    GraphMutation {
        namespace: String,
        #[serde(default)]
        workspace_id: Option<String>,
        #[serde(default)]
        graph_space: Option<String>,
        table: String,
        entity_kind: String,
        event_id: String,
        op: String,
        record: GraphMutationRecord,
        payload: Value,
        embedding_dim: usize,
        #[serde(default)]
        model_envelope: Option<kogwistar_contracts::GraphModelEnvelope>,
    },
    GraphMetadataPatchMutation {
        namespace: String,
        #[serde(default)]
        workspace_id: Option<String>,
        #[serde(default)]
        graph_space: Option<String>,
        table: String,
        entity_kind: String,
        event_id: String,
        op: String,
        entity_id: String,
        #[serde(default)]
        document: Option<String>,
        #[serde(default)]
        metadata_patch: Map<String, Value>,
        payload: Value,
    },
    GraphDeleteMutation {
        namespace: String,
        #[serde(default)]
        workspace_id: Option<String>,
        #[serde(default)]
        graph_space: Option<String>,
        table: String,
        entity_kind: String,
        event_id: String,
        entity_id: String,
        payload: Value,
    },
    UpsertGraphProjection {
        namespace: String,
        #[serde(default)]
        workspace_id: Option<String>,
        #[serde(default)]
        graph_space: Option<String>,
        table: String,
        record: GraphMutationRecord,
        embedding_dim: usize,
    },
    PatchGraphProjectionMetadata {
        namespace: String,
        #[serde(default)]
        workspace_id: Option<String>,
        #[serde(default)]
        graph_space: Option<String>,
        table: String,
        entity_id: String,
        #[serde(default)]
        document: Option<String>,
        #[serde(default)]
        metadata_patch: Map<String, Value>,
        #[serde(default = "default_true")]
        patch_document_metadata: bool,
    },
    GraphProjectionRecords {
        namespace: String,
        #[serde(default)]
        workspace_id: Option<String>,
        #[serde(default)]
        graph_space: Option<String>,
        table: String,
        #[serde(default)]
        ids: Option<Vec<String>>,
        #[serde(default)]
        metadata: BTreeMap<String, Value>,
        #[serde(default)]
        after_id: Option<String>,
        #[serde(default = "default_list_limit")]
        limit: usize,
        #[serde(default)]
        for_update: bool,
    },
    GraphProjectionVectorQuery {
        namespace: String,
        #[serde(default)]
        workspace_id: Option<String>,
        #[serde(default)]
        graph_space: Option<String>,
        table: String,
        embedding: Vec<f32>,
        #[serde(default = "default_list_limit")]
        limit: usize,
        #[serde(default)]
        metadata: BTreeMap<String, Value>,
        #[serde(default)]
        metadata_not_equals: BTreeMap<String, Value>,
        #[serde(default)]
        metric: SnapshotDistanceMetric,
        embedding_dim: usize,
    },
    GraphProjectionVectorQueryBatch {
        namespace: String,
        #[serde(default)]
        workspace_id: Option<String>,
        #[serde(default)]
        graph_space: Option<String>,
        table: String,
        embeddings: Vec<Vec<f32>>,
        #[serde(default = "default_list_limit")]
        limit: usize,
        #[serde(default)]
        metadata: BTreeMap<String, Value>,
        #[serde(default)]
        metadata_not_equals: BTreeMap<String, Value>,
        #[serde(default)]
        metric: SnapshotDistanceMetric,
        embedding_dim: usize,
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
    ApplyRecordedRuntimeTransition {
        transition: Box<RecordedRuntimeTransition>,
        #[serde(default)]
        abort_after_writes: bool,
    },
    ApplyClaimedRecordedRuntimeTransition {
        handoff: RecordedWorkerHandoff,
        transition: Box<RecordedRuntimeTransition>,
        #[serde(default)]
        abort_after_writes: bool,
    },
    ReadRecordedRuntimeState {
        run_id: String,
        workflow_id: String,
        conversation_id: String,
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
    ReconcileJoinIndexJobs {
        #[serde(default = "default_claim_limit")]
        limit: usize,
        #[serde(default = "default_lease_seconds")]
        lease_seconds: i64,
        #[serde(default = "default_optional_namespace")]
        namespace: Option<String>,
        nodes_table: String,
        edges_table: String,
        node_docs_table: String,
        node_refs_table: String,
        edge_refs_table: String,
        edge_endpoints_table: String,
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
        operations: Vec<PostgresStoreOperation>,
        #[serde(default)]
        abort: bool,
    },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GraphMutationRecord {
    id: String,
    #[serde(default)]
    document: Option<String>,
    #[serde(default)]
    metadata: Map<String, Value>,
    #[serde(default)]
    embedding: Option<Vec<f32>>,
}

fn postgres_raw_event_json(event: PostgresRawEntityEvent) -> Value {
    json!({
        "namespace": event.namespace,
        "seq": event.seq,
        "event_id": event.event_id,
        "entity_kind": event.entity_kind,
        "entity_id": event.entity_id,
        "op": event.op,
        "payload_json": event.payload_json,
        "created_at": event.created_at,
    })
}

fn postgres_new_raw_event(
    event_id: String,
    entity_kind: String,
    entity_id: String,
    op: String,
    payload_json: String,
) -> PostgresNewRawEntityEvent {
    PostgresNewRawEntityEvent {
        event_id,
        entity_kind,
        entity_id,
        op,
        payload_json,
    }
}

fn postgres_appended_raw_event_json(appended: kogwistar_store_postgres::AppendedRawEvent) -> Value {
    let seq = appended.event.seq;
    json!({"seq": seq, "inserted": appended.inserted, "event": postgres_raw_event_json(appended.event)})
}

fn graph_scope(
    namespace: String,
    workspace_id: Option<String>,
    graph_space: Option<String>,
) -> kogwistar_store::GraphScope {
    kogwistar_store::GraphScope {
        namespace,
        workspace_id,
        graph_space,
    }
}

fn graph_mutation_record(record: GraphMutationRecord) -> GraphRecord {
    GraphRecord {
        id: record.id,
        document: record.document,
        metadata: record.metadata,
        embedding: record.embedding,
    }
}

fn graph_mutation_json(applied: kogwistar_store::AppliedGraphMutation) -> Value {
    json!({
        "event": event_json(applied.event),
        "inserted": applied.inserted,
        "mutated": applied.mutated,
    })
}

fn optional_graph_mutation_json(applied: Option<kogwistar_store::AppliedGraphMutation>) -> Value {
    applied.map(graph_mutation_json).unwrap_or(Value::Null)
}

fn graph_table_names(
    nodes_table: Option<String>,
    edges_table: Option<String>,
    documents_table: Option<String>,
    domains_table: Option<String>,
) -> kogwistar_store_postgres::GraphTableNames {
    let defaults = kogwistar_store_postgres::GraphTableNames::default();
    kogwistar_store_postgres::GraphTableNames {
        nodes: nodes_table.unwrap_or(defaults.nodes),
        edges: edges_table.unwrap_or(defaults.edges),
        documents: documents_table.unwrap_or(defaults.documents),
        domains: domains_table.unwrap_or(defaults.domains),
    }
}
