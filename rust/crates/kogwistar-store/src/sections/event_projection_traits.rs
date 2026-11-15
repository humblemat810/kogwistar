// Event, named-projection, workflow-design-history, and shadow-inspection traits. Python parity: engine_core event/projection protocols and runtime history services.
pub trait EventReadStore: Send + Sync {
    /// Events strictly after `after_seq`, ascending namespace-local sequence.
    fn replay_events(
        &self,
        namespace: &str,
        after_seq: i64,
        limit: usize,
    ) -> impl Future<Output = StoreResult<Vec<EntityEvent>>> + Send;
    fn replay_cursor(
        &self,
        namespace: &str,
        consumer: &str,
    ) -> impl Future<Output = StoreResult<ReplayCursor>> + Send;
    fn latest_event_seq(&self, namespace: &str) -> impl Future<Output = StoreResult<i64>> + Send;
}

pub trait EventWriteStore: EventReadStore {
    fn append_entity_event(
        &self,
        namespace: &str,
        event: NewEntityEvent,
    ) -> impl Future<Output = StoreResult<AppendedEvent>> + Send;
    fn advance_replay_cursor(
        &self,
        namespace: &str,
        consumer: &str,
        last_seq: i64,
    ) -> impl Future<Output = StoreResult<ReplayCursor>> + Send;
}

/// Explicit destructive event-history capability. Kept separate from ordinary
/// append/cursor writes so consumers opt in to branch pruning deliberately.
pub trait EventPruneStore: EventReadStore {
    /// Delete only rows whose namespace-local sequence is strictly greater
    /// than `to_seq`, returning the exact affected-row count.
    fn prune_entity_events_after(
        &self,
        namespace: &str,
        to_seq: i64,
    ) -> impl Future<Output = StoreResult<u64>> + Send;
}

pub trait ProjectionReadStore: Send + Sync {
    fn named_projection(
        &self,
        namespace: &str,
        key: &str,
    ) -> impl Future<Output = StoreResult<Option<NamedProjection>>> + Send;
    /// Namespace rows, ascending bytewise key, matching Python meta-store list.
    fn named_projections(
        &self,
        namespace: &str,
    ) -> impl Future<Output = StoreResult<Vec<NamedProjection>>> + Send;
}

/// Mutation capability for generic named projections.  Read consumers keep
/// depending on [`ProjectionReadStore`], so shadow inspection stays read-only.
pub trait ProjectionWriteStore: ProjectionReadStore {
    fn replace_named_projection(
        &self,
        namespace: &str,
        key: &str,
        projection: NamedProjectionWrite,
    ) -> impl Future<Output = StoreResult<()>> + Send;

    /// Creates only when both expected sequences are `None`; otherwise updates
    /// only when both persisted sequence values match exactly.
    fn compare_and_swap_named_projection(
        &self,
        namespace: &str,
        key: &str,
        expected_last_authoritative_seq: Option<i64>,
        expected_last_materialized_seq: Option<i64>,
        projection: NamedProjectionWrite,
    ) -> impl Future<Output = StoreResult<bool>> + Send;

    fn clear_named_projection(
        &self,
        namespace: &str,
        key: &str,
    ) -> impl Future<Output = StoreResult<()>> + Send;

    fn clear_projection_namespace(
        &self,
        namespace: &str,
    ) -> impl Future<Output = StoreResult<()>> + Send;
}

/// Read capability for workflow-design checkpoints and reversible deltas.
pub trait WorkflowDesignHistoryReadStore: Send + Sync {
    fn workflow_design_snapshot(
        &self,
        workflow_id: &str,
        max_version: i64,
        schema_version: i64,
    ) -> impl Future<Output = StoreResult<Option<WorkflowDesignSnapshot>>> + Send;

    fn workflow_design_delta(
        &self,
        workflow_id: &str,
        version: i64,
        schema_version: i64,
    ) -> impl Future<Output = StoreResult<Option<WorkflowDesignDelta>>> + Send;
}

/// Mutation capability for workflow-design history. The concrete durable
/// stores also expose these operations on their UoWs for atomic branches.
pub trait WorkflowDesignHistoryWriteStore: WorkflowDesignHistoryReadStore {
    fn put_workflow_design_snapshot(
        &self,
        workflow_id: &str,
        snapshot: WorkflowDesignSnapshotWrite,
    ) -> impl Future<Output = StoreResult<()>> + Send;

    fn clear_workflow_design_snapshots(
        &self,
        workflow_id: &str,
    ) -> impl Future<Output = StoreResult<()>> + Send;

    fn put_workflow_design_delta(
        &self,
        workflow_id: &str,
        delta: WorkflowDesignDeltaWrite,
    ) -> impl Future<Output = StoreResult<()>> + Send;

    fn clear_workflow_design_deltas(
        &self,
        workflow_id: &str,
    ) -> impl Future<Output = StoreResult<()>> + Send;
}

/// Capability-scoped shadow façade. No mutation trait is implemented.
#[derive(Clone, Debug)]
pub struct ShadowInspection<S> {
    inner: S,
}
impl<S> ShadowInspection<S> {
    fn new(inner: S) -> Self {
        Self { inner }
    }
}

pub trait ShadowInspectable:
    Clone + GraphReadStore + EventReadStore + ProjectionReadStore + Sized
{
    fn shadow_inspection(&self) -> ShadowInspection<Self> {
        ShadowInspection::new(self.clone())
    }
}
impl<T> ShadowInspectable for T where
    T: Clone + GraphReadStore + EventReadStore + ProjectionReadStore
{
}

impl<S: GraphReadStore + Send + Sync> GraphReadStore for ShadowInspection<S> {
    async fn graph_record(&self, ns: &str, id: &str) -> StoreResult<Option<GraphRecord>> {
        self.inner.graph_record(ns, id).await
    }
    async fn graph_records(
        &self,
        ns: &str,
        filter: &MetadataFilter,
    ) -> StoreResult<Vec<GraphRecord>> {
        self.inner.graph_records(ns, filter).await
    }
    async fn vector_query(&self, ns: &str, query: &VectorQuery) -> StoreResult<Vec<VectorMatch>> {
        self.inner.vector_query(ns, query).await
    }
    async fn vector_query_of_kind_numeric_lte(
        &self,
        ns: &str,
        kind: GraphEntityKind,
        query: &VectorQuery,
        field: &str,
        maximum: f64,
    ) -> StoreResult<Vec<VectorMatch>> {
        self.inner
            .vector_query_of_kind_numeric_lte(ns, kind, query, field, maximum)
            .await
    }
}

impl<S: EventReadStore + Send + Sync> EventReadStore for ShadowInspection<S> {
    async fn replay_events(
        &self,
        ns: &str,
        after: i64,
        limit: usize,
    ) -> StoreResult<Vec<EntityEvent>> {
        self.inner.replay_events(ns, after, limit).await
    }
    async fn replay_cursor(&self, ns: &str, consumer: &str) -> StoreResult<ReplayCursor> {
        self.inner.replay_cursor(ns, consumer).await
    }
    async fn latest_event_seq(&self, ns: &str) -> StoreResult<i64> {
        self.inner.latest_event_seq(ns).await
    }
}

impl<S: ProjectionReadStore + Send + Sync> ProjectionReadStore for ShadowInspection<S> {
    async fn named_projection(
        &self,
        namespace: &str,
        key: &str,
    ) -> StoreResult<Option<NamedProjection>> {
        self.inner.named_projection(namespace, key).await
    }
    async fn named_projections(&self, namespace: &str) -> StoreResult<Vec<NamedProjection>> {
        self.inner.named_projections(namespace).await
    }
}
