// Graph mutation validation and authority contracts. Python parity: kogwistar.engine_core lifecycle/write/persist subsystems.
fn is_tombstoned(metadata: &Map<String, Value>) -> bool {
    metadata
        .get("lifecycle_status")
        .and_then(Value::as_str)
        .is_some_and(|status| matches!(status, "tombstoned" | "deleted"))
}

/// Storage scope carried by the graph projection. Namespace, workspace, and
/// graph-space are storage coordinates, not graph roles or entity subtypes.
/// Python's `kg_graph_type` identifies the graph role and supplies the default
/// ACL `truth_graph`; it is deliberately independent of this storage scope.
/// The Python pgvector schema keeps graph rows in shared tables, so these scope
/// values are represented by exact JSON metadata predicates rather than by a
/// Rust-owned table or schema. ACL truth-graph selection is policy context and
/// is carried separately by the ACL-aware mutation contract.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct GraphScope {
    pub namespace: String,
    pub workspace_id: Option<String>,
    pub graph_space: Option<String>,
}

impl GraphScope {
    /// Reject metadata that claims a storage-scope dimension absent from this
    /// request, so backends cannot accidentally widen a scoped record.
    pub fn validate_metadata_scope(&self, metadata: &Map<String, Value>) -> StoreResult<()> {
        for (key, expected) in [
            ("namespace", Some(self.namespace.as_str())),
            ("workspace_id", self.workspace_id.as_deref()),
            ("graph_space", self.graph_space.as_deref()),
        ] {
            match expected {
                Some(expected)
                    if metadata
                        .get(key)
                        .is_some_and(|actual| actual.as_str() != Some(expected)) =>
                {
                    return Err(StoreError::InvalidEntityEventPayload {
                        message: format!(
                            "graph metadata scope {key:?} conflicts with request scope"
                        ),
                    });
                }
                None if metadata.get(key).is_some_and(|actual| !actual.is_null()) => {
                    return Err(StoreError::InvalidEntityEventPayload {
                        message: format!(
                            "graph metadata scope {key:?} is absent from request scope"
                        ),
                    });
                }
                _ => {}
            }
        }
        Ok(())
    }
}

/// One authoritative graph change plus its materialized vector projection.
/// `table` names an existing Python pgvector collection (normally
/// `gke_nodes` or `gke_edges`); implementations must not create a parallel
/// graph schema for this capability.
#[derive(Clone, Debug, PartialEq)]
pub struct GraphMutation {
    pub scope: GraphScope,
    pub table: String,
    pub entity_kind: String,
    pub event_id: String,
    pub op: String,
    pub payload: Value,
    pub record: GraphRecord,
    pub embedding_dim: usize,
}

/// Return the canonical projection table for a persisted graph entity kind.
///
/// `document` and `domain` share the graph event/projection transaction path,
/// but are not Node/Edge lifecycle entities.
pub fn graph_projection_table(entity_kind: &str) -> StoreResult<&'static str> {
    match entity_kind {
        "node" => Ok("gke_nodes"),
        "edge" => Ok("gke_edges"),
        "document" => Ok("gke_documents"),
        "domain" => Ok("gke_domains"),
        _ => Err(StoreError::InvalidEntityEventPayload {
            message: format!("unsupported graph entity kind {entity_kind:?}"),
        }),
    }
}

/// Validate a projection target using the shared entity-kind/table mapping.
pub fn validate_graph_projection_target(table: &str, entity_kind: &str) -> StoreResult<()> {
    let expected_table = graph_projection_table(entity_kind)?;
    if table != expected_table {
        return Err(StoreError::InvalidEntityEventPayload {
            message: format!("graph entity kind {entity_kind:?} must use table {expected_table:?}"),
        });
    }
    Ok(())
}

/// Validate a node/edge-only lifecycle target, including its table mapping.
pub fn validate_graph_lifecycle_target(table: &str, entity_kind: &str) -> StoreResult<()> {
    if !matches!(entity_kind, "node" | "edge") {
        return Err(StoreError::InvalidEntityEventPayload {
            message: format!("graph lifecycle operations do not support {entity_kind:?}"),
        });
    }
    validate_graph_projection_target(table, entity_kind)
}

/// A graph mutation batch plus optional indexing intents that must share its
/// commit boundary.  Indexing remains explicit: a store must never infer a
/// queue operation from arbitrary graph metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct GraphMutationBatch {
    pub mutations: Vec<GraphMutation>,
    pub index_jobs: Vec<NewIndexJob>,
}

/// Validate the internal mutation before a store can own its effects.
///
/// The versioned wire DTO performs the same checks at the API boundary, but
/// Rust-native callers can reach a store without going through that adapter.
/// Keeping this guard at the effect owner prevents a direct caller from
/// bypassing the shared graph-mutation contract.
pub fn validate_graph_mutation(mutation: &GraphMutation) -> StoreResult<()> {
    if mutation.scope.namespace.is_empty() {
        return Err(StoreError::EmptyNamespace);
    }
    if mutation.event_id.is_empty() {
        return Err(StoreError::InvalidIdentifier {
            identifier: "graph mutation event_id",
        });
    }
    if mutation.record.id.is_empty() {
        return Err(StoreError::EmptyRecordId);
    }
    // Documents and domains share this internal transaction path without
    // being admitted to the public Node/Edge graph mutation authority.
    validate_graph_projection_target(&mutation.table, &mutation.entity_kind)?;
    if !matches!(mutation.op.as_str(), "ADD" | "REPLACE" | "TOMBSTONE") {
        return Err(StoreError::UnsupportedEntityEventOperation {
            op: mutation.op.clone(),
        });
    }
    if !mutation.payload.is_object() {
        return Err(StoreError::InvalidEntityEventPayload {
            message: "graph mutation payload must be a JSON object".to_owned(),
        });
    }
    mutation
        .scope
        .validate_metadata_scope(&mutation.record.metadata)?;
    if let Some(metadata) = mutation.payload.get("metadata") {
        let metadata =
            metadata
                .as_object()
                .ok_or_else(|| StoreError::InvalidEntityEventPayload {
                    message: "graph mutation payload metadata must be an object".to_owned(),
                })?;
        mutation.scope.validate_metadata_scope(metadata)?;
    }
    if matches!(mutation.entity_kind.as_str(), "node" | "edge")
        && mutation.payload.get("id").and_then(Value::as_str) != Some(mutation.record.id.as_str())
    {
        return Err(StoreError::InvalidEntityEventPayload {
            message: "graph mutation payload id must match the projection record id".to_owned(),
        });
    }
    if let Some(embedding) = &mutation.record.embedding {
        if embedding.iter().any(|value| !value.is_finite()) {
            return Err(StoreError::NonFiniteVector);
        }
        if embedding.len() != mutation.embedding_dim {
            return Err(StoreError::VectorDimensionMismatch {
                expected: mutation.embedding_dim,
                actual: embedding.len(),
            });
        }
    }
    Ok(())
}

/// Ensure an event-ID retry is the same graph effect before treating it as
/// idempotent. Raw event stores intentionally return the first event for a
/// repeated ID; graph mutation owners must apply this stricter comparison.
pub fn validate_graph_mutation_replay(
    mutation: &GraphMutation,
    existing: &EntityEvent,
) -> StoreResult<()> {
    if existing.namespace != mutation.scope.namespace
        || existing.event_id != mutation.event_id
        || existing.entity_kind != mutation.entity_kind
        || existing.entity_id != mutation.record.id
        || existing.op != mutation.op
        || existing.payload != mutation.payload
    {
        return Err(StoreError::InvalidEntityEventPayload {
            message: format!(
                "graph mutation event id {:?} was reused with different input",
                mutation.event_id
            ),
        });
    }
    Ok(())
}

/// Validate a store-owned atomic batch before opening its transaction.
pub fn validate_graph_mutation_batch(batch: &GraphMutationBatch) -> StoreResult<()> {
    let first = batch
        .mutations
        .first()
        .ok_or(StoreError::InvalidEntityEventPayload {
            message: "graph mutation batches must not be empty".to_owned(),
        })?;
    for mutation in &batch.mutations {
        validate_graph_mutation(mutation)?;
        if mutation.scope != first.scope {
            return Err(StoreError::InvalidEntityEventPayload {
                message: "all atomic graph mutations must share one graph scope".to_owned(),
            });
        }
    }
    for job in &batch.index_jobs {
        if job.job_id.is_empty()
            || job.namespace.is_empty()
            || job.entity_kind.is_empty()
            || job.entity_id.is_empty()
            || job.index_kind.is_empty()
        {
            return Err(StoreError::InvalidIdentifier {
                identifier: "index job field",
            });
        }
        if job.namespace != first.scope.namespace {
            return Err(StoreError::InvalidEntityEventPayload {
                message: "index job namespace must match graph mutation namespace".to_owned(),
            });
        }
        if !matches!(job.op.as_str(), "UPSERT" | "DELETE") {
            return Err(StoreError::InvalidEntityEventPayload {
                message: "index job op must be UPSERT or DELETE".to_owned(),
            });
        }
        if !(0..=i32::MAX as i64).contains(&job.max_retries) {
            return Err(StoreError::InvalidEntityEventPayload {
                message: "index job max_retries must fit a non-negative PostgreSQL INTEGER"
                    .to_owned(),
            });
        }
    }
    Ok(())
}

/// Validate the optional typed model that accompanies a graph mutation.
///
/// The legacy mutation payload remains unchanged.  This additive seam lets a
/// Rust-native caller opt into the same model contract as the Python writer
/// without forcing old event/replay callers to construct an envelope.
pub fn validate_typed_graph_mutation(
    mutation: &GraphMutation,
    envelope: &GraphModelEnvelope,
) -> StoreResult<()> {
    validate_graph_mutation(mutation)?;
    let persistent_entity = match envelope.model_class {
        GraphModelClass::Node | GraphModelClass::Edge => {
            Some(envelope.decode_persistent_v1().map_err(|error| {
                StoreError::InvalidEntityEventPayload {
                    message: error.to_string(),
                }
            })?)
        }
        _ => {
            envelope.validate_family_v1().map_err(|error| {
                StoreError::InvalidEntityEventPayload {
                    message: error.to_string(),
                }
            })?;
            None
        }
    };
    let expected_kind = match mutation.entity_kind.as_str() {
        "node" => GraphEntityKind::Node,
        "edge" => GraphEntityKind::Edge,
        _ => {
            return Err(StoreError::InvalidEntityEventPayload {
                message: format!(
                    "typed graph mutation has unsupported entity kind {:?}",
                    mutation.entity_kind
                ),
            });
        }
    };
    if envelope.model_class.entity_kind() != expected_kind {
        return Err(StoreError::InvalidEntityEventPayload {
            message: format!(
                "typed graph mutation entity kind {:?} does not match model class {:?}",
                expected_kind, envelope.model_class
            ),
        });
    }
    if mutation.payload != envelope.payload {
        return Err(StoreError::InvalidEntityEventPayload {
            message: "typed graph mutation payload differs from its model envelope".to_owned(),
        });
    }
    let entity_id = match persistent_entity {
        Some(entity) => match entity {
            PersistentGraphEntityV1::Node(node) => node.core.id,
            PersistentGraphEntityV1::Edge(edge) => edge.core.id,
        },
        None => {
            match envelope
                .inspect()
                .map_err(|error| StoreError::InvalidEntityEventPayload {
                    message: error.to_string(),
                })? {
                kogwistar_contracts::TypedGraphEntity::Node(node) => node.common.id,
                kogwistar_contracts::TypedGraphEntity::Edge(edge) => edge.common.id,
            }
        }
    };
    if entity_id.as_deref() != Some(mutation.record.id.as_str()) {
        return Err(StoreError::InvalidEntityEventPayload {
            message: "typed graph mutation record id differs from its model envelope".to_owned(),
        });
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq)]
pub struct AppliedGraphMutation {
    pub event: EntityEvent,
    pub inserted: bool,
    /// False for an idempotent event retry.  The pre-existing projection is
    /// deliberately left untouched in that case.
    pub mutated: bool,
}

/// Declared graph-authority level for a store implementation.
///
/// This is intentionally explicit and ordered by authority. A caller must
/// not infer graph support from an unrelated metadata or event API; the
/// default remains unsupported so older backends fail closed.
#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
pub enum GraphAuthorityCapability {
    #[default]
    Unsupported,
    SingleMutation,
    AtomicBatchWithIndexJobs,
}

impl GraphAuthorityCapability {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Unsupported => "unsupported",
            Self::SingleMutation => "single_mutation",
            Self::AtomicBatchWithIndexJobs => "atomic_batch_with_index_jobs",
        }
    }

    pub const fn supports_single_mutation(self) -> bool {
        !matches!(self, Self::Unsupported)
    }

    pub const fn supports_atomic_index_jobs(self) -> bool {
        matches!(self, Self::AtomicBatchWithIndexJobs)
    }
}

/// Read request for the materialized Python pgvector graph tables.  Metadata
/// predicates retain exact JSON equality semantics; scope predicates are
/// applied in addition to these user-visible filters.
#[derive(Clone, Debug, PartialEq)]
pub struct GraphProjectionRead {
    pub scope: GraphScope,
    pub table: String,
    pub ids: Option<Vec<String>>,
    pub metadata: MetadataFilter,
    pub limit: usize,
}

/// Keyset-paginated graph projection read. Existing `GraphProjectionRead`
/// remains unchanged for source compatibility.
#[derive(Clone, Debug, PartialEq)]
pub struct GraphProjectionPageRead {
    pub read: GraphProjectionRead,
    pub after_id: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GraphProjectionVectorQuery {
    pub scope: GraphScope,
    pub table: String,
    pub query: VectorQuery,
    pub embedding_dim: usize,
}

/// Narrow authoritative graph capability.  It intentionally combines event
/// append and projection materialization so implementations with a real UoW
/// can make both durable together.  The existing `GraphWriteStore` remains a
/// lightweight, non-evented Phase-2 capability.
pub trait GraphMutationStore: Send + Sync {
    /// Declare the strongest graph authority this store can actually provide.
    /// The default is unsupported for backward-compatible, fail-closed
    /// adoption by existing implementations.
    fn graph_authority_capability(&self) -> GraphAuthorityCapability {
        GraphAuthorityCapability::Unsupported
    }
    fn apply_graph_mutation(
        &self,
        mutation: GraphMutation,
    ) -> impl Future<Output = StoreResult<AppliedGraphMutation>> + Send;
    /// Apply a group under one authoritative transaction.
    ///
    /// The default is deliberately fail-closed.  Implementations must opt in
    /// only when they can preserve one commit boundary for the whole group;
    /// callers must not silently degrade an atomic request to per-item writes.
    fn apply_graph_mutations_atomic(
        &self,
        _mutations: Vec<GraphMutation>,
    ) -> impl Future<Output = StoreResult<Vec<AppliedGraphMutation>>> + Send {
        async {
            Err(StoreError::Backend {
                backend: "graph-mutation".to_owned(),
                message: "atomic graph mutation batches are not supported by this store".to_owned(),
            })
        }
    }
    /// Apply graph mutations and their explicit index intents atomically.
    /// The default is fail-closed because a queue write outside the graph
    /// transaction would create a false durability guarantee.
    fn apply_graph_mutations_with_index_jobs_atomic(
        &self,
        _batch: GraphMutationBatch,
    ) -> impl Future<Output = StoreResult<Vec<AppliedGraphMutation>>> + Send {
        async {
            Err(StoreError::Backend {
                backend: "graph-mutation".to_owned(),
                message: "atomic graph mutation plus index jobs are not supported by this store"
                    .to_owned(),
            })
        }
    }
    fn apply_typed_graph_mutation(
        &self,
        mutation: GraphMutation,
        envelope: GraphModelEnvelope,
    ) -> impl Future<Output = StoreResult<AppliedGraphMutation>> + Send {
        async move {
            if !self.graph_authority_capability().supports_single_mutation() {
                return Err(StoreError::Backend {
                    backend: "graph-mutation".to_owned(),
                    message: "typed graph mutations require declared graph authority".to_owned(),
                });
            }
            validate_typed_graph_mutation(&mutation, &envelope)?;
            self.apply_graph_mutation(mutation).await
        }
    }
    fn graph_projection_records(
        &self,
        read: GraphProjectionRead,
    ) -> impl Future<Output = StoreResult<Vec<GraphRecord>>> + Send;
    /// Read one keyset page. The default fails closed so older store
    /// implementations remain source-compatible without pretending to page.
    fn graph_projection_page(
        &self,
        _read: GraphProjectionPageRead,
    ) -> impl Future<Output = StoreResult<Vec<GraphRecord>>> + Send {
        async {
            Err(StoreError::Backend {
                backend: "graph-projection-pagination".to_owned(),
                message: "keyset pagination is not supported by this store".to_owned(),
            })
        }
    }
    fn graph_projection_vector_query(
        &self,
        query: GraphProjectionVectorQuery,
    ) -> impl Future<Output = StoreResult<Vec<VectorMatch>>> + Send;
}

/// Visit graph projection pages in stable ID order without retaining all rows.
///
/// The helper rejects backends that return unordered or non-advancing pages,
/// preventing silent omissions and infinite loops at language/backend seams.
pub async fn for_each_graph_projection_page<S, F>(
    store: &S,
    read: GraphProjectionRead,
    mut visit: F,
) -> StoreResult<()>
where
    S: GraphMutationStore + Sync,
    F: FnMut(&[GraphRecord]) -> StoreResult<()>,
{
    let page_size = read.limit;
    if page_size == 0 {
        return Ok(());
    }

    let mut after_id = None;
    loop {
        let page = store
            .graph_projection_page(GraphProjectionPageRead {
                read: read.clone(),
                after_id: after_id.clone(),
            })
            .await?;
        let Some(first) = page.first() else {
            break;
        };
        if after_id.as_ref().is_some_and(|cursor| first.id <= *cursor)
            || page.windows(2).any(|pair| pair[0].id >= pair[1].id)
            || page.len() > page_size
        {
            return Err(StoreError::Backend {
                backend: "graph-projection-pagination".to_owned(),
                message: "backend returned a non-advancing or unordered page".to_owned(),
            });
        }
        let has_more = page.len() == page_size;
        after_id = page.last().map(|record| record.id.clone());
        visit(&page)?;
        if !has_more {
            break;
        }
    }
    Ok(())
}

/// Read every page into one vector. Prefer `for_each_graph_projection_page`
/// when the consumer can reduce records incrementally.
pub async fn graph_projection_records_all<S>(
    store: &S,
    read: GraphProjectionRead,
) -> StoreResult<Vec<GraphRecord>>
where
    S: GraphMutationStore + Sync,
{
    let mut records = Vec::new();
    for_each_graph_projection_page(store, read, |page| {
        records.extend_from_slice(page);
        Ok(())
    })
    .await?;
    Ok(records)
}
