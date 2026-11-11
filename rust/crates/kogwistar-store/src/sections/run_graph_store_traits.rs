// Run-registry and graph read/write store traits. Python parity: kogwistar.server.run_registry/runtime_api and engine_core graph backend protocols.
/// Bounded run-registry capability. It intentionally has no transition/CAS
/// policy: Python backends accept every status value and missing updates noop.
pub trait ServerRunReadStore: Send + Sync {
    fn server_run(
        &self,
        run_id: &str,
    ) -> impl Future<Output = StoreResult<Option<ServerRun>>> + Send;
    fn server_runs(
        &self,
        status: Option<&str>,
        workflow_id: Option<&str>,
        conversation_id: Option<&str>,
        limit: usize,
    ) -> impl Future<Output = StoreResult<Vec<ServerRun>>> + Send;
    fn server_run_events(
        &self,
        run_id: &str,
        after_seq: i64,
        limit: usize,
    ) -> impl Future<Output = StoreResult<Vec<ServerRunEvent>>> + Send;
}

pub trait ServerRunWriteStore: ServerRunReadStore {
    fn create_server_run(
        &self,
        run: ServerRunCreate,
    ) -> impl Future<Output = StoreResult<()>> + Send;
    fn append_server_run_event(
        &self,
        run_id: &str,
        event_type: &str,
        payload_json: String,
    ) -> impl Future<Output = StoreResult<ServerRunEvent>> + Send;
    fn update_server_run(
        &self,
        run_id: &str,
        update: ServerRunUpdate,
    ) -> impl Future<Output = StoreResult<()>> + Send;
    fn request_server_run_cancel(
        &self,
        run_id: &str,
    ) -> impl Future<Output = StoreResult<()>> + Send;
}

pub trait GraphReadStore: Send + Sync {
    fn graph_record(
        &self,
        namespace: &str,
        id: &str,
    ) -> impl Future<Output = StoreResult<Option<GraphRecord>>> + Send;
    /// Read by both namespace and entity kind. The default fails closed so
    /// stores with an ambiguous legacy identity model cannot claim parity.
    fn graph_record_of_kind(
        &self,
        _namespace: &str,
        _kind: GraphEntityKind,
        _id: &str,
    ) -> impl Future<Output = StoreResult<Option<GraphRecord>>> + Send {
        async {
            Err(StoreError::Backend {
                backend: "graph-read".to_owned(),
                message: "kind-aware graph record reads are not supported by this store".to_owned(),
            })
        }
    }
    /// Namespace records in implementation-defined stable read order.
    ///
    /// The Phase-2 in-memory implementation preserves first insertion order:
    /// upserting an existing id keeps its position, while delete then reinsert
    /// appends it. Consumers must not infer bytewise-id ordering from this API.
    fn graph_records(
        &self,
        namespace: &str,
        metadata: &MetadataFilter,
    ) -> impl Future<Output = StoreResult<Vec<GraphRecord>>> + Send;
    /// Read a graph entity kind without mixing nodes and edges.
    fn graph_records_of_kind(
        &self,
        _namespace: &str,
        _kind: GraphEntityKind,
        _metadata: &MetadataFilter,
    ) -> impl Future<Output = StoreResult<Vec<GraphRecord>>> + Send {
        async {
            Err(StoreError::Backend {
                backend: "graph-read".to_owned(),
                message: "kind-aware graph record listing is not supported by this store"
                    .to_owned(),
            })
        }
    }
    /// Read a bounded set of records by typed identity. Missing IDs are
    /// omitted, duplicates are returned once, and result order follows the
    /// first occurrence in `ids`. The default preserves compatibility for
    /// stores that only implement single-record reads; backends can override
    /// it with one native batch request.
    fn graph_records_by_ids_of_kind(
        &self,
        namespace: &str,
        kind: GraphEntityKind,
        ids: &[String],
    ) -> impl Future<Output = StoreResult<Vec<GraphRecord>>> + Send {
        let namespace = namespace.to_owned();
        let ids = ids.to_vec();
        async move {
            if ids.len() > 10_000 {
                return Err(StoreError::Backend {
                    backend: "graph-read".to_owned(),
                    message: "typed ID batch exceeds the 10000-record limit".to_owned(),
                });
            }
            let mut seen = BTreeSet::new();
            let mut records = Vec::with_capacity(ids.len());
            for id in ids {
                if id.trim().is_empty() {
                    return Err(StoreError::InvalidIdentifier {
                        identifier: "graph record id",
                    });
                }
                if !seen.insert(id.clone()) {
                    continue;
                }
                if let Some(record) = self.graph_record_of_kind(&namespace, kind, &id).await? {
                    records.push(record);
                }
            }
            Ok(records)
        }
    }
    /// Metadata matches in implementation-defined stable rank order.
    ///
    /// The Phase-2 in-memory implementation's cosine mode follows the public
    /// Python memory backend: ascending distance with insertion-order ties;
    /// absent, dimension-mismatched, and zero-norm candidate vectors rank at
    /// distance `2.0`. L2 retains its strict vector contract.
    fn vector_query(
        &self,
        namespace: &str,
        query: &VectorQuery,
    ) -> impl Future<Output = StoreResult<Vec<VectorMatch>>> + Send;
    /// Run vector search within one graph entity kind.
    fn vector_query_of_kind(
        &self,
        _namespace: &str,
        _kind: GraphEntityKind,
        _query: &VectorQuery,
    ) -> impl Future<Output = StoreResult<Vec<VectorMatch>>> + Send {
        async {
            Err(StoreError::Backend {
                backend: "graph-read".to_owned(),
                message: "kind-aware vector search is not supported by this store".to_owned(),
            })
        }
    }
    /// Run a typed vector query with an inclusive numeric metadata upper bound.
    /// The default fails closed rather than applying the bound after top-N,
    /// which could silently change the candidate set.
    fn vector_query_of_kind_numeric_lte(
        &self,
        _namespace: &str,
        _kind: GraphEntityKind,
        _query: &VectorQuery,
        _field: &str,
        _maximum: f64,
    ) -> impl Future<Output = StoreResult<Vec<VectorMatch>>> + Send {
        async {
            Err(StoreError::Backend {
                backend: "graph-read".to_owned(),
                message: "numeric-bounded vector search is not supported by this store".to_owned(),
            })
        }
    }
}

/// An independently named vector collection, separate from canonical graph
/// entities. Derived indexes such as Python's `node_index` must not be stored
/// as fake Nodes or Edges: graph reads, ACL traversal, and lifecycle writes
/// treat those entity kinds as authoritative graph data.
pub trait VectorCollectionStore: Send + Sync {
    /// Upsert one bounded batch within one namespace and collection. The
    /// backend's transaction/eventual-consistency guarantees remain explicit;
    /// this method does not imply graph-mutation atomicity.
    fn upsert_vector_collection(
        &self,
        namespace: &str,
        collection: &str,
        records: Vec<GraphRecord>,
    ) -> impl Future<Output = StoreResult<()>> + Send;

    /// Delete the requested IDs from only the selected namespace/collection.
    fn delete_vector_collection_records(
        &self,
        namespace: &str,
        collection: &str,
        ids: Vec<String>,
    ) -> impl Future<Output = StoreResult<()>> + Send;

    /// Query only the selected namespace/collection using the shared distance
    /// contract and stable backend ordering.
    fn query_vector_collection(
        &self,
        namespace: &str,
        collection: &str,
        query: &VectorQuery,
    ) -> impl Future<Output = StoreResult<Vec<VectorMatch>>> + Send;
}

/// Backend-native access to Python's dedicated `node_index` projection.
///
/// This is intentionally separate from `VectorCollectionStore`: PostgreSQL
/// stores this projection in the graph-schema-local `gke_node_index` table,
/// while Chroma and memory use their dedicated index collections. `index`
/// names that backend-native target; it is not a graph entity kind and writes
/// through this contract do not imply canonical graph mutation atomicity.
pub trait NodeIndexStore: Send + Sync {
    fn upsert_node_index(
        &self,
        index: &str,
        records: Vec<GraphRecord>,
    ) -> impl Future<Output = StoreResult<()>> + Send;

    fn get_node_index(
        &self,
        index: &str,
        ids: &[String],
    ) -> impl Future<Output = StoreResult<Vec<GraphRecord>>> + Send;

    fn query_node_index(
        &self,
        index: &str,
        query: &VectorQuery,
    ) -> impl Future<Output = StoreResult<Vec<VectorMatch>>> + Send;
}

pub trait GraphWriteStore: GraphReadStore {
    fn upsert_graph_record(
        &self,
        namespace: &str,
        record: GraphRecord,
    ) -> impl Future<Output = StoreResult<()>> + Send;
    /// Write a graph record with an explicit entity kind.
    fn upsert_graph_record_of_kind(
        &self,
        namespace: &str,
        _kind: GraphEntityKind,
        record: GraphRecord,
    ) -> impl Future<Output = StoreResult<()>> + Send {
        async move { self.upsert_graph_record(namespace, record).await }
    }
    fn delete_graph_record(
        &self,
        namespace: &str,
        id: &str,
    ) -> impl Future<Output = StoreResult<bool>> + Send;
    /// Delete by namespace, entity kind, and ID to avoid cross-kind deletion.
    fn delete_graph_record_of_kind(
        &self,
        namespace: &str,
        _kind: GraphEntityKind,
        id: &str,
    ) -> impl Future<Output = StoreResult<bool>> + Send {
        async move { self.delete_graph_record(namespace, id).await }
    }
}

/// Minimal materialized-projection mutation seam for durable index workers.
///
/// The worker does not query graph state. Requiring `GraphWriteStore` also
/// requires unrelated graph reads and prevents native projection stores from
/// participating without pretending to implement those queries.
pub trait GraphProjectionWriteStore: Send + Sync {
    fn upsert_graph_projection_record(
        &self,
        namespace: &str,
        entity_kind: &str,
        record: GraphRecord,
    ) -> impl Future<Output = StoreResult<()>> + Send;

    fn delete_graph_projection_record(
        &self,
        namespace: &str,
        entity_kind: &str,
        id: &str,
    ) -> impl Future<Output = StoreResult<bool>> + Send;

    /// Report whether an accepted projection action changed the materialized
    /// store or was safely superseded by a newer canonical revision. Existing
    /// implementations remain source-compatible and default to `Applied`.
    fn upsert_graph_projection_record_with_outcome(
        &self,
        namespace: &str,
        entity_kind: &str,
        record: GraphRecord,
    ) -> impl Future<Output = StoreResult<ProjectionWriteOutcome>> + Send {
        async move {
            self.upsert_graph_projection_record(namespace, entity_kind, record)
                .await?;
            Ok(ProjectionWriteOutcome::Applied)
        }
    }

    fn delete_graph_projection_record_with_outcome(
        &self,
        namespace: &str,
        entity_kind: &str,
        id: &str,
    ) -> impl Future<Output = StoreResult<ProjectionWriteOutcome>> + Send {
        async move {
            self.delete_graph_projection_record(namespace, entity_kind, id)
                .await?;
            Ok(ProjectionWriteOutcome::Applied)
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProjectionWriteOutcome {
    Applied,
    Superseded,
}

impl<T: GraphWriteStore> GraphProjectionWriteStore for T {
    // Preserve the trait's explicit opaque-future + Send contract here.
    #[allow(clippy::manual_async_fn)]
    fn upsert_graph_projection_record(
        &self,
        namespace: &str,
        entity_kind: &str,
        record: GraphRecord,
    ) -> impl Future<Output = StoreResult<()>> + Send {
        async move {
            let kind = match entity_kind {
                "node" => GraphEntityKind::Node,
                "edge" => GraphEntityKind::Edge,
                _ => {
                    return Err(StoreError::Backend {
                        backend: "graph-projection".to_owned(),
                        message: "graph projection entity kind must be 'node' or 'edge'".to_owned(),
                    });
                }
            };
            self.upsert_graph_record_of_kind(namespace, kind, record)
                .await
        }
    }

    #[allow(clippy::manual_async_fn)]
    fn delete_graph_projection_record(
        &self,
        namespace: &str,
        entity_kind: &str,
        id: &str,
    ) -> impl Future<Output = StoreResult<bool>> + Send {
        async move {
            let kind = match entity_kind {
                "node" => GraphEntityKind::Node,
                "edge" => GraphEntityKind::Edge,
                _ => {
                    return Err(StoreError::Backend {
                        backend: "graph-projection".to_owned(),
                        message: "graph projection entity kind must be 'node' or 'edge'".to_owned(),
                    });
                }
            };
            self.delete_graph_record_of_kind(namespace, kind, id).await
        }
    }
}
