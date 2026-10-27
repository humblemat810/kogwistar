use super::*;

impl std::fmt::Debug for PostgresStore {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PostgresStore")
            .field("schema", &self.tables.schema)
            .finish_non_exhaustive()
    }
}

impl PostgresStore {
    /// Build a store from a parsed `tokio_postgres::Config`.
    pub fn from_config(config: Config, schema: impl AsRef<str>) -> PostgresStoreResult<Self> {
        let tables = Tables::new(schema.as_ref())?;
        let manager = Manager::new(config, NoTls);
        let pool = Pool::builder(manager)
            .build()
            .map_err(|error| PostgresStoreError::Backend(error.to_string()))?;
        Ok(Self { pool, tables })
    }

    /// Build a store from a libpq-style DSN accepted by `tokio-postgres`.
    pub fn from_dsn(dsn: &str, schema: impl AsRef<str>) -> PostgresStoreResult<Self> {
        let config = dsn
            .parse::<Config>()
            .map_err(|error| PostgresStoreError::Backend(error.to_string()))?;
        Self::from_config(config, schema)
    }

    /// Build from an existing pool. This is useful when app wiring owns pool
    /// sizing or connection recycling policy.
    pub fn from_pool(pool: Pool, schema: impl AsRef<str>) -> PostgresStoreResult<Self> {
        Ok(Self {
            pool,
            tables: Tables::new(schema.as_ref())?,
        })
    }

    pub fn schema(&self) -> &str {
        &self.tables.schema
    }

    /// Create the exact Python `EnginePostgresMetaStore` Phase-2b tables.
    pub async fn ensure_schema(&self) -> PostgresStoreResult<()> {
        let mut pooled_client = self.client().await?;
        let client: &mut tokio_postgres::Client = &mut pooled_client;
        let transaction = client.transaction().await.map_err(backend)?;
        transaction
            .query_one(
                "SELECT pg_advisory_xact_lock($1)",
                &[&POSTGRES_BOOTSTRAP_ADVISORY_LOCK_KEY],
            )
            .await
            .map_err(backend)?;
        transaction
            .batch_execute(&schema_sql(&self.tables))
            .await
            .map_err(backend)?;
        transaction.commit().await.map_err(backend)
    }

    /// Explicit operator action: create only Python-compatible pgvector graph
    /// collections.  Ordinary `ensure_schema` deliberately never enables an
    /// extension or creates graph projection tables.
    pub async fn create_graph_schema(
        &self,
        embedding_dim: usize,
        names: GraphTableNames,
    ) -> PostgresStoreResult<()> {
        let client = self.client().await?;
        create_graph_schema(&**client, &self.tables, embedding_dim, names).await
    }

    /// Run append/cursor work in one RAII PostgreSQL transaction.
    ///
    /// The pooled client stays local to this method. `PostgresUnitOfWork`
    /// owns its `Transaction<'_>` borrow, so the client cannot escape and a
    /// cancelled future, panic, or dropped UoW lets `Transaction::drop`
    /// perform PostgreSQL's rollback protection. Returning an error also
    /// explicitly rolls back every UoW operation.
    pub async fn transaction<T, F>(&self, operation: F) -> PostgresStoreResult<T>
    where
        F: for<'a> FnOnce(&'a mut PostgresUnitOfWork) -> BoxFuture<'a, PostgresStoreResult<T>>,
    {
        let mut pooled_client = self.client().await?;
        // `deadpool_postgres::Object::transaction` returns deadpool's wrapper.
        // Bind its underlying tokio client first, so UoW owns the native RAII
        // `tokio_postgres::Transaction`, not a pool wrapper or raw SQL.
        let client: &mut tokio_postgres::Client = &mut pooled_client;
        let transaction = client.transaction().await.map_err(backend)?;
        let mut uow = PostgresUnitOfWork {
            transaction,
            tables: self.tables.clone(),
        };
        match operation(&mut uow).await {
            Ok(value) => {
                uow.transaction.commit().await.map_err(backend)?;
                Ok(value)
            }
            Err(error) => {
                if let Err(rollback) = uow.transaction.rollback().await {
                    return Err(PostgresStoreError::Backend(format!(
                        "{error}; rollback failed: {rollback}"
                    )));
                }
                Err(error)
            }
        }
    }

    pub async fn alloc_event_seq(&self, namespace: &str) -> PostgresStoreResult<i64> {
        require_namespace(namespace)?;
        let namespace = namespace.to_owned();
        self.transaction(move |uow| Box::pin(async move { uow.alloc_event_seq(&namespace).await }))
            .await
    }

    pub async fn next_global_seq(&self) -> PostgresStoreResult<i64> {
        self.transaction(|uow| Box::pin(async move { uow.next_global_seq().await }))
            .await
    }

    pub async fn current_global_seq(&self) -> PostgresStoreResult<i64> {
        let client = self.client().await?;
        current_global_seq(&**client, &self.tables).await
    }

    pub async fn next_user_seq(&self, user_id: &str) -> PostgresStoreResult<i64> {
        let user_id = user_id.to_owned();
        self.transaction(move |uow| Box::pin(async move { uow.next_user_seq(&user_id).await }))
            .await
    }

    pub async fn current_user_seq(&self, user_id: &str) -> PostgresStoreResult<i64> {
        let client = self.client().await?;
        current_user_seq(&**client, &self.tables, user_id).await
    }

    pub async fn set_user_seq(&self, user_id: &str, value: i64) -> PostgresStoreResult<()> {
        let user_id = user_id.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move { uow.set_user_seq(&user_id, value).await })
        })
        .await
    }

    pub async fn get_index_applied_fingerprint(
        &self,
        namespace: &str,
        coalesce_key: &str,
    ) -> PostgresStoreResult<Option<String>> {
        let client = self.client().await?;
        get_index_applied_fingerprint(&**client, &self.tables, namespace, coalesce_key).await
    }

    pub async fn set_index_applied_fingerprint(
        &self,
        namespace: &str,
        coalesce_key: &str,
        applied_fingerprint: Option<String>,
        last_job_id: Option<String>,
    ) -> PostgresStoreResult<()> {
        let namespace = namespace.to_owned();
        let coalesce_key = coalesce_key.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.set_index_applied_fingerprint(
                    &namespace,
                    &coalesce_key,
                    applied_fingerprint,
                    last_job_id,
                )
                .await
            })
        })
        .await
    }

    /// Apply graph projection row and authoritative event in one PostgreSQL
    /// transaction.  `gke_*` remains Python's schema; this merely joins it.
    pub async fn apply_graph_mutation(
        &self,
        mutation: GraphMutation,
    ) -> PostgresStoreResult<AppliedGraphMutation> {
        validate_graph_mutation(&mutation)?;
        self.transaction(move |uow| {
            Box::pin(async move { uow.apply_graph_mutation(mutation).await })
        })
        .await
    }

    /// Apply graph events and projections in one PostgreSQL transaction.
    pub async fn apply_graph_mutations_atomic(
        &self,
        mutations: Vec<GraphMutation>,
    ) -> PostgresStoreResult<Vec<AppliedGraphMutation>> {
        validate_graph_mutation_batch(&GraphMutationBatch {
            mutations: mutations.clone(),
            index_jobs: Vec::new(),
        })?;
        self.transaction(move |uow| {
            Box::pin(async move {
                let mut results = Vec::with_capacity(mutations.len());
                for mutation in mutations {
                    results.push(uow.apply_graph_mutation(mutation).await?);
                }
                Ok(results)
            })
        })
        .await
    }

    /// Apply graph mutations and explicit index intents in one PostgreSQL
    /// transaction. The queue payload is never inferred from graph metadata.
    pub async fn apply_graph_mutations_with_index_jobs_atomic(
        &self,
        batch: GraphMutationBatch,
    ) -> PostgresStoreResult<Vec<AppliedGraphMutation>> {
        validate_graph_mutation_batch(&batch)?;
        self.transaction(move |uow| {
            Box::pin(async move {
                let mut results = Vec::with_capacity(batch.mutations.len());
                for mutation in batch.mutations {
                    results.push(uow.apply_graph_mutation(mutation).await?);
                }
                for job in batch.index_jobs {
                    uow.enqueue_index_job(job).await?;
                }
                Ok(results)
            })
        })
        .await
    }

    pub async fn apply_graph_metadata_patch_mutation(
        &self,
        mutation: GraphMetadataPatchMutation,
    ) -> PostgresStoreResult<Option<AppliedGraphMutation>> {
        self.transaction(move |uow| {
            Box::pin(async move { uow.apply_graph_metadata_patch_mutation(mutation).await })
        })
        .await
    }

    pub async fn apply_graph_delete_mutation(
        &self,
        mutation: GraphDeleteMutation,
    ) -> PostgresStoreResult<Option<AppliedGraphMutation>> {
        self.transaction(move |uow| {
            Box::pin(async move { uow.apply_graph_delete_mutation(mutation).await })
        })
        .await
    }

    pub async fn upsert_graph_projection(
        &self,
        write: GraphProjectionUpsert,
    ) -> PostgresStoreResult<()> {
        self.transaction(move |uow| {
            Box::pin(async move { uow.upsert_graph_projection(write).await })
        })
        .await
    }

    /// Admit an embedding result only if its source document and metadata are
    /// still the current canonical projection. Promotion changes the vector
    /// only; a delayed provider result cannot overwrite a newer graph record.
    pub async fn promote_graph_embedding_projection(
        &self,
        namespace: String,
        entity_kind: String,
        candidate: GraphRecord,
    ) -> PostgresStoreResult<bool> {
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.promote_graph_embedding_projection(&namespace, &entity_kind, candidate)
                    .await
            })
        })
        .await
    }

    /// Physically remove a serving projection only after its canonical row is
    /// tombstoned. The lifecycle event is already durable; this operation only
    /// performs the idempotent materialized cleanup used by index workers.
    pub async fn delete_graph_projection_if_tombstoned(
        &self,
        scope: GraphScope,
        entity_kind: String,
        entity_id: String,
    ) -> PostgresStoreResult<bool> {
        self.transaction(move |uow| {
            Box::pin(async move {
                delete_graph_projection_if_tombstoned(
                    &uow.transaction,
                    &uow.tables,
                    &scope,
                    &entity_kind,
                    &entity_id,
                )
                .await
            })
        })
        .await
    }

    pub async fn patch_graph_projection_metadata(
        &self,
        patch: GraphProjectionMetadataPatch,
    ) -> PostgresStoreResult<bool> {
        self.transaction(move |uow| {
            Box::pin(async move { uow.patch_graph_projection_metadata(patch).await })
        })
        .await
    }

    pub async fn graph_projection_records(
        &self,
        read: GraphProjectionRead,
    ) -> PostgresStoreResult<Vec<GraphRecord>> {
        let client = self.client().await?;
        graph_projection_records(&**client, &self.tables, read, None, false).await
    }

    /// Read and lock graph projection rows within one explicit native UoW.
    /// This is for revision-gated projection updates; ordinary reads should
    /// continue to use `graph_projection_records`.
    pub async fn graph_projection_records_for_update(
        &self,
        read: GraphProjectionRead,
    ) -> PostgresStoreResult<Vec<GraphRecord>> {
        self.transaction(move |uow| {
            Box::pin(async move { uow.graph_projection_records_for_update(read).await })
        })
        .await
    }

    /// Report whether any of the Python-compatible graph/vector tables already
    /// contain rows. Embedding-profile binding uses this to reject unverified
    /// legacy vector spaces before admitting a new provider.
    pub async fn has_graph_projection_rows(&self) -> PostgresStoreResult<bool> {
        let client = self.client().await?;
        for table in ["gke_nodes", "gke_edges", "gke_documents", "gke_domains"] {
            let relation = qualified(&self.tables.quoted_schema, table)?;
            let exists: Option<String> = client
                .query_one("SELECT to_regclass($1)::text", &[&relation])
                .await
                .map_err(backend)?
                .get(0);
            if exists.is_none() {
                continue;
            }
            let sql = format!("SELECT EXISTS (SELECT 1 FROM {relation} LIMIT 1)");
            let has_rows: bool = client.query_one(&sql, &[]).await.map_err(backend)?.get(0);
            if has_rows {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Read ACL records for a bounded set of graph entities without scanning
    /// every ACL record in the namespace. Item-level rules share their entity
    /// ID, so this returns node/edge, grounding, and span rules together.
    pub async fn graph_acl_records_for_entities(
        &self,
        scope: &kogwistar_store::GraphScope,
        truth_graph: &str,
        entity_ids: &[String],
    ) -> PostgresStoreResult<Vec<GraphRecord>> {
        let client = self.client().await?;
        graph_acl_records_for_entities(&**client, &self.tables, scope, truth_graph, entity_ids)
            .await
    }

    pub async fn graph_projection_page(
        &self,
        page: GraphProjectionPageRead,
    ) -> PostgresStoreResult<Vec<GraphRecord>> {
        let client = self.client().await?;
        graph_projection_records(&**client, &self.tables, page.read, page.after_id, true).await
    }

    /// Read related graph projections from one stable PostgreSQL snapshot.
    ///
    /// Each read is keyset-paginated inside the same read-only repeatable-read
    /// transaction, so concurrent graph writes cannot make a caller combine
    /// rows from different committed states.
    pub async fn graph_projection_snapshot(
        &self,
        reads: Vec<GraphProjectionRead>,
    ) -> PostgresStoreResult<Vec<Vec<GraphRecord>>> {
        self.transaction(move |unit| {
            Box::pin(async move {
                unit.transaction
                    .batch_execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
                    .await
                    .map_err(backend)?;

                let mut snapshots = Vec::with_capacity(reads.len());
                for read in reads {
                    let page_size = read.limit;
                    let mut records = Vec::new();
                    let mut after_id: Option<String> = None;
                    if page_size > 0 {
                        loop {
                            let page = graph_projection_records(
                                &unit.transaction,
                                &unit.tables,
                                read.clone(),
                                after_id.clone(),
                                true,
                            )
                            .await?;
                            let Some(first) = page.first() else {
                                break;
                            };
                            if after_id.as_ref().is_some_and(|cursor| first.id <= *cursor)
                                || page.windows(2).any(|pair| pair[0].id >= pair[1].id)
                                || page.len() > page_size
                            {
                                return Err(PostgresStoreError::Backend(
                                    "graph projection snapshot returned a non-advancing or unordered page"
                                        .to_owned(),
                                ));
                            }
                            let has_more = page.len() == page_size;
                            after_id = page.last().map(|record| record.id.clone());
                            records.extend(page);
                            if !has_more {
                                break;
                            }
                        }
                    }
                    snapshots.push(records);
                }
                Ok(snapshots)
            })
        })
        .await
    }

    pub async fn graph_projection_vector_query(
        &self,
        query: GraphProjectionVectorQuery,
    ) -> PostgresStoreResult<Vec<VectorMatch>> {
        self.graph_projection_vector_query_excluding_metadata(query, BTreeMap::new())
            .await
    }

    pub async fn graph_projection_vector_query_excluding_metadata(
        &self,
        query: GraphProjectionVectorQuery,
        metadata_not_equals: BTreeMap<String, Value>,
    ) -> PostgresStoreResult<Vec<VectorMatch>> {
        let client = self.client().await?;
        graph_projection_vector_query(&**client, &self.tables, query, &metadata_not_equals).await
    }

    /// Execute multiple same-scope nearest-neighbor searches in one database
    /// round trip, preserving one ranked result row per query embedding.
    pub async fn graph_projection_vector_query_batch(
        &self,
        queries: Vec<GraphProjectionVectorQuery>,
    ) -> PostgresStoreResult<Vec<Vec<VectorMatch>>> {
        self.graph_projection_vector_query_batch_excluding_metadata(queries, BTreeMap::new())
            .await
    }

    pub async fn graph_projection_vector_query_batch_excluding_metadata(
        &self,
        queries: Vec<GraphProjectionVectorQuery>,
        metadata_not_equals: BTreeMap<String, Value>,
    ) -> PostgresStoreResult<Vec<Vec<VectorMatch>>> {
        if queries.is_empty() {
            return Ok(Vec::new());
        }
        let client = self.client().await?;
        graph_projection_vector_query_batch(&**client, &self.tables, queries, &metadata_not_equals)
            .await
    }

    pub async fn append_raw_entity_event(
        &self,
        namespace: &str,
        event: NewRawEntityEvent,
    ) -> PostgresStoreResult<AppendedRawEvent> {
        require_namespace(namespace)?;
        let namespace = namespace.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move { uow.append_raw_entity_event(&namespace, event).await })
        })
        .await
    }

    /// Raw replay is exclusive of `after_seq`, ordered ascending, and bounded.
    pub async fn replay_raw_events(
        &self,
        namespace: &str,
        after_seq: i64,
        limit: usize,
    ) -> PostgresStoreResult<Vec<RawEntityEvent>> {
        require_namespace(namespace)?;
        let client = self.client().await?;
        replay_raw_events(&**client, &self.tables, namespace, after_seq, limit).await
    }

    /// `MAX(seq)` over retained rows, not allocation state. Gaps are legal.
    pub async fn latest_retained_event_seq(&self, namespace: &str) -> PostgresStoreResult<i64> {
        require_namespace(namespace)?;
        let client = self.client().await?;
        latest_retained_event_seq(&**client, &self.tables, namespace).await
    }

    pub async fn prune_entity_events_after(
        &self,
        namespace: &str,
        to_seq: i64,
    ) -> PostgresStoreResult<u64> {
        require_namespace(namespace)?;
        let namespace = namespace.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move { uow.prune_entity_events_after(&namespace, to_seq).await })
        })
        .await
    }

    pub async fn replay_cursor(
        &self,
        namespace: &str,
        consumer: &str,
    ) -> PostgresStoreResult<ReplayCursor> {
        require_namespace(namespace)?;
        let client = self.client().await?;
        replay_cursor(&**client, &self.tables, namespace, consumer).await
    }

    /// Bounded operator-triggered recovery. Projection replacement and cursor
    /// movement are committed only together with the authoritative replay fold.
    pub async fn recover_entity_projection(
        &self,
        request: EntityRecoveryRequest,
    ) -> PostgresStoreResult<EntityRecoveryReport> {
        validate_entity_recovery_request(&request)?;
        self.transaction(move |uow| {
            Box::pin(async move { uow.recover_entity_projection(&request).await })
        })
        .await
    }

    /// Explicit full rebuild. `ensure_schema` and opening a store never call it.
    pub async fn rebuild_entity_projection(
        &self,
        request: EntityRebuildRequest,
    ) -> PostgresStoreResult<EntityRecoveryReport> {
        validate_entity_rebuild_request(&request)?;
        self.transaction(move |uow| {
            Box::pin(async move { uow.rebuild_entity_projection(&request).await })
        })
        .await
    }

    pub async fn get_named_projection(
        &self,
        namespace: &str,
        key: &str,
    ) -> PostgresStoreResult<Option<NamedProjection>> {
        require_namespace(namespace)?;
        let client = self.client().await?;
        named_projection(&**client, &self.tables, namespace, key).await
    }

    pub async fn list_named_projections(
        &self,
        namespace: &str,
    ) -> PostgresStoreResult<Vec<NamedProjection>> {
        require_namespace(namespace)?;
        let client = self.client().await?;
        named_projections(&**client, &self.tables, namespace).await
    }

    pub async fn replace_named_projection(
        &self,
        namespace: &str,
        key: &str,
        projection: NamedProjectionWrite,
    ) -> PostgresStoreResult<()> {
        require_namespace(namespace)?;
        let namespace = namespace.to_owned();
        let key = key.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.replace_named_projection(&namespace, &key, projection)
                    .await
            })
        })
        .await
    }

    pub async fn compare_and_swap_named_projection(
        &self,
        namespace: &str,
        key: &str,
        expected_last_authoritative_seq: Option<i64>,
        expected_last_materialized_seq: Option<i64>,
        projection: NamedProjectionWrite,
    ) -> PostgresStoreResult<bool> {
        require_namespace(namespace)?;
        let namespace = namespace.to_owned();
        let key = key.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.compare_and_swap_named_projection(
                    &namespace,
                    &key,
                    expected_last_authoritative_seq,
                    expected_last_materialized_seq,
                    projection,
                )
                .await
            })
        })
        .await
    }

    /// Atomically compare-and-swaps a set of named projections.
    pub async fn compare_and_swap_named_projections(
        &self,
        updates: Vec<NamedProjectionCas>,
    ) -> PostgresStoreResult<bool> {
        self.transaction(move |uow| {
            Box::pin(async move { uow.compare_and_swap_named_projections(updates).await })
        })
        .await
    }

    pub async fn clear_named_projection(
        &self,
        namespace: &str,
        key: &str,
    ) -> PostgresStoreResult<()> {
        require_namespace(namespace)?;
        let namespace = namespace.to_owned();
        let key = key.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move { uow.clear_named_projection(&namespace, &key).await })
        })
        .await
    }

    pub async fn clear_projection_namespace(&self, namespace: &str) -> PostgresStoreResult<()> {
        require_namespace(namespace)?;
        let namespace = namespace.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move { uow.clear_projection_namespace(&namespace).await })
        })
        .await
    }

    pub async fn put_workflow_design_snapshot(
        &self,
        workflow_id: &str,
        snapshot: WorkflowDesignSnapshotWrite,
    ) -> PostgresStoreResult<()> {
        let workflow_id = workflow_id.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.put_workflow_design_snapshot(&workflow_id, snapshot)
                    .await
            })
        })
        .await
    }

    pub async fn get_workflow_design_snapshot(
        &self,
        workflow_id: &str,
        max_version: i64,
        schema_version: i64,
    ) -> PostgresStoreResult<Option<WorkflowDesignSnapshot>> {
        let client = self.client().await?;
        workflow_design_snapshot(
            &**client,
            &self.tables,
            workflow_id,
            max_version,
            schema_version,
        )
        .await
    }

    pub async fn clear_workflow_design_snapshots(
        &self,
        workflow_id: &str,
    ) -> PostgresStoreResult<()> {
        let workflow_id = workflow_id.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move { uow.clear_workflow_design_snapshots(&workflow_id).await })
        })
        .await
    }

    pub async fn put_workflow_design_delta(
        &self,
        workflow_id: &str,
        delta: WorkflowDesignDeltaWrite,
    ) -> PostgresStoreResult<()> {
        let workflow_id = workflow_id.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move { uow.put_workflow_design_delta(&workflow_id, delta).await })
        })
        .await
    }

    pub async fn get_workflow_design_delta(
        &self,
        workflow_id: &str,
        version: i64,
        schema_version: i64,
    ) -> PostgresStoreResult<Option<WorkflowDesignDelta>> {
        let client = self.client().await?;
        workflow_design_delta(
            &**client,
            &self.tables,
            workflow_id,
            version,
            schema_version,
        )
        .await
    }

    pub async fn clear_workflow_design_deltas(&self, workflow_id: &str) -> PostgresStoreResult<()> {
        let workflow_id = workflow_id.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move { uow.clear_workflow_design_deltas(&workflow_id).await })
        })
        .await
    }

    pub async fn create_server_run(&self, run: ServerRunCreate) -> PostgresStoreResult<()> {
        self.transaction(move |uow| Box::pin(async move { uow.create_server_run(run).await }))
            .await
    }
    pub async fn get_server_run(&self, run_id: &str) -> PostgresStoreResult<Option<ServerRun>> {
        server_run(&**self.client().await?, &self.tables, run_id).await
    }
    pub async fn list_server_runs(
        &self,
        status: Option<&str>,
        workflow_id: Option<&str>,
        conversation_id: Option<&str>,
        limit: usize,
    ) -> PostgresStoreResult<Vec<ServerRun>> {
        server_runs(
            &**self.client().await?,
            &self.tables,
            status,
            workflow_id,
            conversation_id,
            limit,
        )
        .await
    }
    pub async fn append_server_run_event(
        &self,
        run_id: &str,
        event_type: &str,
        payload_json: String,
    ) -> PostgresStoreResult<ServerRunEvent> {
        let run_id = run_id.to_owned();
        let event_type = event_type.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.append_server_run_event(&run_id, &event_type, payload_json)
                    .await
            })
        })
        .await
    }
    pub async fn list_server_run_events(
        &self,
        run_id: &str,
        after_seq: i64,
        limit: usize,
    ) -> PostgresStoreResult<Vec<ServerRunEvent>> {
        server_run_events(
            &**self.client().await?,
            &self.tables,
            run_id,
            after_seq,
            limit,
        )
        .await
    }
    pub async fn update_server_run(
        &self,
        run_id: &str,
        update: ServerRunUpdate,
    ) -> PostgresStoreResult<()> {
        let run_id = run_id.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move { uow.update_server_run(&run_id, update).await })
        })
        .await
    }
    pub async fn request_server_run_cancel(&self, run_id: &str) -> PostgresStoreResult<()> {
        let run_id = run_id.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move { uow.request_server_run_cancel(&run_id).await })
        })
        .await
    }

    pub async fn apply_recorded_runtime_transition(
        &self,
        transition: RecordedRuntimeTransition,
        abort_after_writes: bool,
    ) -> PostgresStoreResult<RecordedTransitionResult> {
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.apply_recorded_runtime_transition(transition, abort_after_writes)
                    .await
            })
        })
        .await
    }

    pub async fn apply_claimed_recorded_runtime_transition(
        &self,
        handoff: RecordedWorkerHandoff,
        transition: RecordedRuntimeTransition,
        abort_after_writes: bool,
    ) -> PostgresStoreResult<RecordedTransitionResult> {
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.apply_claimed_recorded_runtime_transition(
                    handoff,
                    transition,
                    abort_after_writes,
                )
                .await
            })
        })
        .await
    }

    pub async fn apply_claimed_recorded_worker_effect(
        &self,
        handoff: RecordedWorkerHandoff,
        effect: RecordedWorkerSuccessEffect,
    ) -> PostgresStoreResult<RecordedTransitionResult> {
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.apply_claimed_recorded_worker_effect(handoff, effect)
                    .await
            })
        })
        .await
    }

    pub async fn resume_recorded_runtime_token(
        &self,
        run_id: &str,
        workflow_id: &str,
        conversation_id: &str,
        node_id: &str,
        token_id: &str,
        resume_payload: Option<Value>,
    ) -> PostgresStoreResult<RecordedTransitionResult> {
        let run_id = run_id.to_owned();
        let workflow_id = workflow_id.to_owned();
        let conversation_id = conversation_id.to_owned();
        let node_id = node_id.to_owned();
        let token_id = token_id.to_owned();
        self.transaction(|uow| {
            Box::pin(async move {
                let current = read_recorded_runtime_state(
                    &uow.transaction,
                    &uow.tables,
                    &run_id,
                    &workflow_id,
                    &conversation_id,
                )
                .await?
                .ok_or_else(|| {
                    PostgresStoreError::RecordedRuntimeConflict(format!(
                        "run {run_id:?} has no recorded runtime state"
                    ))
                })?;
                let parent = current
                    .frontier
                    .suspended
                    .iter()
                    .find(|(node, _, token, _)| node == &node_id && token == &token_id)
                    .and_then(|(_, _, _, parent)| parent.clone());
                let frontier = frontier_after_worker_resume(
                    &current.frontier,
                    &node_id,
                    &token_id,
                    parent.as_deref(),
                )?;
                let expected_event_seq =
                    latest_server_run_event_seq(&uow.transaction, &uow.tables, &run_id).await?;
                let result = uow
                    .apply_recorded_runtime_transition(
                        RecordedRuntimeTransition {
                            contract_version: RECORDED_RUNTIME_CONTRACT_VERSION,
                            transition_id: format!(
                                "resume-{run_id}-{token_id}-{expected_event_seq}"
                            ),
                            expected_event_seq,
                            kind: kogwistar_runtime::RecordedTransitionKind::ResumeResult,
                            run_id: run_id.clone(),
                            workflow_id: workflow_id.clone(),
                            conversation_id: conversation_id.clone(),
                            user_id: None,
                            user_turn_node_id: None,
                            step_seq: current.last_step_seq.saturating_add(1),
                            node_id: Some(node_id.clone()),
                            token_id: Some(token_id.clone()),
                            parent_token_id: parent.clone(),
                            initial_state: None,
                            state_update: Vec::new(),
                            update: None,
                            state_schema: Map::new(),
                            frontier: Some(frontier),
                            result: None,
                            wait_reason: None,
                            resume_payload,
                            errors: Vec::new(),
                        },
                        false,
                    )
                    .await?;
                let message_id = format!(
                    "lane|{}",
                    kogwistar_contracts::stable_id(
                        "runtime.worker.request",
                        &[
                            run_id.clone(),
                            token_id.clone(),
                            node_id.clone(),
                            result
                                .reduced
                                .state
                                .last_step_seq
                                .saturating_add(1)
                                .to_string(),
                        ],
                    )
                );
                let now = unix_epoch_millis() / 1000;
                let join_mask = result
                    .reduced
                    .state
                    .frontier
                    .pending
                    .iter()
                    .find(|(_, _, token, _)| token == &token_id)
                    .map(|(_, mask, _, _)| *mask)
                    .unwrap_or(0);
                let op = result
                    .reduced
                    .state
                    .state
                    .get("_rt_node_ops")
                    .and_then(Value::as_object)
                    .and_then(|ops| ops.get(&node_id))
                    .cloned()
                    .unwrap_or_else(|| Value::String("noop".to_owned()));
                let payload = RuntimeStepExecutePayload::from_recorded_state(
                    &result.reduced.state,
                    RuntimeStepExecuteRequest {
                        node_id: node_id.clone(),
                        op,
                        join_mask,
                        token_id: token_id.clone(),
                        parent_token_id: parent.clone(),
                        step_seq: result.reduced.state.last_step_seq.saturating_add(1),
                        expected_event_seq: result.event_seq,
                        resume_effect: result.reduced.state.resume_payload.clone(),
                    },
                );
                uow.project_lane_message(NewProjectedLaneMessage {
                    message_id,
                    namespace: "workflow".to_owned(),
                    purpose: "system".to_owned(),
                    inbox_id: "workflow-runtime".to_owned(),
                    conversation_id: conversation_id.clone(),
                    recipient_id: "python-worker".to_owned(),
                    sender_id: "rust-scheduler".to_owned(),
                    msg_type: "workflow.step.execute".to_owned(),
                    status: "pending".to_owned(),
                    created_at: now,
                    available_at: now,
                    run_id: Some(run_id.clone()),
                    step_id: Some(node_id.clone()),
                    correlation_id: Some(run_id.clone()),
                    payload_json: Some(serde_json::to_string(&payload)?),
                    error_json: None,
                })
                .await?;
                Ok(result)
            })
        })
        .await
    }

    pub async fn read_recorded_runtime_state(
        &self,
        run_id: &str,
        workflow_id: &str,
        conversation_id: &str,
    ) -> PostgresStoreResult<Option<RecordedRuntimeState>> {
        read_recorded_runtime_state(
            &**self.client().await?,
            &self.tables,
            run_id,
            workflow_id,
            conversation_id,
        )
        .await
    }

    pub async fn enqueue_index_job(&self, job: NewIndexJob) -> PostgresStoreResult<String> {
        let namespace = job.namespace.clone();
        require_namespace(&namespace)?;
        self.transaction(move |uow| Box::pin(async move { uow.enqueue_index_job(job).await }))
            .await
    }
    pub async fn claim_index_jobs(
        &self,
        limit: usize,
        lease_seconds: i64,
        namespace: Option<&str>,
    ) -> PostgresStoreResult<Vec<IndexJob>> {
        let namespace = namespace.map(ToOwned::to_owned);
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.claim_index_jobs(limit, lease_seconds, namespace.as_deref())
                    .await
            })
        })
        .await
    }
    pub async fn claim_index_jobs_filtered(
        &self,
        limit: usize,
        lease_seconds: i64,
        filter: IndexJobClaimFilter,
    ) -> PostgresStoreResult<Vec<IndexJob>> {
        if filter.index_kinds.is_empty()
            || filter.index_kinds.iter().any(|kind| kind.trim().is_empty())
        {
            return Err(PostgresStoreError::Backend(
                "filtered index-job claim requires non-empty index kinds".to_owned(),
            ));
        }
        let namespace = filter.namespace;
        let index_kinds = filter.index_kinds;
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.claim_index_jobs_filtered(
                    limit,
                    lease_seconds,
                    namespace.as_deref(),
                    &index_kinds,
                )
                .await
            })
        })
        .await
    }
    pub async fn reconcile_join_index_jobs(
        &self,
        limit: usize,
        lease_seconds: i64,
        namespace: Option<&str>,
        names: GraphIndexTableNames,
    ) -> PostgresStoreResult<usize> {
        let namespace = namespace.map(ToOwned::to_owned);
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.reconcile_join_index_jobs(limit, lease_seconds, namespace.as_deref(), names)
                    .await
            })
        })
        .await
    }
    pub async fn mark_index_job_done(
        &self,
        job_id: &str,
        claim_token: Option<&str>,
    ) -> PostgresStoreResult<bool> {
        let job_id = job_id.to_owned();
        let claim_token = claim_token.map(ToOwned::to_owned);
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.mark_index_job_done(&job_id, claim_token.as_deref())
                    .await
            })
        })
        .await
    }
    pub async fn accept_index_job_result(
        &self,
        job_id: &str,
        claim_token: &str,
        result_json: &str,
        result_sha256: &str,
    ) -> PostgresStoreResult<AcceptedIndexJobResult> {
        let job_id = job_id.to_owned();
        let claim_token = claim_token.to_owned();
        let result_json = result_json.to_owned();
        let result_sha256 = result_sha256.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.accept_index_job_result(&job_id, &claim_token, &result_json, &result_sha256)
                    .await
            })
        })
        .await
    }
    pub async fn index_job_result(
        &self,
        job_id: &str,
    ) -> PostgresStoreResult<Option<AcceptedIndexJobResult>> {
        index_job_result(&**self.client().await?, &self.tables, job_id).await
    }
    pub async fn mark_index_job_failed(
        &self,
        job_id: &str,
        error: &str,
        final_: bool,
        claim_token: Option<&str>,
    ) -> PostgresStoreResult<()> {
        let job_id = job_id.to_owned();
        let error = error.to_owned();
        let claim_token = claim_token.map(ToOwned::to_owned);
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.mark_index_job_failed(&job_id, &error, final_, claim_token.as_deref())
                    .await
            })
        })
        .await
    }
    pub async fn bump_retry_and_requeue(
        &self,
        job_id: &str,
        error: &str,
        delay: i64,
        claim_token: Option<&str>,
    ) -> PostgresStoreResult<()> {
        let job_id = job_id.to_owned();
        let error = error.to_owned();
        let claim_token = claim_token.map(ToOwned::to_owned);
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.bump_retry_and_requeue(&job_id, &error, delay, claim_token.as_deref())
                    .await
            })
        })
        .await
    }
    pub async fn renew_index_job_lease(
        &self,
        job_id: &str,
        claim_token: &str,
        lease_seconds: i64,
    ) -> PostgresStoreResult<bool> {
        let job_id = job_id.to_owned();
        let claim_token = claim_token.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.renew_index_job_lease(&job_id, &claim_token, lease_seconds)
                    .await
            })
        })
        .await
    }
    pub async fn requeue_index_job_at_tail(
        &self,
        job_id: &str,
        payload_json: String,
        delay: i64,
        claim_token: Option<&str>,
    ) -> PostgresStoreResult<()> {
        let job_id = job_id.to_owned();
        let claim_token = claim_token.map(ToOwned::to_owned);
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.requeue_index_job_at_tail(&job_id, payload_json, delay, claim_token.as_deref())
                    .await
            })
        })
        .await
    }
    pub async fn list_index_jobs(
        &self,
        namespace: Option<&str>,
        status: Option<&str>,
        entity_kind: Option<&str>,
        entity_id: Option<&str>,
        index_kind: Option<&str>,
        limit: usize,
    ) -> PostgresStoreResult<Vec<IndexJob>> {
        let client = self.client().await?;
        list_index_jobs(
            &**client,
            &self.tables,
            namespace,
            status,
            entity_kind,
            entity_id,
            index_kind,
            limit,
        )
        .await
    }

    pub async fn project_lane_message(
        &self,
        row: NewProjectedLaneMessage,
    ) -> PostgresStoreResult<()> {
        self.transaction(move |uow| Box::pin(async move { uow.project_lane_message(row).await }))
            .await
    }
    pub async fn get_projected_lane_message(
        &self,
        id: &str,
    ) -> PostgresStoreResult<Option<ProjectedLaneMessage>> {
        projected_lane_message(&**self.client().await?, &self.tables, id).await
    }
    pub async fn list_projected_lane_messages(
        &self,
        filter: LaneMessageFilter,
    ) -> PostgresStoreResult<Vec<ProjectedLaneMessage>> {
        list_projected_lane_messages(&**self.client().await?, &self.tables, &filter).await
    }
    pub async fn update_projected_lane_message_status(
        &self,
        id: &str,
        status: &str,
        error: Option<String>,
    ) -> PostgresStoreResult<()> {
        let id = id.to_owned();
        let status = status.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.update_projected_lane_message_status(&id, &status, error)
                    .await
            })
        })
        .await
    }
    pub async fn update_projected_lane_message_links(
        &self,
        id: &str,
        prev: Option<String>,
        next: Option<String>,
        inbox_tail: Option<String>,
        conversation_tail: Option<String>,
    ) -> PostgresStoreResult<()> {
        let id = id.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.update_projected_lane_message_links(
                    &id,
                    prev,
                    next,
                    inbox_tail,
                    conversation_tail,
                )
                .await
            })
        })
        .await
    }
    pub async fn clear_projected_lane_messages(&self, namespace: &str) -> PostgresStoreResult<u64> {
        let namespace = namespace.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move { uow.clear_projected_lane_messages(&namespace).await })
        })
        .await
    }
    pub async fn claim_projected_lane_messages(
        &self,
        namespace: &str,
        inbox: &str,
        owner: &str,
        limit: usize,
        lease: i64,
    ) -> PostgresStoreResult<Vec<ProjectedLaneMessage>> {
        let namespace = namespace.to_owned();
        let inbox = inbox.to_owned();
        let owner = owner.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.claim_projected_lane_messages(&namespace, &inbox, &owner, limit, lease)
                    .await
            })
        })
        .await
    }
    pub async fn claim_projected_lane_messages_filtered(
        &self,
        filter: LaneMessageClaimFilter,
    ) -> PostgresStoreResult<Vec<ProjectedLaneMessage>> {
        self.transaction(move |uow| {
            Box::pin(async move { uow.claim_projected_lane_messages_filtered(filter).await })
        })
        .await
    }
    pub async fn ack_projected_lane_message(
        &self,
        id: &str,
        owner: &str,
    ) -> PostgresStoreResult<()> {
        let id = id.to_owned();
        let owner = owner.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move { uow.ack_projected_lane_message(&id, &owner).await })
        })
        .await
    }
    pub async fn requeue_projected_lane_message(
        &self,
        id: &str,
        owner: &str,
        error: Option<String>,
        delay: i64,
    ) -> PostgresStoreResult<()> {
        let id = id.to_owned();
        let owner = owner.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.requeue_projected_lane_message(&id, &owner, error, delay)
                    .await
            })
        })
        .await
    }
    pub async fn dead_letter_projected_lane_message(
        &self,
        id: &str,
        owner: &str,
        error: Option<String>,
    ) -> PostgresStoreResult<()> {
        let id = id.to_owned();
        let owner = owner.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.dead_letter_projected_lane_message(&id, &owner, error)
                    .await
            })
        })
        .await
    }
    pub async fn repair_orphaned_claimed_lane_messages(
        &self,
        namespace: &str,
        inbox_id: Option<&str>,
        limit: usize,
    ) -> PostgresStoreResult<Vec<String>> {
        let namespace = namespace.to_owned();
        let inbox_id = inbox_id.map(str::to_owned);
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.repair_orphaned_claimed_lane_messages(&namespace, inbox_id.as_deref(), limit)
                    .await
            })
        })
        .await
    }

    /// Strict trait-compatible advance: monotonic and at most retained latest.
    pub async fn strict_advance_replay_cursor(
        &self,
        namespace: &str,
        consumer: &str,
        last_seq: i64,
    ) -> PostgresStoreResult<ReplayCursor> {
        require_namespace(namespace)?;
        let namespace = namespace.to_owned();
        let consumer = consumer.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.strict_advance_replay_cursor(&namespace, &consumer, last_seq)
                    .await
            })
        })
        .await
    }

    /// Python legacy `cursor_set`: overwrite without monotonic/range checks.
    pub async fn set_replay_cursor_legacy(
        &self,
        namespace: &str,
        consumer: &str,
        last_seq: i64,
    ) -> PostgresStoreResult<ReplayCursor> {
        require_namespace(namespace)?;
        let namespace = namespace.to_owned();
        let consumer = consumer.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.set_replay_cursor_legacy(&namespace, &consumer, last_seq)
                    .await
            })
        })
        .await
    }

    pub(super) async fn client(&self) -> PostgresStoreResult<deadpool_postgres::Object> {
        self.pool.get().await.map_err(backend)
    }
}
