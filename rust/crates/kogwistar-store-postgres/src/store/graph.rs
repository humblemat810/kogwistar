use super::super::*;

impl PostgresStore {
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

    pub async fn graph_projection_vector_query_level_lte_excluding_metadata(
        &self,
        query: GraphProjectionVectorQuery,
        metadata_not_equals: BTreeMap<String, Value>,
        level_lte: i64,
    ) -> PostgresStoreResult<Vec<VectorMatch>> {
        let client = self.client().await?;
        super::super::graph::graph_projection_vector_query_with_level_lte(
            &**client,
            &self.tables,
            query,
            &metadata_not_equals,
            Some(level_lte),
        )
        .await
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
}
