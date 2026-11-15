use super::super::*;

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
        ensure_graph_scope_identity_indexes(&transaction, &self.tables).await?;
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
        self.create_graph_schema_with_node_index(embedding_dim, names, "gke_node_index")
            .await
    }

    /// Create Python-compatible graph projections and its separate node-index
    /// vector collection, preserving the existing schema and column contract.
    ///
    /// PYTHON_PARITY_V1: rust=rust/crates/kogwistar-store-postgres/src/store/connection.rs#create_graph_schema_with_node_index; python=kogwistar/engine_core/postgres_backend.py#PgVectorBackend._ensure_schema_sync; fixture=contracts/golden/pg-node-index-schema-v1.json; rust_test=rust/crates/kogwistar-store-postgres/src/tests/graph.rs#postgres_named_vector_collections_are_scoped_ranked_and_not_graph_entities; python_test=tests/core/test_rust_store_graph_mutation_differential.py#test_graph_mutation_fresh_rust_schema_and_validation_rollback
    pub async fn create_graph_schema_with_node_index(
        &self,
        embedding_dim: usize,
        names: GraphTableNames,
        node_index_table: &str,
    ) -> PostgresStoreResult<()> {
        let mut pooled_client = self.client().await?;
        let client: &mut tokio_postgres::Client = &mut pooled_client;
        let transaction = client.transaction().await.map_err(|error| {
            PostgresStoreError::Backend(format!("graph schema transaction start failed: {error:?}"))
        })?;
        transaction
            .query_one(
                "SELECT pg_advisory_xact_lock(hashtext('kogwistar.pgvector.extension'))",
                &[],
            )
            .await
            .map_err(|error| {
                PostgresStoreError::Backend(format!("graph schema advisory lock failed: {error:?}"))
            })?;
        create_graph_schema(
            &transaction,
            &self.tables,
            embedding_dim,
            names,
            node_index_table,
        )
        .await?;
        transaction.commit().await.map_err(|error| {
            PostgresStoreError::Backend(format!(
                "graph schema transaction commit failed: {error:?}"
            ))
        })
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
        self.transaction_with_graph_schemas(&[], operation).await
    }

    /// Run runtime and graph work in one PostgreSQL transaction while allowing
    /// graph projections to target other schemas in the same database.
    /// The additional names are schema-qualified graph targets only; all
    /// transaction/event ownership remains with this store.
    pub async fn transaction_with_graph_schemas<T, F>(
        &self,
        graph_schemas: &[&str],
        operation: F,
    ) -> PostgresStoreResult<T>
    where
        F: for<'a> FnOnce(&'a mut PostgresUnitOfWork) -> BoxFuture<'a, PostgresStoreResult<T>>,
    {
        let mut pooled_client = self.client().await?;
        // `deadpool_postgres::Object::transaction` returns deadpool's wrapper.
        // Bind its underlying tokio client first, so UoW owns the native RAII
        // `tokio_postgres::Transaction`, not a pool wrapper or raw SQL.
        let client: &mut tokio_postgres::Client = &mut pooled_client;
        let transaction = client.transaction().await.map_err(backend)?;
        let mut graph_tables_by_schema = std::collections::HashMap::new();
        graph_tables_by_schema.insert(self.tables.schema.clone(), self.tables.clone());
        for schema in graph_schemas {
            let tables = Tables::new(schema)?;
            graph_tables_by_schema.insert(tables.schema.clone(), tables);
        }
        let mut uow = PostgresUnitOfWork {
            transaction,
            tables: self.tables.clone(),
            graph_tables_by_schema,
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

    pub(crate) async fn client(&self) -> PostgresStoreResult<deadpool_postgres::Object> {
        self.pool.get().await.map_err(backend)
    }
}
