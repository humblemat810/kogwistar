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

    pub(crate) async fn client(&self) -> PostgresStoreResult<deadpool_postgres::Object> {
        self.pool.get().await.map_err(backend)
    }
}
