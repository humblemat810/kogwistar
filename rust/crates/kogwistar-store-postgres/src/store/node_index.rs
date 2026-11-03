use super::super::*;
use kogwistar_store::{DistanceMetric, GraphRecord, VectorMatch, VectorQuery};

impl PostgresStore {
    /// Upsert records into Python's Chroma-shaped node-index table.
    ///
    /// PYTHON_PARITY_V1: rust=rust/crates/kogwistar-store-postgres/src/store/node_index.rs#upsert_node_index; python=kogwistar/engine_core/postgres_backend.py#PgVectorBackend.node_index_upsert
    pub async fn upsert_node_index(
        &self,
        table_name: &str,
        records: Vec<GraphRecord>,
    ) -> PostgresStoreResult<()> {
        if records.len() > 10_000 {
            return Err(PostgresStoreError::Backend(
                "node-index upsert exceeds the 10000-record limit".to_owned(),
            ));
        }
        let table = qualified(&self.tables.quoted_schema, table_name)?;
        for record in &records {
            if record.id.trim().is_empty() || record.id.contains('\0') {
                return Err(PostgresStoreError::Backend(
                    "node-index record id must be non-empty and contain no NUL".to_owned(),
                ));
            }
            if let Some(embedding) = &record.embedding {
                finite_vector(embedding)?;
            }
        }
        if records.is_empty() {
            return Ok(());
        }
        let mut client = self.client().await?;
        let transaction = client.transaction().await.map_err(backend)?;
        for record in records {
            let metadata = serde_json::to_string(&record.metadata)?;
            let embedding = record.embedding.as_deref().map(vector_literal);
            transaction
                .execute(
                    &format!(
                        "INSERT INTO {table} (id,document,metadata,embedding) \
                         VALUES ($1,$2,$3::text::jsonb,$4::text::vector) \
                         ON CONFLICT(id) DO UPDATE SET document=EXCLUDED.document, \
                         metadata=EXCLUDED.metadata,embedding=EXCLUDED.embedding,updated_at=NOW()"
                    ),
                    &[&record.id, &record.document, &metadata, &embedding],
                )
                .await
                .map_err(backend)?;
        }
        transaction.commit().await.map_err(backend)
    }

    /// Read Python-compatible node-index records by ID, retaining request order.
    ///
    /// PYTHON_PARITY_V1: rust=rust/crates/kogwistar-store-postgres/src/store/node_index.rs#get_node_index; python=kogwistar/engine_core/postgres_backend.py#PgVectorBackend.node_index_get; rust_test=tests/core/test_rust_store_graph_mutation_differential.py#test_graph_mutation_fresh_rust_schema_and_validation_rollback
    pub async fn get_node_index(
        &self,
        table_name: &str,
        ids: &[String],
    ) -> PostgresStoreResult<Vec<GraphRecord>> {
        if ids.len() > 10_000 {
            return Err(PostgresStoreError::Backend(
                "node-index lookup exceeds the 10000-ID limit".to_owned(),
            ));
        }
        let table = qualified(&self.tables.quoted_schema, table_name)?;
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let client = self.client().await?;
        let rows = client
            .query(
                &format!(
                    "SELECT id,document,metadata::TEXT,embedding::TEXT FROM {table} WHERE id=ANY($1::varchar[])"
                ),
                &[&ids],
            )
            .await
            .map_err(backend)?;
        let mut by_id = rows
            .iter()
            .map(|row| {
                let metadata: String = row.try_get(2).map_err(backend)?;
                let embedding: Option<String> = row.try_get(3).map_err(backend)?;
                Ok((
                    row.try_get::<_, String>(0).map_err(backend)?,
                    GraphRecord {
                        id: row.try_get(0).map_err(backend)?,
                        document: row.try_get(1).map_err(backend)?,
                        metadata: serde_json::from_str(&metadata)?,
                        embedding: embedding.as_deref().map(serde_json::from_str).transpose()?,
                    },
                ))
            })
            .collect::<PostgresStoreResult<std::collections::HashMap<_, _>>>()?;
        Ok(ids.iter().filter_map(|id| by_id.remove(id)).collect())
    }

    /// Query the Python-compatible node-index table using pgvector ordering.
    /// The result is storage-only; callers must apply graph ACL and scope checks.
    ///
    /// PYTHON_PARITY_V1: rust=rust/crates/kogwistar-store-postgres/src/store/node_index.rs#query_node_index; python=kogwistar/engine_core/postgres_backend.py#PgVectorBackend.node_index_query; rust_test=tests/core/test_rust_store_graph_mutation_differential.py#test_graph_mutation_fresh_rust_schema_and_validation_rollback
    pub async fn query_node_index(
        &self,
        table_name: &str,
        query: &VectorQuery,
    ) -> PostgresStoreResult<Vec<VectorMatch>> {
        if query.limit > 10_000 {
            return Err(PostgresStoreError::Backend(
                "node-index query limit exceeds 10000".to_owned(),
            ));
        }
        finite_vector(&query.embedding)?;
        if query.limit == 0 {
            return Ok(Vec::new());
        }
        let table = qualified(&self.tables.quoted_schema, table_name)?;
        let operator = match query.metric {
            DistanceMetric::Cosine => "<=>",
            DistanceMetric::L2 => "<->",
            DistanceMetric::InnerProduct => "<#>",
        };
        let metadata = serde_json::to_string(&query.metadata.equals)?;
        let embedding = vector_literal(&query.embedding);
        let limit = i64::try_from(query.limit).unwrap_or(i64::MAX);
        let client = self.client().await?;
        let rows = client
            .query(
                &format!(
                    "SELECT id,document,metadata::TEXT,embedding::TEXT,embedding {operator} $1::text::vector AS distance \
                     FROM {table} WHERE embedding IS NOT NULL AND metadata @> $2::text::jsonb \
                     ORDER BY distance ASC,id ASC LIMIT $3"
                ),
                &[&embedding, &metadata, &limit],
            )
            .await
            .map_err(backend)?;
        rows.iter()
            .map(|row| {
                let record = graph_record_from_row(row)?;
                Ok(VectorMatch {
                    record,
                    distance: row.try_get(4).map_err(backend)?,
                })
            })
            .collect()
    }
}
