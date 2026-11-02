use super::super::*;
use kogwistar_store::VectorCollectionStore;

fn validate_collection_inputs(namespace: &str, collection: &str) -> StoreResult<()> {
    if namespace.trim().is_empty() {
        return Err(StoreError::EmptyNamespace);
    }
    if collection.trim().is_empty() || collection.contains('\0') {
        return Err(StoreError::InvalidIdentifier {
            identifier: "vector collection name",
        });
    }
    Ok(())
}

impl VectorCollectionStore for PostgresStore {
    async fn upsert_vector_collection(
        &self,
        namespace: &str,
        collection: &str,
        records: Vec<GraphRecord>,
    ) -> StoreResult<()> {
        validate_collection_inputs(namespace, collection)?;
        if records.len() > 10_000 {
            return Err(StoreError::Backend {
                backend: "postgres-vector-collection".to_owned(),
                message: "upsert batch exceeds the 10000-record limit".to_owned(),
            });
        }
        for record in &records {
            if record.id.trim().is_empty() || record.id.contains('\0') {
                return Err(StoreError::InvalidIdentifier {
                    identifier: "vector collection record id",
                });
            }
            if record
                .metadata
                .get("namespace")
                .is_some_and(|value| value.as_str() != Some(namespace))
            {
                return Err(StoreError::Backend {
                    backend: "postgres-vector-collection".to_owned(),
                    message: "record metadata namespace conflicts with collection namespace"
                        .to_owned(),
                });
            }
            let embedding = record
                .embedding
                .as_deref()
                .ok_or_else(|| StoreError::Backend {
                    backend: "postgres-vector-collection".to_owned(),
                    message: "vector collection records require embeddings".to_owned(),
                })?;
            finite_vector(embedding).map_err(trait_error)?;
        }
        if records.is_empty() {
            return Ok(());
        }

        let mut client = self.client().await.map_err(trait_error)?;
        let transaction = client
            .transaction()
            .await
            .map_err(backend)
            .map_err(trait_error)?;
        for record in records {
            let metadata =
                serde_json::to_string(&record.metadata).map_err(|error| StoreError::Backend {
                    backend: "postgres-vector-collection".to_owned(),
                    message: error.to_string(),
                })?;
            let embedding = vector_literal(
                record
                    .embedding
                    .as_deref()
                    .expect("embeddings validated before starting transaction"),
            );
            transaction
                .execute(
                    &format!(
                    "INSERT INTO {} (namespace,collection,id,document,metadata,embedding) \
                         VALUES ($1,$2,$3,$4,$5::text::jsonb,'{}'::vector) \
                         ON CONFLICT(namespace,collection,id) DO UPDATE SET \
                         document=EXCLUDED.document,metadata=EXCLUDED.metadata,embedding=EXCLUDED.embedding,updated_at=NOW()",
                        self.tables.vector_collection_records,
                        embedding
                    ),
                    &[
                        &namespace,
                        &collection,
                        &record.id,
                        &record.document,
                        &metadata,
                    ],
                )
                .await
                .map_err(backend)
                .map_err(trait_error)?;
        }
        transaction
            .commit()
            .await
            .map_err(backend)
            .map_err(trait_error)
    }

    async fn delete_vector_collection_records(
        &self,
        namespace: &str,
        collection: &str,
        ids: Vec<String>,
    ) -> StoreResult<()> {
        validate_collection_inputs(namespace, collection)?;
        if ids.len() > 10_000
            || ids
                .iter()
                .any(|id| id.trim().is_empty() || id.contains('\0'))
        {
            return Err(StoreError::Backend {
                backend: "postgres-vector-collection".to_owned(),
                message: "delete IDs must be non-empty and the batch cannot exceed 10000 records"
                    .to_owned(),
            });
        }
        if ids.is_empty() {
            return Ok(());
        }
        let client = self.client().await.map_err(trait_error)?;
        client
            .execute(
                &format!(
                    "DELETE FROM {} WHERE namespace=$1 AND collection=$2 AND id=ANY($3::text[])",
                    self.tables.vector_collection_records
                ),
                &[&namespace, &collection, &ids],
            )
            .await
            .map(|_| ())
            .map_err(backend)
            .map_err(trait_error)
    }

    async fn query_vector_collection(
        &self,
        namespace: &str,
        collection: &str,
        query: &VectorQuery,
    ) -> StoreResult<Vec<VectorMatch>> {
        validate_collection_inputs(namespace, collection)?;
        if query.limit > 10_000 {
            return Err(StoreError::Backend {
                backend: "postgres-vector-collection".to_owned(),
                message: "query limit exceeds the 10000-record limit".to_owned(),
            });
        }
        finite_vector(&query.embedding).map_err(trait_error)?;
        if query.limit == 0 {
            return Ok(Vec::new());
        }
        let metadata_filter =
            serde_json::to_string(&query.metadata.equals).map_err(|error| StoreError::Backend {
                backend: "postgres-vector-collection".to_owned(),
                message: error.to_string(),
            })?;
        let embedding = vector_literal(&query.embedding);
        let operator = match query.metric {
            kogwistar_store::DistanceMetric::Cosine => "<=>",
            kogwistar_store::DistanceMetric::L2 => "<->",
            kogwistar_store::DistanceMetric::InnerProduct => "<#>",
        };
        let client = self.client().await.map_err(trait_error)?;
        let rows = client
            .query(
                &format!(
                    "SELECT id,document,metadata::text,embedding::text,embedding {operator} '{}'::vector AS distance \
                     FROM {} WHERE namespace=$1 AND collection=$2 AND metadata @> $3::text::jsonb \
                     ORDER BY distance ASC,id ASC LIMIT $4",
                    embedding,
                    self.tables.vector_collection_records,
                ),
                &[
                    &namespace,
                    &collection,
                    &metadata_filter,
                    &i64::try_from(query.limit).unwrap_or(i64::MAX),
                ],
            )
            .await
            .map_err(backend)
            .map_err(trait_error)?;
        rows.iter()
            .map(|row| {
                let metadata_json: String = row.try_get(2).map_err(backend).map_err(trait_error)?;
                let metadata =
                    serde_json::from_str(&metadata_json).map_err(|error| StoreError::Backend {
                        backend: "postgres-vector-collection".to_owned(),
                        message: error.to_string(),
                    })?;
                let embedding_json: Option<String> =
                    row.try_get(3).map_err(backend).map_err(trait_error)?;
                let embedding = embedding_json
                    .as_deref()
                    .map(serde_json::from_str)
                    .transpose()
                    .map_err(|error| StoreError::Backend {
                        backend: "postgres-vector-collection".to_owned(),
                        message: error.to_string(),
                    })?;
                Ok(VectorMatch {
                    record: GraphRecord {
                        id: row.try_get(0).map_err(backend).map_err(trait_error)?,
                        document: row.try_get(1).map_err(backend).map_err(trait_error)?,
                        metadata,
                        embedding,
                    },
                    distance: row.try_get(4).map_err(backend).map_err(trait_error)?,
                })
            })
            .collect()
    }
}
