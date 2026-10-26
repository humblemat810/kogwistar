use kogwistar_store::{
    DistanceMetric, GraphReadStore, GraphRecord, GraphWriteStore, MetadataFilter, StoreError,
    StoreResult, VectorMatch, VectorQuery,
};
use reqwest::Method;
use serde_json::{Value, json};

use crate::{client::ChromaHttpStore, records};

#[derive(Clone, Debug, thiserror::Error)]
pub enum ChromaStoreError {
    #[error("invalid Chroma configuration: {0}")]
    Configuration(String),
    #[error("Chroma backend error: {0}")]
    Backend(String),
}

#[derive(Clone, Debug)]
pub struct ChromaGraphStore {
    http: ChromaHttpStore,
    metric: DistanceMetric,
}

impl ChromaGraphStore {
    pub async fn connect(config: crate::ChromaStoreConfig) -> Result<Self, ChromaStoreError> {
        let http = ChromaHttpStore::connect(&config).await?;
        Ok(Self {
            http,
            metric: config.metric,
        })
    }

    fn backend_error(error: impl std::fmt::Display) -> StoreError {
        StoreError::Backend {
            backend: "chroma".to_owned(),
            message: error.to_string(),
        }
    }

    async fn get_page(
        &self,
        namespace: &str,
        filter: &MetadataFilter,
        offset: u32,
        limit: u32,
    ) -> StoreResult<Vec<GraphRecord>> {
        let url = self
            .http
            .collection_url("get")
            .map_err(Self::backend_error)?;
        let response = self
            .http
            .request(
                Method::POST,
                url,
                Some(json!({
                    "where": records::query_where(namespace, filter),
                    "limit": limit,
                    "offset": offset,
                    "include": ["documents", "metadatas", "embeddings"]
                })),
            )
            .await
            .map_err(Self::backend_error)?;
        let payload: Value = response.json().await.map_err(Self::backend_error)?;
        decode_records(&payload, false)
    }
}

impl GraphReadStore for ChromaGraphStore {
    async fn graph_record(&self, namespace: &str, id: &str) -> StoreResult<Option<GraphRecord>> {
        records::validate_namespace(namespace)?;
        let url = self
            .http
            .collection_url("get")
            .map_err(Self::backend_error)?;
        let response = self
            .http
            .request(
                Method::POST,
                url,
                Some(json!({
                    "ids": [id],
                    "where": records::query_where(namespace, &MetadataFilter::default()),
                    "limit": 1,
                    "include": ["documents", "metadatas", "embeddings"]
                })),
            )
            .await
            .map_err(Self::backend_error)?;
        let payload: Value = response.json().await.map_err(Self::backend_error)?;
        Ok(decode_records(&payload, false)?.into_iter().next())
    }

    async fn graph_records(
        &self,
        namespace: &str,
        filter: &MetadataFilter,
    ) -> StoreResult<Vec<GraphRecord>> {
        records::validate_namespace(namespace)?;
        let mut result = Vec::new();
        let mut offset = 0_u32;
        loop {
            let page = self.get_page(namespace, filter, offset, 500).await?;
            let count = page.len();
            result.extend(page);
            if count < 500 {
                break;
            }
            offset = offset
                .checked_add(500)
                .ok_or_else(|| Self::backend_error("Chroma offset overflow"))?;
        }
        Ok(result)
    }

    async fn vector_query(
        &self,
        namespace: &str,
        query: &VectorQuery,
    ) -> StoreResult<Vec<VectorMatch>> {
        records::validate_namespace(namespace)?;
        if query.metric != self.metric {
            return Err(Self::backend_error(
                "query metric differs from the collection's configured metric",
            ));
        }
        if query.limit == 0 {
            return Ok(Vec::new());
        }
        if query.embedding.iter().any(|value| !value.is_finite()) {
            return Err(StoreError::NonFiniteVector);
        }
        let url = self
            .http
            .collection_url("query")
            .map_err(Self::backend_error)?;
        let response = self
            .http
            .request(
                Method::POST,
                url,
                Some(json!({
                    "query_embeddings": [query.embedding],
                    "n_results": u32::try_from(query.limit).unwrap_or(u32::MAX),
                    "where": records::query_where(namespace, &query.metadata),
                    "include": ["documents", "metadatas", "embeddings", "distances"]
                })),
            )
            .await
            .map_err(Self::backend_error)?;
        let payload: Value = response.json().await.map_err(Self::backend_error)?;
        let mut matches = decode_matches(&payload)?;
        if query.metric == DistanceMetric::InnerProduct {
            for item in &mut matches {
                item.distance -= 1.0;
            }
        } else if query.metric == DistanceMetric::L2 {
            for item in &mut matches {
                item.distance = item.distance.max(0.0).sqrt();
            }
        }
        Ok(matches)
    }
}

impl GraphWriteStore for ChromaGraphStore {
    async fn upsert_graph_record(&self, namespace: &str, record: GraphRecord) -> StoreResult<()> {
        records::validate_namespace(namespace)?;
        if record.id.is_empty() {
            return Err(StoreError::EmptyRecordId);
        }
        if let Some(vector) = &record.embedding {
            if vector.iter().any(|value| !value.is_finite()) {
                return Err(StoreError::NonFiniteVector);
            }
        }
        let metadata = records::chroma_metadata(namespace, &record)?;
        let url = self
            .http
            .collection_url("upsert")
            .map_err(Self::backend_error)?;
        self.http
            .request(
                Method::POST,
                url,
                Some(json!({
                    "ids": [record.id],
                    "embeddings": record.embedding.map(|embedding| vec![embedding]),
                    "documents": [record.document],
                    "metadatas": [metadata]
                })),
            )
            .await
            .map_err(Self::backend_error)?;
        Ok(())
    }

    async fn delete_graph_record(&self, namespace: &str, id: &str) -> StoreResult<bool> {
        records::validate_namespace(namespace)?;
        let before = self.graph_record(namespace, id).await?.is_some();
        if !before {
            return Ok(false);
        }
        let url = self
            .http
            .collection_url("delete")
            .map_err(Self::backend_error)?;
        self.http
            .request(
                Method::POST,
                url,
                Some(json!({
                    "ids": [id],
                    "where": records::query_where(namespace, &MetadataFilter::default())
                })),
            )
            .await
            .map_err(Self::backend_error)?;
        Ok(true)
    }
}

fn decode_records(payload: &Value, nested: bool) -> StoreResult<Vec<GraphRecord>> {
    let ids = field_array(payload, "ids", nested);
    let documents = field_array(payload, "documents", nested);
    let metadatas = field_array(payload, "metadatas", nested);
    let embeddings = field_array(payload, "embeddings", nested);
    ids.iter()
        .enumerate()
        .map(|(index, id)| {
            let id = id.as_str().ok_or_else(|| {
                ChromaGraphStore::backend_error("Chroma returned a non-string id")
            })?;
            let document = documents
                .get(index)
                .and_then(Value::as_str)
                .map(str::to_owned);
            let metadata = metadatas
                .get(index)
                .filter(|value| !value.is_null())
                .cloned();
            let embedding = embeddings
                .get(index)
                .and_then(Value::as_array)
                .map(|values| {
                    values
                        .iter()
                        .map(|value| {
                            value.as_f64().map(|value| value as f32).ok_or_else(|| {
                                ChromaGraphStore::backend_error(
                                    "Chroma returned an invalid embedding component",
                                )
                            })
                        })
                        .collect::<StoreResult<Vec<_>>>()
                })
                .transpose()?;
            records::restore_record(id.to_owned(), document, metadata, embedding)
                .map_err(Into::into)
        })
        .collect()
}

fn decode_matches(payload: &Value) -> StoreResult<Vec<VectorMatch>> {
    let records = decode_records(payload, true)?;
    let distances = field_array(payload, "distances", true);
    records
        .into_iter()
        .enumerate()
        .map(|(index, record)| {
            let distance = distances
                .get(index)
                .and_then(Value::as_f64)
                .ok_or_else(|| {
                    ChromaGraphStore::backend_error(
                        "Chroma omitted a numeric distance for a vector result",
                    )
                })?;
            Ok(VectorMatch { record, distance })
        })
        .collect()
}

fn field_array(payload: &Value, field: &str, nested: bool) -> Vec<Value> {
    let values = payload
        .get(field)
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if nested {
        values
            .first()
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default()
    } else {
        values
    }
}
