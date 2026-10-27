use std::collections::BTreeSet;

use kogwistar_store::{
    DistanceMetric, GraphEntityKind, GraphReadStore, GraphRecord, GraphWriteStore, MetadataFilter,
    StoreError, StoreResult, VectorMatch, VectorQuery,
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
    collections: ChromaCollections,
    metric: DistanceMetric,
}

#[derive(Clone, Debug)]
enum ChromaCollections {
    Single(ChromaHttpStore),
    ByKind {
        nodes: ChromaHttpStore,
        edges: ChromaHttpStore,
    },
}

impl ChromaGraphStore {
    pub async fn connect(config: crate::ChromaStoreConfig) -> Result<Self, ChromaStoreError> {
        let http = ChromaHttpStore::connect(&config).await?;
        Ok(Self {
            collections: ChromaCollections::Single(http),
            metric: config.metric,
        })
    }

    /// Connect separate Chroma collections for node and edge identity.
    ///
    /// The legacy `connect` constructor keeps its single-collection behavior;
    /// kind-aware graph operations are available only through this constructor.
    pub async fn connect_entity_collections(
        node_config: crate::ChromaStoreConfig,
        edge_config: crate::ChromaStoreConfig,
    ) -> Result<Self, ChromaStoreError> {
        if node_config.endpoint != edge_config.endpoint
            || node_config.tenant != edge_config.tenant
            || node_config.database != edge_config.database
            || node_config.metric != edge_config.metric
            || node_config.request_timeout != edge_config.request_timeout
            || node_config.headers != edge_config.headers
        {
            return Err(ChromaStoreError::Configuration(
                "node and edge collections must share endpoint, tenant, database, metric, timeout, and headers".to_owned(),
            ));
        }
        if node_config.collection == edge_config.collection {
            return Err(ChromaStoreError::Configuration(
                "node and edge collections must have distinct names".to_owned(),
            ));
        }
        let metric = node_config.metric;
        let (nodes, edges) = tokio::try_join!(
            ChromaHttpStore::connect(&node_config),
            ChromaHttpStore::connect(&edge_config)
        )?;
        Ok(Self {
            collections: ChromaCollections::ByKind { nodes, edges },
            metric,
        })
    }

    fn legacy_http(&self) -> StoreResult<&ChromaHttpStore> {
        match &self.collections {
            ChromaCollections::Single(http) => Ok(http),
            ChromaCollections::ByKind { .. } => Err(Self::backend_error(
                "this store uses separate node/edge collections; specify an entity kind",
            )),
        }
    }

    fn http_for_kind(&self, kind: GraphEntityKind) -> StoreResult<&ChromaHttpStore> {
        match &self.collections {
            ChromaCollections::Single(_) => Err(Self::backend_error(
                "kind-aware operations require separate node and edge collections",
            )),
            ChromaCollections::ByKind { nodes, edges } => Ok(match kind {
                GraphEntityKind::Node => nodes,
                GraphEntityKind::Edge => edges,
            }),
        }
    }

    fn backend_error(error: impl std::fmt::Display) -> StoreError {
        StoreError::Backend {
            backend: "chroma".to_owned(),
            message: error.to_string(),
        }
    }

    async fn get_page(
        http: &ChromaHttpStore,
        namespace: &str,
        filter: &MetadataFilter,
        offset: u32,
        limit: u32,
    ) -> StoreResult<Vec<GraphRecord>> {
        let url = http.collection_url("get").map_err(Self::backend_error)?;
        let response = http
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
        let http = self.legacy_http()?;
        let url = http.collection_url("get").map_err(Self::backend_error)?;
        let response = http
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

    async fn graph_record_of_kind(
        &self,
        namespace: &str,
        kind: GraphEntityKind,
        id: &str,
    ) -> StoreResult<Option<GraphRecord>> {
        records::validate_namespace(namespace)?;
        let http = self.http_for_kind(kind)?;
        let url = http.collection_url("get").map_err(Self::backend_error)?;
        let response = http
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

    async fn graph_records_by_ids_of_kind(
        &self,
        namespace: &str,
        kind: GraphEntityKind,
        ids: &[String],
    ) -> StoreResult<Vec<GraphRecord>> {
        self.graph_records_by_ids_in_collection(namespace, kind, ids)
            .await
    }

    async fn graph_records(
        &self,
        namespace: &str,
        filter: &MetadataFilter,
    ) -> StoreResult<Vec<GraphRecord>> {
        records::validate_namespace(namespace)?;
        let http = self.legacy_http()?;
        let mut result = Vec::new();
        let mut offset = 0_u32;
        loop {
            let page = Self::get_page(http, namespace, filter, offset, 500).await?;
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

    async fn graph_records_of_kind(
        &self,
        namespace: &str,
        kind: GraphEntityKind,
        filter: &MetadataFilter,
    ) -> StoreResult<Vec<GraphRecord>> {
        records::validate_namespace(namespace)?;
        let http = self.http_for_kind(kind)?;
        let mut result = Vec::new();
        let mut offset = 0_u32;
        loop {
            let page = Self::get_page(http, namespace, filter, offset, 500).await?;
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
        let http = self.legacy_http()?;
        self.vector_query_in_collection(http, namespace, query)
            .await
    }

    async fn vector_query_of_kind(
        &self,
        namespace: &str,
        kind: GraphEntityKind,
        query: &VectorQuery,
    ) -> StoreResult<Vec<VectorMatch>> {
        records::validate_namespace(namespace)?;
        let http = self.http_for_kind(kind)?;
        self.vector_query_in_collection(http, namespace, query)
            .await
    }
}

impl ChromaGraphStore {
    async fn graph_records_by_ids_in_collection(
        &self,
        namespace: &str,
        kind: GraphEntityKind,
        ids: &[String],
    ) -> StoreResult<Vec<GraphRecord>> {
        records::validate_namespace(namespace)?;
        if ids.len() > 10_000 {
            return Err(Self::backend_error(
                "typed ID batch exceeds the 10000-record limit",
            ));
        }
        if ids.iter().any(|id| id.trim().is_empty()) {
            return Err(Self::backend_error("typed ID batch contains an empty ID"));
        }
        let mut seen = BTreeSet::new();
        let unique_ids = ids
            .iter()
            .filter(|id| seen.insert(id.as_str()))
            .cloned()
            .collect::<Vec<_>>();
        if unique_ids.is_empty() {
            return Ok(Vec::new());
        }
        let http = self.http_for_kind(kind)?;
        let url = http.collection_url("get").map_err(Self::backend_error)?;
        let response = http
            .request(
                Method::POST,
                url,
                Some(json!({
                    "ids": &unique_ids,
                    "where": records::query_where(namespace, &MetadataFilter::default()),
                    "include": ["documents", "metadatas", "embeddings"]
                })),
            )
            .await
            .map_err(Self::backend_error)?;
        let payload: Value = response.json().await.map_err(Self::backend_error)?;
        let mut records = decode_records(&payload, false)?
            .into_iter()
            .map(|record| (record.id.clone(), record))
            .collect::<std::collections::BTreeMap<_, _>>();
        Ok(unique_ids
            .iter()
            .filter_map(|id| records.remove(id))
            .collect())
    }

    /// Read a bounded entity-kind snapshot for callers that must apply
    /// cross-record policy such as endpoint ACL checks before returning data.
    /// The method fails rather than returning a partial snapshot when the cap
    /// is exceeded.
    pub async fn graph_records_of_kind_bounded(
        &self,
        namespace: &str,
        kind: GraphEntityKind,
        filter: &MetadataFilter,
        max_records: usize,
    ) -> StoreResult<Vec<GraphRecord>> {
        records::validate_namespace(namespace)?;
        if max_records == 0 {
            return Ok(Vec::new());
        }
        let http = self.http_for_kind(kind)?;
        let mut result = Vec::new();
        let mut offset = 0_u32;
        loop {
            let remaining_plus_one = max_records.saturating_sub(result.len()).saturating_add(1);
            let limit = u32::try_from(remaining_plus_one.min(500)).unwrap_or(500);
            let page = Self::get_page(http, namespace, filter, offset, limit).await?;
            let count = page.len();
            if result.len().saturating_add(count) > max_records {
                return Err(Self::backend_error(format!(
                    "kind-aware graph snapshot exceeds the {max_records}-record limit"
                )));
            }
            result.extend(page);
            if count < limit as usize {
                break;
            }
            offset = offset
                .checked_add(limit)
                .ok_or_else(|| Self::backend_error("Chroma offset overflow"))?;
        }
        Ok(result)
    }

    async fn vector_query_in_collection(
        &self,
        http: &ChromaHttpStore,
        namespace: &str,
        query: &VectorQuery,
    ) -> StoreResult<Vec<VectorMatch>> {
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
        let url = http.collection_url("query").map_err(Self::backend_error)?;
        let response = http
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
        for item in &mut matches {
            item.distance = normalize_chroma_distance(query.metric, item.distance);
        }
        Ok(matches)
    }
}

impl GraphWriteStore for ChromaGraphStore {
    async fn upsert_graph_record(&self, namespace: &str, record: GraphRecord) -> StoreResult<()> {
        let http = self.legacy_http()?;
        self.upsert_in_collection(http, namespace, record).await
    }

    async fn upsert_graph_record_of_kind(
        &self,
        namespace: &str,
        kind: GraphEntityKind,
        record: GraphRecord,
    ) -> StoreResult<()> {
        let http = self.http_for_kind(kind)?;
        self.upsert_in_collection(http, namespace, record).await
    }

    async fn delete_graph_record_of_kind(
        &self,
        namespace: &str,
        kind: GraphEntityKind,
        id: &str,
    ) -> StoreResult<bool> {
        records::validate_namespace(namespace)?;
        let http = self.http_for_kind(kind)?;
        self.delete_in_collection(http, namespace, id).await
    }

    async fn delete_graph_record(&self, namespace: &str, id: &str) -> StoreResult<bool> {
        let http = self.legacy_http()?;
        self.delete_in_collection(http, namespace, id).await
    }
}

impl ChromaGraphStore {
    async fn upsert_in_collection(
        &self,
        http: &ChromaHttpStore,
        namespace: &str,
        record: GraphRecord,
    ) -> StoreResult<()> {
        records::validate_namespace(namespace)?;
        if record.id.is_empty() {
            return Err(StoreError::EmptyRecordId);
        }
        if let Some(vector) = &record.embedding
            && vector.iter().any(|value| !value.is_finite())
        {
            return Err(StoreError::NonFiniteVector);
        }
        let metadata = records::chroma_metadata(namespace, &record)?;
        let url = http.collection_url("upsert").map_err(Self::backend_error)?;
        http.request(
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

    async fn delete_in_collection(
        &self,
        http: &ChromaHttpStore,
        namespace: &str,
        id: &str,
    ) -> StoreResult<bool> {
        records::validate_namespace(namespace)?;
        let url = http.collection_url("get").map_err(Self::backend_error)?;
        let response = http
            .request(
                Method::POST,
                url,
                Some(json!({
                    "ids": [id],
                    "where": records::query_where(namespace, &MetadataFilter::default()),
                    "limit": 1,
                    "include": ["metadatas"]
                })),
            )
            .await
            .map_err(Self::backend_error)?;
        let payload: Value = response.json().await.map_err(Self::backend_error)?;
        let before = !decode_records(&payload, false)?.is_empty();
        if !before {
            return Ok(false);
        }
        let url = http.collection_url("delete").map_err(Self::backend_error)?;
        http.request(
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

fn normalize_chroma_distance(metric: DistanceMetric, distance: f64) -> f64 {
    match metric {
        DistanceMetric::Cosine => distance,
        // Chroma reports 1 - inner_product; Kogwistar orders by negative dot product.
        DistanceMetric::InnerProduct => distance - 1.0,
        // Chroma reports squared L2 while Kogwistar's public metric is Euclidean L2.
        DistanceMetric::L2 => distance.max(0.0).sqrt(),
    }
}

#[cfg(test)]
mod tests {
    use super::normalize_chroma_distance;
    use kogwistar_store::{DistanceMetric, similarity_from_distance};

    #[test]
    fn chroma_distances_match_kogwistar_metric_conventions() {
        assert_eq!(
            normalize_chroma_distance(DistanceMetric::Cosine, 0.25),
            0.25
        );
        assert_eq!(
            normalize_chroma_distance(DistanceMetric::InnerProduct, 0.2),
            -0.8
        );
        assert_eq!(normalize_chroma_distance(DistanceMetric::L2, 9.0), 3.0);
        assert_eq!(normalize_chroma_distance(DistanceMetric::L2, -1.0), 0.0);
    }

    #[test]
    fn chroma_distance_semantics_match_shared_python_contract() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/chroma-vector-distance-semantics-v1.json"
        )))
        .expect("shared Chroma vector distance contract");
        assert_eq!(fixture["schema_version"], 1);
        for case in fixture["cases"].as_array().expect("cases") {
            let metric = match case["metric"].as_str() {
                Some("cosine") => DistanceMetric::Cosine,
                Some("ip") => DistanceMetric::InnerProduct,
                Some("l2") => DistanceMetric::L2,
                other => panic!("unknown metric in shared fixture: {other:?}"),
            };
            let normalized = normalize_chroma_distance(
                metric,
                case["chroma_distance"].as_f64().expect("Chroma distance"),
            );
            assert_eq!(
                normalized,
                case["kogwistar_distance"]
                    .as_f64()
                    .expect("normalized distance")
            );
            assert_eq!(
                similarity_from_distance(Some(normalized), metric),
                Some(case["similarity"].as_f64().expect("similarity"))
            );
        }
    }
}
