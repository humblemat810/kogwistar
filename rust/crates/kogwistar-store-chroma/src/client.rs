use kogwistar_store::DistanceMetric;
use reqwest::{
    Client, Method, Response, Url,
    header::{HeaderMap, HeaderName, HeaderValue},
};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::time::Duration;

use crate::store::ChromaStoreError;

#[derive(Clone, Debug)]
pub struct ChromaStoreConfig {
    pub endpoint: String,
    pub tenant: Option<String>,
    pub database: Option<String>,
    pub collection: String,
    pub metric: DistanceMetric,
    pub request_timeout: Duration,
    pub headers: BTreeMap<String, String>,
}

impl Default for ChromaStoreConfig {
    fn default() -> Self {
        Self {
            endpoint: "http://localhost:8000".to_owned(),
            tenant: Some("default_tenant".to_owned()),
            database: Some("default_database".to_owned()),
            collection: "kogwistar_graph_records".to_owned(),
            metric: DistanceMetric::Cosine,
            request_timeout: Duration::from_secs(30),
            headers: BTreeMap::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ChromaHttpStore {
    client: Client,
    endpoint: Url,
    tenant: String,
    database: String,
    collection_id: String,
}

impl ChromaHttpStore {
    pub(crate) async fn connect(config: &ChromaStoreConfig) -> Result<Self, ChromaStoreError> {
        let endpoint = Url::parse(&config.endpoint)
            .map_err(|error| ChromaStoreError::Configuration(error.to_string()))?;
        let tenant = config
            .tenant
            .as_deref()
            .unwrap_or("default_tenant")
            .to_owned();
        let database = config
            .database
            .as_deref()
            .unwrap_or("default_database")
            .to_owned();
        let mut headers = HeaderMap::new();
        for (name, value) in &config.headers {
            let name = HeaderName::from_bytes(name.as_bytes())
                .map_err(|error| ChromaStoreError::Configuration(error.to_string()))?;
            let value = HeaderValue::from_str(value)
                .map_err(|error| ChromaStoreError::Configuration(error.to_string()))?;
            headers.insert(name, value);
        }
        let client = Client::builder()
            .timeout(config.request_timeout)
            .default_headers(headers)
            .build()
            .map_err(|error| ChromaStoreError::Configuration(error.to_string()))?;
        let mut store = Self {
            client,
            endpoint,
            tenant,
            database,
            collection_id: String::new(),
        };
        let url = store.api_url(&[
            "tenants",
            &store.tenant,
            "databases",
            &store.database,
            "collections",
        ])?;
        let response = store
            .request(
                Method::POST,
                url,
                Some(json!({
                    "name": config.collection,
                    "get_or_create": true,
                    "metadata": {
                        "hnsw:space": metric_name(config.metric),
                        "kogwistar:contract": "graph-record-v1"
                    }
                })),
            )
            .await?;
        let body: Value = response.json().await.map_err(backend_error)?;
        let expected_metric = metric_name(config.metric);
        let actual_metric = body
            .get("metadata")
            .and_then(|metadata| metadata.get("hnsw:space"))
            .and_then(Value::as_str);
        if actual_metric != Some(expected_metric) {
            return Err(ChromaStoreError::Configuration(format!(
                "collection metric is {actual_metric:?}; configured metric is {expected_metric:?}"
            )));
        }
        store.collection_id = body
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| ChromaStoreError::Backend("collection response omitted id".to_owned()))?
            .to_owned();
        Ok(store)
    }

    pub(crate) async fn request(
        &self,
        method: Method,
        url: Url,
        body: Option<Value>,
    ) -> Result<Response, ChromaStoreError> {
        let mut request = self.client.request(method, url);
        if let Some(body) = body {
            request = request.json(&body);
        }
        let response = request.send().await.map_err(backend_error)?;
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(ChromaStoreError::Backend(format!(
                "Chroma returned HTTP {status}: {}",
                body.chars().take(512).collect::<String>()
            )));
        }
        Ok(response)
    }

    pub(crate) fn collection_url(&self, operation: &str) -> Result<Url, ChromaStoreError> {
        self.api_url(&[
            "tenants",
            &self.tenant,
            "databases",
            &self.database,
            "collections",
            &self.collection_id,
            operation,
        ])
    }

    fn api_url(&self, segments: &[&str]) -> Result<Url, ChromaStoreError> {
        let mut url = self.endpoint.clone();
        {
            let mut path = url.path_segments_mut().map_err(|_| {
                ChromaStoreError::Configuration("endpoint cannot accept path segments".to_owned())
            })?;
            path.pop_if_empty();
            path.extend(["api", "v2"]);
            path.extend(segments.iter().copied());
        }
        Ok(url)
    }
}

fn metric_name(metric: DistanceMetric) -> &'static str {
    match metric {
        DistanceMetric::Cosine => "cosine",
        DistanceMetric::L2 => "l2",
        DistanceMetric::InnerProduct => "ip",
    }
}

fn backend_error(error: reqwest::Error) -> ChromaStoreError {
    ChromaStoreError::Backend(error.to_string())
}
