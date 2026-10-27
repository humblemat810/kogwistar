//! Rust embedding providers for the shared Kogwistar embedding profile.
//!
//! Storage and query code consume normalized vectors; provider identity and
//! dimensions are kept with the semantic profile so a model switch cannot
//! silently mix incompatible vector spaces.

use reqwest::{Client, Url};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::future::Future;
use std::pin::Pin;
use std::time::Duration;
use thiserror::Error;

pub type EmbeddingFuture<'a> =
    Pin<Box<dyn Future<Output = Result<Vec<Vec<f64>>, EmbeddingError>> + Send + 'a>>;

/// Rust projection of Python's `engine_core.embedding_profile.EmbeddingProfile`.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct EmbeddingProfile {
    pub provider: String,
    pub model: String,
    pub dimension: usize,
    #[serde(default = "default_similarity_metric")]
    pub similarity_metric: String,
    #[serde(default)]
    pub endpoint_fingerprint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_sequence_length: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub crop_token_budget: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenizer_fingerprint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub crop_policy: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_kind: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_revision: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preprocessing_fingerprint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_image_patches: Option<usize>,
}

fn default_similarity_metric() -> String {
    "cosine".to_owned()
}

impl EmbeddingProfile {
    pub fn validate(&self) -> Result<(), EmbeddingError> {
        if self.provider.trim().is_empty() || self.model.trim().is_empty() {
            return Err(EmbeddingError::InvalidProfile(
                "provider and model must not be empty".to_owned(),
            ));
        }
        if self.dimension == 0 {
            return Err(EmbeddingError::InvalidProfile(
                "dimension must be positive".to_owned(),
            ));
        }
        if !matches!(
            self.similarity_metric.to_ascii_lowercase().as_str(),
            "cosine" | "l2" | "ip"
        ) {
            return Err(EmbeddingError::InvalidProfile(
                "similarity_metric must be cosine, l2, or ip".to_owned(),
            ));
        }
        if let (Some(max), Some(crop)) = (self.max_sequence_length, self.crop_token_budget)
            && crop > max
        {
            return Err(EmbeddingError::InvalidProfile(
                "crop_token_budget cannot exceed max_sequence_length".to_owned(),
            ));
        }
        if self.embedding_kind.as_deref().is_some_and(|kind| {
            !matches!(
                kind.to_ascii_lowercase().as_str(),
                "single_vector" | "dense" | "late_interaction"
            )
        }) {
            return Err(EmbeddingError::InvalidProfile(
                "unsupported embedding_kind".to_owned(),
            ));
        }
        if self.max_sequence_length == Some(0)
            || self.crop_token_budget == Some(0)
            || self.max_image_patches == Some(0)
        {
            return Err(EmbeddingError::InvalidProfile(
                "sequence, crop, and image-patch limits must be positive".to_owned(),
            ));
        }
        if self
            .crop_policy
            .as_deref()
            .is_some_and(|value| value.trim().is_empty())
            || self
                .model_revision
                .as_deref()
                .is_some_and(|value| value.trim().is_empty())
            || self
                .preprocessing_fingerprint
                .as_deref()
                .is_some_and(|value| value.trim().is_empty())
        {
            return Err(EmbeddingError::InvalidProfile(
                "optional profile strings must not be empty".to_owned(),
            ));
        }
        Ok(())
    }

    /// Return the same normalized and sparse JSON projection as Python's
    /// `EmbeddingProfile.as_dict()`.
    pub fn as_value(&self) -> Value {
        let mut value = Map::new();
        value.insert(
            "provider".to_owned(),
            Value::String(self.provider.trim().to_ascii_lowercase()),
        );
        value.insert(
            "model".to_owned(),
            Value::String(self.model.trim().to_owned()),
        );
        value.insert("dimension".to_owned(), Value::from(self.dimension));
        value.insert(
            "similarity_metric".to_owned(),
            Value::String(self.similarity_metric.trim().to_ascii_lowercase()),
        );
        value.insert(
            "endpoint_fingerprint".to_owned(),
            self.endpoint_fingerprint
                .clone()
                .map(Value::String)
                .unwrap_or(Value::Null),
        );
        for (key, item) in [
            ("max_sequence_length", self.max_sequence_length),
            ("crop_token_budget", self.crop_token_budget),
            ("max_image_patches", self.max_image_patches),
        ] {
            if let Some(item) = item {
                value.insert(key.to_owned(), Value::from(item));
            }
        }
        for (key, item) in [
            (
                "tokenizer_fingerprint",
                self.tokenizer_fingerprint.as_deref(),
            ),
            ("crop_policy", self.crop_policy.as_deref()),
            ("embedding_kind", self.embedding_kind.as_deref()),
            ("model_revision", self.model_revision.as_deref()),
            (
                "preprocessing_fingerprint",
                self.preprocessing_fingerprint.as_deref(),
            ),
        ] {
            if let Some(item) = item {
                value.insert(key.to_owned(), Value::String(item.to_owned()));
            }
        }
        Value::Object(value)
    }

    /// Match Python's endpoint fingerprinting: exclude credentials, query, and
    /// fragment while retaining normalized scheme, host, port, and path.
    pub fn endpoint_fingerprint(endpoint: &str) -> Result<String, EmbeddingError> {
        let parsed = Url::parse(endpoint).map_err(|_| {
            EmbeddingError::InvalidEndpoint(
                "endpoint must be a valid absolute HTTP(S) URL".to_owned(),
            )
        })?;
        if !matches!(parsed.scheme(), "http" | "https") || parsed.host_str().is_none() {
            return Err(EmbeddingError::InvalidEndpoint(
                "endpoint must be an absolute HTTP(S) URL".to_owned(),
            ));
        }
        let host = parsed.host_str().unwrap_or_default().to_ascii_lowercase();
        let port = explicit_endpoint_port(endpoint)
            .or_else(|| parsed.port())
            .map(|port| format!(":{port}"))
            .unwrap_or_default();
        let path = parsed.path().trim_end_matches('/');
        let path = if path == "/" { "" } else { path };
        let normalized = format!(
            "{}://{}{}{}",
            parsed.scheme().to_ascii_lowercase(),
            host,
            port,
            path
        );
        Ok(hex::encode(Sha256::digest(normalized.as_bytes())))
    }
}

fn explicit_endpoint_port(endpoint: &str) -> Option<u16> {
    let authority = endpoint
        .split_once("://")?
        .1
        .split(['/', '?', '#'])
        .next()?;
    let host_and_port = authority
        .rsplit_once('@')
        .map_or(authority, |(_, host)| host);
    let port = if host_and_port.starts_with('[') {
        let bracket_end = host_and_port.find(']')?;
        host_and_port.get(bracket_end + 1..)?.strip_prefix(':')?
    } else {
        host_and_port.rsplit_once(':')?.1
    };
    port.parse().ok()
}

pub trait EmbeddingProvider: Send + Sync {
    fn profile(&self) -> &EmbeddingProfile;

    fn embed<'a>(&'a self, texts: &'a [String]) -> EmbeddingFuture<'a>;
}

#[derive(Debug, Error)]
pub enum EmbeddingError {
    #[error("invalid embedding profile: {0}")]
    InvalidProfile(String),
    #[error("invalid embedding endpoint: {0}")]
    InvalidEndpoint(String),
    #[error("embedding provider request failed: {0}")]
    Request(String),
    #[error("embedding provider returned an invalid response: {0}")]
    InvalidResponse(String),
}

#[derive(Clone)]
pub struct OllamaEmbeddingProvider {
    client: Client,
    endpoint: Url,
    profile: EmbeddingProfile,
}

impl OllamaEmbeddingProvider {
    pub fn new(
        profile: EmbeddingProfile,
        endpoint: &str,
        request_timeout: Duration,
    ) -> Result<Self, EmbeddingError> {
        profile.validate()?;
        if !profile.provider.eq_ignore_ascii_case("ollama") {
            return Err(EmbeddingError::InvalidProfile(
                "Ollama provider requires provider='ollama'".to_owned(),
            ));
        }
        let endpoint_fingerprint = EmbeddingProfile::endpoint_fingerprint(endpoint)?;
        if profile
            .endpoint_fingerprint
            .as_deref()
            .is_some_and(|expected| expected != endpoint_fingerprint)
        {
            return Err(EmbeddingError::InvalidProfile(
                "configured endpoint does not match embedding profile".to_owned(),
            ));
        }
        let mut endpoint = Url::parse(endpoint).map_err(|_| {
            EmbeddingError::InvalidEndpoint(
                "endpoint must be a valid absolute HTTP(S) URL".to_owned(),
            )
        })?;
        if !matches!(endpoint.scheme(), "http" | "https") || endpoint.host_str().is_none() {
            return Err(EmbeddingError::InvalidEndpoint(
                "endpoint must be an absolute HTTP(S) URL".to_owned(),
            ));
        }
        endpoint.set_query(None);
        endpoint.set_fragment(None);
        let base_path = endpoint.path().trim_end_matches('/');
        endpoint.set_path(&format!("{base_path}/api/embeddings"));
        let client = Client::builder()
            .timeout(request_timeout)
            .build()
            .map_err(|error| EmbeddingError::Request(error.to_string()))?;
        Ok(Self {
            client,
            endpoint,
            profile,
        })
    }
}

impl EmbeddingProvider for OllamaEmbeddingProvider {
    fn profile(&self) -> &EmbeddingProfile {
        &self.profile
    }

    fn embed<'a>(&'a self, texts: &'a [String]) -> EmbeddingFuture<'a> {
        Box::pin(async move {
            let mut embeddings = Vec::with_capacity(texts.len());
            for text in texts {
                let response = self
                    .client
                    .post(self.endpoint.clone())
                    .json(&json!({"model": self.profile.model, "prompt": text}))
                    .send()
                    .await
                    .map_err(|error| EmbeddingError::Request(error.without_url().to_string()))?
                    .error_for_status()
                    .map_err(|error| EmbeddingError::Request(error.without_url().to_string()))?;
                let payload: Value = response.json().await.map_err(|error| {
                    EmbeddingError::InvalidResponse(error.without_url().to_string())
                })?;
                let vector = payload
                    .get("embedding")
                    .and_then(Value::as_array)
                    .ok_or_else(|| {
                        EmbeddingError::InvalidResponse(
                            "response is missing embedding array".to_owned(),
                        )
                    })?
                    .iter()
                    .map(|value| {
                        value.as_f64().ok_or_else(|| {
                            EmbeddingError::InvalidResponse(
                                "embedding values must be finite numbers".to_owned(),
                            )
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                embeddings.push(vector);
            }
            normalize_embeddings(embeddings, texts.len(), self.profile.dimension)
        })
    }
}

pub fn normalize_embeddings(
    vectors: Vec<Vec<f64>>,
    expected_count: usize,
    expected_dimension: usize,
) -> Result<Vec<Vec<f64>>, EmbeddingError> {
    if vectors.len() != expected_count {
        return Err(EmbeddingError::InvalidResponse(format!(
            "provider returned {} vectors for {expected_count} inputs",
            vectors.len()
        )));
    }
    vectors
        .into_iter()
        .map(|vector| {
            if vector.len() != expected_dimension {
                return Err(EmbeddingError::InvalidResponse(format!(
                    "provider returned dimension {}, expected {expected_dimension}",
                    vector.len()
                )));
            }
            if vector.iter().any(|value| !value.is_finite()) {
                return Err(EmbeddingError::InvalidResponse(
                    "embedding vector contains a non-finite number".to_owned(),
                ));
            }
            let scale = vector
                .iter()
                .map(|value| value.abs())
                .fold(0.0_f64, f64::max);
            if scale == 0.0 {
                return Ok(vector);
            }
            let scaled = vector.iter().map(|value| value / scale).collect::<Vec<_>>();
            let norm = scaled.iter().map(|value| value * value).sum::<f64>().sqrt();
            Ok(scaled.into_iter().map(|value| value / norm).collect())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_matches_shared_python_embedding_profile_fixture() {
        let fixture = include_str!("../../../../contracts/golden/embedding-profile-v1.json");
        let profile: EmbeddingProfile = serde_json::from_str(fixture).unwrap();
        profile.validate().unwrap();
        assert_eq!(
            profile.as_value(),
            serde_json::from_str::<Value>(fixture).unwrap()
        );
        assert_eq!(profile.provider, "ollama");
        assert_eq!(profile.model, "all-minilm:l6-v2");
        assert_eq!(profile.dimension, 384);
        assert_eq!(profile.similarity_metric, "cosine");
    }

    #[test]
    fn endpoint_fingerprint_does_not_retain_credentials_or_query() {
        let first = EmbeddingProfile::endpoint_fingerprint(
            "https://user:secret@example.test:8443/embed?token=hidden#fragment",
        )
        .unwrap();
        let second =
            EmbeddingProfile::endpoint_fingerprint("https://example.test:8443/embed").unwrap();
        assert_eq!(first, second);
        assert_eq!(first.len(), 64);
    }

    #[test]
    fn endpoint_fingerprint_matches_python_for_empty_paths_and_explicit_default_ports() {
        assert_eq!(
            EmbeddingProfile::endpoint_fingerprint("https://example.test").unwrap(),
            "f1e2c7da767e6e0fc3eac0819fdd3a622967f18748c6f0a875c3a5b0a165031e"
        );
        assert_eq!(
            EmbeddingProfile::endpoint_fingerprint("https://example.test:443/").unwrap(),
            "74b69725587db261567c6f486dba5a9f7c7146005fcb79f91cbb73dd544e7b03"
        );
        assert_eq!(
            EmbeddingProfile::endpoint_fingerprint("http://example.test:80/a/").unwrap(),
            "10f49e37179dccb6263d6ee480908c0c0e0d5435b45cdb7667d8b60417858bd3"
        );
    }

    #[test]
    fn configured_provider_rejects_endpoint_outside_the_profiled_vector_space() {
        let profile = EmbeddingProfile {
            provider: "ollama".to_owned(),
            model: "model".to_owned(),
            dimension: 2,
            similarity_metric: "cosine".to_owned(),
            endpoint_fingerprint: Some(
                EmbeddingProfile::endpoint_fingerprint("http://old-endpoint:11434").unwrap(),
            ),
            max_sequence_length: None,
            crop_token_budget: None,
            tokenizer_fingerprint: None,
            crop_policy: None,
            embedding_kind: None,
            model_revision: None,
            preprocessing_fingerprint: None,
            max_image_patches: None,
        };
        assert!(matches!(
            OllamaEmbeddingProvider::new(
                profile,
                "http://new-endpoint:11434",
                Duration::from_secs(1)
            ),
            Err(EmbeddingError::InvalidProfile(_))
        ));
    }

    #[test]
    fn normalization_matches_python_zero_unit_and_scaled_vectors() {
        let result = normalize_embeddings(
            vec![vec![0.0, 0.0], vec![3.0, 4.0], vec![1.0e308, -1.0e308]],
            3,
            2,
        )
        .unwrap();
        assert_eq!(result[0], vec![0.0, 0.0]);
        assert_eq!(result[1], vec![0.6, 0.8]);
        let unit_component = 1.0 / 2.0_f64.sqrt();
        assert_eq!(result[2], vec![unit_component, -unit_component]);
    }

    #[test]
    fn normalization_rejects_provider_dimension_and_count_drift() {
        assert!(normalize_embeddings(vec![vec![1.0]], 1, 2).is_err());
        assert!(normalize_embeddings(vec![], 1, 2).is_err());
        assert!(normalize_embeddings(vec![vec![f64::INFINITY, 0.0]], 1, 2).is_err());
    }

    async fn read_request_body(stream: &mut tokio::net::TcpStream) -> Value {
        use tokio::io::AsyncReadExt;

        let mut bytes = Vec::new();
        let mut buffer = [0_u8; 1024];
        let (header_end, content_length) = loop {
            let count = stream.read(&mut buffer).await.unwrap();
            assert_ne!(count, 0, "client closed before sending the request body");
            bytes.extend_from_slice(&buffer[..count]);
            if let Some(header_end) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
                let header_text = String::from_utf8_lossy(&bytes[..header_end]);
                let content_length = header_text
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().unwrap())
                    })
                    .unwrap();
                break (header_end + 4, content_length);
            }
        };
        while bytes.len() < header_end + content_length {
            let count = stream.read(&mut buffer).await.unwrap();
            assert_ne!(count, 0, "client closed before sending the request body");
            bytes.extend_from_slice(&buffer[..count]);
        }
        serde_json::from_slice(&bytes[header_end..header_end + content_length]).unwrap()
    }

    #[tokio::test]
    async fn ollama_provider_uses_python_compatible_request_and_normalized_result() {
        use tokio::io::AsyncWriteExt;

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            for expected_prompt in ["three-four", "zero"] {
                let (mut stream, _) = listener.accept().await.unwrap();
                let request = read_request_body(&mut stream).await;
                assert_eq!(request["model"], "all-minilm:l6-v2");
                assert_eq!(request["prompt"], expected_prompt);
                let embedding = if expected_prompt == "zero" {
                    vec![0.0, 0.0]
                } else {
                    vec![3.0, 4.0]
                };
                let body = serde_json::to_vec(&json!({"embedding": embedding})).unwrap();
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                stream.write_all(response.as_bytes()).await.unwrap();
                stream.write_all(&body).await.unwrap();
            }
        });

        let profile = EmbeddingProfile {
            provider: "ollama".to_owned(),
            model: "all-minilm:l6-v2".to_owned(),
            dimension: 2,
            similarity_metric: "cosine".to_owned(),
            endpoint_fingerprint: None,
            max_sequence_length: None,
            crop_token_budget: None,
            tokenizer_fingerprint: None,
            crop_policy: None,
            embedding_kind: Some("single_vector".to_owned()),
            model_revision: None,
            preprocessing_fingerprint: None,
            max_image_patches: None,
        };
        let provider = OllamaEmbeddingProvider::new(
            profile,
            &format!("http://{address}/"),
            Duration::from_secs(5),
        )
        .unwrap();
        let texts = vec!["three-four".to_owned(), "zero".to_owned()];
        let result = provider.embed(&texts).await.unwrap();
        server.await.unwrap();
        assert_eq!(result[0], vec![0.6, 0.8]);
        assert_eq!(result[1], vec![0.0, 0.0]);
    }
}
