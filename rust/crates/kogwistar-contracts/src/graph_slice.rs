//! Versioned field-slice rules shared by Python and Rust graph paths.
//!
//! Pydantic-extension remains the source of truth for Python model annotations.
//! Python can export those annotations as this manifest, and Rust can apply the
//! same rule without guessing visibility from field names.  Backend callers may
//! preserve unknown fields for legacy compatibility; public/LLM slices fail
//! closed when a field is not listed by the manifest.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

use crate::graph_model::GraphModelClass;

pub const GRAPH_SLICE_MANIFEST_VERSION: u16 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphSliceMode {
    Backend,
    Frontend,
    Dto,
    Llm,
    LlmIn,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphSliceFieldV1 {
    /// Exact modes in which this field is allowed.
    #[serde(default)]
    pub modes: BTreeSet<GraphSliceMode>,
    /// Optional recursive rule for nested models or model containers.
    #[serde(default)]
    pub child: Option<Box<GraphSliceManifestV1>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphSliceManifestV1 {
    pub schema_version: u16,
    #[serde(default)]
    pub model_name: Option<String>,
    /// Unknown fields are retained only for these modes.  A Python model with
    /// ``extra='allow'`` can therefore preserve backend compatibility while
    /// still making public and LLM projections deny-by-default.
    #[serde(default)]
    pub retain_unlisted_modes: BTreeSet<GraphSliceMode>,
    pub fields: BTreeMap<String, GraphSliceFieldV1>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GraphModelSliceV1 {
    pub schema_version: u16,
    pub model_class: GraphModelClass,
    pub mode: GraphSliceMode,
    pub payload: Value,
}

/// Optional declaration carried beside a canonical graph envelope. The
/// declaration selects a projection; it does not replace the canonical
/// payload or grant access to it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GraphSliceDeclarationV1 {
    pub schema_version: u16,
    pub mode: GraphSliceMode,
    pub manifest: GraphSliceManifestV1,
}

#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum GraphSliceError {
    #[error("unsupported graph slice manifest version {0}")]
    UnsupportedVersion(u16),
    #[error("graph slice payload must be a JSON object")]
    PayloadMustBeObject,
    #[error("graph slice manifest contains an empty field name")]
    EmptyFieldName,
    #[error("unlisted graph fields may only be retained in backend mode")]
    UnlistedFieldsOutsideBackend,
}

impl GraphSliceManifestV1 {
    pub fn new(fields: BTreeMap<String, GraphSliceFieldV1>) -> Self {
        Self {
            schema_version: GRAPH_SLICE_MANIFEST_VERSION,
            model_name: None,
            retain_unlisted_modes: BTreeSet::from([GraphSliceMode::Backend]),
            fields,
        }
    }

    pub fn project(&self, payload: &Value, mode: GraphSliceMode) -> Result<Value, GraphSliceError> {
        if self.schema_version != GRAPH_SLICE_MANIFEST_VERSION {
            return Err(GraphSliceError::UnsupportedVersion(self.schema_version));
        }
        if self
            .retain_unlisted_modes
            .iter()
            .any(|retained| *retained != GraphSliceMode::Backend)
        {
            return Err(GraphSliceError::UnlistedFieldsOutsideBackend);
        }
        for name in self.fields.keys() {
            if name.is_empty() {
                return Err(GraphSliceError::EmptyFieldName);
            }
        }
        let object = payload
            .as_object()
            .ok_or(GraphSliceError::PayloadMustBeObject)?;
        Ok(Value::Object(project_object(
            object,
            self,
            mode,
            self.retain_unlisted_modes.contains(&mode),
        )))
    }
}

fn project_object(
    object: &Map<String, Value>,
    manifest: &GraphSliceManifestV1,
    mode: GraphSliceMode,
    retain_unlisted: bool,
) -> Map<String, Value> {
    let mut projected = Map::new();
    for (name, value) in object {
        let Some(rule) = manifest.fields.get(name) else {
            if retain_unlisted {
                projected.insert(name.clone(), value.clone());
            }
            continue;
        };
        if !rule.modes.contains(&mode) {
            continue;
        }
        let value = rule
            .child
            .as_deref()
            .map_or_else(|| value.clone(), |child| project_nested(value, child, mode));
        projected.insert(name.clone(), value);
    }
    projected
}

fn project_nested(value: &Value, manifest: &GraphSliceManifestV1, mode: GraphSliceMode) -> Value {
    match value {
        Value::Object(object) => Value::Object(project_object(
            object,
            manifest,
            mode,
            manifest.retain_unlisted_modes.contains(&mode),
        )),
        Value::Array(items) => Value::Array(
            items
                .iter()
                .map(|item| project_nested(item, manifest, mode))
                .collect(),
        ),
        scalar => scalar.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rule(modes: &[GraphSliceMode]) -> GraphSliceFieldV1 {
        GraphSliceFieldV1 {
            modes: modes.iter().copied().collect(),
            child: None,
        }
    }

    #[test]
    fn public_projection_drops_unlisted_and_nested_backend_fields() {
        let mut span_fields = BTreeMap::new();
        span_fields.insert(
            "excerpt".to_owned(),
            rule(&[GraphSliceMode::Backend, GraphSliceMode::LlmIn]),
        );
        span_fields.insert("doc_id".to_owned(), rule(&[GraphSliceMode::Backend]));
        let span = GraphSliceManifestV1 {
            schema_version: GRAPH_SLICE_MANIFEST_VERSION,
            model_name: Some("Span".to_owned()),
            retain_unlisted_modes: BTreeSet::new(),
            fields: span_fields,
        };

        let mut fields = BTreeMap::new();
        fields.insert(
            "summary".to_owned(),
            rule(&[GraphSliceMode::Backend, GraphSliceMode::LlmIn]),
        );
        fields.insert("secret".to_owned(), rule(&[GraphSliceMode::Backend]));
        fields.insert(
            "mentions".to_owned(),
            GraphSliceFieldV1 {
                modes: BTreeSet::from([GraphSliceMode::Backend, GraphSliceMode::LlmIn]),
                child: Some(Box::new(GraphSliceManifestV1 {
                    schema_version: GRAPH_SLICE_MANIFEST_VERSION,
                    model_name: Some("Grounding".to_owned()),
                    retain_unlisted_modes: BTreeSet::new(),
                    fields: BTreeMap::from([(
                        "spans".to_owned(),
                        GraphSliceFieldV1 {
                            modes: BTreeSet::from([GraphSliceMode::Backend, GraphSliceMode::LlmIn]),
                            child: Some(Box::new(span)),
                        },
                    )]),
                })),
            },
        );

        let manifest = GraphSliceManifestV1::new(fields);
        let payload = serde_json::json!({
            "summary": "visible",
            "secret": "backend-only",
            "mentions": [{"spans": [{"excerpt": "text", "doc_id": "secret-doc"}]}]
        });
        assert_eq!(
            manifest.project(&payload, GraphSliceMode::LlmIn).unwrap(),
            serde_json::json!({"summary": "visible", "mentions": [{"spans": [{"excerpt": "text"}]}]})
        );
        assert_eq!(
            manifest.project(&payload, GraphSliceMode::Backend).unwrap(),
            payload
        );
    }

    #[test]
    fn invalid_manifest_version_fails_closed() {
        let manifest = GraphSliceManifestV1 {
            schema_version: GRAPH_SLICE_MANIFEST_VERSION + 1,
            model_name: None,
            retain_unlisted_modes: BTreeSet::new(),
            fields: BTreeMap::new(),
        };
        assert!(matches!(
            manifest.project(&serde_json::json!({}), GraphSliceMode::LlmIn),
            Err(GraphSliceError::UnsupportedVersion(_))
        ));
    }

    #[test]
    fn unlisted_fields_cannot_be_retained_in_model_visible_modes() {
        let manifest = GraphSliceManifestV1 {
            schema_version: GRAPH_SLICE_MANIFEST_VERSION,
            model_name: Some("GraphEntity".to_owned()),
            retain_unlisted_modes: BTreeSet::from([GraphSliceMode::Llm]),
            fields: BTreeMap::new(),
        };

        assert_eq!(
            manifest.project(&serde_json::json!({"secret": true}), GraphSliceMode::Llm),
            Err(GraphSliceError::UnlistedFieldsOutsideBackend)
        );
    }
}
