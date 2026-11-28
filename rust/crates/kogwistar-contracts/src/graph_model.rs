//! Typed, additive view over Python graph-model payloads.
//!
//! `GraphModelEnvelope` carries the Python model class out-of-band. The tag is
//! deliberately not inserted into the legacy graph payload: older persisted
//! rows therefore remain byte-shape compatible, while Rust can distinguish
//! base, ACL, conversation, extraction, and flattened models when the caller
//! has authoritative type information. Nodes and edges both implement the
//! shared graph-entity interface; edges extend its common fields with typed
//! node and edge endpoint references.

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::value::RawValue;
use serde_json::{Map, Value};
use std::collections::BTreeMap;
use thiserror::Error;

mod types;
pub use types::*;

#[cfg(test)]
mod tests;

use crate::graph_slice::{
    GRAPH_SLICE_MANIFEST_VERSION, GraphModelSliceV1, GraphSliceDeclarationV1, GraphSliceManifestV1,
    GraphSliceMode,
};

pub const GRAPH_MODEL_ENVELOPE_VERSION: u16 = 1;
pub const PYTHON_GRAPH_WIRE_VERSION: u16 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphEntityKind {
    Node,
    Edge,
}

/// Python graph model classes that can cross a typed Rust/Python seam.
///
/// This is envelope metadata, not a discriminator added to persisted JSON.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphModelClass {
    Node,
    Edge,
    AclNode,
    AclEdge,
    ConversationNode,
    ConversationEdge,
    WorkflowNode,
    WorkflowEdge,
    WorkflowRunNode,
    WorkflowStepExecNode,
    WorkflowCheckpointNode,
    WorkflowCompletedNode,
    WorkflowCancelledNode,
    WorkflowFailedNode,
    WorkflowRuntimeEdge,
    PureChromaNode,
    PureChromaEdge,
    DocNode,
    LlmNode,
    LlmEdge,
    LlmNodeExtraction,
    LlmEdgeExtraction,
    FlattenedLlmNode,
    FlattenedLlmEdge,
    AssocFlattenedLlmNode,
    AssocFlattenedLlmEdge,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphModelFamily {
    Persistent,
    Acl,
    Conversation,
    Workflow,
    WorkflowRuntime,
    PureChroma,
    Document,
    Llm,
    FlattenedLlm,
}

/// Capabilities exposed by a graph model at a language seam.
///
/// This is deliberately a description of the model contract, not an
/// authorization grant.  Edge profiles must retain every capability of their
/// corresponding node profile and add incidence operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphCapabilityProfile {
    pub graph_entity: bool,
    pub edge_entity: bool,
    pub metadata: bool,
    pub lifecycle: bool,
    pub level: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphModelCapabilityProfileV1 {
    pub schema_version: u16,
    pub model_class: String,
    pub entity_kind: GraphEntityKind,
    #[serde(flatten)]
    pub capabilities: GraphCapabilityProfile,
}

pub fn graph_model_capability_profiles() -> Vec<GraphModelCapabilityProfileV1> {
    GraphModelClass::all()
        .iter()
        .map(|model_class| GraphModelCapabilityProfileV1 {
            schema_version: GRAPH_MODEL_ENVELOPE_VERSION,
            model_class: model_class.wire_name().to_owned(),
            entity_kind: model_class.entity_kind(),
            capabilities: model_class.capabilities(),
        })
        .collect()
}

const PERSISTENT_NODE_CAPABILITIES: GraphCapabilityProfile = GraphCapabilityProfile {
    graph_entity: true,
    edge_entity: false,
    metadata: true,
    lifecycle: true,
    level: true,
};

const PERSISTENT_EDGE_CAPABILITIES: GraphCapabilityProfile = GraphCapabilityProfile {
    graph_entity: true,
    edge_entity: true,
    metadata: true,
    lifecycle: true,
    level: true,
};

const LLM_NODE_CAPABILITIES: GraphCapabilityProfile = GraphCapabilityProfile {
    graph_entity: true,
    edge_entity: false,
    metadata: false,
    lifecycle: false,
    level: false,
};

const LLM_EDGE_CAPABILITIES: GraphCapabilityProfile = GraphCapabilityProfile {
    graph_entity: true,
    edge_entity: true,
    metadata: false,
    lifecycle: false,
    level: false,
};

const PURE_CHROMA_NODE_CAPABILITIES: GraphCapabilityProfile = GraphCapabilityProfile {
    graph_entity: true,
    edge_entity: false,
    metadata: true,
    lifecycle: false,
    level: false,
};

const PURE_CHROMA_EDGE_CAPABILITIES: GraphCapabilityProfile = GraphCapabilityProfile {
    graph_entity: true,
    edge_entity: true,
    metadata: true,
    lifecycle: false,
    level: false,
};

const DOCUMENT_NODE_CAPABILITIES: GraphCapabilityProfile = GraphCapabilityProfile {
    graph_entity: true,
    edge_entity: false,
    metadata: false,
    lifecycle: false,
    level: false,
};

impl GraphModelClass {
    /// Return every graph class supported by this contract version.
    ///
    /// Keep this list explicit so language bindings can attest that their
    /// registries describe the same wire contract before using the bridge.
    pub const fn all() -> &'static [Self] {
        &[
            Self::Node,
            Self::Edge,
            Self::AclNode,
            Self::AclEdge,
            Self::ConversationNode,
            Self::ConversationEdge,
            Self::WorkflowNode,
            Self::WorkflowEdge,
            Self::WorkflowRunNode,
            Self::WorkflowStepExecNode,
            Self::WorkflowCheckpointNode,
            Self::WorkflowCompletedNode,
            Self::WorkflowCancelledNode,
            Self::WorkflowFailedNode,
            Self::WorkflowRuntimeEdge,
            Self::PureChromaNode,
            Self::PureChromaEdge,
            Self::DocNode,
            Self::LlmNode,
            Self::LlmEdge,
            Self::LlmNodeExtraction,
            Self::LlmEdgeExtraction,
            Self::FlattenedLlmNode,
            Self::FlattenedLlmEdge,
            Self::AssocFlattenedLlmNode,
            Self::AssocFlattenedLlmEdge,
        ]
    }

    /// Return the stable snake-case name used in the Python envelope.
    pub const fn wire_name(self) -> &'static str {
        match self {
            Self::Node => "node",
            Self::Edge => "edge",
            Self::AclNode => "acl_node",
            Self::AclEdge => "acl_edge",
            Self::ConversationNode => "conversation_node",
            Self::ConversationEdge => "conversation_edge",
            Self::WorkflowNode => "workflow_node",
            Self::WorkflowEdge => "workflow_edge",
            Self::WorkflowRunNode => "workflow_run_node",
            Self::WorkflowStepExecNode => "workflow_step_exec_node",
            Self::WorkflowCheckpointNode => "workflow_checkpoint_node",
            Self::WorkflowCompletedNode => "workflow_completed_node",
            Self::WorkflowCancelledNode => "workflow_cancelled_node",
            Self::WorkflowFailedNode => "workflow_failed_node",
            Self::WorkflowRuntimeEdge => "workflow_runtime_edge",
            Self::PureChromaNode => "pure_chroma_node",
            Self::PureChromaEdge => "pure_chroma_edge",
            Self::DocNode => "doc_node",
            Self::LlmNode => "llm_node",
            Self::LlmEdge => "llm_edge",
            Self::LlmNodeExtraction => "llm_node_extraction",
            Self::LlmEdgeExtraction => "llm_edge_extraction",
            Self::FlattenedLlmNode => "flattened_llm_node",
            Self::FlattenedLlmEdge => "flattened_llm_edge",
            Self::AssocFlattenedLlmNode => "assoc_flattened_llm_node",
            Self::AssocFlattenedLlmEdge => "assoc_flattened_llm_edge",
        }
    }

    /// Return the Python class name used to build a slice manifest.
    pub const fn python_name(self) -> &'static str {
        match self {
            Self::Node => "Node",
            Self::Edge => "Edge",
            Self::AclNode => "ACLNode",
            Self::AclEdge => "ACLEdge",
            Self::ConversationNode => "ConversationNode",
            Self::ConversationEdge => "ConversationEdge",
            Self::WorkflowNode => "WorkflowNode",
            Self::WorkflowEdge => "WorkflowEdge",
            Self::WorkflowRunNode => "WorkflowRunNode",
            Self::WorkflowStepExecNode => "WorkflowStepExecNode",
            Self::WorkflowCheckpointNode => "WorkflowCheckpointNode",
            Self::WorkflowCompletedNode => "WorkflowCompletedNode",
            Self::WorkflowCancelledNode => "WorkflowCancelledNode",
            Self::WorkflowFailedNode => "WorkflowFailedNode",
            Self::WorkflowRuntimeEdge => "WorkflowRuntimeEdge",
            Self::PureChromaNode => "PureChromaNode",
            Self::PureChromaEdge => "PureChromaEdge",
            Self::DocNode => "DocNode",
            Self::LlmNode => "LLMNode",
            Self::LlmEdge => "LLMEdge",
            Self::LlmNodeExtraction => "LLMNodeExtraction",
            Self::LlmEdgeExtraction => "LLMEdgeExtraction",
            Self::FlattenedLlmNode => "FlattenedLLMNode",
            Self::FlattenedLlmEdge => "FlattenedLLMEdge",
            Self::AssocFlattenedLlmNode => "AssocFlattenedLLMNode",
            Self::AssocFlattenedLlmEdge => "AssocFlattenedLLMEdge",
        }
    }

    pub const fn family(self) -> GraphModelFamily {
        match self {
            Self::Node | Self::Edge => GraphModelFamily::Persistent,
            Self::AclNode | Self::AclEdge => GraphModelFamily::Acl,
            Self::ConversationNode | Self::ConversationEdge => GraphModelFamily::Conversation,
            Self::WorkflowNode | Self::WorkflowEdge => GraphModelFamily::Workflow,
            Self::WorkflowRunNode
            | Self::WorkflowStepExecNode
            | Self::WorkflowCheckpointNode
            | Self::WorkflowCompletedNode
            | Self::WorkflowCancelledNode
            | Self::WorkflowFailedNode
            | Self::WorkflowRuntimeEdge => GraphModelFamily::WorkflowRuntime,
            Self::PureChromaNode | Self::PureChromaEdge => GraphModelFamily::PureChroma,
            Self::DocNode => GraphModelFamily::Document,
            Self::LlmNode | Self::LlmEdge | Self::LlmNodeExtraction | Self::LlmEdgeExtraction => {
                GraphModelFamily::Llm
            }
            Self::FlattenedLlmNode
            | Self::FlattenedLlmEdge
            | Self::AssocFlattenedLlmNode
            | Self::AssocFlattenedLlmEdge => GraphModelFamily::FlattenedLlm,
        }
    }

    pub const fn entity_kind(self) -> GraphEntityKind {
        match self {
            Self::Edge
            | Self::AclEdge
            | Self::ConversationEdge
            | Self::WorkflowEdge
            | Self::WorkflowRuntimeEdge
            | Self::PureChromaEdge
            | Self::LlmEdge
            | Self::LlmEdgeExtraction
            | Self::FlattenedLlmEdge
            | Self::AssocFlattenedLlmEdge => GraphEntityKind::Edge,
            Self::Node
            | Self::AclNode
            | Self::ConversationNode
            | Self::WorkflowNode
            | Self::WorkflowRunNode
            | Self::WorkflowStepExecNode
            | Self::WorkflowCheckpointNode
            | Self::WorkflowCompletedNode
            | Self::WorkflowCancelledNode
            | Self::WorkflowFailedNode
            | Self::PureChromaNode
            | Self::DocNode
            | Self::LlmNode
            | Self::LlmNodeExtraction
            | Self::FlattenedLlmNode
            | Self::AssocFlattenedLlmNode => GraphEntityKind::Node,
        }
    }

    /// Return the node/edge capability profile represented by this class.
    ///
    /// Specialized persistent families intentionally reuse the persistent
    /// capability profile. Their family-specific metadata is validated by the
    /// family adapter, while the shared graph/lifecycle contract remains the
    /// same as Python's Node/Edge inheritance.
    pub const fn capabilities(self) -> GraphCapabilityProfile {
        match self {
            Self::Node | Self::AclNode | Self::ConversationNode | Self::WorkflowNode => {
                PERSISTENT_NODE_CAPABILITIES
            }
            Self::DocNode => DOCUMENT_NODE_CAPABILITIES,
            Self::Edge | Self::AclEdge | Self::ConversationEdge | Self::WorkflowEdge => {
                PERSISTENT_EDGE_CAPABILITIES
            }
            Self::WorkflowRuntimeEdge => PERSISTENT_EDGE_CAPABILITIES,
            Self::WorkflowRunNode
            | Self::WorkflowStepExecNode
            | Self::WorkflowCheckpointNode
            | Self::WorkflowCompletedNode
            | Self::WorkflowCancelledNode
            | Self::WorkflowFailedNode => PERSISTENT_NODE_CAPABILITIES,
            Self::LlmNode | Self::LlmNodeExtraction => LLM_NODE_CAPABILITIES,
            Self::LlmEdge | Self::LlmEdgeExtraction => LLM_EDGE_CAPABILITIES,
            Self::FlattenedLlmNode | Self::AssocFlattenedLlmNode => LLM_NODE_CAPABILITIES,
            Self::FlattenedLlmEdge | Self::AssocFlattenedLlmEdge => LLM_EDGE_CAPABILITIES,
            Self::PureChromaNode => PURE_CHROMA_NODE_CAPABILITIES,
            Self::PureChromaEdge => PURE_CHROMA_EDGE_CAPABILITIES,
        }
    }

    /// Return the corresponding node or edge model class in this family.
    pub const fn corresponding_class(self, kind: GraphEntityKind) -> Option<Self> {
        match (self.family(), kind) {
            (GraphModelFamily::Persistent, GraphEntityKind::Node) => Some(Self::Node),
            (GraphModelFamily::Persistent, GraphEntityKind::Edge) => Some(Self::Edge),
            (GraphModelFamily::Acl, GraphEntityKind::Node) => Some(Self::AclNode),
            (GraphModelFamily::Acl, GraphEntityKind::Edge) => Some(Self::AclEdge),
            (GraphModelFamily::Conversation, GraphEntityKind::Node) => Some(Self::ConversationNode),
            (GraphModelFamily::Conversation, GraphEntityKind::Edge) => Some(Self::ConversationEdge),
            (GraphModelFamily::Workflow, GraphEntityKind::Node) => Some(Self::WorkflowNode),
            (GraphModelFamily::Workflow, GraphEntityKind::Edge) => Some(Self::WorkflowEdge),
            (GraphModelFamily::WorkflowRuntime, GraphEntityKind::Node) => match self {
                Self::WorkflowRunNode => Some(Self::WorkflowRunNode),
                Self::WorkflowStepExecNode => Some(Self::WorkflowStepExecNode),
                Self::WorkflowCheckpointNode => Some(Self::WorkflowCheckpointNode),
                Self::WorkflowCompletedNode => Some(Self::WorkflowCompletedNode),
                Self::WorkflowCancelledNode => Some(Self::WorkflowCancelledNode),
                Self::WorkflowFailedNode => Some(Self::WorkflowFailedNode),
                Self::WorkflowRuntimeEdge => Some(Self::WorkflowRunNode),
                _ => None,
            },
            (GraphModelFamily::WorkflowRuntime, GraphEntityKind::Edge) => match self {
                // Runtime nodes have several concrete lifecycle states but
                // share one persistent Node capability profile. Use the
                // first runtime node as the profile representative; this is
                // not a wire-type coercion.
                Self::WorkflowRuntimeEdge => Some(Self::WorkflowRunNode),
                _ => None,
            },
            (GraphModelFamily::PureChroma, GraphEntityKind::Node) => Some(Self::PureChromaNode),
            (GraphModelFamily::PureChroma, GraphEntityKind::Edge) => Some(Self::PureChromaEdge),
            (GraphModelFamily::Document, GraphEntityKind::Node) => Some(Self::DocNode),
            (GraphModelFamily::Document, GraphEntityKind::Edge) => None,
            (GraphModelFamily::Llm, GraphEntityKind::Node) => match self {
                Self::LlmEdgeExtraction => Some(Self::LlmNodeExtraction),
                Self::LlmEdge => Some(Self::LlmNode),
                Self::LlmNodeExtraction => Some(Self::LlmNodeExtraction),
                _ => Some(Self::LlmNode),
            },
            (GraphModelFamily::Llm, GraphEntityKind::Edge) => match self {
                Self::LlmNodeExtraction => Some(Self::LlmEdgeExtraction),
                _ => Some(Self::LlmEdge),
            },
            (GraphModelFamily::FlattenedLlm, GraphEntityKind::Node) => match self {
                Self::AssocFlattenedLlmEdge => Some(Self::AssocFlattenedLlmNode),
                _ => Some(Self::FlattenedLlmNode),
            },
            (GraphModelFamily::FlattenedLlm, GraphEntityKind::Edge) => match self {
                Self::AssocFlattenedLlmNode => Some(Self::AssocFlattenedLlmEdge),
                _ => Some(Self::FlattenedLlmEdge),
            },
        }
    }

    /// Whether this class satisfies the shared node/edge inheritance rule.
    pub fn edge_extends_node_capabilities(self) -> bool {
        if self.entity_kind() != GraphEntityKind::Edge {
            return false;
        }
        let Some(node_class) = self.corresponding_class(GraphEntityKind::Node) else {
            return false;
        };
        let node = node_class.capabilities();
        let edge = self.capabilities();
        edge.graph_entity == node.graph_entity
            && (!node.metadata || edge.metadata)
            && (!node.lifecycle || edge.lifecycle)
            && (!node.level || edge.level)
            && edge.edge_entity
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct NodeId(pub String);

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct EdgeId(pub String);

/// Stable wrapper used at language boundaries. `payload` is kept intact so a
/// decode/inspect/re-encode cycle does not normalize or drop legacy fields.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GraphModelEnvelope {
    pub schema_version: u16,
    pub model_class: GraphModelClass,
    pub payload: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub slice: Option<GraphSliceDeclarationV1>,
}

/// Lossless wire envelope for callers that must forward Python JSON without
/// passing arbitrary-precision integers through `serde_json::Value`.
///
/// This is an additive transport type. Callers that need a typed semantic
/// view can continue using [`GraphModelEnvelope`]; callers that only need to
/// validate or forward the canonical payload should retain this raw form.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RawGraphModelEnvelopeV1 {
    pub schema_version: u16,
    pub model_class: GraphModelClass,
    pub payload: Box<RawValue>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub slice: Option<GraphSliceDeclarationV1>,
}

/// Exact decimal representation of a Python integer at the JSON boundary.
///
/// The string is canonicalized (no leading zeroes, except for zero itself) and
/// remains unbounded. Use `to_i64()` only when adapting to a legacy V1 DTO.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PythonIntegerV1(String);

impl PythonIntegerV1 {
    pub fn as_decimal_str(&self) -> &str {
        &self.0
    }

    pub fn to_i64(&self) -> Option<i64> {
        self.0.parse().ok()
    }
}

impl RawGraphModelEnvelopeV1 {
    pub fn from_json(payload_json: &str) -> Result<Self, GraphModelError> {
        serde_json::from_str(payload_json)
            .map_err(|error| GraphModelError::InvalidPayload(error.to_string()))
    }

    /// Serialize the envelope while embedding its payload without reparsing it.
    pub fn to_json(&self) -> Result<String, GraphModelError> {
        serde_json::to_string(self)
            .map_err(|error| GraphModelError::InvalidPayload(error.to_string()))
    }

    pub fn validate_family_v1(&self) -> Result<(), GraphModelError> {
        validate_graph_model_family_from_str(&self.to_json()?)
    }

    /// Read Python's level mixin value without narrowing it to Rust `i64`.
    /// An explicitly present top-level field takes precedence over metadata,
    /// including explicit null, matching the Python model's synchronization.
    pub fn level_from_root(&self) -> Result<Option<PythonIntegerV1>, GraphModelError> {
        let payload: BTreeMap<String, Box<RawValue>> = serde_json::from_str(self.payload.get())
            .map_err(|error| GraphModelError::InvalidPayload(error.to_string()))?;
        if let Some(raw) = payload.get("level_from_root") {
            return parse_python_integer_raw(raw.get());
        }
        let Some(metadata) = payload.get("metadata") else {
            return Ok(None);
        };
        let metadata: BTreeMap<String, Box<RawValue>> = serde_json::from_str(metadata.get())
            .map_err(|error| GraphModelError::InvalidPayload(error.to_string()))?;
        match metadata.get("level_from_root") {
            Some(raw) => parse_python_integer_raw(raw.get()),
            None => Ok(None),
        }
    }

    /// Decode persistent graph fields while keeping Python's unbounded
    /// hierarchy level available without narrowing it to the legacy V1 `i64`.
    /// The typed V1 view uses `null` only when that field is outside `i64`;
    /// `level_from_root` and `raw_payload` retain its exact representations.
    pub fn decode_persistent_v2(&self) -> Result<PersistentGraphEntityV2, GraphModelError> {
        if self.schema_version != GRAPH_MODEL_ENVELOPE_VERSION {
            return Err(GraphModelError::UnsupportedVersion(self.schema_version));
        }
        if !matches!(
            self.model_class,
            GraphModelClass::Node | GraphModelClass::Edge
        ) {
            return Err(GraphModelError::StrictModelClassUnsupported {
                model_class: self.model_class,
            });
        }

        let level_from_root = self.level_from_root()?;
        let payload = persistent_payload_for_v1_decode(self.payload.get())?;
        // The typed persistent decoder performs the Node/Edge family checks.
        // Do not call validate_family_v1() first: for these classes that would
        // decode and validate the same entity once before this lossless V1 view.
        let entity = GraphModelEnvelope::new(self.model_class, payload).decode_persistent_v1()?;
        Ok(PersistentGraphEntityV2 {
            entity,
            level_from_root,
            raw_payload: self.payload.clone(),
        })
    }
}

fn persistent_payload_for_v1_decode(raw_payload: &str) -> Result<Value, GraphModelError> {
    let fields = serde_json::from_str::<BTreeMap<String, Box<RawValue>>>(raw_payload)
        .map_err(|error| GraphModelError::InvalidPayload(error.to_string()))?;
    let mut sanitized = String::from("{");
    for (index, (key, raw_value)) in fields.iter().enumerate() {
        if index > 0 {
            sanitized.push(',');
        }
        sanitized.push_str(
            &serde_json::to_string(key)
                .map_err(|error| GraphModelError::InvalidPayload(error.to_string()))?,
        );
        sanitized.push(':');
        if key == "level_from_root" && level_requires_lossless_sidecar(raw_value.get())? {
            sanitized.push_str("null");
        } else if key == "metadata" {
            sanitized.push_str(&metadata_without_exact_level(raw_value.get())?);
        } else {
            sanitized.push_str(raw_value.get());
        }
    }
    sanitized.push('}');
    serde_json::from_str(&sanitized)
        .map_err(|error| GraphModelError::InvalidPayload(error.to_string()))
}

fn metadata_without_exact_level(raw_metadata: &str) -> Result<String, GraphModelError> {
    let Ok(fields) = serde_json::from_str::<BTreeMap<String, Box<RawValue>>>(raw_metadata) else {
        return Ok(raw_metadata.to_owned());
    };
    let mut sanitized = String::from("{");
    for (index, (key, raw_value)) in fields.iter().enumerate() {
        if index > 0 {
            sanitized.push(',');
        }
        sanitized.push_str(
            &serde_json::to_string(key)
                .map_err(|error| GraphModelError::InvalidPayload(error.to_string()))?,
        );
        sanitized.push(':');
        if key == "level_from_root" && level_requires_lossless_sidecar(raw_value.get())? {
            sanitized.push_str("null");
        } else {
            sanitized.push_str(raw_value.get());
        }
    }
    sanitized.push('}');
    Ok(sanitized)
}

fn level_requires_lossless_sidecar(raw_level: &str) -> Result<bool, GraphModelError> {
    Ok(parse_python_integer_raw(raw_level)?.is_some_and(|level| level.to_i64().is_none()))
}

fn parse_python_integer_raw(raw: &str) -> Result<Option<PythonIntegerV1>, GraphModelError> {
    let token = raw.trim();
    if token == "null" {
        return Ok(None);
    }
    let string_input = token.starts_with('"');
    let decimal = if token == "true" {
        "1".to_owned()
    } else if token == "false" {
        "0".to_owned()
    } else if token.starts_with('"') {
        let value: String = serde_json::from_str(token)
            .map_err(|error| GraphModelError::InvalidPayload(error.to_string()))?;
        value.trim().to_owned()
    } else {
        token.to_owned()
    };
    let (negative, digits) = match decimal.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => match decimal.strip_prefix('+') {
            Some(rest) => (false, rest),
            None => (false, decimal.as_str()),
        },
    };
    if !string_input && (digits.contains('.') || digits.contains('e') || digits.contains('E')) {
        let value = decimal
            .parse::<f64>()
            .map_err(|error| GraphModelError::InvalidPayload(error.to_string()))?;
        if value.is_finite()
            && value.fract() == 0.0
            && value >= i64::MIN as f64
            && value < -(i64::MIN as f64)
        {
            return Ok(Some(PythonIntegerV1((value as i64).to_string())));
        }
        return Err(GraphModelError::InvalidPayload(
            "level_from_root must be an integer".to_owned(),
        ));
    }
    let bytes = digits.as_bytes();
    if bytes.is_empty()
        || !bytes.iter().enumerate().all(|(index, byte)| match byte {
            b'0'..=b'9' => true,
            b'_' => {
                index > 0
                    && index + 1 < bytes.len()
                    && bytes[index - 1].is_ascii_digit()
                    && bytes[index + 1].is_ascii_digit()
            }
            _ => false,
        })
    {
        return Err(GraphModelError::InvalidPayload(
            "level_from_root must be an integer".to_owned(),
        ));
    }
    let digits = digits.replace('_', "");
    let digits = digits.trim_start_matches('0');
    let digits = if digits.is_empty() { "0" } else { digits };
    let canonical = if negative && digits != "0" {
        format!("-{digits}")
    } else {
        digits.to_owned()
    };
    Ok(Some(PythonIntegerV1(canonical)))
}

pub fn round_trip_raw_graph_model_envelope_from_str(
    payload_json: &str,
) -> Result<String, GraphModelError> {
    let envelope = RawGraphModelEnvelopeV1::from_json(payload_json)?;
    envelope.validate_family_v1()?;
    envelope.to_json()
}

/// Decode a persistent raw graph envelope into an additive, versioned
/// language-neutral result. Arbitrary-size hierarchy levels are decimal
/// strings; the legacy entity payload keeps its existing V1 shape.
pub fn decode_persistent_graph_model_v2_from_str(
    payload_json: &str,
) -> Result<String, GraphModelError> {
    let envelope = RawGraphModelEnvelopeV1::from_json(payload_json)?;
    let decoded = envelope.decode_persistent_v2()?;
    let entity = match &decoded.entity {
        PersistentGraphEntityV1::Node(node) => serde_json::to_value(node),
        PersistentGraphEntityV1::Edge(edge) => serde_json::to_value(edge),
    }
    .map_err(|error| GraphModelError::InvalidPayload(error.to_string()))?;
    let level_from_root = decoded
        .level_from_root
        .as_ref()
        .map(|value| value.as_decimal_str());
    serde_json::to_string(&serde_json::json!({
        "schema_version": 2,
        "model_class": envelope.model_class.wire_name(),
        "entity": entity,
        "level_from_root": level_from_root,
        "raw_payload": decoded.raw_payload,
    }))
    .map_err(|error| GraphModelError::InvalidPayload(error.to_string()))
}

pub fn python_level_from_root_from_str(
    payload_json: &str,
) -> Result<Option<String>, GraphModelError> {
    let envelope = RawGraphModelEnvelopeV1::from_json(payload_json)?;
    envelope.validate_family_v1()?;
    Ok(envelope
        .level_from_root()?
        .map(|value| value.as_decimal_str().to_owned()))
}

impl GraphModelEnvelope {
    pub fn new(model_class: GraphModelClass, payload: Value) -> Self {
        Self {
            schema_version: GRAPH_MODEL_ENVELOPE_VERSION,
            model_class,
            payload,
            slice: None,
        }
    }

    /// Attach an optional projection declaration while retaining the canonical
    /// payload for a later language or transport boundary.
    pub fn with_slice(mut self, declaration: GraphSliceDeclarationV1) -> Self {
        self.slice = Some(declaration);
        self
    }

    /// Apply an exported pydantic-extension field policy without changing the
    /// canonical envelope. The caller chooses the boundary explicitly; Rust
    /// never infers a public or LLM slice from field names.
    pub fn project_slice(
        &self,
        manifest: &GraphSliceManifestV1,
        mode: GraphSliceMode,
    ) -> Result<GraphModelSliceV1, GraphModelError> {
        if self.schema_version != GRAPH_MODEL_ENVELOPE_VERSION {
            return Err(GraphModelError::UnsupportedVersion(self.schema_version));
        }
        if let Some(model_name) = manifest.model_name.as_deref()
            && model_name != self.model_class.python_name()
        {
            return Err(GraphModelError::InvalidPayload(format!(
                "slice manifest model {model_name:?} does not match envelope model {}",
                self.model_class.python_name()
            )));
        }
        let payload = manifest
            .project(&self.payload, mode)
            .map_err(|error| GraphModelError::InvalidPayload(error.to_string()))?;
        Ok(GraphModelSliceV1 {
            schema_version: self.schema_version,
            model_class: self.model_class,
            mode,
            payload,
        })
    }

    pub fn project_declared_slice(&self) -> Result<GraphModelSliceV1, GraphModelError> {
        let declaration = self.slice.as_ref().ok_or_else(|| {
            GraphModelError::InvalidPayload("graph envelope has no slice declaration".to_owned())
        })?;
        if declaration.schema_version != GRAPH_SLICE_MANIFEST_VERSION {
            return Err(GraphModelError::UnsupportedVersion(
                declaration.schema_version,
            ));
        }
        self.project_slice(&declaration.manifest, declaration.mode)
    }

    /// Decode to typed graph references without changing the original wire payload.
    pub fn inspect(&self) -> Result<TypedGraphEntity, GraphModelError> {
        if self.schema_version != GRAPH_MODEL_ENVELOPE_VERSION {
            return Err(GraphModelError::UnsupportedVersion(self.schema_version));
        }
        match self.model_class.entity_kind() {
            GraphEntityKind::Node => {
                if has_edge_fields(&self.payload) {
                    return Err(GraphModelError::InvalidEntityFields {
                        model_class: self.model_class,
                        reason: "node model payload contains edge-only fields",
                    });
                }
                let payload: GraphNodePayload =
                    serde_json::from_value(typed_decode_view(&self.payload))
                        .map_err(|error| GraphModelError::InvalidPayload(error.to_string()))?;
                payload.validate(self.model_class)?;
                Ok(TypedGraphEntity::Node(payload))
            }
            GraphEntityKind::Edge => {
                let payload: GraphEdgePayload =
                    serde_json::from_value(typed_decode_view(&self.payload))
                        .map_err(|error| GraphModelError::InvalidPayload(error.to_string()))?;
                payload.validate(self.model_class)?;
                Ok(TypedGraphEntity::Edge(payload))
            }
        }
    }

    /// Decode the strict persistent Node/Edge contract without changing the
    /// original envelope payload. Other model families intentionally use
    /// separate adapters because their required fields differ.
    pub fn decode_persistent_v1(&self) -> Result<PersistentGraphEntityV1, GraphModelError> {
        if self.schema_version != GRAPH_MODEL_ENVELOPE_VERSION {
            return Err(GraphModelError::UnsupportedVersion(self.schema_version));
        }
        let level_field_present = self.payload.get("level_from_root").is_some();
        match self.model_class {
            GraphModelClass::Node => {
                if has_edge_fields(&self.payload) {
                    return Err(GraphModelError::InvalidEntityFields {
                        model_class: self.model_class,
                        reason: "node model payload contains edge-only fields",
                    });
                }
                serde_json::from_value(typed_decode_view(&self.payload))
                    .map(|payload: PersistentNodeV1| {
                        payload.into_validated(self.model_class, level_field_present)
                    })
                    .map_err(|error| GraphModelError::InvalidPayload(error.to_string()))?
            }
            GraphModelClass::Edge => {
                for field in [
                    "source_ids",
                    "target_ids",
                    "relation",
                    "source_edge_ids",
                    "target_edge_ids",
                ] {
                    require_wire_field(&self.payload, field)?;
                }
                serde_json::from_value(typed_decode_view(&self.payload))
                    .map(|payload: PersistentEdgeV1| {
                        payload.into_validated(self.model_class, level_field_present)
                    })
                    .map_err(|error| GraphModelError::InvalidPayload(error.to_string()))?
            }
            model_class => Err(GraphModelError::StrictModelClassUnsupported { model_class }),
        }
    }

    /// Validate the family-specific contract without changing the wire value.
    ///
    /// This deliberately remains separate from `inspect()` and
    /// `decode_persistent_v1()`: callers that only need lossless forwarding can
    /// keep using the permissive path, while typed seams can opt into the
    /// stricter family rules.
    pub fn validate_family_v1(&self) -> Result<(), GraphModelError> {
        if self.schema_version != GRAPH_MODEL_ENVELOPE_VERSION {
            return Err(GraphModelError::UnsupportedVersion(self.schema_version));
        }
        match self.model_class {
            GraphModelClass::Node | GraphModelClass::Edge => {
                self.decode_persistent_v1().map(|_| ())
            }
            GraphModelClass::WorkflowNode => {
                let typed = self.decode_persistent_shape(GraphEntityKind::Node)?;
                validate_workflow_node_metadata(&typed.core.metadata)
            }
            GraphModelClass::WorkflowEdge => {
                let typed = self.decode_persistent_edge_shape()?;
                validate_workflow_edge_metadata(&typed.core.metadata)
            }
            GraphModelClass::WorkflowRunNode => validate_workflow_runtime_node_shape(
                self.model_class,
                &self.payload,
                "workflow_run",
            ),
            GraphModelClass::WorkflowStepExecNode => validate_workflow_runtime_node_shape(
                self.model_class,
                &self.payload,
                "workflow_step_exec",
            ),
            GraphModelClass::WorkflowCheckpointNode => validate_workflow_runtime_node_shape(
                self.model_class,
                &self.payload,
                "workflow_checkpoint",
            ),
            GraphModelClass::WorkflowCompletedNode => validate_workflow_runtime_node_shape(
                self.model_class,
                &self.payload,
                "workflow_completed",
            ),
            GraphModelClass::WorkflowCancelledNode => validate_workflow_runtime_node_shape(
                self.model_class,
                &self.payload,
                "workflow_cancelled",
            ),
            GraphModelClass::WorkflowFailedNode => validate_workflow_runtime_node_shape(
                self.model_class,
                &self.payload,
                "workflow_failed",
            ),
            GraphModelClass::WorkflowRuntimeEdge => {
                let typed = self.decode_persistent_edge_shape()?;
                validate_workflow_runtime_edge_metadata(&typed.core.metadata)
            }
            GraphModelClass::ConversationNode => {
                self.decode_persistent_shape(GraphEntityKind::Node)?;
                validate_conversation_node_payload(&self.payload)
            }
            GraphModelClass::ConversationEdge => {
                self.decode_persistent_edge_shape()?;
                validate_conversation_edge_metadata(&raw_metadata(
                    GraphModelClass::ConversationEdge,
                    &self.payload,
                )?)
            }
            GraphModelClass::LlmNode
            | GraphModelClass::LlmNodeExtraction
            | GraphModelClass::LlmEdge
            | GraphModelClass::LlmEdgeExtraction
            | GraphModelClass::FlattenedLlmNode
            | GraphModelClass::FlattenedLlmEdge
            | GraphModelClass::AssocFlattenedLlmNode
            | GraphModelClass::AssocFlattenedLlmEdge => {
                validate_llm_shape(self.model_class, &self.payload)
            }
            GraphModelClass::PureChromaNode | GraphModelClass::PureChromaEdge => {
                validate_pure_chroma_shape(self.model_class, &self.payload)
            }
            GraphModelClass::DocNode => validate_document_node_shape(&self.payload),
            GraphModelClass::AclNode | GraphModelClass::AclEdge => {
                validate_acl_shape(self.model_class, &self.payload)
            }
        }
    }

    fn decode_persistent_shape(
        &self,
        kind: GraphEntityKind,
    ) -> Result<PersistentNodeV1, GraphModelError> {
        if kind != GraphEntityKind::Node || has_edge_fields(&self.payload) {
            return Err(GraphModelError::InvalidEntityFields {
                model_class: self.model_class,
                reason: "family node payload contains edge-only fields",
            });
        }
        let mut payload: PersistentNodeV1 =
            serde_json::from_value(typed_decode_view(&self.payload))
                .map_err(|error| GraphModelError::InvalidPayload(error.to_string()))?;
        payload.core.normalize_shared_mixin_fields(
            self.model_class,
            self.payload.get("level_from_root").is_some(),
        )?;
        payload.core.validate(self.model_class)?;
        Ok(payload)
    }

    fn decode_persistent_edge_shape(&self) -> Result<PersistentEdgeV1, GraphModelError> {
        for field in [
            "source_ids",
            "target_ids",
            "relation",
            "source_edge_ids",
            "target_edge_ids",
        ] {
            require_wire_field(&self.payload, field)?;
        }
        let mut edge: PersistentEdgeV1 =
            serde_json::from_value(typed_decode_view(&self.payload))
                .map_err(|error| GraphModelError::InvalidPayload(error.to_string()))?;
        edge.core.normalize_shared_mixin_fields(
            self.model_class,
            self.payload.get("level_from_root").is_some(),
        )?;
        edge.core.validate(self.model_class)?;
        Ok(edge)
    }
}

/// Validate one JSON envelope at the language seam without exposing a Rust
/// storage or runtime operation. This is used by differential tests and by
/// optional native callers that already own the envelope class tag.
pub fn validate_graph_model_family_from_str(payload_json: &str) -> Result<(), GraphModelError> {
    if let Ok(envelope) = serde_json::from_str::<GraphModelEnvelope>(payload_json)
        && envelope.validate_family_v1().is_ok()
    {
        return Ok(());
    }

    #[derive(Deserialize)]
    struct RawEnvelope<'a> {
        schema_version: u16,
        model_class: GraphModelClass,
        #[serde(borrow)]
        payload: &'a RawValue,
    }

    let raw_envelope: RawEnvelope<'_> = serde_json::from_str(payload_json)
        .map_err(|error| GraphModelError::InvalidPayload(error.to_string()))?;
    let fields: BTreeMap<String, Box<RawValue>> = serde_json::from_str(raw_envelope.payload.get())
        .map_err(|error| GraphModelError::InvalidPayload(error.to_string()))?;
    let mut payload = Map::new();
    for (name, raw) in fields {
        if name == "level_from_root" && level_requires_lossless_sidecar(raw.get())? {
            payload.insert(name, Value::Number(0.into()));
        } else if name == "metadata" {
            match serde_json::from_str::<BTreeMap<String, Box<RawValue>>>(raw.get()) {
                Ok(metadata_fields) => {
                    let mut metadata = Map::new();
                    for (key, value) in metadata_fields {
                        if key == "level_from_root" && level_requires_lossless_sidecar(value.get())?
                        {
                            metadata.insert(key, Value::Number(0.into()));
                        } else {
                            let parsed = serde_json::from_str(value.get()).map_err(|error| {
                                GraphModelError::InvalidPayload(error.to_string())
                            })?;
                            metadata.insert(key, parsed);
                        }
                    }
                    payload.insert(name, Value::Object(metadata));
                }
                Err(_) => {
                    let parsed = serde_json::from_str(raw.get())
                        .map_err(|error| GraphModelError::InvalidPayload(error.to_string()))?;
                    payload.insert(name, parsed);
                }
            }
        } else {
            let parsed = serde_json::from_str(raw.get())
                .map_err(|error| GraphModelError::InvalidPayload(error.to_string()))?;
            payload.insert(name, parsed);
        }
    }
    let envelope = GraphModelEnvelope {
        schema_version: raw_envelope.schema_version,
        model_class: raw_envelope.model_class,
        payload: Value::Object(payload),
        slice: None,
    };
    envelope.validate_family_v1()
}

/// Project a declared canonical envelope at the language seam.
///
/// The input must retain the backend/canonical payload and an explicit slice
/// declaration. Rust never infers a projection from field names or accepts a
/// caller-selected mode without the manifest exported by the Python model.
pub fn project_graph_model_slice_from_str(payload_json: &str) -> Result<String, GraphModelError> {
    let envelope: GraphModelEnvelope = serde_json::from_str(payload_json)
        .map_err(|error| GraphModelError::InvalidPayload(error.to_string()))?;
    let projected = envelope.project_declared_slice()?;
    serde_json::to_string(&projected)
        .map_err(|error| GraphModelError::InvalidPayload(error.to_string()))
}

/// Project several declared envelopes in one language-boundary crossing.
///
/// The batch is a transport optimization only: each envelope is still
/// validated and projected independently, and one invalid item fails the
/// complete call without returning partial results.
pub fn project_graph_model_slices_from_str(payload_json: &str) -> Result<String, GraphModelError> {
    let envelopes: Vec<GraphModelEnvelope> = serde_json::from_str(payload_json)
        .map_err(|error| GraphModelError::InvalidPayload(error.to_string()))?;
    let projected = envelopes
        .into_iter()
        .map(|envelope| envelope.project_declared_slice())
        .collect::<Result<Vec<_>, _>>()?;
    serde_json::to_string(&projected)
        .map_err(|error| GraphModelError::InvalidPayload(error.to_string()))
}

fn validate_metadata<T: for<'de> Deserialize<'de>>(
    model_class: GraphModelClass,
    metadata: &BTreeMap<String, Value>,
) -> Result<T, GraphModelError> {
    let mut value = Value::Object(metadata.clone().into_iter().collect());
    normalize_unbounded_integer_tokens(&mut value);
    let json = serde_json::to_string(&value).map_err(|error| {
        GraphModelError::InvalidPayload(format!("{model_class:?} metadata: {error}"))
    })?;
    serde_json::from_str(&json).map_err(|error| {
        GraphModelError::InvalidPayload(format!("{model_class:?} metadata: {error}"))
    })
}

fn typed_decode_view(value: &Value) -> Value {
    let mut normalized = value.clone();
    normalize_unbounded_integer_tokens(&mut normalized);
    normalized
}

fn normalize_unbounded_integer_tokens(value: &mut Value) {
    match value {
        Value::Array(items) => items
            .iter_mut()
            .for_each(normalize_unbounded_integer_tokens),
        Value::Object(fields) => fields
            .values_mut()
            .for_each(normalize_unbounded_integer_tokens),
        Value::Number(number) if number.as_i64().is_none() && number.as_u64().is_none() => {
            let raw = number.to_string();
            if python_integer_text_is_nonnegative(&raw).is_some()
                || raw
                    .strip_prefix('-')
                    .is_some_and(|digits| python_integer_text_is_nonnegative(digits).is_some())
            {
                *value = Value::String(raw);
            }
        }
        _ => {}
    }
}

fn raw_metadata(
    model_class: GraphModelClass,
    payload: &Value,
) -> Result<BTreeMap<String, Value>, GraphModelError> {
    match payload.get("metadata") {
        None => Ok(BTreeMap::new()),
        Some(Value::Object(metadata)) => Ok(metadata
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect()),
        Some(_) => Err(GraphModelError::InvalidEntityFields {
            model_class,
            reason: "conversation metadata must be an object",
        }),
    }
}

fn validate_workflow_node_metadata(
    metadata: &BTreeMap<String, Value>,
) -> Result<(), GraphModelError> {
    let value: WorkflowNodeMetadataV1 = validate_metadata(GraphModelClass::WorkflowNode, metadata)?;
    if value.entity_type != "workflow_node"
        || (!value.wf_terminal && value.wf_op.trim().is_empty())
        || value.wf_version.trim().is_empty()
        || value.wf_version.trim() != value.wf_version
        || value
            .wf_environment_id
            .as_deref()
            .is_some_and(|id| id.trim().is_empty() || id.trim() != id)
        || (value.wf_executor_target == Some(WorkflowExecutorTargetV1::PythonWorker)
            && value.wf_op.starts_with("rust-native/"))
    {
        return Err(GraphModelError::InvalidEntityFields {
            model_class: GraphModelClass::WorkflowNode,
            reason: "invalid workflow node metadata",
        });
    }
    Ok(())
}

fn validate_workflow_runtime_node_shape(
    model_class: GraphModelClass,
    payload: &Value,
    expected_entity_type: &str,
) -> Result<(), GraphModelError> {
    let typed = GraphModelEnvelope::new(model_class, payload.clone())
        .decode_persistent_shape(GraphEntityKind::Node)?;
    validate_workflow_runtime_metadata(model_class, &typed.core.metadata, expected_entity_type)
}

fn validate_workflow_runtime_metadata(
    model_class: GraphModelClass,
    metadata: &BTreeMap<String, Value>,
    expected_entity_type: &str,
) -> Result<(), GraphModelError> {
    let string_field = |field: &str, required: bool| {
        metadata.get(field).is_some_and(Value::is_string)
            || (!required && !metadata.contains_key(field))
    };
    let integer_field = |field: &str, required: bool| {
        metadata
            .get(field)
            .is_some_and(|value| python_integer_input(value).is_some())
            || (!required && !metadata.contains_key(field))
    };
    let entity_type_valid = metadata
        .get("entity_type")
        .is_none_or(|actual| actual.as_str() == Some(expected_entity_type));
    let fields_valid = match model_class {
        GraphModelClass::WorkflowRunNode => {
            ["workflow_id", "run_id", "conversation_id", "turn_node_id"]
                .into_iter()
                .all(|field| string_field(field, true))
                && string_field("workflow_version", false)
                && string_field("status", false)
        }
        GraphModelClass::WorkflowStepExecNode => {
            [
                "run_id",
                "workflow_id",
                "workflow_node_id",
                "op",
                "result_json",
            ]
            .into_iter()
            .all(|field| string_field(field, true))
                && integer_field("step_seq", true)
                && integer_field("duration_ms", false)
                && string_field("status", false)
                && metadata
                    .get("step_seq")
                    .and_then(python_integer_input)
                    .is_some_and(|nonnegative| nonnegative)
        }
        GraphModelClass::WorkflowCheckpointNode => {
            ["run_id", "workflow_id", "state_json"]
                .into_iter()
                .all(|field| string_field(field, true))
                && integer_field("step_seq", true)
                && metadata
                    .get("step_seq")
                    .and_then(python_integer_input)
                    .is_some_and(|nonnegative| nonnegative)
        }
        GraphModelClass::WorkflowCompletedNode
        | GraphModelClass::WorkflowCancelledNode
        | GraphModelClass::WorkflowFailedNode => {
            ["workflow_id", "run_id", "conversation_id"]
                .into_iter()
                .all(|field| string_field(field, true))
                && integer_field("accepted_step_seq", true)
                && (model_class != GraphModelClass::WorkflowFailedNode
                    || metadata.get("errors").is_none_or(|errors| {
                        errors
                            .as_array()
                            .is_some_and(|items| items.iter().all(Value::is_string))
                    }))
        }
        _ => false,
    };
    if !entity_type_valid || !fields_valid {
        return Err(GraphModelError::InvalidEntityFields {
            model_class,
            reason: "invalid workflow runtime node metadata",
        });
    }
    Ok(())
}

fn python_integer_input(value: &Value) -> Option<bool> {
    match value {
        Value::Number(number) => number
            .as_i64()
            .map(|number| number >= 0)
            .or_else(|| number.as_u64().map(|_| true))
            .or_else(|| {
                number.as_f64().and_then(|number| {
                    (number.is_finite() && number.fract() == 0.0).then_some(number >= 0.0)
                })
            })
            .or_else(|| python_integer_text_is_nonnegative(&number.to_string())),
        Value::String(value) => python_integer_text_is_nonnegative(value),
        Value::Bool(_) => Some(true),
        _ => None,
    }
}

fn python_integer_text_is_nonnegative(value: &str) -> Option<bool> {
    let value = value.trim();
    let (negative, digits) = match value.as_bytes().first() {
        Some(b'-') => (true, &value[1..]),
        Some(b'+') => (false, &value[1..]),
        _ => (false, value),
    };
    if digits.is_empty() {
        return None;
    }

    let mut previous_was_digit = false;
    let mut nonzero_digit = false;
    let mut digit_count = 0;
    for byte in digits.bytes() {
        match byte {
            b'0'..=b'9' => {
                previous_was_digit = true;
                nonzero_digit |= byte != b'0';
                digit_count += 1;
            }
            b'_' if previous_was_digit => previous_was_digit = false,
            _ => return None,
        }
    }
    if !previous_was_digit || digit_count == 0 {
        return None;
    }
    Some(!negative || !nonzero_digit)
}

fn validate_workflow_runtime_edge_metadata(
    metadata: &BTreeMap<String, Value>,
) -> Result<(), GraphModelError> {
    let is_string = |value: &Value| value.is_string();
    let relation_valid = metadata.get("relation").is_some_and(is_string);
    let optional_string_fields_valid = ["conversation_id", "run_id", "entity_type"]
        .into_iter()
        .all(|field| {
            metadata
                .get(field)
                .is_none_or(|value| value.is_null() && field != "entity_type" || is_string(value))
        });
    if !relation_valid || !optional_string_fields_valid {
        return Err(GraphModelError::InvalidEntityFields {
            model_class: GraphModelClass::WorkflowRuntimeEdge,
            reason: "invalid workflow runtime edge metadata",
        });
    }
    Ok(())
}

fn validate_workflow_edge_metadata(
    metadata: &BTreeMap<String, Value>,
) -> Result<(), GraphModelError> {
    let value: WorkflowEdgeMetadataV1 = validate_metadata(GraphModelClass::WorkflowEdge, metadata)?;
    if value.entity_type != "workflow_edge"
        || python_integer_input(&value.wf_priority) != Some(true)
        || !matches!(value.wf_multiplicity.as_str(), "one" | "many")
    {
        return Err(GraphModelError::InvalidEntityFields {
            model_class: GraphModelClass::WorkflowEdge,
            reason: "invalid workflow edge metadata",
        });
    }
    Ok(())
}

fn validate_conversation_node_payload(payload: &Value) -> Result<(), GraphModelError> {
    let metadata = raw_metadata(GraphModelClass::ConversationNode, payload)?;
    let value: ConversationNodeMetadataV1 =
        validate_metadata(GraphModelClass::ConversationNode, &metadata)?;
    let role_valid = metadata.get("role").is_none_or(|role| {
        role.is_null()
            || matches!(
                role.as_str(),
                Some("user" | "assistant" | "system" | "tool")
            )
    });
    let optional_fields_valid = ["conversation_id", "user_id"].into_iter().all(|field| {
        metadata
            .get(field)
            .is_none_or(|value| value.is_null() || value.is_string())
    });
    // Python projection: ConversationRoleMixin validates these serialized fields before
    // push_conversation_metadata mirrors non-null values into metadata.
    let top_level_role_valid = payload.get("role").is_none_or(|role| {
        role.is_null()
            || matches!(
                role.as_str(),
                Some("user" | "assistant" | "system" | "tool")
            )
    });
    let top_level_turn_index_valid = payload
        .get("turn_index")
        .is_none_or(|index| index.is_null() || python_integer_input(index).is_some());
    let top_level_identifiers_valid = ["conversation_id", "user_id"].into_iter().all(|field| {
        payload
            .get(field)
            .is_none_or(|value| value.is_null() || value.is_string())
    });
    let forbidden = [
        "char_distance_from_last_summary",
        "turn_distance_from_last_summary",
        "prev_node_char_distance_from_last_summary",
        "prev_node_distance_from_last_summary",
        "prev_turn_meta_summary",
    ];
    if python_integer_input(&value.level_from_root) != Some(true)
        || value
            .turn_index
            .as_ref()
            .is_some_and(|index| python_integer_input(index).is_none())
        || !role_valid
        || !optional_fields_valid
        || !top_level_role_valid
        || !top_level_turn_index_valid
        || !top_level_identifiers_valid
        || forbidden.iter().any(|key| metadata.contains_key(*key))
    {
        return Err(GraphModelError::InvalidEntityFields {
            model_class: GraphModelClass::ConversationNode,
            reason: "invalid conversation node metadata",
        });
    }
    Ok(())
}

fn validate_conversation_edge_metadata(
    metadata: &BTreeMap<String, Value>,
) -> Result<(), GraphModelError> {
    let _: ConversationEdgeMetadataV1 =
        validate_metadata(GraphModelClass::ConversationEdge, metadata)?;
    let numeric_fields_valid = [
        "char_distance_from_last_summary",
        "turn_distance_from_last_summary",
    ]
    .into_iter()
    .all(|field| {
        metadata
            .get(field)
            .is_none_or(|value| value.is_null() || python_integer_input(value).is_some())
    });
    if !numeric_fields_valid {
        return Err(GraphModelError::InvalidEntityFields {
            model_class: GraphModelClass::ConversationEdge,
            reason: "conversation edge distances must be integers or null",
        });
    }
    Ok(())
}

fn validate_llm_shape(
    model_class: GraphModelClass,
    payload: &Value,
) -> Result<(), GraphModelError> {
    if payload.get("embedding").is_some() {
        return Err(GraphModelError::InvalidEntityFields {
            model_class,
            reason: "LLM graph payload must not contain an embedding",
        });
    }
    if matches!(
        model_class,
        GraphModelClass::FlattenedLlmNode
            | GraphModelClass::FlattenedLlmEdge
            | GraphModelClass::AssocFlattenedLlmNode
            | GraphModelClass::AssocFlattenedLlmEdge
    ) {
        return validate_flattened_llm_shape(model_class, payload);
    }
    if model_class == GraphModelClass::LlmNodeExtraction {
        let node: LlmNodeExtractionV1 = serde_json::from_value(payload.clone())
            .map_err(|error| GraphModelError::InvalidPayload(error.to_string()))?;
        if node.mentions.is_empty() {
            return Err(GraphModelError::InvalidEntityFields {
                model_class,
                reason: "LLM extraction node requires mentions",
            });
        }
        for (index, span) in node.mentions.iter().enumerate() {
            if span.end_char <= span.start_char || span.page_number == 0 {
                return Err(GraphModelError::InvalidEntityFields {
                    model_class,
                    reason: "LLM extraction span bounds or page number are invalid",
                });
            }
            if !payload["mentions"][index]["excerpt"].is_string() {
                return Err(GraphModelError::InvalidEntityFields {
                    model_class,
                    reason: "LLM extraction spans require excerpt strings",
                });
            }
        }
        return validate_llm_ids(model_class, payload);
    }
    if model_class == GraphModelClass::LlmEdgeExtraction {
        let edge: LlmEdgeExtractionV1 = serde_json::from_value(payload.clone())
            .map_err(|error| GraphModelError::InvalidPayload(error.to_string()))?;
        if edge.mentions.is_empty() {
            return Err(GraphModelError::InvalidEntityFields {
                model_class,
                reason: "LLM extraction edge requires mentions",
            });
        }
        if edge
            .source_ids
            .iter()
            .chain(&edge.target_ids)
            .any(|endpoint| endpoint.0.is_empty())
            || edge
                .source_edge_ids
                .as_option()
                .into_iter()
                .flatten()
                .chain(edge.target_edge_ids.as_option().into_iter().flatten())
                .any(|endpoint| endpoint.0.is_empty())
        {
            return Err(GraphModelError::InvalidEntityFields {
                model_class,
                reason: "LLM edge endpoint IDs must not be empty",
            });
        }
        for (index, span) in edge.mentions.iter().enumerate() {
            if span.end_char <= span.start_char || span.page_number == 0 {
                return Err(GraphModelError::InvalidEntityFields {
                    model_class,
                    reason: "LLM extraction span bounds or page number are invalid",
                });
            }
            if !payload["mentions"][index]["excerpt"].is_string() {
                return Err(GraphModelError::InvalidEntityFields {
                    model_class,
                    reason: "LLM extraction spans require excerpt strings",
                });
            }
        }
        return validate_llm_ids(model_class, payload);
    }
    if model_class.entity_kind() == GraphEntityKind::Edge {
        let edge: LlmEdgeV1 = serde_json::from_value(payload.clone())
            .map_err(|error| GraphModelError::InvalidPayload(error.to_string()))?;
        validate_llm_groundings(model_class, &edge.mentions, payload)
    } else {
        let node: LlmNodeV1 = serde_json::from_value(payload.clone())
            .map_err(|error| GraphModelError::InvalidPayload(error.to_string()))?;
        validate_llm_groundings(model_class, &node.mentions, payload)
    }
}

fn validate_llm_groundings(
    model_class: GraphModelClass,
    mentions: &[LlmGroundingV1],
    payload: &Value,
) -> Result<(), GraphModelError> {
    if mentions.is_empty() {
        return Err(GraphModelError::InvalidEntityFields {
            model_class,
            reason: "LLM graph payload requires mentions",
        });
    }
    for grounding in mentions {
        grounding.validate(model_class)?;
        if matches!(
            model_class,
            GraphModelClass::LlmNodeExtraction | GraphModelClass::LlmEdgeExtraction
        ) && grounding.spans.iter().any(|span| span.excerpt.is_empty())
        {
            return Err(GraphModelError::InvalidEntityFields {
                model_class,
                reason: "LLM extraction spans require excerpts",
            });
        }
    }
    validate_llm_ids(model_class, payload)
}

fn validate_llm_ids(model_class: GraphModelClass, payload: &Value) -> Result<(), GraphModelError> {
    for id_field in ["id", "local_id"] {
        if payload
            .get(id_field)
            .is_some_and(|value| value.is_string() && value.as_str().unwrap_or("").is_empty())
        {
            return Err(GraphModelError::InvalidEntityFields {
                model_class,
                reason: "LLM IDs must not be empty",
            });
        }
    }
    Ok(())
}

fn validate_flattened_llm_shape(
    model_class: GraphModelClass,
    payload: &Value,
) -> Result<(), GraphModelError> {
    if model_class == GraphModelClass::AssocFlattenedLlmNode {
        let _: AssocFlattenedLlmNodeV1 = serde_json::from_value(payload.clone())
            .map_err(|error| GraphModelError::InvalidPayload(error.to_string()))?;
        return validate_llm_ids(model_class, payload);
    }
    if model_class == GraphModelClass::AssocFlattenedLlmEdge {
        let edge: AssocFlattenedLlmEdgeV1 = serde_json::from_value(payload.clone())
            .map_err(|error| GraphModelError::InvalidPayload(error.to_string()))?;
        if edge
            .source_ids
            .iter()
            .chain(edge.target_ids.iter())
            .any(|id| id.0.is_empty())
            || edge
                .source_edge_ids
                .as_option()
                .into_iter()
                .flatten()
                .chain(edge.target_edge_ids.as_option().into_iter().flatten())
                .any(|id| id.0.is_empty())
        {
            return Err(GraphModelError::InvalidEntityFields {
                model_class,
                reason: "edge endpoint IDs must not be empty",
            });
        }
        return validate_llm_ids(model_class, payload);
    }
    if model_class.entity_kind() == GraphEntityKind::Edge {
        let edge: FlattenedLlmEdgeV1 = serde_json::from_value(payload.clone())
            .map_err(|error| GraphModelError::InvalidPayload(error.to_string()))?;
        if edge
            .source_ids
            .iter()
            .chain(&edge.target_ids)
            .any(|endpoint| endpoint.0.is_empty())
            || edge
                .source_edge_ids
                .as_option()
                .into_iter()
                .flatten()
                .chain(edge.target_edge_ids.as_option().into_iter().flatten())
                .any(|endpoint| endpoint.0.is_empty())
        {
            return Err(GraphModelError::InvalidEntityFields {
                model_class,
                reason: "edge endpoint IDs must not be empty",
            });
        }
        if edge.mentions.is_empty() {
            return Err(GraphModelError::InvalidEntityFields {
                model_class,
                reason: "flattened LLM payload requires mentions",
            });
        }
        for grounding in &edge.mentions {
            grounding.validate(model_class)?;
        }
    } else {
        let node: FlattenedLlmNodeV1 = serde_json::from_value(payload.clone())
            .map_err(|error| GraphModelError::InvalidPayload(error.to_string()))?;
        if node.mentions.is_empty() {
            return Err(GraphModelError::InvalidEntityFields {
                model_class,
                reason: "flattened LLM payload requires mentions",
            });
        }
        for grounding in &node.mentions {
            grounding.validate(model_class)?;
        }
    }
    Ok(())
}

fn validate_pure_chroma_shape(
    model_class: GraphModelClass,
    payload: &Value,
) -> Result<(), GraphModelError> {
    if payload.get("mentions").is_some() {
        return Err(GraphModelError::InvalidEntityFields {
            model_class,
            reason: "PureChroma graph payload must not contain grounding",
        });
    }
    if model_class.entity_kind() == GraphEntityKind::Edge {
        let edge: PureChromaEdgeV1 = serde_json::from_value(payload.clone())
            .map_err(|error| GraphModelError::InvalidPayload(error.to_string()))?;
        if edge.fields.label.trim().is_empty()
            || edge.fields.summary.trim().is_empty()
            || edge
                .fields
                .properties
                .as_ref()
                .is_some_and(|value| !valid_graph_properties(value))
        {
            return Err(GraphModelError::InvalidEntityFields {
                model_class,
                reason: "invalid PureChroma label, summary, or properties",
            });
        }
        Ok(())
    } else {
        let node: PureChromaNodeV1 = serde_json::from_value(payload.clone())
            .map_err(|error| GraphModelError::InvalidPayload(error.to_string()))?;
        if node.fields.label.trim().is_empty()
            || node.fields.summary.trim().is_empty()
            || node
                .fields
                .properties
                .as_ref()
                .is_some_and(|value| !valid_graph_properties(value))
        {
            return Err(GraphModelError::InvalidEntityFields {
                model_class,
                reason: "invalid PureChroma label, summary, or properties",
            });
        }
        Ok(())
    }
}

fn validate_document_node_shape(payload: &Value) -> Result<(), GraphModelError> {
    let node: DocumentNodeV1 = serde_json::from_value(payload.clone())
        .map_err(|error| GraphModelError::InvalidPayload(format!("DocNode: {error}")))?;
    if node.doc_id.trim().is_empty() || node.mentions.is_empty() {
        return Err(GraphModelError::InvalidEntityFields {
            model_class: GraphModelClass::DocNode,
            reason: "DocNode requires a non-empty doc_id and grounding",
        });
    }
    for grounding in &node.mentions {
        grounding.validate(GraphModelClass::DocNode)?;
    }
    Ok(())
}

fn validate_acl_shape(
    model_class: GraphModelClass,
    payload: &Value,
) -> Result<(), GraphModelError> {
    // ACLNode/ACLEdge are persistent Node/Edge subclasses in Python. Validate
    // the shared persistent shape first so metadata checks cannot accept an
    // ACL edge without incidence fields.
    match model_class {
        GraphModelClass::AclNode => {
            let node: PersistentNodeV1 = serde_json::from_value(payload.clone())
                .map_err(|error| GraphModelError::InvalidPayload(format!("ACL node: {error}")))?;
            node.core.validate(model_class)?;
        }
        GraphModelClass::AclEdge => {
            let edge: PersistentEdgeV1 = serde_json::from_value(payload.clone())
                .map_err(|error| GraphModelError::InvalidPayload(format!("ACL edge: {error}")))?;
            edge.core.validate(model_class)?;
            if edge
                .source_ids
                .iter()
                .chain(edge.target_ids.iter())
                .any(|id| id.0.is_empty())
                || edge
                    .source_edge_ids
                    .as_option()
                    .into_iter()
                    .chain(edge.target_edge_ids.as_option())
                    .flatten()
                    .any(|id| id.0.is_empty())
            {
                return Err(GraphModelError::InvalidEntityFields {
                    model_class,
                    reason: "ACL edge endpoint IDs must not be empty",
                });
            }
        }
        _ => unreachable!("validate_acl_shape only accepts ACL model classes"),
    }
    let metadata = payload
        .get("metadata")
        .cloned()
        .ok_or_else(|| GraphModelError::InvalidPayload("ACL metadata is required".to_owned()))?;
    if model_class == GraphModelClass::AclNode {
        let value: AclNodeMetadataV1 = serde_json::from_value(metadata).map_err(|error| {
            GraphModelError::InvalidPayload(format!("ACL node metadata: {error}"))
        })?;
        if value.entity_type != "acl_record"
            || value.acl_truth_graph.trim().is_empty()
            || value.acl_target_entity_id.trim().is_empty()
            || value.acl_version == 0
            || value.level_from_root < 0
        {
            return Err(GraphModelError::InvalidEntityFields {
                model_class,
                reason: "invalid ACL node metadata",
            });
        }
    } else {
        let value: AclEdgeMetadataV1 = serde_json::from_value(metadata).map_err(|error| {
            GraphModelError::InvalidPayload(format!("ACL edge metadata: {error}"))
        })?;
        if value.acl_truth_graph.trim().is_empty()
            || value.acl_target_entity_id.trim().is_empty()
            || value.level_from_root < 0
        {
            return Err(GraphModelError::InvalidEntityFields {
                model_class,
                reason: "invalid ACL edge metadata",
            });
        }
    }
    Ok(())
}

fn require_wire_field(payload: &Value, field: &str) -> Result<(), GraphModelError> {
    if payload
        .as_object()
        .is_none_or(|object| !object.contains_key(field))
    {
        return Err(GraphModelError::InvalidPayload(format!(
            "missing required wire field {field}"
        )));
    }
    Ok(())
}

fn has_edge_fields(payload: &Value) -> bool {
    [
        "source_ids",
        "target_ids",
        "relation",
        "source_edge_ids",
        "target_edge_ids",
    ]
    .iter()
    .any(|field| payload.get(field).is_some())
}
