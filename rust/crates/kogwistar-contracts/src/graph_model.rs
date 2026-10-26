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
        if let Some(model_name) = manifest.model_name.as_deref() {
            if model_name != self.model_class.python_name() {
                return Err(GraphModelError::InvalidPayload(format!(
                    "slice manifest model {model_name:?} does not match envelope model {}",
                    self.model_class.python_name()
                )));
            }
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
                validate_conversation_node_metadata(&raw_metadata(
                    GraphModelClass::ConversationNode,
                    &self.payload,
                )?)
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

fn validate_conversation_node_metadata(
    metadata: &BTreeMap<String, Value>,
) -> Result<(), GraphModelError> {
    let value: ConversationNodeMetadataV1 =
        validate_metadata(GraphModelClass::ConversationNode, metadata)?;
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
    if model_class.entity_kind() == GraphEntityKind::Edge {
        let edge: FlattenedLlmEdgeV1 = serde_json::from_value(payload.clone())
            .map_err(|error| GraphModelError::InvalidPayload(error.to_string()))?;
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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
pub struct GraphEntityPayload {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub local_id: Option<String>,
    #[serde(default)]
    pub label: Option<String>,
    #[serde(rename = "type", default)]
    pub entity_type: Option<String>,
    #[serde(default)]
    pub summary: Option<String>,
    #[serde(default)]
    pub domain_id: Option<String>,
    #[serde(default)]
    pub canonical_entity_id: Option<String>,
    #[serde(default)]
    pub properties: Option<Value>,
    #[serde(default)]
    pub mentions: Option<Value>,
    #[serde(default)]
    pub doc_id: Option<String>,
    #[serde(default)]
    pub embedding: Option<Vec<f64>>,
    #[serde(default)]
    pub metadata: Option<Map<String, Value>>,
    #[serde(default)]
    pub level_from_root: Option<i64>,
    #[serde(default)]
    pub role: Option<Value>,
    #[serde(default)]
    pub turn_index: Option<Value>,
    #[serde(default)]
    pub conversation_id: Option<Value>,
    #[serde(default)]
    pub user_id: Option<Value>,
    #[serde(flatten)]
    pub extension_fields: Map<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
pub struct GraphNodePayload {
    #[serde(flatten)]
    pub common: GraphEntityPayload,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
pub struct GraphEdgePayload {
    #[serde(flatten)]
    pub common: GraphEntityPayload,
    #[serde(default)]
    pub source_ids: Option<Vec<NodeId>>,
    #[serde(default)]
    pub target_ids: Option<Vec<NodeId>>,
    #[serde(default)]
    pub relation: Option<String>,
    #[serde(default)]
    pub source_edge_ids: Option<Vec<EdgeId>>,
    #[serde(default)]
    pub target_edge_ids: Option<Vec<EdgeId>>,
}

/// A field which is required to be present on the wire, but may contain JSON
/// null. `Option<T>` alone cannot express that distinction in a Rust struct.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RequiredNullable<T> {
    Null,
    Value(T),
}

impl<T> RequiredNullable<T> {
    pub const fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }

    pub const fn as_option(&self) -> Option<&T> {
        match self {
            Self::Null => None,
            Self::Value(value) => Some(value),
        }
    }
}

impl<T> Serialize for RequiredNullable<T>
where
    T: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Null => serializer.serialize_none(),
            Self::Value(value) => value.serialize(serializer),
        }
    }
}

impl<'de, T> Deserialize<'de> for RequiredNullable<T>
where
    T: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Option::<T>::deserialize(deserializer).map(|value| value.map_or(Self::Null, Self::Value))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphEntityType {
    Entity,
    Relationship,
    ReferencePointer,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LifecycleStatus {
    Active,
    Tombstoned,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MentionVerificationMethod {
    Llm,
    Levenshtein,
    Regex,
    Heuristic,
    Human,
    Ensemble,
    System,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MentionVerificationV1 {
    pub method: MentionVerificationMethod,
    pub is_verified: bool,
    #[serde(default)]
    pub score: Option<f64>,
    #[serde(default)]
    pub notes: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TextSpanV1 {
    pub collection_page_url: String,
    pub document_page_url: String,
    pub doc_id: String,
    pub insertion_method: String,
    #[serde(default = "default_page_number")]
    pub page_number: u32,
    pub start_char: u64,
    pub end_char: u64,
    #[serde(default)]
    pub excerpt: String,
    #[serde(default)]
    pub context_before: String,
    #[serde(default)]
    pub context_after: String,
    #[serde(default)]
    pub chunk_id: Option<String>,
    #[serde(default)]
    pub source_cluster_id: Option<String>,
    #[serde(default)]
    pub verification: Option<MentionVerificationV1>,
}

/// LLM prompt evidence is intentionally leaner than persistent evidence:
/// backend insertion provenance is not part of the LLM projection.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LlmTextSpanV1 {
    pub collection_page_url: String,
    pub document_page_url: String,
    pub doc_id: String,
    #[serde(default = "default_page_number")]
    pub page_number: u32,
    pub start_char: u64,
    pub end_char: u64,
    #[serde(default)]
    pub excerpt: String,
    #[serde(default)]
    pub context_before: String,
    #[serde(default)]
    pub context_after: String,
    #[serde(default)]
    pub chunk_id: Option<String>,
    #[serde(default)]
    pub source_cluster_id: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LlmGroundingV1 {
    #[serde(default)]
    pub spans: Vec<LlmTextSpanV1>,
    #[serde(default)]
    pub multimodal_spans: Vec<Value>,
}

impl LlmGroundingV1 {
    fn validate(&self, model_class: GraphModelClass) -> Result<(), GraphModelError> {
        if self.spans.is_empty() && self.multimodal_spans.is_empty() {
            return Err(GraphModelError::InvalidEntityFields {
                model_class,
                reason: "LLM grounding requires text or multimodal evidence",
            });
        }
        for span in &self.spans {
            if span.end_char <= span.start_char || span.page_number == 0 {
                return Err(GraphModelError::InvalidEntityFields {
                    model_class,
                    reason: "LLM span bounds or page number are invalid",
                });
            }
        }
        Ok(())
    }
}

fn default_page_number() -> u32 {
    1
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GroundingV1 {
    #[serde(default)]
    pub spans: Vec<TextSpanV1>,
    /// Multimodal span schemas have independent versioning. Preserve them as
    /// typed-container members until their dedicated Rust contract is added.
    #[serde(default)]
    pub multimodal_spans: Vec<Value>,
}

impl GroundingV1 {
    fn validate(&self, model_class: GraphModelClass) -> Result<(), GraphModelError> {
        if self.spans.is_empty() && self.multimodal_spans.is_empty() {
            return Err(GraphModelError::InvalidEntityFields {
                model_class,
                reason: "grounding requires text or multimodal evidence",
            });
        }
        for span in &self.spans {
            if span.end_char <= span.start_char {
                return Err(GraphModelError::InvalidEntityFields {
                    model_class,
                    reason: "span end_char must be greater than start_char",
                });
            }
            if span.page_number == 0 {
                return Err(GraphModelError::InvalidEntityFields {
                    model_class,
                    reason: "span page_number must be positive",
                });
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GraphEntityCoreV1 {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub local_id: Option<String>,
    pub label: String,
    #[serde(rename = "type")]
    pub entity_type: GraphEntityType,
    pub summary: String,
    #[serde(default)]
    pub domain_id: Option<String>,
    #[serde(default)]
    pub canonical_entity_id: Option<String>,
    #[serde(default)]
    pub properties: Option<Value>,
    pub mentions: Vec<GroundingV1>,
    #[serde(default)]
    pub doc_id: Option<String>,
    #[serde(default)]
    pub embedding: Option<Vec<f64>>,
    #[serde(default)]
    pub metadata: BTreeMap<String, Value>,
    #[serde(default, deserialize_with = "deserialize_python_integer_option")]
    pub level_from_root: Option<i64>,
    #[serde(flatten)]
    pub extension_fields: BTreeMap<String, Value>,
}

fn coerce_python_integer(value: &Value) -> Result<Option<i64>, &'static str> {
    match value {
        Value::Null => Ok(None),
        Value::Bool(value) => Ok(Some(i64::from(*value))),
        Value::Number(value) => {
            if let Some(integer) = value.as_i64() {
                return Ok(Some(integer));
            }
            if let Some(unsigned) = value.as_u64() {
                return i64::try_from(unsigned)
                    .map(Some)
                    .map_err(|_| "integer is outside the Rust i64 range");
            }
            let float = value.as_f64().ok_or("value is not an integer")?;
            if !float.is_finite() || float.fract() != 0.0 {
                return Err("value is not an integer");
            }
            if float < i64::MIN as f64 || float >= -(i64::MIN as f64) {
                return Err("integer is outside the Rust i64 range");
            }
            Ok(Some(float as i64))
        }
        Value::String(value) => value
            .trim()
            .parse::<i64>()
            .map(Some)
            .map_err(|_| "string is not a valid integer"),
        _ => Err("value is not a valid integer"),
    }
}

fn deserialize_python_integer_option<'de, D>(deserializer: D) -> Result<Option<i64>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Value::deserialize(deserializer)?;
    coerce_python_integer(&value).map_err(serde::de::Error::custom)
}

impl GraphEntityCoreV1 {
    fn normalize_shared_mixin_fields(
        &mut self,
        model_class: GraphModelClass,
        level_field_present: bool,
    ) -> Result<(), GraphModelError> {
        match self.metadata.get("lifecycle_status") {
            Some(Value::String(status)) if status == "deleted" => {
                self.metadata.insert(
                    "lifecycle_status".to_owned(),
                    Value::String("tombstoned".to_owned()),
                );
            }
            Some(_) => {}
            None => {
                self.metadata.insert(
                    "lifecycle_status".to_owned(),
                    Value::String("active".to_owned()),
                );
            }
        }
        self.metadata
            .entry("redirect_to_id".to_owned())
            .or_insert(Value::Null);

        if let Some(level) = self.level_from_root {
            self.metadata
                .insert("level_from_root".to_owned(), Value::Number(level.into()));
        } else if !level_field_present {
            if let Some(value) = self.metadata.get("level_from_root") {
                self.level_from_root = coerce_python_integer(value).map_err(|reason| {
                    GraphModelError::InvalidEntityFields {
                        model_class,
                        reason,
                    }
                })?;
                if let Some(level) = self.level_from_root {
                    self.metadata
                        .insert("level_from_root".to_owned(), Value::Number(level.into()));
                }
            }
        }
        Ok(())
    }

    fn validate(&self, model_class: GraphModelClass) -> Result<(), GraphModelError> {
        for id in [&self.id, &self.local_id] {
            if id.as_deref().is_some_and(str::is_empty) {
                return Err(GraphModelError::InvalidEntityFields {
                    model_class,
                    reason: "entity IDs must not be empty",
                });
            }
        }
        if self.mentions.is_empty() {
            return Err(GraphModelError::InvalidEntityFields {
                model_class,
                reason: "persistent entity requires at least one grounding",
            });
        }
        if self
            .properties
            .as_ref()
            .is_some_and(|value| !valid_graph_properties(value))
        {
            return Err(GraphModelError::InvalidEntityFields {
                model_class,
                reason: "properties must contain only scalars, scalar lists, or scalar maps",
            });
        }
        for grounding in &self.mentions {
            grounding.validate(model_class)?;
        }
        if let Some(status) = self.metadata.get("lifecycle_status") {
            if !matches!(status.as_str(), Some("active" | "tombstoned")) {
                return Err(GraphModelError::InvalidEntityFields {
                    model_class,
                    reason: "lifecycle_status must be active or tombstoned",
                });
            }
        }
        if let Some(redirect) = self.metadata.get("redirect_to_id")
            && !redirect.is_null()
            && redirect.as_str().is_none()
        {
            return Err(GraphModelError::InvalidEntityFields {
                model_class,
                reason: "redirect_to_id must be a string or null",
            });
        }
        Ok(())
    }

    fn effective_level(&self) -> Option<i64> {
        self.level_from_root
    }
}

fn is_json_primitive(value: &Value) -> bool {
    matches!(
        value,
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_)
    )
}

fn valid_graph_properties(value: &Value) -> bool {
    let Some(properties) = value.as_object() else {
        return false;
    };
    properties.values().all(|value| match value {
        scalar if is_json_primitive(scalar) => true,
        Value::Array(items) => items.iter().all(is_json_primitive),
        Value::Object(items) => items.values().all(is_json_primitive),
        _ => false,
    })
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PersistentNodeV1 {
    #[serde(flatten)]
    pub core: GraphEntityCoreV1,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PersistentEdgeV1 {
    #[serde(flatten)]
    pub core: GraphEntityCoreV1,
    pub source_ids: Vec<NodeId>,
    pub target_ids: Vec<NodeId>,
    pub relation: String,
    pub source_edge_ids: RequiredNullable<Vec<EdgeId>>,
    pub target_edge_ids: RequiredNullable<Vec<EdgeId>>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum PersistentGraphEntityV1 {
    Node(PersistentNodeV1),
    Edge(PersistentEdgeV1),
}

/// Additive persistent graph view that keeps hierarchy levels outside the
/// legacy signed-64-bit field while retaining the unmodified source payload.
#[derive(Clone, Debug)]
pub struct PersistentGraphEntityV2 {
    pub entity: PersistentGraphEntityV1,
    pub level_from_root: Option<PythonIntegerV1>,
    pub raw_payload: Box<RawValue>,
}

impl PersistentGraphEntityV2 {
    pub fn as_node(&self) -> Option<&PersistentNodeV1> {
        match &self.entity {
            PersistentGraphEntityV1::Node(node) => Some(node),
            PersistentGraphEntityV1::Edge(_) => None,
        }
    }

    pub fn as_edge(&self) -> Option<&PersistentEdgeV1> {
        match &self.entity {
            PersistentGraphEntityV1::Node(_) => None,
            PersistentGraphEntityV1::Edge(edge) => Some(edge),
        }
    }
}

/// Typed metadata contracts for graph families whose Python models extend the
/// persistent Node/Edge shape. These are validation views only; they are not
/// serialized into the legacy payload or used by storage.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WorkflowNodeMetadataV1 {
    #[serde(default = "workflow_node_entity_type")]
    pub entity_type: String,
    pub workflow_id: String,
    #[serde(default = "workflow_noop")]
    pub wf_op: String,
    #[serde(default = "workflow_v1")]
    pub wf_version: String,
    #[serde(default)]
    pub wf_environment_id: Option<String>,
    #[serde(default)]
    pub wf_executor_target: Option<WorkflowExecutorTargetV1>,
    #[serde(default)]
    pub wf_start: bool,
    #[serde(default)]
    pub wf_terminal: bool,
    #[serde(default)]
    pub wf_fanout: bool,
    #[serde(default)]
    pub wf_join: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum WorkflowExecutorTargetV1 {
    PythonWorker,
    RustNative,
}

fn workflow_node_entity_type() -> String {
    "workflow_node".to_owned()
}

fn workflow_edge_entity_type() -> String {
    "workflow_edge".to_owned()
}

fn workflow_noop() -> String {
    "noop".to_owned()
}

fn workflow_v1() -> String {
    "v1".to_owned()
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WorkflowEdgeMetadataV1 {
    #[serde(default = "workflow_edge_entity_type")]
    pub entity_type: String,
    pub workflow_id: String,
    #[serde(default)]
    pub wf_predicate: Option<String>,
    #[serde(default = "workflow_default_priority")]
    pub wf_priority: Value,
    #[serde(default)]
    pub wf_is_default: bool,
    #[serde(default = "workflow_one")]
    pub wf_multiplicity: String,
}

fn workflow_default_priority() -> Value {
    Value::from(100)
}

fn workflow_one() -> String {
    "one".to_owned()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConversationRoleV1 {
    User,
    Assistant,
    System,
    Tool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConversationNodeMetadataV1 {
    #[serde(default = "conversation_node_entity_type")]
    pub entity_type: String,
    pub level_from_root: Value,
    #[serde(default)]
    pub role: Option<ConversationRoleV1>,
    #[serde(default)]
    pub turn_index: Option<Value>,
    #[serde(default)]
    pub conversation_id: Option<String>,
    #[serde(default)]
    pub user_id: Option<String>,
}

fn conversation_node_entity_type() -> String {
    "conversation_node".to_owned()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConversationCausalTypeV1 {
    Chain,
    Dependency,
    Reuse,
    Annotation,
    Summary,
    Reference,
    Internal,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConversationEdgeMetadataV1 {
    #[serde(default)]
    pub causal_type: Option<ConversationCausalTypeV1>,
    #[serde(default)]
    pub char_distance_from_last_summary: Option<Value>,
    #[serde(default)]
    pub turn_distance_from_last_summary: Option<Value>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DocumentNodeTypeV1 {
    Page,
    Chunk,
    Summary,
}

/// Common LLM-facing fields. Unlike the persistent core, this deliberately
/// has no embedding and does not carry storage-only lifecycle or hierarchy
/// fields.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LlmEntityFieldsV1 {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub local_id: Option<String>,
    pub label: String,
    #[serde(rename = "type")]
    pub entity_type: GraphEntityType,
    pub summary: String,
    #[serde(default)]
    pub domain_id: Option<String>,
    #[serde(default)]
    pub canonical_entity_id: Option<String>,
    #[serde(default)]
    pub properties: Option<Value>,
    #[serde(default)]
    pub doc_id: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LlmNodeV1 {
    #[serde(flatten)]
    pub fields: LlmEntityFieldsV1,
    pub mentions: Vec<LlmGroundingV1>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LlmEdgeV1 {
    #[serde(flatten)]
    pub fields: LlmEntityFieldsV1,
    pub source_ids: Vec<NodeId>,
    pub target_ids: Vec<NodeId>,
    pub relation: String,
    pub source_edge_ids: RequiredNullable<Vec<EdgeId>>,
    pub target_edge_ids: RequiredNullable<Vec<EdgeId>>,
    pub mentions: Vec<LlmGroundingV1>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FlattenedGroundingV1 {
    #[serde(default)]
    pub span_ids: Vec<String>,
    #[serde(default)]
    pub multimodal_span_ids: Vec<String>,
}

impl FlattenedGroundingV1 {
    fn validate(&self, model_class: GraphModelClass) -> Result<(), GraphModelError> {
        if self.span_ids.is_empty() && self.multimodal_span_ids.is_empty() {
            return Err(GraphModelError::InvalidEntityFields {
                model_class,
                reason: "flattened grounding requires at least one span ID",
            });
        }
        if self
            .span_ids
            .iter()
            .chain(self.multimodal_span_ids.iter())
            .any(|id| id.trim().is_empty())
        {
            return Err(GraphModelError::InvalidEntityFields {
                model_class,
                reason: "flattened grounding IDs must not be empty",
            });
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FlattenedLlmNodeV1 {
    #[serde(flatten)]
    pub fields: LlmEntityFieldsV1,
    pub mentions: Vec<FlattenedGroundingV1>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FlattenedLlmEdgeV1 {
    #[serde(flatten)]
    pub fields: LlmEntityFieldsV1,
    pub source_ids: Vec<NodeId>,
    pub target_ids: Vec<NodeId>,
    pub relation: String,
    pub source_edge_ids: RequiredNullable<Vec<EdgeId>>,
    pub target_edge_ids: RequiredNullable<Vec<EdgeId>>,
    pub mentions: Vec<FlattenedGroundingV1>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PureChromaFieldsV1 {
    #[serde(default)]
    pub id: Option<String>,
    pub label: String,
    #[serde(rename = "type")]
    pub entity_type: GraphEntityType,
    pub summary: String,
    #[serde(default)]
    pub doc_id: Option<String>,
    #[serde(default)]
    pub embedding: Option<Vec<f64>>,
    #[serde(default)]
    pub metadata: BTreeMap<String, Value>,
    #[serde(default)]
    pub properties: Option<Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PureChromaNodeV1 {
    #[serde(flatten)]
    pub fields: PureChromaFieldsV1,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PureChromaEdgeV1 {
    #[serde(flatten)]
    pub fields: PureChromaFieldsV1,
    pub source_ids: Vec<NodeId>,
    pub target_ids: Vec<NodeId>,
    pub relation: String,
    pub source_edge_ids: RequiredNullable<Vec<EdgeId>>,
    pub target_edge_ids: RequiredNullable<Vec<EdgeId>>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DocumentNodeV1 {
    #[serde(default)]
    pub id: Option<String>,
    pub label: String,
    #[serde(rename = "type")]
    pub node_type: DocumentNodeTypeV1,
    pub summary: String,
    pub doc_id: String,
    pub mentions: Vec<GroundingV1>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AclNodeMetadataV1 {
    pub entity_type: String,
    pub acl_truth_graph: String,
    pub acl_target_entity_id: String,
    pub acl_target_grain: AclGrainV1,
    #[serde(default)]
    pub acl_target_item_id: String,
    pub acl_version: u64,
    pub acl_mode: AclModeV1,
    #[serde(default)]
    pub created_by: Option<String>,
    #[serde(default)]
    pub owner_id: Option<String>,
    #[serde(default)]
    pub security_scope: Option<String>,
    #[serde(default)]
    pub tombstoned: bool,
    #[serde(default)]
    pub supersedes_version: Option<u64>,
    #[serde(default)]
    pub level_from_root: i64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AclEdgeMetadataV1 {
    pub entity_type: AclEdgeTypeV1,
    pub acl_truth_graph: String,
    pub acl_target_entity_id: String,
    pub acl_target_grain: AclGrainV1,
    #[serde(default)]
    pub acl_target_item_id: String,
    #[serde(default)]
    pub level_from_root: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AclGrainV1 {
    Document,
    Grounding,
    Span,
    Node,
    Edge,
    Artifact,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AclModeV1 {
    Private,
    Shared,
    Scope,
    Group,
    Public,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AclEdgeTypeV1 {
    AclSupersedes,
    AclTargetsTruth,
    AclCoversUsage,
}

impl PersistentNodeV1 {
    fn into_validated(
        mut self,
        model_class: GraphModelClass,
        level_field_present: bool,
    ) -> Result<PersistentGraphEntityV1, GraphModelError> {
        self.core
            .normalize_shared_mixin_fields(model_class, level_field_present)?;
        self.core.validate(GraphModelClass::Node)?;
        Ok(PersistentGraphEntityV1::Node(self))
    }
}

impl PersistentEdgeV1 {
    fn into_validated(
        mut self,
        model_class: GraphModelClass,
        level_field_present: bool,
    ) -> Result<PersistentGraphEntityV1, GraphModelError> {
        self.core
            .normalize_shared_mixin_fields(model_class, level_field_present)?;
        self.core.validate(GraphModelClass::Edge)?;
        if self
            .source_ids
            .iter()
            .chain(self.target_ids.iter())
            .any(|id| id.0.is_empty())
        {
            return Err(GraphModelError::InvalidEntityFields {
                model_class: GraphModelClass::Edge,
                reason: "node endpoint IDs must not be empty",
            });
        }
        if self
            .source_edge_ids
            .as_option()
            .into_iter()
            .chain(self.target_edge_ids.as_option())
            .flatten()
            .any(|id| id.0.is_empty())
        {
            return Err(GraphModelError::InvalidEntityFields {
                model_class: GraphModelClass::Edge,
                reason: "edge endpoint IDs must not be empty",
            });
        }
        Ok(PersistentGraphEntityV1::Edge(self))
    }
}

/// Permissive, loss-preserving inspection view for legacy payloads. Use
/// `PersistentGraphEntityV1` when semantic validation is required.
#[derive(Clone, Debug, PartialEq)]
pub enum TypedGraphEntity {
    Node(GraphNodePayload),
    Edge(GraphEdgePayload),
}

/// Shared structural accessors corresponding to Python's `NodeLike`.
/// `GraphEntity` intentionally does not require a concrete storage layout.
pub trait GraphEntity {
    fn graph_entity_kind(&self) -> GraphEntityKind;
    fn graph_entity_id(&self) -> Option<&str>;
    fn graph_entity_label(&self) -> Option<&str>;
    fn graph_entity_summary(&self) -> Option<&str>;
}

/// Edge-specific incidence behavior corresponding to Python's
/// `EdgeLike(NodeLike)`. It is a capability extension, not node containment.
pub trait EdgeEntity: GraphEntity {
    fn source_node_ids(&self) -> Option<&[NodeId]>;
    fn target_node_ids(&self) -> Option<&[NodeId]>;
    fn relation(&self) -> Option<&str>;
    fn source_edge_ids(&self) -> Option<&[EdgeId]>;
    fn target_edge_ids(&self) -> Option<&[EdgeId]>;

    /// Python represents hyperedges as ordinary `Edge` identities with more
    /// than one node incidence or with edge-to-edge incidence. Keep that
    /// meaning at the shared interface instead of inventing a second wire
    /// entity whose IDs could not round-trip through legacy payloads.
    fn is_hyperedge(&self) -> bool {
        self.source_node_ids().is_some_and(|ids| ids.len() > 1)
            || self.target_node_ids().is_some_and(|ids| ids.len() > 1)
            || self.source_edge_ids().is_some_and(|ids| !ids.is_empty())
            || self.target_edge_ids().is_some_and(|ids| !ids.is_empty())
    }
}

pub trait HasMetadata {
    fn metadata(&self) -> &BTreeMap<String, Value>;
}

pub trait HasLifecycle: HasMetadata {
    fn lifecycle_status(&self) -> LifecycleStatus {
        match self
            .metadata()
            .get("lifecycle_status")
            .and_then(Value::as_str)
        {
            Some("tombstoned") => LifecycleStatus::Tombstoned,
            _ => LifecycleStatus::Active,
        }
    }
}

pub trait HasLevel: HasMetadata {
    fn level_from_root(&self) -> Option<i64>;
}

/// Exact Python-compatible hierarchy value for additive graph-model views.
/// `HasLevel` remains the legacy bounded interface for source compatibility.
pub trait HasExactLevel {
    fn exact_level_from_root(&self) -> Option<&PythonIntegerV1>;
}

impl GraphEntity for GraphNodePayload {
    fn graph_entity_kind(&self) -> GraphEntityKind {
        GraphEntityKind::Node
    }

    fn graph_entity_id(&self) -> Option<&str> {
        self.common.id.as_deref()
    }

    fn graph_entity_label(&self) -> Option<&str> {
        self.common.label.as_deref()
    }

    fn graph_entity_summary(&self) -> Option<&str> {
        self.common.summary.as_deref()
    }
}

impl GraphEntity for GraphEdgePayload {
    fn graph_entity_kind(&self) -> GraphEntityKind {
        GraphEntityKind::Edge
    }

    fn graph_entity_id(&self) -> Option<&str> {
        self.common.id.as_deref()
    }

    fn graph_entity_label(&self) -> Option<&str> {
        self.common.label.as_deref()
    }

    fn graph_entity_summary(&self) -> Option<&str> {
        self.common.summary.as_deref()
    }
}

impl EdgeEntity for GraphEdgePayload {
    fn source_node_ids(&self) -> Option<&[NodeId]> {
        self.source_ids.as_deref()
    }

    fn target_node_ids(&self) -> Option<&[NodeId]> {
        self.target_ids.as_deref()
    }

    fn relation(&self) -> Option<&str> {
        self.relation.as_deref()
    }

    fn source_edge_ids(&self) -> Option<&[EdgeId]> {
        self.source_edge_ids.as_deref()
    }

    fn target_edge_ids(&self) -> Option<&[EdgeId]> {
        self.target_edge_ids.as_deref()
    }
}

impl GraphEntity for PersistentNodeV1 {
    fn graph_entity_kind(&self) -> GraphEntityKind {
        GraphEntityKind::Node
    }

    fn graph_entity_id(&self) -> Option<&str> {
        self.core.id.as_deref()
    }

    fn graph_entity_label(&self) -> Option<&str> {
        Some(&self.core.label)
    }

    fn graph_entity_summary(&self) -> Option<&str> {
        Some(&self.core.summary)
    }
}

impl GraphEntity for PersistentEdgeV1 {
    fn graph_entity_kind(&self) -> GraphEntityKind {
        GraphEntityKind::Edge
    }

    fn graph_entity_id(&self) -> Option<&str> {
        self.core.id.as_deref()
    }

    fn graph_entity_label(&self) -> Option<&str> {
        Some(&self.core.label)
    }

    fn graph_entity_summary(&self) -> Option<&str> {
        Some(&self.core.summary)
    }
}

impl EdgeEntity for PersistentEdgeV1 {
    fn source_node_ids(&self) -> Option<&[NodeId]> {
        Some(&self.source_ids)
    }

    fn target_node_ids(&self) -> Option<&[NodeId]> {
        Some(&self.target_ids)
    }

    fn relation(&self) -> Option<&str> {
        Some(&self.relation)
    }

    fn source_edge_ids(&self) -> Option<&[EdgeId]> {
        self.source_edge_ids.as_option().map(Vec::as_slice)
    }

    fn target_edge_ids(&self) -> Option<&[EdgeId]> {
        self.target_edge_ids.as_option().map(Vec::as_slice)
    }
}

impl GraphEntity for LlmNodeV1 {
    fn graph_entity_kind(&self) -> GraphEntityKind {
        GraphEntityKind::Node
    }

    fn graph_entity_id(&self) -> Option<&str> {
        self.fields.id.as_deref()
    }

    fn graph_entity_label(&self) -> Option<&str> {
        Some(&self.fields.label)
    }

    fn graph_entity_summary(&self) -> Option<&str> {
        Some(&self.fields.summary)
    }
}

impl GraphEntity for LlmEdgeV1 {
    fn graph_entity_kind(&self) -> GraphEntityKind {
        GraphEntityKind::Edge
    }

    fn graph_entity_id(&self) -> Option<&str> {
        self.fields.id.as_deref()
    }

    fn graph_entity_label(&self) -> Option<&str> {
        Some(&self.fields.label)
    }

    fn graph_entity_summary(&self) -> Option<&str> {
        Some(&self.fields.summary)
    }
}

impl EdgeEntity for LlmEdgeV1 {
    fn source_node_ids(&self) -> Option<&[NodeId]> {
        Some(&self.source_ids)
    }

    fn target_node_ids(&self) -> Option<&[NodeId]> {
        Some(&self.target_ids)
    }

    fn relation(&self) -> Option<&str> {
        Some(&self.relation)
    }

    fn source_edge_ids(&self) -> Option<&[EdgeId]> {
        self.source_edge_ids.as_option().map(Vec::as_slice)
    }

    fn target_edge_ids(&self) -> Option<&[EdgeId]> {
        self.target_edge_ids.as_option().map(Vec::as_slice)
    }
}

impl GraphEntity for FlattenedLlmNodeV1 {
    fn graph_entity_kind(&self) -> GraphEntityKind {
        GraphEntityKind::Node
    }

    fn graph_entity_id(&self) -> Option<&str> {
        self.fields.id.as_deref()
    }

    fn graph_entity_label(&self) -> Option<&str> {
        Some(&self.fields.label)
    }

    fn graph_entity_summary(&self) -> Option<&str> {
        Some(&self.fields.summary)
    }
}

impl GraphEntity for FlattenedLlmEdgeV1 {
    fn graph_entity_kind(&self) -> GraphEntityKind {
        GraphEntityKind::Edge
    }

    fn graph_entity_id(&self) -> Option<&str> {
        self.fields.id.as_deref()
    }

    fn graph_entity_label(&self) -> Option<&str> {
        Some(&self.fields.label)
    }

    fn graph_entity_summary(&self) -> Option<&str> {
        Some(&self.fields.summary)
    }
}

impl EdgeEntity for FlattenedLlmEdgeV1 {
    fn source_node_ids(&self) -> Option<&[NodeId]> {
        Some(&self.source_ids)
    }

    fn target_node_ids(&self) -> Option<&[NodeId]> {
        Some(&self.target_ids)
    }

    fn relation(&self) -> Option<&str> {
        Some(&self.relation)
    }

    fn source_edge_ids(&self) -> Option<&[EdgeId]> {
        self.source_edge_ids.as_option().map(Vec::as_slice)
    }

    fn target_edge_ids(&self) -> Option<&[EdgeId]> {
        self.target_edge_ids.as_option().map(Vec::as_slice)
    }
}

impl GraphEntity for PureChromaNodeV1 {
    fn graph_entity_kind(&self) -> GraphEntityKind {
        GraphEntityKind::Node
    }

    fn graph_entity_id(&self) -> Option<&str> {
        self.fields.id.as_deref()
    }

    fn graph_entity_label(&self) -> Option<&str> {
        Some(&self.fields.label)
    }

    fn graph_entity_summary(&self) -> Option<&str> {
        Some(&self.fields.summary)
    }
}

impl HasMetadata for PureChromaNodeV1 {
    fn metadata(&self) -> &BTreeMap<String, Value> {
        &self.fields.metadata
    }
}

impl GraphEntity for PureChromaEdgeV1 {
    fn graph_entity_kind(&self) -> GraphEntityKind {
        GraphEntityKind::Edge
    }

    fn graph_entity_id(&self) -> Option<&str> {
        self.fields.id.as_deref()
    }

    fn graph_entity_label(&self) -> Option<&str> {
        Some(&self.fields.label)
    }

    fn graph_entity_summary(&self) -> Option<&str> {
        Some(&self.fields.summary)
    }
}

impl EdgeEntity for PureChromaEdgeV1 {
    fn source_node_ids(&self) -> Option<&[NodeId]> {
        Some(&self.source_ids)
    }

    fn target_node_ids(&self) -> Option<&[NodeId]> {
        Some(&self.target_ids)
    }

    fn relation(&self) -> Option<&str> {
        Some(&self.relation)
    }

    fn source_edge_ids(&self) -> Option<&[EdgeId]> {
        self.source_edge_ids.as_option().map(Vec::as_slice)
    }

    fn target_edge_ids(&self) -> Option<&[EdgeId]> {
        self.target_edge_ids.as_option().map(Vec::as_slice)
    }
}

impl HasMetadata for PureChromaEdgeV1 {
    fn metadata(&self) -> &BTreeMap<String, Value> {
        &self.fields.metadata
    }
}

impl GraphEntity for DocumentNodeV1 {
    fn graph_entity_kind(&self) -> GraphEntityKind {
        GraphEntityKind::Node
    }

    fn graph_entity_id(&self) -> Option<&str> {
        self.id.as_deref()
    }

    fn graph_entity_label(&self) -> Option<&str> {
        Some(&self.label)
    }

    fn graph_entity_summary(&self) -> Option<&str> {
        Some(&self.summary)
    }
}

impl HasMetadata for PersistentNodeV1 {
    fn metadata(&self) -> &BTreeMap<String, Value> {
        &self.core.metadata
    }
}

impl HasMetadata for PersistentEdgeV1 {
    fn metadata(&self) -> &BTreeMap<String, Value> {
        &self.core.metadata
    }
}

impl HasLifecycle for PersistentNodeV1 {}
impl HasLifecycle for PersistentEdgeV1 {}

impl HasLevel for PersistentNodeV1 {
    fn level_from_root(&self) -> Option<i64> {
        self.core.effective_level()
    }
}

impl HasLevel for PersistentEdgeV1 {
    fn level_from_root(&self) -> Option<i64> {
        self.core.effective_level()
    }
}

impl GraphEntity for PersistentGraphEntityV2 {
    fn graph_entity_kind(&self) -> GraphEntityKind {
        self.entity.graph_entity_kind()
    }

    fn graph_entity_id(&self) -> Option<&str> {
        self.entity.graph_entity_id()
    }

    fn graph_entity_label(&self) -> Option<&str> {
        self.entity.graph_entity_label()
    }

    fn graph_entity_summary(&self) -> Option<&str> {
        self.entity.graph_entity_summary()
    }
}

impl HasMetadata for PersistentGraphEntityV2 {
    fn metadata(&self) -> &BTreeMap<String, Value> {
        match &self.entity {
            PersistentGraphEntityV1::Node(node) => node.metadata(),
            PersistentGraphEntityV1::Edge(edge) => edge.metadata(),
        }
    }
}

impl HasLifecycle for PersistentGraphEntityV2 {}

impl HasLevel for PersistentGraphEntityV2 {
    fn level_from_root(&self) -> Option<i64> {
        self.level_from_root
            .as_ref()
            .and_then(PythonIntegerV1::to_i64)
    }
}

impl HasExactLevel for PersistentGraphEntityV2 {
    fn exact_level_from_root(&self) -> Option<&PythonIntegerV1> {
        self.level_from_root.as_ref()
    }
}

impl GraphEntity for PersistentGraphEntityV1 {
    fn graph_entity_kind(&self) -> GraphEntityKind {
        match self {
            Self::Node(node) => node.graph_entity_kind(),
            Self::Edge(edge) => edge.graph_entity_kind(),
        }
    }

    fn graph_entity_id(&self) -> Option<&str> {
        match self {
            Self::Node(node) => node.graph_entity_id(),
            Self::Edge(edge) => edge.graph_entity_id(),
        }
    }

    fn graph_entity_label(&self) -> Option<&str> {
        match self {
            Self::Node(node) => node.graph_entity_label(),
            Self::Edge(edge) => edge.graph_entity_label(),
        }
    }

    fn graph_entity_summary(&self) -> Option<&str> {
        match self {
            Self::Node(node) => node.graph_entity_summary(),
            Self::Edge(edge) => edge.graph_entity_summary(),
        }
    }
}

impl GraphEntity for TypedGraphEntity {
    fn graph_entity_kind(&self) -> GraphEntityKind {
        match self {
            Self::Node(_) => GraphEntityKind::Node,
            Self::Edge(_) => GraphEntityKind::Edge,
        }
    }

    fn graph_entity_id(&self) -> Option<&str> {
        match self {
            Self::Node(node) => node.graph_entity_id(),
            Self::Edge(edge) => edge.graph_entity_id(),
        }
    }

    fn graph_entity_label(&self) -> Option<&str> {
        match self {
            Self::Node(node) => node.graph_entity_label(),
            Self::Edge(edge) => edge.graph_entity_label(),
        }
    }

    fn graph_entity_summary(&self) -> Option<&str> {
        match self {
            Self::Node(node) => node.graph_entity_summary(),
            Self::Edge(edge) => edge.graph_entity_summary(),
        }
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum GraphModelError {
    #[error("unsupported graph model envelope version {0}")]
    UnsupportedVersion(u16),
    #[error("invalid graph model payload: {0}")]
    InvalidPayload(String),
    #[error("graph model {model_class:?} has invalid entity fields: {reason}")]
    InvalidEntityFields {
        model_class: GraphModelClass,
        reason: &'static str,
    },
    #[error("strict graph model decoding does not support model class {model_class:?}")]
    StrictModelClassUnsupported { model_class: GraphModelClass },
}

impl GraphEntityPayload {
    fn validate(&self, model_class: GraphModelClass) -> Result<(), GraphModelError> {
        if let Some(id) = &self.id {
            if id.is_empty() {
                return Err(GraphModelError::InvalidEntityFields {
                    model_class,
                    reason: "id must not be empty",
                });
            }
        }
        if let Some(id) = &self.local_id {
            if id.is_empty() {
                return Err(GraphModelError::InvalidEntityFields {
                    model_class,
                    reason: "local_id must not be empty",
                });
            }
        }
        Ok(())
    }
}

impl GraphNodePayload {
    fn validate(&self, model_class: GraphModelClass) -> Result<(), GraphModelError> {
        self.common.validate(model_class)
    }
}

impl GraphEdgePayload {
    fn validate(&self, model_class: GraphModelClass) -> Result<(), GraphModelError> {
        self.common.validate(model_class)?;
        if self.source_ids.is_none() || self.target_ids.is_none() || self.relation.is_none() {
            return Err(GraphModelError::InvalidEntityFields {
                model_class,
                reason: "edge requires source_ids, target_ids, and relation",
            });
        }
        for id in self
            .source_ids
            .iter()
            .chain(self.target_ids.iter())
            .flatten()
        {
            if id.0.is_empty() {
                return Err(GraphModelError::InvalidEntityFields {
                    model_class,
                    reason: "node endpoint IDs must not be empty",
                });
            }
        }
        for id in self
            .source_edge_ids
            .iter()
            .chain(self.target_edge_ids.iter())
            .flatten()
        {
            if id.0.is_empty() {
                return Err(GraphModelError::InvalidEntityFields {
                    model_class,
                    reason: "edge endpoint IDs must not be empty",
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_slice::{
        GRAPH_SLICE_MANIFEST_VERSION, GraphSliceDeclarationV1, GraphSliceFieldV1,
        GraphSliceManifestV1, GraphSliceMode,
    };
    use std::collections::BTreeSet;

    fn fixture() -> Value {
        serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/python-graph-model-payloads-v1.json"
        )))
        .expect("valid Python graph payload fixture")
    }

    fn assert_persistent_v2_validation_matches_v1(model_class: GraphModelClass, payload: &Value) {
        let legacy_result =
            GraphModelEnvelope::new(model_class, payload.clone()).validate_family_v1();
        let wire = serde_json::json!({
            "schema_version": GRAPH_MODEL_ENVELOPE_VERSION,
            "model_class": model_class.wire_name(),
            "payload": payload,
        })
        .to_string();
        let v2_result = RawGraphModelEnvelopeV1::from_json(&wire)
            .and_then(|envelope| envelope.decode_persistent_v2());

        assert_eq!(
            v2_result.is_ok(),
            legacy_result.is_ok(),
            "persistent V2 validation must preserve the V1 family contract for {}: V1={legacy_result:?}, V2={v2_result:?}",
            model_class.wire_name()
        );
    }

    fn capability_fixture() -> Value {
        serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/graph-model-capabilities-v1.json"
        )))
        .expect("valid shared graph capability fixture")
    }

    #[test]
    fn every_python_node_and_edge_fixture_has_a_typed_rust_view() {
        for case in fixture()["cases"].as_array().expect("fixture cases") {
            let model_class = match case["kind"].as_str().expect("kind") {
                "node" => GraphModelClass::Node,
                "edge" | "edge_with_edge_endpoints" => GraphModelClass::Edge,
                other => panic!("unexpected fixture graph model kind: {other}"),
            };
            let envelope = GraphModelEnvelope::new(model_class, case["payload"].clone());
            assert_persistent_v2_validation_matches_v1(model_class, &case["payload"]);
            let typed = envelope.inspect().expect("valid typed graph view");
            match typed {
                TypedGraphEntity::Node(node) => {
                    assert_eq!(node.common.id.as_deref(), Some("parity-node"));
                }
                TypedGraphEntity::Edge(edge) => {
                    assert_eq!(edge.source_ids.as_ref().unwrap()[0].0, "parity-node");
                    if case["kind"] == "edge_with_edge_endpoints" {
                        assert_eq!(edge.source_edge_ids.as_ref().unwrap()[0].0, "parity-edge");
                        assert_eq!(
                            edge.target_edge_ids.as_ref().unwrap()[0].0,
                            "parity-edge-target"
                        );
                    }
                }
            }
            assert_eq!(envelope.payload, case["payload"]);
        }
    }

    #[test]
    fn workflow_runtime_edge_matches_python_metadata_defaults_and_requirements() {
        let contract: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/workflow-runtime-edge-metadata-v1.json"
        )))
        .expect("valid shared workflow runtime edge metadata contract");
        assert_eq!(contract["schema_version"], 1);

        let baseline = fixture()["cases"][1]["payload"].clone();
        for case in contract["cases"].as_array().expect("contract cases") {
            let mut payload = baseline.clone();
            payload["metadata"] = case["metadata"].clone();
            let result = GraphModelEnvelope::new(GraphModelClass::WorkflowRuntimeEdge, payload)
                .validate_family_v1();
            assert_eq!(
                result.is_ok(),
                case["valid"].as_bool().expect("validity flag"),
                "workflow runtime edge metadata case {}",
                case["id"]
            );
        }
    }

    #[test]
    fn workflow_runtime_node_metadata_matches_python_model_contract() {
        let contract: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/workflow-runtime-node-metadata-v1.json"
        )))
        .expect("valid shared workflow runtime node metadata contract");
        assert_eq!(contract["schema_version"], 1);

        let baseline = fixture()["cases"][0]["payload"].clone();
        for case in contract["cases"].as_array().expect("contract cases") {
            let model_class: GraphModelClass = serde_json::from_value(case["model_class"].clone())
                .expect("known runtime node model class");
            let mut payload = baseline.clone();
            payload["metadata"] = case["metadata"].clone();
            let result = GraphModelEnvelope::new(model_class, payload).validate_family_v1();
            assert_eq!(
                result.is_ok(),
                case["valid"].as_bool().expect("validity flag"),
                "workflow runtime node metadata case {}",
                case["id"]
            );
        }
    }

    #[test]
    fn conversation_metadata_matches_python_model_contract() {
        let contract: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/conversation-graph-metadata-v1.json"
        )))
        .expect("valid shared conversation graph metadata contract");
        assert_eq!(contract["schema_version"], 1);

        let graph_fixture = fixture();
        for case in contract["cases"].as_array().expect("contract cases") {
            let model_class: GraphModelClass = serde_json::from_value(case["model_class"].clone())
                .expect("known conversation model class");
            let baseline_index = usize::from(model_class == GraphModelClass::ConversationEdge);
            let mut payload = graph_fixture["cases"][baseline_index]["payload"].clone();
            payload["metadata"] = case["metadata"].clone();
            let result = GraphModelEnvelope::new(model_class, payload).validate_family_v1();
            assert_eq!(
                result.is_ok(),
                case["valid"].as_bool().expect("validity flag"),
                "conversation metadata case {}",
                case["id"]
            );
        }
    }

    #[test]
    fn workflow_design_metadata_matches_python_model_contract() {
        let contract: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/workflow-design-metadata-v1.json"
        )))
        .expect("valid shared workflow design metadata contract");
        assert_eq!(contract["schema_version"], 1);

        let graph_fixture = fixture();
        for case in contract["cases"].as_array().expect("contract cases") {
            let model_class: GraphModelClass = serde_json::from_value(case["model_class"].clone())
                .expect("known workflow design model class");
            let baseline_index = usize::from(model_class == GraphModelClass::WorkflowEdge);
            let mut payload = graph_fixture["cases"][baseline_index]["payload"].clone();
            payload["metadata"] = if case["valid"].as_bool() == Some(true) {
                case["serialized_metadata"].clone()
            } else {
                case["input_metadata"].clone()
            };
            let result = GraphModelEnvelope::new(model_class, payload).validate_family_v1();
            assert_eq!(
                result.is_ok(),
                case["valid"].as_bool().expect("validity flag"),
                "workflow design metadata case {}: {result:?}",
                case["id"]
            );
        }
    }

    #[test]
    fn persistent_graph_invalid_cases_match_shared_python_rejection_fixture() {
        let baseline: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/python-graph-model-payloads-v1.json"
        )))
        .expect("valid Python graph payload fixture");
        let invalid: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/graph-model-invalid-v1.json"
        )))
        .expect("valid graph invalid-case fixture");
        assert_eq!(invalid["schema_version"], 1);

        for case in invalid["cases"].as_array().expect("invalid cases") {
            let case_id = case.as_str().expect("case id");
            let (kind, mutation) = case_id.split_once('.').expect("kind.mutation");
            let baseline_kind = match kind {
                "pure_chroma_node" => "node",
                "pure_chroma_edge" => "edge",
                _ => kind,
            };
            let baseline_case = baseline["cases"]
                .as_array()
                .expect("baseline cases")
                .iter()
                .find(|candidate| candidate["kind"] == baseline_kind)
                .expect("matching baseline model");
            let mut payload = baseline_case["payload"].clone();
            if kind.starts_with("pure_chroma_") {
                payload
                    .as_object_mut()
                    .expect("PureChroma payload object")
                    .remove("mentions");
            }
            let model_class = match kind {
                "node" => GraphModelClass::Node,
                "edge" => GraphModelClass::Edge,
                "pure_chroma_node" => GraphModelClass::PureChromaNode,
                "pure_chroma_edge" => GraphModelClass::PureChromaEdge,
                _ => panic!("unexpected fixture model class: {kind}"),
            };
            match mutation {
                "invalid_lifecycle_status" => {
                    payload["metadata"]["lifecycle_status"] = Value::String("purged".into());
                }
                "empty_id" => {
                    payload["id"] = Value::String(String::new());
                }
                "invalid_redirect_id" => {
                    payload["metadata"]["redirect_to_id"] = Value::Number(7.into());
                }
                "empty_node_endpoint_id" if kind == "edge" => {
                    payload["source_ids"] = serde_json::json!([""]);
                }
                "empty_edge_endpoint_id" if kind == "edge" => {
                    payload["source_edge_ids"] = serde_json::json!([""]);
                }
                "missing_relation" if kind == "edge" => {
                    payload
                        .as_object_mut()
                        .expect("edge payload object")
                        .remove("relation");
                }
                "properties_not_mapping" => {
                    payload["properties"] = serde_json::json!(["not", "a", "mapping"]);
                }
                "properties_nested_mapping" if kind == "node" => {
                    payload["properties"] = serde_json::json!({
                        "invalid": {"nested": {"value": 1}}
                    });
                }
                "properties_nested_list" if kind == "edge" => {
                    payload["properties"] = serde_json::json!({"invalid": [[1]]});
                }
                "properties_nested_mapping" if kind == "pure_chroma_node" => {
                    payload["properties"] = serde_json::json!({
                        "invalid": {"nested": {"value": 1}}
                    });
                }
                "properties_nested_list" if kind == "pure_chroma_edge" => {
                    payload["properties"] = serde_json::json!({"invalid": [[1]]});
                }
                _ => panic!("unexpected graph-model invalid fixture: {case_id}"),
            }
            assert!(
                GraphModelEnvelope::new(model_class, payload.clone())
                    .validate_family_v1()
                    .is_err(),
                "Rust accepted shared invalid graph-model case {case_id}"
            );
            if matches!(model_class, GraphModelClass::Node | GraphModelClass::Edge) {
                assert_persistent_v2_validation_matches_v1(model_class, &payload);
            }
        }
    }

    #[test]
    fn persistent_graph_legacy_defaults_match_shared_compatibility_fixture() {
        let baseline: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/python-graph-model-payloads-v1.json"
        )))
        .expect("valid Python graph payload fixture");
        let compatibility: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/graph-model-legacy-compat-v1.json"
        )))
        .expect("valid graph compatibility fixture");
        assert_eq!(compatibility["schema_version"], 1);

        for case in compatibility["cases"]
            .as_array()
            .expect("compatibility cases")
        {
            let case_id = case.as_str().expect("case id");
            let (kind, compatibility_case) = case_id.split_once('.').expect("kind.case");
            let baseline_case = baseline["cases"]
                .as_array()
                .expect("baseline cases")
                .iter()
                .find(|candidate| candidate["kind"] == kind)
                .expect("matching baseline model");
            let mut payload = baseline_case["payload"].clone();
            let model_class = match kind {
                "node" => GraphModelClass::Node,
                "edge" => GraphModelClass::Edge,
                _ => panic!("unexpected fixture model class: {kind}"),
            };
            let expected_status = match compatibility_case {
                "omitted_lifecycle_defaults" => {
                    let metadata = payload
                        .get_mut("metadata")
                        .and_then(Value::as_object_mut)
                        .expect("fixture metadata");
                    metadata.remove("lifecycle_status");
                    metadata.remove("redirect_to_id");
                    "active"
                }
                "deleted_alias_normalizes" => {
                    payload["metadata"]["lifecycle_status"] = Value::String("deleted".into());
                    "tombstoned"
                }
                "level_metadata_fallback" => {
                    payload
                        .as_object_mut()
                        .expect("graph payload object")
                        .remove("level_from_root");
                    payload["metadata"]["level_from_root"] = Value::Number(2.into());
                    "active"
                }
                "level_top_level_precedence" => {
                    payload["level_from_root"] = Value::Number(3.into());
                    payload["metadata"]["level_from_root"] = Value::Number(2.into());
                    "active"
                }
                _ => panic!("unexpected graph-model compatibility case: {case_id}"),
            };
            let decoded = GraphModelEnvelope::new(model_class, payload)
                .decode_persistent_v1()
                .expect("legacy graph payload remains accepted");
            let core = match decoded {
                PersistentGraphEntityV1::Node(node) => node.core,
                PersistentGraphEntityV1::Edge(edge) => edge.core,
            };
            assert_eq!(core.metadata["lifecycle_status"], expected_status);
            assert_eq!(core.metadata["redirect_to_id"], Value::Null);
            match compatibility_case {
                "level_metadata_fallback" => {
                    assert_eq!(core.level_from_root, Some(2));
                    assert_eq!(core.metadata["level_from_root"], 2);
                }
                "level_top_level_precedence" => {
                    assert_eq!(core.level_from_root, Some(3));
                    assert_eq!(core.metadata["level_from_root"], 3);
                }
                _ => {}
            }
        }
    }

    #[test]
    fn model_class_tag_is_out_of_band_and_preserves_legacy_payload_shape() {
        let payload = fixture()["cases"][0]["payload"].clone();
        let envelope = GraphModelEnvelope::new(GraphModelClass::AclNode, payload.clone());
        assert_eq!(envelope.payload, payload);
        assert_eq!(envelope.model_class.entity_kind(), GraphEntityKind::Node);
        assert!(
            !serde_json::to_value(envelope)
                .unwrap()
                .get("payload")
                .unwrap()
                .as_object()
                .unwrap()
                .contains_key("model_class")
        );
    }

    #[test]
    fn node_and_edge_endpoint_ids_are_distinct_rust_types() {
        let payload = fixture()["cases"][2]["payload"].clone();
        let envelope = GraphModelEnvelope::new(GraphModelClass::Edge, payload);
        let TypedGraphEntity::Edge(edge) = envelope.inspect().unwrap() else {
            panic!("edge model class must decode to an edge");
        };
        assert_eq!(
            edge.source_ids.unwrap()[0],
            NodeId("parity-node".to_owned())
        );
        assert_eq!(
            edge.source_edge_ids.unwrap()[0],
            EdgeId("parity-edge".to_owned())
        );
    }

    #[test]
    fn hyperedges_reuse_edge_identity_and_preserve_all_incidences() {
        let mut payload = fixture()["cases"][1]["payload"].clone();
        payload["source_ids"] = serde_json::json!(["node-a", "node-b"]);
        payload["target_ids"] = serde_json::json!(["node-c", "node-d"]);
        payload["source_edge_ids"] = serde_json::json!(["edge-parent"]);
        let envelope = GraphModelEnvelope::new(GraphModelClass::Edge, payload);
        let TypedGraphEntity::Edge(edge) = envelope.inspect().unwrap() else {
            panic!("hyperedge must retain the ordinary edge identity");
        };
        assert_eq!(edge.source_ids.as_ref().unwrap().len(), 2);
        assert_eq!(edge.target_ids.as_ref().unwrap().len(), 2);
        assert_eq!(edge.source_edge_ids.as_ref().unwrap().len(), 1);
        assert!(edge.is_hyperedge());
        assert_eq!(edge.graph_entity_id(), Some("parity-edge"));
    }

    #[test]
    fn nodes_and_edges_share_the_graph_entity_interface() {
        fn inspect_entity(entity: &impl GraphEntity) -> (&'static str, Option<&str>) {
            let kind = match entity.graph_entity_kind() {
                GraphEntityKind::Node => "node",
                GraphEntityKind::Edge => "edge",
            };
            (kind, entity.graph_entity_id())
        }

        let fixture = fixture();
        let node = GraphModelEnvelope::new(
            GraphModelClass::Node,
            fixture["cases"][0]["payload"].clone(),
        )
        .inspect()
        .unwrap();
        let edge = GraphModelEnvelope::new(
            GraphModelClass::Edge,
            fixture["cases"][1]["payload"].clone(),
        )
        .inspect()
        .unwrap();

        assert_eq!(inspect_entity(&node), ("node", Some("parity-node")));
        assert_eq!(inspect_entity(&edge), ("edge", Some("parity-edge")));
    }

    #[test]
    fn explicit_model_class_rejects_node_edge_shape_confusion() {
        let edge_payload = fixture()["cases"][1]["payload"].clone();
        let wrongly_tagged = GraphModelEnvelope::new(GraphModelClass::Node, edge_payload);
        assert!(matches!(
            wrongly_tagged.inspect(),
            Err(GraphModelError::InvalidEntityFields {
                reason: "node model payload contains edge-only fields",
                ..
            })
        ));
        assert!(matches!(
            wrongly_tagged.decode_persistent_v1(),
            Err(GraphModelError::InvalidEntityFields {
                reason: "node model payload contains edge-only fields",
                ..
            })
        ));

        let mut incomplete = fixture()["cases"][1]["payload"].clone();
        incomplete.as_object_mut().unwrap().remove("relation");
        let wrongly_shaped = GraphModelEnvelope::new(GraphModelClass::Edge, incomplete);
        assert!(matches!(
            wrongly_shaped.inspect(),
            Err(GraphModelError::InvalidEntityFields {
                reason: "edge requires source_ids, target_ids, and relation",
                ..
            })
        ));
    }

    #[test]
    fn strict_persistent_models_preserve_required_nullable_edge_fields() {
        let fixture = fixture();
        let payload = fixture["cases"][1]["payload"].clone();
        let decoded = GraphModelEnvelope::new(GraphModelClass::Edge, payload.clone())
            .decode_persistent_v1()
            .expect("strict edge payload");
        let PersistentGraphEntityV1::Edge(edge) = decoded else {
            panic!("expected strict edge");
        };
        assert!(matches!(edge.source_edge_ids, RequiredNullable::Value(_)));
        assert!(matches!(edge.target_edge_ids, RequiredNullable::Value(_)));
        assert_eq!(edge.lifecycle_status(), LifecycleStatus::Active);
        assert_eq!(edge.level_from_root(), None);

        let mut explicit_null = fixture["cases"][1]["payload"].clone();
        explicit_null["source_edge_ids"] = Value::Null;
        explicit_null["target_edge_ids"] = Value::Null;
        let decoded = GraphModelEnvelope::new(GraphModelClass::Edge, explicit_null)
            .decode_persistent_v1()
            .expect("explicit null remains valid");
        let PersistentGraphEntityV1::Edge(edge) = decoded else {
            panic!("expected strict edge");
        };
        assert!(edge.source_edge_ids.is_null());
        assert!(edge.target_edge_ids.is_null());

        let mut missing = fixture["cases"][1]["payload"].clone();
        missing
            .as_object_mut()
            .expect("object payload")
            .remove("source_edge_ids");
        assert!(matches!(
            GraphModelEnvelope::new(GraphModelClass::Edge, missing).decode_persistent_v1(),
            Err(GraphModelError::InvalidPayload(_))
        ));
    }

    #[test]
    fn strict_persistent_models_reject_invalid_grounding_and_lifecycle() {
        let fixture = fixture();
        let mut payload = fixture["cases"][0]["payload"].clone();
        payload["mentions"] = Value::Array(Vec::new());
        assert!(matches!(
            GraphModelEnvelope::new(GraphModelClass::Node, payload).decode_persistent_v1(),
            Err(GraphModelError::InvalidEntityFields {
                reason: "persistent entity requires at least one grounding",
                ..
            })
        ));

        let mut invalid_lifecycle = fixture["cases"][0]["payload"].clone();
        invalid_lifecycle["metadata"]["lifecycle_status"] = Value::String("purged".into());
        assert!(matches!(
            GraphModelEnvelope::new(GraphModelClass::Node, invalid_lifecycle)
                .decode_persistent_v1(),
            Err(GraphModelError::InvalidEntityFields {
                reason: "lifecycle_status must be active or tombstoned",
                ..
            })
        ));
    }

    #[test]
    fn persistent_node_and_edge_omitted_lifecycle_metadata_keep_python_defaults() {
        let fixture = fixture();
        for (model_class, case_name) in [
            (GraphModelClass::Node, "node"),
            (GraphModelClass::Edge, "edge"),
        ] {
            let mut payload = fixture["cases"]
                .as_array()
                .expect("fixture cases")
                .iter()
                .find(|case| case["kind"] == case_name)
                .expect("fixture model case")["payload"]
                .clone();
            let metadata = payload
                .get_mut("metadata")
                .and_then(Value::as_object_mut)
                .expect("fixture metadata");
            metadata.remove("lifecycle_status");
            metadata.remove("redirect_to_id");

            let decoded = GraphModelEnvelope::new(model_class, payload)
                .decode_persistent_v1()
                .expect("omitted lifecycle metadata remains backward compatible");
            match decoded {
                PersistentGraphEntityV1::Node(node) => {
                    assert_eq!(node.lifecycle_status(), LifecycleStatus::Active);
                    assert_eq!(node.core.metadata["redirect_to_id"], Value::Null);
                }
                PersistentGraphEntityV1::Edge(edge) => {
                    assert_eq!(edge.lifecycle_status(), LifecycleStatus::Active);
                    assert_eq!(edge.core.metadata["redirect_to_id"], Value::Null);
                }
            }
        }
    }

    #[test]
    fn persistent_node_and_edge_normalize_python_mixin_fields() {
        let fixture = fixture();
        for (model_class, case_name) in [
            (GraphModelClass::Node, "node"),
            (GraphModelClass::Edge, "edge"),
        ] {
            let mut payload = fixture["cases"]
                .as_array()
                .expect("fixture cases")
                .iter()
                .find(|case| case["kind"] == case_name)
                .expect("fixture model case")["payload"]
                .clone();
            payload["metadata"]["lifecycle_status"] = Value::String("deleted".to_owned());
            payload["metadata"]["level_from_root"] = Value::Number(2.into());
            payload
                .as_object_mut()
                .expect("object payload")
                .remove("level_from_root");
            let envelope = GraphModelEnvelope::new(model_class, payload.clone());

            let decoded = envelope
                .decode_persistent_v1()
                .expect("legacy model fields normalize like Python mixins");
            let core = match decoded {
                PersistentGraphEntityV1::Node(node) => node.core,
                PersistentGraphEntityV1::Edge(edge) => edge.core,
            };
            assert_eq!(core.metadata["lifecycle_status"], "tombstoned");
            assert_eq!(core.metadata["redirect_to_id"], Value::Null);
            assert_eq!(core.level_from_root, Some(2));
            assert_eq!(core.metadata["level_from_root"], 2);
            assert_eq!(envelope.payload["metadata"]["lifecycle_status"], "deleted");

            let mut conflicting = fixture["cases"]
                .as_array()
                .expect("fixture cases")
                .iter()
                .find(|case| case["kind"] == case_name)
                .expect("fixture model case")["payload"]
                .clone();
            conflicting["level_from_root"] = Value::Number(3.into());
            conflicting["metadata"]["level_from_root"] = Value::Number(2.into());
            let decoded = GraphModelEnvelope::new(model_class, conflicting)
                .decode_persistent_v1()
                .expect("explicit level field overrides metadata like Python mixin");
            let core = match decoded {
                PersistentGraphEntityV1::Node(node) => node.core,
                PersistentGraphEntityV1::Edge(edge) => edge.core,
            };
            assert_eq!(core.level_from_root, Some(3));
            assert_eq!(core.metadata["level_from_root"], 3);

            for (raw_level, expected) in [
                (Value::String("2".to_owned()), 2),
                (Value::Bool(true), 1),
                (serde_json::json!(2.0), 2),
            ] {
                let mut coerced = fixture["cases"]
                    .as_array()
                    .expect("fixture cases")
                    .iter()
                    .find(|case| case["kind"] == case_name)
                    .expect("fixture model case")["payload"]
                    .clone();
                coerced["metadata"]["level_from_root"] = raw_level;
                coerced
                    .as_object_mut()
                    .expect("object payload")
                    .remove("level_from_root");
                let decoded = GraphModelEnvelope::new(model_class, coerced)
                    .decode_persistent_v1()
                    .expect("coerce metadata level as Python's integer field does");
                let core = match decoded {
                    PersistentGraphEntityV1::Node(node) => node.core,
                    PersistentGraphEntityV1::Edge(edge) => edge.core,
                };
                assert_eq!(core.level_from_root, Some(expected));
                assert_eq!(core.metadata["level_from_root"], expected);
            }

            let mut fractional = fixture["cases"]
                .as_array()
                .expect("fixture cases")
                .iter()
                .find(|case| case["kind"] == case_name)
                .expect("fixture model case")["payload"]
                .clone();
            fractional["metadata"]["level_from_root"] = serde_json::json!(2.5);
            fractional
                .as_object_mut()
                .expect("object payload")
                .remove("level_from_root");
            assert!(
                GraphModelEnvelope::new(model_class, fractional)
                    .decode_persistent_v1()
                    .is_err()
            );

            let mut negative = fixture["cases"]
                .as_array()
                .expect("fixture cases")
                .iter()
                .find(|case| case["kind"] == case_name)
                .expect("fixture model case")["payload"]
                .clone();
            negative["level_from_root"] = Value::Number((-1).into());
            let decoded = GraphModelEnvelope::new(model_class, negative)
                .decode_persistent_v1()
                .expect("preserve Python's existing integer level acceptance");
            let core = match decoded {
                PersistentGraphEntityV1::Node(node) => node.core,
                PersistentGraphEntityV1::Edge(edge) => edge.core,
            };
            assert_eq!(core.level_from_root, Some(-1));
            assert_eq!(core.metadata["level_from_root"], -1);

            let mut explicit_null = fixture["cases"]
                .as_array()
                .expect("fixture cases")
                .iter()
                .find(|case| case["kind"] == case_name)
                .expect("fixture model case")["payload"]
                .clone();
            explicit_null["level_from_root"] = Value::Null;
            explicit_null["metadata"]["level_from_root"] = Value::Number(2.into());
            let decoded = GraphModelEnvelope::new(model_class, explicit_null)
                .decode_persistent_v1()
                .expect("explicit null remains distinct from omitted field");
            let core = match decoded {
                PersistentGraphEntityV1::Node(node) => node.core,
                PersistentGraphEntityV1::Edge(edge) => edge.core,
            };
            assert_eq!(core.level_from_root, None);
            assert_eq!(core.metadata["level_from_root"], 2);
        }
    }

    #[test]
    fn raw_family_validator_accepts_oversized_python_hierarchy_levels() {
        let fixture = fixture();
        let baseline = fixture["cases"][0]["payload"].clone();
        let huge = "18446744073709551616";

        let top_level_payload = serde_json::to_string(&baseline)
            .expect("serialize baseline payload")
            .replace(
                "\"level_from_root\":0",
                &format!("\"level_from_root\":{huge}"),
            );
        let top_level_wire = format!(
            "{{\"schema_version\":1,\"model_class\":\"node\",\"payload\":{top_level_payload}}}"
        );
        assert!(top_level_wire.contains(huge));
        validate_graph_model_family_from_str(&top_level_wire)
            .expect("large top-level Python integer validates");
        assert!(
            validate_graph_model_family_from_str(&top_level_wire.replace(huge, "2.5")).is_err()
        );
        let negative_huge = format!("-{huge}");
        let negative_top_level = top_level_wire.replace(huge, &negative_huge);
        validate_graph_model_family_from_str(&negative_top_level)
            .expect("large negative top-level Python integer validates");
        let negative_v2 = RawGraphModelEnvelopeV1::from_json(&negative_top_level)
            .expect("negative raw envelope parses")
            .decode_persistent_v2()
            .expect("negative V2 decode retains the exact integer");
        assert_eq!(
            negative_v2
                .level_from_root
                .expect("negative level present")
                .as_decimal_str(),
            negative_huge
        );

        let mut metadata_only = baseline;
        metadata_only
            .as_object_mut()
            .expect("node payload object")
            .remove("level_from_root");
        let metadata_payload = serde_json::to_string(&metadata_only)
            .expect("serialize metadata-only payload")
            .replace(
                "\"level_from_root\":0",
                &format!("\"level_from_root\":{huge}"),
            );
        let metadata_wire = format!(
            "{{\"schema_version\":1,\"model_class\":\"node\",\"payload\":{metadata_payload}}}"
        );
        validate_graph_model_family_from_str(&metadata_wire)
            .expect("large metadata fallback integer validates");
        let negative_metadata = metadata_wire.replace(huge, &negative_huge);
        validate_graph_model_family_from_str(&negative_metadata)
            .expect("large negative metadata fallback integer validates");
    }

    #[test]
    fn raw_graph_envelope_round_trip_preserves_python_big_integer_tokens() {
        let fixture = fixture();
        let baseline = serde_json::to_string(&fixture["cases"][0]["payload"])
            .expect("serialize graph payload");
        let huge = "1606938044258990275541962092341162602522202993782792835301376";
        let payload = baseline.replace(
            "\"level_from_root\":0",
            &format!("\"level_from_root\":{huge}"),
        );
        let wire =
            format!("{{\"schema_version\":1,\"model_class\":\"node\",\"payload\":{payload}}}");

        let envelope = RawGraphModelEnvelopeV1::from_json(&wire)
            .expect("raw envelope accepts valid JSON without materializing payload numbers");
        envelope
            .validate_family_v1()
            .expect("raw envelope validates using the shared family contract");
        let round_trip = envelope.to_json().expect("serialize raw envelope");

        assert!(round_trip.contains(huge));
        assert_eq!(envelope.payload.get(), payload);
    }

    #[test]
    fn raw_graph_envelope_reads_python_integer_without_narrowing_or_losing_precedence() {
        let huge = "1606938044258990275541962092341162602522202993782792835301376";
        let wire = format!(
            "{{\"schema_version\":1,\"model_class\":\"node\",\"payload\":{{\
             \"level_from_root\":{huge},\"metadata\":{{\"level_from_root\":7}}}}}}"
        );
        let envelope = RawGraphModelEnvelopeV1::from_json(&wire).expect("valid raw envelope");
        assert_eq!(
            envelope
                .level_from_root()
                .expect("read exact top-level integer")
                .expect("level is present")
                .as_decimal_str(),
            huge
        );

        let metadata_wire = wire.replace(&format!("\"level_from_root\":{huge},"), "");
        let metadata_envelope =
            RawGraphModelEnvelopeV1::from_json(&metadata_wire).expect("valid raw envelope");
        assert_eq!(
            metadata_envelope
                .level_from_root()
                .expect("read metadata fallback")
                .expect("metadata level is present")
                .as_decimal_str(),
            "7"
        );

        let null_wire = metadata_wire.replace("\"level_from_root\":7", "\"level_from_root\":null");
        let null_envelope =
            RawGraphModelEnvelopeV1::from_json(&null_wire).expect("valid raw envelope");
        assert_eq!(
            null_envelope
                .level_from_root()
                .expect("explicit null is valid"),
            None
        );
    }

    #[test]
    fn persistent_v2_keeps_unbounded_level_and_lossless_payload_without_changing_v1() {
        let fixture = fixture();
        let huge = "1606938044258990275541962092341162602522202993782792835301376";
        let baseline = serde_json::to_string(&fixture["cases"][0]["payload"])
            .expect("serialize graph payload");
        let payload = baseline.replace(
            "\"level_from_root\":0",
            &format!("\"level_from_root\":{huge}"),
        );
        let wire =
            format!("{{\"schema_version\":1,\"model_class\":\"node\",\"payload\":{payload}}}");
        let raw = RawGraphModelEnvelopeV1::from_json(&wire).expect("valid raw envelope");
        let decoded = raw
            .decode_persistent_v2()
            .expect("typed V2 decoder retains the Python integer");

        assert_eq!(
            decoded
                .level_from_root
                .as_ref()
                .expect("level remains available")
                .as_decimal_str(),
            huge
        );
        assert!(decoded.raw_payload.get().contains(huge));
        let PersistentGraphEntityV1::Node(node) = decoded.entity else {
            panic!("node envelope must decode as a node");
        };
        assert_eq!(node.core.level_from_root, None);

        let ordinary_wire =
            format!("{{\"schema_version\":1,\"model_class\":\"node\",\"payload\":{baseline}}}");
        let ordinary = RawGraphModelEnvelopeV1::from_json(&ordinary_wire)
            .expect("ordinary level envelope")
            .decode_persistent_v2()
            .expect("ordinary level remains compatible");
        assert_eq!(
            ordinary
                .level_from_root
                .expect("ordinary level is present")
                .as_decimal_str(),
            "0"
        );
        let PersistentGraphEntityV1::Node(node) = ordinary.entity else {
            panic!("node envelope must decode as a node");
        };
        assert_eq!(node.core.level_from_root, Some(0));
    }

    #[test]
    fn persistent_v2_reads_unbounded_metadata_level_without_narrowing() {
        let fixture = fixture();
        let huge = "1606938044258990275541962092341162602522202993782792835301376";
        let mut payload = fixture["cases"][0]["payload"].clone();
        let object = payload.as_object_mut().expect("fixture payload object");
        object.remove("level_from_root");
        object
            .get_mut("metadata")
            .and_then(Value::as_object_mut)
            .expect("metadata object")
            .insert("level_from_root".to_owned(), Value::Number(7.into()));
        let payload = serde_json::to_string(&payload)
            .expect("serialize graph payload")
            .replace(
                "\"level_from_root\":7",
                &format!("\"level_from_root\":{huge}"),
            );
        let wire =
            format!("{{\"schema_version\":1,\"model_class\":\"node\",\"payload\":{payload}}}");
        let decoded = RawGraphModelEnvelopeV1::from_json(&wire)
            .expect("valid raw envelope")
            .decode_persistent_v2()
            .expect("metadata level remains available");

        assert_eq!(
            decoded
                .level_from_root
                .expect("metadata level is present")
                .as_decimal_str(),
            huge
        );
        assert!(decoded.raw_payload.get().contains(huge));
    }

    #[test]
    fn strict_edge_extends_the_shared_graph_entity_interface() {
        fn assert_persistent_capabilities<T>()
        where
            T: GraphEntity + HasMetadata + HasLifecycle + HasLevel,
        {
        }

        fn assert_edge_capabilities<T>()
        where
            T: EdgeEntity + HasMetadata + HasLifecycle + HasLevel,
        {
        }

        assert_persistent_capabilities::<PersistentNodeV1>();
        assert_persistent_capabilities::<PersistentEdgeV1>();
        assert_edge_capabilities::<PersistentEdgeV1>();

        fn inspect(entity: &impl GraphEntity) -> (&'static str, Option<&str>) {
            let kind = match entity.graph_entity_kind() {
                GraphEntityKind::Node => "node",
                GraphEntityKind::Edge => "edge",
            };
            (kind, entity.graph_entity_id())
        }

        let fixture = fixture();
        let node = GraphModelEnvelope::new(
            GraphModelClass::Node,
            fixture["cases"][0]["payload"].clone(),
        )
        .decode_persistent_v1()
        .unwrap();
        let edge = GraphModelEnvelope::new(
            GraphModelClass::Edge,
            fixture["cases"][1]["payload"].clone(),
        )
        .decode_persistent_v1()
        .unwrap();
        assert_eq!(inspect(&node), ("node", Some("parity-node")));
        assert_eq!(inspect(&edge), ("edge", Some("parity-edge")));

        let PersistentGraphEntityV1::Edge(edge) = edge else {
            panic!("expected strict edge");
        };
        assert_eq!(edge.source_node_ids().unwrap().len(), 1);
        assert_eq!(edge.relation(), Some("supports"));
    }

    #[test]
    fn persistent_v2_keeps_shared_entity_capabilities_and_exact_hierarchy_level() {
        fn assert_common_capabilities<T>()
        where
            T: GraphEntity + HasMetadata + HasLifecycle + HasLevel + HasExactLevel,
        {
        }

        assert_common_capabilities::<PersistentGraphEntityV2>();

        let fixture = fixture();
        let huge = "1606938044258990275541962092341162602522202993782792835301376";
        let huge_number: Value = serde_json::from_str(huge).expect("arbitrary JSON integer");

        for (case_index, model_class, expected_id) in [
            (0, GraphModelClass::Node, "parity-node"),
            (1, GraphModelClass::Edge, "parity-edge"),
        ] {
            let mut payload = fixture["cases"][case_index]["payload"].clone();
            let object = payload.as_object_mut().expect("entity payload object");
            object.insert("level_from_root".to_owned(), huge_number.clone());
            object
                .get_mut("metadata")
                .and_then(Value::as_object_mut)
                .expect("entity metadata object")
                .insert("level_from_root".to_owned(), huge_number.clone());
            let payload_json = serde_json::to_string(&payload).expect("serialize graph payload");
            let envelope_json = format!(
                "{{\"schema_version\":1,\"model_class\":{},\"payload\":{payload_json}}}",
                serde_json::to_string(&model_class).expect("serialize model class")
            );
            let decoded = RawGraphModelEnvelopeV1::from_json(&envelope_json)
                .expect("raw envelope parses")
                .decode_persistent_v2()
                .expect("V2 entity validates");

            assert_eq!(decoded.graph_entity_id(), Some(expected_id));
            assert_eq!(
                decoded
                    .metadata()
                    .get("lifecycle_status")
                    .and_then(Value::as_str),
                Some("active")
            );
            assert_eq!(
                decoded.level_from_root(),
                None,
                "legacy i64 view stays bounded"
            );
            assert_eq!(
                decoded
                    .exact_level_from_root()
                    .expect("exact level remains available")
                    .as_decimal_str(),
                huge
            );

            match model_class {
                GraphModelClass::Node => assert!(decoded.as_node().is_some()),
                GraphModelClass::Edge => {
                    let edge = decoded
                        .as_edge()
                        .expect("edge variant exposes edge capability");
                    assert_eq!(edge.relation(), Some("supports"));
                }
                _ => unreachable!("only persistent nodes and edges are tested"),
            }
        }

        let mut invalid_payload = fixture["cases"][0]["payload"].clone();
        invalid_payload["metadata"]["lifecycle_status"] = Value::String("purged".to_owned());
        let invalid_wire = format!(
            "{{\"schema_version\":1,\"model_class\":\"node\",\"payload\":{}}}",
            serde_json::to_string(&invalid_payload).expect("serialize invalid payload")
        );
        assert!(
            RawGraphModelEnvelopeV1::from_json(&invalid_wire)
                .expect("invalid family payload remains valid JSON")
                .decode_persistent_v2()
                .is_err(),
            "V2 decoding must retain fail-closed lifecycle validation"
        );
    }

    #[test]
    fn every_rust_graph_model_pair_keeps_node_capabilities_on_edges() {
        fn assert_node<T: GraphEntity>() {}
        fn assert_edge<T: GraphEntity + EdgeEntity>() {}

        assert_node::<GraphNodePayload>();
        assert_edge::<GraphEdgePayload>();
        assert_node::<PersistentNodeV1>();
        assert_edge::<PersistentEdgeV1>();
        assert_node::<LlmNodeV1>();
        assert_edge::<LlmEdgeV1>();
        assert_node::<FlattenedLlmNodeV1>();
        assert_edge::<FlattenedLlmEdgeV1>();
        assert_node::<PureChromaNodeV1>();
        assert_edge::<PureChromaEdgeV1>();

        // These assertions are intentionally compile-time contracts: an edge
        // is a graph entity plus incidence capabilities, never a narrower
        // replacement for its corresponding node family.
    }

    #[test]
    fn specialized_edges_resolve_to_their_matching_node_family() {
        let pairs = [
            (GraphModelClass::Edge, GraphModelClass::Node),
            (GraphModelClass::AclEdge, GraphModelClass::AclNode),
            (
                GraphModelClass::ConversationEdge,
                GraphModelClass::ConversationNode,
            ),
            (GraphModelClass::WorkflowEdge, GraphModelClass::WorkflowNode),
            (
                GraphModelClass::WorkflowRuntimeEdge,
                GraphModelClass::WorkflowRunNode,
            ),
            (
                GraphModelClass::PureChromaEdge,
                GraphModelClass::PureChromaNode,
            ),
            (GraphModelClass::LlmEdge, GraphModelClass::LlmNode),
            (
                GraphModelClass::LlmEdgeExtraction,
                GraphModelClass::LlmNodeExtraction,
            ),
            (
                GraphModelClass::FlattenedLlmEdge,
                GraphModelClass::FlattenedLlmNode,
            ),
            (
                GraphModelClass::AssocFlattenedLlmEdge,
                GraphModelClass::AssocFlattenedLlmNode,
            ),
        ];
        for (edge, node) in pairs {
            assert_eq!(
                edge.corresponding_class(GraphEntityKind::Node),
                Some(node),
                "edge {:?} must retain its specialized node family",
                edge
            );
            assert!(edge.edge_extends_node_capabilities());
        }
    }

    #[test]
    fn document_node_does_not_claim_persistent_capabilities() {
        let capabilities = GraphModelClass::DocNode.capabilities();
        assert!(capabilities.graph_entity);
        assert!(!capabilities.edge_entity);
        assert!(!capabilities.metadata);
        assert!(!capabilities.lifecycle);
        assert!(!capabilities.level);
    }

    #[test]
    fn capability_report_covers_every_declared_graph_class() {
        let profiles = graph_model_capability_profiles();
        assert_eq!(profiles.len(), GraphModelClass::all().len());
        assert_eq!(
            profiles
                .iter()
                .map(|profile| profile.model_class.as_str())
                .collect::<Vec<_>>(),
            GraphModelClass::all()
                .iter()
                .map(|model_class| model_class.wire_name())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            profiles
                .iter()
                .find(|profile| profile.model_class == "doc_node")
                .unwrap()
                .capabilities,
            GraphModelClass::DocNode.capabilities()
        );
    }

    #[test]
    fn graph_model_capabilities_match_shared_cross_language_fixture() {
        let fixture = capability_fixture();
        assert_eq!(fixture["schema_version"], GRAPH_MODEL_ENVELOPE_VERSION);

        let mut expected = BTreeMap::new();
        for group in fixture["profiles"].as_array().expect("profile groups") {
            let entity_kind = group["entity_kind"].as_str().expect("entity kind");
            for model_class in group["classes"].as_array().expect("class names") {
                let model_class = model_class.as_str().expect("model class name");
                let mut profile = group["capabilities"].clone();
                let object = profile.as_object_mut().expect("capability object");
                object.insert(
                    "schema_version".to_owned(),
                    Value::from(GRAPH_MODEL_ENVELOPE_VERSION),
                );
                object.insert(
                    "model_class".to_owned(),
                    Value::String(model_class.to_owned()),
                );
                object.insert(
                    "entity_kind".to_owned(),
                    Value::String(entity_kind.to_owned()),
                );
                let typed: GraphModelCapabilityProfileV1 =
                    serde_json::from_value(profile).expect("typed fixture profile");
                assert!(expected.insert(model_class.to_owned(), typed).is_none());
            }
        }

        let actual: BTreeMap<_, _> = graph_model_capability_profiles()
            .into_iter()
            .map(|profile| (profile.model_class.clone(), profile))
            .collect();
        assert_eq!(actual, expected);

        let edge_pairs = fixture["edge_node_pairs"]
            .as_array()
            .expect("edge/node pairs");
        let paired_edges: BTreeSet<_> = edge_pairs
            .iter()
            .map(|pair| pair[0].as_str().expect("edge model name"))
            .collect();
        let registered_edges: BTreeSet<_> = expected
            .values()
            .filter(|profile| profile.entity_kind == GraphEntityKind::Edge)
            .map(|profile| profile.model_class.as_str())
            .collect();
        assert_eq!(
            paired_edges.len(),
            edge_pairs.len(),
            "edge pairs must be unique"
        );
        assert_eq!(paired_edges, registered_edges);

        let paired_nodes: BTreeSet<_> = edge_pairs
            .iter()
            .map(|pair| pair[1].as_str().expect("node model name"))
            .collect();
        assert_eq!(
            paired_nodes.len(),
            edge_pairs.len(),
            "node pairs must be unique"
        );
        let node_only: BTreeSet<_> = fixture["node_only_classes"]
            .as_array()
            .expect("node-only classes")
            .iter()
            .map(|model| model.as_str().expect("node-only class name"))
            .collect();
        let registered_nodes: BTreeSet<_> = expected
            .values()
            .filter(|profile| profile.entity_kind == GraphEntityKind::Node)
            .map(|profile| profile.model_class.as_str())
            .collect();
        assert_eq!(
            registered_nodes
                .difference(&paired_nodes)
                .copied()
                .collect::<BTreeSet<_>>(),
            node_only
        );

        for pair in edge_pairs {
            let edge_name = pair[0].as_str().expect("edge model name");
            let node_name = pair[1].as_str().expect("node model name");
            let edge = GraphModelClass::all()
                .iter()
                .copied()
                .find(|model| model.wire_name() == edge_name)
                .expect("fixture edge exists in Rust registry");
            let node = GraphModelClass::all()
                .iter()
                .copied()
                .find(|model| model.wire_name() == node_name)
                .expect("fixture node exists in Rust registry");
            assert_eq!(edge.corresponding_class(GraphEntityKind::Node), Some(node));
            assert!(edge.edge_extends_node_capabilities());
        }
    }

    #[test]
    fn model_family_classification_keeps_specialized_shapes_separate() {
        assert_eq!(GraphModelClass::Node.family(), GraphModelFamily::Persistent);
        assert_eq!(
            GraphModelClass::ConversationEdge.family(),
            GraphModelFamily::Conversation
        );
        assert_eq!(
            GraphModelClass::WorkflowNode.family(),
            GraphModelFamily::Workflow
        );
        assert_eq!(
            GraphModelClass::PureChromaEdge.family(),
            GraphModelFamily::PureChroma
        );
        assert_eq!(
            GraphModelClass::LlmEdgeExtraction.family(),
            GraphModelFamily::Llm
        );
    }

    #[test]
    fn every_edge_family_explicitly_extends_its_node_capabilities() {
        let edge_classes: Vec<_> = GraphModelClass::all()
            .iter()
            .copied()
            .filter(|model_class| model_class.entity_kind() == GraphEntityKind::Edge)
            .collect();

        assert!(
            !edge_classes.is_empty(),
            "the graph contract must define edges"
        );
        for edge_class in edge_classes {
            let node_class = edge_class
                .corresponding_class(GraphEntityKind::Node)
                .unwrap_or_else(|| {
                    panic!(
                        "edge class {} has no node counterpart",
                        edge_class.wire_name()
                    )
                });
            assert_eq!(edge_class.entity_kind(), GraphEntityKind::Edge);
            assert_eq!(node_class.entity_kind(), GraphEntityKind::Node);
            assert_eq!(edge_class.family(), node_class.family());
            assert!(
                edge_class.edge_extends_node_capabilities(),
                "edge class {} must include the capabilities of node class {}",
                edge_class.wire_name(),
                node_class.wire_name()
            );
        }

        assert!(
            GraphModelClass::DocNode
                .corresponding_class(GraphEntityKind::Edge)
                .is_none()
        );
    }

    #[test]
    fn workflow_and_conversation_family_metadata_is_typed_and_validated() {
        let fixture = fixture();
        let mut workflow_node = fixture["cases"][0]["payload"].clone();
        workflow_node["metadata"] = serde_json::json!({
            "entity_type": "workflow_node",
            "workflow_id": "wf:demo",
            "wf_op": "parse",
            "wf_version": "v1",
            "wf_start": true
        });
        assert!(
            GraphModelEnvelope::new(GraphModelClass::WorkflowNode, workflow_node)
                .validate_family_v1()
                .is_ok()
        );

        let mut conversation_node = fixture["cases"][0]["payload"].clone();
        conversation_node["metadata"] = serde_json::json!({
            "entity_type": "conversation_node",
            "level_from_root": 0,
            "role": "user",
            "turn_index": 1,
            "conversation_id": "conversation:demo",
            "user_id": "user:demo"
        });
        assert!(
            GraphModelEnvelope::new(GraphModelClass::ConversationNode, conversation_node)
                .validate_family_v1()
                .is_ok()
        );

        for (entity_type, role, turn_index) in [
            ("conversation_start", "system", -1),
            ("conversation_turn", "user", 0),
            ("conversation_summary", "system", 1),
            ("assistant_turn", "assistant", 2),
        ] {
            let mut conversation_subtype = fixture["cases"][0]["payload"].clone();
            conversation_subtype["metadata"] = serde_json::json!({
                "entity_type": entity_type,
                "level_from_root": 0,
                "role": role,
                "turn_index": turn_index,
                "conversation_id": "conversation:demo",
                "user_id": "user:demo"
            });
            assert!(
                GraphModelEnvelope::new(GraphModelClass::ConversationNode, conversation_subtype)
                    .validate_family_v1()
                    .is_ok(),
                "Python ConversationNode subtype {entity_type:?} must remain valid"
            );
        }

        let mut invalid_conversation = fixture["cases"][0]["payload"].clone();
        invalid_conversation["metadata"] = serde_json::json!({
            "entity_type": "conversation_node",
            "level_from_root": 0,
            "turn_distance_from_last_summary": 1
        });
        assert!(matches!(
            GraphModelEnvelope::new(GraphModelClass::ConversationNode, invalid_conversation)
                .validate_family_v1(),
            Err(GraphModelError::InvalidEntityFields {
                model_class: GraphModelClass::ConversationNode,
                ..
            })
        ));
    }

    #[test]
    fn workflow_operation_version_matches_shared_python_contract_fixture() {
        let versions: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/workflow-operation-version-v1.json"
        )))
        .expect("valid workflow operation version fixture");

        let validate_version = |version: Value| {
            let metadata: BTreeMap<String, Value> = serde_json::from_value(serde_json::json!({
                "entity_type": "workflow_node",
                "workflow_id": "wf:version-fixture",
                "wf_op": "compute",
                "wf_version": version
            }))
            .expect("metadata object");
            validate_workflow_node_metadata(&metadata)
        };

        for version in versions["valid"].as_array().expect("valid versions") {
            assert!(validate_version(version.clone()).is_ok());
        }
        for version in versions["invalid"].as_array().expect("invalid versions") {
            assert!(
                validate_version(version.clone()).is_err(),
                "Rust accepted invalid operation version {version}"
            );
        }

        let legacy_metadata: BTreeMap<String, Value> = serde_json::from_value(serde_json::json!({
            "entity_type": "workflow_node",
            "workflow_id": "wf:version-fixture",
            "wf_op": "compute"
        }))
        .expect("legacy metadata object");
        let parsed: WorkflowNodeMetadataV1 = serde_json::from_value(serde_json::Value::Object(
            legacy_metadata.clone().into_iter().collect(),
        ))
        .expect("legacy metadata defaults");
        assert_eq!(
            parsed.wf_version,
            versions["legacy_default"]
                .as_str()
                .expect("legacy default version string")
        );
        assert!(validate_workflow_node_metadata(&legacy_metadata).is_ok());
    }

    #[test]
    fn workflow_edge_and_conversation_edge_metadata_preserve_edge_shape() {
        let fixture = fixture();
        let mut workflow_edge = fixture["cases"][1]["payload"].clone();
        workflow_edge["metadata"] = serde_json::json!({
            "entity_type": "workflow_edge",
            "workflow_id": "wf:demo",
            "wf_predicate": "ok",
            "wf_priority": 10,
            "wf_multiplicity": "many"
        });
        assert!(
            GraphModelEnvelope::new(GraphModelClass::WorkflowEdge, workflow_edge)
                .validate_family_v1()
                .is_ok()
        );

        let mut conversation_edge = fixture["cases"][1]["payload"].clone();
        conversation_edge["metadata"] = serde_json::json!({
            "causal_type": "chain",
            "char_distance_from_last_summary": 12,
            "turn_distance_from_last_summary": 1
        });
        assert!(
            GraphModelEnvelope::new(GraphModelClass::ConversationEdge, conversation_edge)
                .validate_family_v1()
                .is_ok()
        );
    }

    #[test]
    fn workflow_runtime_subtypes_preserve_node_edge_shape_and_metadata_tag() {
        let fixture = fixture();
        let runtime_contract: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/workflow-runtime-node-metadata-v1.json"
        )))
        .expect("valid shared workflow runtime node metadata contract");
        let node_payload = fixture["cases"][0]["payload"].clone();
        for (model_class, wire_name) in [
            (GraphModelClass::WorkflowRunNode, "workflow_run_node"),
            (
                GraphModelClass::WorkflowStepExecNode,
                "workflow_step_exec_node",
            ),
            (
                GraphModelClass::WorkflowCheckpointNode,
                "workflow_checkpoint_node",
            ),
            (
                GraphModelClass::WorkflowCompletedNode,
                "workflow_completed_node",
            ),
            (
                GraphModelClass::WorkflowCancelledNode,
                "workflow_cancelled_node",
            ),
            (GraphModelClass::WorkflowFailedNode, "workflow_failed_node"),
        ] {
            let mut payload = node_payload.clone();
            payload["metadata"] = runtime_contract["cases"]
                .as_array()
                .unwrap()
                .iter()
                .find(|case| {
                    case["model_class"] == wire_name && case["valid"].as_bool() == Some(true)
                })
                .expect("shared fixture has valid runtime-node case")["metadata"]
                .clone();
            assert!(
                GraphModelEnvelope::new(model_class, payload)
                    .validate_family_v1()
                    .is_ok(),
                "runtime node should validate: {model_class:?}"
            );
        }

        let mut edge_payload = fixture["cases"][1]["payload"].clone();
        edge_payload["metadata"] = serde_json::json!({"relation": "next"});
        assert!(
            GraphModelEnvelope::new(GraphModelClass::WorkflowRuntimeEdge, edge_payload)
                .validate_family_v1()
                .is_ok()
        );

        let mut wrong = fixture["cases"][0]["payload"].clone();
        wrong["metadata"] = serde_json::json!({"entity_type": "workflow_failed"});
        assert!(matches!(
            GraphModelEnvelope::new(GraphModelClass::WorkflowRunNode, wrong).validate_family_v1(),
            Err(GraphModelError::InvalidEntityFields { .. })
        ));
    }

    #[test]
    fn llm_pure_chroma_and_document_families_have_distinct_requirements() {
        let fixture = fixture();

        let mut llm = fixture["cases"][0]["payload"].clone();
        llm.as_object_mut().unwrap().remove("embedding");
        assert!(
            GraphModelEnvelope::new(GraphModelClass::LlmNodeExtraction, llm)
                .validate_family_v1()
                .is_ok()
        );

        let llm_with_embedding = fixture["cases"][0]["payload"].clone();
        assert!(matches!(
            GraphModelEnvelope::new(GraphModelClass::LlmNode, llm_with_embedding)
                .validate_family_v1(),
            Err(GraphModelError::InvalidEntityFields {
                reason: "LLM graph payload must not contain an embedding",
                ..
            })
        ));

        let mut pure_chroma = fixture["cases"][0]["payload"].clone();
        pure_chroma.as_object_mut().unwrap().remove("mentions");
        assert!(
            GraphModelEnvelope::new(GraphModelClass::PureChromaNode, pure_chroma)
                .validate_family_v1()
                .is_ok()
        );

        let mut doc = serde_json::json!({
            "id": "doc-node:1",
            "label": "Page 1",
            "type": "page",
            "summary": "A document page",
            "doc_id": "document:1"
        });
        doc["mentions"] = fixture["cases"][0]["payload"]["mentions"].clone();
        assert!(
            GraphModelEnvelope::new(GraphModelClass::DocNode, doc)
                .validate_family_v1()
                .is_ok()
        );

        let invalid_doc = serde_json::json!({"type": "section", "doc_id": "document:1"});
        assert!(
            GraphModelEnvelope::new(GraphModelClass::DocNode, invalid_doc)
                .validate_family_v1()
                .is_err()
        );

        let mut acl = fixture["cases"][0]["payload"].clone();
        acl["metadata"] = serde_json::json!({
            "entity_type": "acl_record",
            "acl_truth_graph": "truth:main",
            "acl_target_entity_id": "parity-node",
            "acl_target_grain": "node",
            "acl_version": 1,
            "acl_mode": "private",
            "level_from_root": 0
        });
        assert!(
            GraphModelEnvelope::new(GraphModelClass::AclNode, acl)
                .validate_family_v1()
                .is_ok()
        );

        let mut flattened = fixture["cases"][0]["payload"].clone();
        flattened.as_object_mut().unwrap().remove("embedding");
        flattened["mentions"] = serde_json::json!([{"span_ids": ["sp:1"]}]);
        assert!(
            GraphModelEnvelope::new(GraphModelClass::FlattenedLlmNode, flattened)
                .validate_family_v1()
                .is_ok()
        );

        let mut invalid_acl = fixture["cases"][0]["payload"].clone();
        invalid_acl["metadata"] = serde_json::json!({
            "entity_type": "acl_record",
            "acl_truth_graph": "truth:main",
            "acl_target_entity_id": "parity-node",
            "acl_target_grain": "unknown-grain",
            "acl_version": 1,
            "acl_mode": "private"
        });
        assert!(
            GraphModelEnvelope::new(GraphModelClass::AclNode, invalid_acl)
                .validate_family_v1()
                .is_err()
        );
    }

    #[test]
    fn family_dtos_share_graph_entity_and_edge_entity_capabilities() {
        fn identify(entity: &impl GraphEntity) -> (GraphEntityKind, Option<&str>) {
            (entity.graph_entity_kind(), entity.graph_entity_id())
        }

        let fixture = fixture();
        let mut llm_node_payload = fixture["cases"][0]["payload"].clone();
        llm_node_payload
            .as_object_mut()
            .unwrap()
            .remove("embedding");
        let llm_node: LlmNodeV1 = serde_json::from_value(llm_node_payload).unwrap();
        assert_eq!(
            identify(&llm_node),
            (GraphEntityKind::Node, Some("parity-node"))
        );

        let mut llm_edge_payload = fixture["cases"][1]["payload"].clone();
        llm_edge_payload
            .as_object_mut()
            .unwrap()
            .remove("embedding");
        let llm_edge: LlmEdgeV1 = serde_json::from_value(llm_edge_payload).unwrap();
        assert_eq!(
            identify(&llm_edge),
            (GraphEntityKind::Edge, Some("parity-edge"))
        );
        assert_eq!(llm_edge.source_node_ids().unwrap().len(), 1);

        let mut pure_payload = fixture["cases"][0]["payload"].clone();
        pure_payload.as_object_mut().unwrap().remove("mentions");
        let pure: PureChromaNodeV1 = serde_json::from_value(pure_payload).unwrap();
        assert_eq!(
            identify(&pure),
            (GraphEntityKind::Node, Some("parity-node"))
        );

        let doc = DocumentNodeV1 {
            id: Some("doc-node:1".to_owned()),
            label: "Page 1".to_owned(),
            node_type: DocumentNodeTypeV1::Page,
            summary: "A page".to_owned(),
            doc_id: "document:1".to_owned(),
            mentions: vec![],
        };
        assert_eq!(identify(&doc), (GraphEntityKind::Node, Some("doc-node:1")));
    }

    #[test]
    fn acl_edge_validation_preserves_the_persistent_edge_shape() {
        let fixture = fixture();
        let mut acl_edge = fixture["cases"][1]["payload"].clone();
        acl_edge["metadata"] = serde_json::json!({
            "entity_type": "acl_targets_truth",
            "acl_truth_graph": "truth:main",
            "acl_target_entity_id": "parity-node",
            "acl_target_grain": "node",
            "acl_version": 1,
            "acl_mode": "shared",
            "level_from_root": 0
        });
        assert!(
            GraphModelEnvelope::new(GraphModelClass::AclEdge, acl_edge.clone())
                .validate_family_v1()
                .is_ok()
        );

        acl_edge
            .as_object_mut()
            .expect("edge payload object")
            .remove("source_ids");
        assert!(matches!(
            GraphModelEnvelope::new(GraphModelClass::AclEdge, acl_edge).validate_family_v1(),
            Err(GraphModelError::InvalidPayload(_))
        ));
    }

    #[test]
    fn declared_slice_projects_canonical_payload_and_requires_declaration() {
        let manifest = GraphSliceManifestV1 {
            schema_version: GRAPH_SLICE_MANIFEST_VERSION,
            model_name: Some("Node".to_owned()),
            retain_unlisted_modes: BTreeSet::new(),
            fields: BTreeMap::from([(
                "summary".to_owned(),
                GraphSliceFieldV1 {
                    modes: BTreeSet::from([GraphSliceMode::LlmIn]),
                    child: None,
                    child_location: Default::default(),
                },
            )]),
        };
        let envelope = GraphModelEnvelope::new(
            GraphModelClass::Node,
            serde_json::json!({"summary": "visible", "secret": "backend-only"}),
        )
        .with_slice(GraphSliceDeclarationV1 {
            schema_version: GRAPH_SLICE_MANIFEST_VERSION,
            mode: GraphSliceMode::LlmIn,
            manifest,
        });

        let projected = envelope.project_declared_slice().unwrap();
        assert_eq!(projected.payload, serde_json::json!({"summary": "visible"}));
        let projected_wire =
            project_graph_model_slice_from_str(&serde_json::to_string(&envelope).unwrap()).unwrap();
        let projected_value: Value = serde_json::from_str(&projected_wire).unwrap();
        assert_eq!(
            projected_value["payload"],
            serde_json::json!({"summary": "visible"})
        );
        assert!(
            GraphModelEnvelope::new(
                GraphModelClass::Node,
                serde_json::json!({"summary": "visible"}),
            )
            .project_declared_slice()
            .is_err()
        );

        let mismatched = GraphModelEnvelope::new(
            GraphModelClass::Edge,
            serde_json::json!({"summary": "visible"}),
        )
        .with_slice(GraphSliceDeclarationV1 {
            schema_version: GRAPH_SLICE_MANIFEST_VERSION,
            mode: GraphSliceMode::LlmIn,
            manifest: envelope.slice.as_ref().unwrap().manifest.clone(),
        });
        assert!(matches!(
            mismatched.project_declared_slice(),
            Err(GraphModelError::InvalidPayload(message))
                if message.contains("does not match envelope model Edge")
        ));
    }

    #[test]
    fn batch_slice_projection_matches_individual_projection_and_is_fail_closed() {
        let manifest = GraphSliceManifestV1 {
            schema_version: GRAPH_SLICE_MANIFEST_VERSION,
            model_name: Some("Node".to_owned()),
            retain_unlisted_modes: BTreeSet::new(),
            fields: BTreeMap::from([(
                "summary".to_owned(),
                GraphSliceFieldV1 {
                    modes: BTreeSet::from([GraphSliceMode::LlmIn]),
                    child: None,
                    child_location: Default::default(),
                },
            )]),
        };
        let envelopes: Vec<_> = ["one", "two"]
            .into_iter()
            .map(|summary| {
                GraphModelEnvelope::new(
                    GraphModelClass::Node,
                    serde_json::json!({"summary": summary, "secret": "hidden"}),
                )
                .with_slice(GraphSliceDeclarationV1 {
                    schema_version: GRAPH_SLICE_MANIFEST_VERSION,
                    mode: GraphSliceMode::LlmIn,
                    manifest: manifest.clone(),
                })
            })
            .collect();
        let request = serde_json::to_string(&envelopes).unwrap();
        let batch: Vec<GraphModelSliceV1> =
            serde_json::from_str(&project_graph_model_slices_from_str(&request).unwrap()).unwrap();
        let individual: Vec<GraphModelSliceV1> = envelopes
            .iter()
            .map(|envelope| {
                serde_json::from_str(
                    &project_graph_model_slice_from_str(&serde_json::to_string(envelope).unwrap())
                        .unwrap(),
                )
                .unwrap()
            })
            .collect();
        assert_eq!(batch, individual);

        let mut invalid = envelopes;
        invalid[1].slice.as_mut().unwrap().manifest.model_name = Some("Edge".to_owned());
        assert!(
            project_graph_model_slices_from_str(&serde_json::to_string(&invalid).unwrap()).is_err()
        );
    }
}
