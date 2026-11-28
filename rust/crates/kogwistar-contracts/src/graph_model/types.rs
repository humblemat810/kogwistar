use super::*;

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
    #[serde(default, skip_serializing)]
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
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub multimodal_spans: Vec<Value>,
}

impl LlmGroundingV1 {
    pub(super) fn validate(&self, model_class: GraphModelClass) -> Result<(), GraphModelError> {
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
    pub(super) fn validate(&self, model_class: GraphModelClass) -> Result<(), GraphModelError> {
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
    #[serde(default, skip_serializing)]
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
    pub(super) fn normalize_shared_mixin_fields(
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
        } else if !level_field_present && let Some(value) = self.metadata.get("level_from_root") {
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
        Ok(())
    }

    pub(super) fn validate(&self, model_class: GraphModelClass) -> Result<(), GraphModelError> {
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
        if let Some(status) = self.metadata.get("lifecycle_status")
            && !matches!(status.as_str(), Some("active" | "tombstoned"))
        {
            return Err(GraphModelError::InvalidEntityFields {
                model_class,
                reason: "lifecycle_status must be active or tombstoned",
            });
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

pub(super) fn valid_graph_properties(value: &Value) -> bool {
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub doc_id: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LlmNodeV1 {
    #[serde(flatten)]
    pub fields: LlmEntityFieldsV1,
    pub mentions: Vec<LlmGroundingV1>,
}

/// Extraction output uses direct excerpt spans, unlike `LlmNodeV1`'s grouped
/// grounding representation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LlmNodeExtractionV1 {
    #[serde(flatten)]
    pub fields: LlmEntityFieldsV1,
    pub mentions: Vec<LlmTextSpanV1>,
}

/// Extraction edges use the same direct excerpt spans as extraction nodes,
/// while retaining typed node and edge incidence endpoints.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LlmEdgeExtractionV1 {
    #[serde(flatten)]
    pub fields: LlmEntityFieldsV1,
    pub source_ids: Vec<NodeId>,
    pub target_ids: Vec<NodeId>,
    pub relation: String,
    pub source_edge_ids: RequiredNullable<Vec<EdgeId>>,
    pub target_edge_ids: RequiredNullable<Vec<EdgeId>>,
    pub mentions: Vec<LlmTextSpanV1>,
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
    pub(super) fn validate(&self, model_class: GraphModelClass) -> Result<(), GraphModelError> {
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

/// Associated-flattened nodes carry entity fields only; grounding relations
/// live in the separate association tables.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AssocFlattenedLlmNodeV1 {
    #[serde(flatten)]
    pub fields: LlmEntityFieldsV1,
}

/// Associated-flattened edges carry incidence fields only; grounding links
/// live in the separate association tables.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AssocFlattenedLlmEdgeV1 {
    #[serde(flatten)]
    pub fields: LlmEntityFieldsV1,
    pub source_ids: Vec<NodeId>,
    pub target_ids: Vec<NodeId>,
    pub relation: String,
    pub source_edge_ids: RequiredNullable<Vec<EdgeId>>,
    pub target_edge_ids: RequiredNullable<Vec<EdgeId>>,
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
    pub(super) fn into_validated(
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
    pub(super) fn into_validated(
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

impl GraphEntity for LlmEdgeExtractionV1 {
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

impl EdgeEntity for LlmEdgeExtractionV1 {
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

impl GraphEntity for AssocFlattenedLlmNodeV1 {
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

impl GraphEntity for AssocFlattenedLlmEdgeV1 {
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

impl EdgeEntity for AssocFlattenedLlmEdgeV1 {
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
    pub(super) fn validate(&self, model_class: GraphModelClass) -> Result<(), GraphModelError> {
        if let Some(id) = &self.id
            && id.is_empty()
        {
            return Err(GraphModelError::InvalidEntityFields {
                model_class,
                reason: "id must not be empty",
            });
        }
        if let Some(id) = &self.local_id
            && id.is_empty()
        {
            return Err(GraphModelError::InvalidEntityFields {
                model_class,
                reason: "local_id must not be empty",
            });
        }
        Ok(())
    }
}

impl GraphNodePayload {
    pub(super) fn validate(&self, model_class: GraphModelClass) -> Result<(), GraphModelError> {
        self.common.validate(model_class)
    }
}

impl GraphEdgePayload {
    pub(super) fn validate(&self, model_class: GraphModelClass) -> Result<(), GraphModelError> {
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
