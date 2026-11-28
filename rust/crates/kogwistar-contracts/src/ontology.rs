//! Rust projection of `kogwistar/ontology/models.py` and
//! `kogwistar/ontology/composition.py`. Packages are declarative data only;
//! they grant neither graph mutation nor authorization authority.

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum OntologyContractError {
    #[error("invalid ontology package: {0}")]
    Package(String),
    #[error("ontology composition failed: {0}")]
    Composition(String),
    #[error("ontology validation failed: {0}")]
    Validation(String),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct OntologyImportV1 {
    pub ontology_id: String,
    pub version: String,
    pub content_sha256: String,
}

impl OntologyImportV1 {
    fn qualified_id(&self) -> String {
        format!(
            "{}@{}#{}",
            self.ontology_id, self.version, self.content_sha256
        )
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OntologyManifestV1 {
    #[serde(default = "schema_version")]
    pub schema_version: u8,
    pub ontology_id: String,
    pub version: String,
    pub title: String,
    #[serde(default)]
    pub summary: String,
    pub content_sha256: String,
    #[serde(default)]
    pub imports: Vec<OntologyImportV1>,
    #[serde(default)]
    pub extensions: BTreeMap<String, Value>,
}

fn schema_version() -> u8 {
    1
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct OntologyRoleV1 {
    pub role_id: String,
    pub target_kind: String,
    #[serde(default)]
    pub allowed_class_ids: Vec<String>,
    #[serde(default)]
    pub min_count: usize,
    #[serde(default = "one_count")]
    pub max_count: Option<usize>,
}

fn one_count() -> Option<usize> {
    Some(1)
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "kind", deny_unknown_fields)]
pub enum OntologyDescriptorV1 {
    #[serde(rename = "ontology_property")]
    Property {
        descriptor_id: String,
        name: String,
        #[serde(default)]
        summary: String,
        #[serde(default)]
        aliases: Vec<String>,
        #[serde(default)]
        extensions: BTreeMap<String, Value>,
        value_kind: String,
        #[serde(default)]
        min_count: usize,
        #[serde(default = "one_count")]
        max_count: Option<usize>,
    },
    #[serde(rename = "ontology_class")]
    Class {
        descriptor_id: String,
        name: String,
        #[serde(default)]
        summary: String,
        #[serde(default)]
        aliases: Vec<String>,
        #[serde(default)]
        extensions: BTreeMap<String, Value>,
        #[serde(default)]
        property_ids: Vec<String>,
        #[serde(default)]
        allow_additional_properties: bool,
    },
    #[serde(rename = "ontology_relation")]
    Relation {
        descriptor_id: String,
        name: String,
        #[serde(default)]
        summary: String,
        #[serde(default)]
        aliases: Vec<String>,
        #[serde(default)]
        extensions: BTreeMap<String, Value>,
        source_class_ids: Vec<String>,
        target_class_ids: Vec<String>,
    },
    #[serde(rename = "ontology_edge_shape")]
    EdgeShape {
        descriptor_id: String,
        name: String,
        #[serde(default)]
        summary: String,
        #[serde(default)]
        aliases: Vec<String>,
        #[serde(default)]
        extensions: BTreeMap<String, Value>,
        #[serde(default)]
        relation_id: Option<String>,
        roles: Vec<OntologyRoleV1>,
    },
}

impl OntologyDescriptorV1 {
    fn id(&self) -> &str {
        match self {
            Self::Property { descriptor_id, .. }
            | Self::Class { descriptor_id, .. }
            | Self::Relation { descriptor_id, .. }
            | Self::EdgeShape { descriptor_id, .. } => descriptor_id,
        }
    }

    fn kind(&self) -> &'static str {
        match self {
            Self::Property { .. } => "ontology_property",
            Self::Class { .. } => "ontology_class",
            Self::Relation { .. } => "ontology_relation",
            Self::EdgeShape { .. } => "ontology_edge_shape",
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OntologyPackageV1 {
    pub manifest: OntologyManifestV1,
    pub descriptors: Vec<OntologyDescriptorV1>,
}

impl OntologyPackageV1 {
    pub fn from_json(value: &str) -> Result<Self, OntologyContractError> {
        Self::from_value(serde_json::from_str(value)?)
    }

    pub fn from_value(value: Value) -> Result<Self, OntologyContractError> {
        let package: Self = serde_json::from_value(value)?;
        package.validate()?;
        Ok(package)
    }

    pub fn identity(&self) -> OntologyImportV1 {
        OntologyImportV1 {
            ontology_id: self.manifest.ontology_id.clone(),
            version: self.manifest.version.clone(),
            content_sha256: self.manifest.content_sha256.clone(),
        }
    }

    pub fn content_digest(&self) -> Result<String, OntologyContractError> {
        let mut canonical = serde_json::to_value(self)?;
        let manifest = canonical["manifest"]
            .as_object_mut()
            .ok_or_else(|| OntologyContractError::Package("manifest must be an object".into()))?;
        manifest.remove("content_sha256");
        if let Some(imports) = manifest.get_mut("imports").and_then(Value::as_array_mut) {
            imports.sort_by_key(|item| {
                (
                    item["ontology_id"].as_str().unwrap_or_default().to_owned(),
                    item["version"].as_str().unwrap_or_default().to_owned(),
                    item["content_sha256"]
                        .as_str()
                        .unwrap_or_default()
                        .to_owned(),
                )
            });
        }
        canonical["descriptors"]
            .as_array_mut()
            .ok_or_else(|| OntologyContractError::Package("descriptors must be an array".into()))?
            .sort_by_key(|item| {
                item["descriptor_id"]
                    .as_str()
                    .unwrap_or_default()
                    .to_owned()
            });
        digest_json(&canonical)
    }

    fn validate(&self) -> Result<(), OntologyContractError> {
        let manifest = &self.manifest;
        if manifest.schema_version != 1
            || !valid_identifier(&manifest.ontology_id)
            || !valid_semver(&manifest.version)
            || manifest.title.is_empty()
            || manifest.title.chars().count() > 240
            || manifest.summary.chars().count() > 4000
            || !valid_sha256(&manifest.content_sha256)
        {
            return Err(OntologyContractError::Package(
                "invalid manifest fields".into(),
            ));
        }
        validate_extensions(&manifest.extensions)?;
        let mut imports = BTreeSet::new();
        for item in &manifest.imports {
            if !valid_identifier(&item.ontology_id)
                || !valid_semver(&item.version)
                || !valid_sha256(&item.content_sha256)
                || !imports.insert(item.qualified_id())
            {
                return Err(OntologyContractError::Package(
                    "invalid or duplicate import".into(),
                ));
            }
        }
        if self.descriptors.is_empty() {
            return Err(OntologyContractError::Package(
                "descriptors must not be empty".into(),
            ));
        }
        let mut ids = BTreeSet::new();
        for descriptor in &self.descriptors {
            if !ids.insert(descriptor.id().to_owned()) {
                return Err(OntologyContractError::Package(
                    "duplicate descriptor_id".into(),
                ));
            }
            validate_descriptor(descriptor)?;
        }
        let digest = self.content_digest()?;
        if digest != manifest.content_sha256 {
            return Err(OntologyContractError::Package(format!(
                "content_sha256 mismatch: expected {digest}, got {}",
                manifest.content_sha256
            )));
        }
        Ok(())
    }
}

fn validate_descriptor(descriptor: &OntologyDescriptorV1) -> Result<(), OntologyContractError> {
    let invalid = |message: &str| OntologyContractError::Package(message.to_owned());
    let (id, name, summary, extensions) = match descriptor {
        OntologyDescriptorV1::Property {
            descriptor_id,
            name,
            summary,
            extensions,
            value_kind,
            min_count,
            max_count,
            ..
        } => {
            if !matches!(
                value_kind.as_str(),
                "string" | "integer" | "number" | "boolean" | "timestamp" | "logical_ref" | "json"
            ) || !valid_cardinality(*min_count, *max_count)
            {
                return Err(invalid("invalid ontology property"));
            }
            (descriptor_id, name, summary, extensions)
        }
        OntologyDescriptorV1::Class {
            descriptor_id,
            name,
            summary,
            extensions,
            property_ids,
            ..
        } => {
            if property_ids.iter().any(|value| !valid_reference(value)) {
                return Err(invalid("invalid class property reference"));
            }
            (descriptor_id, name, summary, extensions)
        }
        OntologyDescriptorV1::Relation {
            descriptor_id,
            name,
            summary,
            extensions,
            source_class_ids,
            target_class_ids,
            ..
        } => {
            if source_class_ids.is_empty()
                || target_class_ids.is_empty()
                || source_class_ids
                    .iter()
                    .chain(target_class_ids)
                    .any(|value| !valid_reference(value))
            {
                return Err(invalid("invalid relation class references"));
            }
            (descriptor_id, name, summary, extensions)
        }
        OntologyDescriptorV1::EdgeShape {
            descriptor_id,
            name,
            summary,
            extensions,
            relation_id,
            roles,
            ..
        } => {
            if roles.is_empty()
                || relation_id
                    .as_ref()
                    .is_some_and(|value| !valid_reference(value))
            {
                return Err(invalid("invalid edge shape"));
            }
            let mut role_ids = BTreeSet::new();
            for role in roles {
                if !valid_identifier(&role.role_id)
                    || !role_ids.insert(role.role_id.as_str())
                    || !matches!(role.target_kind.as_str(), "node" | "edge")
                    || role
                        .allowed_class_ids
                        .iter()
                        .any(|value| !valid_reference(value))
                    || !valid_cardinality(role.min_count, role.max_count)
                {
                    return Err(invalid("invalid edge role"));
                }
            }
            (descriptor_id, name, summary, extensions)
        }
    };
    if !valid_identifier(id)
        || name.is_empty()
        || name.chars().count() > 240
        || summary.chars().count() > 4000
    {
        return Err(invalid("invalid descriptor fields"));
    }
    validate_extensions(extensions)
}

fn valid_cardinality(min: usize, max: Option<usize>) -> bool {
    max.is_none_or(|max| max >= 1 && max >= min)
}

fn validate_extensions(values: &BTreeMap<String, Value>) -> Result<(), OntologyContractError> {
    if values
        .keys()
        .any(|key| key.starts_with('/') || key.ends_with('/') || !key.contains('/'))
    {
        return Err(OntologyContractError::Package(
            "extension keys must be namespaced".into(),
        ));
    }
    Ok(())
}

fn valid_identifier(value: &str) -> bool {
    let mut chars = value.chars();
    chars
        .next()
        .is_some_and(|first| first.is_ascii_alphabetic())
        && chars.all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '.' | '-'))
}

fn valid_reference(value: &str) -> bool {
    let mut parts = value.split(':');
    match (parts.next(), parts.next(), parts.next()) {
        (Some(local), None, None) => valid_identifier(local),
        (Some(package), Some(local), None) => valid_identifier(package) && valid_identifier(local),
        _ => false,
    }
}

fn valid_semver(value: &str) -> bool {
    let mut base = value
        .split(['-', '+'])
        .next()
        .unwrap_or_default()
        .split('.');
    let numbers: Vec<_> = base.by_ref().collect();
    numbers.len() == 3
        && numbers
            .iter()
            .all(|part| !part.is_empty() && part.bytes().all(|b| b.is_ascii_digit()))
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'-' | b'+'))
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn digest_json(value: &Value) -> Result<String, OntologyContractError> {
    Ok(hex::encode(Sha256::digest(
        crate::canonical_json(value, false).as_bytes(),
    )))
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ComposedOntologyDescriptorV1 {
    pub qualified_id: String,
    pub descriptor: OntologyDescriptorV1,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ComposedOntologyViewV1 {
    pub schema_version: u8,
    pub package_identities: Vec<OntologyImportV1>,
    pub descriptors: Vec<ComposedOntologyDescriptorV1>,
    pub composition_sha256: String,
}

impl ComposedOntologyViewV1 {
    pub fn get(&self, qualified_id: &str) -> Option<&OntologyDescriptorV1> {
        self.descriptors
            .iter()
            .find(|item| item.qualified_id == qualified_id)
            .map(|item| &item.descriptor)
    }
}

pub fn compose_ontology_packages(
    packages: &[OntologyPackageV1],
    root_ontology_ids: &[&str],
) -> Result<ComposedOntologyViewV1, OntologyContractError> {
    let mut by_identity = BTreeMap::new();
    let mut by_id: BTreeMap<&str, Vec<&OntologyPackageV1>> = BTreeMap::new();
    for package in packages {
        package.validate()?;
        let identity = package.identity();
        let key = identity.qualified_id();
        if by_identity
            .insert(key.clone(), package)
            .is_some_and(|prior| prior != package)
        {
            return Err(OntologyContractError::Composition(
                "duplicate package identity has differing payloads".into(),
            ));
        }
        by_id
            .entry(package.manifest.ontology_id.as_str())
            .or_default()
            .push(package);
    }
    let roots: BTreeSet<String> = if root_ontology_ids.is_empty() {
        by_id.keys().map(|key| (*key).to_owned()).collect()
    } else {
        root_ontology_ids
            .iter()
            .map(|root| (*root).to_owned())
            .collect()
    };
    let mut selected = BTreeMap::<String, &OntologyPackageV1>::new();
    let mut visiting = BTreeSet::new();
    fn visit<'a>(
        package: &'a OntologyPackageV1,
        by_identity: &BTreeMap<String, &'a OntologyPackageV1>,
        selected: &mut BTreeMap<String, &'a OntologyPackageV1>,
        visiting: &mut BTreeSet<String>,
    ) -> Result<(), OntologyContractError> {
        let key = package.identity().qualified_id();
        if selected.contains_key(&key) {
            return Ok(());
        }
        if !visiting.insert(key.clone()) {
            return Err(OntologyContractError::Composition(format!(
                "import cycle at {key}"
            )));
        }
        for import in &package.manifest.imports {
            let imported = by_identity.get(&import.qualified_id()).ok_or_else(|| {
                OntologyContractError::Composition(format!(
                    "missing exact ontology import {}",
                    import.qualified_id()
                ))
            })?;
            visit(imported, by_identity, selected, visiting)?;
        }
        visiting.remove(&key);
        selected.insert(key, package);
        Ok(())
    }
    for root in roots {
        let candidates = by_id.get(root.as_str()).ok_or_else(|| {
            OntologyContractError::Composition(format!("missing root ontology {root}"))
        })?;
        if candidates.len() != 1 {
            return Err(OntologyContractError::Composition(format!(
                "ambiguous root ontology {root}"
            )));
        }
        visit(candidates[0], &by_identity, &mut selected, &mut visiting)?;
    }

    let mut descriptor_map = BTreeMap::<String, OntologyDescriptorV1>::new();
    for package in selected.values() {
        for descriptor in &package.descriptors {
            let qualified = format!("{}:{}", package.manifest.ontology_id, descriptor.id());
            if descriptor_map
                .insert(qualified.clone(), descriptor.clone())
                .is_some()
            {
                return Err(OntologyContractError::Composition(format!(
                    "duplicate qualified descriptor {qualified}"
                )));
            }
        }
    }
    for package in selected.values() {
        for descriptor in &package.descriptors {
            validate_descriptor_references(
                &package.manifest.ontology_id,
                descriptor,
                &descriptor_map,
            )?;
        }
    }
    let identities: Vec<_> = selected
        .values()
        .map(|package| package.identity())
        .collect();
    let descriptors: Vec<_> = descriptor_map
        .into_iter()
        .map(|(qualified_id, descriptor)| ComposedOntologyDescriptorV1 {
            qualified_id,
            descriptor,
        })
        .collect();
    let canonical =
        json!({"schema_version": 1, "package_identities": identities, "descriptors": descriptors});
    let composition_sha256 = digest_json(&canonical)?;
    let view = ComposedOntologyViewV1 {
        schema_version: 1,
        package_identities: identities,
        descriptors,
        composition_sha256,
    };
    Ok(view)
}

fn qualify(package_id: &str, reference: &str) -> String {
    if reference.contains(':') {
        reference.to_owned()
    } else {
        format!("{package_id}:{reference}")
    }
}

fn validate_descriptor_references(
    package_id: &str,
    descriptor: &OntologyDescriptorV1,
    descriptors: &BTreeMap<String, OntologyDescriptorV1>,
) -> Result<(), OntologyContractError> {
    let require = |reference: &str, kinds: &[&str]| -> Result<(), OntologyContractError> {
        let qualified = qualify(package_id, reference);
        match descriptors.get(&qualified) {
            None => Err(OntologyContractError::Composition(format!(
                "missing descriptor {qualified}"
            ))),
            Some(target) if !kinds.contains(&target.kind()) => {
                Err(OntologyContractError::Composition(format!(
                    "{qualified} has incompatible descriptor kind"
                )))
            }
            Some(_) => Ok(()),
        }
    };
    match descriptor {
        OntologyDescriptorV1::Class { property_ids, .. } => {
            for value in property_ids {
                require(value, &["ontology_property"])?;
            }
        }
        OntologyDescriptorV1::Relation {
            source_class_ids,
            target_class_ids,
            ..
        } => {
            for value in source_class_ids.iter().chain(target_class_ids) {
                require(value, &["ontology_class"])?;
            }
        }
        OntologyDescriptorV1::EdgeShape {
            relation_id, roles, ..
        } => {
            if let Some(value) = relation_id {
                require(value, &["ontology_relation"])?;
            }
            for role in roles {
                for value in &role.allowed_class_ids {
                    require(
                        value,
                        if role.target_kind == "node" {
                            &["ontology_class"]
                        } else {
                            &["ontology_relation", "ontology_edge_shape"]
                        },
                    )?;
                }
            }
        }
        OntologyDescriptorV1::Property { .. } => {}
    }
    Ok(())
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct OntologyEndpointV1 {
    pub target_kind: String,
    pub class_id: String,
}

pub fn validate_ontology_payload(
    view: &ComposedOntologyViewV1,
    class_id: &str,
    payload: &Value,
) -> Result<(), OntologyContractError> {
    let descriptor = view.get(class_id);
    let Some(OntologyDescriptorV1::Class {
        property_ids,
        allow_additional_properties,
        ..
    }) = descriptor
    else {
        return Err(OntologyContractError::Validation(format!(
            "unknown ontology class: {class_id}"
        )));
    };
    let package_id = class_id
        .split_once(':')
        .map(|(package, _)| package)
        .unwrap_or_default();
    let object = payload
        .as_object()
        .ok_or_else(|| OntologyContractError::Validation("payload must be an object".into()))?;
    let mut properties = BTreeMap::new();
    for id in property_ids {
        let qualified = qualify(package_id, id);
        match view.get(&qualified) {
            Some(property @ OntologyDescriptorV1::Property { .. }) => {
                properties.insert(qualified, property);
            }
            _ => {
                return Err(OntologyContractError::Validation(format!(
                    "class property unavailable: {qualified}"
                )));
            }
        }
    }
    for (key, value) in object {
        let qualified = qualify(package_id, key);
        let Some(OntologyDescriptorV1::Property {
            value_kind,
            min_count,
            max_count,
            ..
        }) = properties.get(&qualified).copied()
        else {
            if !allow_additional_properties {
                return Err(OntologyContractError::Validation(format!(
                    "unexpected property for {class_id}: {key}"
                )));
            }
            continue;
        };
        let values: Vec<&Value> = match value {
            Value::Array(_items) if *max_count == Some(1) => {
                return Err(OntologyContractError::Validation(format!(
                    "property {key} does not accept a list"
                )));
            }
            Value::Array(items) => items.iter().collect(),
            _ if *max_count != Some(1) => {
                return Err(OntologyContractError::Validation(format!(
                    "property {key} requires a list"
                )));
            }
            _ => vec![value],
        };
        if values.len() < *min_count || max_count.is_some_and(|max| values.len() > max) {
            return Err(OntologyContractError::Validation(format!(
                "property {key} cardinality mismatch"
            )));
        }
        if values.iter().any(|item| !value_matches(value_kind, item)) {
            return Err(OntologyContractError::Validation(format!(
                "property {key} expects {value_kind}"
            )));
        }
    }
    for id in property_ids {
        if let Some(OntologyDescriptorV1::Property { min_count, .. }) =
            properties.get(&qualify(package_id, id))
        {
            if *min_count > 0
                && !object.contains_key(id)
                && !object.contains_key(&qualify(package_id, id))
            {
                return Err(OntologyContractError::Validation(format!(
                    "required property is missing: {id}"
                )));
            }
        }
    }
    Ok(())
}

fn value_matches(kind: &str, value: &Value) -> bool {
    match kind {
        "string" | "timestamp" | "logical_ref" => value.is_string(),
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "number" => value.is_number(),
        "boolean" => value.is_boolean(),
        "json" => true,
        _ => false,
    }
}

pub fn validate_ontology_edge_roles(
    view: &ComposedOntologyViewV1,
    shape_id: &str,
    roles: &BTreeMap<String, Vec<OntologyEndpointV1>>,
) -> Result<(), OntologyContractError> {
    let Some(OntologyDescriptorV1::EdgeShape {
        roles: declared_roles,
        ..
    }) = view.get(shape_id)
    else {
        return Err(OntologyContractError::Validation(format!(
            "unknown ontology edge shape: {shape_id}"
        )));
    };
    if let Some(unknown) = roles
        .keys()
        .find(|key| !declared_roles.iter().any(|role| role.role_id == **key))
    {
        return Err(OntologyContractError::Validation(format!(
            "unknown edge role: {unknown}"
        )));
    }
    let package_id = shape_id
        .split_once(':')
        .map(|(package, _)| package)
        .unwrap_or_default();
    for role in declared_roles {
        let endpoints = roles
            .get(&role.role_id)
            .map(Vec::as_slice)
            .unwrap_or_default();
        if endpoints.len() < role.min_count
            || role.max_count.is_some_and(|max| endpoints.len() > max)
        {
            return Err(OntologyContractError::Validation(format!(
                "edge role {} cardinality mismatch",
                role.role_id
            )));
        }
        for endpoint in endpoints {
            if endpoint.target_kind != role.target_kind {
                return Err(OntologyContractError::Validation(format!(
                    "edge role {} has an invalid target kind",
                    role.role_id
                )));
            }
            let kind = view.get(&endpoint.class_id).map(OntologyDescriptorV1::kind);
            let valid = if role.target_kind == "node" {
                kind == Some("ontology_class")
            } else {
                matches!(kind, Some("ontology_relation" | "ontology_edge_shape"))
            };
            if !valid {
                return Err(OntologyContractError::Validation(format!(
                    "edge role {} has an invalid target",
                    role.role_id
                )));
            }
            let allowed: BTreeSet<_> = role
                .allowed_class_ids
                .iter()
                .map(|value| qualify(package_id, value))
                .collect();
            if !allowed.is_empty() && !allowed.contains(&endpoint.class_id) {
                return Err(OntologyContractError::Validation(format!(
                    "edge role {} has an invalid class",
                    role.role_id
                )));
            }
        }
    }
    Ok(())
}
