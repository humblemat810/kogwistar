use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use thiserror::Error;

pub const DELEGATION_AUTHORITY_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum DelegationError {
    #[error("child requested capabilities absent from parent: {0}")]
    CapabilityExpansion(String),
    #[error("delegated_by must be a non-empty subject")]
    EmptyDelegatingSubject,
}

/// Versioned, language-neutral authority carried by a nested workflow.
///
/// This is an execution contract, not an authentication credential. The
/// effect owner must derive it from the parent authority and never trust a
/// child-provided effective set.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DelegationAuthorityV1 {
    pub schema_version: u32,
    pub delegated_by: String,
    pub effective_capabilities: Vec<String>,
}

impl DelegationAuthorityV1 {
    pub fn new(
        delegated_by: impl Into<String>,
        effective_capabilities: &BTreeSet<String>,
    ) -> Result<Self, DelegationError> {
        let delegated_by = delegated_by.into();
        if delegated_by.trim().is_empty() {
            return Err(DelegationError::EmptyDelegatingSubject);
        }
        Ok(Self {
            schema_version: DELEGATION_AUTHORITY_SCHEMA_VERSION,
            delegated_by,
            effective_capabilities: effective_capabilities.iter().cloned().collect(),
        })
    }
}

/// Enforce the non-expansion invariant shared by Python and Rust runtimes.
pub fn attenuate_capabilities(
    parent: &BTreeSet<String>,
    requested: Option<&BTreeSet<String>>,
) -> Result<BTreeSet<String>, DelegationError> {
    let requested = requested.unwrap_or(parent);
    let missing = requested.difference(parent).cloned().collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(DelegationError::CapabilityExpansion(missing.join(",")));
    }
    Ok(requested.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attenuation_accepts_equal_or_narrower_sets_only() {
        let parent = BTreeSet::from(["graph.read".to_owned(), "workflow.run.write".to_owned()]);
        let child = BTreeSet::from(["graph.read".to_owned()]);
        assert_eq!(
            attenuate_capabilities(&parent, Some(&child)).unwrap(),
            child
        );
        assert_eq!(attenuate_capabilities(&parent, None).unwrap(), parent);
        let wider = BTreeSet::from(["service.manage".to_owned()]);
        assert!(matches!(
            attenuate_capabilities(&parent, Some(&wider)),
            Err(DelegationError::CapabilityExpansion(_))
        ));
    }

    #[test]
    fn authority_is_deterministic_and_versioned() {
        let caps = BTreeSet::from(["workflow.run.write".to_owned(), "graph.read".to_owned()]);
        let authority = DelegationAuthorityV1::new("agent:parent", &caps).unwrap();
        assert_eq!(authority.schema_version, 1);
        assert_eq!(
            authority.effective_capabilities,
            vec!["graph.read", "workflow.run.write"]
        );
    }

    #[test]
    fn authority_matches_shared_golden_fixture() {
        let authority: DelegationAuthorityV1 = serde_json::from_str(include_str!(
            "../../../../contracts/golden/delegation-authority-v1.json"
        ))
        .unwrap();
        assert_eq!(
            authority.schema_version,
            DELEGATION_AUTHORITY_SCHEMA_VERSION
        );
        assert_eq!(authority.delegated_by, "agent:parent");
        assert_eq!(
            authority.effective_capabilities,
            vec!["graph.read", "workflow.run.write"]
        );
    }
}
