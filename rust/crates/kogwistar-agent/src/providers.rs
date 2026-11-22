use crate::error::AgentError;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ProviderInactiveError {
    #[error("provider lifecycle is no longer active: {0}")]
    Inactive(String),
}

pub trait Provider: Send + Sync {
    fn provider_id(&self) -> &str;
    fn close(&self) {}
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ProviderIdentity {
    pub provider_id: String,
    pub version: String,
}

impl ProviderIdentity {
    pub fn new(
        provider_id: impl Into<String>,
        version: impl Into<String>,
    ) -> Result<Self, AgentError> {
        let provider_id = provider_id.into();
        let version = version.into();
        if provider_id.trim().is_empty() || version.trim().is_empty() {
            return Err(AgentError::Validation(
                "provider_id and version must be non-empty".to_owned(),
            ));
        }
        Ok(Self {
            provider_id: provider_id.trim().to_lowercase(),
            version,
        })
    }

    pub fn qualified_id(&self) -> String {
        format!("{}@{}", self.provider_id, self.version)
    }
}

#[derive(Clone)]
pub struct ProviderRegistration {
    pub identity: ProviderIdentity,
    pub provider: Arc<dyn Provider>,
    pub generation: u64,
}

impl std::fmt::Debug for ProviderRegistration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderRegistration")
            .field("identity", &self.identity)
            .field("generation", &self.generation)
            .finish()
    }
}

#[derive(Default)]
pub struct ProviderRegistry {
    registrations: RwLock<HashMap<String, ProviderRegistration>>,
    generations: RwLock<HashMap<String, u64>>,
}

impl ProviderRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(
        &self,
        provider: Arc<dyn Provider>,
        version: &str,
    ) -> Result<ProviderRegistration, AgentError> {
        let identity = ProviderIdentity::new(provider.provider_id(), version)?;
        let key = identity.qualified_id();
        let mut registrations = self.registrations.write().unwrap();
        if registrations.contains_key(&key) {
            return Err(AgentError::Validation(format!(
                "provider identity already registered: {key}"
            )));
        }
        let mut generations = self.generations.write().unwrap();
        let generation = generations.get(&key).copied().unwrap_or(0) + 1;
        let registration = ProviderRegistration {
            identity,
            provider,
            generation,
        };
        registrations.insert(key.clone(), registration.clone());
        generations.insert(key, generation);
        Ok(registration)
    }

    pub fn get(
        &self,
        provider_id: &str,
        version: &str,
    ) -> Result<ProviderRegistration, AgentError> {
        let key = ProviderIdentity::new(provider_id, version)?.qualified_id();
        self.registrations
            .read()
            .unwrap()
            .get(&key)
            .cloned()
            .ok_or_else(|| AgentError::Validation(format!("unknown provider identity: {key}")))
    }

    pub fn list(&self) -> Vec<ProviderRegistration> {
        self.registrations
            .read()
            .unwrap()
            .values()
            .cloned()
            .collect()
    }

    pub fn dispose(&self, registration: &ProviderRegistration) -> Result<(), AgentError> {
        let key = registration.identity.qualified_id();
        let mut registrations = self.registrations.write().unwrap();
        match registrations.get(&key) {
            Some(current) if current.generation == registration.generation => {
                registrations.remove(&key);
                Ok(())
            }
            _ => Err(AgentError::Validation(format!(
                "registration is not active owner: {key}"
            ))),
        }
    }

    pub fn run_if_active<T>(
        &self,
        registration: &ProviderRegistration,
        operation: impl FnOnce() -> T,
    ) -> Result<T, ProviderInactiveError> {
        let key = registration.identity.qualified_id();
        let active = self
            .registrations
            .read()
            .unwrap()
            .get(&key)
            .is_some_and(|current| current.generation == registration.generation);
        if active {
            Ok(operation())
        } else {
            Err(ProviderInactiveError::Inactive(format!(
                "{key}@{}",
                registration.generation
            )))
        }
    }
}
