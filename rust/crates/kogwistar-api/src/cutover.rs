use serde::Serialize;
use std::collections::BTreeSet;

use super::FROZEN_OPENAPI_ROUTES;

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ImplementationSnapshot {
    pub mode: String,
    pub contract_version: u32,
    pub schema_version: u32,
    pub frozen_route_operations: usize,
    pub implemented_route_operations: usize,
    pub runtime_cutover_ready: bool,
    pub server_cutover_ready: bool,
}

impl ImplementationSnapshot {
    pub fn effective_server_cutover_ready(&self) -> bool {
        self.server_cutover_ready
            && PENDING_SERVER_CUTOVER_ROUTES.is_empty()
            && pending_server_routes_are_valid()
            && self.implemented_route_operations == frozen_application_route_operations()
    }
}

fn pending_server_routes_are_valid() -> bool {
    let mut seen = BTreeSet::new();
    PENDING_SERVER_CUTOVER_ROUTES
        .iter()
        .all(|route| seen.insert(*route) && FROZEN_OPENAPI_ROUTES.contains(route))
}

pub const PENDING_SERVER_CUTOVER_ROUTES: &[(&str, &str)] = &[
    ("DELETE", "/admin/doc/{doc_id}"),
    ("GET", "/api/conversations"),
    ("GET", "/api/conversations/{conversation_id}"),
    ("GET", "/api/conversations/{conversation_id}/turns"),
    ("GET", "/api/search_index_hybrid"),
    ("GET", "/api/viz/cytoscape.json"),
    ("GET", "/api/viz/d3.json"),
    ("GET", "/api/workflow/tools/audit"),
    ("GET", "/viz/d3.bundle"),
    ("POST", "/api/add_index_entries"),
    ("POST", "/api/conversations"),
    ("POST", "/api/conversations/{conversation_id}/turns:answer"),
    ("POST", "/api/document"),
    ("POST", "/api/document.upsert_tree"),
    ("POST", "/api/graph/upsert"),
];

pub const TRANSPORT_OWNED_HTTP_ROUTES: &[(&str, &str)] = &[
    ("GET", "/health"),
    ("GET", "/api/auth/login"),
    ("GET", "/api/auth/callback"),
    ("GET", "/api/auth/me"),
    ("POST", "/api/auth/logout"),
    ("POST", "/auth/dev-token"),
];

pub(crate) fn is_transport_owned_http_route(method: &str, path: &str) -> bool {
    TRANSPORT_OWNED_HTTP_ROUTES.contains(&(method, path))
}

pub(crate) fn frozen_application_route_operations() -> usize {
    FROZEN_OPENAPI_ROUTES
        .iter()
        .filter(|(method, path)| !is_transport_owned_http_route(method, path))
        .count()
}

pub const PENDING_SYSCALL_CUTOVER_OPS: &[&str] = &[
    "send_message",
    "receive_message",
    "mount_memory",
    "invoke_tool",
];

impl Default for ImplementationSnapshot {
    fn default() -> Self {
        Self {
            mode: "rust".to_owned(),
            contract_version: 1,
            schema_version: 1,
            frozen_route_operations: FROZEN_OPENAPI_ROUTES.len(),
            implemented_route_operations: frozen_application_route_operations()
                .saturating_sub(PENDING_SERVER_CUTOVER_ROUTES.len()),
            runtime_cutover_ready: false,
            server_cutover_ready: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pending_server_routes_are_unique_and_frozen() {
        assert!(pending_server_routes_are_valid());
        assert_eq!(
            PENDING_SERVER_CUTOVER_ROUTES.len(),
            PENDING_SERVER_CUTOVER_ROUTES
                .iter()
                .collect::<BTreeSet<_>>()
                .len()
        );
    }

    #[test]
    fn default_snapshot_cannot_claim_server_cutover() {
        assert!(!ImplementationSnapshot::default().effective_server_cutover_ready());
    }
}
