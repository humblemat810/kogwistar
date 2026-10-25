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

pub(crate) fn is_pending_server_cutover_route(method: &str, path_and_query: &str) -> bool {
    let path = path_and_query
        .split_once('?')
        .map(|(path, _)| path)
        .unwrap_or(path_and_query);
    PENDING_SERVER_CUTOVER_ROUTES
        .iter()
        .any(|(pending_method, template)| {
            *pending_method == method && route_template_matches(template, path)
        })
}

fn route_template_matches(template: &str, path: &str) -> bool {
    let template_segments = template.split('/').collect::<Vec<_>>();
    let path_segments = path.split('/').collect::<Vec<_>>();
    template_segments.len() == path_segments.len()
        && template_segments
            .iter()
            .zip(path_segments)
            .all(|(expected, actual)| {
                if expected.starts_with('{') && expected.ends_with('}') {
                    !actual.is_empty()
                } else {
                    *expected == actual
                }
            })
}

pub const PENDING_SERVER_CUTOVER_ROUTES: &[(&str, &str)] = &[
    ("DELETE", "/admin/doc/{doc_id}"),
    ("GET", "/api/search_index_hybrid"),
    ("GET", "/api/viz/cytoscape.json"),
    ("GET", "/api/viz/d3.json"),
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
    fn pending_route_matcher_handles_query_and_path_parameters() {
        assert!(is_pending_server_cutover_route(
            "POST",
            "/api/conversations/c-1/turns:answer?mode=sync"
        ));
        assert!(is_pending_server_cutover_route(
            "DELETE",
            "/admin/doc/doc-1"
        ));
        assert!(!is_pending_server_cutover_route(
            "GET",
            "/api/workflow/tools/audit"
        ));
        assert!(!is_pending_server_cutover_route("POST", "/admin/doc/doc-1"));
    }

    #[test]
    fn default_snapshot_cannot_claim_server_cutover() {
        assert!(!ImplementationSnapshot::default().effective_server_cutover_ready());
    }
}
