use serde::Serialize;
use std::collections::BTreeSet;

use super::{FROZEN_OPENAPI_ROUTES, PENDING_SERVER_CUTOVER_ROUTES};

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
    // Keep this explicit release gate: it becomes true only after the pending
    // route inventory is deliberately emptied as part of a verified cutover.
    #[allow(clippy::const_is_empty)]
    pub fn effective_server_cutover_ready(&self) -> bool {
        effective_server_cutover_ready_for(
            self.server_cutover_ready,
            PENDING_SERVER_CUTOVER_ROUTES,
            FROZEN_OPENAPI_ROUTES,
            self.implemented_route_operations,
            frozen_application_route_operations(),
        )
    }
}

fn pending_server_routes_are_valid() -> bool {
    pending_server_routes_are_valid_for(PENDING_SERVER_CUTOVER_ROUTES, FROZEN_OPENAPI_ROUTES)
}

fn pending_server_routes_are_valid_for(
    pending_routes: &[(&str, &str)],
    frozen_routes: &[(&str, &str)],
) -> bool {
    let mut seen = BTreeSet::new();
    pending_routes
        .iter()
        .all(|route| seen.insert(*route) && frozen_routes.contains(route))
}

fn effective_server_cutover_ready_for(
    declared_ready: bool,
    pending_routes: &[(&str, &str)],
    frozen_routes: &[(&str, &str)],
    implemented_routes: usize,
    expected_implemented_routes: usize,
) -> bool {
    declared_ready
        && pending_routes.is_empty()
        && pending_server_routes_are_valid_for(pending_routes, frozen_routes)
        && implemented_routes == expected_implemented_routes
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

pub const TRANSPORT_OWNED_HTTP_ROUTES: &[(&str, &str)] = &[
    ("GET", "/health"),
    ("GET", "/api/viz/cytoscape.json"),
    ("GET", "/api/viz/d3.json"),
    ("GET", "/viz/d3.bundle"),
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
    fn transport_owned_routes_are_unique_and_frozen() {
        assert!(
            TRANSPORT_OWNED_HTTP_ROUTES
                .iter()
                .all(|route| FROZEN_OPENAPI_ROUTES.contains(route))
        );
        assert_eq!(
            TRANSPORT_OWNED_HTTP_ROUTES.len(),
            TRANSPORT_OWNED_HTTP_ROUTES
                .iter()
                .collect::<BTreeSet<_>>()
                .len()
        );
    }

    #[test]
    fn pending_route_matcher_handles_query_and_path_parameters() {
        assert!(!is_pending_server_cutover_route(
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

    #[test]
    fn server_cutover_requires_closed_routes_and_reconciled_counts() {
        let frozen = [("GET", "/health"), ("GET", "/api/items")];
        let pending = [("GET", "/api/items")];

        assert!(!effective_server_cutover_ready_for(
            true, &pending, &frozen, 1, 1
        ));
        assert!(!effective_server_cutover_ready_for(
            true,
            &[],
            &frozen,
            0,
            1
        ));
        assert!(!effective_server_cutover_ready_for(
            false,
            &[],
            &frozen,
            1,
            1
        ));
        assert!(effective_server_cutover_ready_for(true, &[], &frozen, 1, 1));
    }

    #[test]
    fn pending_route_inventory_rejects_duplicates_and_unfrozen_routes() {
        let frozen = [("GET", "/health"), ("GET", "/api/items")];

        assert!(pending_server_routes_are_valid_for(
            &[("GET", "/api/items")],
            &frozen
        ));
        assert!(!pending_server_routes_are_valid_for(
            &[("GET", "/api/items"), ("GET", "/api/items")],
            &frozen
        ));
        assert!(!pending_server_routes_are_valid_for(
            &[("DELETE", "/api/items")],
            &frozen
        ));
    }
}
