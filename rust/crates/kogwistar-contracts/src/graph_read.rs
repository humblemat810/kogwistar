use serde::{Deserialize, Serialize};

pub const GRAPH_NEIGHBORS_REQUEST_VERSION: u32 = 1;

fn default_request_version() -> u32 {
    GRAPH_NEIGHBORS_REQUEST_VERSION
}

fn default_truth_graph() -> String {
    "knowledge".to_owned()
}

/// Shared request DTO for graph-neighbor reads.
///
/// The schema version is optional on input so existing internal callers keep
/// working, but every parsed request is normalized to an explicit version.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GraphNeighborsRequestV1 {
    #[serde(default = "default_request_version")]
    pub schema_version: u32,
    pub rid: String,
    #[serde(default)]
    pub doc_id: Option<String>,
    #[serde(default = "default_truth_graph")]
    pub truth_graph: String,
}

impl GraphNeighborsRequestV1 {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.schema_version != GRAPH_NEIGHBORS_REQUEST_VERSION {
            return Err("unsupported graph-neighbor request schema version");
        }
        if self.rid.trim().is_empty() {
            return Err("rid is required");
        }
        if self.truth_graph.trim().is_empty() {
            return Err("truth_graph is required");
        }
        Ok(())
    }
}

pub fn graph_neighbors_request_from_str(
    payload_json: &str,
) -> Result<GraphNeighborsRequestV1, serde_json::Error> {
    serde_json::from_str(payload_json)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_neighbor_request_defaults_version_and_validates() {
        let request = graph_neighbors_request_from_str(r#"{"rid":"n1"}"#).unwrap();
        assert_eq!(request.schema_version, GRAPH_NEIGHBORS_REQUEST_VERSION);
        assert_eq!(request.doc_id, None);
        assert_eq!(request.truth_graph, "knowledge");
        assert!(request.validate().is_ok());
    }

    #[test]
    fn graph_neighbor_request_rejects_unknown_or_invalid_fields() {
        assert!(graph_neighbors_request_from_str(r#"{"rid":"n1","write":true}"#).is_err());
        let empty = graph_neighbors_request_from_str(r#"{"schema_version":1,"rid":" "}"#).unwrap();
        assert_eq!(empty.validate(), Err("rid is required"));
        let empty_graph = graph_neighbors_request_from_str(
            r#"{"schema_version":1,"rid":"n1","truth_graph":" "}"#,
        )
        .unwrap();
        assert_eq!(empty_graph.validate(), Err("truth_graph is required"));
        let future =
            graph_neighbors_request_from_str(r#"{"schema_version":2,"rid":"n1"}"#).unwrap();
        assert_eq!(
            future.validate(),
            Err("unsupported graph-neighbor request schema version")
        );
    }
}
