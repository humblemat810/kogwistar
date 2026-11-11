// Storage-neutral adjacency, traversal, and graph visualization kernels. Python parity: kogwistar.graph_query and server graph-visualization projections.
/// Compute Python-compatible node/edge adjacency from canonical edge records.
///
/// This helper is deliberately storage-neutral: callers provide the already
/// scope-filtered node records and edge projections. It does not authorize access,
/// infer a namespace, or treat a projection row as authoritative outside the
/// caller's read boundary. Tombstoned rows are excluded so a stale projection
/// cannot reintroduce a deleted relationship.
pub fn graph_neighbors_from_records(
    rid: &str,
    nodes: &[GraphRecord],
    edges: &[GraphRecord],
    doc_id: Option<&str>,
) -> GraphNeighbors {
    graph_neighbors_from_records_with_jump(rid, nodes, edges, doc_id, true)
}

/// Resolve a node ID to its active redirect target using a scoped graph snapshot.
///
/// This mirrors Python's `resolve_mode="redirect"`: active nodes resolve to
/// themselves, tombstones without a target and broken/cyclic chains resolve to
/// no node. Callers must still apply ACL filtering to the returned target.
pub fn resolve_graph_node_redirect_id(rid: &str, nodes: &[GraphRecord]) -> Option<String> {
    resolve_graph_entity_redirect_id(rid, nodes)
}

/// Resolve an edge ID to its active redirect target from an edge snapshot.
/// The supplied records must be edges; callers remain responsible for ACL checks.
pub fn resolve_graph_edge_redirect_id(rid: &str, edges: &[GraphRecord]) -> Option<String> {
    resolve_graph_entity_redirect_id(rid, edges)
}

fn resolve_graph_entity_redirect_id(rid: &str, records: &[GraphRecord]) -> Option<String> {
    let mut current = rid.trim().to_owned();
    if current.is_empty() {
        return None;
    }

    let mut visited = BTreeSet::new();
    loop {
        if !visited.insert(current.clone()) {
            return None;
        }
        let record = records.iter().find(|record| record.id == current)?;
        if let Some(target) = record
            .metadata
            .get("redirect_to_id")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|target| !target.is_empty())
        {
            current = target.to_owned();
            continue;
        }
        if is_tombstoned(&record.metadata) {
            return None;
        }
        return Some(current);
    }
}

/// Compute adjacency with the caller-selected node-to-opposite-node jump.
///
/// Python's `GraphQuery.k_hop` intentionally disables this jump so traversal
/// advances through explicit edge IDs one layer at a time. Keeping the mode
/// in the storage kernel prevents a native caller from silently changing the
/// graph distance semantics.
// PYTHON_PARITY_V1: rust=rust/crates/kogwistar-store/src/lib.rs#graph_neighbors_from_records_with_jump; python=kogwistar/graph_query.py#GraphQuery.neighbors; fixture=contracts/golden/graph-neighbors-cross-kind-id-v1.json; rust_test=rust/crates/kogwistar-store/src/lib.rs#graph_neighbors_preserve_cross_kind_same_id_from_shared_fixture; python_test=tests/core/test_edge_endpoint_rows.py#test_graph_query_sync_endpoint_merge_preserves_cross_kind_same_id
pub fn graph_neighbors_from_records_with_jump(
    rid: &str,
    nodes: &[GraphRecord],
    edges: &[GraphRecord],
    doc_id: Option<&str>,
    allow_jump_edge: bool,
) -> GraphNeighbors {
    let mut result = GraphNeighbors::default();
    let node_ids: BTreeSet<&str> = nodes
        .iter()
        .filter(|node| !is_tombstoned(&node.metadata))
        .map(|node| node.id.as_str())
        .collect();
    let edge_ids: BTreeSet<&str> = edges
        .iter()
        .filter(|edge| !is_tombstoned(&edge.metadata))
        .map(|edge| edge.id.as_str())
        .collect();
    let is_node = node_ids.contains(rid);

    for edge in edges {
        if is_tombstoned(&edge.metadata) {
            continue;
        }
        let Some(document) = edge
            .document
            .as_deref()
            .and_then(|value| serde_json::from_str::<Value>(value).ok())
        else {
            continue;
        };
        if doc_id.is_some_and(|expected| {
            edge.metadata
                .get("doc_id")
                .and_then(Value::as_str)
                .or_else(|| document.get("doc_id").and_then(Value::as_str))
                != Some(expected)
        }) {
            continue;
        }

        let ids = |field: &str| {
            document
                .get(field)
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
        };
        let source_nodes: Vec<&str> = ids("source_ids").collect();
        let target_nodes: Vec<&str> = ids("target_ids").collect();
        let source_edges: Vec<&str> = ids("source_edge_ids").collect();
        let target_edges: Vec<&str> = ids("target_edge_ids").collect();

        if is_node && (source_nodes.contains(&rid) || target_nodes.contains(&rid)) {
            result.edges.insert(edge.id.clone());
            if allow_jump_edge {
                for endpoint in source_nodes.iter().chain(target_nodes.iter()) {
                    if *endpoint != rid && node_ids.contains(endpoint) {
                        result.nodes.insert((*endpoint).to_owned());
                    }
                }
            }
        }
        if edge.id == rid {
            for endpoint in source_nodes.iter().chain(target_nodes.iter()) {
                if node_ids.contains(endpoint) {
                    result.nodes.insert((*endpoint).to_owned());
                }
            }
            for endpoint in source_edges.iter().chain(target_edges.iter()) {
                if edge_ids.contains(endpoint) {
                    result.edges.insert((*endpoint).to_owned());
                }
            }
        }
    }

    result
}

/// Find visible edge IDs using the same exact relation/document and
/// case-insensitive endpoint-label semantics as Python `GraphQuery.find_edges`.
/// Callers must provide records after scope and ACL filtering.
pub fn graph_find_edges_from_records(
    query: &GraphFindEdgesQuery,
    nodes: &[GraphRecord],
    edges: &[GraphRecord],
) -> Vec<String> {
    let node_labels = nodes
        .iter()
        .filter(|node| !is_tombstoned(&node.metadata))
        .filter_map(|node| {
            let document = node.document.as_deref()?.parse::<Value>().ok()?;
            Some((
                node.id.clone(),
                document
                    .get("label")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
            ))
        })
        .collect::<BTreeMap<_, _>>();
    let labels_match = |labels: &[&str], needle: Option<&str>| {
        needle.is_none_or(|needle| {
            let needle = needle.to_lowercase();
            labels
                .iter()
                .any(|label| label.to_lowercase().contains(&needle))
        })
    };
    let mut result = edges
        .iter()
        .filter(|edge| !is_tombstoned(&edge.metadata))
        .filter_map(|edge| {
            let document = edge.document.as_deref()?.parse::<Value>().ok()?;
            let relation = document.get("relation").and_then(Value::as_str);
            if query
                .relation
                .as_deref()
                .is_some_and(|expected| relation != Some(expected))
            {
                return None;
            }
            let document_id = edge
                .metadata
                .get("doc_id")
                .and_then(Value::as_str)
                .or_else(|| document.get("doc_id").and_then(Value::as_str));
            if query
                .doc_id
                .as_deref()
                .is_some_and(|expected| document_id != Some(expected))
            {
                return None;
            }
            let ids = |field: &str| {
                document
                    .get(field)
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter_map(Value::as_str)
            };
            let source_labels = ids("source_ids")
                .filter_map(|id| node_labels.get(id).map(String::as_str))
                .collect::<Vec<_>>();
            let target_labels = ids("target_ids")
                .filter_map(|id| node_labels.get(id).map(String::as_str))
                .collect::<Vec<_>>();
            if !labels_match(&source_labels, query.src_label_contains.as_deref())
                || !labels_match(&target_labels, query.tgt_label_contains.as_deref())
            {
                return None;
            }
            Some(edge.id.clone())
        })
        .collect::<Vec<_>>();
    result.sort();
    result
}

/// Find a bounded path through visible node and edge records.
///
/// The caller must provide records after scope and ACL filtering. Traversal
/// reuses the same adjacency kernel as `kg_neighbors`, so a Rust-native query
/// cannot bypass the authority boundary by walking raw projections.
pub fn graph_shortest_path_from_records(
    query: &GraphShortestPathQuery,
    nodes: &[GraphRecord],
    edges: &[GraphRecord],
) -> Vec<String> {
    if query.src_id == query.dst_id {
        return vec![query.src_id.clone()];
    }
    let mut queue =
        std::collections::VecDeque::from([(query.src_id.clone(), vec![query.src_id.clone()])]);
    let mut seen = BTreeSet::from([query.src_id.clone()]);
    while let Some((current, path)) = queue.pop_front() {
        if path.len().saturating_sub(1) >= query.max_depth {
            continue;
        }
        let neighbors = graph_neighbors_from_records_with_jump(
            &current,
            nodes,
            edges,
            query.doc_id.as_deref(),
            false,
        );
        for next in neighbors.nodes.into_iter().chain(neighbors.edges) {
            if !seen.insert(next.clone()) {
                continue;
            }
            let mut candidate = path.clone();
            candidate.push(next.clone());
            if next == query.dst_id {
                return candidate;
            }
            queue.push_back((next, candidate));
        }
    }
    Vec::new()
}

/// Expand a bounded frontier in deterministic layers.
///
/// The result mirrors Python `GraphQuery.k_hop`: each layer contains the
/// nodes and edges discovered from the prior frontier, while visited IDs are
/// not re-expanded. Records must already be scope- and ACL-filtered.
pub fn graph_k_hop_from_records(
    query: &GraphKHopQuery,
    nodes: &[GraphRecord],
    edges: &[GraphRecord],
) -> Vec<GraphNeighbors> {
    let mut visited = BTreeSet::new();
    let mut frontier = query.start_ids.iter().cloned().collect::<BTreeSet<_>>();
    let mut layers = Vec::with_capacity(query.k);
    for _ in 0..query.k {
        let mut next_frontier = BTreeSet::new();
        let mut layer = GraphNeighbors::default();
        for rid in &frontier {
            if !visited.insert(rid.clone()) {
                continue;
            }
            let neighbors = graph_neighbors_from_records_with_jump(
                rid,
                nodes,
                edges,
                query.doc_id.as_deref(),
                false,
            );
            layer.nodes.extend(neighbors.nodes.iter().cloned());
            layer.edges.extend(neighbors.edges.iter().cloned());
            next_frontier.extend(neighbors.nodes);
            next_frontier.extend(neighbors.edges);
        }
        next_frontier.retain(|id| !visited.contains(id));
        layers.push(layer);
        frontier = next_frontier;
    }
    layers
}

/// Project scope- and ACL-filtered graph records to the Python D3 MCP shape.
///
/// This is a serialization kernel only: callers own scope selection and ACL
/// filtering. Missing or inaccessible endpoints are omitted from links so a
/// visible edge cannot disclose identifiers for hidden entities.
fn graph_record_has_reference(
    record: &GraphRecord,
    insertion_method: &str,
    doc_id: Option<&str>,
) -> bool {
    let Some(document) = record_json(record) else {
        return false;
    };
    let references = document
        .get("references")
        .filter(|value| value.as_array().is_some_and(|items| !items.is_empty()))
        .or_else(|| document.get("mentions"))
        .and_then(Value::as_array);
    references.is_some_and(|references| {
        references.iter().any(|reference| {
            reference.get("insertion_method").and_then(Value::as_str) == Some(insertion_method)
                && doc_id.is_none_or(|expected_doc| {
                    reference.get("doc_id").and_then(Value::as_str) == Some(expected_doc)
                        || reference
                            .get("document_page_url")
                            .and_then(Value::as_str)
                            .is_some_and(|url| url.contains(expected_doc))
                })
        })
    })
}

fn graph_viz_selected_records<'a>(
    records: &'a [GraphRecord],
    doc_id: Option<&str>,
    insertion_method: Option<&str>,
) -> Vec<&'a GraphRecord> {
    // Python's `_ids_by_doc` treats an empty query value like an omitted filter.
    let selection_doc_id = doc_id.filter(|value| !value.is_empty());
    let base = records
        .iter()
        .filter(|record| {
            !is_tombstoned(&record.metadata)
                && selection_doc_id.is_none_or(|expected| {
                    record_field(record, "doc_id")
                        .and_then(|value| value.as_str().map(str::to_owned))
                        .as_deref()
                        == Some(expected)
                })
        })
        .collect::<Vec<_>>();
    // Python's `_filter_by_insertion_method` treats an empty query as no filter.
    let Some(expected_method) = insertion_method.filter(|value| !value.is_empty()) else {
        return base;
    };
    let matching = base
        .iter()
        .copied()
        .filter(|record| graph_record_has_reference(record, expected_method, selection_doc_id))
        .collect::<Vec<_>>();
    if matching.is_empty() { base } else { matching }
}

fn graph_viz_endpoint_complete_edges<'a>(
    nodes: &[GraphRecord],
    edges: &'a [GraphRecord],
    doc_id: Option<&str>,
    insertion_method: Option<&str>,
) -> Vec<&'a GraphRecord> {
    let node_ids = graph_viz_selected_records(nodes, doc_id, insertion_method)
        .into_iter()
        .map(|node| node.id.as_str())
        .collect::<BTreeSet<_>>();
    let candidates = graph_viz_selected_records(edges, doc_id, insertion_method);
    let mut visible_edge_ids = candidates
        .iter()
        .map(|edge| edge.id.as_str())
        .collect::<BTreeSet<_>>();

    loop {
        let next = candidates
            .iter()
            .filter(|edge| {
                ["source_ids", "target_ids"]
                    .into_iter()
                    .flat_map(|field| graph_record_string_list(edge, field))
                    .all(|endpoint_id| node_ids.contains(endpoint_id.as_str()))
                    && ["source_edge_ids", "target_edge_ids"]
                        .into_iter()
                        .flat_map(|field| graph_record_string_list(edge, field))
                        .all(|endpoint_id| visible_edge_ids.contains(endpoint_id.as_str()))
            })
            .map(|edge| edge.id.as_str())
            .collect::<BTreeSet<_>>();
        if next == visible_edge_ids {
            break;
        }
        visible_edge_ids = next;
    }

    candidates
        .into_iter()
        .filter(|edge| visible_edge_ids.contains(edge.id.as_str()))
        .collect()
}

pub fn graph_viz_d3_from_records(
    nodes: &[GraphRecord],
    edges: &[GraphRecord],
    doc_id: Option<&str>,
    mode: &str,
    insertion_method: Option<&str>,
) -> Value {
    let mut node_values = BTreeMap::<String, Value>::new();
    let mut node_order = Vec::<String>::new();
    let selected_nodes = graph_viz_selected_records(nodes, doc_id, insertion_method);
    for node in selected_nodes {
        let mut value = record_json(node)
            .filter(Value::is_object)
            .unwrap_or_else(|| Value::Object(Map::new()));
        let object = value.as_object_mut().expect("object was checked");
        object.insert("id".to_owned(), Value::String(node.id.clone()));
        object.insert(
            "label".to_owned(),
            record_field(node, "label")
                .filter(|value| !value.is_null())
                .unwrap_or_else(|| Value::String(node.id.clone())),
        );
        object.insert("type".to_owned(), Value::String("entity".to_owned()));
        if let Some(summary) = record_field(node, "summary") {
            object.insert("summary".to_owned(), summary);
        }
        object.insert(
            "properties".to_owned(),
            record_field(node, "properties")
                .filter(Value::is_object)
                .unwrap_or_else(|| Value::Object(Map::new())),
        );
        if !node_values.contains_key(&node.id) {
            node_order.push(node.id.clone());
        }
        node_values.insert(node.id.clone(), value);
    }
    let node_ids = node_values.keys().cloned().collect::<BTreeSet<_>>();
    let visible_edges = graph_viz_endpoint_complete_edges(nodes, edges, doc_id, insertion_method);
    let edge_ids = visible_edges
        .iter()
        .map(|record| record.id.clone())
        .collect::<BTreeSet<_>>();
    let reify = mode.eq_ignore_ascii_case("reify");
    let mut links = Vec::new();

    for edge in &visible_edges {
        let relation = graph_visualization_edge_relation(edge);
        let edge_label = relation
            .clone()
            .unwrap_or_else(|| Value::String("edge".to_owned()));
        let properties = record_field(edge, "properties")
            .filter(Value::is_object)
            .unwrap_or_else(|| Value::Object(Map::new()));
        let sources = graph_record_string_list(edge, "source_ids");
        let targets = graph_record_string_list(edge, "target_ids");
        let source_edges = graph_record_string_list(edge, "source_edge_ids");
        let target_edges = graph_record_string_list(edge, "target_edge_ids");

        if reify {
            if !node_values.contains_key(&edge.id) {
                let mut edge_node = record_json(edge)
                    .filter(Value::is_object)
                    .unwrap_or_else(|| Value::Object(Map::new()));
                let object = edge_node.as_object_mut().expect("object was checked");
                object.insert("id".to_owned(), Value::String(edge.id.clone()));
                object.insert("label".to_owned(), edge_label);
                object.insert("type".to_owned(), Value::String("edge-node".to_owned()));
                object.insert(
                    "summary".to_owned(),
                    record_field(edge, "summary").unwrap_or(Value::Null),
                );
                object.insert("properties".to_owned(), properties.clone());
                node_order.push(edge.id.clone());
                node_values.insert(edge.id.clone(), edge_node);
            }
            for source in sources.iter().filter(|id| node_ids.contains(*id)) {
                links.push(json!({"source": source, "target": edge.id, "relation": relation, "role": "src", "properties": properties}));
            }
            for source in source_edges.iter().filter(|id| edge_ids.contains(*id)) {
                links.push(json!({"source": source, "target": edge.id, "relation": relation, "role": "src", "properties": properties}));
            }
            for target in targets.iter().filter(|id| node_ids.contains(*id)) {
                links.push(json!({"source": edge.id, "target": target, "relation": relation, "role": "tgt", "properties": properties}));
            }
            for target in target_edges.iter().filter(|id| edge_ids.contains(*id)) {
                links.push(json!({"source": edge.id, "target": target, "relation": relation, "role": "tgt", "properties": properties}));
            }
        } else {
            for source in sources.iter().filter(|id| node_ids.contains(*id)) {
                for target in targets.iter().filter(|id| node_ids.contains(*id)) {
                    links.push(json!({"source": source, "target": target, "relation": relation, "properties": properties}));
                }
            }
        }
    }
    json!({
        "nodes": node_order.into_iter().filter_map(|id| node_values.remove(&id)).collect::<Vec<_>>(),
        "links": links,
        "mode": mode,
        "doc_id": doc_id,
    })
}

/// Project the D3 projection into Python Cytoscape's elements envelope.
pub fn graph_viz_cytoscape_from_records(
    nodes: &[GraphRecord],
    edges: &[GraphRecord],
    doc_id: Option<&str>,
    mode: &str,
    insertion_method: Option<&str>,
) -> Value {
    let d3 = graph_viz_d3_from_records(nodes, edges, doc_id, mode, insertion_method);
    let mut elements = Vec::new();
    let visible_node_ids = d3["nodes"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|node| node["type"] == "entity")
        .filter_map(|node| node["id"].as_str().map(str::to_owned))
        .collect::<BTreeSet<_>>();
    let selected_edges = graph_viz_endpoint_complete_edges(nodes, edges, doc_id, insertion_method);
    let visible_edge_ids = selected_edges
        .iter()
        .map(|edge| edge.id.clone())
        .collect::<BTreeSet<_>>();
    let d3_nodes = d3["nodes"]
        .as_array()
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    for node in d3_nodes.iter().filter(|node| node["type"] == "entity") {
        let data = if node["type"] == "entity" {
            json!({"id": node["id"], "label": node["label"], "type": "entity"})
        } else {
            json!({"id": node["id"], "label": node["label"], "type": "edge-node"})
        };
        elements.push(json!({
            "data": data,
            "classes": if node["type"] == "edge-node" { "edge-node" } else { "" },
        }));
    }
    if mode.eq_ignore_ascii_case("reify") {
        let links = d3["links"].as_array().into_iter().flatten().fold(
            BTreeMap::<String, Vec<&Value>>::new(),
            |mut grouped, link| {
                let role = link["role"].as_str().unwrap_or_default();
                let edge_id = if role == "src" {
                    link["target"].as_str().unwrap_or_default()
                } else {
                    link["source"].as_str().unwrap_or_default()
                };
                grouped.entry(edge_id.to_owned()).or_default().push(link);
                grouped
            },
        );
        for edge in &selected_edges {
            if let Some(node) = d3_nodes
                .iter()
                .find(|node| node["type"] == "edge-node" && node["id"] == edge.id)
            {
                elements.push(json!({
                    "data": {"id": node["id"], "label": node["label"], "type": "edge-node"},
                    "classes": "edge-node",
                }));
            }
            for link in links.get(&edge.id).into_iter().flatten() {
                let role = link["role"].as_str().unwrap_or_default();
                let edge_id = if role == "src" {
                    link["target"].as_str().unwrap_or_default()
                } else {
                    link["source"].as_str().unwrap_or_default()
                };
                let endpoint_id = if role == "src" {
                    link["source"].as_str().unwrap_or_default()
                } else {
                    link["target"].as_str().unwrap_or_default()
                };
                let endpoint_is_edge = visible_edge_ids.contains(endpoint_id);
                let endpoint_role = match (role, endpoint_is_edge) {
                    ("src", true) => "srcE",
                    ("src", false) => "src",
                    (_, true) => "tgtE",
                    (_, false) => "tgt",
                };
                elements.push(json!({
                    "data": {
                        "id": format!("{edge_id}::{endpoint_role}::{endpoint_id}"),
                        "source": link["source"],
                        "target": link["target"],
                        "label": link["relation"],
                    },
                    "classes": role,
                }));
            }
        }
    } else {
        for edge in &selected_edges {
            let relation = graph_visualization_edge_relation(edge);
            for source in graph_record_string_list(edge, "source_ids")
                .iter()
                .filter(|id| visible_node_ids.contains(*id))
            {
                for target in graph_record_string_list(edge, "target_ids")
                    .iter()
                    .filter(|id| visible_node_ids.contains(*id))
                {
                    elements.push(json!({
                        "data": {
                            "id": format!("{}::{}->{}", edge.id, source, target),
                            "source": source,
                            "target": target,
                            "label": relation,
                        },
                        "classes": "",
                    }));
                }
            }
        }
    }
    json!({"elements": elements, "mode": mode, "doc_id": doc_id})
}

fn graph_record_string_list(record: &GraphRecord, field: &str) -> Vec<String> {
    let value = record_field(record, field).and_then(|value| match value {
        Value::String(encoded) => serde_json::from_str::<Value>(&encoded).ok(),
        value => Some(value),
    });
    value
        .and_then(|value| value.as_array().cloned())
        .unwrap_or_default()
        .into_iter()
        .filter_map(|value| value.as_str().map(str::to_owned))
        .filter(|value| !value.is_empty())
        .collect()
}
