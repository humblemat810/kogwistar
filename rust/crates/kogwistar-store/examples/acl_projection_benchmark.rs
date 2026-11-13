use kogwistar_store::{GraphAclPrincipal, GraphAclProjection, GraphRecord};
use serde_json::{Map, Value};
use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::hint::black_box;
use std::time::Instant;

fn records(value: &Value, key: &str) -> Vec<GraphRecord> {
    value[key]
        .as_array()
        .expect("workload array")
        .iter()
        .map(|item| GraphRecord {
            id: item["id"].as_str().expect("record id").to_owned(),
            document: None,
            metadata: item["metadata"]
                .as_object()
                .cloned()
                .unwrap_or_else(Map::new),
            embedding: None,
        })
        .collect()
}

fn main() {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let input = args
        .windows(2)
        .find(|pair| pair[0] == "--input")
        .map(|pair| pair[1].as_str())
        .expect("--input PATH is required");
    let samples = args
        .windows(2)
        .find(|pair| pair[0] == "--samples")
        .map(|pair| pair[1].parse::<usize>().expect("sample count"))
        .unwrap_or(5);
    let iterations = args
        .windows(2)
        .find(|pair| pair[0] == "--iterations")
        .map(|pair| pair[1].parse::<usize>().expect("iteration count"))
        .unwrap_or(20);
    assert!(samples >= 2 && iterations > 0);

    let workload: Value =
        serde_json::from_slice(&fs::read(input).expect("read shared ACL workload"))
            .expect("parse shared ACL workload");
    let nodes = records(&workload, "nodes");
    let edges = records(&workload, "edges");
    let targets = nodes
        .iter()
        .filter(|record| {
            record.metadata.get("entity_type").and_then(Value::as_str) != Some("acl_record")
        })
        .cloned()
        .collect::<Vec<_>>();
    let principal = GraphAclPrincipal {
        principal_id: workload["principal"]["id"]
            .as_str()
            .expect("principal id")
            .to_owned(),
        groups: workload["principal"]["groups"]
            .as_array()
            .expect("principal groups")
            .iter()
            .map(|group| group.as_str().expect("group").to_owned())
            .collect::<BTreeSet<_>>(),
        security_scope: workload["principal"]["security_scope"]
            .as_str()
            .map(str::to_owned),
    };
    let truth_graph = workload["truth_graph"].as_str().expect("truth graph");
    let mut output = Vec::with_capacity(samples + 1);
    for sample in 0..=samples {
        let projection =
            GraphAclProjection::from_complete_snapshot(&nodes, &edges, truth_graph, &principal);
        let started = Instant::now();
        let mut visible_ids = Vec::new();
        for _ in 0..iterations {
            let filtered = projection.filter_records(black_box(&targets), black_box(&edges));
            visible_ids = filtered
                .nodes()
                .iter()
                .map(|record| record.id.clone())
                .collect();
            black_box(&visible_ids);
        }
        let elapsed_ns = started.elapsed().as_nanos();
        output.push(serde_json::json!({
            "kind": if sample == 0 { "warmup" } else { "sample" },
            "sample": sample,
            "iterations": iterations,
            "elapsed_ns": elapsed_ns,
            "visible_count": visible_ids.len(),
            "visible_ids": visible_ids,
        }));
    }
    println!(
        "{}",
        serde_json::json!({
            "schema": "kogwistar-acl-projection-rust-result/v1",
            "items": nodes.len(),
            "edges": edges.len(),
            "results": output,
        })
    );
}
