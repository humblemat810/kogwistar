//! Benchmark typed graph-slice projection without Python or PyO3.
//!
//! The input is the same JSON envelope fixture emitted by the Python seam
//! benchmark. JSON is read once at startup; timed iterations retain typed Rust
//! envelopes and therefore measure the no-seam projection path.

use kogwistar_contracts::GraphModelEnvelope;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::hint::black_box;
use std::time::Instant;

fn main() {
    let mut args = env::args().skip(1);
    let input = args
        .next()
        .expect("usage: graph_slice_native_benchmark <input> [iterations]");
    let iterations: usize = args
        .next()
        .as_deref()
        .unwrap_or("30")
        .parse()
        .expect("iterations must be a positive integer");
    assert!(iterations > 0, "iterations must be positive");

    let source = fs::read_to_string(input).expect("read graph-slice fixture");
    let envelopes: Vec<GraphModelEnvelope> =
        serde_json::from_str(&source).expect("parse graph-slice fixture");
    assert!(
        !envelopes.is_empty(),
        "fixture must contain at least one envelope"
    );

    let mut output = Vec::new();
    let start = Instant::now();
    for _ in 0..iterations {
        output.clear();
        output.extend(
            envelopes
                .iter()
                .map(|envelope| envelope.project_declared_slice().expect("valid slice")),
        );
        black_box(&output);
    }
    let payloads: Vec<&Value> = output.iter().map(|slice| &slice.payload).collect();
    let digest = Sha256::digest(
        serde_json::to_string(&payloads)
            .expect("serialize projected payloads")
            .as_bytes(),
    );
    let digest = hex::encode(digest);
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    println!(
        "{{\"schema\":\"kogwistar-graph-slice-native-benchmark/v1\",\"envelopes\":{},\"iterations\":{},\"payload_digest\":\"{}\",\"total_ms\":{:.6},\"per_iteration_ms\":{:.6}}}",
        envelopes.len(),
        iterations,
        digest,
        elapsed_ms,
        elapsed_ms / iterations as f64
    );
}
