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

fn project_json_pipeline(source: &str, validate_family: bool) -> String {
    let parsed: Vec<GraphModelEnvelope> =
        serde_json::from_str(source).expect("parse graph-slice fixture");
    if validate_family {
        for envelope in &parsed {
            envelope
                .validate_family_v1()
                .expect("validate graph-model family");
        }
    }
    let projected = parsed
        .iter()
        .map(|envelope| envelope.project_declared_slice().expect("valid slice"))
        .collect::<Vec<_>>();
    let payloads = projected
        .iter()
        .map(|slice| &slice.payload)
        .collect::<Vec<_>>();
    serde_json::to_string(&payloads).expect("serialize output")
}

fn median_ms(samples: &[f64]) -> f64 {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    } else {
        sorted[middle]
    }
}

fn p95_ms(samples: &[f64]) -> f64 {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = ((sorted.len() as f64 * 0.95).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    sorted[index]
}

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

    // Measure the shared JSON parse/project/encode work without family checks.
    let projection_warmup = project_json_pipeline(&source, false);
    black_box(&projection_warmup);
    let mut projection_samples_ms = Vec::with_capacity(iterations);
    let mut projection_output = String::new();
    for _ in 0..iterations {
        let start = Instant::now();
        projection_output = project_json_pipeline(&source, false);
        projection_samples_ms.push(start.elapsed().as_secs_f64() * 1000.0);
        black_box(&projection_output);
    }
    let json_projection_elapsed_ms: f64 = projection_samples_ms.iter().sum();
    let json_projection_digest = hex::encode(Sha256::digest(projection_output.as_bytes()));
    assert_eq!(
        json_projection_digest, digest,
        "JSON projection output differs"
    );

    // Measure validation separately: decode the same canonical JSON, validate
    // typed graph families, project, and serialize the projected output.
    let validation_warmup = project_json_pipeline(&source, true);
    black_box(&validation_warmup);
    let mut validation_samples_ms = Vec::with_capacity(iterations);
    let mut serialized_output = String::new();
    for _ in 0..iterations {
        let start = Instant::now();
        serialized_output = project_json_pipeline(&source, true);
        validation_samples_ms.push(start.elapsed().as_secs_f64() * 1000.0);
        black_box(&serialized_output);
    }
    let json_pipeline_elapsed_ms: f64 = validation_samples_ms.iter().sum();
    let json_pipeline_digest = hex::encode(Sha256::digest(serialized_output.as_bytes()));
    assert_eq!(json_pipeline_digest, digest, "JSON pipeline output differs");

    println!(
        "{{\"schema\":\"kogwistar-graph-slice-native-benchmark/v1\",\"envelopes\":{},\"iterations\":{},\"payload_digest\":\"{}\",\"kernel_total_ms\":{:.6},\"kernel_per_iteration_ms\":{:.6},\"json_projection_total_ms\":{:.6},\"json_projection_per_iteration_ms\":{:.6},\"json_projection_median_ms\":{:.6},\"json_projection_p95_ms\":{:.6},\"json_pipeline_total_ms\":{:.6},\"json_pipeline_per_iteration_ms\":{:.6},\"json_pipeline_median_ms\":{:.6},\"json_pipeline_p95_ms\":{:.6}}}",
        envelopes.len(),
        iterations,
        digest,
        elapsed_ms,
        elapsed_ms / iterations as f64,
        json_projection_elapsed_ms,
        json_projection_elapsed_ms / iterations as f64,
        median_ms(&projection_samples_ms),
        p95_ms(&projection_samples_ms),
        json_pipeline_elapsed_ms,
        json_pipeline_elapsed_ms / iterations as f64,
        median_ms(&validation_samples_ms),
        p95_ms(&validation_samples_ms)
    );
}
