//! Fixed-input PostgreSQL projection-worker attribution benchmark.
//!
//! Run with `KOGWISTAR_TEST_POSTGRES_DSN` set. Each warmup/sample gets a fresh
//! schema; provider vectors are precomputed so no embedding inference is timed.

use kogwistar_engine::projection_worker::{
    ProjectionWorkerAttribution, ProjectionWorkerSummary,
    run_graph_projection_jobs_once_with_batch_provider_input_attributed,
};
use kogwistar_store::{
    GraphProjectionRead, GraphRecord, GraphScope, IndexJobReadStore, NewIndexJob,
};
use kogwistar_store_postgres::{GraphProjectionUpsert, GraphTableNames, PostgresStore};
use serde::Serialize;
use serde_json::json;
use std::{env, time::Instant};
use tokio_postgres::NoTls;
use uuid::Uuid;

const WORKER_BATCH_SIZE: usize = 100;

#[derive(Clone, Debug, Serialize)]
struct Sample {
    elapsed_ms: f64,
    provider_ms: f64,
    graph_projection_ms: f64,
    worker_transfer_ms: f64,
    worker_batch_size: usize,
    jobs: usize,
    records: usize,
}

async fn cleanup_schema(dsn: &str, schema: &str) -> Result<(), Box<dyn std::error::Error>> {
    let (client, connection) = tokio_postgres::connect(dsn, NoTls).await?;
    tokio::spawn(async move {
        let _ = connection.await;
    });
    client
        .batch_execute(&format!("DROP SCHEMA IF EXISTS \"{schema}\" CASCADE"))
        .await?;
    Ok(())
}

async fn run_sample(
    dsn: &str,
    schema: &str,
    items: usize,
) -> Result<Sample, Box<dyn std::error::Error>> {
    let store = PostgresStore::from_dsn(dsn, schema)?;
    store.ensure_schema().await?;
    store
        .create_graph_schema(384, GraphTableNames::default())
        .await?;

    for index in 0..items {
        let mut record = GraphRecord::new(format!("node-{index:06}"));
        record.document = Some(format!("benchmark node {}", record.id));
        record
            .metadata
            .insert("entity_type".to_owned(), json!("node"));
        store
            .upsert_graph_projection(GraphProjectionUpsert {
                scope: GraphScope {
                    namespace: "benchmark".to_owned(),
                    workspace_id: None,
                    graph_space: None,
                },
                table: "gke_nodes".to_owned(),
                record,
                embedding_dim: 384,
            })
            .await?;
    }

    for index in 0..items {
        store
            .enqueue_index_job(NewIndexJob {
                job_id: format!("projection-{index:06}"),
                namespace: "benchmark".to_owned(),
                entity_kind: "node".to_owned(),
                entity_id: format!("node-{index:06}"),
                index_kind: "node_embedding".to_owned(),
                op: "UPSERT".to_owned(),
                payload_json: Some(json!({"ordinal": index}).to_string()),
                max_retries: 3,
            })
            .await?;
    }

    let precomputed_embedding = {
        let mut embedding = vec![0.0_f32; 384];
        embedding[0] = 1.0;
        embedding
    };
    let started = Instant::now();
    let mut summary = ProjectionWorkerSummary::default();
    let mut attribution = ProjectionWorkerAttribution::default();
    while summary.claimed < items {
        let batch_limit = (items - summary.claimed).min(WORKER_BATCH_SIZE);
        let batch_embedding = precomputed_embedding.clone();
        let (batch_summary, batch_attribution) =
            run_graph_projection_jobs_once_with_batch_provider_input_attributed(
                &store,
                &store,
                batch_limit,
                60,
                Some("benchmark"),
                move |inputs| {
                    let embedding = batch_embedding.clone();
                    async move {
                        Ok(inputs
                            .into_iter()
                            .map(|input| {
                                let mut record = GraphRecord::new(input.entity_id.clone());
                                record.document =
                                    Some(format!("benchmark node {}", input.entity_id));
                                record
                                    .metadata
                                    .insert("entity_type".to_owned(), json!("node"));
                                record.embedding = Some(embedding.clone());
                                (input.job_id, Ok(record))
                            })
                            .collect())
                    }
                },
            )
            .await?;
        if batch_summary.claimed != batch_limit
            || batch_summary.projected + batch_summary.reused != batch_limit
            || batch_summary.claim_lost != 0
            || batch_summary.retried != 0
            || batch_summary.failed != 0
            || batch_attribution
                .provider_and_projection
                .completed_provider_elapsed_ns
                == 0
            || batch_attribution
                .provider_and_projection
                .graph_projection_elapsed_ns
                == 0
            || batch_attribution.worker_transfer_elapsed_ns == 0
        {
            return Err(format!(
                "worker batch mismatch: requested={batch_limit}, claimed={}, projected={}, reused={}, claim_lost={}, retried={}, failed={}",
                batch_summary.claimed,
                batch_summary.projected,
                batch_summary.reused,
                batch_summary.claim_lost,
                batch_summary.retried,
                batch_summary.failed
            )
            .into());
        }
        summary.claimed += batch_summary.claimed;
        summary.projected += batch_summary.projected;
        summary.reused += batch_summary.reused;
        summary.claim_lost += batch_summary.claim_lost;
        summary.retried += batch_summary.retried;
        summary.failed += batch_summary.failed;
        attribution
            .provider_and_projection
            .completed_provider_elapsed_ns += batch_attribution
            .provider_and_projection
            .completed_provider_elapsed_ns;
        attribution
            .provider_and_projection
            .graph_projection_elapsed_ns += batch_attribution
            .provider_and_projection
            .graph_projection_elapsed_ns;
        attribution.worker_transfer_elapsed_ns += batch_attribution.worker_transfer_elapsed_ns;
    }
    let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;

    if summary.claimed != items || summary.projected + summary.reused != items {
        let all_jobs = store
            .index_jobs(Some("benchmark"), None, None, None, None, items + 1)
            .await?;
        let failure = all_jobs
            .iter()
            .find(|job| job.last_error.is_some())
            .map(|job| {
                format!(
                    "{}: {}",
                    job.status,
                    job.last_error.as_deref().unwrap_or_default()
                )
            })
            .unwrap_or_else(|| "no job error recorded".to_owned());
        return Err(format!(
            "worker result mismatch: claimed={}/{items}, projected={}, reused={}, claim_lost={}, retried={}, failed={}; {failure}",
            summary.claimed,
            summary.projected,
            summary.reused,
            summary.claim_lost,
            summary.retried,
            summary.failed
        )
        .into());
    }
    let jobs = store
        .index_jobs(Some("benchmark"), Some("DONE"), None, None, None, items + 1)
        .await?;
    let graph_scope = kogwistar_store::GraphScope {
        namespace: "benchmark".to_owned(),
        workspace_id: None,
        graph_space: None,
    };
    let records = store
        .graph_projection_records(GraphProjectionRead {
            scope: graph_scope.clone(),
            table: "gke_nodes".to_owned(),
            ids: None,
            metadata: Default::default(),
            limit: items + 1,
        })
        .await?;
    if jobs.len() != items || records.len() != items {
        return Err(format!(
            "durability mismatch: DONE jobs={}, projected records={}, expected={items}",
            jobs.len(),
            records.len()
        )
        .into());
    }
    if items > 0
        && store
            .graph_projection_records(GraphProjectionRead {
                scope: graph_scope,
                table: "gke_nodes".to_owned(),
                ids: Some(vec!["node-000000".to_owned()]),
                metadata: Default::default(),
                limit: 1,
            })
            .await?
            .into_iter()
            .next()
            .and_then(|record| record.embedding)
            .is_none()
    {
        return Err("projected record missing embedding".into());
    }

    Ok(Sample {
        elapsed_ms,
        provider_ms: attribution
            .provider_and_projection
            .completed_provider_elapsed_ns as f64
            / 1_000_000.0,
        graph_projection_ms: attribution
            .provider_and_projection
            .graph_projection_elapsed_ns as f64
            / 1_000_000.0,
        worker_transfer_ms: attribution.worker_transfer_elapsed_ns as f64 / 1_000_000.0,
        worker_batch_size: WORKER_BATCH_SIZE,
        jobs: jobs.len(),
        records: records.len(),
    })
}

fn median(values: &[f64]) -> f64 {
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    }
}

#[cfg(test)]
mod tests {
    use super::median;

    #[test]
    fn median_selects_middle_sample_for_odd_count() {
        assert_eq!(median(&[9.0, 1.0, 5.0]), 5.0);
    }

    #[test]
    fn median_averages_middle_samples_for_even_count() {
        assert_eq!(median(&[10.0, 2.0]), 6.0);
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dsn = env::var("KOGWISTAR_TEST_POSTGRES_DSN")?;
    let mut args = env::args().skip(1);
    let items = args.next().as_deref().unwrap_or("100").parse::<usize>()?;
    let sample_count = args.next().as_deref().unwrap_or("3").parse::<usize>()?;
    if args.next().is_some() || items == 0 || sample_count == 0 {
        return Err("usage: projection_worker_attribution [items>0] [samples>0]".into());
    }

    let mut measured = Vec::with_capacity(sample_count);
    let mut warmup = None;
    for ordinal in 0..=sample_count {
        let schema = format!("kg_projection_bench_{}", Uuid::new_v4().simple());
        let result = run_sample(&dsn, &schema, items).await;
        let cleanup = cleanup_schema(&dsn, &schema).await;
        let sample = result?;
        cleanup?;
        if ordinal == 0 {
            warmup = Some(sample);
        } else {
            measured.push(sample);
        }
    }

    let elapsed = measured
        .iter()
        .map(|sample| sample.elapsed_ms)
        .collect::<Vec<_>>();
    let provider = measured
        .iter()
        .map(|sample| sample.provider_ms)
        .collect::<Vec<_>>();
    let projection = measured
        .iter()
        .map(|sample| sample.graph_projection_ms)
        .collect::<Vec<_>>();
    let worker_transfer = measured
        .iter()
        .map(|sample| sample.worker_transfer_ms)
        .collect::<Vec<_>>();
    println!(
        "{}",
        json!({
            "contract": "kogwistar-projection-worker-attribution/v1",
            "backend": "postgresql",
            "items": items,
            "embedding": {"dimension": 384, "inference": "precomputed_not_timed"},
            "warmup": warmup,
            "samples": measured,
            "median_elapsed_ms": median(&elapsed),
            "median_provider_ms": median(&provider),
            "median_graph_projection_ms": median(&projection),
            "median_worker_transfer_ms": median(&worker_transfer),
            "median_throughput_jobs_per_sec": items as f64 * 1_000.0 / median(&elapsed),
            "scope": "background_worker_only_not_mcp_request_latency"
        })
    );
    Ok(())
}
