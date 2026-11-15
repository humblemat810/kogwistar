# Kogwistar Chroma Graph Store

This crate implements Kogwistar's `GraphReadStore` and `GraphWriteStore` over
the Chroma v2 HTTP API. It stores vectors and graph records in a Chroma
collection; durable event history, workflow/runtime state, indexing jobs, and
lane-message queues remain in the separately selected SQLite or PostgreSQL
metadata store.

```rust,no_run
use kogwistar_store::DistanceMetric;
use kogwistar_store_chroma::{ChromaGraphStore, ChromaStoreConfig};

# async fn example() -> Result<(), Box<dyn std::error::Error>> {
let graph = ChromaGraphStore::connect(ChromaStoreConfig {
    endpoint: "http://127.0.0.1:8000".to_owned(),
    collection: "knowledge-v1".to_owned(),
    metric: DistanceMetric::Cosine,
    ..Default::default()
}).await?;
# let _ = graph;
# Ok(())
# }
```

This is a server-backed Rust adapter, not an opener for Python Chroma's
embedded `persist_directory`. It uses Chroma's v2 HTTP endpoints and expects
the server and collection to be provisioned/reachable separately. The
collection metric is fixed at creation; use a distinct collection for a
different metric. The metadata filter and namespace are enforced in Chroma's
query predicate, and Kogwistar metadata stays canonical in a reserved JSON
metadata field so nested metadata survives round-trips.
