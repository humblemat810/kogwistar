//! Compile-time composition contracts for independently selected stores.

use kogwistar_store::{GraphWriteStore, IndexJobWriteStore};
use kogwistar_store_chroma::ChromaGraphStore;
use kogwistar_store_postgres::PostgresStore;

#[test]
fn postgres_durable_queue_can_pair_with_chroma_graph_projection() {
    fn composition_bounds<Queue, Graph>()
    where
        Queue: IndexJobWriteStore,
        Graph: GraphWriteStore,
    {
    }

    // The projection worker depends on separate queue and graph capabilities;
    // it does not require both stores to share a transaction implementation.
    composition_bounds::<PostgresStore, ChromaGraphStore>();
}
