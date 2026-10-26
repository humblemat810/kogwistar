//! Chroma-backed graph/vector storage. Durable event and queue metadata remain
//! owned by the separately selected SQLite or PostgreSQL metadata store.

mod client;
mod records;
mod store;

pub use client::ChromaStoreConfig;
pub use store::ChromaGraphStore;
