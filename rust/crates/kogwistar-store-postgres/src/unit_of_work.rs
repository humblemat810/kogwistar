use super::*;

/// Operations sharing one RAII PostgreSQL transaction.
pub struct PostgresUnitOfWork<'transaction> {
    pub(super) transaction: Transaction<'transaction>,
    pub(super) tables: Tables,
    pub(super) graph_tables_by_schema: std::collections::HashMap<String, Tables>,
}

mod events_recovery;
mod graph;
mod index_jobs;
mod lane_messages;
mod projections;
mod recorded_runtime;
mod runs;
mod workflow_design;
