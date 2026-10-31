use super::*;

impl SqliteUnitOfWork<'_> {
    /// Append canonical node/edge mutations and durable projection work in the
    /// caller's transaction. Chroma is updated later by the revision-fenced
    /// projection worker.
    pub fn apply_canonical_graph_mutation_batch(
        &mut self,
        batch: GraphMutationBatch,
    ) -> SqliteStoreResult<Vec<AppliedGraphMutation>> {
        let Some(first) = batch.mutations.first() else {
            return Err(SqliteStoreError::TransactionAborted(
                "graph mutation batches must not be empty".to_owned(),
            ));
        };
        let scope = first.scope.clone();
        for mutation in &batch.mutations {
            validate_graph_mutation(mutation)?;
            if mutation.scope != scope {
                return Err(SqliteStoreError::TransactionAborted(
                    "atomic graph mutations must share one graph scope".to_owned(),
                ));
            }
            if !matches!(mutation.entity_kind.as_str(), "node" | "edge") {
                return Err(SqliteStoreError::TransactionAborted(
                    "SQLite canonical graph projection supports node and edge entities only"
                        .to_owned(),
                ));
            }
        }
        for job in &batch.index_jobs {
            if job.namespace != scope.namespace
                || job.job_id.is_empty()
                || job.entity_id.is_empty()
                || job.entity_kind.is_empty()
                || job.index_kind.is_empty()
                || !matches!(job.op.as_str(), "UPSERT" | "DELETE")
                || job.max_retries < 0
            {
                return Err(SqliteStoreError::TransactionAborted(
                    "index job intent is invalid or outside the graph namespace".to_owned(),
                ));
            }
        }

        let mut applied = Vec::with_capacity(batch.mutations.len());
        for mutation in batch.mutations {
            let payload_json = serde_json::to_string(&mutation.payload)
                .expect("serde_json::Value serialization cannot fail");
            let appended = self.append_raw_entity_event(
                &mutation.scope.namespace,
                NewRawEntityEvent {
                    event_id: mutation.event_id.clone(),
                    entity_kind: mutation.entity_kind.clone(),
                    entity_id: mutation.record.id.clone(),
                    op: mutation.op.clone(),
                    payload_json,
                },
            )?;
            let event = raw_to_entity_event(appended.event).map_err(SqliteStoreError::from)?;
            if !appended.inserted {
                validate_graph_mutation_replay(&mutation, &event)?;
            } else {
                let state = match mutation.op.as_str() {
                    "TOMBSTONE" => "deleted",
                    "ADD" | "REPLACE"
                        if mutation
                            .payload
                            .get("lifecycle_patch")
                            .and_then(Value::as_object)
                            .and_then(|patch| patch.get("lifecycle_status"))
                            .and_then(Value::as_str)
                            == Some("tombstoned") =>
                    {
                        "deleted"
                    }
                    _ => "active",
                };
                let revision = kogwistar_engine::canonical_revision::CanonicalEntityRevision {
                    namespace: mutation.scope.namespace.clone(),
                    entity_kind: mutation.entity_kind.clone(),
                    entity_id: mutation.record.id.clone(),
                    revision: event.seq,
                    state: state.to_owned(),
                    payload: event.payload.clone(),
                };
                let projection_job = NewIndexJob {
                    job_id: format!("canonical-projection-{}", mutation.event_id),
                    namespace: mutation.scope.namespace.clone(),
                    entity_kind: mutation.entity_kind.clone(),
                    entity_id: mutation.record.id.clone(),
                    index_kind: "node_embedding".to_owned(),
                    op: if revision.is_deleted() {
                        "DELETE".to_owned()
                    } else {
                        "UPSERT".to_owned()
                    },
                    payload_json: Some(
                        kogwistar_engine::canonical_revision::canonical_revision_job_payload(
                            &revision,
                        ),
                    ),
                    max_retries: 10,
                };
                self.enqueue_index_job(projection_job)?;
            }
            applied.push(AppliedGraphMutation {
                event,
                inserted: appended.inserted,
                mutated: appended.inserted,
            });
        }
        for job in batch.index_jobs {
            self.enqueue_index_job(job)?;
        }
        Ok(applied)
    }
}
