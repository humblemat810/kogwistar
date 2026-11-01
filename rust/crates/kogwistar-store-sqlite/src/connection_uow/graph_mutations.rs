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
        let canonical_embedding_entities = batch
            .mutations
            .iter()
            .map(|mutation| (mutation.entity_kind.clone(), mutation.record.id.clone()))
            .collect::<BTreeSet<_>>();
        let mut mutations = batch.mutations;
        for mutation in &mut mutations {
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
            persist_graph_scope(mutation)?;
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

        let mut applied = Vec::with_capacity(mutations.len());
        for mutation in mutations {
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
            let current = self.latest_entity_events(
                &mutation.scope.namespace,
                &mutation.entity_kind,
                std::slice::from_ref(&mutation.record.id),
            )?;
            let current_events = current
                .into_values()
                .map(raw_to_entity_event)
                .collect::<Result<Vec<_>, _>>()
                .map_err(SqliteStoreError::from)?;
            if let Some(revision) =
                kogwistar_engine::canonical_revision::read_canonical_entity_revision(
                    &current_events,
                    &mutation.scope.namespace,
                    &mutation.entity_kind,
                    &mutation.record.id,
                )?
            {
                let projection_key = format!("{}:{}", revision.entity_kind, revision.entity_id);
                if revision.is_deleted() {
                    self.clear_stage1_node_projection(&revision.namespace, &projection_key)?;
                } else if revision
                    .payload
                    .get("mentions")
                    .and_then(Value::as_array)
                    .is_none_or(Vec::is_empty)
                {
                    // Stage-1 is the source-grounded graph projection, not a
                    // fabricated fallback for ungrounded runtime artifacts.
                    self.clear_stage1_node_projection(&revision.namespace, &projection_key)?;
                } else {
                    let job_payload =
                        kogwistar_engine::canonical_revision::canonical_revision_job_payload(
                            &revision,
                        );
                    let fingerprint = serde_json::from_str::<serde_json::Value>(&job_payload)
                        .ok()
                        .and_then(|payload| {
                            payload
                                .get("source_fingerprint")
                                .and_then(serde_json::Value::as_str)
                                .map(str::to_owned)
                        })
                        .ok_or_else(|| {
                            SqliteStoreError::TransactionAborted(
                                "canonical Stage-1 fingerprint could not be derived".to_owned(),
                            )
                        })?;
                    let projection = crate::stage1_projection_from_canonical_event(
                        &revision.namespace,
                        &revision.entity_kind,
                        &revision.entity_id,
                        revision.revision,
                        &revision.payload,
                        &fingerprint,
                    )
                    .map_err(SqliteStoreError::from)?;
                    self.replace_stage1_node_projection(
                        &revision.namespace,
                        &projection_key,
                        projection,
                    )?;
                }
            }
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
                    // Python's indexing contract uses one embedding job kind
                    // for both entity types; entity_kind selects the collection.
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
            // Canonical mutations already enqueue a revision-fenced embedding
            // job. A duplicate without that revision payload must not replace it.
            if job.index_kind == "node_embedding"
                && canonical_embedding_entities
                    .contains(&(job.entity_kind.clone(), job.entity_id.clone()))
            {
                continue;
            }
            self.enqueue_index_job(job)?;
        }
        Ok(applied)
    }
}

fn persist_graph_scope(mutation: &mut GraphMutation) -> SqliteStoreResult<()> {
    let metadata = mutation
        .payload
        .as_object_mut()
        .and_then(|payload| {
            payload
                .entry("metadata".to_owned())
                .or_insert_with(|| Value::Object(Default::default()))
                .as_object_mut()
        })
        .ok_or_else(|| {
            SqliteStoreError::TransactionAborted(
                "canonical graph payload metadata must be an object".to_owned(),
            )
        })?;
    mutation
        .scope
        .validate_metadata_scope(metadata)
        .map_err(|error| SqliteStoreError::TransactionAborted(error.to_string()))?;
    mutation
        .scope
        .validate_metadata_scope(&mutation.record.metadata)
        .map_err(|error| SqliteStoreError::TransactionAborted(error.to_string()))?;
    for (key, expected) in [
        ("namespace", Some(mutation.scope.namespace.as_str())),
        ("workspace_id", mutation.scope.workspace_id.as_deref()),
        ("graph_space", mutation.scope.graph_space.as_deref()),
    ] {
        match expected {
            Some(expected) => {
                if metadata
                    .get(key)
                    .is_some_and(|actual| actual.as_str() != Some(expected))
                {
                    return Err(SqliteStoreError::TransactionAborted(format!(
                        "canonical graph metadata {key:?} conflicts with its event scope"
                    )));
                }
                metadata.insert(key.to_owned(), Value::String(expected.to_owned()));
                mutation
                    .record
                    .metadata
                    .insert(key.to_owned(), Value::String(expected.to_owned()));
            }
            None if metadata.get(key).is_some_and(|value| !value.is_null()) => {
                return Err(SqliteStoreError::TransactionAborted(format!(
                    "canonical graph metadata {key:?} has a scope absent from its event scope"
                )));
            }
            None => {}
        }
    }
    Ok(())
}
