use super::*;

impl PostgresUnitOfWork<'_> {
    pub async fn lock_entity_identity(
        &self,
        namespace: &str,
        entity_kind: &str,
        entity_id: &str,
    ) -> PostgresStoreResult<()> {
        let key = format!("{namespace}\u{1f}{entity_kind}\u{1f}{entity_id}");
        self.transaction
            .query_one(
                "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
                &[&key],
            )
            .await
            .map_err(backend)?;
        Ok(())
    }

    pub async fn lock_runtime_queue_admission(&self) -> PostgresStoreResult<()> {
        advisory_lock(
            &self.transaction,
            &format!(
                "runtime-queue-admission\u{1f}{}",
                self.tables.projected_lane_messages
            ),
            73,
        )
        .await
    }

    pub async fn apply_graph_mutation(
        &mut self,
        mutation: GraphMutation,
    ) -> PostgresStoreResult<AppliedGraphMutation> {
        validate_graph_mutation(&mutation)?;
        apply_graph_mutation(&self.transaction, &self.tables, mutation).await
    }

    pub async fn apply_graph_metadata_patch_mutation(
        &mut self,
        mutation: GraphMetadataPatchMutation,
    ) -> PostgresStoreResult<Option<AppliedGraphMutation>> {
        apply_graph_metadata_patch_mutation(&self.transaction, &self.tables, mutation).await
    }

    pub async fn apply_graph_delete_mutation(
        &mut self,
        mutation: GraphDeleteMutation,
    ) -> PostgresStoreResult<Option<AppliedGraphMutation>> {
        apply_graph_delete_mutation(&self.transaction, &self.tables, mutation).await
    }

    pub async fn upsert_graph_projection(
        &mut self,
        write: GraphProjectionUpsert,
    ) -> PostgresStoreResult<()> {
        upsert_graph_projection(&self.transaction, &self.tables, write).await
    }

    pub async fn promote_graph_embedding_projection(
        &mut self,
        namespace: &str,
        entity_kind: &str,
        mut candidate: GraphRecord,
    ) -> PostgresStoreResult<bool> {
        require_namespace(namespace)?;
        let table = match entity_kind {
            "node" => "gke_nodes",
            "edge" => "gke_edges",
            _ => {
                return Err(StoreError::InvalidEntityEventPayload {
                    message: "embedding promotion supports node or edge only".to_owned(),
                }
                .into());
            }
        };
        let embedding_dim = candidate
            .embedding
            .as_ref()
            .map(Vec::len)
            .filter(|dimension| *dimension > 0)
            .ok_or_else(|| {
                PostgresStoreError::Backend(
                    "embedding promotion requires a non-empty vector".to_owned(),
                )
            })?;
        let scope = GraphScope {
            namespace: namespace.to_owned(),
            workspace_id: None,
            graph_space: None,
        };
        let stage1_marker_names = [
            "_kogwistar_stage1_source_fingerprint",
            "_kogwistar_stage1_expected_fingerprint",
            "_kogwistar_stage1_revision",
            "_kogwistar_stage1_expected_revision",
        ];
        let has_any_stage1_marker = stage1_marker_names
            .iter()
            .any(|name| candidate.metadata.contains_key(*name));
        let staged_fingerprint = candidate
            .metadata
            .remove("_kogwistar_stage1_source_fingerprint")
            .and_then(|value| value.as_str().map(str::to_owned));
        let expected_fingerprint = candidate
            .metadata
            .remove("_kogwistar_stage1_expected_fingerprint")
            .and_then(|value| value.as_str().map(str::to_owned));
        let staged_revision = candidate
            .metadata
            .remove("_kogwistar_stage1_revision")
            .and_then(|value| value.as_i64());
        let expected_revision = candidate
            .metadata
            .remove("_kogwistar_stage1_expected_revision")
            .and_then(|value| value.as_i64());
        let has_stage1_contract = staged_fingerprint.is_some()
            && expected_fingerprint.is_some()
            && staged_revision.is_some()
            && expected_revision.is_some();
        if has_any_stage1_marker && !has_stage1_contract {
            return Err(StoreError::InvalidEntityEventPayload {
                message: "embedding promotion has an incomplete Stage-1 fencing contract"
                    .to_owned(),
            }
            .into());
        }
        if has_stage1_contract {
            let source_fingerprint = staged_fingerprint
                .as_deref()
                .expect("Stage-1 contract checked above");
            let mut final_metadata = graph_metadata(&scope, &candidate.metadata)?;
            final_metadata.insert("_kogwistar_stage2_ready".to_owned(), Value::Bool(true));
            final_metadata.insert(
                "_kogwistar_source_fingerprint".to_owned(),
                Value::String(source_fingerprint.to_owned()),
            );
            candidate.metadata = final_metadata;
        }
        // Match Python's Stage-1 writer lock order: invalidate/read the
        // canonical graph projection before taking the Stage-1 row lock.
        let current = self
            .graph_projection_records_for_update(GraphProjectionRead {
                scope: scope.clone(),
                table: table.to_owned(),
                ids: Some(vec![candidate.id.clone()]),
                metadata: Default::default(),
                limit: 1,
            })
            .await?
            .into_iter()
            .next();
        let stage1_source = if has_stage1_contract {
            self.transaction
                .query_opt(
                    &format!(
                        "SELECT document, metadata::TEXT, source_fingerprint, revision, \
                                materialization_status \
                         FROM {} WHERE namespace=$1 AND entity_kind=$2 AND entity_id=$3 \
                         FOR UPDATE",
                        self.tables.stage1_projections
                    ),
                    &[&namespace, &entity_kind, &candidate.id],
                )
                .await
                .map_err(backend)?
                .map(|row| {
                    let document: String = row.try_get(0).map_err(backend)?;
                    let metadata_json: String = row.try_get(1).map_err(backend)?;
                    let metadata: Map<String, Value> = serde_json::from_str(&metadata_json)?;
                    let source_fingerprint: String = row.try_get(2).map_err(backend)?;
                    let revision: i64 = row.try_get(3).map_err(backend)?;
                    let materialization_status: String = row.try_get(4).map_err(backend)?;
                    Ok::<_, PostgresStoreError>((
                        document,
                        metadata,
                        source_fingerprint,
                        revision,
                        materialization_status,
                    ))
                })
                .transpose()?
        } else {
            None
        };
        if let Some((document, metadata, source_fingerprint, revision, status)) = &stage1_source {
            let mut expected_source_metadata = candidate.metadata.clone();
            expected_source_metadata.remove("_kogwistar_stage2_ready");
            expected_source_metadata.remove("_kogwistar_source_fingerprint");
            if source_fingerprint != staged_fingerprint.as_deref().unwrap_or_default()
                || source_fingerprint != expected_fingerprint.as_deref().unwrap_or_default()
                || *revision != staged_revision.unwrap_or_default()
                || *revision != expected_revision.unwrap_or_default()
                || status != "pending"
                || Some(document.as_str()) != candidate.document.as_deref()
                || graph_metadata(&scope, metadata)?
                    != graph_metadata(&scope, &expected_source_metadata)?
            {
                return Ok(false);
            }
        }
        let mut current = if let Some(current) = current {
            current
        } else if let Some((document, _metadata, _, _, _)) = &stage1_source {
            GraphRecord {
                id: candidate.id.clone(),
                document: Some(document.clone()),
                metadata: graph_metadata(&scope, &candidate.metadata)?,
                embedding: None,
            }
        } else {
            return Ok(false);
        };
        if current
            .metadata
            .get("lifecycle_status")
            .and_then(Value::as_str)
            .is_some_and(|status| matches!(status, "tombstoned" | "deleted"))
        {
            return Ok(false);
        }
        let expected_metadata = graph_metadata(&scope, &candidate.metadata)?;
        if current.document != candidate.document || current.metadata != expected_metadata {
            return Ok(false);
        }

        // The provider supplies only the vector. The row lock and Stage-1
        // comparison above fence delayed results against newer staged writes.
        current.embedding = candidate.embedding.take();
        self.upsert_graph_projection(GraphProjectionUpsert {
            scope: scope.clone(),
            table: table.to_owned(),
            record: current,
            embedding_dim,
        })
        .await?;
        if has_stage1_contract {
            let deleted = self
                .transaction
                .execute(
                    &format!(
                        "DELETE FROM {} WHERE namespace=$1 AND entity_kind=$2 \
                         AND entity_id=$3 AND source_fingerprint=$4 AND revision=$5",
                        self.tables.stage1_projections
                    ),
                    &[
                        &namespace,
                        &entity_kind,
                        &candidate.id,
                        &staged_fingerprint.as_deref().unwrap_or_default(),
                        &staged_revision.unwrap_or_default(),
                    ],
                )
                .await
                .map_err(backend)?;
            if stage1_source.is_some() && deleted != 1 {
                return Err(PostgresStoreError::Backend(
                    "locked Stage-1 source changed before promotion commit".to_owned(),
                ));
            }
        }
        Ok(true)
    }

    pub async fn patch_graph_projection_metadata(
        &mut self,
        patch: GraphProjectionMetadataPatch,
    ) -> PostgresStoreResult<bool> {
        patch_graph_projection_metadata(&self.transaction, &self.tables, patch).await
    }

    pub async fn graph_projection_records(
        &self,
        read: GraphProjectionRead,
    ) -> PostgresStoreResult<Vec<GraphRecord>> {
        graph_projection_records(&self.transaction, &self.tables, read, None, false).await
    }

    pub async fn graph_projection_records_for_update(
        &self,
        read: GraphProjectionRead,
    ) -> PostgresStoreResult<Vec<GraphRecord>> {
        graph_projection_records_with_lock(&self.transaction, &self.tables, read, None, false, true)
            .await
    }

    /// Read ACL truth through this unit of work's transaction. Callers that
    /// derive graph effects from ACL state must not use the store-level read
    /// helper, which may observe a different committed snapshot.
    pub async fn graph_acl_records_for_entities(
        &self,
        scope: &GraphScope,
        truth_graph: &str,
        entity_ids: &[String],
    ) -> PostgresStoreResult<Vec<GraphRecord>> {
        graph_acl_records_for_entities(
            &self.transaction,
            &self.tables,
            scope,
            truth_graph,
            entity_ids,
        )
        .await
    }

    pub async fn graph_projection_vector_query(
        &self,
        query: GraphProjectionVectorQuery,
    ) -> PostgresStoreResult<Vec<VectorMatch>> {
        graph_projection_vector_query(&self.transaction, &self.tables, query, &BTreeMap::new())
            .await
    }
}
