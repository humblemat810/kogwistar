// PostgreSQL store and unit-of-work operation adapters. Python parity: kogwistar.engine_core.engine_postgres.
async fn postgres_store_operation_json(
    store: &PostgresStore,
    operation: PostgresStoreOperation,
) -> Result<Value, PostgresStoreError> {
    match operation {
        PostgresStoreOperation::CloseStore
        | PostgresStoreOperation::BeginTransaction
        | PostgresStoreOperation::CommitTransaction
        | PostgresStoreOperation::RollbackTransaction => Err(postgres_session_error(
            "PostgreSQL transaction control cannot be nested as an operation",
        )),
        PostgresStoreOperation::LockEntityIdentity { .. } => Err(postgres_session_error(
            "entity identity locks require an active PostgreSQL transaction",
        )),
        PostgresStoreOperation::EnsureSchema => {
            store.ensure_schema().await?;
            Ok(json!({"initialized": true}))
        }
        PostgresStoreOperation::GraphAuthorityCapability => {
            let capability = store.graph_authority_capability();
            Ok(json!({
                "level": capability.as_str(),
                "single_mutation": capability.supports_single_mutation(),
                "atomic_index_jobs": capability.supports_atomic_index_jobs(),
            }))
        }
        PostgresStoreOperation::NextGlobalSeq => Ok(json!(store.next_global_seq().await?)),
        PostgresStoreOperation::CurrentGlobalSeq => Ok(json!(store.current_global_seq().await?)),
        PostgresStoreOperation::NextUserSeq { user_id } => {
            Ok(json!(store.next_user_seq(&user_id).await?))
        }
        PostgresStoreOperation::CurrentUserSeq { user_id } => {
            Ok(json!(store.current_user_seq(&user_id).await?))
        }
        PostgresStoreOperation::SetUserSeq { user_id, value } => {
            store.set_user_seq(&user_id, value).await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::GetIndexAppliedFingerprint {
            namespace,
            coalesce_key,
        } => Ok(json!(
            store
                .get_index_applied_fingerprint(&namespace, &coalesce_key)
                .await?
        )),
        PostgresStoreOperation::SetIndexAppliedFingerprint {
            namespace,
            coalesce_key,
            applied_fingerprint,
            last_job_id,
        } => {
            store
                .set_index_applied_fingerprint(
                    &namespace,
                    &coalesce_key,
                    applied_fingerprint,
                    last_job_id,
                )
                .await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::CreateGraphSchema {
            embedding_dim,
            nodes_table,
            edges_table,
            documents_table,
            domains_table,
            node_index_table,
        } => {
            let names = graph_table_names(nodes_table, edges_table, documents_table, domains_table);
            if let Some(node_index_table) = node_index_table {
                store
                    .create_graph_schema_with_node_index(embedding_dim, names, &node_index_table)
                    .await?;
            } else {
                store.create_graph_schema(embedding_dim, names).await?;
            }
            Ok(json!({"initialized": true}))
        }
        PostgresStoreOperation::NodeIndexUpsert { table, records } => {
            store
                .upsert_node_index(
                    &table,
                    records.into_iter().map(graph_mutation_record).collect(),
                )
                .await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::NodeIndexGet { table, ids } => Ok(json!(store
            .get_node_index(&table, &ids)
            .await?
            .into_iter()
            .map(graph_record_json)
            .collect::<Vec<_>>())),
        PostgresStoreOperation::NodeIndexQuery {
            table,
            embedding,
            limit,
            metadata,
            metric,
        } => Ok(json!(store
            .query_node_index(
                &table,
                &VectorQuery {
                    embedding,
                    limit,
                    metadata: metadata_filter(metadata),
                    metric: metric.into(),
                },
            )
            .await?
            .into_iter()
            .map(|item| json!({"record": graph_record_json(item.record), "distance": item.distance}))
            .collect::<Vec<_>>())),
        PostgresStoreOperation::AllocEventSeq { namespace } => {
            Ok(json!(store.alloc_event_seq(&namespace).await?))
        }
        PostgresStoreOperation::RawAppend {
            namespace,
            event_id,
            entity_kind,
            entity_id,
            op,
            payload_json,
        } => Ok(postgres_appended_raw_event_json(
            store
                .append_raw_entity_event(
                    &namespace,
                    postgres_new_raw_event(event_id, entity_kind, entity_id, op, payload_json),
                )
                .await?,
        )),
        PostgresStoreOperation::GraphMutation {
            namespace,
            workspace_id,
            graph_space,
            table,
            entity_kind,
            event_id,
            op,
            record,
            payload,
            embedding_dim,
            model_envelope,
        } => {
            let mutation = kogwistar_store::GraphMutation {
                scope: graph_scope(namespace, workspace_id, graph_space),
                table,
                entity_kind,
                event_id,
                op,
                payload,
                record: graph_mutation_record(record),
                embedding_dim,
            };
            let applied = match model_envelope {
                Some(envelope) => store.apply_typed_graph_mutation(mutation, envelope).await?,
                None => store.apply_graph_mutation(mutation).await?,
            };
            Ok(graph_mutation_json(applied))
        }
        PostgresStoreOperation::GraphMetadataPatchMutation {
            namespace,
            workspace_id,
            graph_space,
            table,
            entity_kind,
            event_id,
            op,
            entity_id,
            document,
            metadata_patch,
            payload,
        } => Ok(optional_graph_mutation_json(
            store
                .apply_graph_metadata_patch_mutation(
                    kogwistar_store_postgres::GraphMetadataPatchMutation {
                        scope: graph_scope(namespace, workspace_id, graph_space),
                        table,
                        entity_kind,
                        event_id,
                        op,
                        entity_id,
                        document,
                        metadata_patch,
                        payload,
                    },
                )
                .await?,
        )),
        PostgresStoreOperation::GraphDeleteMutation {
            namespace,
            workspace_id,
            graph_space,
            table,
            entity_kind,
            event_id,
            entity_id,
            payload,
        } => Ok(optional_graph_mutation_json(
            store
                .apply_graph_delete_mutation(kogwistar_store_postgres::GraphDeleteMutation {
                    scope: graph_scope(namespace, workspace_id, graph_space),
                    table,
                    entity_kind,
                    event_id,
                    entity_id,
                    payload,
                })
                .await?,
        )),
        PostgresStoreOperation::UpsertGraphProjection {
            namespace,
            workspace_id,
            graph_space,
            table,
            record,
            embedding_dim,
        } => {
            store
                .upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                    scope: graph_scope(namespace, workspace_id, graph_space),
                    table,
                    record: graph_mutation_record(record),
                    embedding_dim,
                })
                .await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::PatchGraphProjectionMetadata {
            namespace,
            workspace_id,
            graph_space,
            table,
            entity_id,
            document,
            metadata_patch,
            patch_document_metadata,
        } => Ok(json!(
            store
                .patch_graph_projection_metadata(
                    kogwistar_store_postgres::GraphProjectionMetadataPatch {
                        scope: graph_scope(namespace, workspace_id, graph_space),
                        table,
                        entity_id,
                        document,
                        metadata_patch,
                        patch_document_metadata,
                    },
                )
                .await?
        )),
        PostgresStoreOperation::GraphProjectionRecords {
            namespace,
            workspace_id,
            graph_space,
            table,
            ids,
            metadata,
            after_id,
            limit,
            for_update,
        } => {
            let read = GraphProjectionRead {
                scope: graph_scope(namespace, workspace_id, graph_space),
                table,
                ids,
                metadata: metadata_filter(metadata),
                limit,
            };
            let records = match (after_id, for_update) {
                (Some(after_id), false) => {
                    store
                        .graph_projection_page(GraphProjectionPageRead {
                            read,
                            after_id: Some(after_id),
                        })
                        .await?
                }
                (None, false) => store.graph_projection_records(read).await?,
                (None, true) => store.graph_projection_records_for_update(read).await?,
                (Some(_), true) => {
                    return Err(PostgresStoreError::Backend(
                        "row-locking graph projection reads cannot be paginated".to_owned(),
                    ));
                }
            };
            Ok(Value::Array(
                records.into_iter().map(graph_record_json).collect(),
            ))
        }
        PostgresStoreOperation::GraphProjectionVectorQuery {
            namespace,
            workspace_id,
            graph_space,
            table,
            embedding,
            limit,
            metadata,
            metadata_not_equals,
            metric,
            embedding_dim,
        } => {
            let query = kogwistar_store::GraphProjectionVectorQuery {
                scope: graph_scope(namespace, workspace_id, graph_space),
                table,
                query: VectorQuery {
                    embedding,
                    limit,
                    metadata: metadata_filter(metadata),
                    metric: metric.into(),
                },
                embedding_dim,
            };
            let matches = if metadata_not_equals.is_empty() {
                store.graph_projection_vector_query(query).await?
            } else {
                store
                    .graph_projection_vector_query_excluding_metadata(query, metadata_not_equals)
                    .await?
            };
            Ok(Value::Array(
                matches
                .into_iter()
                .map(|item| json!({"record": graph_record_json(item.record), "distance": item.distance}))
                .collect(),
            ))
        }
        PostgresStoreOperation::GraphProjectionVectorQueryBatch {
            namespace,
            workspace_id,
            graph_space,
            table,
            embeddings,
            limit,
            metadata,
            metadata_not_equals,
            metric,
            embedding_dim,
        } => {
            let scope = graph_scope(namespace, workspace_id, graph_space);
            let queries = embeddings
                .into_iter()
                .map(|embedding| kogwistar_store::GraphProjectionVectorQuery {
                    scope: scope.clone(),
                    table: table.clone(),
                    query: VectorQuery {
                        embedding,
                        limit,
                        metadata: metadata_filter(metadata.clone()),
                        metric: metric.into(),
                    },
                    embedding_dim,
                })
                .collect();
            let matches_by_embedding = if metadata_not_equals.is_empty() {
                store.graph_projection_vector_query_batch(queries).await?
            } else {
                store
                    .graph_projection_vector_query_batch_excluding_metadata(
                        queries,
                        metadata_not_equals,
                    )
                    .await?
            };
            Ok(Value::Array(
                matches_by_embedding
                    .into_iter()
                    .map(|matches| {
                        Value::Array(
                            matches
                                .into_iter()
                                .map(|item| {
                                    json!({"record": graph_record_json(item.record), "distance": item.distance})
                                })
                                .collect(),
                        )
                    })
                    .collect(),
            ))
        }
        PostgresStoreOperation::ExclusiveRawReplay {
            namespace,
            after_seq,
            limit,
        } => Ok(Value::Array(
            store
                .replay_raw_events(&namespace, after_seq, limit)
                .await?
                .into_iter()
                .map(postgres_raw_event_json)
                .collect(),
        )),
        PostgresStoreOperation::LatestRetainedEventSeq { namespace } => {
            Ok(json!(store.latest_retained_event_seq(&namespace).await?))
        }
        PostgresStoreOperation::PruneEntityEventsAfter { namespace, to_seq } => Ok(json!(
            store.prune_entity_events_after(&namespace, to_seq).await?
        )),
        PostgresStoreOperation::CursorGet {
            namespace,
            consumer,
        } => Ok(cursor_json(
            store.replay_cursor(&namespace, &consumer).await?,
        )),
        PostgresStoreOperation::LegacyCursorSet {
            namespace,
            consumer,
            last_seq,
        } => Ok(cursor_json(
            store
                .set_replay_cursor_legacy(&namespace, &consumer, last_seq)
                .await?,
        )),
        PostgresStoreOperation::StrictCursorAdvance {
            namespace,
            consumer,
            last_seq,
        } => Ok(cursor_json(
            store
                .strict_advance_replay_cursor(&namespace, &consumer, last_seq)
                .await?,
        )),
        PostgresStoreOperation::RecoverEntityProjection {
            namespace,
            consumer,
            projection_namespace,
            projection_key,
            batch_limit,
            abort_after_projection,
        } => Ok(entity_recovery_report_json(
            store
                .recover_entity_projection(entity_recovery_request(
                    namespace,
                    consumer,
                    projection_namespace,
                    projection_key,
                    batch_limit,
                    abort_after_projection,
                ))
                .await?,
        )),
        PostgresStoreOperation::RebuildEntityProjection {
            namespace,
            consumer,
            projection_namespace,
            projection_key,
            abort_after_projection,
        } => Ok(entity_recovery_report_json(
            store
                .rebuild_entity_projection(entity_rebuild_request(
                    namespace,
                    consumer,
                    projection_namespace,
                    projection_key,
                    abort_after_projection,
                ))
                .await?,
        )),
        PostgresStoreOperation::GetNamedProjection { namespace, key } => Ok(store
            .get_named_projection(&namespace, &key)
            .await?
            .map(projection_json)
            .unwrap_or(Value::Null)),
        PostgresStoreOperation::ListNamedProjections { namespace } => Ok(Value::Array(
            store
                .list_named_projections(&namespace)
                .await?
                .into_iter()
                .map(projection_json)
                .collect(),
        )),
        PostgresStoreOperation::ReplaceNamedProjection {
            namespace,
            key,
            payload,
            last_authoritative_seq,
            last_materialized_seq,
            projection_schema_version,
            materialization_status,
        } => {
            store
                .replace_named_projection(
                    &namespace,
                    &key,
                    projection_write(
                        payload,
                        last_authoritative_seq,
                        last_materialized_seq,
                        projection_schema_version,
                        materialization_status,
                    ),
                )
                .await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::CompareAndSwapNamedProjection {
            namespace,
            key,
            payload,
            expected_last_authoritative_seq,
            expected_last_materialized_seq,
            last_authoritative_seq,
            last_materialized_seq,
            projection_schema_version,
            materialization_status,
        } => Ok(json!(
            store
                .compare_and_swap_named_projection(
                    &namespace,
                    &key,
                    expected_last_authoritative_seq,
                    expected_last_materialized_seq,
                    projection_write(
                        payload,
                        last_authoritative_seq,
                        last_materialized_seq,
                        projection_schema_version,
                        materialization_status
                    ),
                )
                .await?
        )),
        PostgresStoreOperation::CompareAndSwapNamedProjections { updates } => {
            let updates = updates
                .into_iter()
                .map(|update| PostgresNamedProjectionCas {
                    namespace: update.namespace,
                    key: update.key,
                    expected_last_authoritative_seq: update.expected_last_authoritative_seq,
                    expected_last_materialized_seq: update.expected_last_materialized_seq,
                    projection: projection_write(
                        update.payload,
                        update.last_authoritative_seq,
                        update.last_materialized_seq,
                        update.projection_schema_version,
                        update.materialization_status,
                    ),
                })
                .collect();
            Ok(json!(
                store.compare_and_swap_named_projections(updates).await?
            ))
        }
        PostgresStoreOperation::ClearNamedProjection { namespace, key } => {
            store.clear_named_projection(&namespace, &key).await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::ClearProjectionNamespace { namespace } => {
            store.clear_projection_namespace(&namespace).await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::PutWorkflowDesignSnapshot {
            workflow_id,
            version,
            seq,
            payload_json,
            schema_version,
        } => {
            store
                .put_workflow_design_snapshot(
                    &workflow_id,
                    workflow_design_snapshot_write(version, seq, payload_json, schema_version),
                )
                .await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::GetWorkflowDesignSnapshot {
            workflow_id,
            max_version,
            schema_version,
        } => Ok(store
            .get_workflow_design_snapshot(&workflow_id, max_version, schema_version)
            .await?
            .map(workflow_design_snapshot_json)
            .unwrap_or(Value::Null)),
        PostgresStoreOperation::ClearWorkflowDesignSnapshots { workflow_id } => {
            store.clear_workflow_design_snapshots(&workflow_id).await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::PutWorkflowDesignDelta {
            workflow_id,
            version,
            prev_version,
            target_seq,
            forward_json,
            inverse_json,
            schema_version,
        } => {
            store
                .put_workflow_design_delta(
                    &workflow_id,
                    workflow_design_delta_write(
                        version,
                        prev_version,
                        target_seq,
                        forward_json,
                        inverse_json,
                        schema_version,
                    ),
                )
                .await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::GetWorkflowDesignDelta {
            workflow_id,
            version,
            schema_version,
        } => Ok(store
            .get_workflow_design_delta(&workflow_id, version, schema_version)
            .await?
            .map(workflow_design_delta_json)
            .unwrap_or(Value::Null)),
        PostgresStoreOperation::ClearWorkflowDesignDeltas { workflow_id } => {
            store.clear_workflow_design_deltas(&workflow_id).await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::CreateServerRun {
            run_id,
            conversation_id,
            workflow_id,
            user_id,
            user_turn_node_id,
            status,
        } => {
            store
                .create_server_run(ServerRunCreate {
                    run_id,
                    conversation_id,
                    workflow_id,
                    user_id,
                    user_turn_node_id,
                    status,
                })
                .await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::GetServerRun { run_id } => Ok(store
            .get_server_run(&run_id)
            .await?
            .map(server_run_json)
            .transpose()?
            .unwrap_or(Value::Null)),
        PostgresStoreOperation::ListServerRuns {
            status,
            workflow_id,
            conversation_id,
            limit,
        } => Ok(Value::Array(
            store
                .list_server_runs(
                    status.as_deref(),
                    workflow_id.as_deref(),
                    conversation_id.as_deref(),
                    limit,
                )
                .await?
                .into_iter()
                .map(server_run_json)
                .collect::<Result<_, _>>()?,
        )),
        PostgresStoreOperation::AppendServerRunEvent {
            run_id,
            event_type,
            payload_json,
        } => Ok(server_run_event_json(
            store
                .append_server_run_event(&run_id, &event_type, payload_json)
                .await?,
        )?),
        PostgresStoreOperation::ListServerRunEvents {
            run_id,
            after_seq,
            limit,
        } => Ok(Value::Array(
            store
                .list_server_run_events(&run_id, after_seq, limit)
                .await?
                .into_iter()
                .map(server_run_event_json)
                .collect::<Result<_, _>>()?,
        )),
        PostgresStoreOperation::UpdateServerRun {
            run_id,
            status,
            assistant_turn_node_id,
            result_json,
            error_json,
            started_at_ms,
            finished_at_ms,
            cancel_requested,
        } => {
            store
                .update_server_run(
                    &run_id,
                    ServerRunUpdate {
                        status,
                        assistant_turn_node_id,
                        result_json,
                        error_json,
                        started_at_ms,
                        finished_at_ms,
                        cancel_requested,
                    },
                )
                .await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::RequestServerRunCancel { run_id } => {
            store.request_server_run_cancel(&run_id).await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::ApplyRecordedRuntimeTransition {
            transition,
            abort_after_writes,
        } => serde_json::to_value(
            store
                .apply_recorded_runtime_transition(*transition, abort_after_writes)
                .await?,
        )
        .map_err(|error| PostgresStoreError::TransactionAborted(error.to_string())),
        PostgresStoreOperation::ApplyClaimedRecordedRuntimeTransition {
            handoff,
            transition,
            abort_after_writes,
        } => serde_json::to_value(
            store
                .apply_claimed_recorded_runtime_transition(handoff, *transition, abort_after_writes)
                .await?,
        )
        .map_err(|error| PostgresStoreError::TransactionAborted(error.to_string())),
        PostgresStoreOperation::ReadRecordedRuntimeState {
            run_id,
            workflow_id,
            conversation_id,
        } => serde_json::to_value(
            store
                .read_recorded_runtime_state(&run_id, &workflow_id, &conversation_id)
                .await?,
        )
        .map_err(|error| PostgresStoreError::TransactionAborted(error.to_string())),
        PostgresStoreOperation::EnqueueIndexJob {
            job_id,
            namespace,
            entity_kind,
            entity_id,
            index_kind,
            op,
            payload_json,
            max_retries,
        } => Ok(json!(
            store
                .enqueue_index_job(new_index_job(
                    job_id,
                    namespace,
                    entity_kind,
                    entity_id,
                    index_kind,
                    op,
                    payload_json,
                    max_retries
                ))
                .await?
        )),
        PostgresStoreOperation::ClaimIndexJobs {
            limit,
            lease_seconds,
            namespace,
        } => Ok(Value::Array(
            store
                .claim_index_jobs(limit, lease_seconds, namespace.as_deref())
                .await?
                .into_iter()
                .map(index_job_json)
                .collect(),
        )),
        PostgresStoreOperation::ReconcileJoinIndexJobs {
            limit,
            lease_seconds,
            namespace,
            nodes_table,
            edges_table,
            node_docs_table,
            node_refs_table,
            edge_refs_table,
            edge_endpoints_table,
        } => Ok(json!(
            store
                .reconcile_join_index_jobs(
                    limit,
                    lease_seconds,
                    namespace.as_deref(),
                    GraphIndexTableNames {
                        nodes: nodes_table,
                        edges: edges_table,
                        node_docs: node_docs_table,
                        node_refs: node_refs_table,
                        edge_refs: edge_refs_table,
                        edge_endpoints: edge_endpoints_table,
                    },
                )
                .await?
        )),
        PostgresStoreOperation::MarkIndexJobDone {
            job_id,
            claim_token,
        } => Ok(json!(
            store
                .mark_index_job_done(&job_id, claim_token.as_deref())
                .await?
        )),
        PostgresStoreOperation::AcceptIndexJobResult {
            job_id,
            claim_token,
            result_json,
            result_sha256,
        } => Ok(accepted_index_job_result_json(
            store
                .accept_index_job_result(&job_id, &claim_token, &result_json, &result_sha256)
                .await?,
        )),
        PostgresStoreOperation::GetIndexJobResult { job_id } => Ok(store
            .index_job_result(&job_id)
            .await?
            .map(accepted_index_job_result_json)
            .unwrap_or(Value::Null)),
        PostgresStoreOperation::MarkIndexJobFailed {
            job_id,
            error,
            final_,
            claim_token,
        } => {
            store
                .mark_index_job_failed(&job_id, &error, final_, claim_token.as_deref())
                .await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::BumpRetryAndRequeue {
            job_id,
            error,
            next_run_at_seconds,
            claim_token,
        } => {
            store
                .bump_retry_and_requeue(
                    &job_id,
                    &error,
                    next_run_at_seconds,
                    claim_token.as_deref(),
                )
                .await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::RenewIndexJobLease {
            job_id,
            claim_token,
            lease_seconds,
        } => Ok(json!(
            store
                .renew_index_job_lease(&job_id, &claim_token, lease_seconds)
                .await?
        )),
        PostgresStoreOperation::RequeueIndexJobAtTail {
            job_id,
            payload_json,
            delay_seconds,
            claim_token,
        } => {
            store
                .requeue_index_job_at_tail(
                    &job_id,
                    payload_json,
                    delay_seconds,
                    claim_token.as_deref(),
                )
                .await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::ListIndexJobs {
            namespace,
            status,
            entity_kind,
            entity_id,
            index_kind,
            limit,
        } => Ok(Value::Array(
            store
                .list_index_jobs(
                    namespace.as_deref(),
                    status.as_deref(),
                    entity_kind.as_deref(),
                    entity_id.as_deref(),
                    index_kind.as_deref(),
                    limit,
                )
                .await?
                .into_iter()
                .map(index_job_json)
                .collect(),
        )),
        PostgresStoreOperation::ProjectLaneMessage {
            message_id,
            namespace,
            purpose,
            inbox_id,
            conversation_id,
            recipient_id,
            sender_id,
            msg_type,
            status,
            created_at,
            available_at,
            run_id,
            step_id,
            correlation_id,
            payload_json,
            error_json,
        } => {
            store
                .project_lane_message(new_lane_message(
                    message_id,
                    namespace,
                    purpose,
                    inbox_id,
                    conversation_id,
                    recipient_id,
                    sender_id,
                    msg_type,
                    status,
                    created_at,
                    available_at,
                    run_id,
                    step_id,
                    correlation_id,
                    payload_json,
                    error_json,
                ))
                .await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::GetProjectedLaneMessage { message_id } => Ok(store
            .get_projected_lane_message(&message_id)
            .await?
            .map(lane_message_json)
            .unwrap_or(Value::Null)),
        PostgresStoreOperation::UpdateProjectedLaneMessageStatus {
            message_id,
            status,
            error_json,
        } => {
            store
                .update_projected_lane_message_status(&message_id, &status, error_json)
                .await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::UpdateProjectedLaneMessageLinks {
            message_id,
            prev_message_id,
            next_message_id,
            inbox_tail_message_id,
            conversation_tail_message_id,
        } => {
            store
                .update_projected_lane_message_links(
                    &message_id,
                    prev_message_id,
                    next_message_id,
                    inbox_tail_message_id,
                    conversation_tail_message_id,
                )
                .await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::ListProjectedLaneMessages {
            namespace,
            purpose,
            inbox_id,
            conversation_id,
            run_id,
            status,
            msg_type,
            sender_id,
            recipient_id,
            correlation_id,
            reply_to_message_id,
            created_at_gte,
            created_at_lte,
            available_at_gte,
            available_at_lte,
            limit,
            newest_first,
        } => {
            let native_limit = if run_id.is_some() { usize::MAX } else { limit };
            let rows = store
                .list_projected_lane_messages(lane_filter(
                    namespace,
                    purpose,
                    inbox_id,
                    conversation_id,
                    status,
                    msg_type,
                    sender_id,
                    recipient_id,
                    correlation_id,
                    reply_to_message_id,
                    created_at_gte,
                    created_at_lte,
                    available_at_gte,
                    available_at_lte,
                    native_limit,
                    newest_first,
                ))
                .await?;
            Ok(Value::Array(
                rows.into_iter()
                    .filter(|row| {
                        run_id
                            .as_ref()
                            .is_none_or(|value| row.run_id.as_deref() == Some(value))
                    })
                    .take(limit)
                    .map(lane_message_json)
                    .collect(),
            ))
        }
        PostgresStoreOperation::ClearProjectedLaneMessages { namespace } => Ok(json!(
            store.clear_projected_lane_messages(&namespace).await?
        )),
        PostgresStoreOperation::ClaimProjectedLaneMessages {
            namespace,
            inbox_id,
            claimed_by,
            limit,
            lease_seconds,
            message_ids,
            run_id,
            msg_type,
            recipient_id,
        } => Ok(Value::Array(
            store
                .claim_projected_lane_messages_filtered(lane_claim_filter(
                    namespace,
                    inbox_id,
                    claimed_by,
                    message_ids,
                    run_id,
                    msg_type,
                    recipient_id,
                    limit,
                    lease_seconds,
                ))
                .await?
                .into_iter()
                .map(lane_message_json)
                .collect(),
        )),
        PostgresStoreOperation::AckProjectedLaneMessage {
            message_id,
            claimed_by,
        } => {
            store
                .ack_projected_lane_message(&message_id, &claimed_by)
                .await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::RequeueProjectedLaneMessage {
            message_id,
            claimed_by,
            error_json,
            delay_seconds,
        } => {
            store
                .requeue_projected_lane_message(&message_id, &claimed_by, error_json, delay_seconds)
                .await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::DeadLetterProjectedLaneMessage {
            message_id,
            claimed_by,
            error_json,
        } => {
            store
                .dead_letter_projected_lane_message(&message_id, &claimed_by, error_json)
                .await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::Batch { operations, abort } => {
            store
                .transaction(move |uow| {
                    Box::pin(async move {
                        let mut results = Vec::with_capacity(operations.len());
                        for operation in operations {
                            results.push(postgres_uow_operation_json(uow, operation).await?);
                        }
                        if abort {
                            return Err(PostgresStoreError::TransactionAborted(
                                "requested by PostgreSQL JSON ABI abort flag".to_owned(),
                            ));
                        }
                        Ok(Value::Array(results))
                    })
                })
                .await
        }
    }
}

async fn postgres_uow_operation_json(
    uow: &mut PostgresUnitOfWork<'_>,
    operation: PostgresStoreOperation,
) -> Result<Value, PostgresStoreError> {
    match operation {
        PostgresStoreOperation::LockEntityIdentity {
            namespace,
            entity_kind,
            entity_id,
        } => {
            uow.lock_entity_identity(&namespace, &entity_kind, &entity_id)
                .await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::NextGlobalSeq => Ok(json!(uow.next_global_seq().await?)),
        PostgresStoreOperation::CurrentGlobalSeq => Ok(json!(uow.current_global_seq().await?)),
        PostgresStoreOperation::NextUserSeq { user_id } => {
            Ok(json!(uow.next_user_seq(&user_id).await?))
        }
        PostgresStoreOperation::CurrentUserSeq { user_id } => {
            Ok(json!(uow.current_user_seq(&user_id).await?))
        }
        PostgresStoreOperation::SetUserSeq { user_id, value } => {
            uow.set_user_seq(&user_id, value).await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::GetIndexAppliedFingerprint {
            namespace,
            coalesce_key,
        } => Ok(json!(
            uow.get_index_applied_fingerprint(&namespace, &coalesce_key)
                .await?
        )),
        PostgresStoreOperation::SetIndexAppliedFingerprint {
            namespace,
            coalesce_key,
            applied_fingerprint,
            last_job_id,
        } => {
            uow.set_index_applied_fingerprint(
                &namespace,
                &coalesce_key,
                applied_fingerprint,
                last_job_id,
            )
            .await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::AllocEventSeq { namespace } => {
            Ok(json!(uow.alloc_event_seq(&namespace).await?))
        }
        PostgresStoreOperation::RawAppend {
            namespace,
            event_id,
            entity_kind,
            entity_id,
            op,
            payload_json,
        } => Ok(postgres_appended_raw_event_json(
            uow.append_raw_entity_event(
                &namespace,
                postgres_new_raw_event(event_id, entity_kind, entity_id, op, payload_json),
            )
            .await?,
        )),
        PostgresStoreOperation::GraphMutation {
            namespace,
            workspace_id,
            graph_space,
            table,
            entity_kind,
            event_id,
            op,
            record,
            payload,
            embedding_dim,
            model_envelope,
        } => {
            let mutation = kogwistar_store::GraphMutation {
                scope: graph_scope(namespace, workspace_id, graph_space),
                table,
                entity_kind,
                event_id,
                op,
                payload,
                record: graph_mutation_record(record),
                embedding_dim,
            };
            if let Some(envelope) = model_envelope {
                kogwistar_store::validate_typed_graph_mutation(&mutation, &envelope)?;
            }
            Ok(graph_mutation_json(
                uow.apply_graph_mutation(mutation).await?,
            ))
        }
        PostgresStoreOperation::ExclusiveRawReplay {
            namespace,
            after_seq,
            limit,
        } => Ok(Value::Array(
            uow.replay_raw_events(&namespace, after_seq, limit)
                .await?
                .into_iter()
                .map(postgres_raw_event_json)
                .collect(),
        )),
        PostgresStoreOperation::GraphMetadataPatchMutation {
            namespace,
            workspace_id,
            graph_space,
            table,
            entity_kind,
            event_id,
            op,
            entity_id,
            document,
            metadata_patch,
            payload,
        } => Ok(optional_graph_mutation_json(
            uow.apply_graph_metadata_patch_mutation(
                kogwistar_store_postgres::GraphMetadataPatchMutation {
                    scope: graph_scope(namespace, workspace_id, graph_space),
                    table,
                    entity_kind,
                    event_id,
                    op,
                    entity_id,
                    document,
                    metadata_patch,
                    payload,
                },
            )
            .await?,
        )),
        PostgresStoreOperation::GraphDeleteMutation {
            namespace,
            workspace_id,
            graph_space,
            table,
            entity_kind,
            event_id,
            entity_id,
            payload,
        } => Ok(optional_graph_mutation_json(
            uow.apply_graph_delete_mutation(kogwistar_store_postgres::GraphDeleteMutation {
                scope: graph_scope(namespace, workspace_id, graph_space),
                table,
                entity_kind,
                event_id,
                entity_id,
                payload,
            })
            .await?,
        )),
        PostgresStoreOperation::UpsertGraphProjection {
            namespace,
            workspace_id,
            graph_space,
            table,
            record,
            embedding_dim,
        } => {
            uow.upsert_graph_projection(kogwistar_store_postgres::GraphProjectionUpsert {
                scope: graph_scope(namespace, workspace_id, graph_space),
                table,
                record: graph_mutation_record(record),
                embedding_dim,
            })
            .await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::PatchGraphProjectionMetadata {
            namespace,
            workspace_id,
            graph_space,
            table,
            entity_id,
            document,
            metadata_patch,
            patch_document_metadata,
        } => Ok(json!(
            uow.patch_graph_projection_metadata(
                kogwistar_store_postgres::GraphProjectionMetadataPatch {
                    scope: graph_scope(namespace, workspace_id, graph_space),
                    table,
                    entity_id,
                    document,
                    metadata_patch,
                    patch_document_metadata,
                },
            )
            .await?
        )),
        PostgresStoreOperation::GraphProjectionRecords {
            namespace,
            workspace_id,
            graph_space,
            table,
            ids,
            metadata,
            after_id,
            limit,
            for_update,
        } => {
            if after_id.is_some() || !for_update {
                return Err(PostgresStoreError::Backend(
                    "transactional graph projection reads require for_update=true and no page cursor"
                        .to_owned(),
                ));
            }
            let records = uow
                .graph_projection_records_for_update(GraphProjectionRead {
                    scope: graph_scope(namespace, workspace_id, graph_space),
                    table,
                    ids,
                    metadata: metadata_filter(metadata),
                    limit,
                })
                .await?;
            Ok(Value::Array(
                records.into_iter().map(graph_record_json).collect(),
            ))
        }
        PostgresStoreOperation::PruneEntityEventsAfter { namespace, to_seq } => Ok(json!(
            uow.prune_entity_events_after(&namespace, to_seq).await?
        )),
        PostgresStoreOperation::LegacyCursorSet {
            namespace,
            consumer,
            last_seq,
        } => Ok(cursor_json(
            uow.set_replay_cursor_legacy(&namespace, &consumer, last_seq)
                .await?,
        )),
        PostgresStoreOperation::StrictCursorAdvance {
            namespace,
            consumer,
            last_seq,
        } => Ok(cursor_json(
            uow.strict_advance_replay_cursor(&namespace, &consumer, last_seq)
                .await?,
        )),
        PostgresStoreOperation::ReplaceNamedProjection {
            namespace,
            key,
            payload,
            last_authoritative_seq,
            last_materialized_seq,
            projection_schema_version,
            materialization_status,
        } => {
            uow.replace_named_projection(
                &namespace,
                &key,
                projection_write(
                    payload,
                    last_authoritative_seq,
                    last_materialized_seq,
                    projection_schema_version,
                    materialization_status,
                ),
            )
            .await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::CompareAndSwapNamedProjection {
            namespace,
            key,
            payload,
            expected_last_authoritative_seq,
            expected_last_materialized_seq,
            last_authoritative_seq,
            last_materialized_seq,
            projection_schema_version,
            materialization_status,
        } => Ok(json!(
            uow.compare_and_swap_named_projection(
                &namespace,
                &key,
                expected_last_authoritative_seq,
                expected_last_materialized_seq,
                projection_write(
                    payload,
                    last_authoritative_seq,
                    last_materialized_seq,
                    projection_schema_version,
                    materialization_status
                ),
            )
            .await?
        )),
        PostgresStoreOperation::CompareAndSwapNamedProjections { updates } => {
            let updates = updates
                .into_iter()
                .map(|update| PostgresNamedProjectionCas {
                    namespace: update.namespace,
                    key: update.key,
                    expected_last_authoritative_seq: update.expected_last_authoritative_seq,
                    expected_last_materialized_seq: update.expected_last_materialized_seq,
                    projection: projection_write(
                        update.payload,
                        update.last_authoritative_seq,
                        update.last_materialized_seq,
                        update.projection_schema_version,
                        update.materialization_status,
                    ),
                })
                .collect();
            Ok(json!(
                uow.compare_and_swap_named_projections(updates).await?
            ))
        }
        PostgresStoreOperation::ClearNamedProjection { namespace, key } => {
            uow.clear_named_projection(&namespace, &key).await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::ClearProjectionNamespace { namespace } => {
            uow.clear_projection_namespace(&namespace).await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::PutWorkflowDesignSnapshot {
            workflow_id,
            version,
            seq,
            payload_json,
            schema_version,
        } => {
            uow.put_workflow_design_snapshot(
                &workflow_id,
                workflow_design_snapshot_write(version, seq, payload_json, schema_version),
            )
            .await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::ClearWorkflowDesignSnapshots { workflow_id } => {
            uow.clear_workflow_design_snapshots(&workflow_id).await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::PutWorkflowDesignDelta {
            workflow_id,
            version,
            prev_version,
            target_seq,
            forward_json,
            inverse_json,
            schema_version,
        } => {
            uow.put_workflow_design_delta(
                &workflow_id,
                workflow_design_delta_write(
                    version,
                    prev_version,
                    target_seq,
                    forward_json,
                    inverse_json,
                    schema_version,
                ),
            )
            .await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::ClearWorkflowDesignDeltas { workflow_id } => {
            uow.clear_workflow_design_deltas(&workflow_id).await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::CreateServerRun {
            run_id,
            conversation_id,
            workflow_id,
            user_id,
            user_turn_node_id,
            status,
        } => {
            uow.create_server_run(ServerRunCreate {
                run_id,
                conversation_id,
                workflow_id,
                user_id,
                user_turn_node_id,
                status,
            })
            .await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::AppendServerRunEvent {
            run_id,
            event_type,
            payload_json,
        } => Ok(server_run_event_json(
            uow.append_server_run_event(&run_id, &event_type, payload_json)
                .await?,
        )?),
        PostgresStoreOperation::UpdateServerRun {
            run_id,
            status,
            assistant_turn_node_id,
            result_json,
            error_json,
            started_at_ms,
            finished_at_ms,
            cancel_requested,
        } => {
            uow.update_server_run(
                &run_id,
                ServerRunUpdate {
                    status,
                    assistant_turn_node_id,
                    result_json,
                    error_json,
                    started_at_ms,
                    finished_at_ms,
                    cancel_requested,
                },
            )
            .await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::RequestServerRunCancel { run_id } => {
            uow.request_server_run_cancel(&run_id).await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::ApplyRecordedRuntimeTransition {
            transition,
            abort_after_writes,
        } => serde_json::to_value(
            uow.apply_recorded_runtime_transition(*transition, abort_after_writes)
                .await?,
        )
        .map_err(|error| PostgresStoreError::TransactionAborted(error.to_string())),
        PostgresStoreOperation::ApplyClaimedRecordedRuntimeTransition {
            handoff,
            transition,
            abort_after_writes,
        } => serde_json::to_value(
            uow.apply_claimed_recorded_runtime_transition(handoff, *transition, abort_after_writes)
                .await?,
        )
        .map_err(|error| PostgresStoreError::TransactionAborted(error.to_string())),
        PostgresStoreOperation::EnqueueIndexJob {
            job_id,
            namespace,
            entity_kind,
            entity_id,
            index_kind,
            op,
            payload_json,
            max_retries,
        } => Ok(json!(
            uow.enqueue_index_job(new_index_job(
                job_id,
                namespace,
                entity_kind,
                entity_id,
                index_kind,
                op,
                payload_json,
                max_retries
            ))
            .await?
        )),
        PostgresStoreOperation::ClaimIndexJobs {
            limit,
            lease_seconds,
            namespace,
        } => Ok(Value::Array(
            uow.claim_index_jobs(limit, lease_seconds, namespace.as_deref())
                .await?
                .into_iter()
                .map(index_job_json)
                .collect(),
        )),
        PostgresStoreOperation::ReconcileJoinIndexJobs {
            limit,
            lease_seconds,
            namespace,
            nodes_table,
            edges_table,
            node_docs_table,
            node_refs_table,
            edge_refs_table,
            edge_endpoints_table,
        } => Ok(json!(
            uow.reconcile_join_index_jobs(
                limit,
                lease_seconds,
                namespace.as_deref(),
                GraphIndexTableNames {
                    nodes: nodes_table,
                    edges: edges_table,
                    node_docs: node_docs_table,
                    node_refs: node_refs_table,
                    edge_refs: edge_refs_table,
                    edge_endpoints: edge_endpoints_table,
                },
            )
            .await?
        )),
        PostgresStoreOperation::MarkIndexJobDone {
            job_id,
            claim_token,
        } => Ok(json!(
            uow.mark_index_job_done(&job_id, claim_token.as_deref())
                .await?
        )),
        PostgresStoreOperation::AcceptIndexJobResult {
            job_id,
            claim_token,
            result_json,
            result_sha256,
        } => Ok(accepted_index_job_result_json(
            uow.accept_index_job_result(&job_id, &claim_token, &result_json, &result_sha256)
                .await?,
        )),
        PostgresStoreOperation::GetIndexJobResult { .. } => {
            Err(PostgresStoreError::TransactionAborted(
                "get_index_job_result is not available inside a transaction".to_owned(),
            ))
        }
        PostgresStoreOperation::MarkIndexJobFailed {
            job_id,
            error,
            final_,
            claim_token,
        } => {
            uow.mark_index_job_failed(&job_id, &error, final_, claim_token.as_deref())
                .await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::BumpRetryAndRequeue {
            job_id,
            error,
            next_run_at_seconds,
            claim_token,
        } => {
            uow.bump_retry_and_requeue(
                &job_id,
                &error,
                next_run_at_seconds,
                claim_token.as_deref(),
            )
            .await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::RenewIndexJobLease {
            job_id,
            claim_token,
            lease_seconds,
        } => Ok(json!(
            uow.renew_index_job_lease(&job_id, &claim_token, lease_seconds)
                .await?
        )),
        PostgresStoreOperation::RequeueIndexJobAtTail {
            job_id,
            payload_json,
            delay_seconds,
            claim_token,
        } => {
            uow.requeue_index_job_at_tail(
                &job_id,
                payload_json,
                delay_seconds,
                claim_token.as_deref(),
            )
            .await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::ProjectLaneMessage {
            message_id,
            namespace,
            purpose,
            inbox_id,
            conversation_id,
            recipient_id,
            sender_id,
            msg_type,
            status,
            created_at,
            available_at,
            run_id,
            step_id,
            correlation_id,
            payload_json,
            error_json,
        } => {
            uow.project_lane_message(new_lane_message(
                message_id,
                namespace,
                purpose,
                inbox_id,
                conversation_id,
                recipient_id,
                sender_id,
                msg_type,
                status,
                created_at,
                available_at,
                run_id,
                step_id,
                correlation_id,
                payload_json,
                error_json,
            ))
            .await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::UpdateProjectedLaneMessageStatus {
            message_id,
            status,
            error_json,
        } => {
            uow.update_projected_lane_message_status(&message_id, &status, error_json)
                .await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::UpdateProjectedLaneMessageLinks {
            message_id,
            prev_message_id,
            next_message_id,
            inbox_tail_message_id,
            conversation_tail_message_id,
        } => {
            uow.update_projected_lane_message_links(
                &message_id,
                prev_message_id,
                next_message_id,
                inbox_tail_message_id,
                conversation_tail_message_id,
            )
            .await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::ClearProjectedLaneMessages { namespace } => {
            Ok(json!(uow.clear_projected_lane_messages(&namespace).await?))
        }
        PostgresStoreOperation::ClaimProjectedLaneMessages {
            namespace,
            inbox_id,
            claimed_by,
            limit,
            lease_seconds,
            message_ids,
            run_id,
            msg_type,
            recipient_id,
        } => Ok(Value::Array(
            uow.claim_projected_lane_messages_filtered(lane_claim_filter(
                namespace,
                inbox_id,
                claimed_by,
                message_ids,
                run_id,
                msg_type,
                recipient_id,
                limit,
                lease_seconds,
            ))
            .await?
            .into_iter()
            .map(lane_message_json)
            .collect(),
        )),
        PostgresStoreOperation::AckProjectedLaneMessage {
            message_id,
            claimed_by,
        } => {
            uow.ack_projected_lane_message(&message_id, &claimed_by)
                .await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::RequeueProjectedLaneMessage {
            message_id,
            claimed_by,
            error_json,
            delay_seconds,
        } => {
            uow.requeue_projected_lane_message(&message_id, &claimed_by, error_json, delay_seconds)
                .await?;
            Ok(Value::Null)
        }
        PostgresStoreOperation::DeadLetterProjectedLaneMessage {
            message_id,
            claimed_by,
            error_json,
        } => {
            uow.dead_letter_projected_lane_message(&message_id, &claimed_by, error_json)
                .await?;
            Ok(Value::Null)
        }
        _ => Err(PostgresStoreError::TransactionAborted(
            "unsupported PostgreSQL batch operation".to_owned(),
        )),
    }
}
