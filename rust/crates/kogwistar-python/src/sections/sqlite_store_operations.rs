// SQLite operation translation, sessions, and cache policy. Python parity: kogwistar.engine_core.engine_sqlite and backend bridge.
fn raw_event_json(event: RawEntityEvent) -> Value {
    json!({
        "namespace": event.namespace,
        "seq": event.seq,
        "event_id": event.event_id,
        "entity_kind": event.entity_kind,
        "entity_id": event.entity_id,
        "op": event.op,
        "payload_json": event.payload_json,
        "created_at": event.created_at,
    })
}

fn cursor_json(cursor: ReplayCursor) -> Value {
    json!({"namespace": cursor.namespace, "consumer": cursor.consumer, "last_seq": cursor.last_seq})
}

fn entity_recovery_report_json(report: EntityRecoveryReport) -> Value {
    json!({
        "processed_count": report.processed_count,
        "prior_cursor": report.prior_cursor,
        "new_cursor": report.new_cursor,
        "latest_authoritative_seq": report.latest_authoritative_seq,
        "caught_up": report.caught_up,
        "canonical_payload": report.canonical_payload,
        "digest": report.digest,
    })
}

fn entity_recovery_request(
    namespace: String,
    consumer: String,
    projection_namespace: String,
    projection_key: String,
    batch_limit: usize,
    abort_after_projection: bool,
) -> EntityRecoveryRequest {
    EntityRecoveryRequest {
        namespace,
        consumer,
        projection_namespace,
        projection_key,
        batch_limit,
        abort_after_projection,
    }
}

fn entity_rebuild_request(
    namespace: String,
    consumer: String,
    projection_namespace: String,
    projection_key: String,
    abort_after_projection: bool,
) -> EntityRebuildRequest {
    EntityRebuildRequest {
        namespace,
        consumer,
        projection_namespace,
        projection_key,
        abort_after_projection,
    }
}

fn new_raw_event(
    event_id: String,
    entity_kind: String,
    entity_id: String,
    op: String,
    payload_json: String,
) -> NewRawEntityEvent {
    NewRawEntityEvent {
        event_id,
        entity_kind,
        entity_id,
        op,
        payload_json,
    }
}

fn appended_raw_event_json(appended: kogwistar_store_sqlite::AppendedRawEvent) -> Value {
    let seq = appended.event.seq;
    json!({"seq": seq, "inserted": appended.inserted, "event": raw_event_json(appended.event)})
}

fn sqlite_store_operation_json(
    store: &SqliteStore,
    operation: SqliteStoreOperation,
) -> Result<Value, SqliteStoreError> {
    match operation {
        SqliteStoreOperation::OpenInit => Ok(json!({"initialized": true})),
        SqliteStoreOperation::Close => Err(SqliteStoreError::TransactionAborted(
            "SQLite close must be handled by the session".to_owned(),
        )),
        SqliteStoreOperation::BeginTransaction
        | SqliteStoreOperation::CommitTransaction
        | SqliteStoreOperation::RollbackTransaction => Err(SqliteStoreError::TransactionAborted(
            "SQLite transaction control must be handled by the session".to_owned(),
        )),
        SqliteStoreOperation::CurrentGlobalSeq => Ok(json!(store.current_global_seq()?)),
        SqliteStoreOperation::NextGlobalSeq => Ok(json!(store.next_global_seq()?)),
        SqliteStoreOperation::CurrentUserSeq { user_id } => {
            Ok(json!(store.current_user_seq(&user_id)?))
        }
        SqliteStoreOperation::NextUserSeq { user_id } => Ok(json!(store.next_user_seq(&user_id)?)),
        SqliteStoreOperation::SetUserSeq { user_id, value } => {
            store.set_user_seq(&user_id, value)?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::CurrentScopedSeq { scope_id } => {
            Ok(json!(store.current_scoped_seq(&scope_id)?))
        }
        SqliteStoreOperation::NextScopedSeq { scope_id } => {
            Ok(json!(store.next_scoped_seq(&scope_id)?))
        }
        SqliteStoreOperation::SetScopedSeq { scope_id, value } => {
            store.set_scoped_seq(&scope_id, value)?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::AllocEventSeq { namespace } => {
            Ok(json!(store.alloc_event_seq(&namespace)?))
        }
        SqliteStoreOperation::RawAppend {
            namespace,
            event_id,
            entity_kind,
            entity_id,
            op,
            payload_json,
        } => Ok(appended_raw_event_json(store.append_raw_entity_event(
            &namespace,
            new_raw_event(event_id, entity_kind, entity_id, op, payload_json),
        )?)),
        SqliteStoreOperation::RawRestore {
            namespace,
            seq,
            event_id,
            entity_kind,
            entity_id,
            op,
            payload_json,
            created_at,
        } => Ok(appended_raw_event_json(
            store.append_restored_raw_entity_event(
                &namespace,
                RestoredRawEntityEvent {
                    seq,
                    event_id,
                    entity_kind,
                    entity_id,
                    op,
                    payload_json,
                    created_at,
                },
            )?,
        )),
        SqliteStoreOperation::ExclusiveRawReplay {
            namespace,
            after_seq,
            limit,
        } => Ok(Value::Array(
            store
                .replay_raw_events(&namespace, after_seq, limit)?
                .into_iter()
                .map(raw_event_json)
                .collect(),
        )),
        SqliteStoreOperation::LatestRetainedEventSeq { namespace } => {
            Ok(json!(store.latest_retained_event_seq(&namespace)?))
        }
        SqliteStoreOperation::PruneEntityEventsAfter { namespace, to_seq } => {
            Ok(json!(store.prune_entity_events_after(&namespace, to_seq)?))
        }
        SqliteStoreOperation::CursorGet {
            namespace,
            consumer,
        } => Ok(cursor_json(store.replay_cursor(&namespace, &consumer)?)),
        SqliteStoreOperation::LegacyCursorSet {
            namespace,
            consumer,
            last_seq,
        } => Ok(cursor_json(
            store.set_replay_cursor_legacy(&namespace, &consumer, last_seq)?,
        )),
        SqliteStoreOperation::StrictCursorAdvance {
            namespace,
            consumer,
            last_seq,
        } => Ok(cursor_json(store.strict_advance_replay_cursor(
            &namespace, &consumer, last_seq,
        )?)),
        SqliteStoreOperation::RecoverEntityProjection {
            namespace,
            consumer,
            projection_namespace,
            projection_key,
            batch_limit,
            abort_after_projection,
        } => Ok(entity_recovery_report_json(
            store.recover_entity_projection(entity_recovery_request(
                namespace,
                consumer,
                projection_namespace,
                projection_key,
                batch_limit,
                abort_after_projection,
            ))?,
        )),
        SqliteStoreOperation::RebuildEntityProjection {
            namespace,
            consumer,
            projection_namespace,
            projection_key,
            abort_after_projection,
        } => Ok(entity_recovery_report_json(
            store.rebuild_entity_projection(entity_rebuild_request(
                namespace,
                consumer,
                projection_namespace,
                projection_key,
                abort_after_projection,
            ))?,
        )),
        SqliteStoreOperation::GetNamedProjection { namespace, key } => Ok(store
            .get_named_projection(&namespace, &key)?
            .map(projection_json)
            .unwrap_or(Value::Null)),
        SqliteStoreOperation::ListNamedProjections { namespace } => Ok(Value::Array(
            store
                .list_named_projections(&namespace)?
                .into_iter()
                .map(projection_json)
                .collect(),
        )),
        SqliteStoreOperation::ReplaceNamedProjection {
            namespace,
            key,
            payload,
            last_authoritative_seq,
            last_materialized_seq,
            projection_schema_version,
            materialization_status,
        } => {
            store.replace_named_projection(
                &namespace,
                &key,
                projection_write(
                    payload,
                    last_authoritative_seq,
                    last_materialized_seq,
                    projection_schema_version,
                    materialization_status,
                ),
            )?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::CompareAndSwapNamedProjection {
            namespace,
            key,
            payload,
            expected_last_authoritative_seq,
            expected_last_materialized_seq,
            last_authoritative_seq,
            last_materialized_seq,
            projection_schema_version,
            materialization_status,
        } => Ok(json!(store.compare_and_swap_named_projection(
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
        )?)),
        SqliteStoreOperation::CompareAndSwapNamedProjections { updates } => {
            let updates = updates
                .into_iter()
                .map(|update| NamedProjectionCas {
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
            Ok(json!(store.compare_and_swap_named_projections(updates)?))
        }
        SqliteStoreOperation::ClearNamedProjection { namespace, key } => {
            store.clear_named_projection(&namespace, &key)?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::ClearProjectionNamespace { namespace } => {
            store.clear_projection_namespace(&namespace)?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::GetStage1NodeProjection { namespace, key } => Ok(store
            .get_stage1_node_projection(&namespace, &key)?
            .map(projection_json)
            .unwrap_or(Value::Null)),
        SqliteStoreOperation::ListStage1NodeProjections { namespace } => Ok(Value::Array(
            store
                .list_stage1_node_projections(&namespace)?
                .into_iter()
                .map(projection_json)
                .collect(),
        )),
        SqliteStoreOperation::ReplaceStage1NodeProjection {
            namespace,
            key,
            payload,
            last_authoritative_seq,
            last_materialized_seq,
            projection_schema_version,
            materialization_status,
        } => {
            store.replace_stage1_node_projection(
                &namespace,
                &key,
                projection_write(
                    payload,
                    last_authoritative_seq,
                    last_materialized_seq,
                    projection_schema_version,
                    materialization_status,
                ),
            )?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::ClearStage1NodeProjection { namespace, key } => {
            store.clear_stage1_node_projection(&namespace, &key)?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::ClearStage1NodeProjectionIfCurrent {
            namespace,
            key,
            expected_last_authoritative_seq,
        } => Ok(json!(store.clear_stage1_node_projection_if_current(
            &namespace,
            &key,
            expected_last_authoritative_seq,
        )?)),
        SqliteStoreOperation::PutWorkflowDesignSnapshot {
            workflow_id,
            version,
            seq,
            payload_json,
            schema_version,
        } => {
            store.put_workflow_design_snapshot(
                &workflow_id,
                workflow_design_snapshot_write(version, seq, payload_json, schema_version),
            )?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::GetWorkflowDesignSnapshot {
            workflow_id,
            max_version,
            schema_version,
        } => Ok(store
            .get_workflow_design_snapshot(&workflow_id, max_version, schema_version)?
            .map(workflow_design_snapshot_json)
            .unwrap_or(Value::Null)),
        SqliteStoreOperation::ClearWorkflowDesignSnapshots { workflow_id } => {
            store.clear_workflow_design_snapshots(&workflow_id)?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::PutWorkflowDesignDelta {
            workflow_id,
            version,
            prev_version,
            target_seq,
            forward_json,
            inverse_json,
            schema_version,
        } => {
            store.put_workflow_design_delta(
                &workflow_id,
                workflow_design_delta_write(
                    version,
                    prev_version,
                    target_seq,
                    forward_json,
                    inverse_json,
                    schema_version,
                ),
            )?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::GetWorkflowDesignDelta {
            workflow_id,
            version,
            schema_version,
        } => Ok(store
            .get_workflow_design_delta(&workflow_id, version, schema_version)?
            .map(workflow_design_delta_json)
            .unwrap_or(Value::Null)),
        SqliteStoreOperation::ClearWorkflowDesignDeltas { workflow_id } => {
            store.clear_workflow_design_deltas(&workflow_id)?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::CreateServerRun {
            run_id,
            conversation_id,
            workflow_id,
            user_id,
            user_turn_node_id,
            status,
        } => {
            store.create_server_run(ServerRunCreate {
                run_id,
                conversation_id,
                workflow_id,
                user_id,
                user_turn_node_id,
                status,
            })?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::GetServerRun { run_id } => Ok(store
            .get_server_run(&run_id)?
            .map(server_run_json)
            .transpose()?
            .unwrap_or(Value::Null)),
        SqliteStoreOperation::ListServerRuns {
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
                )?
                .into_iter()
                .map(server_run_json)
                .collect::<Result<_, _>>()?,
        )),
        SqliteStoreOperation::AppendServerRunEvent {
            run_id,
            event_type,
            payload_json,
        } => Ok(server_run_event_json(store.append_server_run_event(
            &run_id,
            &event_type,
            payload_json,
        )?)?),
        SqliteStoreOperation::ListServerRunEvents {
            run_id,
            after_seq,
            limit,
        } => Ok(Value::Array(
            store
                .list_server_run_events(&run_id, after_seq, limit)?
                .into_iter()
                .map(server_run_event_json)
                .collect::<Result<_, _>>()?,
        )),
        SqliteStoreOperation::UpdateServerRun {
            run_id,
            status,
            assistant_turn_node_id,
            result_json,
            error_json,
            started_at_ms,
            finished_at_ms,
            cancel_requested,
        } => {
            store.update_server_run(
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
            )?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::RequestServerRunCancel { run_id } => {
            store.request_server_run_cancel(&run_id)?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::ApplyRecordedRuntimeTransition {
            transition,
            abort_after_writes,
        } => serde_json::to_value(
            store.apply_recorded_runtime_transition(*transition, abort_after_writes)?,
        )
        .map_err(|error| SqliteStoreError::TransactionAborted(error.to_string())),
        SqliteStoreOperation::ApplyClaimedRecordedRuntimeTransition {
            handoff,
            transition,
            abort_after_writes,
        } => serde_json::to_value(store.apply_claimed_recorded_runtime_transition(
            handoff,
            *transition,
            abort_after_writes,
        )?)
        .map_err(|error| SqliteStoreError::TransactionAborted(error.to_string())),
        SqliteStoreOperation::ReadRecordedRuntimeState {
            run_id,
            workflow_id,
            conversation_id,
        } => serde_json::to_value(store.read_recorded_runtime_state(
            &run_id,
            &workflow_id,
            &conversation_id,
        )?)
        .map_err(|error| SqliteStoreError::TransactionAborted(error.to_string())),
        SqliteStoreOperation::GetIndexAppliedFingerprint {
            namespace,
            coalesce_key,
        } => Ok(json!(
            store.index_applied_fingerprint(&namespace, &coalesce_key)?
        )),
        SqliteStoreOperation::SetIndexAppliedFingerprint {
            namespace,
            coalesce_key,
            applied_fingerprint,
            last_job_id,
        } => {
            store.set_index_applied_fingerprint(
                &namespace,
                &coalesce_key,
                applied_fingerprint.as_deref(),
                last_job_id.as_deref(),
            )?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::EnqueueIndexJob {
            job_id,
            namespace,
            entity_kind,
            entity_id,
            index_kind,
            op,
            payload_json,
            max_retries,
        } => Ok(json!(store.enqueue_index_job(new_index_job(
            job_id,
            namespace,
            entity_kind,
            entity_id,
            index_kind,
            op,
            payload_json,
            max_retries
        ))?)),
        SqliteStoreOperation::ClaimIndexJobs {
            limit,
            lease_seconds,
            namespace,
        } => Ok(Value::Array(
            store
                .claim_index_jobs(limit, lease_seconds, namespace.as_deref())?
                .into_iter()
                .map(index_job_json)
                .collect(),
        )),
        SqliteStoreOperation::MarkIndexJobDone {
            job_id,
            claim_token,
        } => Ok(json!(
            store.mark_index_job_done(&job_id, claim_token.as_deref())?
        )),
        SqliteStoreOperation::AcceptIndexJobResult {
            job_id,
            claim_token,
            result_json,
            result_sha256,
        } => Ok(accepted_index_job_result_json(
            store.accept_index_job_result(&job_id, &claim_token, &result_json, &result_sha256)?,
        )),
        SqliteStoreOperation::GetIndexJobResult { job_id } => Ok(store
            .index_job_result(&job_id)?
            .map(accepted_index_job_result_json)
            .unwrap_or(Value::Null)),
        SqliteStoreOperation::MarkIndexJobFailed {
            job_id,
            error,
            final_,
            claim_token,
        } => {
            store.mark_index_job_failed(&job_id, &error, final_, claim_token.as_deref())?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::BumpRetryAndRequeue {
            job_id,
            error,
            next_run_at_seconds,
            claim_token,
        } => {
            store.bump_retry_and_requeue(
                &job_id,
                &error,
                next_run_at_seconds,
                claim_token.as_deref(),
            )?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::RenewIndexJobLease {
            job_id,
            claim_token,
            lease_seconds,
        } => Ok(json!(store.renew_index_job_lease(
            &job_id,
            &claim_token,
            lease_seconds
        )?)),
        SqliteStoreOperation::RequeueIndexJobAtTail {
            job_id,
            payload_json,
            delay_seconds,
            claim_token,
        } => {
            store.requeue_index_job_at_tail(
                &job_id,
                payload_json,
                delay_seconds,
                claim_token.as_deref(),
            )?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::ListIndexJobs {
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
                )?
                .into_iter()
                .map(index_job_json)
                .collect(),
        )),
        SqliteStoreOperation::ProjectLaneMessage {
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
            store.project_lane_message(new_lane_message(
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
            ))?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::GetProjectedLaneMessage { message_id } => Ok(store
            .get_projected_lane_message(&message_id)?
            .map(lane_message_json)
            .unwrap_or(Value::Null)),
        SqliteStoreOperation::UpdateProjectedLaneMessageStatus {
            message_id,
            status,
            error_json,
        } => {
            store.update_projected_lane_message_status(&message_id, &status, error_json)?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::UpdateProjectedLaneMessageLinks {
            message_id,
            prev_message_id,
            next_message_id,
            inbox_tail_message_id,
            conversation_tail_message_id,
        } => {
            store.update_projected_lane_message_links(
                &message_id,
                prev_message_id,
                next_message_id,
                inbox_tail_message_id,
                conversation_tail_message_id,
            )?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::ListProjectedLaneMessages {
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
            let rows = store.list_projected_lane_messages(lane_filter(
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
            ))?;
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
        SqliteStoreOperation::ClearProjectedLaneMessages { namespace } => {
            Ok(json!(store.clear_projected_lane_messages(&namespace)?))
        }
        SqliteStoreOperation::ClaimProjectedLaneMessages {
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
                ))?
                .into_iter()
                .map(lane_message_json)
                .collect(),
        )),
        SqliteStoreOperation::AckProjectedLaneMessage {
            message_id,
            claimed_by,
        } => {
            store.ack_projected_lane_message(&message_id, &claimed_by)?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::RequeueProjectedLaneMessage {
            message_id,
            claimed_by,
            error_json,
            delay_seconds,
        } => {
            store.requeue_projected_lane_message(
                &message_id,
                &claimed_by,
                error_json,
                delay_seconds,
            )?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::DeadLetterProjectedLaneMessage {
            message_id,
            claimed_by,
            error_json,
        } => {
            store.dead_letter_projected_lane_message(&message_id, &claimed_by, error_json)?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::Batch { operations, abort } => store.immediate_transaction(|uow| {
            let mut results = Vec::with_capacity(operations.len());
            for operation in operations {
                results.push(sqlite_batch_operation_json(uow, operation)?);
            }
            if abort {
                return Err(SqliteStoreError::TransactionAborted(
                    "requested by SQLite JSON ABI abort flag".to_owned(),
                ));
            }
            Ok(Value::Array(results))
        }),
    }
}

fn sqlite_batch_operation_json(
    uow: &mut SqliteUnitOfWork<'_>,
    operation: SqliteStoreOperation,
) -> Result<Value, SqliteStoreError> {
    match operation {
        SqliteStoreOperation::NextGlobalSeq => Ok(json!(uow.next_global_seq()?)),
        SqliteStoreOperation::NextUserSeq { user_id } => Ok(json!(uow.next_user_seq(&user_id)?)),
        SqliteStoreOperation::SetUserSeq { user_id, value } => {
            uow.set_user_seq(&user_id, value)?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::NextScopedSeq { scope_id } => {
            Ok(json!(uow.next_scoped_seq(&scope_id)?))
        }
        SqliteStoreOperation::SetScopedSeq { scope_id, value } => {
            uow.set_scoped_seq(&scope_id, value)?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::AllocEventSeq { namespace } => {
            Ok(json!(uow.alloc_event_seq(&namespace)?))
        }
        SqliteStoreOperation::RawAppend {
            namespace,
            event_id,
            entity_kind,
            entity_id,
            op,
            payload_json,
        } => Ok(appended_raw_event_json(uow.append_raw_entity_event(
            &namespace,
            new_raw_event(event_id, entity_kind, entity_id, op, payload_json),
        )?)),
        SqliteStoreOperation::RawRestore {
            namespace,
            seq,
            event_id,
            entity_kind,
            entity_id,
            op,
            payload_json,
            created_at,
        } => Ok(appended_raw_event_json(
            uow.append_restored_raw_entity_event(
                &namespace,
                RestoredRawEntityEvent {
                    seq,
                    event_id,
                    entity_kind,
                    entity_id,
                    op,
                    payload_json,
                    created_at,
                },
            )?,
        )),
        SqliteStoreOperation::PruneEntityEventsAfter { namespace, to_seq } => {
            Ok(json!(uow.prune_entity_events_after(&namespace, to_seq)?))
        }
        SqliteStoreOperation::LegacyCursorSet {
            namespace,
            consumer,
            last_seq,
        } => Ok(cursor_json(
            uow.set_replay_cursor_legacy(&namespace, &consumer, last_seq)?,
        )),
        SqliteStoreOperation::StrictCursorAdvance {
            namespace,
            consumer,
            last_seq,
        } => Ok(cursor_json(uow.strict_advance_replay_cursor(
            &namespace, &consumer, last_seq,
        )?)),
        SqliteStoreOperation::ReplaceNamedProjection {
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
            )?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::CompareAndSwapNamedProjection {
            namespace,
            key,
            payload,
            expected_last_authoritative_seq,
            expected_last_materialized_seq,
            last_authoritative_seq,
            last_materialized_seq,
            projection_schema_version,
            materialization_status,
        } => Ok(json!(uow.compare_and_swap_named_projection(
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
        )?)),
        SqliteStoreOperation::CompareAndSwapNamedProjections { updates } => {
            let updates = updates
                .into_iter()
                .map(|update| NamedProjectionCas {
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
            Ok(json!(uow.compare_and_swap_named_projections(updates)?))
        }
        SqliteStoreOperation::ClearNamedProjection { namespace, key } => {
            uow.clear_named_projection(&namespace, &key)?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::ClearProjectionNamespace { namespace } => {
            uow.clear_projection_namespace(&namespace)?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::ReplaceStage1NodeProjection {
            namespace,
            key,
            payload,
            last_authoritative_seq,
            last_materialized_seq,
            projection_schema_version,
            materialization_status,
        } => {
            uow.replace_stage1_node_projection(
                &namespace,
                &key,
                projection_write(
                    payload,
                    last_authoritative_seq,
                    last_materialized_seq,
                    projection_schema_version,
                    materialization_status,
                ),
            )?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::ClearStage1NodeProjection { namespace, key } => {
            uow.clear_stage1_node_projection(&namespace, &key)?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::ClearStage1NodeProjectionIfCurrent {
            namespace,
            key,
            expected_last_authoritative_seq,
        } => Ok(json!(uow.clear_stage1_node_projection_if_current(
            &namespace,
            &key,
            expected_last_authoritative_seq,
        )?)),
        SqliteStoreOperation::PutWorkflowDesignSnapshot {
            workflow_id,
            version,
            seq,
            payload_json,
            schema_version,
        } => {
            uow.put_workflow_design_snapshot(
                &workflow_id,
                workflow_design_snapshot_write(version, seq, payload_json, schema_version),
            )?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::ClearWorkflowDesignSnapshots { workflow_id } => {
            uow.clear_workflow_design_snapshots(&workflow_id)?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::PutWorkflowDesignDelta {
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
            )?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::ClearWorkflowDesignDeltas { workflow_id } => {
            uow.clear_workflow_design_deltas(&workflow_id)?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::CreateServerRun {
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
            })?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::AppendServerRunEvent {
            run_id,
            event_type,
            payload_json,
        } => Ok(server_run_event_json(uow.append_server_run_event(
            &run_id,
            &event_type,
            payload_json,
        )?)?),
        SqliteStoreOperation::UpdateServerRun {
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
            )?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::RequestServerRunCancel { run_id } => {
            uow.request_server_run_cancel(&run_id)?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::SetIndexAppliedFingerprint {
            namespace,
            coalesce_key,
            applied_fingerprint,
            last_job_id,
        } => {
            uow.set_index_applied_fingerprint(
                &namespace,
                &coalesce_key,
                applied_fingerprint.as_deref(),
                last_job_id.as_deref(),
            )?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::EnqueueIndexJob {
            job_id,
            namespace,
            entity_kind,
            entity_id,
            index_kind,
            op,
            payload_json,
            max_retries,
        } => Ok(json!(uow.enqueue_index_job(new_index_job(
            job_id,
            namespace,
            entity_kind,
            entity_id,
            index_kind,
            op,
            payload_json,
            max_retries
        ))?)),
        SqliteStoreOperation::ClaimIndexJobs {
            limit,
            lease_seconds,
            namespace,
        } => Ok(Value::Array(
            uow.claim_index_jobs(limit, lease_seconds, namespace.as_deref())?
                .into_iter()
                .map(index_job_json)
                .collect(),
        )),
        SqliteStoreOperation::MarkIndexJobDone {
            job_id,
            claim_token,
        } => Ok(json!(
            uow.mark_index_job_done(&job_id, claim_token.as_deref())?
        )),
        SqliteStoreOperation::AcceptIndexJobResult {
            job_id,
            claim_token,
            result_json,
            result_sha256,
        } => Ok(accepted_index_job_result_json(
            uow.accept_index_job_result(&job_id, &claim_token, &result_json, &result_sha256)?,
        )),
        SqliteStoreOperation::GetIndexJobResult { .. } => {
            Err(SqliteStoreError::TransactionAborted(
                "get_index_job_result is not available inside a transaction".to_owned(),
            ))
        }
        SqliteStoreOperation::MarkIndexJobFailed {
            job_id,
            error,
            final_,
            claim_token,
        } => {
            uow.mark_index_job_failed(&job_id, &error, final_, claim_token.as_deref())?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::BumpRetryAndRequeue {
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
            )?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::RenewIndexJobLease {
            job_id,
            claim_token,
            lease_seconds,
        } => Ok(json!(uow.renew_index_job_lease(
            &job_id,
            &claim_token,
            lease_seconds
        )?)),
        SqliteStoreOperation::RequeueIndexJobAtTail {
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
            )?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::ProjectLaneMessage {
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
            ))?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::UpdateProjectedLaneMessageStatus {
            message_id,
            status,
            error_json,
        } => {
            uow.update_projected_lane_message_status(&message_id, &status, error_json)?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::UpdateProjectedLaneMessageLinks {
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
            )?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::ClearProjectedLaneMessages { namespace } => {
            Ok(json!(uow.clear_projected_lane_messages(&namespace)?))
        }
        SqliteStoreOperation::ClaimProjectedLaneMessages {
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
            ))?
            .into_iter()
            .map(lane_message_json)
            .collect(),
        )),
        SqliteStoreOperation::AckProjectedLaneMessage {
            message_id,
            claimed_by,
        } => {
            uow.ack_projected_lane_message(&message_id, &claimed_by)?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::RequeueProjectedLaneMessage {
            message_id,
            claimed_by,
            error_json,
            delay_seconds,
        } => {
            uow.requeue_projected_lane_message(
                &message_id,
                &claimed_by,
                error_json,
                delay_seconds,
            )?;
            Ok(Value::Null)
        }
        SqliteStoreOperation::DeadLetterProjectedLaneMessage {
            message_id,
            claimed_by,
            error_json,
        } => {
            uow.dead_letter_projected_lane_message(&message_id, &claimed_by, error_json)?;
            Ok(Value::Null)
        }
        _ => Err(SqliteStoreError::TransactionAborted(
            "unsupported SQLite batch operation".to_owned(),
        )),
    }
}

fn sqlite_store_json_impl(payload_json: &str) -> Result<String, (&'static str, String)> {
    let value: Value = serde_json::from_str(payload_json)
        .map_err(|error| (STORE_INVALID_JSON, format!("invalid JSON: {error}")))?;
    validate_sqlite_operation(value.get("operation").ok_or_else(|| {
        (
            STORE_INVALID_PAYLOAD,
            "invalid SQLite store payload: missing operation".to_owned(),
        )
    })?)?;
    let request: SqliteStoreRequest = serde_json::from_value(value).map_err(|error| {
        (
            STORE_INVALID_PAYLOAD,
            format!("invalid SQLite store payload: {error}"),
        )
    })?;
    let context_id = request
        .context_id
        .clone()
        .unwrap_or_else(|| "rust-internal".to_owned());
    if matches!(&request.operation, SqliteStoreOperation::Close) {
        close_cached_sqlite_store(&request.path, &context_id)?;
        return Ok("null".to_owned());
    }
    let result: Result<Value, SqliteStoreError> = (|| match request.operation {
        SqliteStoreOperation::BeginTransaction => {
            let entry = cached_sqlite_store(&request.path, &context_id)
                .map_err(sqlite_cached_store_error)?;
            let mut entry = entry
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let transaction_id = request.transaction_id.ok_or_else(|| {
                SqliteStoreError::TransactionAborted(
                    "SQLite begin requires transaction_id".to_owned(),
                )
            })?;
            if entry.transaction_id.is_some() {
                return Err(SqliteStoreError::TransactionAborted(
                    "SQLite external transaction is already active".to_owned(),
                ));
            }
            entry.store.begin_external_transaction()?;
            entry.transaction_id = Some(transaction_id);
            Ok(Value::Null)
        }
        SqliteStoreOperation::CommitTransaction => {
            let entry = cached_sqlite_store(&request.path, &context_id)
                .map_err(sqlite_cached_store_error)?;
            let result = (|| {
                let mut entry = entry
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                require_sqlite_transaction(&entry, request.transaction_id.as_deref())?;
                entry.store.commit_external_transaction()?;
                entry.transaction_id = None;
                Ok::<Value, SqliteStoreError>(Value::Null)
            })();
            if result.is_ok() && !request.reuse_session {
                close_cached_sqlite_store(&request.path, &context_id)
                    .map_err(|(_, message)| SqliteStoreError::TransactionAborted(message))?;
            }
            result
        }
        SqliteStoreOperation::RollbackTransaction => {
            let entry = cached_sqlite_store(&request.path, &context_id)
                .map_err(sqlite_cached_store_error)?;
            let result = (|| {
                let mut entry = entry
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                require_sqlite_transaction(&entry, request.transaction_id.as_deref())?;
                entry.store.rollback_external_transaction()?;
                entry.transaction_id = None;
                Ok::<Value, SqliteStoreError>(Value::Null)
            })();
            if result.is_ok() && !request.reuse_session {
                close_cached_sqlite_store(&request.path, &context_id)
                    .map_err(|(_, message)| SqliteStoreError::TransactionAborted(message))?;
            }
            result
        }
        operation => {
            if request.reuse_session {
                let entry = cached_sqlite_store(&request.path, &context_id)
                    .map_err(sqlite_cached_store_error)?;
                let entry = entry
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                match (&entry.transaction_id, request.transaction_id.as_deref()) {
                    (None, None) | (Some(_), Some(_)) => {
                        require_sqlite_session_owner(&entry, request.transaction_id.as_deref())?;
                        return sqlite_store_operation_json(&entry.store, operation);
                    }
                    (Some(_), _) => {
                        return Err(SqliteStoreError::TransactionAborted(
                            "SQLite operation does not own the active transaction".to_owned(),
                        ));
                    }
                    (None, Some(_)) => {
                        return Err(SqliteStoreError::TransactionAborted(
                            "SQLite transaction_id is stale".to_owned(),
                        ));
                    }
                }
            }
            if let Some(entry) = existing_cached_sqlite_store(&request.path, &context_id) {
                let entry = entry
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                if entry.context_id != context_id {
                    return Err(SqliteStoreError::ContextMismatch);
                }
                match (&entry.transaction_id, request.transaction_id.as_deref()) {
                    (Some(active), Some(requested)) if active == requested => {
                        return sqlite_store_operation_json(&entry.store, operation);
                    }
                    (Some(_), _) => {
                        return Err(SqliteStoreError::TransactionAborted(
                            "SQLite operation does not own the active transaction".to_owned(),
                        ));
                    }
                    (None, Some(_)) => {
                        return Err(SqliteStoreError::TransactionAborted(
                            "SQLite transaction_id is stale".to_owned(),
                        ));
                    }
                    (None, None) => {}
                }
            } else if request.transaction_id.is_some() {
                if any_cached_sqlite_store(&request.path).is_some() {
                    return Err(SqliteStoreError::ContextMismatch);
                }
                return Err(SqliteStoreError::TransactionAborted(
                    "SQLite transaction_id is stale".to_owned(),
                ));
            }

            // This bridge shares database files with Python EngineSQLite.
            // Calls outside an explicit Rust transaction use no persistent
            // native session, so every call sees Python's latest commit.
            let fresh = SqliteStore::open(&request.path)?;
            sqlite_store_operation_json(&fresh, operation)
        }
    })();
    result
        .and_then(|value| {
            serde_json::to_string(&value).map_err(|error| {
                SqliteStoreError::TransactionAborted(format!("cannot encode result: {error}"))
            })
        })
        .map_err(|error| {
            let code = match error {
                SqliteStoreError::EventIdNamespaceCollision { .. } => {
                    STORE_EVENT_ID_NAMESPACE_COLLISION
                }
                SqliteStoreError::NegativeSequenceValue { .. } => STORE_INVALID_SEQUENCE_VALUE,
                SqliteStoreError::Store(kogwistar_store::StoreError::CursorOutOfRange {
                    ..
                }) => STORE_CURSOR_OUT_OF_RANGE,
                SqliteStoreError::Store(kogwistar_store::StoreError::CursorRegresses {
                    ..
                }) => STORE_CURSOR_REGRESSES,
                SqliteStoreError::Store(
                    kogwistar_store::StoreError::InvalidIdentifier { .. }
                    | kogwistar_store::StoreError::InvalidRecoveryBatchLimit
                    | kogwistar_store::StoreError::InvalidEntityEventPayload { .. }
                    | kogwistar_store::StoreError::UnsupportedEntityEventOperation { .. },
                ) => STORE_INVALID_ENTITY_EVENT,
                SqliteStoreError::TransactionAborted(ref message)
                    if message == "unsupported SQLite batch operation" =>
                {
                    STORE_OPERATION_INVALID
                }
                SqliteStoreError::ContextMismatch => STORE_SQLITE_ENGINE_MISMATCH,
                SqliteStoreError::TransactionAborted(_) => STORE_TRANSACTION_ABORTED,
                _ => STORE_PERSISTENCE_FAILED,
            };
            (code, error.to_string())
        })
}

fn require_sqlite_transaction(
    entry: &CachedSqliteStore,
    requested: Option<&str>,
) -> Result<(), SqliteStoreError> {
    match (&entry.transaction_id, requested) {
        (Some(active), Some(requested)) if active == requested => Ok(()),
        _ => Err(SqliteStoreError::TransactionAborted(
            "SQLite transaction control does not own the active transaction".to_owned(),
        )),
    }
}

fn require_sqlite_session_owner(
    entry: &CachedSqliteStore,
    requested: Option<&str>,
) -> Result<(), SqliteStoreError> {
    match (&entry.transaction_id, requested) {
        (None, None) => Ok(()),
        (Some(active), Some(requested)) if active == requested => Ok(()),
        _ => Err(SqliteStoreError::TransactionAborted(
            "SQLite operation does not own the active transaction".to_owned(),
        )),
    }
}

/// Cache handles only for explicit Rust transaction sessions. Ordinary bridge
/// calls deliberately use one fresh handle per call, so Python commits cannot
/// be hidden behind a native session snapshot.
struct CachedSqliteStore {
    store: SqliteStore,
    transaction_id: Option<String>,
    context_id: String,
}

fn sqlite_cached_store_error((code, message): (&'static str, String)) -> SqliteStoreError {
    if code == STORE_SQLITE_ENGINE_MISMATCH {
        SqliteStoreError::ContextMismatch
    } else {
        SqliteStoreError::TransactionAborted(message)
    }
}

type SharedCachedSqliteStore = std::sync::Arc<Mutex<CachedSqliteStore>>;

fn existing_cached_sqlite_store(path: &str, context_id: &str) -> Option<SharedCachedSqliteStore> {
    let key = (PathBuf::from(path), context_id.to_owned());
    sqlite_store_cache()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .get(&key)
        .cloned()
}

fn any_cached_sqlite_store(path: &str) -> Option<SharedCachedSqliteStore> {
    let path = PathBuf::from(path);
    sqlite_store_cache()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .iter()
        .find(|((cached_path, _), _)| *cached_path == path)
        .map(|(_, store)| store.clone())
}

fn cached_sqlite_store(
    path: &str,
    context_id: &str,
) -> Result<SharedCachedSqliteStore, (&'static str, String)> {
    let path = PathBuf::from(path);
    let key = (path.clone(), context_id.to_owned());
    let stores = sqlite_store_cache();
    let mut stores = stores
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    // A caller may remove a temporary database between calls. Evict the old
    // handle and initialize the recreated file.
    if path.exists()
        && let Some(store) = stores.get(&key)
    {
        let entry = store
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if entry.context_id != context_id {
            return Err((
                STORE_SQLITE_ENGINE_MISMATCH,
                "SQLite cached session belongs to another execution context".to_owned(),
            ));
        }
        drop(entry);
        return Ok(store.clone());
    }

    let store =
        SqliteStore::open(&path).map_err(|error| (STORE_PERSISTENCE_FAILED, error.to_string()))?;
    let entry = std::sync::Arc::new(Mutex::new(CachedSqliteStore {
        store,
        transaction_id: None,
        context_id: context_id.to_owned(),
    }));
    stores.insert(key, entry.clone());
    Ok(entry)
}

fn sqlite_store_cache() -> &'static Mutex<BTreeMap<(PathBuf, String), SharedCachedSqliteStore>> {
    static STORES: OnceLock<Mutex<BTreeMap<(PathBuf, String), SharedCachedSqliteStore>>> =
        OnceLock::new();
    STORES.get_or_init(|| Mutex::new(BTreeMap::new()))
}

fn close_cached_sqlite_store(path: &str, context_id: &str) -> Result<(), (&'static str, String)> {
    let key = (PathBuf::from(path), context_id.to_owned());
    let stores = sqlite_store_cache();
    let entry = stores
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .get(&key)
        .cloned();
    if let Some(entry) = entry {
        let entry = entry
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if entry.transaction_id.is_some() {
            return Err((
                STORE_INVALID_PAYLOAD,
                "cannot close SQLite store while transaction is active".to_owned(),
            ));
        }
    }
    stores
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .remove(&key);
    Ok(())
}

fn validate_sqlite_operation(value: &Value) -> Result<(), (&'static str, String)> {
    let object = value.as_object().ok_or_else(|| {
        (
            STORE_INVALID_PAYLOAD,
            "invalid SQLite store payload: operation must be an object".to_owned(),
        )
    })?;
    if let Some(kind) = object.get("kind").and_then(Value::as_str)
        && kind == "batch"
        && let Some(ops) = object.get("operations").and_then(Value::as_array)
    {
        for op in ops {
            validate_sqlite_operation(op)?;
        }
    }
    Ok(())
}
