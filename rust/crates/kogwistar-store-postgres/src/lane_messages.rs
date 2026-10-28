use super::*;

pub(super) const LANE_SELECT: &str = "message_id,namespace,purpose,inbox_id,conversation_id,recipient_id,sender_id,msg_type,status,seq,conversation_seq,claimed_by,lease_until::TEXT,retry_count,created_at,available_at,run_id,step_id,correlation_id,payload_json,error_json,prev_message_id,next_message_id,inbox_tail_message_id,conversation_tail_message_id";
pub(super) const LANE_RETURNING: &str = "x.message_id,x.namespace,x.purpose,x.inbox_id,x.conversation_id,x.recipient_id,x.sender_id,x.msg_type,x.status,x.seq,x.conversation_seq,x.claimed_by,x.lease_until::TEXT,x.retry_count,x.created_at,x.available_at,x.run_id,x.step_id,x.correlation_id,x.payload_json,x.error_json,x.prev_message_id,x.next_message_id,x.inbox_tail_message_id,x.conversation_tail_message_id";

pub(super) fn lane_from_row(row: &Row) -> PostgresStoreResult<ProjectedLaneMessage> {
    Ok(ProjectedLaneMessage {
        message_id: row.try_get(0).map_err(backend)?,
        namespace: row.try_get(1).map_err(backend)?,
        purpose: row.try_get(2).map_err(backend)?,
        inbox_id: row.try_get(3).map_err(backend)?,
        conversation_id: row.try_get(4).map_err(backend)?,
        recipient_id: row.try_get(5).map_err(backend)?,
        sender_id: row.try_get(6).map_err(backend)?,
        msg_type: row.try_get(7).map_err(backend)?,
        status: row.try_get(8).map_err(backend)?,
        seq: row.try_get(9).map_err(backend)?,
        conversation_seq: row.try_get(10).map_err(backend)?,
        claimed_by: row.try_get(11).map_err(backend)?,
        lease_until: row
            .try_get::<_, Option<String>>(12)
            .map_err(backend)?
            .map(serde_json::Value::String),
        retry_count: i64::from(row.try_get::<_, i32>(13).map_err(backend)?),
        created_at: row.try_get(14).map_err(backend)?,
        available_at: row.try_get(15).map_err(backend)?,
        run_id: row.try_get(16).map_err(backend)?,
        step_id: row.try_get(17).map_err(backend)?,
        correlation_id: row.try_get(18).map_err(backend)?,
        payload_json: row.try_get(19).map_err(backend)?,
        error_json: row.try_get(20).map_err(backend)?,
        prev_message_id: row.try_get(21).map_err(backend)?,
        next_message_id: row.try_get(22).map_err(backend)?,
        inbox_tail_message_id: row.try_get(23).map_err(backend)?,
        conversation_tail_message_id: row.try_get(24).map_err(backend)?,
    })
}
pub(super) async fn project_lane_message<C>(
    c: &C,
    t: &Tables,
    row: NewProjectedLaneMessage,
) -> PostgresStoreResult<()>
where
    C: GenericClient + Sync,
{
    require_namespace(&row.namespace)?;
    advisory_lock(
        c,
        &format!("lane-inbox\u{1f}{}\u{1f}{}", row.namespace, row.inbox_id),
        31,
    )
    .await?;
    advisory_lock(
        c,
        &format!(
            "lane-conversation\u{1f}{}\u{1f}{}",
            row.namespace, row.conversation_id
        ),
        32,
    )
    .await?;
    c.execute(&format!("INSERT INTO {}(message_id,namespace,purpose,inbox_id,conversation_id,recipient_id,sender_id,msg_type,status,seq,conversation_seq,claimed_by,lease_until,retry_count,created_at,available_at,run_id,step_id,correlation_id,payload_json,error_json,prev_message_id,next_message_id,inbox_tail_message_id,conversation_tail_message_id) SELECT $1,$2,$3,$4,$5,$6,$7,$8,$9,COALESCE((SELECT MAX(seq)+1 FROM {} WHERE namespace=$2 AND inbox_id=$4),1),COALESCE((SELECT MAX(conversation_seq)+1 FROM {} WHERE namespace=$2 AND conversation_id=$5),1),NULL,NULL,0,$10,$11,$12,$13,$14,$15,$16,(SELECT message_id FROM {} WHERE namespace=$2 AND inbox_id=$4 ORDER BY seq DESC,created_at DESC LIMIT 1),NULL,$1,$1 ON CONFLICT(message_id) DO NOTHING",t.projected_lane_messages,t.projected_lane_messages,t.projected_lane_messages,t.projected_lane_messages),&[&row.message_id,&row.namespace,&if row.purpose.is_empty(){"user_visible".to_owned()}else{row.purpose},&row.inbox_id,&row.conversation_id,&row.recipient_id,&row.sender_id,&row.msg_type,&row.status,&row.created_at,&row.available_at,&row.run_id,&row.step_id,&row.correlation_id,&row.payload_json,&row.error_json]).await.map_err(backend)?;
    Ok(())
}
pub(super) async fn projected_lane_message<C>(
    c: &C,
    t: &Tables,
    id: &str,
) -> PostgresStoreResult<Option<ProjectedLaneMessage>>
where
    C: GenericClient + Sync,
{
    c.query_opt(
        &format!(
            "SELECT {LANE_SELECT} FROM {} WHERE message_id=$1",
            t.projected_lane_messages
        ),
        &[&id],
    )
    .await
    .map_err(backend)?
    .map(|r| lane_from_row(&r))
    .transpose()
}
pub(super) async fn update_projected_lane_message_status<C>(
    c: &C,
    t: &Tables,
    id: &str,
    status: &str,
    error: Option<String>,
) -> PostgresStoreResult<()>
where
    C: GenericClient + Sync,
{
    c.execute(&format!("UPDATE {} SET status=$1,error_json=COALESCE($2,error_json),claimed_by=CASE WHEN $1 IN ('completed','failed','cancelled') THEN NULL ELSE claimed_by END,lease_until=CASE WHEN $1 IN ('completed','failed','cancelled') THEN NULL ELSE lease_until END WHERE message_id=$3",t.projected_lane_messages),&[&status,&error,&id]).await.map_err(backend)?;
    Ok(())
}
pub(super) async fn update_projected_lane_message_links<C>(
    c: &C,
    t: &Tables,
    id: &str,
    prev: Option<String>,
    next: Option<String>,
    inbox_tail: Option<String>,
    conversation_tail: Option<String>,
) -> PostgresStoreResult<()>
where
    C: GenericClient + Sync,
{
    c.execute(&format!("UPDATE {} SET prev_message_id=$1,next_message_id=$2,inbox_tail_message_id=$3,conversation_tail_message_id=$4 WHERE message_id=$5",t.projected_lane_messages),&[&prev,&next,&inbox_tail,&conversation_tail,&id]).await.map_err(backend)?;
    Ok(())
}
pub(super) async fn clear_projected_lane_messages<C>(
    c: &C,
    t: &Tables,
    namespace: &str,
) -> PostgresStoreResult<u64>
where
    C: GenericClient + Sync,
{
    c.execute(
        &format!(
            "DELETE FROM {} WHERE namespace=$1",
            t.projected_lane_messages
        ),
        &[&namespace],
    )
    .await
    .map_err(backend)
}
pub(super) async fn claim_projected_lane_messages_filtered<C>(
    c: &C,
    t: &Tables,
    filter: &LaneMessageClaimFilter,
) -> PostgresStoreResult<Vec<ProjectedLaneMessage>>
where
    C: GenericClient + Sync,
{
    if filter.limit == 0 {
        return Ok(vec![]);
    }
    let read_filter = LaneMessageFilter {
        namespace: Some(filter.namespace.clone()),
        purpose: None,
        inbox_id: Some(filter.inbox_id.clone()),
        conversation_id: None,
        status: None,
        msg_type: filter.msg_type.clone(),
        sender_id: None,
        recipient_id: filter.recipient_id.clone(),
        correlation_id: None,
        reply_to_message_id: None,
        created_at_gte: None,
        created_at_lte: None,
        available_at_gte: None,
        available_at_lte: None,
        limit: usize::MAX,
        newest_first: false,
    };
    let allowed_ids = filter
        .message_ids
        .as_ref()
        .map(|ids| ids.iter().collect::<std::collections::HashSet<_>>());
    let mut candidates = list_projected_lane_messages(c, t, &read_filter).await?;
    candidates.retain(|row| {
        allowed_ids
            .as_ref()
            .is_none_or(|ids| ids.contains(&row.message_id))
            && filter
                .run_id
                .as_ref()
                .is_none_or(|value| row.run_id.as_deref() == Some(value))
            && matches!(row.status.as_str(), "pending" | "claimed")
    });
    candidates.sort_by_key(|row| (row.seq, row.created_at, row.message_id.clone()));
    let lease = filter.lease_seconds.to_string();
    let mut claimed = Vec::with_capacity(filter.limit);
    for row in candidates {
        // An active older lease must not consume the caller's claim limit.
        // The conditional UPDATE below rejects it; keep scanning so newer
        // pending work remains claimable while another worker is running.
        let rows = c
            .query(
                &format!(
                    "UPDATE {} AS x SET status='claimed',claimed_by=$1,lease_until=NOW()+($2::TEXT||' seconds')::interval \
                     WHERE message_id=$3 AND namespace=$4 AND inbox_id=$5 \
                     AND ($6::TEXT IS NULL OR run_id=$6) \
                     AND ($7::TEXT IS NULL OR msg_type=$7) \
                     AND ($8::TEXT IS NULL OR recipient_id=$8) \
                     AND ((status='pending' AND available_at<=EXTRACT(EPOCH FROM NOW())::BIGINT) \
                          OR (status='claimed' AND lease_until IS NOT NULL AND lease_until<NOW())) \
                     RETURNING {}",
                    t.projected_lane_messages, LANE_RETURNING
                ),
                &[
                    &filter.claimed_by,
                    &lease,
                    &row.message_id,
                    &filter.namespace,
                    &filter.inbox_id,
                    &filter.run_id,
                    &filter.msg_type,
                    &filter.recipient_id,
                ],
            )
            .await
            .map_err(backend)?;
        if let Some(row) = rows.first() {
            claimed.push(lane_from_row(row)?);
            if claimed.len() == filter.limit {
                break;
            }
        }
    }
    Ok(claimed)
}

pub(super) async fn claim_projected_lane_messages<C>(
    c: &C,
    t: &Tables,
    namespace: &str,
    inbox: &str,
    owner: &str,
    limit: usize,
    lease: i64,
) -> PostgresStoreResult<Vec<ProjectedLaneMessage>>
where
    C: GenericClient + Sync,
{
    claim_projected_lane_messages_filtered(
        c,
        t,
        &LaneMessageClaimFilter {
            namespace: namespace.to_owned(),
            inbox_id: inbox.to_owned(),
            claimed_by: owner.to_owned(),
            message_ids: None,
            run_id: None,
            msg_type: None,
            recipient_id: None,
            limit,
            lease_seconds: lease,
        },
    )
    .await
}
#[allow(clippy::too_many_arguments)]
pub(super) async fn claim_projected_lane_messages_for_run<C>(
    c: &C,
    t: &Tables,
    namespace: &str,
    inbox: &str,
    run_id: &str,
    owner: &str,
    limit: usize,
    lease: i64,
) -> PostgresStoreResult<Vec<ProjectedLaneMessage>>
where
    C: GenericClient + Sync,
{
    claim_projected_lane_messages_filtered(
        c,
        t,
        &LaneMessageClaimFilter {
            namespace: namespace.to_owned(),
            inbox_id: inbox.to_owned(),
            claimed_by: owner.to_owned(),
            message_ids: None,
            run_id: Some(run_id.to_owned()),
            msg_type: None,
            recipient_id: None,
            limit,
            lease_seconds: lease,
        },
    )
    .await
}
pub(super) async fn ack_projected_lane_message<C>(
    c: &C,
    t: &Tables,
    id: &str,
    owner: &str,
) -> PostgresStoreResult<()>
where
    C: GenericClient + Sync,
{
    c.execute(&format!("UPDATE {} SET status='completed',claimed_by=NULL,lease_until=NULL WHERE message_id=$1 AND status NOT IN ('completed','failed','cancelled','dead-letter') AND (claimed_by IS NULL OR (status='claimed' AND claimed_by=$2 AND lease_until>=NOW()))",t.projected_lane_messages),&[&id,&owner]).await.map_err(backend)?;
    Ok(())
}
pub(super) async fn requeue_projected_lane_message<C>(
    c: &C,
    t: &Tables,
    id: &str,
    owner: &str,
    error: Option<String>,
    delay: i64,
) -> PostgresStoreResult<()>
where
    C: GenericClient + Sync,
{
    let delay = delay.max(0).to_string();
    c.execute(&format!("UPDATE {} SET status='pending',claimed_by=NULL,lease_until=NULL,retry_count=retry_count+1,available_at=EXTRACT(EPOCH FROM(NOW()+($1::TEXT||' seconds')::interval))::BIGINT,error_json=COALESCE($2,error_json) WHERE message_id=$3 AND status NOT IN ('completed','failed','cancelled','dead-letter') AND (claimed_by IS NULL OR (status='claimed' AND claimed_by=$4 AND lease_until>=NOW()))",t.projected_lane_messages),&[&delay,&error,&id,&owner]).await.map_err(backend)?;
    Ok(())
}
pub(super) async fn dead_letter_projected_lane_message<C>(
    c: &C,
    t: &Tables,
    id: &str,
    owner: &str,
    error: Option<String>,
) -> PostgresStoreResult<()>
where
    C: GenericClient + Sync,
{
    c.execute(&format!("UPDATE {} SET status='dead-letter',claimed_by=NULL,lease_until=NULL,error_json=COALESCE($1,error_json) WHERE message_id=$2 AND status NOT IN ('completed','failed','cancelled','dead-letter') AND (claimed_by IS NULL OR (status='claimed' AND claimed_by=$3 AND lease_until>=NOW()))",t.projected_lane_messages),&[&error,&id,&owner]).await.map_err(backend)?;
    Ok(())
}
pub(super) async fn repair_orphaned_claimed_lane_messages<C>(
    client: &C,
    tables: &Tables,
    namespace: &str,
    inbox_id: Option<&str>,
    limit: usize,
) -> PostgresStoreResult<Vec<String>>
where
    C: GenericClient + Sync,
{
    if limit == 0 {
        return Ok(Vec::new());
    }
    let limit = i64::try_from(limit).unwrap_or(i64::MAX);
    let rows = client
        .query(
            &format!(
                "SELECT message_id FROM {} WHERE namespace=$1 \
                 AND ($2::TEXT IS NULL OR inbox_id=$2) AND status='claimed' \
                 AND (lease_until IS NULL OR lease_until<=NOW()) \
                 ORDER BY seq ASC, created_at ASC LIMIT $3 FOR UPDATE SKIP LOCKED",
                tables.projected_lane_messages
            ),
            &[&namespace, &inbox_id, &limit],
        )
        .await
        .map_err(backend)?;
    let ids = rows
        .iter()
        .map(|row| row.try_get::<_, String>(0).map_err(backend))
        .collect::<PostgresStoreResult<Vec<_>>>()?;
    for id in &ids {
        client
            .execute(
                &format!(
                    "UPDATE {} SET status='pending', claimed_by=NULL, lease_until=NULL \
                     WHERE message_id=$1 AND status='claimed' \
                     AND (lease_until IS NULL OR lease_until<=NOW())",
                    tables.projected_lane_messages
                ),
                &[id],
            )
            .await
            .map_err(backend)?;
    }
    Ok(ids)
}
pub(super) async fn list_projected_lane_messages<C>(
    c: &C,
    t: &Tables,
    f: &LaneMessageFilter,
) -> PostgresStoreResult<Vec<ProjectedLaneMessage>>
where
    C: GenericClient + Sync,
{
    let limit = i64::try_from(f.limit).unwrap_or(i64::MAX);
    let order = if f.newest_first { "DESC" } else { "ASC" };
    let rows=c.query(&format!("SELECT {LANE_SELECT} FROM {} WHERE ($1::TEXT IS NULL OR namespace=$1) AND ($2::TEXT IS NULL OR purpose=$2) AND ($3::TEXT IS NULL OR inbox_id=$3) AND ($4::TEXT IS NULL OR conversation_id=$4) AND ($5::TEXT IS NULL OR status=$5) AND ($6::TEXT IS NULL OR msg_type=$6) AND ($7::TEXT IS NULL OR sender_id=$7) AND ($8::TEXT IS NULL OR recipient_id=$8) AND ($9::TEXT IS NULL OR correlation_id=$9) AND ($10::BIGINT IS NULL OR created_at>=$10) AND ($11::BIGINT IS NULL OR created_at<=$11) AND ($12::BIGINT IS NULL OR available_at>=$12) AND ($13::BIGINT IS NULL OR available_at<=$13) ORDER BY created_at {order},seq {order},message_id {order} LIMIT $14",t.projected_lane_messages),&[&f.namespace,&f.purpose,&f.inbox_id,&f.conversation_id,&f.status,&f.msg_type,&f.sender_id,&f.recipient_id,&f.correlation_id,&f.created_at_gte,&f.created_at_lte,&f.available_at_gte,&f.available_at_lte,&limit]).await.map_err(backend)?;
    rows.iter()
        .map(lane_from_row)
        .collect::<PostgresStoreResult<Vec<_>>>()
}
