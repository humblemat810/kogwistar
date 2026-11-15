use super::*;

pub(crate) fn lane_from_row(row: &Row) -> PostgresStoreResult<ProjectedLaneMessage> {
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
pub(crate) async fn project_lane_message<C>(
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
pub(crate) async fn projected_lane_message<C>(
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
pub(crate) async fn update_projected_lane_message_status<C>(
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
pub(crate) async fn update_projected_lane_message_links<C>(
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
pub(crate) async fn clear_projected_lane_messages<C>(
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
