use super::*;

pub(crate) fn lane_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<ProjectedLaneMessage> {
    Ok(ProjectedLaneMessage {
        message_id: row.get(0)?,
        namespace: row.get(1)?,
        purpose: row.get(2)?,
        inbox_id: row.get(3)?,
        conversation_id: row.get(4)?,
        recipient_id: row.get(5)?,
        sender_id: row.get(6)?,
        msg_type: row.get(7)?,
        status: row.get(8)?,
        seq: row.get(9)?,
        conversation_seq: row.get(10)?,
        claimed_by: row.get(11)?,
        lease_until: row.get::<_, Option<i64>>(12)?.map(serde_json::Value::from),
        retry_count: row.get(13)?,
        created_at: row.get(14)?,
        available_at: row.get(15)?,
        run_id: row.get(16)?,
        step_id: row.get(17)?,
        correlation_id: row.get(18)?,
        payload_json: row.get(19)?,
        error_json: row.get(20)?,
        prev_message_id: row.get(21)?,
        next_message_id: row.get(22)?,
        inbox_tail_message_id: row.get(23)?,
        conversation_tail_message_id: row.get(24)?,
    })
}
pub(crate) fn projected_lane_message(
    conn: &Connection,
    id: &str,
) -> SqliteStoreResult<Option<ProjectedLaneMessage>> {
    conn.query_row(
        &format!("SELECT {LANE_SELECT} FROM projected_lane_messages WHERE message_id=?1"),
        [id],
        lane_from_row,
    )
    .optional()
    .map_err(Into::into)
}
pub(crate) fn project_lane_message(
    conn: &Connection,
    row: NewProjectedLaneMessage,
) -> SqliteStoreResult<()> {
    require_queue_namespace(&row.namespace)?;
    let seq:i64=conn.query_row("SELECT COALESCE(MAX(seq),0)+1 FROM projected_lane_messages WHERE namespace=?1 AND inbox_id=?2",params![row.namespace,row.inbox_id],|r|r.get(0))?;
    let conversation_seq:i64=conn.query_row("SELECT COALESCE(MAX(conversation_seq),0)+1 FROM projected_lane_messages WHERE namespace=?1 AND conversation_id=?2",params![row.namespace,row.conversation_id],|r|r.get(0))?;
    let prev:Option<String>=conn.query_row("SELECT message_id FROM projected_lane_messages WHERE namespace=?1 AND inbox_id=?2 ORDER BY seq DESC,created_at DESC LIMIT 1",params![row.namespace,row.inbox_id],|r|r.get(0)).optional()?;
    conn.execute("INSERT OR IGNORE INTO projected_lane_messages(message_id,namespace,purpose,inbox_id,conversation_id,recipient_id,sender_id,msg_type,status,seq,conversation_seq,claimed_by,lease_until,retry_count,created_at,available_at,run_id,step_id,correlation_id,payload_json,error_json,prev_message_id,next_message_id,inbox_tail_message_id,conversation_tail_message_id) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,NULL,NULL,0,?12,?13,?14,?15,?16,?17,?18,?19,NULL,?1,?1)",params![row.message_id,row.namespace,if row.purpose.is_empty(){"user_visible".to_owned()}else{row.purpose},row.inbox_id,row.conversation_id,row.recipient_id,row.sender_id,row.msg_type,row.status,seq,conversation_seq,row.created_at,row.available_at,row.run_id,row.step_id,row.correlation_id,row.payload_json,row.error_json,prev])?;
    Ok(())
}
pub(crate) fn update_projected_lane_message_status(
    conn: &Connection,
    id: &str,
    status: &str,
    error: Option<String>,
) -> SqliteStoreResult<()> {
    conn.execute("UPDATE projected_lane_messages SET status=?1,error_json=COALESCE(?2,error_json),claimed_by=CASE WHEN ?1 IN ('completed','failed','cancelled') THEN NULL ELSE claimed_by END,lease_until=CASE WHEN ?1 IN ('completed','failed','cancelled') THEN NULL ELSE lease_until END WHERE message_id=?3",params![status,error,id])?;
    Ok(())
}
pub(crate) fn update_projected_lane_message_links(
    conn: &Connection,
    id: &str,
    prev: Option<String>,
    next: Option<String>,
    inbox_tail: Option<String>,
    conversation_tail: Option<String>,
) -> SqliteStoreResult<()> {
    conn.execute("UPDATE projected_lane_messages SET prev_message_id=?1,next_message_id=?2,inbox_tail_message_id=?3,conversation_tail_message_id=?4 WHERE message_id=?5",params![prev,next,inbox_tail,conversation_tail,id])?;
    Ok(())
}
pub(crate) fn clear_projected_lane_messages(
    conn: &Connection,
    namespace: &str,
) -> SqliteStoreResult<u64> {
    Ok(conn.execute(
        "DELETE FROM projected_lane_messages WHERE namespace=?1",
        [namespace],
    )? as u64)
}
