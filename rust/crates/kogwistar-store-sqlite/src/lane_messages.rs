use super::*;

pub(super) fn lane_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<ProjectedLaneMessage> {
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
pub(super) fn projected_lane_message(
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
pub(super) fn project_lane_message(
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
pub(super) fn update_projected_lane_message_status(
    conn: &Connection,
    id: &str,
    status: &str,
    error: Option<String>,
) -> SqliteStoreResult<()> {
    conn.execute("UPDATE projected_lane_messages SET status=?1,error_json=COALESCE(?2,error_json),claimed_by=CASE WHEN ?1 IN ('completed','failed','cancelled') THEN NULL ELSE claimed_by END,lease_until=CASE WHEN ?1 IN ('completed','failed','cancelled') THEN NULL ELSE lease_until END WHERE message_id=?3",params![status,error,id])?;
    Ok(())
}
pub(super) fn update_projected_lane_message_links(
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
pub(super) fn clear_projected_lane_messages(
    conn: &Connection,
    namespace: &str,
) -> SqliteStoreResult<u64> {
    Ok(conn.execute(
        "DELETE FROM projected_lane_messages WHERE namespace=?1",
        [namespace],
    )? as u64)
}
pub(super) fn claim_projected_lane_messages(
    conn: &Connection,
    namespace: &str,
    inbox: &str,
    owner: &str,
    limit: usize,
    lease: i64,
) -> SqliteStoreResult<Vec<ProjectedLaneMessage>> {
    claim_projected_lane_messages_filtered(
        conn,
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
}

pub(super) fn claim_projected_lane_messages_filtered(
    conn: &Connection,
    filter: &LaneMessageClaimFilter,
) -> SqliteStoreResult<Vec<ProjectedLaneMessage>> {
    if filter.limit == 0 {
        return Ok(vec![]);
    }
    let now = unix_epoch_seconds();
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
    let mut rows = list_projected_lane_messages(conn, &read_filter)?;
    rows.retain(|row| {
        allowed_ids
            .as_ref()
            .is_none_or(|ids| ids.contains(&row.message_id))
            && filter
                .run_id
                .as_ref()
                .is_none_or(|value| row.run_id.as_deref() == Some(value))
            && ((row.status == "pending" && row.available_at <= now)
                || (row.status == "claimed"
                    && row
                        .lease_until
                        .as_ref()
                        .and_then(Value::as_i64)
                        .is_some_and(|value| value < now)))
    });
    rows.sort_by_key(|row| (row.seq, row.created_at, row.message_id.clone()));
    rows.truncate(filter.limit);
    let until = now + filter.lease_seconds;
    for row in &mut rows {
        conn.execute("UPDATE projected_lane_messages SET status='claimed',claimed_by=?1,lease_until=?2 WHERE message_id=?3",params![filter.claimed_by,until,row.message_id])?;
        row.status = "claimed".to_owned();
        row.claimed_by = Some(filter.claimed_by.clone());
        row.lease_until = Some(Value::from(until));
    }
    Ok(rows)
}
pub(super) fn claim_projected_lane_messages_for_run(
    conn: &Connection,
    namespace: &str,
    inbox: &str,
    run_id: &str,
    owner: &str,
    limit: usize,
    lease: i64,
) -> SqliteStoreResult<Vec<ProjectedLaneMessage>> {
    if limit == 0 {
        return Ok(vec![]);
    }
    let now = unix_epoch_seconds();
    let until = now + lease;
    let mut stmt=conn.prepare("SELECT message_id FROM projected_lane_messages WHERE namespace=?1 AND inbox_id=?2 AND run_id=?3 AND ((status='pending' AND available_at<=?4) OR (status='claimed' AND lease_until IS NOT NULL AND lease_until<?4)) ORDER BY seq ASC,created_at ASC LIMIT ?5")?;
    let ids = stmt
        .query_map(
            params![
                namespace,
                inbox,
                run_id,
                now,
                i64::try_from(limit).unwrap_or(i64::MAX)
            ],
            |row| row.get::<_, String>(0),
        )?
        .collect::<Result<Vec<_>, _>>()?;
    for id in &ids {
        conn.execute("UPDATE projected_lane_messages SET status='claimed',claimed_by=?1,lease_until=?2 WHERE message_id=?3",params![owner,until,id])?;
    }
    let mut out = Vec::with_capacity(ids.len());
    for id in ids {
        if let Some(row) = projected_lane_message(conn, &id)? {
            out.push(row);
        }
    }
    Ok(out)
}
pub(super) fn ack_projected_lane_message(
    conn: &Connection,
    id: &str,
    owner: &str,
) -> SqliteStoreResult<()> {
    conn.execute("UPDATE projected_lane_messages SET status='completed',claimed_by=NULL,lease_until=NULL WHERE message_id=?1 AND status NOT IN ('completed','failed','cancelled','dead-letter') AND (claimed_by IS NULL OR (status='claimed' AND claimed_by=?2 AND lease_until>=?3))",params![id,owner,unix_epoch_seconds()])?;
    Ok(())
}
pub(super) fn requeue_projected_lane_message(
    conn: &Connection,
    id: &str,
    owner: &str,
    error: Option<String>,
    delay: i64,
) -> SqliteStoreResult<()> {
    let now = unix_epoch_seconds();
    conn.execute("UPDATE projected_lane_messages SET status='pending',claimed_by=NULL,lease_until=NULL,retry_count=retry_count+1,available_at=?1,error_json=COALESCE(?2,error_json) WHERE message_id=?3 AND status NOT IN ('completed','failed','cancelled','dead-letter') AND (claimed_by IS NULL OR (status='claimed' AND claimed_by=?4 AND lease_until>=?5))",params![now+delay.max(0),error,id,owner,now])?;
    Ok(())
}
pub(super) fn dead_letter_projected_lane_message(
    conn: &Connection,
    id: &str,
    owner: &str,
    error: Option<String>,
) -> SqliteStoreResult<()> {
    conn.execute("UPDATE projected_lane_messages SET status='dead-letter',claimed_by=NULL,lease_until=NULL,error_json=COALESCE(?1,error_json) WHERE message_id=?2 AND status NOT IN ('completed','failed','cancelled','dead-letter') AND (claimed_by IS NULL OR (status='claimed' AND claimed_by=?3 AND lease_until>=?4))",params![error,id,owner,unix_epoch_seconds()])?;
    Ok(())
}
pub(super) fn repair_orphaned_claimed_lane_messages(
    conn: &Connection,
    namespace: &str,
    inbox_id: Option<&str>,
    limit: usize,
) -> SqliteStoreResult<Vec<String>> {
    if limit == 0 {
        return Ok(Vec::new());
    }
    let now = unix_epoch_seconds();
    let mut statement = conn.prepare(
        "SELECT message_id FROM projected_lane_messages \
         WHERE namespace=?1 AND (?2 IS NULL OR inbox_id=?2) AND status='claimed' \
         AND (lease_until IS NULL OR lease_until<=?3) \
         ORDER BY seq ASC, created_at ASC LIMIT ?4",
    )?;
    let ids = statement
        .query_map(
            params![
                namespace,
                inbox_id,
                now,
                i64::try_from(limit).unwrap_or(i64::MAX)
            ],
            |row| row.get::<_, String>(0),
        )?
        .collect::<Result<Vec<_>, _>>()?;
    drop(statement);
    for id in &ids {
        conn.execute(
            "UPDATE projected_lane_messages SET status='pending', claimed_by=NULL, \
             lease_until=NULL WHERE message_id=?1 AND status='claimed' \
             AND (lease_until IS NULL OR lease_until<=?2)",
            params![id, now],
        )?;
    }
    Ok(ids)
}
pub(super) fn list_projected_lane_messages(
    conn: &Connection,
    f: &LaneMessageFilter,
) -> SqliteStoreResult<Vec<ProjectedLaneMessage>> {
    let mut rows = Vec::new();
    let mut stmt = conn.prepare(&format!(
        "SELECT {LANE_SELECT} FROM projected_lane_messages"
    ))?;
    for item in stmt.query_map([], lane_from_row)? {
        let row = item?;
        let eq = |x: &Option<String>, v: &str| x.as_ref().is_none_or(|x| x == v);
        if eq(&f.namespace, &row.namespace)
            && eq(&f.purpose, &row.purpose)
            && eq(&f.inbox_id, &row.inbox_id)
            && eq(&f.conversation_id, &row.conversation_id)
            && eq(&f.status, &row.status)
            && eq(&f.msg_type, &row.msg_type)
            && eq(&f.sender_id, &row.sender_id)
            && eq(&f.recipient_id, &row.recipient_id)
            && f.correlation_id
                .as_ref()
                .is_none_or(|x| row.correlation_id.as_ref() == Some(x))
            && f.created_at_gte.is_none_or(|x| row.created_at >= x)
            && f.created_at_lte.is_none_or(|x| row.created_at <= x)
            && f.available_at_gte.is_none_or(|x| row.available_at >= x)
            && f.available_at_lte.is_none_or(|x| row.available_at <= x)
        {
            rows.push(row)
        }
    }
    rows.sort_by(|a, b| {
        (a.created_at, a.seq, &a.message_id).cmp(&(b.created_at, b.seq, &b.message_id))
    });
    if f.newest_first {
        rows.reverse()
    }
    rows.truncate(f.limit);
    Ok(rows)
}
pub(super) fn truncate_error(error: &str) -> String {
    error.chars().take(2000).collect()
}
