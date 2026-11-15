use super::projection::lane_from_row;
use super::*;

pub(crate) async fn claim_projected_lane_messages_filtered<C>(
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

pub(crate) async fn claim_projected_lane_messages<C>(
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
pub(crate) async fn claim_projected_lane_messages_for_run<C>(
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
