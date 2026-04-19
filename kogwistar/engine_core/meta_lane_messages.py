from __future__ import annotations

import time
from dataclasses import replace

from ..messaging.models import ProjectedLaneMessageRow


class LaneMessageMetaStoreMixin:
    """Shared lane-message metastore behavior.

    The public lane-message API belongs to the metastore abstraction layer.
    Concrete stores only provide row fetch/insert/update primitives.
    """

    def _lane_message_get_row(self, *, message_id: str) -> ProjectedLaneMessageRow | None:
        raise NotImplementedError

    def _lane_message_insert_row(self, *, row: ProjectedLaneMessageRow) -> None:
        raise NotImplementedError

    def _lane_message_update_row(self, *, row: ProjectedLaneMessageRow) -> None:
        raise NotImplementedError

    def _lane_message_list_rows(
        self,
        *,
        namespace: str | None = None,
        inbox_id: str | None = None,
        conversation_id: str | None = None,
        status: str | None = None,
    ) -> list[ProjectedLaneMessageRow]:
        raise NotImplementedError

    @staticmethod
    def _lane_message_now_epoch() -> int:
        return int(time.time())

    def project_lane_message(
        self,
        *,
        message_id: str,
        namespace: str,
        inbox_id: str,
        conversation_id: str,
        recipient_id: str,
        sender_id: str,
        msg_type: str,
        status: str,
        created_at: int,
        available_at: int,
        run_id: str | None,
        step_id: str | None,
        correlation_id: str | None,
        payload_json: str | None = None,
        error_json: str | None = None,
    ) -> None:
        if self._lane_message_get_row(message_id=str(message_id)) is not None:
            return
        inbox_rows = self._lane_message_list_rows(
            namespace=str(namespace),
            inbox_id=str(inbox_id),
        )
        conversation_rows = self._lane_message_list_rows(
            namespace=str(namespace),
            conversation_id=str(conversation_id),
        )
        row = ProjectedLaneMessageRow(
            message_id=str(message_id),
            namespace=str(namespace),
            inbox_id=str(inbox_id),
            conversation_id=str(conversation_id),
            recipient_id=str(recipient_id),
            sender_id=str(sender_id),
            msg_type=str(msg_type),
            status=str(status),
            seq=max((item.seq for item in inbox_rows), default=0) + 1,
            conversation_seq=max((item.conversation_seq for item in conversation_rows), default=0) + 1,
            claimed_by=None,
            lease_until=None,
            retry_count=0,
            created_at=int(created_at),
            available_at=int(available_at),
            run_id=None if run_id is None else str(run_id),
            step_id=None if step_id is None else str(step_id),
            correlation_id=None if correlation_id is None else str(correlation_id),
            payload_json=payload_json,
            error_json=error_json,
        )
        self._lane_message_insert_row(row=row)

    def update_projected_lane_message_status(
        self,
        *,
        message_id: str,
        status: str,
        error_json: str | None = None,
    ) -> None:
        row = self._lane_message_get_row(message_id=str(message_id))
        if row is None:
            return
        terminal = str(status) in {"completed", "failed", "cancelled"}
        self._lane_message_update_row(
            row=replace(
                row,
                status=str(status),
                claimed_by=None if terminal else row.claimed_by,
                lease_until=None if terminal else row.lease_until,
                error_json=row.error_json if error_json is None else str(error_json),
            )
        )

    def claim_projected_lane_messages(
        self,
        *,
        namespace: str = "default",
        inbox_id: str,
        claimed_by: str,
        limit: int = 50,
        lease_seconds: int = 60,
    ) -> list[ProjectedLaneMessageRow]:
        if int(limit) <= 0:
            return []
        now = self._lane_message_now_epoch()
        eligible = []
        for row in self._lane_message_list_rows(namespace=str(namespace), inbox_id=str(inbox_id)):
            if row.status == "pending" and int(row.available_at) <= now:
                eligible.append(row)
            elif row.status == "claimed" and row.lease_until is not None and int(row.lease_until) < now:
                eligible.append(row)
        picked = sorted(eligible, key=lambda item: (item.seq, item.created_at))[: int(limit)]
        out: list[ProjectedLaneMessageRow] = []
        lease_until = now + int(lease_seconds)
        for row in picked:
            updated = replace(
                row,
                status="claimed",
                claimed_by=str(claimed_by),
                lease_until=int(lease_until),
            )
            self._lane_message_update_row(row=updated)
            out.append(updated)
        return out

    def ack_projected_lane_message(self, *, message_id: str, claimed_by: str) -> None:
        row = self._lane_message_get_row(message_id=str(message_id))
        if row is None or row.claimed_by != str(claimed_by):
            return
        self._lane_message_update_row(
            row=replace(
                row,
                status="completed",
                claimed_by=None,
                lease_until=None,
            )
        )

    def requeue_projected_lane_message(
        self,
        *,
        message_id: str,
        claimed_by: str,
        error_json: str | None = None,
        delay_seconds: int = 0,
    ) -> None:
        row = self._lane_message_get_row(message_id=str(message_id))
        if row is None or row.claimed_by != str(claimed_by):
            return
        self._lane_message_update_row(
            row=replace(
                row,
                status="pending",
                claimed_by=None,
                lease_until=None,
                retry_count=int(row.retry_count) + 1,
                available_at=self._lane_message_now_epoch() + max(0, int(delay_seconds)),
                error_json=row.error_json if error_json is None else str(error_json),
            )
        )

    def list_projected_lane_messages(
        self,
        *,
        namespace: str = "default",
        inbox_id: str | None = None,
        status: str | None = None,
    ) -> list[ProjectedLaneMessageRow]:
        rows = self._lane_message_list_rows(
            namespace=str(namespace),
            inbox_id=inbox_id,
            status=status,
        )
        return sorted(rows, key=lambda item: (item.seq, item.created_at))


__all__ = ["LaneMessageMetaStoreMixin"]
