from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

import pytest

from graph_knowledge_engine.engine_core.postgres_backend import where_jsonb

pytestmark = pytest.mark.ci


def _compile(clause: sa.ClauseElement) -> str:
    return str(
        clause.compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    )


def test_where_and_in_and_gte_typed() -> None:
    """Typed metadata parity: numeric comparisons must be typed (cast), not TEXT."""
    md = sa.MetaData()
    t = sa.Table("t", md, sa.Column("metadata", postgresql.JSONB))

    clause = where_jsonb(
        t.c.metadata,
        {
            "$and": [
                {"entity_type": "workflow_node"},  # string
                {"workflow_id": {"$in": ["wf1", "wf2"]}},  # string IN
                {"seq": {"$gte": 10}},  # numeric
            ]
        },
        numeric_keys={"seq"},
    )
    sql = _compile(clause)

    assert "metadata" in sql
    assert "entity_type" in sql
    assert "workflow_id" in sql
    assert "IN" in sql
    assert ">=" in sql

    # Must show typed compare somewhere (CAST or ::). This prevents the classic bug:
    #   (metadata ->> 'seq') >= 10   -- TEXT >= INT (wrong / may error)
    assert ("CAST" in sql.upper()) or ("::" in sql), sql


def test_where_bool_and_numeric_typed() -> None:
    """Regression for conversation filters: bool + numeric in the same where."""
    md = sa.MetaData()
    t = sa.Table("t", md, sa.Column("metadata", postgresql.JSONB))

    clause = where_jsonb(
        t.c.metadata,
        {
            "$and": [
                {"in_conversation_chain": True},  # boolean
                {"turn_index": {"$gte": 1}},  # numeric
            ]
        },
        numeric_keys={"turn_index"},
    )
    sql = _compile(clause)

    assert "in_conversation_chain" in sql
    assert "turn_index" in sql
    assert ">=" in sql

    assert ("CAST" in sql.upper()) or ("::" in sql), sql


def test_where_or_ne_compile() -> None:
    md = sa.MetaData()
    t = sa.Table("t", md, sa.Column("metadata", postgresql.JSONB))

    clause = where_jsonb(
        t.c.metadata,
        {"$or": [{"doc_id": "d1"}, {"doc_id": {"$ne": "d2"}}]},
    )
    sql = _compile(clause)
    assert "OR" in sql
    assert "!=" in sql or "<>" in sql
    assert "doc_id" in sql
