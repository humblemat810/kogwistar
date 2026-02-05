from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from graph_knowledge_engine.postgres_backend import where_jsonb


def _compile(clause: sa.ClauseElement) -> str:
    return str(clause.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}))


def test_where_and_in_and_gte_compile() -> None:
    md = sa.MetaData()
    t = sa.Table("t", md, sa.Column("metadata", postgresql.JSONB))

    clause = where_jsonb(
        t.c.metadata,
        {
            "$and": [
                {"entity_type": "workflow_node"},
                {"workflow_id": {"$in": ["wf1", "wf2"]}},
                {"seq": {"$gte": 10}},
            ]
        },
    )
    sql = _compile(clause)

    # We don't assert exact SQL (dialect can vary), but we do assert key fragments.
    assert "metadata" in sql
    assert "->>" in sql
    assert "workflow_id" in sql
    assert "IN" in sql
    assert ">=" in sql


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
