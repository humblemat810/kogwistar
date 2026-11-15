import pytest

pytest.importorskip("sqlalchemy")

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql as psql

from kogwistar.engine_core.postgres_backend import where_jsonb

pytestmark = pytest.mark.ci_full


def test_where_jsonb_typed_roundtrip(sa_engine, pg_schema):
    """
    End-to-end regression test for typed JSONB metadata.

    Verifies:
    - numeric JSON values are compared numerically (not as text)
    - boolean JSON values are compared as booleans
    - where_jsonb works against a real Postgres execution, not just SQL compilation
    """
    if sa_engine is None or pg_schema is None:
        pytest.skip("PostgreSQL fixtures are unavailable")
    md = sa.MetaData(schema=pg_schema)

    t = sa.Table(
        "jsonb_roundtrip",
        md,
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("metadata", psql.JSONB, nullable=False),
    )

    md.create_all(sa_engine)

    with sa_engine.begin() as conn:
        conn.execute(
            sa.insert(t),
            [
                {
                    "id": "A",
                    "metadata": {"turn_index": 0, "in_conversation_chain": True},
                },
                {
                    "id": "B",
                    "metadata": {"turn_index": 1, "in_conversation_chain": True},
                },
                {
                    "id": "C",
                    "metadata": {"turn_index": 2, "in_conversation_chain": False},
                },
                {"id": "D", "metadata": {"in_conversation_chain": True}},
                {"id": "E", "metadata": {"entity_type": "acl_record"}},
                {"id": "F", "metadata": {"entity_type": None}},
            ],
        )

        clause = where_jsonb(
            t.c.metadata,
            {
                "$and": [
                    {"in_conversation_chain": True},
                    {"turn_index": {"$gte": 1}},
                ]
            },
            numeric_keys={"turn_index"},
        )

        q = sa.select(t.c.id).where(clause).order_by(t.c.id)
        got = [r[0] for r in conn.execute(q).fetchall()]

        not_acl_record = where_jsonb(
            t.c.metadata, {"entity_type": {"$ne": "acl_record"}}
        )
        non_acl_ids = [
            r[0]
            for r in conn.execute(
                sa.select(t.c.id).where(not_acl_record).order_by(t.c.id)
            ).fetchall()
        ]
        not_null_ids = [
            r[0]
            for r in conn.execute(
                sa.select(t.c.id)
                .where(where_jsonb(t.c.metadata, {"entity_type": {"$ne": None}}))
                .order_by(t.c.id)
            ).fetchall()
        ]

    assert got == ["B"]
    assert non_acl_ids == ["A", "B", "C", "D", "F"]
    assert not_null_ids == ["A", "B", "C", "D", "E"]
