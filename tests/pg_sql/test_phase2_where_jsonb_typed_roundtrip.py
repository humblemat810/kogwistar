import sqlalchemy as sa
from sqlalchemy.dialects import postgresql as psql
from graph_knowledge_engine.engine_core.postgres_backend import where_jsonb


def test_where_jsonb_typed_roundtrip(sa_engine, pg_schema):
    """
    End-to-end regression test for typed JSONB metadata.

    Verifies:
    - numeric JSON values are compared numerically (not as text)
    - boolean JSON values are compared as booleans
    - where_jsonb works against a real Postgres execution, not just SQL compilation
    """
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

    assert got == ["B"]
