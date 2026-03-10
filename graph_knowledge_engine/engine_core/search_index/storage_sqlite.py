from __future__ import annotations

import sqlite3


_EXTERNAL_CONTENT_FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS semantic_index_fts
USING fts5(
    index_key UNINDEXED,
    canonical_title,
    provision,
    keywords,
    aliases,
    content='semantic_index',
    content_rowid='id'
);
""".strip()


def _fts_is_contentless(cur: sqlite3.Cursor) -> bool:
    row = cur.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'semantic_index_fts'"
    ).fetchone()
    if not row or not row[0]:
        return False
    sql = str(row[0]).lower()
    return "content=''" in sql or 'content=""' in sql


def _drop_fts_objects(cur: sqlite3.Cursor) -> None:
    cur.executescript(
        """
        DROP TRIGGER IF EXISTS semantic_index_ai;
        DROP TRIGGER IF EXISTS semantic_index_ad;
        DROP TRIGGER IF EXISTS semantic_index_au;
        DROP TABLE IF EXISTS semantic_index_fts;
        """
    )


def rebuild_semantic_index_fts(conn: sqlite3.Connection) -> None:
    conn.execute("INSERT INTO semantic_index_fts(semantic_index_fts) VALUES('rebuild')")
    conn.commit()


def ensure_index_tables(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS semantic_index (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            index_key       TEXT    NOT NULL UNIQUE,
            node_id         TEXT    NOT NULL,
            canonical_title TEXT    NOT NULL,
            keywords        TEXT    NOT NULL DEFAULT '',
            aliases         TEXT    NOT NULL DEFAULT '',
            provision       TEXT    NOT NULL,
            document_id     TEXT,
            created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (node_id, canonical_title, provision)
        );
        """
    )

    # Migrate old incompatible contentless FTS schema if present.
    if _fts_is_contentless(cur):
        _drop_fts_objects(cur)

    cur.execute(_EXTERNAL_CONTENT_FTS_SQL)

    cur.executescript(
        """
        CREATE TRIGGER IF NOT EXISTS semantic_index_ai
        AFTER INSERT ON semantic_index
        BEGIN
            INSERT INTO semantic_index_fts(
                rowid,
                index_key,
                canonical_title,
                provision,
                keywords,
                aliases
            )
            VALUES (
                new.id,
                new.index_key,
                new.canonical_title,
                new.provision,
                new.keywords,
                new.aliases
            );
        END;

        CREATE TRIGGER IF NOT EXISTS semantic_index_ad
        AFTER DELETE ON semantic_index
        BEGIN
            INSERT INTO semantic_index_fts(
                semantic_index_fts,
                rowid,
                index_key,
                canonical_title,
                provision,
                keywords,
                aliases
            )
            VALUES (
                'delete',
                old.id,
                old.index_key,
                old.canonical_title,
                old.provision,
                old.keywords,
                old.aliases
            );
        END;

        CREATE TRIGGER IF NOT EXISTS semantic_index_au
        AFTER UPDATE ON semantic_index
        BEGIN
            INSERT INTO semantic_index_fts(
                semantic_index_fts,
                rowid,
                index_key,
                canonical_title,
                provision,
                keywords,
                aliases
            )
            VALUES (
                'delete',
                old.id,
                old.index_key,
                old.canonical_title,
                old.provision,
                old.keywords,
                old.aliases
            );

            INSERT INTO semantic_index_fts(
                rowid,
                index_key,
                canonical_title,
                provision,
                keywords,
                aliases
            )
            VALUES (
                new.id,
                new.index_key,
                new.canonical_title,
                new.provision,
                new.keywords,
                new.aliases
            );
        END;
        """
    )

    conn.commit()
    rebuild_semantic_index_fts(conn)
