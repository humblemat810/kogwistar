from __future__ import annotations

import sqlite3


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

    cur.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS semantic_index_fts
        USING fts5(
            index_key UNINDEXED,
            canonical_title,
            provision,
            keywords,
            aliases,
            content=''
        );
        """
    )

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
            DELETE FROM semantic_index_fts WHERE rowid = old.id;
        END;

        CREATE TRIGGER IF NOT EXISTS semantic_index_au
        AFTER UPDATE ON semantic_index
        BEGIN
            DELETE FROM semantic_index_fts WHERE rowid = old.id;
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