//! Python-compatible lexical projection for the derived search index.
//!
//! PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/search_index_fts.rs#upsert_entries; python=kogwistar/engine_core/search_index/service.py#SearchIndexService.upsert_entries; fixture=contracts/golden/search-index-entry-v1.json; rust_test=rust/crates/kogwistar-api/src/search_index_fts.rs#rust_fts_projection_matches_python_index_schema; python_test=tests/core/test_search_index_contract.py#test_search_index_entries_match_shared_rust_golden_fixture

use std::path::Path;

use kogwistar_contracts::{SearchIndexEntryV1, make_search_index_key};
use rusqlite::{Connection, OptionalExtension, params};

const INDEX_TABLE_SCHEMA: &str = "CREATE TABLE IF NOT EXISTS semantic_index (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    index_key TEXT NOT NULL UNIQUE,
    node_id TEXT NOT NULL,
    canonical_title TEXT NOT NULL,
    keywords TEXT NOT NULL DEFAULT '',
    aliases TEXT NOT NULL DEFAULT '',
    provision TEXT NOT NULL,
    document_id TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(node_id, canonical_title, provision)
);";

const FTS_SCHEMA: &str = "CREATE VIRTUAL TABLE IF NOT EXISTS semantic_index_fts
USING fts5(index_key UNINDEXED, canonical_title, provision, keywords, aliases,
           content='semantic_index', content_rowid='id');
CREATE TRIGGER IF NOT EXISTS semantic_index_ai AFTER INSERT ON semantic_index BEGIN
    INSERT INTO semantic_index_fts(rowid,index_key,canonical_title,provision,keywords,aliases)
    VALUES(new.id,new.index_key,new.canonical_title,new.provision,new.keywords,new.aliases);
END;
CREATE TRIGGER IF NOT EXISTS semantic_index_ad AFTER DELETE ON semantic_index BEGIN
    INSERT INTO semantic_index_fts(semantic_index_fts,rowid,index_key,canonical_title,provision,keywords,aliases)
    VALUES('delete',old.id,old.index_key,old.canonical_title,old.provision,old.keywords,old.aliases);
END;
CREATE TRIGGER IF NOT EXISTS semantic_index_au AFTER UPDATE ON semantic_index BEGIN
    INSERT INTO semantic_index_fts(semantic_index_fts,rowid,index_key,canonical_title,provision,keywords,aliases)
    VALUES('delete',old.id,old.index_key,old.canonical_title,old.provision,old.keywords,old.aliases);
    INSERT INTO semantic_index_fts(rowid,index_key,canonical_title,provision,keywords,aliases)
    VALUES(new.id,new.index_key,new.canonical_title,new.provision,new.keywords,new.aliases);
END;";

#[derive(Clone, Debug, PartialEq)]
pub struct LexicalIndexHit {
    pub index_key: String,
    pub node_id: String,
    pub canonical_title: String,
    pub provision: String,
    pub document_id: Option<String>,
    pub bm25_score: f64,
}

pub fn open_index(path: impl AsRef<Path>) -> rusqlite::Result<Connection> {
    if path.as_ref() != Path::new(":memory:")
        && let Some(parent) = path
            .as_ref()
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)
            .map_err(|error| rusqlite::Error::ToSqlConversionFailure(Box::new(error)))?;
    }
    let connection = Connection::open(path)?;
    connection.execute_batch("PRAGMA foreign_keys = ON;")?;
    connection.execute_batch(INDEX_TABLE_SCHEMA)?;
    let existing_fts_schema = connection
        .query_row(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='semantic_index_fts'",
            [],
            |row| row.get::<_, String>(0),
        )
        .optional()?;
    if existing_fts_schema.is_some_and(|sql| {
        let sql = sql.to_ascii_lowercase();
        sql.contains("content=''") || sql.contains("content=\"\"")
    }) {
        connection.execute_batch(
            "DROP TRIGGER IF EXISTS semantic_index_ai;
             DROP TRIGGER IF EXISTS semantic_index_ad;
             DROP TRIGGER IF EXISTS semantic_index_au;
             DROP TABLE IF EXISTS semantic_index_fts;",
        )?;
    }
    connection.execute_batch(FTS_SCHEMA)?;
    connection.execute(
        "INSERT INTO semantic_index_fts(semantic_index_fts) VALUES('rebuild')",
        [],
    )?;
    Ok(connection)
}

pub fn upsert_entries(
    connection: &mut Connection,
    entries: &[SearchIndexEntryV1],
) -> rusqlite::Result<()> {
    let transaction = connection.transaction()?;
    {
        let mut statement = transaction.prepare_cached(
            "INSERT INTO semantic_index
                (index_key,node_id,canonical_title,keywords,aliases,provision,document_id)
             VALUES (?1,?2,?3,?4,?5,?6,?7)
             ON CONFLICT(index_key) DO UPDATE SET
                node_id=excluded.node_id,
                canonical_title=excluded.canonical_title,
                keywords=excluded.keywords,
                aliases=excluded.aliases,
                provision=excluded.provision,
                document_id=excluded.document_id,
                updated_at=CURRENT_TIMESTAMP",
        )?;
        for entry in entries {
            statement.execute(params![
                make_search_index_key(&entry.node_id, &entry.canonical_title, &entry.provision),
                entry.node_id,
                entry.canonical_title,
                entry.keywords.join(" "),
                entry.aliases.join(" "),
                entry.provision,
                entry.doc_id,
            ])?;
        }
    }
    transaction.commit()
}

pub fn search_fts(
    connection: &Connection,
    query: &str,
    limit: usize,
) -> rusqlite::Result<Vec<LexicalIndexHit>> {
    let mut statement = connection.prepare(
        "SELECT s.index_key,s.node_id,s.canonical_title,s.provision,s.document_id,
                bm25(semantic_index_fts) AS fts_score
         FROM semantic_index AS s
         JOIN semantic_index_fts ON semantic_index_fts.rowid=s.id
         WHERE semantic_index_fts MATCH ?1
         ORDER BY fts_score
         LIMIT ?2",
    )?;
    let rows = statement.query_map(params![query, limit as i64], |row| {
        Ok(LexicalIndexHit {
            index_key: row.get(0)?,
            node_id: row.get(1)?,
            canonical_title: row.get(2)?,
            provision: row.get(3)?,
            document_id: row.get(4)?,
            bm25_score: row.get(5)?,
        })
    })?;
    rows.collect()
}

#[cfg(test)]
mod tests {
    use super::{open_index, search_fts, upsert_entries};
    use kogwistar_contracts::SearchIndexUpsertV1;
    use serde_json::Value;
    use std::path::PathBuf;

    fn fixture() -> (SearchIndexUpsertV1, Value) {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/search-index-entry-v1.json"
        )))
        .expect("search index fixture is valid");
        let request = serde_json::from_value(fixture["request"].clone())
            .expect("fixture request follows shared contract");
        (request, fixture)
    }

    #[test]
    fn rust_fts_projection_matches_python_index_schema() {
        let (request, fixture) = fixture();
        let mut connection = open_index(":memory:").expect("open in-memory index");
        upsert_entries(&mut connection, &request.index).expect("persist lexical entries");

        let count: i64 = connection
            .query_row("SELECT count(*) FROM semantic_index", [], |row| row.get(0))
            .expect("read persisted entry count");
        assert_eq!(count as usize, request.index.len());

        for (entry, expected) in request
            .index
            .iter()
            .zip(fixture["expected"].as_array().expect("expected entries"))
        {
            let key = kogwistar_contracts::make_search_index_key(
                &entry.node_id,
                &entry.canonical_title,
                &entry.provision,
            );
            let row: (String, String, String, String, Option<String>) = connection
                .query_row(
                    "SELECT index_key,node_id,canonical_title,provision,document_id
                     FROM semantic_index WHERE index_key=?1",
                    [&key],
                    |row| {
                        Ok((
                            row.get(0)?,
                            row.get(1)?,
                            row.get(2)?,
                            row.get(3)?,
                            row.get(4)?,
                        ))
                    },
                )
                .expect("lookup Python-compatible row");
            assert_eq!(row.0, expected["index_key"]);
            assert_eq!(row.1, entry.node_id);
            assert_eq!(row.2, entry.canonical_title);
            assert_eq!(row.3, entry.provision);
            assert_eq!(row.4.as_deref(), entry.doc_id.as_deref());
        }
        let hits = search_fts(&connection, "permit", 10).expect("FTS query");
        assert!(hits.iter().any(|hit| hit.node_id == "node-42"));
    }

    #[test]
    fn index_reopen_and_upsert_keep_fts_projection_in_sync() {
        let (request, _) = fixture();
        let path = PathBuf::from(std::env::temp_dir()).join(format!(
            "kogwistar-search-index-{}.sqlite3",
            uuid::Uuid::new_v4()
        ));
        {
            let mut connection = open_index(&path).expect("open persistent index");
            upsert_entries(&mut connection, &request.index).expect("initial upsert");
        }
        {
            let mut connection = open_index(&path).expect("reopen persistent index");
            upsert_entries(&mut connection, &request.index).expect("idempotent upsert");
            let hits = search_fts(&connection, "permit", 10).expect("query reopened FTS");
            assert!(hits.iter().any(|hit| hit.node_id == "node-42"));
            let count: i64 = connection
                .query_row("SELECT count(*) FROM semantic_index", [], |row| row.get(0))
                .expect("verify upsert identity");
            assert_eq!(count as usize, request.index.len());
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn update_and_delete_keep_external_content_fts_in_sync() {
        let (request, _) = fixture();
        let mut connection = open_index(":memory:").expect("open in-memory index");
        upsert_entries(&mut connection, &request.index).expect("initial upsert");

        let mut revised = request.index[0].clone();
        revised.keywords = vec!["revisedtoken".to_owned()];
        revised.aliases.clear();
        upsert_entries(&mut connection, &[revised.clone()]).expect("update entry");

        assert!(
            search_fts(&connection, "authorization", 10)
                .expect("query removed alias")
                .iter()
                .all(|hit| hit.node_id != revised.node_id)
        );
        assert!(
            search_fts(&connection, "revisedtoken", 10)
                .expect("query revised keyword")
                .iter()
                .any(|hit| hit.node_id == revised.node_id)
        );

        let index_key = kogwistar_contracts::make_search_index_key(
            &revised.node_id,
            &revised.canonical_title,
            &revised.provision,
        );
        connection
            .execute(
                "DELETE FROM semantic_index WHERE index_key=?1",
                [&index_key],
            )
            .expect("delete canonical row");
        assert!(
            search_fts(&connection, "revisedtoken", 10)
                .expect("query deleted entry")
                .iter()
                .all(|hit| hit.node_id != revised.node_id)
        );
    }

    #[test]
    fn opening_legacy_contentless_fts_database_migrates_and_rebuilds() {
        let path = std::env::temp_dir().join(format!(
            "kogwistar-search-index-legacy-{}.sqlite3",
            uuid::Uuid::new_v4()
        ));
        {
            let connection = rusqlite::Connection::open(&path).expect("legacy database");
            connection
                .execute_batch(
                    "CREATE TABLE semantic_index (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        index_key TEXT NOT NULL UNIQUE,
                        node_id TEXT NOT NULL,
                        canonical_title TEXT NOT NULL,
                        keywords TEXT NOT NULL DEFAULT '',
                        aliases TEXT NOT NULL DEFAULT '',
                        provision TEXT NOT NULL,
                        document_id TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(node_id, canonical_title, provision)
                    );
                    CREATE VIRTUAL TABLE semantic_index_fts USING fts5(
                        index_key UNINDEXED, canonical_title, provision, keywords, aliases, content=''
                    );",
                )
                .expect("create legacy contentless FTS schema");
            connection
                .execute(
                    "INSERT INTO semantic_index VALUES
                        (1,'node-42|Permit Review|Review the permit before work begins.',
                         'node-42','Permit Review','permit review','approval authorization',
                         'Review the permit before work begins.','safety-manual',CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)",
                    [],
                )
                .expect("insert canonical legacy row");
        }
        {
            let connection = open_index(&path).expect("migrate legacy index");
            let schema: String = connection
                .query_row(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name='semantic_index_fts'",
                    [],
                    |row| row.get(0),
                )
                .expect("read migrated FTS schema");
            assert!(schema.contains("content='semantic_index'"));
            assert!(
                search_fts(&connection, "permit", 10)
                    .expect("query rebuilt index")
                    .iter()
                    .any(|hit| hit.node_id == "node-42")
            );
        }
        let _ = std::fs::remove_file(&path);
    }
}
