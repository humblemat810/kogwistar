use serde::{Deserialize, Serialize};

/// PYTHON_PARITY_V1: rust=rust/crates/kogwistar-contracts/src/search_index.rs#build_search_index_embedding_text; python=kogwistar/engine_core/search_index/models.py#build_embedding_text; fixture=contracts/golden/search-index-entry-v1.json; rust_test=rust/crates/kogwistar-contracts/src/search_index.rs#search_index_entries_match_python_golden_fixture; python_test=tests/core/test_search_index_contract.py#test_search_index_entries_match_shared_rust_golden_fixture
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct SearchIndexEntryV1 {
    pub node_id: String,
    pub canonical_title: String,
    #[serde(default)]
    pub keywords: Vec<String>,
    #[serde(default)]
    pub aliases: Vec<String>,
    pub provision: String,
    #[serde(default)]
    pub doc_id: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct SearchIndexUpsertV1 {
    pub index: Vec<SearchIndexEntryV1>,
}

pub fn make_search_index_key(node_id: &str, canonical_title: &str, provision: &str) -> String {
    format!("{node_id}|{canonical_title}|{provision}")
}

pub fn build_search_index_embedding_text(item: &SearchIndexEntryV1) -> String {
    let mut lines = vec![
        "ENTITY SEARCH INDEX ENTRY".to_owned(),
        format!("CANONICAL TITLE: {}", item.canonical_title),
        format!("PROVISION/DESCRIPTION: {}", item.provision),
    ];
    let keywords = item.keywords.join(", ");
    if !keywords.is_empty() {
        lines.push(format!("KEYWORDS: {keywords}"));
    }
    let aliases = item.aliases.join(", ");
    if !aliases.is_empty() {
        lines.push(format!("ALIASES/KNOWN AS: {aliases}"));
    }
    if let Some(doc_id) = item.doc_id.as_deref().filter(|value| !value.is_empty()) {
        lines.push(format!("SOURCE DOCUMENT: {doc_id}"));
    }
    lines.push(format!("TARGET NODE ID: {}", item.node_id));
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::{SearchIndexUpsertV1, build_search_index_embedding_text, make_search_index_key};
    use serde_json::Value;

    #[test]
    fn search_index_entries_match_python_golden_fixture() {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/search-index-entry-v1.json"
        )))
        .expect("shared search-index fixture");
        let request: SearchIndexUpsertV1 =
            serde_json::from_value(fixture["request"].clone()).expect("valid request");
        let expected = fixture["expected"].as_array().expect("expected entries");
        assert_eq!(request.index.len(), expected.len());
        for (item, expected) in request.index.iter().zip(expected) {
            assert_eq!(
                make_search_index_key(&item.node_id, &item.canonical_title, &item.provision),
                expected["index_key"]
            );
            assert_eq!(
                build_search_index_embedding_text(item),
                expected["embedding_text"]
            );
        }
    }
}
