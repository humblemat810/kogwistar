/// Normalize a graph role using the Python `kogwistar.graph_kinds` aliases.
///
/// Unknown roles are preserved so graph roles remain extensible.
pub fn normalize_graph_kind(kind: Option<&str>) -> String {
    let raw = match kind {
        None | Some("") => "knowledge".to_owned(),
        Some(value) => value.trim().to_ascii_lowercase(),
    };
    match raw.as_str() {
        "knowledge" | "kg" => "knowledge".to_owned(),
        "workflow" | "wf" => "workflow".to_owned(),
        "wisdom" | "w" => "wisdom".to_owned(),
        "conversation" | "chat" | "conv" | "dialog" => "conversation".to_owned(),
        _ => raw,
    }
}

#[cfg(test)]
mod tests {
    use super::normalize_graph_kind;
    use serde_json::Value;

    #[test]
    fn graph_kind_normalization_matches_shared_python_fixture() {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/graph-kind-v1.json"
        )))
        .expect("graph-kind fixture is valid JSON");
        for case in fixture["cases"].as_array().expect("cases array") {
            let input = case["input"].as_str();
            let expected = case["expected"].as_str().expect("expected string");
            assert_eq!(normalize_graph_kind(input), expected);
        }
    }
}
