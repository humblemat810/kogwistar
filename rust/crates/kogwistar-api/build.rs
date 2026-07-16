use std::{env, fs, path::PathBuf};

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let openapi_path = manifest_dir.join("../../../contracts/golden/openapi.json");
    let mcp_tools_path = manifest_dir.join("../../../contracts/golden/mcp-tools.json");
    println!("cargo:rerun-if-changed={}", openapi_path.display());
    println!("cargo:rerun-if-changed={}", mcp_tools_path.display());
    let document: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&openapi_path).unwrap()).unwrap();
    let mut routes = Vec::new();
    for (path, operations) in document["paths"].as_object().unwrap() {
        for method in operations.as_object().unwrap().keys() {
            if ["get", "post", "put", "patch", "delete", "head", "options"]
                .contains(&method.as_str())
            {
                routes.push((method.to_ascii_uppercase(), path.clone()));
            }
        }
    }
    routes.sort();
    let entries = routes
        .iter()
        .map(|(method, path)| format!("    ({method:?}, {path:?}),\n"))
        .collect::<String>();
    let output = format!("pub const FROZEN_OPENAPI_ROUTES: &[(&str, &str)] = &[\n{entries}];\n");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    fs::write(out_dir.join("frozen_routes.rs"), output).unwrap();
    fs::copy(&openapi_path, out_dir.join("openapi.json")).unwrap();
    fs::copy(&mcp_tools_path, out_dir.join("mcp-tools.json")).unwrap();
}
