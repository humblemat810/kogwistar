use std::{collections::BTreeSet, env, fs, path::PathBuf};

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let openapi_path = manifest_dir.join("../../../contracts/golden/openapi.json");
    let mcp_tools_path = manifest_dir.join("../../../contracts/golden/mcp-tools.json");
    let port_manifest_path = manifest_dir.join("../../../contracts/rust-port-v1.json");
    let templates_dir = manifest_dir.join("../../../kogwistar/templates");
    println!("cargo:rerun-if-changed={}", openapi_path.display());
    println!("cargo:rerun-if-changed={}", mcp_tools_path.display());
    println!("cargo:rerun-if-changed={}", port_manifest_path.display());
    for name in ["cytoscape.html", "d3.html", "go.html"] {
        println!(
            "cargo:rerun-if-changed={}",
            templates_dir.join(name).display()
        );
    }
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
    let port_manifest: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(&port_manifest_path).expect("read Rust port manifest"),
    )
    .expect("parse Rust port manifest");
    let deferrals = &port_manifest["server_operation_deferrals"];
    assert_eq!(
        deferrals["contract_version"].as_u64(),
        Some(1),
        "unsupported server deferral contract version"
    );
    assert_eq!(
        deferrals["status"].as_str(),
        Some("versioned-intentionally-deferred"),
        "server deferrals must remain explicitly versioned"
    );

    let mut pending_routes = BTreeSet::new();
    for group in deferrals["route_groups"]
        .as_array()
        .expect("server deferral route_groups must be an array")
    {
        for operation in group["operations"]
            .as_array()
            .expect("each server deferral group's operations must be an array")
        {
            let method = operation["method"]
                .as_str()
                .expect("deferred route method must be a string");
            let path = operation["path"]
                .as_str()
                .expect("deferred route path must be a string");
            assert!(
                ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"].contains(&method),
                "unsupported deferred route method: {method}"
            );
            assert!(
                path.starts_with('/'),
                "deferred route must be absolute: {path}"
            );
            assert!(
                pending_routes.insert((method.to_owned(), path.to_owned())),
                "duplicate deferred route: {method} {path}"
            );
        }
    }

    let mut pending_syscalls = BTreeSet::new();
    for operation in deferrals["syscall_group"]["operations"]
        .as_array()
        .expect("syscall deferral operations must be an array")
    {
        let operation = operation
            .as_str()
            .expect("deferred syscall operation must be a string");
        assert!(
            !operation.is_empty()
                && operation
                    .bytes()
                    .all(|byte| byte.is_ascii_lowercase() || byte == b'_'),
            "invalid deferred syscall operation: {operation}"
        );
        assert!(
            pending_syscalls.insert(operation.to_owned()),
            "duplicate deferred syscall operation: {operation}"
        );
    }

    let pending_route_entries = pending_routes
        .iter()
        .map(|(method, path)| format!("    ({method:?}, {path:?}),\n"))
        .collect::<String>();
    let pending_syscall_entries = pending_syscalls
        .iter()
        .map(|operation| format!("    {operation:?},\n"))
        .collect::<String>();
    let output = format!(
        "pub const FROZEN_OPENAPI_ROUTES: &[(&str, &str)] = &[\n{entries}];\n\
         pub const PENDING_SERVER_CUTOVER_ROUTES: &[(&str, &str)] = &[\n{pending_route_entries}];\n\
         pub const PENDING_SYSCALL_CUTOVER_OPS: &[&str] = &[\n{pending_syscall_entries}];\n"
    );
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    fs::write(out_dir.join("frozen_routes.rs"), output).unwrap();
    fs::copy(&openapi_path, out_dir.join("openapi.json")).unwrap();
    fs::copy(&mcp_tools_path, out_dir.join("mcp-tools.json")).unwrap();
    for name in ["cytoscape.html", "d3.html", "go.html"] {
        fs::copy(templates_dir.join(name), out_dir.join(name)).unwrap();
    }
}
