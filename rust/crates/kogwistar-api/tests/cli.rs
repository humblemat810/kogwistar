use std::process::Command;

// Mirrors kogwistar._rust_bridge.api_cli_health through the shipped CLI binary.
#[test]
fn health_command_prints_selected_backend() {
    let output = Command::new(env!("CARGO_BIN_EXE_kogwistar-cli"))
        .args(["health", "sqlite"])
        .output()
        .expect("run kogwistar-cli health");

    assert!(output.status.success());
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        "ok backend=sqlite\n"
    );
    assert!(output.stderr.is_empty());
}

#[test]
fn health_command_rejects_unsupported_backend() {
    let output = Command::new(env!("CARGO_BIN_EXE_kogwistar-cli"))
        .args(["health", "chroma"])
        .output()
        .expect("run kogwistar-cli with unsupported backend");

    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
    assert!(
        String::from_utf8(output.stderr)
            .unwrap()
            .contains("KOGWISTAR_BACKEND must be pg, sqlite, or in_memory")
    );
}
