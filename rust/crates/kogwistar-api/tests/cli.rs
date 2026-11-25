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

#[test]
fn conversation_transcript_command_uses_python_http_client_contract() {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;

    let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock HTTP server");
    listener
        .set_nonblocking(true)
        .expect("make mock accept bounded");
    let address = listener.local_addr().expect("mock server address");
    let server = thread::spawn(move || {
        let (mut stream, _) = (0..200)
            .find_map(|_| match listener.accept() {
                Ok(connection) => Some(connection),
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    thread::sleep(std::time::Duration::from_millis(10));
                    None
                }
                Err(error) => panic!("accept CLI request: {error}"),
            })
            .expect("CLI request arrives within two seconds");
        let mut request = Vec::new();
        let mut buffer = [0_u8; 1024];
        loop {
            let read = stream.read(&mut buffer).expect("read CLI request");
            request.extend_from_slice(&buffer[..read]);
            if request.windows(4).any(|window| window == b"\r\n\r\n") {
                break;
            }
        }
        let request = String::from_utf8(request).expect("HTTP request is UTF-8");
        assert!(request.starts_with("GET /api/conversations/c-1/turns HTTP/1.1\r\n"));
        assert!(
            request
                .to_ascii_lowercase()
                .contains("authorization: bearer cli-test-token\r\n")
        );

        let body = r#"{"conversation_id":"c-1","turns":[]}"#;
        write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        )
        .expect("write mock response");
    });

    let output = Command::new(env!("CARGO_BIN_EXE_kogwistar-cli"))
        .args([
            "api",
            "conversation",
            "conversation.get_transcript",
            "--conversation-id",
            "c-1",
            "--base-url",
            &format!("http://{address}"),
        ])
        .env("KOGWISTAR_API_TOKEN", "cli-test-token")
        .output()
        .expect("run kogwistar-cli transcript command");
    server.join().expect("mock HTTP server completes");

    assert!(output.status.success());
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&output.stdout).unwrap(),
        serde_json::json!({"conversation_id": "c-1", "turns": []})
    );
    assert!(output.stderr.is_empty());
}
