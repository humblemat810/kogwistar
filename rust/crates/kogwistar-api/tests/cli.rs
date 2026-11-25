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

fn run_mock_http_cli(
    arguments: impl FnOnce(String) -> Vec<String>,
    response_body: &'static str,
) -> (std::process::Output, String) {
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
        stream
            .set_read_timeout(Some(std::time::Duration::from_secs(2)))
            .expect("bound request read");
        let mut request = Vec::new();
        let mut buffer = [0_u8; 1024];
        loop {
            let read = stream.read(&mut buffer).expect("read CLI request");
            request.extend_from_slice(&buffer[..read]);
            let Some(header_end) = request.windows(4).position(|window| window == b"\r\n\r\n")
            else {
                continue;
            };
            let headers = String::from_utf8_lossy(&request[..header_end]);
            let content_length = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().ok())
                        .flatten()
                })
                .unwrap_or(0);
            if request.len() >= header_end + 4 + content_length {
                break;
            }
        }
        let request = String::from_utf8(request).expect("HTTP request is UTF-8");
        write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            response_body.len(),
            response_body
        )
        .expect("write mock response");
        request
    });

    let output = Command::new(env!("CARGO_BIN_EXE_kogwistar-cli"))
        .args(arguments(format!("http://{address}")))
        .env("KOGWISTAR_API_TOKEN", "cli-test-token")
        .output()
        .expect("run kogwistar-cli HTTP command");
    let request = server.join().expect("mock HTTP server completes");
    (output, request)
}

fn assert_cli_json(output: &std::process::Output, expected: serde_json::Value) {
    assert!(output.status.success());
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&output.stdout).unwrap(),
        expected
    );
    assert!(output.stderr.is_empty());
}

#[test]
fn conversation_transcript_command_uses_python_http_client_contract() {
    let (output, request) = run_mock_http_cli(
        |base| {
            [
                "api",
                "conversation",
                "conversation.get_transcript",
                "--conversation-id",
                "c-1",
                "--base-url",
            ]
            .into_iter()
            .map(str::to_owned)
            .chain([base])
            .collect()
        },
        r#"{"conversation_id":"c-1","turns":[]}"#,
    );
    assert!(request.starts_with("GET /api/conversations/c-1/turns HTTP/1.1\r\n"));
    assert!(
        request
            .to_ascii_lowercase()
            .contains("authorization: bearer cli-test-token\r\n")
    );
    assert_cli_json(
        &output,
        serde_json::json!({"conversation_id": "c-1", "turns": []}),
    );
}

#[test]
fn conversation_create_command_posts_python_compatible_payload() {
    let (output, request) = run_mock_http_cli(
        |base| {
            [
                "api",
                "conversation",
                "conversation.create",
                "--user-id",
                "alice",
                "--conversation-id",
                "c-2",
                "--start-node-id",
                "n-start",
                "--base-url",
            ]
            .into_iter()
            .map(str::to_owned)
            .chain([base])
            .collect()
        },
        r#"{"conversation_id":"c-2","user_id":"alice"}"#,
    );
    assert!(request.starts_with("POST /api/conversations HTTP/1.1\r\n"));
    assert!(
        request
            .to_ascii_lowercase()
            .contains("authorization: bearer cli-test-token\r\n")
    );
    let body = request.split_once("\r\n\r\n").unwrap().1;
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(body).unwrap(),
        serde_json::json!({"user_id": "alice", "conversation_id": "c-2", "start_node_id": "n-start"})
    );
    assert_cli_json(
        &output,
        serde_json::json!({"conversation_id": "c-2", "user_id": "alice"}),
    );
}

#[test]
fn conversation_ask_command_uses_python_defaults_and_payload() {
    let (output, request) = run_mock_http_cli(
        |base| {
            [
                "api",
                "conversation",
                "conversation.ask",
                "--conversation-id",
                "c-3",
                "--text",
                "hello",
                "--base-url",
            ]
            .into_iter()
            .map(str::to_owned)
            .chain([base])
            .collect()
        },
        r#"{"run_id":"r-3","status":"queued"}"#,
    );
    assert!(request.starts_with("POST /api/conversations/c-3/turns:answer HTTP/1.1\r\n"));
    let body = request.split_once("\r\n\r\n").unwrap().1;
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(body).unwrap(),
        serde_json::json!({"text": "hello", "user_id": null, "workflow_id": "agentic_answering.v2"})
    );
    assert_cli_json(
        &output,
        serde_json::json!({"run_id": "r-3", "status": "queued"}),
    );
}

#[test]
fn conversation_run_status_command_gets_python_endpoint() {
    let (output, request) = run_mock_http_cli(
        |base| {
            [
                "api",
                "conversation",
                "conversation.run_status",
                "--run-id",
                "r-4",
                "--base-url",
            ]
            .into_iter()
            .map(str::to_owned)
            .chain([base])
            .collect()
        },
        r#"{"run_id":"r-4","status":"completed"}"#,
    );
    assert!(request.starts_with("GET /api/runs/r-4 HTTP/1.1\r\n"));
    assert_cli_json(
        &output,
        serde_json::json!({"run_id": "r-4", "status": "completed"}),
    );
}

#[test]
fn conversation_cancel_run_command_posts_python_endpoint() {
    let (output, request) = run_mock_http_cli(
        |base| {
            [
                "api",
                "conversation",
                "conversation.cancel_run",
                "--run-id",
                "r-5",
                "--base-url",
            ]
            .into_iter()
            .map(str::to_owned)
            .chain([base])
            .collect()
        },
        r#"{"run_id":"r-5","status":"cancelling"}"#,
    );
    assert!(request.starts_with("POST /api/runs/r-5/cancel HTTP/1.1\r\n"));
    assert_cli_json(
        &output,
        serde_json::json!({"run_id": "r-5", "status": "cancelling"}),
    );
}
