use kogwistar_api::{HealthSnapshot, cli_health, validate_server_backend};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::process::ExitCode;

fn usage() -> ! {
    eprintln!(
        "usage: kogwistar-cli health [backend] | api conversation <conversation-command> [options] | api workflow <workflow-command> [options]"
    );
    std::process::exit(2)
}

fn health(backend: String) -> ExitCode {
    if let Err(error) = validate_server_backend(&backend) {
        eprintln!("{error}");
        return ExitCode::from(2);
    }
    println!(
        "{}",
        cli_health(&HealthSnapshot {
            backend,
            persist_directory: String::new(),
            conversation_persist_directory: String::new(),
            workflow_persist_directory: String::new(),
            wisdom_persist_directory: String::new(),
            pg_schema_base: None,
        })
    );
    ExitCode::SUCCESS
}

fn option_values(args: impl Iterator<Item = String>) -> HashMap<String, String> {
    let mut values = HashMap::new();
    let mut args = args;
    while let Some(option) = args.next() {
        if !option.starts_with("--") {
            usage();
        }
        let Some(value) = args.next() else {
            usage();
        };
        values.insert(option, value);
    }
    values
}

fn validate_options(options: &HashMap<String, String>, allowed: &[&str]) {
    if options
        .keys()
        .any(|option| !allowed.contains(&option.as_str()))
    {
        usage();
    }
}

fn required_option<'a>(options: &'a HashMap<String, String>, name: &str) -> &'a str {
    options
        .get(name)
        .map(String::as_str)
        .unwrap_or_else(|| usage())
}

fn optional_integer(options: &HashMap<String, String>, name: &str) -> Option<i64> {
    options
        .get(name)
        .map(|value| value.parse::<i64>().unwrap_or_else(|_| usage()))
}

fn request_json(method: reqwest::Method, url: String, body: Option<Value>) -> ExitCode {
    let token = std::env::var("KOGWISTAR_API_TOKEN").ok();
    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => {
            eprintln!("could not initialize HTTP runtime: {error}");
            return ExitCode::FAILURE;
        }
    };
    let response = runtime.block_on(async {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()?;
        let mut request = client.request(method, url);
        if let Some(body) = body {
            request = request.json(&body);
        }
        if let Some(token) = token {
            request = request.bearer_auth(token);
        }
        request
            .send()
            .await?
            .error_for_status()?
            .json::<Value>()
            .await
    });
    match response {
        Ok(payload) => match serde_json::to_string_pretty(&payload) {
            Ok(json) => {
                println!("{json}");
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("could not encode server response: {error}");
                ExitCode::FAILURE
            }
        },
        Err(error) => {
            eprintln!("API request failed: {error}");
            ExitCode::FAILURE
        }
    }
}

fn conversation_api_command(mut args: impl Iterator<Item = String>) -> ExitCode {
    let Some(command) = args.next() else {
        usage();
    };
    let options = option_values(args);
    let base_url = options
        .get("--base-url")
        .cloned()
        .or_else(|| std::env::var("KOGWISTAR_BASE_URL").ok())
        .unwrap_or_else(|| "http://127.0.0.1:8787".to_owned());
    let base_url = base_url.trim_end_matches('/');
    let (method, path, body) = match command.as_str() {
        "conversation.create" => {
            validate_options(
                &options,
                &[
                    "--base-url",
                    "--user-id",
                    "--conversation-id",
                    "--start-node-id",
                ],
            );
            (
                reqwest::Method::POST,
                "/api/conversations".to_owned(),
                Some(json!({
                    "user_id": required_option(&options, "--user-id"),
                    "conversation_id": options.get("--conversation-id"),
                    "start_node_id": options.get("--start-node-id"),
                })),
            )
        }
        "conversation.get_transcript" => {
            validate_options(&options, &["--base-url", "--conversation-id"]);
            (
                reqwest::Method::GET,
                format!(
                    "/api/conversations/{}/turns",
                    required_option(&options, "--conversation-id")
                ),
                None,
            )
        }
        "conversation.ask" => {
            validate_options(
                &options,
                &[
                    "--base-url",
                    "--conversation-id",
                    "--text",
                    "--user-id",
                    "--workflow-id",
                ],
            );
            (
                reqwest::Method::POST,
                format!(
                    "/api/conversations/{}/turns:answer",
                    required_option(&options, "--conversation-id")
                ),
                Some(json!({
                    "text": required_option(&options, "--text"),
                    "user_id": options.get("--user-id"),
                    "workflow_id": options.get("--workflow-id").map(String::as_str)
                        .unwrap_or("agentic_answering.v2"),
                })),
            )
        }
        "conversation.run_status" | "conversation.cancel_run" => {
            validate_options(&options, &["--base-url", "--run-id"]);
            let run_id = required_option(&options, "--run-id");
            let is_cancel = command == "conversation.cancel_run";
            (
                if is_cancel {
                    reqwest::Method::POST
                } else {
                    reqwest::Method::GET
                },
                if is_cancel {
                    format!("/api/runs/{run_id}/cancel")
                } else {
                    format!("/api/runs/{run_id}")
                },
                None,
            )
        }
        _ => usage(),
    };
    request_json(method, format!("{base_url}{path}"), body)
}

fn workflow_api_command(mut args: impl Iterator<Item = String>) -> ExitCode {
    let Some(command) = args.next() else {
        usage();
    };
    let options = option_values(args);
    let base_url = options
        .get("--base-url")
        .cloned()
        .or_else(|| std::env::var("KOGWISTAR_BASE_URL").ok())
        .unwrap_or_else(|| "http://127.0.0.1:8787".to_owned());
    let base_url = base_url.trim_end_matches('/');
    if command == "workflow.run_submit" {
        validate_options(
            &options,
            &[
                "--base-url",
                "--workflow-id",
                "--conversation-id",
                "--turn-node-id",
                "--user-id",
                "--initial-state",
                "--priority-class",
                "--token-budget",
                "--time-budget-ms",
            ],
        );
        let initial_state = match options.get("--initial-state") {
            Some(value) => match serde_json::from_str::<Value>(value) {
                Ok(state) => state,
                Err(error) => {
                    eprintln!("invalid JSON for --initial-state: {error}");
                    return ExitCode::from(2);
                }
            },
            None => json!({}),
        };
        let body = json!({
            "workflow_id": required_option(&options, "--workflow-id"),
            "conversation_id": required_option(&options, "--conversation-id"),
            "turn_node_id": options.get("--turn-node-id"),
            "user_id": options.get("--user-id"),
            "initial_state": initial_state,
            "priority_class": options.get("--priority-class").map(String::as_str)
                .unwrap_or("foreground"),
            "token_budget": optional_integer(&options, "--token-budget"),
            "time_budget_ms": optional_integer(&options, "--time-budget-ms"),
        });
        return request_json(
            reqwest::Method::POST,
            format!("{base_url}/api/workflow/runs"),
            Some(body),
        );
    }
    if command == "workflow.run_resume" {
        validate_options(
            &options,
            &[
                "--base-url",
                "--run-id",
                "--suspended-node-id",
                "--suspended-token-id",
                "--client-result",
                "--workflow-id",
                "--conversation-id",
                "--turn-node-id",
                "--user-id",
            ],
        );
        let client_result = match options.get("--client-result") {
            Some(value) => match serde_json::from_str::<Value>(value) {
                Ok(result) => result,
                Err(error) => {
                    eprintln!("invalid JSON for --client-result: {error}");
                    return ExitCode::from(2);
                }
            },
            None => json!({}),
        };
        let body = json!({
            "suspended_node_id": required_option(&options, "--suspended-node-id"),
            "suspended_token_id": required_option(&options, "--suspended-token-id"),
            "client_result": client_result,
            "workflow_id": required_option(&options, "--workflow-id"),
            "conversation_id": required_option(&options, "--conversation-id"),
            "turn_node_id": required_option(&options, "--turn-node-id"),
            "user_id": options.get("--user-id"),
        });
        return request_json(
            reqwest::Method::POST,
            format!(
                "{base_url}/api/workflow/runs/{}/resume",
                required_option(&options, "--run-id")
            ),
            Some(body),
        );
    }
    let (method, path) = match command.as_str() {
        "workflow.process_table" => {
            validate_options(
                &options,
                &[
                    "--base-url",
                    "--status",
                    "--workflow-id",
                    "--conversation-id",
                    "--limit",
                ],
            );
            let limit = options
                .get("--limit")
                .map(|value| value.parse::<i64>().unwrap_or_else(|_| usage()))
                .unwrap_or(100);
            (
                reqwest::Method::GET,
                format!("/api/workflow/operator/dashboard?limit={limit}"),
            )
        }
        "workflow.run_checkpoint_get" => {
            validate_options(&options, &["--base-url", "--run-id", "--step-seq"]);
            let run_id = required_option(&options, "--run-id");
            let step_seq = required_option(&options, "--step-seq")
                .parse::<i64>()
                .unwrap_or_else(|_| usage());
            (
                reqwest::Method::GET,
                format!("/api/workflow/runs/{run_id}/checkpoints/{step_seq}"),
            )
        }
        "workflow.run_resume_contract" => {
            validate_options(&options, &["--base-url", "--run-id"]);
            let run_id = required_option(&options, "--run-id");
            (
                reqwest::Method::GET,
                format!("/api/workflow/runs/{run_id}/resume-contract"),
            )
        }
        "workflow.run_replay" => {
            validate_options(&options, &["--base-url", "--run-id", "--target-step-seq"]);
            let run_id = required_option(&options, "--run-id");
            let target_step_seq = required_option(&options, "--target-step-seq")
                .parse::<i64>()
                .unwrap_or_else(|_| usage());
            (
                reqwest::Method::GET,
                format!("/api/workflow/runs/{run_id}/replay?target_step_seq={target_step_seq}"),
            )
        }
        "workflow.run_status" | "workflow.run_cancel" => {
            validate_options(&options, &["--base-url", "--run-id"]);
            let is_cancel = command == "workflow.run_cancel";
            let run_id = required_option(&options, "--run-id");
            (
                if is_cancel {
                    reqwest::Method::POST
                } else {
                    reqwest::Method::GET
                },
                if is_cancel {
                    format!("/api/workflow/runs/{run_id}/cancel")
                } else {
                    format!("/api/workflow/runs/{run_id}")
                },
            )
        }
        "workflow.run_events" => {
            validate_options(
                &options,
                &["--base-url", "--run-id", "--after-seq", "--limit"],
            );
            let run_id = required_option(&options, "--run-id");
            let after_seq = options
                .get("--after-seq")
                .map(String::as_str)
                .unwrap_or("0")
                .parse::<i64>()
                .unwrap_or_else(|_| usage());
            let _limit = options
                .get("--limit")
                .map(|value| value.parse::<i64>().unwrap_or_else(|_| usage()))
                .unwrap_or(200);
            (
                reqwest::Method::GET,
                format!("/api/workflow/runs/{run_id}/events?after_seq={after_seq}"),
            )
        }
        _ => usage(),
    };
    request_json(method, format!("{base_url}{path}"), None)
}

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("health") => {
            let backend = args.next().unwrap_or_else(|| "sqlite".to_owned());
            if args.next().is_some() {
                usage();
            }
            health(backend)
        }
        Some("api") => match args.next().as_deref() {
            Some("conversation") => conversation_api_command(args),
            Some("workflow") => workflow_api_command(args),
            _ => usage(),
        },
        _ => usage(),
    }
}
