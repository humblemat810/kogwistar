use kogwistar_api::{HealthSnapshot, cli_health, validate_server_backend};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::process::ExitCode;

fn usage() -> ! {
    eprintln!(
        "usage: kogwistar-cli health [backend] | api conversation <conversation.create|conversation.get_transcript|conversation.ask|conversation.run_status|conversation.cancel_run> [options]"
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
        Some("api") if args.next().as_deref() == Some("conversation") => {
            conversation_api_command(args)
        }
        _ => usage(),
    }
}
