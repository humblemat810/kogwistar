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
        if value.starts_with("--") {
            eprintln!("missing value for option: {option}");
            usage();
        }
        values.insert(option, value);
    }
    values
}

fn option_values_with_boolean_flags(
    args: impl Iterator<Item = String>,
    boolean_flags: &[&str],
) -> HashMap<String, String> {
    let mut values = HashMap::new();
    let mut args = args;
    while let Some(option) = args.next() {
        if !option.starts_with("--") {
            usage();
        }
        if boolean_flags.contains(&option.as_str()) {
            values.insert(option, "true".to_owned());
            continue;
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
    options.get(name).map(String::as_str).unwrap_or_else(|| {
        eprintln!("missing required option: {name}");
        usage()
    })
}

fn optional_integer(options: &HashMap<String, String>, name: &str) -> Option<i64> {
    options
        .get(name)
        .map(|value| value.parse::<i64>().unwrap_or_else(|_| usage()))
}

fn boolean_option(
    options: &HashMap<String, String>,
    affirmative: &str,
    negative: &str,
    default: bool,
) -> bool {
    options
        .get(negative)
        .map(|_| false)
        .or_else(|| options.get(affirmative).map(|_| true))
        .unwrap_or(default)
}

fn json_option(options: &HashMap<String, String>, name: &str) -> Value {
    options
        .get(name)
        .map(|value| {
            serde_json::from_str::<Value>(value).unwrap_or_else(|error| {
                eprintln!("invalid JSON for {name}: {error}");
                std::process::exit(2);
            })
        })
        .unwrap_or_else(|| json!({}))
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
    let options = match command.as_str() {
        "workflow.service_declare" => option_values_with_boolean_flags(
            args,
            &["--enabled", "--no-enabled", "--autostart", "--no-autostart"],
        ),
        "workflow.design_node_upsert" => option_values_with_boolean_flags(
            args,
            &[
                "--start",
                "--no-start",
                "--terminal",
                "--no-terminal",
                "--fanout",
                "--no-fanout",
            ],
        ),
        "workflow.design_edge_upsert" => {
            option_values_with_boolean_flags(args, &["--is-default", "--no-is-default"])
        }
        _ => option_values(args),
    };
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
    if command == "workflow.service_heartbeat" || command == "workflow.service_trigger" {
        let is_heartbeat = command == "workflow.service_heartbeat";
        let allowed = if is_heartbeat {
            &["--base-url", "--service-id", "--instance-id", "--payload"][..]
        } else {
            &["--base-url", "--service-id", "--trigger-type", "--payload"][..]
        };
        validate_options(&options, allowed);
        let service_id = required_option(&options, "--service-id");
        let payload = match options.get("--payload") {
            Some(value) => match serde_json::from_str::<Value>(value) {
                Ok(payload) => payload,
                Err(error) => {
                    eprintln!("invalid JSON for --payload: {error}");
                    return ExitCode::from(2);
                }
            },
            None => json!({}),
        };
        let (action, body) = if is_heartbeat {
            (
                "heartbeat",
                json!({
                    "instance_id": required_option(&options, "--instance-id"),
                    "payload": payload,
                }),
            )
        } else {
            (
                "trigger",
                json!({
                    "trigger_type": required_option(&options, "--trigger-type"),
                    "payload": payload,
                }),
            )
        };
        return request_json(
            reqwest::Method::POST,
            format!("{base_url}/api/workflow/services/{service_id}/{action}"),
            Some(body),
        );
    }
    if command == "workflow.service_declare" {
        validate_options(
            &options,
            &[
                "--base-url",
                "--service-id",
                "--service-kind",
                "--target-kind",
                "--target-ref",
                "--target-config",
                "--enabled",
                "--no-enabled",
                "--autostart",
                "--no-autostart",
                "--restart-policy",
                "--heartbeat-ttl-ms",
                "--trigger-specs",
            ],
        );
        let parse_json_option = |name: &str| -> Value {
            options
                .get(name)
                .map(|value| {
                    serde_json::from_str::<Value>(value).unwrap_or_else(|error| {
                        eprintln!("invalid JSON for {name}: {error}");
                        std::process::exit(2);
                    })
                })
                .unwrap_or_else(|| json!({}))
        };
        let enabled = options
            .get("--no-enabled")
            .map(|_| false)
            .or_else(|| options.get("--enabled").map(|_| true))
            .unwrap_or(true);
        let autostart = options
            .get("--no-autostart")
            .map(|_| false)
            .or_else(|| options.get("--autostart").map(|_| true))
            .unwrap_or(false);
        let body = json!({
            "service_id": required_option(&options, "--service-id"),
            "service_kind": required_option(&options, "--service-kind"),
            "target_kind": required_option(&options, "--target-kind"),
            "target_ref": required_option(&options, "--target-ref"),
            "target_config": parse_json_option("--target-config"),
            "enabled": enabled,
            "autostart": autostart,
            "restart_policy": parse_json_option("--restart-policy"),
            "heartbeat_ttl_ms": options
                .get("--heartbeat-ttl-ms")
                .map(|value| value.parse::<i64>().unwrap_or_else(|_| usage()))
                .unwrap_or(60000),
            "trigger_specs": options
                .get("--trigger-specs")
                .map(|value| serde_json::from_str::<Value>(value).unwrap_or_else(|error| {
                    eprintln!("invalid JSON for --trigger-specs: {error}");
                    std::process::exit(2);
                }))
                .unwrap_or_else(|| json!([])),
        });
        return request_json(
            reqwest::Method::POST,
            format!("{base_url}/api/workflow/services"),
            Some(body),
        );
    }
    if command == "workflow.capability_approve" || command == "workflow.capability_revoke" {
        let approve = command == "workflow.capability_approve";
        if approve {
            validate_options(
                &options,
                &["--base-url", "--action", "--capabilities", "--subject"],
            );
        } else {
            validate_options(&options, &["--base-url", "--capability", "--subject"]);
        }
        let subject = options
            .get("--subject")
            .map(|value| json!(value))
            .unwrap_or(Value::Null);
        let body = if approve {
            let capabilities =
                serde_json::from_str::<Value>(required_option(&options, "--capabilities"))
                    .unwrap_or_else(|error| {
                        eprintln!("invalid JSON for --capabilities: {error}");
                        std::process::exit(2);
                    });
            json!({
                "action": required_option(&options, "--action"),
                "capabilities": capabilities,
                "subject": subject,
            })
        } else {
            json!({
                "capability": required_option(&options, "--capability"),
                "subject": subject,
            })
        };
        let endpoint = if approve { "approve" } else { "revoke" };
        return request_json(
            reqwest::Method::POST,
            format!("{base_url}/api/workflow/capabilities/{endpoint}"),
            Some(body),
        );
    }
    if command == "workflow.design_node_upsert" {
        validate_options(
            &options,
            &[
                "--base-url",
                "--workflow-id",
                "--designer-id",
                "--label",
                "--node-id",
                "--op",
                "--start",
                "--no-start",
                "--terminal",
                "--no-terminal",
                "--fanout",
                "--no-fanout",
                "--metadata",
            ],
        );
        let body = json!({
            "designer_id": required_option(&options, "--designer-id"),
            "node_id": options.get("--node-id"),
            "label": required_option(&options, "--label"),
            "op": options.get("--op"),
            "start": boolean_option(&options, "--start", "--no-start", false),
            "terminal": boolean_option(&options, "--terminal", "--no-terminal", false),
            "fanout": boolean_option(&options, "--fanout", "--no-fanout", false),
            "metadata": json_option(&options, "--metadata"),
        });
        return request_json(
            reqwest::Method::POST,
            format!(
                "{base_url}/api/workflow/design/{}/nodes",
                required_option(&options, "--workflow-id")
            ),
            Some(body),
        );
    }
    if command == "workflow.design_edge_upsert" {
        validate_options(
            &options,
            &[
                "--base-url",
                "--workflow-id",
                "--designer-id",
                "--src",
                "--dst",
                "--edge-id",
                "--relation",
                "--predicate",
                "--priority",
                "--is-default",
                "--no-is-default",
                "--multiplicity",
                "--metadata",
            ],
        );
        let priority = options
            .get("--priority")
            .map(|value| value.parse::<i64>().unwrap_or_else(|_| usage()))
            .unwrap_or(100);
        let body = json!({
            "designer_id": required_option(&options, "--designer-id"),
            "edge_id": options.get("--edge-id"),
            "src": required_option(&options, "--src"),
            "dst": required_option(&options, "--dst"),
            "relation": options.get("--relation").map(String::as_str).unwrap_or("wf_next"),
            "predicate": options.get("--predicate"),
            "priority": priority,
            "is_default": boolean_option(&options, "--is-default", "--no-is-default", false),
            "multiplicity": options.get("--multiplicity").map(String::as_str).unwrap_or("one"),
            "metadata": json_option(&options, "--metadata"),
        });
        return request_json(
            reqwest::Method::POST,
            format!(
                "{base_url}/api/workflow/design/{}/edges",
                required_option(&options, "--workflow-id")
            ),
            Some(body),
        );
    }
    if command == "workflow.design_node_delete" || command == "workflow.design_edge_delete" {
        let node = command == "workflow.design_node_delete";
        let entity_option = if node { "--node-id" } else { "--edge-id" };
        validate_options(
            &options,
            &[
                "--base-url",
                "--workflow-id",
                "--designer-id",
                entity_option,
            ],
        );
        let entity = required_option(&options, entity_option);
        let kind = if node { "nodes" } else { "edges" };
        let body = json!({"designer_id": required_option(&options, "--designer-id")});
        return request_json(
            reqwest::Method::DELETE,
            format!(
                "{base_url}/api/workflow/design/{}/{kind}/{entity}",
                required_option(&options, "--workflow-id")
            ),
            Some(body),
        );
    }
    if command == "workflow.design_undo" || command == "workflow.design_redo" {
        validate_options(&options, &["--base-url", "--workflow-id", "--designer-id"]);
        let action = if command == "workflow.design_undo" {
            "undo"
        } else {
            "redo"
        };
        let body = json!({"designer_id": required_option(&options, "--designer-id")});
        return request_json(
            reqwest::Method::POST,
            format!(
                "{base_url}/api/workflow/design/{}/{action}",
                required_option(&options, "--workflow-id")
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
        "workflow.operator_dashboard" => {
            validate_options(&options, &["--base-url", "--limit"]);
            let limit = options
                .get("--limit")
                .map(|value| value.parse::<i64>().unwrap_or_else(|_| usage()))
                .unwrap_or(100);
            (
                reqwest::Method::GET,
                format!("/api/workflow/operator/dashboard?limit={limit}"),
            )
        }
        "workflow.visibility_snapshot" => {
            validate_options(&options, &["--base-url"]);
            (reqwest::Method::GET, "/api/workflow/visibility".to_owned())
        }
        "workflow.scheduler_timeline" => {
            validate_options(&options, &["--base-url", "--run-id", "--limit"]);
            let limit = options
                .get("--limit")
                .map(|value| value.parse::<i64>().unwrap_or_else(|_| usage()))
                .unwrap_or(200);
            let run_query = options
                .get("--run-id")
                .map(|run_id| format!("run_id={run_id}&"))
                .unwrap_or_default();
            (
                reqwest::Method::GET,
                format!("/api/workflow/scheduler/timeline?{run_query}limit={limit}"),
            )
        }
        "workflow.budget_snapshot" => {
            validate_options(&options, &["--base-url"]);
            (reqwest::Method::GET, "/api/workflow/budget".to_owned())
        }
        "workflow.budget_history" => {
            validate_options(&options, &["--base-url", "--limit"]);
            let limit = options
                .get("--limit")
                .map(|value| value.parse::<i64>().unwrap_or_else(|_| usage()))
                .unwrap_or(200);
            (
                reqwest::Method::GET,
                format!("/api/workflow/budget/history?limit={limit}"),
            )
        }
        "workflow.tool_audit" => {
            validate_options(&options, &["--base-url", "--conversation-id", "--limit"]);
            let limit = options
                .get("--limit")
                .map(|value| value.parse::<i64>().unwrap_or_else(|_| usage()))
                .unwrap_or(200);
            let conversation_query = options
                .get("--conversation-id")
                .map(|conversation_id| format!("conversation_id={conversation_id}&"))
                .unwrap_or_default();
            (
                reqwest::Method::GET,
                format!("/api/workflow/tools/audit?{conversation_query}limit={limit}"),
            )
        }
        "workflow.service_list" => {
            validate_options(&options, &["--base-url", "--limit"]);
            let limit = options
                .get("--limit")
                .map(|value| value.parse::<i64>().unwrap_or_else(|_| usage()))
                .unwrap_or(200);
            (
                reqwest::Method::GET,
                format!("/api/workflow/services?limit={limit}"),
            )
        }
        "workflow.service_get" => {
            validate_options(&options, &["--base-url", "--service-id"]);
            (
                reqwest::Method::GET,
                format!(
                    "/api/workflow/services/{}",
                    required_option(&options, "--service-id")
                ),
            )
        }
        "workflow.service_events" => {
            validate_options(&options, &["--base-url", "--service-id", "--limit"]);
            let service_id = required_option(&options, "--service-id");
            let limit = options
                .get("--limit")
                .map(|value| value.parse::<i64>().unwrap_or_else(|_| usage()))
                .unwrap_or(500);
            (
                reqwest::Method::GET,
                format!("/api/workflow/services/{service_id}/events?limit={limit}"),
            )
        }
        "workflow.service_enable" | "workflow.service_disable" => {
            validate_options(&options, &["--base-url", "--service-id"]);
            let service_id = required_option(&options, "--service-id");
            let action = if command == "workflow.service_enable" {
                "enable"
            } else {
                "disable"
            };
            (
                reqwest::Method::POST,
                format!("/api/workflow/services/{service_id}/{action}"),
            )
        }
        "workflow.services_repair" => {
            validate_options(&options, &["--base-url", "--limit"]);
            let limit = options
                .get("--limit")
                .map(|value| value.parse::<i64>().unwrap_or_else(|_| usage()))
                .unwrap_or(10000);
            (
                reqwest::Method::POST,
                format!("/api/workflow/services/repair?limit={limit}"),
            )
        }
        "workflow.message_orphans_repair" => {
            validate_options(&options, &["--base-url", "--inbox-id", "--limit"]);
            let limit = options
                .get("--limit")
                .map(|value| value.parse::<i64>().unwrap_or_else(|_| usage()))
                .unwrap_or(100);
            let inbox_query = options
                .get("--inbox-id")
                .map(|inbox_id| format!("inbox_id={inbox_id}&"))
                .unwrap_or_default();
            (
                reqwest::Method::POST,
                format!("/api/workflow/messages/repair-orphans?{inbox_query}limit={limit}"),
            )
        }
        "workflow.dead_letters" => {
            validate_options(&options, &["--base-url", "--limit"]);
            let limit = options
                .get("--limit")
                .map(|value| value.parse::<i64>().unwrap_or_else(|_| usage()))
                .unwrap_or(100);
            (
                reqwest::Method::GET,
                format!("/api/workflow/dead-letters?limit={limit}"),
            )
        }
        "workflow.dead_letter_replay" => {
            validate_options(&options, &["--base-url", "--run-id"]);
            (
                reqwest::Method::POST,
                format!(
                    "/api/workflow/dead-letters/{}/replay",
                    required_option(&options, "--run-id")
                ),
            )
        }
        "workflow.design_history" => {
            validate_options(&options, &["--base-url", "--workflow-id"]);
            (
                reqwest::Method::GET,
                format!(
                    "/api/workflow/design/{}/history",
                    required_option(&options, "--workflow-id")
                ),
            )
        }
        "workflow.capabilities_snapshot" => {
            validate_options(&options, &["--base-url", "--subject"]);
            let subject_query = options
                .get("--subject")
                .map(|subject| format!("?subject={subject}"))
                .unwrap_or_default();
            (
                reqwest::Method::GET,
                format!("/api/workflow/capabilities{subject_query}"),
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
