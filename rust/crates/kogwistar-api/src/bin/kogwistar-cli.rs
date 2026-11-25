use kogwistar_api::{HealthSnapshot, cli_health, validate_server_backend};
use serde_json::Value;
use std::process::ExitCode;

fn usage() -> ! {
    eprintln!(
        "usage: kogwistar-cli health [backend] | api conversation conversation.get_transcript --conversation-id ID [--base-url URL]"
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

fn transcript_command(mut args: impl Iterator<Item = String>) -> ExitCode {
    let mut conversation_id = None;
    let mut base_url =
        std::env::var("KOGWISTAR_BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:8787".to_owned());
    while let Some(option) = args.next() {
        let Some(value) = args.next() else {
            usage();
        };
        match option.as_str() {
            "--conversation-id" => conversation_id = Some(value),
            "--base-url" => base_url = value,
            _ => usage(),
        }
    }
    let Some(conversation_id) = conversation_id else {
        usage();
    };
    let url = format!(
        "{}/api/conversations/{conversation_id}/turns",
        base_url.trim_end_matches('/')
    );
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
        let mut request = client.get(url);
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
            eprintln!("transcript request failed: {error}");
            ExitCode::FAILURE
        }
    }
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
        Some("api")
            if args.next().as_deref() == Some("conversation")
                && args.next().as_deref() == Some("conversation.get_transcript") =>
        {
            transcript_command(args)
        }
        _ => usage(),
    }
}
