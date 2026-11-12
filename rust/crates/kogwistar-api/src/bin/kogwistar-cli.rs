use kogwistar_api::{HealthSnapshot, cli_health, validate_server_backend};

fn usage() -> ! {
    eprintln!("usage: kogwistar-cli health [backend]");
    std::process::exit(2)
}

fn main() {
    let mut args = std::env::args().skip(1);
    if args.next().as_deref() != Some("health") {
        usage();
    }
    let backend = args.next().unwrap_or_else(|| "sqlite".to_owned());
    if args.next().is_some() {
        usage();
    }
    if let Err(error) = validate_server_backend(&backend) {
        eprintln!("{error}");
        std::process::exit(2);
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
}
