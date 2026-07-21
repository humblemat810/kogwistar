use kogwistar_api::{HealthSnapshot, cli_health};

fn usage() -> ! {
    eprintln!("usage: kogwistar-cli health [backend]");
    std::process::exit(2)
}

fn main() {
    let mut args = std::env::args().skip(1);
    if args.next().as_deref() != Some("health") {
        usage();
    }
    let backend = args.next().unwrap_or_else(|| "in_memory".to_owned());
    if args.next().is_some() {
        usage();
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
