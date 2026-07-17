#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    kogwistar_api::run_server_from_environment().await
}
