#[tokio::test]
async fn postgres_identity_claim_rejects_unregistered_principal_before_storage() {
    let service = PostgresRunApplicationService::from_dsn(
        "postgres://unused:unused@127.0.0.1:1/unused",
        "worker_identity_registration",
    )
    .unwrap();
    let recipient = runtime_worker_identity_recipient(
        "python-cpu-v1",
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    );
    let response = service
        .execute(ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: "/internal/runtime/claim".to_owned(),
            body: json!({
                "claimed_by":"unregistered-worker",
                "recipient_ids":[recipient]
            })
            .to_string()
            .into_bytes(),
            principal: json!({"sub":"unregistered-worker", "role":"reader"}),
        })
        .await;

    assert_eq!(response.status, StatusCode::FORBIDDEN.as_u16());
    let body: Value = serde_json::from_slice(&response.body).unwrap();
    assert_eq!(body["code"], "KOGWISTAR_WORKER_IDENTITY_FORBIDDEN");
}
