#[test]
fn ocr_change_event_payload_matches_shared_python_model_fixture() {
    let fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../contracts/golden/document-ocr-change-event-payload-v1.json"
    )))
    .expect("valid OCR change-event fixture");
    let input: kogwistar_contracts::DocumentUpsertV1 =
        serde_json::from_value(fixture["input"].clone()).expect("valid document input");
    let payload = document_upsert_change_payload(&input).expect("valid document payload");

    assert_eq!(payload, fixture["expected_payload"]);
}

#[tokio::test]
async fn change_event_publisher_posts_python_compatible_event_without_blocking_caller() {
    use crate::change_events::ChangeEventPublisher;
    use axum::{
        Json, Router,
        routing::post,
    };
    use tokio::sync::oneshot;

    let (event_tx, event_rx) = oneshot::channel();
    let event_tx = Arc::new(Mutex::new(Some(event_tx)));
    let app = Router::new().route(
        "/ingest",
        post({
            let event_tx = Arc::clone(&event_tx);
            move |Json(event): Json<Value>| {
                let event_tx = Arc::clone(&event_tx);
                async move {
                    if let Some(tx) = event_tx.lock().unwrap().take() {
                        let _ = tx.send(event);
                    }
                    StatusCode::OK
                }
            }
        }),
    );
    let listener = bind_reachable_loopback_listener().await;
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let publisher = ChangeEventPublisher::new(Some(endpoint), 1);
    let publish_started = std::time::Instant::now();
    publisher.publish(
        "doc.upsert",
        json!({"id":"doc-ocr","type":"ocr_document"}),
        json!({
            "kind":"doc_node",
            "id":"doc-ocr",
            "kg_graph_type":"knowledge",
            "url":null
        }),
    );
    assert!(publish_started.elapsed() < std::time::Duration::from_millis(500));
    let event = tokio::time::timeout(std::time::Duration::from_secs(3), event_rx)
        .await
        .expect("publisher should deliver within bounded timeout")
        .expect("event should be received");

    assert_eq!(event["op"], "doc.upsert");
    assert_eq!(event["seq"], 1);
    assert_eq!(event["entity"]["id"], "doc-ocr");
    assert_eq!(event["payload"]["type"], "ocr_document");
    assert!(event["ts_unix_ms"].as_i64().unwrap() > 0);
    assert!(event["run_id"].is_null());
    assert!(event["step_id"].is_null());
}
