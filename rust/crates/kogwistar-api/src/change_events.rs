//! Best-effort counterpart to Python `kogwistar.cdc.ChangeBus`'s HTTP sink.
//! Notifications are never canonical storage and never affect write results.

use serde_json::{Value, json};
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
    mpsc::{self, SyncSender},
};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[derive(Clone)]
pub(crate) struct ChangeEventPublisher {
    sequence: Arc<AtomicU64>,
    sender: Option<SyncSender<Value>>,
}

impl ChangeEventPublisher {
    pub(crate) fn from_env() -> Self {
        let endpoint = std::env::var("CDC_PUBLISH_ENDPOINT")
            .ok()
            .filter(|value| !value.trim().is_empty());
        Self::new(endpoint, 5_000)
    }

    pub(crate) fn new(endpoint: Option<String>, capacity: usize) -> Self {
        let Some(endpoint) = endpoint else {
            return Self {
                sequence: Arc::new(AtomicU64::new(0)),
                sender: None,
            };
        };
        let (sender, receiver) = mpsc::sync_channel(capacity.max(1));
        let endpoint = format!("{}/ingest", endpoint.trim_end_matches('/'));
        let worker = std::thread::Builder::new()
            .name("kogwistar-change-http-sink".to_owned())
            .spawn(move || {
                let Ok(runtime) = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                else {
                    return;
                };
                runtime.block_on(async move {
                    let Ok(client) = reqwest::Client::builder()
                        .timeout(Duration::from_millis(2_500))
                        .build()
                    else {
                        return;
                    };
                    while let Ok(event) = receiver.recv() {
                        let _ = client.post(&endpoint).json(&event).send().await;
                    }
                });
            });
        Self {
            sequence: Arc::new(AtomicU64::new(0)),
            sender: worker.ok().map(|_| sender),
        }
    }

    pub(crate) fn publish(&self, op: &str, payload: Value, entity: Value) {
        let Some(sender) = &self.sender else {
            return;
        };
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
            .min(i64::MAX as u128) as i64;
        let event = json!({
            "seq": self.sequence.fetch_add(1, Ordering::Relaxed) + 1,
            "op": op,
            "ts_unix_ms": timestamp,
            "entity": entity,
            "payload": payload,
            "run_id": null,
            "step_id": null
        });
        // Mirror Python's bounded, non-blocking sink queue: a slow sink may lose
        // notifications, but cannot delay or roll back the canonical write.
        let _ = sender.try_send(event);
    }
}
