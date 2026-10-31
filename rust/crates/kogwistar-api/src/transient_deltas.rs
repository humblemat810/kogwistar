use serde_json::{Value, json};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

const MAX_RUNS: usize = 64;
const MAX_DELTAS_PER_RUN: usize = 64;
const MAX_DELTA_PAGE: usize = MAX_DELTAS_PER_RUN + 1;
const MAX_DELTA_CHARS: usize = 4096;

#[derive(Clone, Debug, PartialEq, Eq)]
struct TransientDelta {
    sequence: i64,
    delta: String,
}

#[derive(Default)]
struct TransientDeltaState {
    runs: HashMap<String, VecDeque<TransientDelta>>,
    run_order: VecDeque<String>,
}

/// Bounded process-local text deltas, deliberately separate from durable run/graph events.
#[derive(Clone, Default)]
pub(crate) struct TransientRunDeltas {
    state: Arc<Mutex<TransientDeltaState>>,
}

impl TransientRunDeltas {
    pub(crate) fn publish(&self, run_id: &str, sequence: i64, delta: &str) -> Result<bool, String> {
        if run_id.is_empty() || run_id.len() > 256 {
            return Err("transient delta run ID is invalid".to_owned());
        }
        if sequence < 1 {
            return Err("transient delta sequence must be positive".to_owned());
        }
        if delta.chars().count() > MAX_DELTA_CHARS {
            return Err("transient text delta exceeds the configured size limit".to_owned());
        }
        let mut state = self
            .state
            .lock()
            .map_err(|_| "transient delta buffer is unavailable".to_owned())?;
        if !state.runs.contains_key(run_id) {
            if state.runs.len() >= MAX_RUNS {
                if let Some(evicted) = state.run_order.pop_front() {
                    state.runs.remove(&evicted);
                }
            }
            state.run_order.push_back(run_id.to_owned());
            state.runs.insert(
                run_id.to_owned(),
                VecDeque::with_capacity(MAX_DELTAS_PER_RUN),
            );
        }
        let entries = state
            .runs
            .get_mut(run_id)
            .expect("run buffer was inserted above");
        if entries
            .back()
            .is_some_and(|previous| sequence <= previous.sequence)
        {
            return Ok(false);
        }
        if entries.len() == MAX_DELTAS_PER_RUN {
            entries.pop_front();
        }
        entries.push_back(TransientDelta {
            sequence,
            delta: delta.to_owned(),
        });
        Ok(true)
    }

    pub(crate) fn list(
        &self,
        run_id: &str,
        after_sequence: i64,
        limit: usize,
    ) -> Result<Vec<Value>, String> {
        let state = self
            .state
            .lock()
            .map_err(|_| "transient delta buffer is unavailable".to_owned())?;
        let Some(entries) = state.runs.get(run_id) else {
            return Ok(Vec::new());
        };
        let mut output = Vec::new();
        let page_limit = limit.clamp(1, MAX_DELTA_PAGE);
        if let Some(oldest) = entries.front()
            && after_sequence < oldest.sequence - 1
        {
            output.push(json!({
                "event_type": "response.text.delta_gap",
                "client_sse_sequence": oldest.sequence - 1,
                "dropped_through_sequence": oldest.sequence - 1,
            }));
        }
        let remaining = page_limit.saturating_sub(output.len());
        output.extend(
            entries
                .iter()
                .filter(|entry| entry.sequence > after_sequence)
                .take(remaining)
                .map(|entry| {
                    json!({
                        "event_type": "response.text.delta",
                        "client_sse_sequence": entry.sequence,
                        "delta": entry.delta,
                    })
                }),
        );
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transient_deltas_are_bounded_deduplicated_and_report_gaps() {
        let deltas = TransientRunDeltas::default();
        for sequence in 1..=MAX_DELTAS_PER_RUN as i64 + 2 {
            assert!(deltas.publish("run", sequence, "chunk").unwrap());
        }
        assert!(
            !deltas
                .publish("run", MAX_DELTAS_PER_RUN as i64, "retry")
                .unwrap()
        );
        let rows = deltas.list("run", 0, 1000).unwrap();
        assert_eq!(rows.len(), MAX_DELTAS_PER_RUN + 1);
        assert_eq!(rows[0]["event_type"], "response.text.delta_gap");
        assert_eq!(rows[0]["client_sse_sequence"], 2);
        assert_eq!(rows[1]["client_sse_sequence"], 3);
        assert_eq!(rows.last().unwrap()["client_sse_sequence"], 66);
        assert_eq!(deltas.list("run", 0, 1).unwrap().len(), 1);
        assert_eq!(deltas.list("run", 0, 2).unwrap().len(), 2);
    }

    #[test]
    fn transient_delta_buffers_are_isolated_and_validate_size_and_sequence() {
        let deltas = TransientRunDeltas::default();
        assert!(deltas.publish("run-a", 1, "a").unwrap());
        assert!(deltas.list("run-b", 0, 64).unwrap().is_empty());
        assert!(deltas.publish("run-a", 0, "bad").is_err());
        assert!(
            deltas
                .publish("run-a", 2, &"x".repeat(MAX_DELTA_CHARS + 1))
                .is_err()
        );
    }

    #[test]
    fn transient_run_count_is_bounded_by_oldest_run_eviction() {
        let deltas = TransientRunDeltas::default();
        for index in 0..=MAX_RUNS {
            deltas.publish(&format!("run-{index}"), 1, "chunk").unwrap();
        }
        assert!(deltas.list("run-0", 0, 64).unwrap().is_empty());
        assert_eq!(deltas.list("run-64", 0, 64).unwrap().len(), 1);
    }
}
