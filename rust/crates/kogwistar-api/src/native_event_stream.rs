use kogwistar_contracts::{ProviderEventV1, provider_event_from_str};
use serde_json::{Value, json};
use std::collections::BTreeSet;

const MAX_STREAM_BYTES: usize = 1024 * 1024;
const MAX_EVENT_BYTES: usize = 256 * 1024;
const MAX_COMPLETED_FIELD_EVENTS: usize = 256;

/// Normalizes OpenAI-compatible Chat Completions SSE into the shared event contract.
/// It deliberately does not execute tool calls; streamed tool arguments need a
/// separate authorization and completion boundary before they can be dispatched.
pub(crate) struct ChatCompletionEventStream {
    buffer: Vec<u8>,
    response_bytes: usize,
    sequence: i64,
    response_id: Option<String>,
    text: String,
    completed_fields: BTreeSet<String>,
    field_scanner: JsonFieldScanner,
    started: bool,
    completed: bool,
    saw_tool_calls: bool,
}

impl Default for ChatCompletionEventStream {
    fn default() -> Self {
        Self {
            buffer: Vec::new(),
            response_bytes: 0,
            sequence: 0,
            response_id: None,
            text: String::new(),
            completed_fields: BTreeSet::new(),
            field_scanner: JsonFieldScanner::default(),
            started: false,
            completed: false,
            saw_tool_calls: false,
        }
    }
}

impl ChatCompletionEventStream {
    pub(crate) fn push(&mut self, bytes: &[u8]) -> Result<Vec<ProviderEventV1>, String> {
        self.response_bytes = self.response_bytes.saturating_add(bytes.len());
        if self.response_bytes > MAX_STREAM_BYTES {
            return Err("provider event stream exceeds the configured size limit".to_owned());
        }
        self.buffer.extend_from_slice(bytes);
        let mut output = Vec::new();
        while let Some((frame_end, delimiter_len)) = frame_boundary(&self.buffer) {
            if frame_end > MAX_EVENT_BYTES {
                return Err("provider SSE event exceeds the configured size limit".to_owned());
            }
            let frame = self.buffer.drain(..frame_end).collect::<Vec<_>>();
            self.buffer.drain(..delimiter_len);
            self.process_frame(&frame, &mut output)?;
        }
        if self.buffer.len() > MAX_EVENT_BYTES {
            return Err("provider SSE event exceeds the configured size limit".to_owned());
        }
        Ok(output)
    }

    pub(crate) fn finish(&mut self) -> Result<Vec<ProviderEventV1>, String> {
        let mut output = Vec::new();
        if !self.buffer.is_empty() {
            let frame = std::mem::take(&mut self.buffer);
            self.process_frame(&frame, &mut output)?;
        }
        if !self.completed && self.started {
            self.complete(&mut output, None);
        }
        Ok(output)
    }

    fn process_frame(
        &mut self,
        frame: &[u8],
        output: &mut Vec<ProviderEventV1>,
    ) -> Result<(), String> {
        let mut data = Vec::new();
        for line in frame.split(|byte| *byte == b'\n') {
            let line = line.strip_suffix(b"\r").unwrap_or(line);
            if let Some(value) = line.strip_prefix(b"data:") {
                let value = value.strip_prefix(b" ").unwrap_or(value);
                if !data.is_empty() {
                    data.push(b'\n');
                }
                data.extend_from_slice(value);
            }
        }
        if data.is_empty() {
            return Ok(());
        }
        if data == b"[DONE]" {
            if self.saw_tool_calls {
                self.fail(
                    output,
                    "streamed_tool_calls_unsupported",
                    "provider emitted streamed tool arguments",
                );
            } else {
                self.complete(output, None);
            }
            return Ok(());
        }
        let value: Value = serde_json::from_slice(&data)
            .map_err(|_| "provider SSE data is not valid JSON".to_owned())?;
        if let Some(error) = value.get("error") {
            let message = error
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("provider stream failed");
            self.fail(output, "provider_stream_error", message);
            return Ok(());
        }
        if let Some(id) = value.get("id").and_then(Value::as_str) {
            self.response_id = Some(id.to_owned());
        }
        if !self.started {
            self.started = true;
            output.push(self.event(json!({
                "kind": "response_started",
                "response_id": self.response_id,
            }))?);
        }
        let Some(choice) = value
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|v| v.first())
        else {
            return Ok(());
        };
        if choice.pointer("/delta/tool_calls").is_some() {
            self.saw_tool_calls = true;
        }
        if let Some(delta) = choice.pointer("/delta/content").and_then(Value::as_str) {
            self.text.push_str(delta);
            if self.text.len() > MAX_STREAM_BYTES {
                return Err("provider response text exceeds the configured size limit".to_owned());
            }
            output.push(self.event(json!({
                "kind": "text_delta",
                "response_id": self.response_id,
                "delta": delta,
            }))?);
            for (field_name, field_value) in self.field_scanner.scan(&self.text) {
                if self.completed_fields.len() < MAX_COMPLETED_FIELD_EVENTS
                    && self.completed_fields.insert(field_name.clone())
                {
                    output.push(self.event(json!({
                        "kind": "field_completed",
                        "response_id": self.response_id,
                        "field_name": field_name,
                        "value": field_value,
                    }))?);
                }
            }
        }
        if choice
            .get("finish_reason")
            .is_some_and(|reason| !reason.is_null())
        {
            if self.saw_tool_calls {
                self.fail(
                    output,
                    "streamed_tool_calls_unsupported",
                    "provider emitted streamed tool arguments",
                );
            } else {
                self.complete(output, None);
            }
        }
        Ok(())
    }

    fn complete(&mut self, output: &mut Vec<ProviderEventV1>, tokens: Option<(i64, i64)>) {
        if self.completed {
            return;
        }
        self.completed = true;
        let structured_output = serde_json::from_str::<Value>(&self.text)
            .ok()
            .and_then(|value| value.as_object().cloned())
            .map(|object| {
                object
                    .into_iter()
                    .collect::<std::collections::BTreeMap<_, _>>()
            });
        let (input_tokens, output_tokens) =
            tokens.map_or((None, None), |(input, output)| (Some(input), Some(output)));
        if let Ok(event) = self.event(json!({
            "kind": "response_completed",
            "response_id": self.response_id,
            "text": self.text,
            "structured_output": structured_output,
            "has_tool_calls": false,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        })) {
            output.push(event);
        }
    }

    fn fail(&mut self, output: &mut Vec<ProviderEventV1>, code: &str, message: &str) {
        if self.completed {
            return;
        }
        self.completed = true;
        if let Ok(event) = self.event(json!({
            "kind": "response_failed",
            "response_id": self.response_id,
            "error_code": code,
            "message": message,
            "retryable": false,
        })) {
            output.push(event);
        }
    }

    fn event(&mut self, mut value: Value) -> Result<ProviderEventV1, String> {
        self.sequence += 1;
        let object = value
            .as_object_mut()
            .ok_or_else(|| "normalized provider event is not an object".to_owned())?;
        object.insert("provider_sequence".to_owned(), json!(self.sequence));
        if let Some(id) = &self.response_id {
            object.insert(
                "provider_event_id".to_owned(),
                json!(format!("{id}:{}", self.sequence)),
            );
        }
        provider_event_from_str(&value.to_string())
            .map_err(|_| "normalized provider event failed contract validation".to_owned())
    }
}

#[derive(Default)]
struct JsonFieldScanner {
    cursor: usize,
    state: JsonFieldScanState,
}

#[derive(Default)]
enum JsonFieldScanState {
    #[default]
    Start,
    KeyOrEnd,
    Key {
        start: usize,
        escaped: bool,
    },
    Colon {
        key: String,
    },
    ValueStart {
        key: String,
    },
    Value {
        key: String,
        start: usize,
        depth: usize,
        in_string: bool,
        escaped: bool,
    },
    Done,
    Failed,
}

impl JsonFieldScanner {
    // Advance only across bytes not visited by earlier deltas. serde_json still
    // validates each completed value before it becomes a semantic event.
    fn scan(&mut self, text: &str) -> Vec<(String, Value)> {
        let bytes = text.as_bytes();
        let mut fields = Vec::new();
        loop {
            let state = std::mem::take(&mut self.state);
            match state {
                JsonFieldScanState::Start => match bytes.get(self.cursor) {
                    Some(byte) if is_json_whitespace(*byte) => {
                        self.cursor += 1;
                        self.state = JsonFieldScanState::Start;
                    }
                    Some(b'{') => {
                        self.cursor += 1;
                        self.state = JsonFieldScanState::KeyOrEnd;
                    }
                    Some(_) => self.state = JsonFieldScanState::Failed,
                    None => {
                        self.state = JsonFieldScanState::Start;
                        break;
                    }
                },
                JsonFieldScanState::KeyOrEnd => match bytes.get(self.cursor) {
                    Some(byte) if is_json_whitespace(*byte) => {
                        self.cursor += 1;
                        self.state = JsonFieldScanState::KeyOrEnd;
                    }
                    Some(b'"') => {
                        let start = self.cursor;
                        self.cursor += 1;
                        self.state = JsonFieldScanState::Key {
                            start,
                            escaped: false,
                        };
                    }
                    Some(b'}') => {
                        self.cursor += 1;
                        self.state = JsonFieldScanState::Done;
                    }
                    Some(_) => self.state = JsonFieldScanState::Failed,
                    None => {
                        self.state = JsonFieldScanState::KeyOrEnd;
                        break;
                    }
                },
                JsonFieldScanState::Key { start, mut escaped } => {
                    let Some(byte) = bytes.get(self.cursor).copied() else {
                        self.state = JsonFieldScanState::Key { start, escaped };
                        break;
                    };
                    self.cursor += 1;
                    if escaped {
                        escaped = false;
                    } else if byte == b'\\' {
                        escaped = true;
                    } else if byte == b'"' {
                        match serde_json::from_str::<String>(&text[start..self.cursor]) {
                            Ok(key) => self.state = JsonFieldScanState::Colon { key },
                            Err(_) => self.state = JsonFieldScanState::Failed,
                        }
                        continue;
                    }
                    self.state = JsonFieldScanState::Key { start, escaped };
                }
                JsonFieldScanState::Colon { key } => match bytes.get(self.cursor) {
                    Some(byte) if is_json_whitespace(*byte) => {
                        self.cursor += 1;
                        self.state = JsonFieldScanState::Colon { key };
                    }
                    Some(b':') => {
                        self.cursor += 1;
                        self.state = JsonFieldScanState::ValueStart { key };
                    }
                    Some(_) => self.state = JsonFieldScanState::Failed,
                    None => {
                        self.state = JsonFieldScanState::Colon { key };
                        break;
                    }
                },
                JsonFieldScanState::ValueStart { key } => match bytes.get(self.cursor) {
                    Some(byte) if is_json_whitespace(*byte) => {
                        self.cursor += 1;
                        self.state = JsonFieldScanState::ValueStart { key };
                    }
                    Some(_) => {
                        self.state = JsonFieldScanState::Value {
                            key,
                            start: self.cursor,
                            depth: 0,
                            in_string: false,
                            escaped: false,
                        };
                    }
                    None => {
                        self.state = JsonFieldScanState::ValueStart { key };
                        break;
                    }
                },
                JsonFieldScanState::Value {
                    key,
                    start,
                    mut depth,
                    mut in_string,
                    mut escaped,
                } => {
                    let Some(byte) = bytes.get(self.cursor).copied() else {
                        self.state = JsonFieldScanState::Value {
                            key,
                            start,
                            depth,
                            in_string,
                            escaped,
                        };
                        break;
                    };
                    if in_string {
                        if escaped {
                            escaped = false;
                        } else if byte == b'\\' {
                            escaped = true;
                        } else if byte == b'"' {
                            in_string = false;
                        }
                        self.cursor += 1;
                    } else {
                        match byte {
                            b'"' => {
                                in_string = true;
                                self.cursor += 1;
                            }
                            b'{' | b'[' => {
                                depth += 1;
                                self.cursor += 1;
                            }
                            b']' if depth == 0 => {
                                self.state = JsonFieldScanState::Failed;
                                continue;
                            }
                            b']' => {
                                depth -= 1;
                                self.cursor += 1;
                            }
                            b'}' if depth == 0 => {
                                if let Ok(value) =
                                    serde_json::from_str::<Value>(&text[start..self.cursor])
                                {
                                    fields.push((key, value));
                                    self.cursor += 1;
                                    self.state = JsonFieldScanState::Done;
                                } else {
                                    self.state = JsonFieldScanState::Failed;
                                }
                                continue;
                            }
                            b'}' => {
                                depth -= 1;
                                self.cursor += 1;
                            }
                            b',' if depth == 0 => {
                                if let Ok(value) =
                                    serde_json::from_str::<Value>(&text[start..self.cursor])
                                {
                                    fields.push((key, value));
                                    self.cursor += 1;
                                    self.state = JsonFieldScanState::KeyOrEnd;
                                } else {
                                    self.state = JsonFieldScanState::Failed;
                                }
                                continue;
                            }
                            _ => self.cursor += 1,
                        }
                    }
                    self.state = JsonFieldScanState::Value {
                        key,
                        start,
                        depth,
                        in_string,
                        escaped,
                    };
                }
                JsonFieldScanState::Done => {
                    self.state = JsonFieldScanState::Done;
                    break;
                }
                JsonFieldScanState::Failed => {
                    self.state = JsonFieldScanState::Failed;
                    break;
                }
            }
        }
        fields
    }
}

fn is_json_whitespace(byte: u8) -> bool {
    matches!(byte, b' ' | b'\n' | b'\r' | b'\t')
}

fn frame_boundary(buffer: &[u8]) -> Option<(usize, usize)> {
    buffer
        .windows(2)
        .position(|window| window == b"\n\n")
        .map(|index| (index, 2))
        .or_else(|| {
            buffer
                .windows(4)
                .position(|window| window == b"\r\n\r\n")
                .map(|index| (index, 4))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_split_sse_frames_and_structured_json() {
        let mut parser = ChatCompletionEventStream::default();
        let frames = [
            r#"data: {"id":"resp-1","choices":[{"delta":{"content":"{\"answer\":\"café\"}"},"finish_reason":null}]}"#.as_bytes(),
            b"\r\n\r\ndata: {\"id\":\"resp-1\",\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\r\n\r\n",
        ];
        let mut events = Vec::new();
        for frame in frames {
            for chunk in frame.chunks(7) {
                events.extend(parser.push(chunk).unwrap());
            }
        }
        assert!(
            events
                .iter()
                .any(|event| matches!(event, ProviderEventV1::TextDelta { .. }))
        );
        assert!(
            events
                .iter()
                .any(|event| matches!(event, ProviderEventV1::ResponseCompleted { .. }))
        );
    }

    #[test]
    fn rejects_malformed_and_oversized_streams() {
        let mut malformed = ChatCompletionEventStream::default();
        assert!(malformed.push(b"data: {nope}\n\n").is_err());
        let mut oversized = ChatCompletionEventStream::default();
        assert!(oversized.push(&vec![b'x'; MAX_STREAM_BYTES + 1]).is_err());
    }

    #[test]
    fn tool_stream_is_explicitly_failed_closed() {
        let mut parser = ChatCompletionEventStream::default();
        let events = parser.push(b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{}]},\"finish_reason\":null}]}\n\n").unwrap();
        assert!(
            events.is_empty()
                || events
                    .iter()
                    .all(|event| !matches!(event, ProviderEventV1::ToolCallCompleted { .. }))
        );
        let events = parser.push(b"data: [DONE]\n\n").unwrap();
        assert!(events.iter().any(|event| matches!(event, ProviderEventV1::ResponseFailed { error_code, .. } if error_code == "streamed_tool_calls_unsupported")));
    }

    #[test]
    fn emits_a_field_only_after_its_complete_json_value_and_only_once() {
        let mut parser = ChatCompletionEventStream::default();
        let first = br#"data: {"id":"resp-fields","choices":[{"delta":{"content":"{\"answer\":\"complete"},"finish_reason":null}]}"#;
        let events = parser.push(&[first.as_slice(), b"\n\n"].concat()).unwrap();
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, ProviderEventV1::FieldCompleted { .. }))
        );

        let second = br#"data: {"id":"resp-fields","choices":[{"delta":{"content":"\",\"metadata\":{\"items\":[1,2]}}"},"finish_reason":null}]}"#;
        let events = parser.push(&[second.as_slice(), b"\n\n"].concat()).unwrap();
        let completed = events
            .iter()
            .filter_map(|event| match event {
                ProviderEventV1::FieldCompleted {
                    field_name, value, ..
                } => Some((field_name.as_str(), value)),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(completed.len(), 2);
        assert_eq!(completed[0], ("answer", &json!("complete")));
        assert_eq!(completed[1], ("metadata", &json!({"items": [1, 2]})));

        let repeated = parser.push(&[second.as_slice(), b"\n\n"].concat()).unwrap();
        assert!(
            !repeated
                .iter()
                .any(|event| matches!(event, ProviderEventV1::FieldCompleted { .. }))
        );
    }

    #[test]
    fn completed_field_scanner_handles_escaped_strings_and_nested_values() {
        let json = r#"{"quote":"say \"hi\"","nested":{"ok":true},"tail":[1,"雪"]}"#;
        let mut scanner = JsonFieldScanner::default();
        let fields = json
            .char_indices()
            .map(|(index, character)| index + character.len_utf8())
            .flat_map(|end| scanner.scan(&json[..end]))
            .collect::<Vec<_>>();
        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0], ("quote".to_owned(), json!("say \"hi\"")));
        assert_eq!(fields[1], ("nested".to_owned(), json!({"ok": true})));
        assert_eq!(fields[2], ("tail".to_owned(), json!([1, "雪"])));

        let mut incomplete = JsonFieldScanner::default();
        let incomplete_json = r#"{"partial":{"x":1"#;
        let incomplete_fields = incomplete_json
            .char_indices()
            .map(|(index, character)| index + character.len_utf8())
            .flat_map(|end| incomplete.scan(&incomplete_json[..end]))
            .collect::<Vec<_>>();
        assert!(incomplete_fields.is_empty());
    }
}
