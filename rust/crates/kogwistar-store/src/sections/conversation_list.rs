// Incremental conversation-list projection over authorized graph pages. Python parity: conversation listing in engine_core read/query paths.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConversationListEntry {
    pub id: String,
    pub start_node_id: String,
    pub status: String,
    pub turn_count: i64,
}

/// Incrementally projects rows without retaining every graph node.
#[derive(Clone, Debug, Default)]
pub struct ConversationListAccumulator {
    user_id: String,
    turn_counts: BTreeMap<String, i64>,
    starts: Vec<(String, String, String)>,
}

impl ConversationListAccumulator {
    pub fn new(user_id: impl Into<String>) -> Self {
        Self {
            user_id: user_id.into(),
            ..Self::default()
        }
    }

    /// Add one page after the caller has applied namespace and ACL filtering.
    pub fn extend_visible_records(&mut self, records: &[GraphRecord]) {
        for record in records
            .iter()
            .filter(|record| !is_tombstoned(&record.metadata))
        {
            match value_string(record, "entity_type").as_deref() {
                Some("conversation_turn" | "assistant_turn")
                    if record_field(record, "turn_index")
                        .and_then(|value| value.as_i64())
                        .is_some() =>
                {
                    if let Some(conversation_id) = value_string(record, "conversation_id") {
                        *self.turn_counts.entry(conversation_id).or_default() += 1;
                    }
                }
                Some("conversation_start")
                    if value_string(record, "user_id").as_deref()
                        == Some(self.user_id.as_str()) =>
                {
                    if let Some(conversation_id) = value_string(record, "conversation_id") {
                        let status =
                            value_string(record, "status").unwrap_or_else(|| "active".to_owned());
                        self.starts
                            .push((conversation_id, record.id.clone(), status));
                    }
                }
                _ => {}
            }
        }
    }

    pub fn finish(self) -> Vec<ConversationListEntry> {
        let mut entries = self
            .starts
            .into_iter()
            .map(
                |(conversation_id, start_node_id, status)| ConversationListEntry {
                    turn_count: self
                        .turn_counts
                        .get(&conversation_id)
                        .copied()
                        .unwrap_or_default(),
                    id: conversation_id,
                    start_node_id,
                    status,
                },
            )
            .collect::<Vec<_>>();
        entries.sort_by_key(|entry| std::cmp::Reverse(entry.turn_count));
        entries
    }
}

/// Keep only conversation-list fields; discard source documents and embeddings.
pub fn compact_conversation_list_candidates(records: &[GraphRecord]) -> Vec<GraphRecord> {
    const FIELDS: &[&str] = &[
        "entity_type",
        "turn_index",
        "conversation_id",
        "user_id",
        "status",
        "lifecycle_status",
    ];
    records
        .iter()
        .filter_map(|record| {
            let entity_type = value_string(record, "entity_type")?;
            if !matches!(
                entity_type.as_str(),
                "conversation_start" | "conversation_turn" | "assistant_turn"
            ) {
                return None;
            }
            let metadata = FIELDS
                .iter()
                .filter_map(|field| {
                    record_field(record, field).map(|value| ((*field).to_owned(), value))
                })
                .collect();
            Some(GraphRecord {
                id: record.id.clone(),
                document: None,
                metadata,
                embedding: None,
            })
        })
        .collect()
}
