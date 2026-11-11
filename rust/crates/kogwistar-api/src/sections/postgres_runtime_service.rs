const EMBEDDING_PROFILE_NAMESPACE: &str = "__kogwistar_embedding_profiles_v1__";
const EMBEDDING_PROFILE_SCHEMA_VERSION: i64 = 1;

#[derive(Clone, Copy)]
// PostgreSQL run-service construction and shared native read/context helpers.
// Python counterparts: kogwistar.server.chat_service.ChatRunService and its
// conversation query helpers in kogwistar.server.chat_service_conversation_queries.
enum RuntimeGraphCommitDenial {
    Owner,
    Capability,
    Scope,
}

enum RuntimeGraphMutationBatch<'a> {
    V1(&'a GraphMutationBatchInput),
    V2(&'a kogwistar_contracts::GraphMutationBatchV2),
}

fn validate_provider_event_publications(
    publications: &[kogwistar_contracts::DurableProviderEventPublicationV1],
    mutations: &[GraphMutation],
    run_id: &str,
) -> Result<(), String> {
    let mut published_ids = BTreeSet::new();
    for publication in publications {
        let event = &publication.event;
        if publication.namespace.trim().is_empty()
            || publication.conversation_id.trim().is_empty()
            || event.event_id.trim().is_empty()
            || event.event_id.len() > 255
            || event.trace.run_id != run_id
            || matches!(
                &event.event,
                kogwistar_contracts::ProviderEventV1::TextDelta { .. }
            )
        {
            return Err(
                "provider event publication is invalid, belongs to another run, or contains a transient token delta"
                    .to_owned(),
            );
        }
        let message_id = format!("msg:{}", event.event_id);
        if !published_ids.insert(message_id.clone()) {
            return Err("provider event publication contains a duplicate event id".to_owned());
        }
        let matching_nodes = mutations
            .iter()
            .filter(|mutation| mutation.entity_kind == "node" && mutation.record.id == message_id)
            .collect::<Vec<_>>();
        if matching_nodes.len() != 1 {
            return Err(
                "provider event must have exactly one matching graph message node".to_owned(),
            );
        }
        let node = matching_nodes[0];
        if node.scope.graph_space.as_deref() != Some("conversation") {
            return Err(
                "provider event graph node must use the conversation graph space".to_owned(),
            );
        }
        if node.scope.namespace != publication.namespace {
            return Err("provider event namespace must match its graph mutation scope".to_owned());
        }
        let metadata = &node.record.metadata;
        let expected_payload = serde_json::to_value(&publication.event)
            .map_err(|_| "provider event payload is invalid".to_owned())?;
        let graph_payload = metadata
            .get("payload_json")
            .and_then(Value::as_str)
            .and_then(|value| serde_json::from_str::<Value>(value).ok());
        let fields_match = [
            ("artifact_kind", "lane_message"),
            ("conversation_id", publication.conversation_id.as_str()),
            ("inbox_id", "agent-provider-events"),
            ("recipient_id", "agent-event-consumer"),
            ("sender_id", "agent-runtime"),
            ("msg_type", "agent.provider_event"),
            ("purpose", "agent_event"),
            ("run_id", event.trace.run_id.as_str()),
            ("step_id", event.trace.step_id.as_str()),
            ("correlation_id", event.event_id.as_str()),
            ("idempotency_key", event.event_id.as_str()),
            ("visibility", "private"),
        ]
        .into_iter()
        .all(|(key, expected)| metadata.get(key).and_then(Value::as_str) == Some(expected));
        if !fields_match
            || metadata.get("shared_scope").and_then(Value::as_bool) != Some(false)
            || graph_payload.as_ref() != Some(&expected_payload)
        {
            return Err(
                "provider event graph node does not match its durable publication".to_owned(),
            );
        }
        let has_conversation_edge = mutations.iter().any(|mutation| {
            mutation.entity_kind == "edge"
                && mutation.payload.get("relation").and_then(Value::as_str)
                    == Some("in_conversation")
                && mutation.scope.namespace == publication.namespace
                && mutation.scope.graph_space.as_deref() == Some("conversation")
                && mutation
                    .payload
                    .get("source_ids")
                    .and_then(Value::as_array)
                    .is_some_and(|ids| {
                        ids.len() == 1 && ids[0].as_str() == Some(message_id.as_str())
                    })
                && mutation
                    .payload
                    .get("target_ids")
                    .and_then(Value::as_array)
                    .is_some_and(|ids| ids.len() == 1)
                && mutation
                    .record
                    .metadata
                    .get("conversation_id")
                    .and_then(Value::as_str)
                    == Some(publication.conversation_id.as_str())
        });
        if !has_conversation_edge {
            return Err(
                "provider event graph batch must link the event into its conversation".to_owned(),
            );
        }
    }
    let graph_event_ids = mutations
        .iter()
        .filter(|mutation| {
            mutation.entity_kind == "node"
                && mutation
                    .record
                    .metadata
                    .get("msg_type")
                    .and_then(Value::as_str)
                    == Some("agent.provider_event")
        })
        .map(|mutation| mutation.record.id.clone())
        .collect::<BTreeSet<_>>();
    if graph_event_ids != published_ids {
        return Err(
            "provider event graph nodes and lane publications must match exactly".to_owned(),
        );
    }
    Ok(())
}

fn validate_provider_event_acknowledgements(
    acknowledgements: &[kogwistar_runtime::ProviderEventAcknowledgementV1],
) -> Result<(), &'static str> {
    if acknowledgements.len() > 100 {
        return Err("provider event acknowledgement batch exceeds 100 entries");
    }
    let mut seen = std::collections::BTreeSet::new();
    for acknowledgement in acknowledgements {
        if acknowledgement.event_id.trim().is_empty()
            || acknowledgement.claimed_by.trim().is_empty()
            || acknowledgement.message_id != format!("msg:{}", acknowledgement.event_id)
        {
            return Err("provider event acknowledgement identity is invalid");
        }
        if !seen.insert(acknowledgement.event_id.as_str()) {
            return Err("provider event acknowledgement event IDs must be unique");
        }
    }
    Ok(())
}

enum NativeEvidenceContextError {
    CallerNotRunOwner,
    OwnerAuthorization,
    GraphReadRejected,
    GraphReadUnavailable,
}

fn runtime_owner_binding_v1(
    principal: &Value,
    capabilities: &SharedCapabilityState,
) -> Option<RuntimeOwnerBindingV1> {
    let user_id = authenticated_run_owner_id(principal)?.to_owned();
    let state = capabilities
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let admitted_capabilities = effective_capabilities(principal, &state);
    let mut admitted_scope = serde_json::Map::new();
    for key in [
        "ns",
        "storage_ns",
        "execution_ns",
        "security_scope",
        "tenant",
        "workspace",
        "project",
        "graph_space",
    ] {
        if let Some(value) = principal.get(key) {
            admitted_scope.insert(key.to_owned(), value.clone());
        }
    }
    if principal_strings(principal, "ns").is_empty() {
        admitted_scope.insert("ns".to_owned(), json!("docs"));
    }
    Some(RuntimeOwnerBindingV1 {
        version: 1,
        user_id,
        admitted_capabilities,
        admitted_scope: Value::Object(admitted_scope),
    })
}

fn revalidated_runtime_owner_principal(
    binding: &RuntimeOwnerBindingV1,
    user: &AuthUser,
    capabilities: &SharedCapabilityState,
    required_capability: &str,
) -> Result<Value, String> {
    if binding.version != 1 || binding.user_id != user.user_id || !user.is_active {
        return Err("run owner binding is missing, mismatched, or inactive".to_owned());
    }
    let mut principal = runtime_principal_from_auth_user(user);
    let admitted_namespaces = principal_strings(&binding.admitted_scope, "ns")
        .into_iter()
        .collect::<BTreeSet<_>>();
    let current_namespaces = principal_strings(&principal, "ns")
        .into_iter()
        .collect::<BTreeSet<_>>();
    let namespaces = admitted_namespaces
        .intersection(&current_namespaces)
        .cloned()
        .collect::<Vec<_>>();
    if namespaces.is_empty() {
        return Err("run owner no longer has an admitted namespace".to_owned());
    }
    principal["ns"] = json!(namespaces.clone());
    for key in [
        "storage_ns",
        "execution_ns",
        "security_scope",
        "tenant",
        "workspace",
        "project",
        "graph_space",
    ] {
        if let Some(value) = binding.admitted_scope.get(key) {
            principal[key] = value.clone();
        }
    }
    let current_scope = security_scope_value(&principal);
    if !namespaces
        .iter()
        .any(|namespace| current_scope["storage_namespace"].as_str() == Some(namespace))
    {
        return Err("run owner storage namespace is no longer authorized".to_owned());
    }
    let state = capabilities
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let current_capabilities = effective_capabilities(&principal, &state);
    let effective = current_capabilities
        .intersection(&binding.admitted_capabilities)
        .cloned()
        .collect::<BTreeSet<_>>();
    if !effective.contains(required_capability) {
        return Err(format!(
            "run owner no longer has {required_capability} capability"
        ));
    }
    principal["capabilities"] = json!(effective);
    Ok(principal)
}

#[derive(Debug)]
enum InlineOwnerPrincipalError {
    Denied(String),
    Unavailable(String),
}

impl std::fmt::Display for InlineOwnerPrincipalError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Denied(message) | Self::Unavailable(message) => formatter.write_str(message),
        }
    }
}

impl From<InlineOwnerPrincipalError> for String {
    fn from(error: InlineOwnerPrincipalError) -> Self {
        error.to_string()
    }
}

fn rust_pgvector_embedding_scope(
    config: &tokio_postgres::Config,
    schema: &str,
) -> Result<String, String> {
    if config.get_hosts().len() > 1 {
        return Err(
            "embedding profile binding requires a single PostgreSQL host in the DSN".to_owned(),
        );
    }
    let host = match config.get_hosts().first() {
        Some(tokio_postgres::config::Host::Tcp(host)) => host.to_ascii_lowercase(),
        #[cfg(unix)]
        Some(tokio_postgres::config::Host::Unix(_)) => {
            return Err(
                "embedding profile binding does not support Unix-socket PostgreSQL DSNs".to_owned(),
            );
        }
        None => "localhost".to_owned(),
    };
    let port = config.get_ports().first().copied().unwrap_or(5432);
    let database = config.get_dbname().unwrap_or_default().trim();
    let tables = "gke_nodes:gke_edges:gke_documents:gke_domains";
    let canonical =
        format!("host={host}|port={port}|database={database}|schema={schema}|tables={tables}");
    let digest = Sha256::digest(canonical.as_bytes());
    let fingerprint = digest
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    Ok(format!("pgvector:{}", &fingerprint[..32]))
}

fn validate_embedding_profile_projection(
    projection: &NamedProjection,
    configured: &kogwistar_engine::embedding::EmbeddingProfile,
    storage_scope: &str,
) -> Result<(), String> {
    let payload_version = projection
        .payload
        .get("profile_schema_version")
        .and_then(Value::as_i64);
    if projection.projection_schema_version != EMBEDDING_PROFILE_SCHEMA_VERSION
        || payload_version != Some(EMBEDDING_PROFILE_SCHEMA_VERSION)
    {
        return Err(format!(
            "unsupported or corrupt embedding profile registry record for {storage_scope}"
        ));
    }
    let registered: kogwistar_engine::embedding::EmbeddingProfile = serde_json::from_value(
        projection
            .payload
            .get("embedding_profile")
            .cloned()
            .ok_or_else(|| "embedding profile registry payload is missing".to_owned())?,
    )
    .map_err(|_| "embedding profile registry payload is corrupt".to_owned())?;
    registered
        .validate()
        .map_err(|_| format!("embedding profile registry record is invalid for {storage_scope}"))?;
    if registered.fingerprint() != configured.fingerprint() {
        return Err(format!(
            "embedding profile mismatch for {storage_scope}: configured {}/{}/{}D, registered {}/{}/{}D",
            configured.provider,
            configured.model,
            configured.dimension,
            registered.provider,
            registered.model,
            registered.dimension
        ));
    }
    Ok(())
}

async fn ensure_embedding_profile_binding_for_store(
    store: &PostgresStore,
    configured: &kogwistar_engine::embedding::EmbeddingProfile,
    storage_scope: &str,
) -> Result<(), String> {
    configured.validate().map_err(|error| error.to_string())?;
    if let Some(current) = store
        .get_named_projection(EMBEDDING_PROFILE_NAMESPACE, storage_scope)
        .await
        .map_err(|error| error.to_string())?
    {
        return validate_embedding_profile_projection(&current, configured, storage_scope);
    }
    if store
        .has_graph_projection_rows()
        .await
        .map_err(|error| error.to_string())?
    {
        return Err(format!(
            "embedding profile is unbound for non-empty PostgreSQL vector storage {}; bind or migrate it with the Python embedding profile registry before enabling Rust embedding",
            storage_scope
        ));
    }

    let payload = json!({
        "profile_schema_version": EMBEDDING_PROFILE_SCHEMA_VERSION,
        "embedding_profile": configured.as_value(),
        "embedding_fingerprint": configured.fingerprint(),
        "legacy_adopted": false,
    })
    .as_object()
    .cloned()
    .ok_or_else(|| "could not serialize embedding profile binding".to_owned())?;
    let inserted = store
        .compare_and_swap_named_projection(
            EMBEDDING_PROFILE_NAMESPACE,
            storage_scope,
            None,
            None,
            NamedProjectionWrite {
                payload,
                last_authoritative_seq: 0,
                last_materialized_seq: 0,
                projection_schema_version: EMBEDDING_PROFILE_SCHEMA_VERSION,
                materialization_status: "bound".to_owned(),
            },
        )
        .await
        .map_err(|error| error.to_string())?;
    if inserted {
        return Ok(());
    }
    let winner = store
        .get_named_projection(EMBEDDING_PROFILE_NAMESPACE, storage_scope)
        .await
        .map_err(|error| error.to_string())?
        .ok_or_else(|| "embedding profile binding disappeared during startup".to_owned())?;
    validate_embedding_profile_projection(&winner, configured, storage_scope)
}

fn conversation_record_value(record: &GraphRecord, field: &str) -> Option<Value> {
    record.metadata.get(field).cloned().or_else(|| {
        record
            .document
            .as_deref()
            .and_then(|document| serde_json::from_str::<Value>(document).ok())
            .and_then(|document| document.get(field).cloned())
    })
}

fn conversation_record_string(record: &GraphRecord, field: &str) -> Option<String> {
    conversation_record_value(record, field).and_then(|value| value.as_str().map(str::to_owned))
}

fn conversation_record_i64(record: &GraphRecord, field: &str) -> Option<i64> {
    conversation_record_value(record, field).and_then(|value| value.as_i64())
}

fn conversation_span(conversation_id: &str, turn_id: &str, text: &str) -> Value {
    json!({
        "collection_page_url": format!("conversation/{conversation_id}"),
        "document_page_url": format!("conversation/{conversation_id}#{turn_id}"),
        "doc_id": format!("conv:{conversation_id}"),
        "insertion_method": "conversation_turn",
        "page_number": 1,
        "start_char": 0,
        "end_char": text.chars().count(),
        "excerpt": text,
        "context_before": "",
        "context_after": "",
        "source_cluster_id": null,
        "verification": {
            "method": "system",
            "is_verified": true,
            "score": 1.0,
            "notes": "verbatim user input",
        },
    })
}

fn conversation_index_job(
    namespace: &str,
    entity_kind: &str,
    entity_id: &str,
    index_kind: &str,
) -> NewIndexJob {
    NewIndexJob {
        job_id: uuid::Uuid::new_v4().to_string(),
        namespace: namespace.to_owned(),
        entity_kind: entity_kind.to_owned(),
        entity_id: entity_id.to_owned(),
        index_kind: index_kind.to_owned(),
        op: "UPSERT".to_owned(),
        payload_json: None,
        max_retries: 3,
    }
}

fn conversation_index_job_kinds(entity_kind: &str) -> Option<&'static [&'static str]> {
    match entity_kind {
        "node" => Some(&["node_docs", "node_refs"]),
        "edge" => Some(&["edge_refs", "edge_endpoints"]),
        _ => None,
    }
}

fn conversation_next_turn_distances(
    previous: &GraphRecord,
    text: &str,
    turn_index: i64,
) -> (i64, i64, i64) {
    let previous_char_distance =
        conversation_record_i64(previous, "char_distance_from_last_summary").unwrap_or(0)
            + previous
                .document
                .as_deref()
                .and_then(|document| serde_json::from_str::<Value>(document).ok())
                .and_then(|document| {
                    document
                        .get("summary")
                        .and_then(Value::as_str)
                        .map(|summary| summary.chars().count() as i64)
                })
                .unwrap_or(0);
    let char_distance = previous_char_distance + text.chars().count() as i64;
    let turn_distance =
        conversation_record_i64(previous, "turn_distance_from_last_summary").unwrap_or(-1) + 2;
    (char_distance, turn_distance, turn_index)
}

impl PostgresRunApplicationService {
    pub fn from_dsn(dsn: &str, schema: &str) -> Result<Self, String> {
        let legacy_wisdom_schema = format!("{schema}_wisdom");
        Self::from_dsn_with_wisdom_schema(dsn, schema, &legacy_wisdom_schema)
    }

    pub fn from_dsn_with_wisdom_schema(
        dsn: &str,
        schema: &str,
        wisdom_schema: &str,
    ) -> Result<Self, String> {
        let conversation_schema = format!("{schema}_conversation");
        let workflow_schema = format!("{schema}_workflow");
        Self::from_dsn_with_graph_schemas(
            dsn,
            schema,
            &conversation_schema,
            &workflow_schema,
            wisdom_schema,
        )
    }

    pub fn from_dsn_with_graph_schemas(
        dsn: &str,
        knowledge_schema: &str,
        conversation_schema: &str,
        workflow_schema: &str,
        wisdom_schema: &str,
    ) -> Result<Self, String> {
        let dsn = dsn
            .strip_prefix("postgresql+psycopg://")
            .map(|rest| format!("postgresql://{rest}"))
            .or_else(|| {
                dsn.strip_prefix("postgresql+psycopg2://")
                    .map(|rest| format!("postgresql://{rest}"))
            })
            .unwrap_or_else(|| dsn.to_owned());
        let config = dsn
            .parse::<tokio_postgres::Config>()
            .map_err(|_| "invalid PostgreSQL DSN".to_owned())?;
        let embedding_storage_scope = rust_pgvector_embedding_scope(&config, knowledge_schema);
        let conversation_embedding_storage_scope =
            rust_pgvector_embedding_scope(&config, conversation_schema);
        let workflow_embedding_storage_scope =
            rust_pgvector_embedding_scope(&config, workflow_schema);
        let native_operations = crate::native_answer_persist::registry().map_err(|error| {
            format!("could not register native conversation operations: {error}")
        })?;
        let wisdom_embedding_storage_scope = rust_pgvector_embedding_scope(&config, wisdom_schema);
        let store = PostgresStore::from_config(config.clone(), knowledge_schema)
            .map_err(|error| error.to_string())?;
        let conversation_store = PostgresStore::from_config(config.clone(), conversation_schema)
            .map_err(|error| error.to_string())?;
        let workflow_store = PostgresStore::from_config(config.clone(), workflow_schema)
            .map_err(|error| error.to_string())?;
        let wisdom_store =
            PostgresStore::from_config(config, wisdom_schema).map_err(|error| error.to_string())?;
        Ok(Self {
            store,
            conversation_store,
            workflow_store,
            wisdom_store,
            graph_role: "knowledge".to_owned(),
            max_queue: runtime_max_queue(),
            capabilities: Arc::new(Mutex::new(CapabilityState::default())),
            native_operations,
            owner_auth_store: None,
            embedding_provider: None,
            search_index_fts: None,
            embedding_storage_scope,
            conversation_embedding_storage_scope,
            workflow_embedding_storage_scope,
            wisdom_embedding_storage_scope,
            chat_completion_provider: None,
            transient_run_deltas: crate::transient_deltas::TransientRunDeltas::default(),
        })
    }

    pub async fn ensure_schema(&self) -> Result<(), String> {
        self.store
            .ensure_schema()
            .await
            .map_err(|error| error.to_string())?;
        self.wisdom_store
            .ensure_schema()
            .await
            .map_err(|error| error.to_string())?;
        if let Some(provider) = &self.embedding_provider {
            self.ensure_embedding_profile_binding(provider.profile())
                .await?;
        }
        Ok(())
    }

    /// Select the host-owned graph role used by the generic graph-upsert route
    /// and its ACL truth-graph checks. The request payload cannot override it.
    pub fn with_graph_role(mut self, role: impl AsRef<str>) -> Result<Self, String> {
        self.graph_role = normalize_graph_role(role.as_ref())?;
        Ok(self)
    }

    async fn ensure_embedding_profile_binding(
        &self,
        configured: &kogwistar_engine::embedding::EmbeddingProfile,
    ) -> Result<(), String> {
        let storage_scope = self
            .embedding_storage_scope
            .as_ref()
            .map_err(Clone::clone)?;
        ensure_embedding_profile_binding_for_store(&self.store, configured, storage_scope).await?;
        let wisdom_scope = self
            .wisdom_embedding_storage_scope
            .as_ref()
            .map_err(Clone::clone)?;
        ensure_embedding_profile_binding_for_store(&self.wisdom_store, configured, wisdom_scope)
            .await
    }

    // PYTHON_PARITY: kogwistar/server/bootstrap.py::ServerStorageSettings.schema_for
    pub async fn ensure_graph_type_schemas(&self) -> Result<(), String> {
        self.ensure_schema().await?;
        self.conversation_store
            .ensure_schema()
            .await
            .map_err(|error| error.to_string())?;
        self.workflow_store
            .ensure_schema()
            .await
            .map_err(|error| error.to_string())?;
        if let Some(provider) = &self.embedding_provider {
            let conversation_scope = self
                .conversation_embedding_storage_scope
                .as_ref()
                .map_err(Clone::clone)?;
            ensure_embedding_profile_binding_for_store(
                &self.conversation_store,
                provider.profile(),
                conversation_scope,
            )
            .await?;
            let workflow_scope = self
                .workflow_embedding_storage_scope
                .as_ref()
                .map_err(Clone::clone)?;
            ensure_embedding_profile_binding_for_store(
                &self.workflow_store,
                provider.profile(),
                workflow_scope,
            )
            .await?;
        }
        Ok(())
    }

    /// Compose trusted application-owned Rust operations into the opt-in native
    /// workflow lane. The default registry remains unchanged for existing users.
    pub fn with_native_operations(
        mut self,
        native_operations: kogwistar_runtime::NativeOperationRegistry,
    ) -> Self {
        self.native_operations = native_operations;
        self
    }

    pub async fn with_owner_auth_config(mut self, config: &AuthConfig) -> Result<Self, String> {
        self.owner_auth_store = Some(Arc::new(AuthStore::from_config(config).await?));
        Ok(self)
    }

    async fn resolve_inline_native_owner_principal(
        &self,
        run_id: &str,
        required_capability: &str,
    ) -> Result<Value, InlineOwnerPrincipalError> {
        let run = self
            .store
            .server_run(run_id)
            .await
            .map_err(|error| InlineOwnerPrincipalError::Unavailable(error.to_string()))?
            .ok_or_else(|| {
                InlineOwnerPrincipalError::Denied("native worker run is missing".to_owned())
            })?;
        let owner_id = run
            .user_id
            .as_deref()
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| {
                InlineOwnerPrincipalError::Denied(
                    "native worker run has no authenticated owner".to_owned(),
                )
            })?;
        let created = self
            .store
            .server_run_events(run_id, 0, 1)
            .await
            .map_err(|error| InlineOwnerPrincipalError::Unavailable(error.to_string()))?
            .into_iter()
            .find(|event| event.event_type == "run.created")
            .ok_or_else(|| {
                InlineOwnerPrincipalError::Denied(
                    "native worker run has no admission event".to_owned(),
                )
            })?;
        let created_payload: Value = serde_json::from_str(&created.payload_json).map_err(|_| {
            InlineOwnerPrincipalError::Denied("native worker admission event is invalid".to_owned())
        })?;
        let binding: RuntimeOwnerBindingV1 = serde_json::from_value(
            created_payload
                .get("runtime_owner_binding")
                .cloned()
                .ok_or_else(|| {
                    InlineOwnerPrincipalError::Denied(
                        "native worker run has no trusted owner binding".to_owned(),
                    )
                })?,
        )
        .map_err(|_| {
            InlineOwnerPrincipalError::Denied("native worker owner binding is invalid".to_owned())
        })?;
        if binding.user_id != owner_id {
            return Err(InlineOwnerPrincipalError::Denied(
                "native worker owner binding does not match the durable run owner".to_owned(),
            ));
        }
        let auth_store = self.owner_auth_store.as_ref().ok_or_else(|| {
            InlineOwnerPrincipalError::Unavailable(
                "native worker owner revalidation is not configured".to_owned(),
            )
        })?;
        let user = auth_store
            .auth_user(owner_id)
            .await
            .map_err(InlineOwnerPrincipalError::Unavailable)?
            .filter(|user| user.is_active)
            .ok_or_else(|| {
                InlineOwnerPrincipalError::Denied(
                    "native worker run owner is missing or inactive".to_owned(),
                )
            })?;
        revalidated_runtime_owner_principal(
            &binding,
            &user,
            &self.capabilities,
            required_capability,
        )
        .map_err(InlineOwnerPrincipalError::Denied)
    }

    async fn native_read_principal(
        &self,
        request: &kogwistar_runtime::RuntimeNativeStepRequest,
        caller: Option<&Value>,
    ) -> Result<Value, NativeEvidenceContextError> {
        let run = self
            .store
            .server_run(&request.run_id)
            .await
            .map_err(|_| NativeEvidenceContextError::OwnerAuthorization)?
            .ok_or(NativeEvidenceContextError::OwnerAuthorization)?;
        let durable_owner = run
            .user_id
            .as_deref()
            .filter(|value| !value.trim().is_empty())
            .ok_or(NativeEvidenceContextError::OwnerAuthorization)?;
        if let Some(caller) = caller {
            let Some(caller_id) = authenticated_run_owner_id(caller) else {
                return Err(NativeEvidenceContextError::CallerNotRunOwner);
            };
            if caller_id != durable_owner {
                return Err(NativeEvidenceContextError::CallerNotRunOwner);
            }
        }
        let mut principal = self
            .resolve_inline_native_owner_principal(&request.run_id, "read_graph")
            .await
            .map_err(|_| NativeEvidenceContextError::OwnerAuthorization)?;
        // ACL records are keyed by the durable run-owner ID, while the refreshed
        // auth principal's `sub` may be an email address.
        principal["actor_id"] = json!(durable_owner);
        Ok(principal)
    }

    async fn native_answer_view_context(
        &self,
        request: &kogwistar_runtime::RuntimeNativeStepRequest,
        caller: Option<&Value>,
    ) -> Result<Value, NativeEvidenceContextError> {
        let principal = self.native_read_principal(request, caller).await?;
        let turn_id = request
            .turn_node_id
            .as_deref()
            .or_else(|| request.state.get("turn_node_id").and_then(Value::as_str))
            .filter(|id| !id.trim().is_empty())
            .ok_or(NativeEvidenceContextError::GraphReadRejected)?;
        if request
            .state
            .get("turn_node_id")
            .and_then(Value::as_str)
            .is_some_and(|state_turn_id| state_turn_id != turn_id)
        {
            return Err(NativeEvidenceContextError::GraphReadRejected);
        }
        let scope_claims = security_scope_value(&principal);
        let scope = GraphScope {
            namespace: scope_claims["storage_namespace"]
                .as_str()
                .unwrap_or("default")
                .to_owned(),
            workspace_id: principal
                .get("workspace_id")
                .or_else(|| principal.get("workspace"))
                .and_then(Value::as_str)
                .map(str::to_owned),
            graph_space: principal
                .get("graph_space")
                .and_then(Value::as_str)
                .map(str::to_owned),
        };
        let acl_principal =
            graph_acl_principal_from_claims(&principal, scope_claims["security_scope"].as_str());
        let conversation_id = request.conversation_id.clone();
        let turn_id = turn_id.to_owned();
        self.conversation_store
            .transaction(move |uow| {
                Box::pin(async move {
                    let mut metadata = BTreeMap::new();
                    metadata.insert("conversation_id".to_owned(), json!(conversation_id));
                    let read = GraphProjectionRead {
                        scope: scope.clone(),
                        table: "gke_nodes".to_owned(),
                        ids: None,
                        metadata: MetadataFilter { equals: metadata },
                        limit: 20_000,
                    };
                    let nodes = uow.graph_projection_records(read).await?;
                    let mut edge_metadata = BTreeMap::new();
                    edge_metadata.insert(
                        "doc_id".to_owned(),
                        json!(format!("conv:{conversation_id}")),
                    );
                    let edges = uow
                        .graph_projection_records(GraphProjectionRead {
                            scope: scope.clone(),
                            table: "gke_edges".to_owned(),
                            ids: None,
                            metadata: MetadataFilter {
                                equals: edge_metadata,
                            },
                            limit: 20_000,
                        })
                        .await?;
                    let ids = nodes
                        .iter()
                        .chain(edges.iter())
                        .map(|record| record.id.clone())
                        .collect::<Vec<_>>();
                    let acl_records = uow
                        .graph_acl_records_for_entities(&scope, "conversation", &ids)
                        .await?;
                    let mut acl_index = GraphAclIndex::default();
                    acl_index.extend_from_records(&acl_records, "conversation");
                    let (visible, visible_edges) = graph_records_visible_by_acl_with_index(
                        &nodes,
                        &edges,
                        &acl_index,
                        &acl_principal,
                    );
                    let context_nodes = visible
                        .iter()
                        .filter(|record| {
                            record
                                .document
                                .as_deref()
                                .and_then(|raw| serde_json::from_str::<Value>(raw).ok())
                                .and_then(|document| {
                                    document
                                        .get("conversation_id")
                                        .or_else(|| {
                                            document.get("metadata").and_then(|metadata| {
                                                metadata.get("conversation_id")
                                            })
                                        })
                                        .and_then(Value::as_str)
                                        .map(|value| value == conversation_id)
                                })
                                .unwrap_or(false)
                        })
                        .map(|record| {
                            json!({
                                "id": record.id,
                                "document": record.document,
                                "metadata": record.metadata
                            })
                        })
                        .collect::<Vec<_>>();
                    let context_edges = visible_edges
                        .iter()
                        .map(|record| {
                            json!({
                                "id": record.id,
                                "document": record.document,
                                "metadata": record.metadata
                            })
                        })
                        .collect::<Vec<_>>();
                    let Some(record) = visible.iter().find(|record| record.id == turn_id) else {
                        return Ok(Err(NativeEvidenceRecordError::Rejected));
                    };
                    let Some(document) = record
                        .document
                        .as_deref()
                        .and_then(|raw| serde_json::from_str::<Value>(raw).ok())
                    else {
                        return Ok(Err(NativeEvidenceRecordError::Rejected));
                    };
                    let document_metadata = document.get("metadata").and_then(Value::as_object);
                    let field = |key: &str| {
                        document
                            .get(key)
                            .or_else(|| document_metadata.and_then(|metadata| metadata.get(key)))
                            .or_else(|| record.metadata.get(key))
                    };
                    if field("entity_type").and_then(Value::as_str) != Some("conversation_turn")
                        || field("conversation_id").and_then(Value::as_str)
                            != Some(conversation_id.as_str())
                        || field("role").and_then(Value::as_str) != Some("user")
                        || matches!(
                            field("lifecycle_status").and_then(Value::as_str),
                            Some("tombstoned" | "deleted")
                        )
                    {
                        return Ok(Err(NativeEvidenceRecordError::Rejected));
                    }
                    Ok(Ok(json!({
                        "turn": {
                            "id": record.id,
                            "document": record.document,
                            "metadata": record.metadata
                        },
                        "context_nodes": context_nodes,
                        "context_edges": context_edges
                    })))
                })
            })
            .await
            .map_err(|_| NativeEvidenceContextError::GraphReadUnavailable)?
            .map_err(|error| match error {
                NativeEvidenceRecordError::Rejected => {
                    NativeEvidenceContextError::GraphReadRejected
                }
                NativeEvidenceRecordError::Unavailable => {
                    NativeEvidenceContextError::GraphReadUnavailable
                }
            })
    }

    async fn native_candidate_context(
        &self,
        request: &kogwistar_runtime::RuntimeNativeStepRequest,
        caller: Option<&Value>,
    ) -> Result<Value, NativeEvidenceContextError> {
        let principal = self.native_read_principal(request, caller).await?;
        let question = request
            .state
            .get("question")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty() && value.len() <= 100_000)
            .ok_or(NativeEvidenceContextError::GraphReadRejected)?;
        let max_candidates = request
            .state
            .get("max_candidates")
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .filter(|value| *value <= 200)
            .ok_or(NativeEvidenceContextError::GraphReadRejected)?;
        let max_retrieval_level = request
            .state
            .get("max_retrieval_level")
            .and_then(Value::as_i64)
            .ok_or(NativeEvidenceContextError::GraphReadRejected)?;
        if max_candidates == 0 {
            return Ok(Value::Array(Vec::new()));
        }
        let provider = self
            .embedding_provider
            .as_ref()
            .ok_or(NativeEvidenceContextError::GraphReadRejected)?;
        let profile = provider.profile();
        profile
            .validate()
            .map_err(|_| NativeEvidenceContextError::GraphReadRejected)?;
        if profile.dimension > 8192 {
            return Err(NativeEvidenceContextError::GraphReadRejected);
        }
        let query_text = vec![question.to_owned()];
        let mut vectors = provider
            .embed(&query_text)
            .await
            .map_err(|_| NativeEvidenceContextError::GraphReadUnavailable)?;
        if vectors.len() != 1 || vectors[0].len() != profile.dimension {
            return Err(NativeEvidenceContextError::GraphReadRejected);
        }
        let vector = vectors
            .pop()
            .expect("the embedding result count was validated")
            .into_iter()
            .map(|value| {
                let value = value as f32;
                value
                    .is_finite()
                    .then_some(value)
                    .ok_or(NativeEvidenceContextError::GraphReadRejected)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let metric = match profile.similarity_metric.to_ascii_lowercase().as_str() {
            "cosine" => DistanceMetric::Cosine,
            "l2" => DistanceMetric::L2,
            "ip" => DistanceMetric::InnerProduct,
            _ => return Err(NativeEvidenceContextError::GraphReadRejected),
        };
        let scope_value = security_scope_value(&principal);
        let scope = GraphScope {
            namespace: scope_value["storage_namespace"]
                .as_str()
                .unwrap_or("default")
                .to_owned(),
            workspace_id: principal
                .get("workspace_id")
                .or_else(|| principal.get("workspace"))
                .and_then(Value::as_str)
                .map(str::to_owned),
            graph_space: principal
                .get("graph_space")
                .and_then(Value::as_str)
                .map(str::to_owned),
        };
        let query = GraphProjectionVectorQuery {
            scope: scope.clone(),
            table: "gke_nodes".to_owned(),
            query: VectorQuery {
                embedding: vector,
                limit: max_candidates,
                metadata: MetadataFilter::default(),
                metric,
            },
            embedding_dim: profile.dimension,
        };
        let candidates = self
            .store
            .graph_projection_vector_query_level_lte_excluding_metadata(
                query,
                BTreeMap::from([("entity_type".to_owned(), json!("acl_record"))]),
                max_retrieval_level,
            )
            .await
            .map_err(|_| NativeEvidenceContextError::GraphReadUnavailable)?;
        let candidate_rows = vec![
            candidates
                .iter()
                .map(|matched| matched.record.clone())
                .collect::<Vec<_>>(),
        ];
        let acl_principal =
            graph_acl_principal_from_claims(&principal, scope_value["security_scope"].as_str());
        let visible_ids = self
            .authorized_vector_candidate_ids(
                &self.store,
                &scope,
                "knowledge",
                "node",
                &candidate_rows,
                &acl_principal,
            )
            .await
            .map_err(|_| NativeEvidenceContextError::GraphReadUnavailable)?;
        let mut projected = candidates
            .into_iter()
            .filter(|matched| {
                visible_ids.contains(&matched.record.id)
                    && !matched
                        .record
                        .metadata
                        .get("lifecycle_status")
                        .and_then(Value::as_str)
                        .is_some_and(|status| matches!(status, "tombstoned" | "deleted"))
            })
            .map(|matched| {
                let record = matched.record;
                json!({
                    "id": record.id,
                    "label": record.metadata.get("label").cloned().unwrap_or(Value::Null),
                    "summary": record.metadata.get("summary").cloned().unwrap_or(Value::Null),
                    "doc": record.document.unwrap_or_default(),
                })
            })
            .collect::<Vec<_>>();
        projected.sort_by(|left, right| {
            left.get("id")
                .and_then(Value::as_str)
                .cmp(&right.get("id").and_then(Value::as_str))
        });
        Ok(Value::Array(projected))
    }

    async fn native_evidence_context(
        &self,
        request: &kogwistar_runtime::RuntimeNativeStepRequest,
        caller: Option<&Value>,
    ) -> Result<Value, NativeEvidenceContextError> {
        let principal = self.native_read_principal(request, caller).await?;
        let read_ids = |field: &str| -> Result<Vec<String>, NativeEvidenceContextError> {
            let Some(value) = request.state.get(field) else {
                return Ok(Vec::new());
            };
            let Some(values) = value.as_array() else {
                return Err(NativeEvidenceContextError::GraphReadRejected);
            };
            if values.len() > 100 {
                return Err(NativeEvidenceContextError::GraphReadRejected);
            }
            let ids = values
                .iter()
                .map(|value| {
                    value
                        .as_str()
                        .filter(|id| !id.trim().is_empty())
                        .map(str::to_owned)
                        .ok_or(NativeEvidenceContextError::GraphReadRejected)
                })
                .collect::<Result<Vec<_>, _>>()?;
            if ids.iter().collect::<BTreeSet<_>>().len() != ids.len() {
                return Err(NativeEvidenceContextError::GraphReadRejected);
            }
            Ok(ids)
        };
        let node_ids = read_ids("used_node_ids")?;
        let edge_ids = read_ids("used_edge_ids")?;
        self.authorized_evidence_records(&principal, &node_ids, &edge_ids, "knowledge")
            .await
            .map_err(|error| match error {
                NativeEvidenceRecordError::Rejected => {
                    NativeEvidenceContextError::GraphReadRejected
                }
                NativeEvidenceRecordError::Unavailable => {
                    NativeEvidenceContextError::GraphReadUnavailable
                }
            })
    }

    async fn native_pointer_context(
        &self,
        request: &kogwistar_runtime::RuntimeNativeStepRequest,
        caller: Option<&Value>,
    ) -> Result<Value, NativeEvidenceContextError> {
        let principal = self.native_read_principal(request, caller).await?;
        let read_ids = |field: &str| -> Result<Vec<String>, NativeEvidenceContextError> {
            let Some(value) = request.state.get(field) else {
                return Ok(Vec::new());
            };
            let Some(values) = value.as_array() else {
                return Err(NativeEvidenceContextError::GraphReadRejected);
            };
            if values.len() > 100 {
                return Err(NativeEvidenceContextError::GraphReadRejected);
            }
            values
                .iter()
                .map(|value| {
                    value
                        .as_str()
                        .filter(|id| !id.trim().is_empty())
                        .map(str::to_owned)
                        .ok_or(NativeEvidenceContextError::GraphReadRejected)
                })
                .collect()
        };
        let used_ids = read_ids("used_node_ids")?;
        let conversation_id = request
            .state
            .get("conversation_id")
            .and_then(Value::as_str)
            .filter(|id| *id == request.conversation_id)
            .ok_or(NativeEvidenceContextError::GraphReadRejected)?;
        let run_node_id = request
            .state
            .get("run_node_id")
            .and_then(Value::as_str)
            .filter(|id| !id.trim().is_empty())
            .ok_or(NativeEvidenceContextError::GraphReadRejected)?;
        let pointer_ids = used_ids
            .iter()
            .map(|id| {
                crate::native_answer_persist::stable_conversation_id(
                    "ptr",
                    &[
                        ("scope", &format!("conv:{conversation_id}")),
                        ("pointer_kind", "kg_node"),
                        ("target_kind", "node"),
                        ("target_id", id),
                    ],
                )
            })
            .collect::<BTreeSet<_>>();
        let run_node_id = run_node_id.to_owned();
        let conversation_node_ids = std::iter::once(run_node_id.clone())
            .chain(pointer_ids.iter().cloned())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let edge_ids = pointer_ids
            .iter()
            .map(|pointer_id| {
                crate::native_answer_persist::stable_conversation_id(
                    "e",
                    &[
                        ("scope", &format!("conv:{conversation_id}")),
                        ("rel", "used_evidence"),
                        ("src", &run_node_id),
                        ("dst", pointer_id),
                    ],
                )
            })
            .collect::<Vec<_>>();
        let scope_claims = security_scope_value(&principal);
        let scope = GraphScope {
            namespace: scope_claims["storage_namespace"]
                .as_str()
                .unwrap_or("default")
                .to_owned(),
            workspace_id: principal
                .get("workspace_id")
                .or_else(|| principal.get("workspace"))
                .and_then(Value::as_str)
                .map(str::to_owned),
            graph_space: principal
                .get("graph_space")
                .and_then(Value::as_str)
                .map(str::to_owned),
        };
        let acl_principal =
            graph_acl_principal_from_claims(&principal, scope_claims["security_scope"].as_str());
        let conversation_id = conversation_id.to_owned();
        let conversation_schema = self.conversation_store.schema().to_owned();
        let result = self.store
            .transaction_with_graph_schemas(&[conversation_schema.as_str()], |uow| {
                let conversation_schema = conversation_schema.clone();
                Box::pin(async move {
                    let read = |table: &str, ids: &[String]| GraphProjectionRead {
                        scope: scope.clone(),
                        table: table.to_owned(),
                        ids: Some(ids.to_vec()),
                        metadata: MetadataFilter::default(),
                        limit: ids.len().max(1),
                    };
                    let knowledge_nodes = if used_ids.is_empty() {
                        Vec::new()
                    } else {
                        uow.graph_projection_records(read("gke_nodes", &used_ids)).await?
                    };
                    let conversation_nodes = uow
                        .graph_projection_records_in_schema(
                            &conversation_schema,
                            read("gke_nodes", &conversation_node_ids),
                        )
                        .await?;
                    let conversation_edges = if edge_ids.is_empty() {
                        Vec::new()
                    } else {
                        uow.graph_projection_records_in_schema(
                            &conversation_schema,
                            read("gke_edges", &edge_ids),
                        )
                        .await?
                    };
                    let knowledge_ids = knowledge_nodes
                        .iter()
                        .map(|record| record.id.clone())
                        .collect::<Vec<_>>();
                    let conversation_entity_ids = conversation_nodes
                        .iter()
                        .chain(conversation_edges.iter())
                        .map(|record| record.id.clone())
                        .collect::<Vec<_>>();
                    let knowledge_acl = uow
                        .graph_acl_records_for_entities(&scope, "knowledge", &knowledge_ids)
                        .await?;
                    let conversation_acl = uow
                        .graph_acl_records_for_entities_in_schema(
                            &conversation_schema,
                            &scope,
                            "conversation",
                            &conversation_entity_ids,
                        )
                        .await?;
                    let mut knowledge_index = GraphAclIndex::default();
                    knowledge_index.extend_from_records(&knowledge_acl, "knowledge");
                    let mut conversation_index = GraphAclIndex::default();
                    conversation_index.extend_from_records(&conversation_acl, "conversation");
                    let (visible_knowledge, _) = graph_records_visible_by_acl_with_index(
                        &knowledge_nodes,
                        &[],
                        &knowledge_index,
                        &acl_principal,
                    );
                    let (visible_conversation_nodes, visible_conversation_edges) =
                        graph_records_visible_by_acl_with_index(
                            &conversation_nodes,
                            &conversation_edges,
                            &conversation_index,
                            &acl_principal,
                        );
                    let selected = used_ids.iter().cloned().collect::<BTreeSet<_>>();
                    let visible_knowledge_ids = visible_knowledge
                        .iter()
                        .filter(|record| {
                            selected.contains(&record.id)
                                && !matches!(
                                    record.metadata.get("lifecycle_status").and_then(Value::as_str),
                                    Some("tombstoned" | "deleted")
                                )
                        })
                        .map(|record| record.id.clone())
                        .collect::<BTreeSet<_>>();
                    if visible_knowledge_ids != selected {
                        return Ok(Err(NativeEvidenceRecordError::Rejected));
                    }
                    let visible_node_ids = visible_conversation_nodes
                        .iter()
                        .map(|record| record.id.clone())
                        .collect::<BTreeSet<_>>();
                    let visible_edge_ids = visible_conversation_edges
                        .iter()
                        .map(|record| record.id.clone())
                        .collect::<BTreeSet<_>>();
                    let required_nodes = BTreeSet::from([run_node_id.clone()]);
                    if !required_nodes.is_subset(&visible_node_ids)
                        || conversation_nodes.iter().any(|record| {
                            pointer_ids.contains(&record.id) && !visible_node_ids.contains(&record.id)
                        })
                        || conversation_edges.iter().any(|record| {
                            edge_ids.contains(&record.id) && !visible_edge_ids.contains(&record.id)
                        })
                    {
                        return Ok(Err(NativeEvidenceRecordError::Rejected));
                    }
                    let run_record = visible_conversation_nodes
                        .iter()
                        .find(|record| record.id == run_node_id);
                    if !run_record.is_some_and(|record| {
                        record.metadata.get("entity_type").and_then(Value::as_str)
                            == Some("agent_run")
                            && record.metadata.get("conversation_id").and_then(Value::as_str)
                                == Some(conversation_id.as_str())
                            && !matches!(
                                record.metadata.get("lifecycle_status").and_then(Value::as_str),
                                Some("tombstoned" | "deleted")
                            )
                    }) {
                        return Ok(Err(NativeEvidenceRecordError::Rejected));
                    }
                    let selected_records = |records: &[GraphRecord], ids: &BTreeSet<String>| {
                        records
                            .iter()
                            .filter(|record| ids.contains(&record.id))
                            .map(|record| json!({"id":record.id,"document":record.document,"metadata":record.metadata}))
                            .collect::<Vec<_>>()
                    };
                    Ok(Ok(json!({
                        "knowledge_nodes": selected_records(&visible_knowledge, &selected),
                        "conversation_nodes": selected_records(&visible_conversation_nodes, &conversation_node_ids.iter().cloned().collect()),
                        "conversation_edges": selected_records(&visible_conversation_edges, &edge_ids.iter().cloned().collect())
                    })))
                })
            })
            .await
            .map_err(|_| NativeEvidenceContextError::GraphReadUnavailable)?;
        result.map_err(|error| match error {
            NativeEvidenceRecordError::Rejected => NativeEvidenceContextError::GraphReadRejected,
            NativeEvidenceRecordError::Unavailable => {
                NativeEvidenceContextError::GraphReadUnavailable
            }
        })
    }

    async fn native_turn_link_context(
        &self,
        request: &kogwistar_runtime::RuntimeNativeStepRequest,
        caller: Option<&Value>,
    ) -> Result<Value, NativeEvidenceContextError> {
        let Some(response_id) = request
            .state
            .get("answer")
            .and_then(Value::as_object)
            .and_then(|answer| {
                answer
                    .get("response_node_id")
                    .or_else(|| answer.get("assistant_turn_node_id"))
            })
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .or_else(|| {
                request
                    .state
                    .get("agentic_answering_result")
                    .and_then(Value::as_object)
                    .and_then(|result| {
                        result
                            .get("assistant_turn_node_id")
                            .or_else(|| result.get("response_node_id"))
                    })
                    .and_then(Value::as_str)
                    .filter(|value| !value.trim().is_empty())
            })
        else {
            return Ok(json!({"nodes": [], "edges": []}));
        };
        let principal = self.native_read_principal(request, caller).await?;
        let user_turn_id = request
            .state
            .get("turn_node_id")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .ok_or(NativeEvidenceContextError::GraphReadRejected)?;
        let conversation_id = request
            .state
            .get("conversation_id")
            .and_then(Value::as_str)
            .filter(|value| *value == request.conversation_id)
            .ok_or(NativeEvidenceContextError::GraphReadRejected)?;
        let records = self
            .authorized_evidence_records(
                &principal,
                &[user_turn_id.to_owned(), response_id.to_owned()],
                &[],
                "conversation",
            )
            .await
            .map_err(|error| match error {
                NativeEvidenceRecordError::Rejected => {
                    NativeEvidenceContextError::GraphReadRejected
                }
                NativeEvidenceRecordError::Unavailable => {
                    NativeEvidenceContextError::GraphReadUnavailable
                }
            })?;
        let nodes = records
            .get("nodes")
            .and_then(Value::as_array)
            .ok_or(NativeEvidenceContextError::GraphReadRejected)?;
        if nodes.len() != 2 {
            return Err(NativeEvidenceContextError::GraphReadRejected);
        }
        for (expected_id, expected_type) in [
            (user_turn_id, "conversation_turn"),
            (response_id, "assistant_turn"),
        ] {
            let record = nodes
                .iter()
                .find(|record| record.get("id").and_then(Value::as_str) == Some(expected_id))
                .ok_or(NativeEvidenceContextError::GraphReadRejected)?;
            let metadata = record
                .get("metadata")
                .and_then(Value::as_object)
                .ok_or(NativeEvidenceContextError::GraphReadRejected)?;
            let actual_type = metadata.get("entity_type").and_then(Value::as_str);
            let valid_type = actual_type == Some(expected_type)
                || (expected_type == "assistant_turn" && actual_type == Some("conversation_turn"));
            if record.get("id").and_then(Value::as_str) != Some(expected_id)
                || !valid_type
                || metadata.get("conversation_id").and_then(Value::as_str) != Some(conversation_id)
                || matches!(
                    metadata.get("lifecycle_status").and_then(Value::as_str),
                    Some("tombstoned" | "deleted")
                )
            {
                return Err(NativeEvidenceContextError::GraphReadRejected);
            }
        }
        Ok(records)
    }
}
