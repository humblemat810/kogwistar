const EMBEDDING_PROFILE_NAMESPACE: &str = "__kogwistar_embedding_profiles_v1__";
const EMBEDDING_PROFILE_SCHEMA_VERSION: i64 = 1;

#[derive(Clone, Copy)]
enum RuntimeGraphCommitDenial {
    Owner,
    Capability,
    Scope,
}

enum RuntimeGraphMutationBatch<'a> {
    V1(&'a GraphMutationBatchInput),
    V2(&'a kogwistar_contracts::GraphMutationBatchV2),
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
                "embedding profile binding does not support Unix-socket PostgreSQL DSNs"
                    .to_owned(),
            );
        }
        None => "localhost".to_owned(),
    };
    let port = config.get_ports().first().copied().unwrap_or(5432);
    let database = config.get_dbname().unwrap_or_default().trim();
    let tables = "gke_nodes:gke_edges:gke_documents:gke_domains";
    let canonical = format!("host={host}|port={port}|database={database}|schema={schema}|tables={tables}");
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
    registered.validate().map_err(|_| {
        format!("embedding profile registry record is invalid for {storage_scope}")
    })?;
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

fn conversation_record_value(record: &GraphRecord, field: &str) -> Option<Value> {
    record
        .metadata
        .get(field)
        .cloned()
        .or_else(|| {
            record
                .document
                .as_deref()
                .and_then(|document| serde_json::from_str::<Value>(document).ok())
                .and_then(|document| document.get(field).cloned())
        })
}

fn conversation_record_string(record: &GraphRecord, field: &str) -> Option<String> {
    conversation_record_value(record, field)
        .and_then(|value| value.as_str().map(str::to_owned))
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
    let previous_char_distance = conversation_record_i64(
        previous,
        "char_distance_from_last_summary",
    )
    .unwrap_or(0)
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
    let turn_distance = conversation_record_i64(
        previous,
        "turn_distance_from_last_summary",
    )
    .unwrap_or(-1)
        + 2;
    (char_distance, turn_distance, turn_index)
}

impl PostgresRunApplicationService {
    pub fn from_dsn(dsn: &str, schema: &str) -> Result<Self, String> {
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
        let embedding_storage_scope = rust_pgvector_embedding_scope(&config, schema);
        let native_operations = crate::native_answer_persist::registry()
            .map_err(|error| format!("could not register native conversation operations: {error}"))?;
        PostgresStore::from_dsn(&dsn, schema)
            .map(|store| Self {
                store,
                max_queue: runtime_max_queue(),
                capabilities: Arc::new(Mutex::new(CapabilityState::default())),
                native_operations,
                owner_auth_store: None,
                embedding_provider: None,
                embedding_storage_scope,
            })
            .map_err(|error| error.to_string())
    }

    pub async fn ensure_schema(&self) -> Result<(), String> {
        self.store
            .ensure_schema()
            .await
            .map_err(|error| error.to_string())?;
        if let Some(provider) = &self.embedding_provider {
            self.ensure_embedding_profile_binding(provider.profile()).await?;
        }
        Ok(())
    }

    async fn ensure_embedding_profile_binding(
        &self,
        configured: &kogwistar_engine::embedding::EmbeddingProfile,
    ) -> Result<(), String> {
        configured.validate().map_err(|error| error.to_string())?;
        let storage_scope = self
            .embedding_storage_scope
            .as_ref()
            .map_err(Clone::clone)?;
        if let Some(current) = self
            .store
            .get_named_projection(EMBEDDING_PROFILE_NAMESPACE, storage_scope)
            .await
            .map_err(|error| error.to_string())?
        {
            return validate_embedding_profile_projection(&current, configured, storage_scope);
        }
        if self
            .store
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
        let write = NamedProjectionWrite {
            payload,
            last_authoritative_seq: 0,
            last_materialized_seq: 0,
            projection_schema_version: EMBEDDING_PROFILE_SCHEMA_VERSION,
            materialization_status: "bound".to_owned(),
        };
        let inserted = self
            .store
            .compare_and_swap_named_projection(
                EMBEDDING_PROFILE_NAMESPACE,
                storage_scope,
                None,
                None,
                write,
            )
            .await
            .map_err(|error| error.to_string())?;
        if inserted {
            return Ok(());
        }
        let winner = self
            .store
            .get_named_projection(EMBEDDING_PROFILE_NAMESPACE, storage_scope)
            .await
            .map_err(|error| error.to_string())?
            .ok_or_else(|| "embedding profile binding disappeared during startup".to_owned())?;
        validate_embedding_profile_projection(&winner, configured, storage_scope)
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

    pub async fn with_owner_auth_config(
        mut self,
        config: &AuthConfig,
    ) -> Result<Self, String> {
        self.owner_auth_store = Some(Arc::new(AuthStore::from_config(config).await?));
        Ok(self)
    }

    async fn resolve_inline_native_owner_principal(
        &self,
        run_id: &str,
        required_capability: &str,
    ) -> Result<Value, String> {
        let run = self
            .store
            .server_run(run_id)
            .await
            .map_err(|error| error.to_string())?
            .ok_or_else(|| "native worker run is missing".to_owned())?;
        let owner_id = run
            .user_id
            .as_deref()
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| "native worker run has no authenticated owner".to_owned())?;
        let created = self
            .store
            .server_run_events(run_id, 0, 1)
            .await
            .map_err(|error| error.to_string())?
            .into_iter()
            .find(|event| event.event_type == "run.created")
            .ok_or_else(|| "native worker run has no admission event".to_owned())?;
        let created_payload: Value = serde_json::from_str(&created.payload_json)
            .map_err(|_| "native worker admission event is invalid".to_owned())?;
        let binding: RuntimeOwnerBindingV1 = serde_json::from_value(
            created_payload
                .get("runtime_owner_binding")
                .cloned()
                .ok_or_else(|| "native worker run has no trusted owner binding".to_owned())?,
        )
        .map_err(|_| "native worker owner binding is invalid".to_owned())?;
        if binding.user_id != owner_id {
            return Err("native worker owner binding does not match the durable run owner".to_owned());
        }
        let auth_store = self
            .owner_auth_store
            .as_ref()
            .ok_or_else(|| "native worker owner revalidation is not configured".to_owned())?;
        let user = auth_store
            .auth_user(owner_id)
            .await?
            .filter(|user| user.is_active)
            .ok_or_else(|| "native worker run owner is missing or inactive".to_owned())?;
        revalidated_runtime_owner_principal(
            &binding,
            &user,
            &self.capabilities,
            required_capability,
        )
    }

    async fn native_evidence_context(
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
        self.authorized_evidence_records(&principal, &node_ids, &edge_ids)
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

    pub fn with_max_queue(mut self, max_queue: usize) -> Self {
        self.max_queue = max_queue.max(1);
        self
    }

    /// Opt in to server-side embedding for synchronous graph mutation batches.
    /// Batches carrying index jobs retain their asynchronous two-stage path.
    pub fn with_embedding_provider(mut self, provider: Arc<dyn EmbeddingProvider>) -> Self {
        self.embedding_provider = Some(provider);
        self
    }

    /// Run one bounded batch of durable embedding jobs entirely through the
    /// Rust-owned PostgreSQL queue, canonical projection reads, and promotion
    /// transaction. Other index-job kinds remain unclaimed.
    pub async fn run_embedding_projection_jobs_once(
        &self,
        limit: usize,
        lease_seconds: i64,
        namespace: Option<&str>,
    ) -> Result<kogwistar_engine::projection_worker::ProjectionWorkerSummary, String> {
        if limit == 0 || limit > 1000 || lease_seconds <= 0 {
            return Err("limit must be 1..=1000 and lease_seconds must be positive".to_owned());
        }
        if namespace.is_some_and(|value| value.trim().is_empty()) {
            return Err("namespace must not be empty".to_owned());
        }
        let provider = self
            .embedding_provider
            .as_ref()
            .cloned()
            .ok_or_else(|| "Rust embedding provider is not configured".to_owned())?;
        provider
            .profile()
            .validate()
            .map_err(|error| error.to_string())?;
        self.ensure_schema().await?;

        let store = self.store.clone();
        let index_filter = IndexJobClaimFilter {
            namespace: namespace.map(str::to_owned),
            index_kinds: vec!["node_embedding".to_owned()],
        };
        kogwistar_engine::projection_worker::run_graph_projection_jobs_once_filtered_with_provider_input(
            &self.store,
            &self.store,
            limit,
            lease_seconds,
            index_filter,
            move |input| {
                let provider = provider.clone();
                let store = store.clone();
                async move {
                    if input.schema_version != 1
                        || input.index_kind != "node_embedding"
                        || !matches!(input.entity_kind.as_str(), "node" | "edge")
                    {
                        return Err(kogwistar_store::StoreError::Backend {
                            backend: "postgres-embedding-worker".to_owned(),
                            message: "unsupported embedding job contract".to_owned(),
                        });
                    }
                    let table = kogwistar_store::graph_projection_table(&input.entity_kind)?;
                    let mut records = store
                        .graph_projection_records(GraphProjectionRead {
                            scope: GraphScope {
                                namespace: input.namespace.clone(),
                                workspace_id: None,
                                graph_space: None,
                            },
                            table: table.to_owned(),
                            ids: Some(vec![input.entity_id.clone()]),
                            metadata: MetadataFilter::default(),
                            limit: 1,
                        })
                        .await
                        .map_err(|_| kogwistar_store::StoreError::Backend {
                            backend: "postgres-embedding-worker".to_owned(),
                            message: "canonical graph projection read failed".to_owned(),
                        })?;
                    let Some(mut record) = records.pop() else {
                        return Err(kogwistar_store::StoreError::Backend {
                            backend: "postgres-embedding-worker".to_owned(),
                            message: "canonical graph projection is missing".to_owned(),
                        });
                    };
                    if record.id != input.entity_id {
                        return Err(kogwistar_store::StoreError::Backend {
                            backend: "postgres-embedding-worker".to_owned(),
                            message: "canonical graph projection identity mismatch".to_owned(),
                        });
                    }
                    let document = record.document.clone().ok_or_else(|| {
                        kogwistar_store::StoreError::Backend {
                            backend: "postgres-embedding-worker".to_owned(),
                            message: "canonical graph document is missing".to_owned(),
                        }
                    })?;
                    let vectors = provider
                        .embed(std::slice::from_ref(&document))
                        .await
                        .map_err(|_| kogwistar_store::StoreError::Backend {
                            backend: "postgres-embedding-worker".to_owned(),
                            message: "embedding provider request failed".to_owned(),
                        })?;
                    if vectors.len() != 1
                        || vectors[0].len() != provider.profile().dimension
                        || vectors[0].iter().any(|value| !value.is_finite())
                    {
                        return Err(kogwistar_store::StoreError::Backend {
                            backend: "postgres-embedding-worker".to_owned(),
                            message: "embedding provider returned an invalid vector".to_owned(),
                        });
                    }
                    let vector = vectors[0]
                        .iter()
                        .map(|value| *value as f32)
                        .collect::<Vec<_>>();
                    if vector.iter().any(|value| !value.is_finite()) {
                        return Err(kogwistar_store::StoreError::NonFiniteVector);
                    }
                    record.embedding = Some(vector);
                    Ok(record)
                }
            },
        )
        .await
        .map_err(|error| error.to_string())
    }

    async fn send_message_syscall(
        &self,
        request: &ApiEffectRequest,
        input: &SyscallInput,
    ) -> ApiEffectResponse {
        let conversation_id = input.args.get("conversation_id")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty());
        let Some(conversation_id) = conversation_id else {
            return effect_error(
                StatusCode::BAD_REQUEST,
                "KOGWISTAR_INVALID_REQUEST",
                "conversation_id is required",
            );
        };
        let text = input.args.get("text")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty());
        let Some(text) = text else {
            return effect_error(
                StatusCode::BAD_REQUEST,
                "KOGWISTAR_INVALID_REQUEST",
                "text must be non-empty",
            );
        };
        let workflow_id = input.args["workflow_id"]
            .as_str()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or("agentic_answering.v2");
        let (scope, acl_principal) =
            conversation_graph_scope_and_acl_principal(&request.principal);
        let nodes = match self.authorized_conversation_nodes(&request.principal).await {
            Ok(nodes) => nodes,
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error,
                );
            }
        };
        let conversation_nodes = nodes
            .into_iter()
            .filter(|record| {
                conversation_record_string(record, "conversation_id").as_deref()
                    == Some(conversation_id)
                    && !matches!(
                        conversation_record_string(record, "lifecycle_status").as_deref(),
                        Some("tombstoned" | "deleted")
                    )
            })
            .collect::<Vec<_>>();
        let mut starts = conversation_nodes
            .iter()
            .filter(|record| {
                conversation_record_string(record, "entity_type").as_deref()
                    == Some("conversation_start")
            })
            .collect::<Vec<_>>();
        starts.sort_by_key(|record| record.id.clone());
        let Some(start) = starts.first() else {
            return effect_error(
                StatusCode::NOT_FOUND,
                "KOGWISTAR_CONVERSATION_NOT_FOUND",
                format!("Unknown conversation_id: {conversation_id}"),
            );
        };
        if starts.len() != 1 {
            return effect_error(
                StatusCode::CONFLICT,
                "KOGWISTAR_CONVERSATION_INVALID",
                "conversation must have exactly one visible conversation_start node",
            );
        }
        let Some(owner_id) = conversation_record_string(start, "user_id")
            .filter(|value| !value.trim().is_empty())
        else {
            return effect_error(
                StatusCode::CONFLICT,
                "KOGWISTAR_CONVERSATION_INVALID",
                "conversation owner is missing",
            );
        };
        let requested_user_id = match input.args.get("user_id") {
            None | Some(Value::Null) => None,
            Some(Value::String(requested)) if requested.trim().is_empty() => None,
            Some(Value::String(requested)) => Some(requested.as_str()),
            Some(_) => {
                return effect_error(
                    StatusCode::BAD_REQUEST,
                    "KOGWISTAR_INVALID_REQUEST",
                    "user_id must be a string when provided",
                );
            }
        };
        let effective_user_id = requested_user_id.or_else(|| {
            authenticated_run_owner_id(&request.principal)
        });
        if effective_user_id != Some(owner_id.as_str()) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_RUN_OWNER_FORBIDDEN",
                "user_id must match the owner of the target conversation",
            );
        }
        let Some(auth_store) = self.owner_auth_store.as_ref() else {
            return effect_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "KOGWISTAR_OWNER_REVALIDATION_UNAVAILABLE",
                "conversation owner revalidation is not configured",
            );
        };
        let owner_user = match auth_store.auth_user(&owner_id).await {
            Ok(Some(user)) if user.is_active => user,
            Ok(_) => {
                return effect_error(
                    StatusCode::FORBIDDEN,
                    "KOGWISTAR_RUN_OWNER_FORBIDDEN",
                    "conversation owner is missing or inactive",
                );
            }
            Err(error) => {
                return effect_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "KOGWISTAR_OWNER_REVALIDATION_UNAVAILABLE",
                    error,
                );
            }
        };
        let mut owner_principal = runtime_principal_from_auth_user(&owner_user);
        if !principal_strings(&owner_principal, "ns")
            .iter()
            .any(|namespace| namespace == &scope.namespace || namespace == "*")
        {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_SCOPE_FORBIDDEN",
                "conversation owner is not authorized for the conversation namespace",
            );
        }
        owner_principal["ns"] = json!([scope.namespace]);
        owner_principal["storage_ns"] = json!(scope.namespace);
        let Some(owner_binding) = runtime_owner_binding_v1(&owner_principal, &self.capabilities)
        else {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_RUN_OWNER_FORBIDDEN",
                "conversation owner has no authenticated run identity",
            );
        };
        owner_principal = match revalidated_runtime_owner_principal(
            &owner_binding,
            &owner_user,
            &self.capabilities,
            "write_graph",
        ) {
            Ok(principal) => principal,
            Err(error) => {
                return effect_error(
                    StatusCode::FORBIDDEN,
                    "KOGWISTAR_RUN_OWNER_FORBIDDEN",
                    error,
                );
            }
        };

        let Some(provider) = self.embedding_provider.as_ref() else {
            return effect_error(
                StatusCode::BAD_GATEWAY,
                "KOGWISTAR_EMBEDDING_FAILED",
                "conversation message admission requires a configured embedding provider",
            );
        };
        let profile = provider.profile();
        if let Err(error) = profile.validate() {
            return effect_error(
                StatusCode::BAD_GATEWAY,
                "KOGWISTAR_EMBEDDING_FAILED",
                error.to_string(),
            );
        }
        let turn_id = uuid::Uuid::new_v4().to_string();
        let run_id = uuid::Uuid::new_v4().to_string();
        let mut tail = conversation_nodes
            .iter()
            .filter(|record| {
                matches!(
                    conversation_record_string(record, "entity_type").as_deref(),
                    Some("conversation_start" | "conversation_turn" | "conversation_summary" | "assistant_turn")
                ) && conversation_record_value(record, "in_conversation_chain")
                    .and_then(|value| value.as_bool())
                    .unwrap_or(true)
            })
            .cloned()
            .collect::<Vec<_>>();
        tail.sort_by_key(|record| {
            (
                conversation_record_i64(record, "turn_index").unwrap_or(-1),
                record.id.clone(),
            )
        });
        let previous = tail.last();
        let turn_index = previous
            .and_then(|record| conversation_record_i64(record, "turn_index"))
            .map(|index| index + 1)
            .unwrap_or(0);
        let step_seq = previous
            .and_then(|record| conversation_record_i64(record, "run_step_seq"))
            .unwrap_or(0)
            + 1;
        let span = conversation_span(conversation_id, &turn_id, text);
        let node_metadata = json!({
            "entity_type": "conversation_turn",
            "turn_index": turn_index,
            "level_from_root": 0,
            "in_conversation_chain": true,
            "in_ui_chain": true,
            "run_id": conversation_id,
            "run_step_seq": step_seq,
            "attempt_seq": 0,
            "lifecycle_status": "active",
            "redirect_to_id": null,
            "role": "user",
            "conversation_id": conversation_id,
            "user_id": owner_id,
        });
        let node_mentions = json!([{"spans": [span], "multimodal_spans": []}]);
        let node_payload = json!({
            "id": turn_id,
            "label": format!("Turn {turn_index} (user)"),
            "type": "entity",
            "summary": text,
            "domain_id": null,
            "canonical_entity_id": null,
            "properties": {},
            "mentions": node_mentions,
            "doc_id": turn_id,
            "metadata": node_metadata,
            "level_from_root": 0,
            "role": "user",
            "turn_index": turn_index,
            "conversation_id": conversation_id,
            "user_id": owner_id,
        });
        if let Err(error) = kogwistar_contracts::GraphModelEnvelope::new(
            kogwistar_contracts::GraphModelClass::ConversationNode,
            node_payload.clone(),
        )
        .validate_family_v1()
        {
            return effect_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "KOGWISTAR_CONVERSATION_MODEL_INVALID",
                error.to_string(),
            );
        }
        let node_document = {
            let mut value = node_payload.clone();
            value.as_object_mut().unwrap().remove("metadata");
            serde_json::to_string(&value).expect("conversation node payload serializes")
        };
        let mut node_metadata_projection = node_metadata.as_object().unwrap().clone();
        node_metadata_projection.extend([
            ("doc_id".to_owned(), json!(turn_id)),
            ("label".to_owned(), json!(format!("Turn {turn_index} (user)"))),
            ("type".to_owned(), json!("entity")),
            ("summary".to_owned(), json!(text)),
            ("properties".to_owned(), json!("{}")),
            ("mentions".to_owned(), json!(serde_json::to_string(&node_mentions).unwrap())),
        ]);
        if let Some(graph_space) = &scope.graph_space {
            node_metadata_projection.insert("graph_space".to_owned(), json!(graph_space));
        }
        if let Some(workspace_id) = &scope.workspace_id {
            node_metadata_projection.insert("workspace_id".to_owned(), json!(workspace_id));
        }

        let mut mutations = Vec::new();
        let Some(node_index_kinds) = conversation_index_job_kinds("node") else {
            return effect_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "KOGWISTAR_INDEX_CONTRACT_INVALID",
                "conversation node index contract is unavailable",
            );
        };
        let mut index_jobs = node_index_kinds
            .iter()
            .map(|kind| conversation_index_job(&scope.namespace, "node", &turn_id, kind))
            .collect::<Vec<_>>();
        let mut edge_payload_and_document = None;
        if let Some(previous) = previous {
            let previous_id = previous.id.clone();
            let (char_distance, turn_distance, tail_turn_index) =
                conversation_next_turn_distances(previous, text, turn_index);
            let edge_index = turn_index + 1;
            let edge_id = kogwistar_contracts::stable_id(
                "conversation.edge",
                &[
                    owner_id.clone(),
                    conversation_id.to_owned(),
                    "next_turn".to_owned(),
                    edge_index.to_string(),
                    format!("['{previous_id}']"),
                    format!("['{turn_id}']"),
                    "[]".to_owned(),
                    "[]".to_owned(),
                    "conversation_edge".to_owned(),
                ],
            )
            .to_string();
            let edge_metadata = json!({
                "relation": "next_turn",
                "target_id": turn_id,
                "char_distance_from_last_summary": char_distance,
                "turn_distance_from_last_summary": turn_distance,
                "tail_turn_index": tail_turn_index,
                "causal_type": "chain",
                "run_id": conversation_id,
                "run_step_seq": step_seq,
                "attempt_seq": 0,
                "lifecycle_status": "active",
                "redirect_to_id": null,
            });
            let edge_payload = json!({
                "id": edge_id,
                "source_ids": [previous_id],
                "target_ids": [turn_id],
                "source_edge_ids": [],
                "target_edge_ids": [],
                "relation": "next_turn",
                "label": "next_turn",
                "type": "relationship",
                "summary": "Sequential flow",
                "doc_id": format!("conv:{conversation_id}"),
                "mentions": node_mentions,
                "domain_id": null,
                "canonical_entity_id": null,
                "properties": {"entity_type": "conversation_edge"},
                "metadata": edge_metadata,
            });
            if let Err(error) = kogwistar_contracts::GraphModelEnvelope::new(
                kogwistar_contracts::GraphModelClass::ConversationEdge,
                edge_payload.clone(),
            )
            .validate_family_v1()
            {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_CONVERSATION_MODEL_INVALID",
                    error.to_string(),
                );
            }
            let edge_document = serde_json::to_string(&edge_payload)
                .expect("conversation edge payload serializes");
            edge_payload_and_document = Some((edge_payload, edge_document, edge_id));
        }
        let embedding_texts = edge_payload_and_document
            .as_ref()
            .map(|(_, edge_document, _)| vec![format!("user: {text}"), edge_document.clone()])
            .unwrap_or_else(|| vec![format!("user: {text}")]);
        let embeddings = match provider.embed(&embedding_texts).await {
            Ok(vectors)
                if vectors.len() == embedding_texts.len()
                    && vectors.iter().all(|vector| {
                        vector.len() == profile.dimension && vector.iter().all(|v| v.is_finite())
                    }) =>
            {
                vectors
            }
            Ok(_) => {
                return effect_error(
                    StatusCode::BAD_GATEWAY,
                    "KOGWISTAR_EMBEDDING_FAILED",
                    "embedding provider returned an invalid conversation vector",
                );
            }
            Err(error) => {
                return effect_error(
                    StatusCode::BAD_GATEWAY,
                    "KOGWISTAR_EMBEDDING_FAILED",
                    error.to_string(),
                );
            }
        };
        let node_mutation = GraphMutation {
            scope: scope.clone(),
            table: "gke_nodes".to_owned(),
            entity_kind: "node".to_owned(),
            event_id: uuid::Uuid::new_v4().to_string(),
            op: "ADD".to_owned(),
            payload: node_payload.clone(),
            record: GraphRecord {
                id: turn_id.clone(),
                document: Some(node_document),
                metadata: node_metadata_projection,
                embedding: Some(embeddings[0].iter().map(|v| *v as f32).collect()),
            },
            embedding_dim: profile.dimension,
        };
        mutations.push((
            node_mutation,
            kogwistar_contracts::GraphModelEnvelope::new(
                kogwistar_contracts::GraphModelClass::ConversationNode,
                node_payload,
            ),
        ));
        if let Some((edge_payload, edge_document, edge_id)) = edge_payload_and_document {
            let edge_metadata = edge_payload["metadata"].as_object().unwrap().clone();
            let source_ids = serde_json::to_string(&edge_payload["source_ids"])
                .expect("edge source IDs serialize");
            let target_ids = serde_json::to_string(&edge_payload["target_ids"])
                .expect("edge target IDs serialize");
            let source_edge_ids = serde_json::to_string(&edge_payload["source_edge_ids"])
                .expect("edge source-edge IDs serialize");
            let target_edge_ids = serde_json::to_string(&edge_payload["target_edge_ids"])
                .expect("edge target-edge IDs serialize");
            let properties = serde_json::to_string(&edge_payload["properties"])
                .expect("edge properties serialize");
            let mentions = serde_json::to_string(&edge_payload["mentions"])
                .expect("edge mentions serialize");
            let mut metadata = edge_metadata;
            metadata.extend([
                ("doc_id".to_owned(), edge_payload["doc_id"].clone()),
                ("relation".to_owned(), json!("next_turn")),
                ("source_ids".to_owned(), json!(source_ids)),
                ("target_ids".to_owned(), json!(target_ids)),
                ("source_edge_ids".to_owned(), json!(source_edge_ids)),
                ("target_edge_ids".to_owned(), json!(target_edge_ids)),
                ("type".to_owned(), json!("relationship")),
                ("summary".to_owned(), json!("Sequential flow")),
                ("properties".to_owned(), json!(properties)),
                ("mentions".to_owned(), json!(mentions)),
                ("node_endpoint_count".to_owned(), json!(2)),
                ("edge_endpoint_count".to_owned(), json!(0)),
                ("total_endpoint_count".to_owned(), json!(2)),
            ]);
            if let Some(graph_space) = &scope.graph_space {
                metadata.insert("graph_space".to_owned(), json!(graph_space));
            }
            if let Some(workspace_id) = &scope.workspace_id {
                metadata.insert("workspace_id".to_owned(), json!(workspace_id));
            }
            mutations.push((
                GraphMutation {
                    scope: scope.clone(),
                    table: "gke_edges".to_owned(),
                    entity_kind: "edge".to_owned(),
                    event_id: uuid::Uuid::new_v4().to_string(),
                    op: "ADD".to_owned(),
                    payload: edge_payload.clone(),
                    record: GraphRecord {
                        id: edge_id.clone(),
                        document: Some(edge_document),
                        metadata,
                        embedding: Some(embeddings[1].iter().map(|v| *v as f32).collect()),
                    },
                    embedding_dim: profile.dimension,
                },
                kogwistar_contracts::GraphModelEnvelope::new(
                    kogwistar_contracts::GraphModelClass::ConversationEdge,
                    edge_payload,
                ),
            ));
            let Some(edge_index_kinds) = conversation_index_job_kinds("edge") else {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_INDEX_CONTRACT_INVALID",
                    "conversation edge index contract is unavailable",
                );
            };
            index_jobs.extend(edge_index_kinds.iter().map(|kind| {
                conversation_index_job(&scope.namespace, "edge", &edge_id, kind)
            }));
        }
        let prepared = PreparedConversationAdmission {
            conversation_id: conversation_id.to_owned(),
            owner_id: owner_id.clone(),
            expected_tail_id: previous.map(|record| record.id.clone()),
            scope: scope.clone(),
            acl_principal,
            mutations,
            index_jobs,
            actor_user_id: authenticated_run_owner_id(&request.principal)
                .map(str::to_owned)
                .unwrap_or_else(|| principal_subject(&request.principal)),
        };
        let mut initial_state = serde_json::Map::new();
        initial_state.insert("user_text".to_owned(), json!(text));
        initial_state.insert("turn_node_id".to_owned(), json!(turn_id));
        initial_state.insert("user_id".to_owned(), json!(owner_id));
        let nested = ApiEffectRequest {
            contract_version: request.contract_version,
            method: "POST".to_owned(),
            path_and_query: "/api/workflow/runs".to_owned(),
            body: serde_json::to_vec(&json!({
                "run_id": run_id,
                "workflow_id": workflow_id,
                "conversation_id": conversation_id,
                "turn_node_id": turn_id,
                "user_id": owner_id,
                "initial_state": initial_state,
            }))
            .expect("message workflow request serializes"),
            principal: owner_principal,
        };
        self.submit_run_with_conversation_admission(&nested, Some(prepared))
            .await
    }

    async fn submit_run(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
        self.submit_run_with_conversation_admission(request, None)
            .await
    }

    async fn submit_run_with_conversation_admission(
        &self,
        request: &ApiEffectRequest,
        conversation_admission: Option<PreparedConversationAdmission>,
    ) -> ApiEffectResponse {
        let input: RuntimeSubmitRun = match serde_json::from_slice(&request.body) {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error.to_string(),
                );
            }
        };
        let workflow_id = input.workflow_id.trim().to_owned();
        let conversation_id = input.conversation_id.trim().to_owned();
        if workflow_id.is_empty() || conversation_id.is_empty() {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                "workflow_id and conversation_id are required",
            );
        }
        let authenticated_owner = authenticated_run_owner_id(&request.principal);
        if input
            .user_id
            .as_deref()
            .is_some_and(|requested| {
                authenticated_owner.is_none_or(|owner| owner != requested.trim())
            })
        {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_RUN_OWNER_FORBIDDEN",
                "run user_id must match the authenticated principal",
            );
        }
        let run_user_id = authenticated_owner.map(str::to_owned);
        let owner_binding = runtime_owner_binding_v1(&request.principal, &self.capabilities);
        let run_id = input
            .run_id
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let turn_node_id = input
            .turn_node_id
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
            .unwrap_or_else(|| format!("wf_turn|{run_id}"));
        let message_id = format!("lane|{}", uuid::Uuid::new_v4());
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|value| value.as_secs() as i64)
            .unwrap_or(0);
        let priority_class = if input.priority_class.is_empty() {
            "foreground".to_owned()
        } else {
            input.priority_class.clone()
        };
        let runtime_kind = if input.runtime_kind.is_empty() {
            "sync".to_owned()
        } else {
            input.runtime_kind.clone()
        };
        let worker_protocol = worker_protocol_for_runtime_kind(&runtime_kind);
        let graph_plan = match (
            self.store
                .get_named_projection("workflow_design", &workflow_id)
                .await,
            self.store
                .get_workflow_design_snapshot(&workflow_id, i64::MAX, 1)
                .await,
        ) {
            (Ok(projection), Ok(snapshot)) => exact_runtime_graph_plan(projection, snapshot),
            _ => None,
        };
        if !answer_workflow_admission_ready(
            &workflow_id,
            graph_plan.as_ref(),
            &self.native_operations,
        ) {
            return effect_error(
                StatusCode::CONFLICT,
                "KOGWISTAR_ANSWER_WORKFLOW_NOT_READY",
                "the selected answer workflow has no complete authoritative plan with a compatible executor",
            );
        }
        if graph_plan
            .as_ref()
            .is_some_and(|plan| submitted_runtime_graph_conflicts(plan, &input))
        {
            return effect_error(
                StatusCode::CONFLICT,
                "KOGWISTAR_RUNTIME_GRAPH_CONFLICT",
                "submitted frozen runtime graph differs from authoritative workflow design",
            );
        }
        let effective_routes = graph_plan
            .as_ref()
            .map(|plan| plan.routes.clone())
            .unwrap_or(input.runtime_routes);
        let effective_join_node_ids = graph_plan
            .as_ref()
            .map(|plan| plan.join_node_ids.clone())
            .unwrap_or(input.join_node_ids);
        let effective_start_join_mask = graph_plan
            .as_ref()
            .map(|plan| plan.start_join_mask)
            .unwrap_or(input.start_join_mask);
        let effective_start_node_id = graph_plan
            .as_ref()
            .map(|plan| plan.start_node_id.clone())
            .or_else(|| {
                input
                    .start_node_id
                    .as_deref()
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .map(str::to_owned)
            })
            .unwrap_or_else(|| "start".to_owned());
        let effective_node_ops = graph_plan
            .as_ref()
            .map(|plan| plan.node_ops.clone())
            .unwrap_or(input.node_ops);
        let effective_node_versions = graph_plan
            .as_ref()
            .map(|plan| plan.node_versions.clone())
            .unwrap_or_else(|| {
                if input.node_versions.is_empty() {
                    effective_node_ops
                        .keys()
                        .map(|node_id| (node_id.clone(), "v1".to_owned()))
                        .collect()
                } else {
                    input.node_versions
                }
            });
        let effective_node_environment_ids = graph_plan
            .as_ref()
            .map(|plan| plan.node_environment_ids.clone())
            .unwrap_or(input.node_environment_ids);
        let effective_node_executor_targets = graph_plan
            .as_ref()
            .map(|plan| plan.node_executor_targets.clone())
            .unwrap_or(input.node_executor_targets);
        // Keep PostgreSQL admission wire-identical to SQLite: the worker must
        // receive a frozen operation, never a null value requiring a node-id
        // resolver guess.
        let effective_start_op = effective_node_ops
            .get(&effective_start_node_id)
            .cloned()
            .unwrap_or_else(|| "noop".to_owned());
        let effective_start_op_version = effective_node_versions
            .get(&effective_start_node_id)
            .cloned()
            .unwrap_or_else(|| "v1".to_owned());
        let effective_start_environment_id = effective_node_environment_ids
            .get(&effective_start_node_id)
            .cloned();
        let effective_start_executor_target = effective_node_executor_targets
            .get(&effective_start_node_id)
            .cloned();
        let mut initial_state = input.initial_state;
        initial_state.remove("_rt_graph_scope");
        initial_state.remove("_rt_run_owner_id");
        initial_state.insert(
            "_rt_graph_scope".to_owned(),
            security_scope_value(&request.principal),
        );
        if let Some(owner_id) = run_user_id.as_deref() {
            initial_state.insert("_rt_run_owner_id".to_owned(), json!(owner_id));
        }
        install_authoritative_node_executor_targets(
            &mut initial_state,
            &effective_node_executor_targets,
        );
        inject_authoritative_capability_state(
            &mut initial_state,
            &request.principal,
            &self.capabilities,
        );
        if !effective_routes.is_empty() {
            initial_state.insert(
                "_rt_routes".to_owned(),
                serde_json::to_value(&effective_routes).expect("static runtime routes serialize"),
            );
        }
        if !effective_node_ops.is_empty() {
            initial_state.insert(
                "_rt_node_ops".to_owned(),
                serde_json::to_value(&effective_node_ops).expect("runtime node ops serialize"),
            );
        }
        if effective_node_versions
            .values()
            .any(|version| version != "v1")
        {
            initial_state.insert(
                "_rt_node_versions".to_owned(),
                serde_json::to_value(&effective_node_versions)
                    .expect("runtime node versions serialize"),
            );
        }
        if !effective_node_environment_ids.is_empty() {
            initial_state.insert(
                "_rt_node_environment_ids".to_owned(),
                serde_json::to_value(&effective_node_environment_ids)
                    .expect("runtime node environment ids serialize"),
            );
        }
        initial_state.insert(
            "_rt_worker_protocol".to_owned(),
            Value::String(worker_protocol.to_owned()),
        );
        if let Some(value) = input.token_budget {
            initial_state.insert("token_budget".to_owned(), json!(value));
        }
        if let Some(value) = input.time_budget_ms {
            initial_state.insert("time_budget_ms".to_owned(), json!(value));
        }
        let response = json!({
            "run_id": run_id,
            "conversation_id": conversation_id,
            "workflow_id": workflow_id,
            "turn_node_id": turn_node_id,
            "status": "queued",
            "priority_class": priority_class,
            "token_budget": input.token_budget,
            "time_budget_ms": input.time_budget_ms,
            "admission": "accepted",
            "lane_message_id": message_id,
        });
        if let Ok(Some(existing)) = self.store.get_server_run(&run_id).await {
            let lanes = self
                .store
                .list_projected_lane_messages(LaneMessageFilter {
                    namespace: Some("workflow".to_owned()),
                    inbox_id: Some("workflow-runtime".to_owned()),
                    correlation_id: Some(run_id.clone()),
                    limit: 10,
                    ..LaneMessageFilter::default()
                })
                .await
                .unwrap_or_default();
            let mut expected_worker_payload = json!({
                "contract_version": 1,
                "kind": "workflow.run.execute",
                "run_id": run_id,
                "workflow_id": workflow_id,
                "conversation_id": conversation_id,
                "turn_node_id": turn_node_id,
                "user_id": run_user_id,
                "initial_state": initial_state,
                "priority_class": priority_class,
                "token_budget": input.token_budget,
                "time_budget_ms": input.time_budget_ms,
                "runtime_kind": runtime_kind,
                "worker_protocol": worker_protocol,
                "join_node_ids": effective_join_node_ids,
                "start_join_mask": effective_start_join_mask,
                "start_node_id": effective_start_node_id,
                "op": effective_start_op,
                "runtime_routes": effective_routes,
            });
            if effective_start_op_version != "v1" {
                expected_worker_payload["op_version"] = json!(effective_start_op_version);
            }
            if let Some(environment_id) = effective_start_environment_id.as_ref() {
                expected_worker_payload["op_environment_id"] = json!(environment_id);
            }
            add_executor_target(
                &mut expected_worker_payload,
                &effective_start_op,
                effective_start_executor_target.as_deref(),
            );
            return match runtime_submit_retry_response(&existing, &lanes, &expected_worker_payload)
            {
                Ok(Some(value)) => json_effect(StatusCode::ACCEPTED, value),
                Ok(None) | Err(_) => effect_error(
                    StatusCode::CONFLICT,
                    "KOGWISTAR_RUNTIME_ADMISSION_CONFLICT",
                    format!("run_id {run_id:?} already exists with different admission state"),
                ),
            };
        }
        if let Err((code, message)) = validate_native_operation_registry(
            &effective_node_ops,
            &effective_node_versions,
            &effective_node_executor_targets,
            &self.native_operations,
        ) {
            return effect_error(StatusCode::UNPROCESSABLE_ENTITY, code, message);
        }
        let max_queue = self.max_queue;
        let queued_priority_class = priority_class.clone();
        let queue_full_admission = if matches!(priority_class.as_str(), "background" | "batch") {
            "rejected"
        } else {
            "deferred"
        };
        let admission_actor_user_id = conversation_admission
            .as_ref()
            .map(|admission| admission.actor_user_id.clone());
        let outcome = self
            .store
            .transaction(|uow| {
                Box::pin(async move {
                    uow.lock_runtime_queue_admission().await?;
                    let pending = uow
                        .list_projected_lane_messages(&LaneMessageFilter {
                            namespace: Some("workflow".to_owned()),
                            inbox_id: Some("workflow-runtime".to_owned()),
                            status: Some("pending".to_owned()),
                            limit: max_queue,
                            ..LaneMessageFilter::default()
                        })
                        .await?;
                    let claimed = uow
                        .list_projected_lane_messages(&LaneMessageFilter {
                            namespace: Some("workflow".to_owned()),
                            inbox_id: Some("workflow-runtime".to_owned()),
                            status: Some("claimed".to_owned()),
                            limit: max_queue,
                            ..LaneMessageFilter::default()
                        })
                        .await?;
                    if pending.len() + claimed.len() >= max_queue {
                        return Ok(RuntimeAdmissionOutcome::QueueFull);
                    }
                    if let Some(admission) = conversation_admission.as_ref() {
                        uow.lock_entity_identity(
                            &admission.scope.namespace,
                            "conversation",
                            &admission.conversation_id,
                        )
                        .await?;
                        let current_nodes = uow
                            .graph_projection_records_for_update(GraphProjectionRead {
                                scope: admission.scope.clone(),
                                table: "gke_nodes".to_owned(),
                                ids: None,
                                metadata: MetadataFilter {
                                    equals: BTreeMap::from([(
                                        "conversation_id".to_owned(),
                                        json!(admission.conversation_id),
                                    )]),
                                },
                                limit: 10_000,
                            })
                            .await?;
                        let acl_nodes = uow
                            .graph_projection_records(GraphProjectionRead {
                                scope: admission.scope.clone(),
                                table: "gke_nodes".to_owned(),
                                ids: None,
                                metadata: MetadataFilter {
                                    equals: BTreeMap::from([(
                                        "entity_type".to_owned(),
                                        json!("acl_record"),
                                    )]),
                                },
                                limit: 10_000,
                            })
                            .await?;
                        let mut acl_input = current_nodes;
                        acl_input.extend(acl_nodes);
                        let (visible_nodes, _) = graph_records_visible_by_acl(
                            &acl_input,
                            &[],
                            "conversation",
                            &admission.acl_principal,
                        );
                        let owner_matches = visible_nodes.iter().any(|record| {
                            conversation_record_string(record, "entity_type").as_deref()
                                == Some("conversation_start")
                                && conversation_record_string(record, "user_id").as_deref()
                                    == Some(admission.owner_id.as_str())
                        });
                        let mut current_tail = visible_nodes
                            .iter()
                            .filter(|record| {
                                matches!(
                                    conversation_record_string(record, "entity_type").as_deref(),
                                    Some("conversation_start" | "conversation_turn" | "conversation_summary" | "assistant_turn")
                                ) && conversation_record_value(record, "in_conversation_chain")
                                    .and_then(|value| value.as_bool())
                                    .unwrap_or(true)
                                    && !matches!(
                                        conversation_record_string(record, "lifecycle_status")
                                            .as_deref(),
                                        Some("tombstoned" | "deleted")
                                    )
                            })
                            .collect::<Vec<_>>();
                        current_tail.sort_by_key(|record| {
                            (
                                conversation_record_i64(record, "turn_index").unwrap_or(-1),
                                record.id.clone(),
                            )
                        });
                        let actual_tail_id = current_tail.last().map(|record| record.id.as_str());
                        if !owner_matches || actual_tail_id != admission.expected_tail_id.as_deref()
                        {
                            return Ok(RuntimeAdmissionOutcome::ConversationChanged);
                        }
                    }
                    uow.create_server_run(ServerRunCreate {
                        run_id: run_id.clone(),
                        conversation_id: conversation_id.clone(),
                        workflow_id: workflow_id.clone(),
                        user_id: run_user_id.clone(),
                        user_turn_node_id: turn_node_id.clone(),
                        status: "queued".to_owned(),
                    })
                    .await?;
                    if let Some(admission) = conversation_admission.as_ref() {
                        for (mutation, envelope) in &admission.mutations {
                            kogwistar_store::validate_typed_graph_mutation(mutation, envelope)?;
                            uow.apply_graph_mutation(mutation.clone()).await?;
                        }
                        for job in &admission.index_jobs {
                            uow.enqueue_index_job(job.clone()).await?;
                        }
                    }
                    let mut created_payload = json!({
                        "run_id": run_id,
                        "run_kind": "workflow_runtime",
                        "conversation_id": conversation_id,
                        "workflow_id": workflow_id,
                        "status": "queued",
                        "turn_node_id": turn_node_id,
                        "runtime_owner_binding": owner_binding,
                        "implementation_attestation": runtime_implementation_attestation(&runtime_kind),
                    });
                    if let Some(actor_user_id) = admission_actor_user_id.as_deref() {
                        created_payload["admission_actor"] = json!({
                            "user_id": actor_user_id,
                            "operation": "send_message",
                        });
                    }
                    let created = uow
                        .append_server_run_event(
                            &run_id,
                            "run.created",
                            serde_json::to_string(&created_payload)?,
                        )
                        .await?;
                    let mut payload = json!({
                        "contract_version": 1,
                        "kind": "workflow.run.execute",
                        "run_id": run_id,
                        "workflow_id": workflow_id,
                        "conversation_id": conversation_id,
                        "turn_node_id": turn_node_id,
                        "user_id": run_user_id,
                        "initial_state": initial_state,
                        "priority_class": queued_priority_class,
                        "token_budget": input.token_budget,
                        "time_budget_ms": input.time_budget_ms,
                        "runtime_kind": runtime_kind,
                        "worker_protocol": worker_protocol,
                        "join_node_ids": effective_join_node_ids,
                        "start_join_mask": effective_start_join_mask,
                        "start_node_id": effective_start_node_id,
                        "op": effective_start_op,
                        "runtime_routes": effective_routes,
                        "expected_event_seq": created.seq,
                    });
                    if effective_start_op_version != "v1" {
                        payload["op_version"] = json!(effective_start_op_version);
                    }
                    if let Some(environment_id) = effective_start_environment_id.as_ref() {
                        payload["op_environment_id"] = json!(environment_id);
                    }
                    add_executor_target(
                        &mut payload,
                        &effective_start_op,
                        effective_start_executor_target.as_deref(),
                    );
                    uow.project_lane_message(NewProjectedLaneMessage {
                        message_id,
                        namespace: "workflow".to_owned(),
                        purpose: "system".to_owned(),
                        inbox_id: "workflow-runtime".to_owned(),
                        conversation_id,
                        recipient_id: runtime_worker_recipient_for_operation(
                            &effective_start_op,
                            effective_start_environment_id.as_deref(),
                            effective_start_executor_target.as_deref(),
                        ),
                        sender_id: "rust-scheduler".to_owned(),
                        msg_type: "workflow.run.execute".to_owned(),
                        status: "pending".to_owned(),
                        created_at: now,
                        available_at: now,
                        run_id: Some(run_id.clone()),
                        step_id: Some(effective_start_node_id),
                        correlation_id: Some(run_id),
                        payload_json: Some(serde_json::to_string(&payload)?),
                        error_json: None,
                    })
                    .await?;
                    Ok(RuntimeAdmissionOutcome::Queued)
                })
            })
            .await;
        match outcome {
            Ok(RuntimeAdmissionOutcome::Queued) => json_effect(StatusCode::ACCEPTED, response),
            Ok(RuntimeAdmissionOutcome::QueueFull) => json_effect(
                StatusCode::TOO_MANY_REQUESTS,
                json!({
                    "admission": queue_full_admission,
                    "reason": "queue_full",
                    "max_queue": max_queue,
                }),
            ),
            Ok(RuntimeAdmissionOutcome::ConversationChanged) => effect_error(
                StatusCode::CONFLICT,
                "KOGWISTAR_CONVERSATION_CHANGED",
                "conversation owner, ACL visibility, or tail changed while preparing the message",
            ),
            Err(error) => effect_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "KOGWISTAR_STORE_ERROR",
                error.to_string(),
            ),
        }
    }

    async fn claim_runtime_work(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
        let input: RuntimeClaimRequest = match serde_json::from_slice(&request.body) {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error.to_string(),
                );
            }
        };
        if input.claimed_by.trim().is_empty() || input.limit == 0 || input.lease_seconds <= 0 {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                "claimed_by, positive limit, and positive lease_seconds are required",
            );
        }
        let recipient_ids = match effective_runtime_claim_recipients(&input.recipient_ids) {
            Ok(values) => values,
            Err(message) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    message,
                );
            }
        };
        match self.claim_runtime_work_items(input, recipient_ids).await {
            Ok(work) => json_effect(StatusCode::OK, json!({"work": work})),
            Err(error) => effect_error(
                StatusCode::CONFLICT,
                "KOGWISTAR_RUNTIME_CLAIM_CONFLICT",
                error,
            ),
        }
    }

    async fn claim_runtime_work_items(
        &self,
        input: RuntimeClaimRequest,
        recipient_ids: Vec<String>,
    ) -> Result<Vec<RuntimeClaimedWork>, String> {
        let claim_owner = format!("{}#{}", input.claimed_by, uuid::Uuid::new_v4());
        let supports_graph_mutation_batch = self
            .store
            .graph_authority_capability()
            .supports_atomic_index_jobs();
        self.store
            .transaction(|uow| {
                Box::pin(async move {
                    let mut claimed = Vec::new();
                    for recipient_id in &recipient_ids {
                        let remaining = input.limit.saturating_sub(claimed.len());
                        if remaining == 0 {
                            break;
                        }
                        claimed.extend(
                            uow.claim_projected_lane_messages_filtered(LaneMessageClaimFilter {
                                namespace: "workflow".to_owned(),
                                inbox_id: "workflow-runtime".to_owned(),
                                claimed_by: claim_owner.clone(),
                                message_ids: None,
                                run_id: input.run_id.clone(),
                                msg_type: None,
                                recipient_id: Some(recipient_id.clone()),
                                limit: remaining,
                                lease_seconds: input.lease_seconds,
                            })
                            .await?,
                        );
                    }
                    let mut values = Vec::new();
                    for lane in claimed {
                        let mut payload: Value =
                            serde_json::from_str(lane.payload_json.as_deref().unwrap_or("{}"))?;
                        let run_id = lane.run_id.clone().unwrap_or_default();
                        if uow
                            .get_server_run(&run_id)
                            .await?
                            .is_some_and(|run| run.cancel_requested)
                        {
                            uow.update_projected_lane_message_status(
                                &lane.message_id,
                                "cancelled",
                                None,
                            )
                            .await?;
                            continue;
                        }
                        let starts_runtime = lane.msg_type == "workflow.run.execute"
                            && payload["runtime_started"].as_bool() != Some(true);
                        let expected_event_seq = if starts_runtime {
                            uow.apply_recorded_runtime_transition(
                                runtime_start_transition(&payload, &run_id),
                                false,
                            )
                            .await?
                            .event_seq
                        } else {
                            payload["expected_event_seq"].as_i64().ok_or_else(|| {
                                kogwistar_store_postgres::PostgresStoreError::RecordedRuntimeConflict(
                                    format!(
                                        "continuation lane message {:?} lacks expected_event_seq",
                                        lane.message_id
                                    ),
                                )
                            })?
                        };
                        payload["expected_event_seq"] = json!(expected_event_seq);
                        if lane.msg_type == "workflow.run.execute" {
                            payload["runtime_started"] = json!(true);
                        }
                        uow.update_projected_lane_message_payload(
                            &lane.message_id,
                            serde_json::to_string(&payload)?,
                        )
                        .await?;
                        let mut work = runtime_claimed_work(
                            lane,
                            &claim_owner,
                            run_id,
                            payload,
                            expected_event_seq,
                        );
                        if supports_graph_mutation_batch {
                            work.worker_effect_capabilities
                                .push("graph_mutation_batch_v1".to_owned());
                        }
                        values.push(work);
                    }
                    Ok(values)
                })
            })
            .await
            .map_err(|error| error.to_string())
    }

    async fn apply_runtime_result(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
        let input: RuntimeWorkerResultRequest = match serde_json::from_slice(&request.body) {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error.to_string(),
                );
            }
        };
        self.apply_runtime_result_input(input, Some(request), None, "python-worker")
            .await
    }

    async fn apply_runtime_result_input(
        &self,
        input: RuntimeWorkerResultRequest,
        request: Option<&ApiEffectRequest>,
        inline_principal: Option<Value>,
        executor_target: &str,
    ) -> ApiEffectResponse {
        if input.transition.is_some() == input.effect.is_some() {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                "exactly one of transition or effect is required",
            );
        }
        if let Some(effect) = input.effect.as_ref()
            && let Err(message) =
                validate_python_worker_submission_trace_binding(
                    &effect.trace_events,
                    &input.handoff.claimed_by,
                )
        {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_WORKER_TRACE",
                message,
            );
        }
        let effect_batch: Option<Result<(RuntimeGraphMutationBatch<'_>, Option<&str>), String>> = input.effect.as_ref().and_then(|effect| {
            match (
                effect.graph_mutation_batch.as_ref(),
                effect.graph_mutation_batch_v2.as_ref(),
            ) {
                (Some(_), Some(_)) => Some(Err("worker effect cannot carry both graph batch versions".to_owned())),
                (Some(v1), None) => Some(Ok((RuntimeGraphMutationBatch::V1(v1), None))),
                (None, Some(v2)) => Some(Ok((RuntimeGraphMutationBatch::V2(v2), Some(v2.default_acl.truth_graph.as_str())))),
                (None, None) => None,
            }
        });
        let graph_effects = if let Some(batch_result) = effect_batch {
            let (batch, default_acl_truth_graph) = match batch_result {
                Ok(value) => value,
                Err(error) => {
                    return effect_error(
                        StatusCode::UNPROCESSABLE_ENTITY,
                        "KOGWISTAR_INVALID_REQUEST",
                        error,
                    );
                }
            };
            if !self
                .store
                .graph_authority_capability()
                .supports_atomic_index_jobs()
            {
                return effect_error(
                    StatusCode::NOT_IMPLEMENTED,
                    "KOGWISTAR_GRAPH_AUTHORITY_UNSUPPORTED",
                    "Worker graph effects require atomic graph authority with index jobs",
                );
            }
            let principal = request
                .map(|request| &request.principal)
                .or(inline_principal.as_ref());
            let Some(principal) = principal else {
                return effect_error(
                    StatusCode::FORBIDDEN,
                    "KOGWISTAR_CAPABILITY_FORBIDDEN",
                    "Worker graph effects require an authenticated write_graph principal",
                );
            };
            if !capability_allowed_for_principal(
                principal,
                &self.capabilities,
                "write_graph",
                &["write_graph"],
            )
            {
                return effect_error(
                    StatusCode::FORBIDDEN,
                    "KOGWISTAR_CAPABILITY_FORBIDDEN",
                    "Worker graph effects require write_graph capability",
                );
            }
            if inline_principal.is_some()
                && !principal_strings(principal, "capabilities")
                    .iter()
                    .any(|capability| capability == "write_graph")
            {
                return effect_error(
                    StatusCode::FORBIDDEN,
                    "KOGWISTAR_CAPABILITY_FORBIDDEN",
                    "Inline graph effects exceed the run's admitted capability ceiling",
                );
            }
            let Some(run_owner_id) = authenticated_run_owner_id(principal) else {
                return effect_error(
                    StatusCode::FORBIDDEN,
                    "KOGWISTAR_RUN_OWNER_FORBIDDEN",
                    "Worker graph effects require an authenticated run owner",
                );
            };
            let (mutations, index_jobs) = match batch {
                RuntimeGraphMutationBatch::V1(batch) => {
                    if let Err(error) = batch.validate() {
                        return effect_error(
                            StatusCode::UNPROCESSABLE_ENTITY,
                            "KOGWISTAR_INVALID_REQUEST",
                            error.to_string(),
                        );
                    }
                    let input = batch.clone();
                    let mutations = match graph_mutation_batch_value(input.clone()) {
                        Ok(value) => value,
                        Err(error) => return effect_error(StatusCode::UNPROCESSABLE_ENTITY, "KOGWISTAR_INVALID_REQUEST", error),
                    };
                    if let Err(error) = validate_graph_mutation_envelopes(&input, &mutations) {
                        return effect_error(StatusCode::UNPROCESSABLE_ENTITY, "KOGWISTAR_INVALID_REQUEST", error);
                    }
                    let index_jobs = match graph_index_jobs_value(&input, &mutations) {
                        Ok(value) => value,
                        Err(error) => return effect_error(StatusCode::UNPROCESSABLE_ENTITY, "KOGWISTAR_INVALID_REQUEST", error),
                    };
                    (mutations, index_jobs)
                }
                RuntimeGraphMutationBatch::V2(batch) => match graph_mutation_batch_v2_value((*batch).clone()) {
                    Ok(value) => value,
                    Err(error) => return effect_error(StatusCode::UNPROCESSABLE_ENTITY, "KOGWISTAR_INVALID_REQUEST", error),
                },
            };
            let security_scope = security_scope_value(principal);
            if mutations
                .iter()
                .any(|mutation| !graph_mutation_scope_allowed(&mutation.scope, &security_scope))
            {
                return effect_error(
                    StatusCode::FORBIDDEN,
                    "KOGWISTAR_SCOPE_FORBIDDEN",
                    "Worker graph effect scope is outside the principal storage scope",
                );
            }
            Some((
                mutations,
                index_jobs,
                default_acl_truth_graph.map(str::to_owned),
                run_owner_id.to_owned(),
                principal.clone(),
                inline_principal.is_some(),
                input.effect.as_ref().map(|effect| effect.effect_id.clone()).unwrap_or_default(),
            ))
        } else {
            None
        };
        let commit_capabilities = self.capabilities.clone();
        let expected_executor_target = executor_target.to_owned();
        let outcome = self
            .store
            .transaction(|uow| {
                Box::pin(async move {
                    let lane = uow
                        .get_projected_lane_message(&input.handoff.message_id)
                        .await?
                        .ok_or_else(|| {
                            kogwistar_store_postgres::PostgresStoreError::RecordedRuntimeConflict(
                                "claimed runtime lane does not exist".to_owned(),
                            )
                        })?;
                    validate_runtime_lane_executor(&lane, &expected_executor_target).map_err(|message| {
                        kogwistar_store_postgres::PostgresStoreError::RecordedRuntimeConflict(
                            message.to_owned(),
                        )
                    })?;
                    if let Some((mutations, _, _, run_owner_id, principal, inline, _)) = graph_effects.as_ref() {
                        if !capability_allowed_for_principal(
                            principal,
                            &commit_capabilities,
                            "write_graph.commit",
                            &["write_graph"],
                        ) || (*inline
                            && !principal_strings(principal, "capabilities")
                                .iter()
                                .any(|capability| capability == "write_graph"))
                        {
                            return Ok(Err(RuntimeGraphCommitDenial::Capability));
                        }
                        let commit_scope = security_scope_value(principal);
                        if mutations.iter().any(|mutation| {
                            !graph_mutation_scope_allowed(&mutation.scope, &commit_scope)
                        }) {
                            return Ok(Err(RuntimeGraphCommitDenial::Scope));
                        }
                        let run = uow.get_server_run(&input.handoff.run_id).await?;
                        if !run.is_some_and(|run| {
                            run.user_id.as_deref() == Some(run_owner_id.as_str())
                        }) {
                            return Ok(Err(RuntimeGraphCommitDenial::Owner));
                        }
                    }
                    if let Some((mutations, mut index_jobs, default_acl_truth_graph, _, principal, _, effect_id)) = graph_effects {
                        let mut mutations = mutations;
                        if let Some(truth_graph) = default_acl_truth_graph {
                            let scope = mutations.first().map(|mutation| mutation.scope.clone()).ok_or_else(|| {
                                kogwistar_store_postgres::PostgresStoreError::RecordedRuntimeConflict(
                                    "ACL-aware graph batch has no mutations".to_owned(),
                                )
                            })?;
                            let owner_id = authenticated_run_owner_id(&principal).ok_or_else(|| {
                                kogwistar_store_postgres::PostgresStoreError::RecordedRuntimeConflict(
                                    "ACL-aware graph batch has no authenticated run owner".to_owned(),
                                )
                            })?.to_ascii_lowercase();
                            let security_scope = principal.get("security_scope").and_then(Value::as_str).map(str::trim).filter(|value| !value.is_empty()).map(str::to_ascii_lowercase);
                            let (acl_mutations, acl_jobs) = resolve_default_acl_graph_effects(
                                uow,
                                &scope,
                                &mutations,
                                &truth_graph,
                                &owner_id,
                                security_scope.as_deref(),
                                &effect_id,
                            ).await?;
                            mutations.extend(acl_mutations);
                            index_jobs.extend(acl_jobs);
                        }
                        for mutation in mutations {
                            uow.apply_graph_mutation(mutation).await?;
                        }
                        for index_job in index_jobs {
                            uow.enqueue_index_job(index_job).await?;
                        }
                    }
                    let handoff = input.handoff;
                    let (applied, effect_id, terminal_result, usage, trace_events) =
                        if let Some(transition) = input.transition {
                            let effect_id = transition.transition_id.clone();
                            let terminal_result = transition.result.clone();
                            let applied = uow
                                .apply_claimed_recorded_runtime_transition(
                                    handoff, transition, false,
                                )
                                .await?;
                            (applied, effect_id, terminal_result, None, Vec::new())
                        } else {
                            let effect = input.effect.expect("effect checked above");
                            let effect_id = effect.effect_id.clone();
                            let terminal_result = effect.result.clone();
                            let usage = effect.usage.clone();
                            let trace_events = effect.trace_events.clone();
                            let applied = uow
                                .apply_claimed_recorded_worker_effect(handoff, effect)
                                .await?;
                            (applied, effect_id, terminal_result, usage, trace_events)
                        };
                    let state = applied.reduced.state.clone();
                    let frontier = &state.frontier;
                    let unfinished = !frontier.pending.is_empty()
                        || !frontier.suspended.is_empty()
                        || frontier.join_outstanding.iter().any(|value| *value != 0)
                        || frontier
                            .join_waiters
                            .values()
                            .any(|waiters| !waiters.is_empty());
                    let applied_idempotent = applied.idempotent;
                    let result = if unfinished || state.status.is_terminal() {
                        for (node_id, join_mask, token_id, parent_token_id) in &frontier.pending {
                            let existing_lanes = uow
                                .list_projected_lane_messages(&LaneMessageFilter {
                                    namespace: Some("workflow".to_owned()),
                                    inbox_id: Some("workflow-runtime".to_owned()),
                                    correlation_id: Some(state.run_id.clone()),
                                    ..LaneMessageFilter::default()
                                })
                                .await?;
                            if active_runtime_lane_covers_token(&existing_lanes, token_id) {
                                continue;
                            }
                            let next_step_seq = state.last_step_seq.saturating_add(1);
                            let message_id = format!(
                                "lane|{}",
                                kogwistar_contracts::stable_id(
                                    "runtime.worker.request",
                                    &[
                                        state.run_id.clone(),
                                        token_id.clone(),
                                        node_id.clone(),
                                        next_step_seq.to_string(),
                                    ],
                                )
                            );
                            let now = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .map(|value| value.as_secs() as i64)
                                .unwrap_or(0);
                            if uow.get_projected_lane_message(&message_id).await?.is_some() {
                                continue;
                            }
                            let op = state
                                .state
                                .get("_rt_node_ops")
                                .and_then(Value::as_object)
                                .and_then(|ops| ops.get(node_id))
                                .cloned()
                                .unwrap_or_else(|| Value::String("noop".to_owned()));
                            let payload = RuntimeStepExecutePayload::from_recorded_state(
                                &state,
                                RuntimeStepExecuteRequest {
                                    node_id: node_id.clone(),
                                    op,
                                    join_mask: *join_mask,
                                    token_id: token_id.clone(),
                                    parent_token_id: parent_token_id.clone(),
                                    step_seq: next_step_seq,
                                    expected_event_seq: applied.event_seq,
                                    resume_effect: None,
                                },
                            );
                            uow.project_lane_message(NewProjectedLaneMessage {
                                message_id,
                                namespace: "workflow".to_owned(),
                                purpose: "system".to_owned(),
                                inbox_id: "workflow-runtime".to_owned(),
                                conversation_id: state.conversation_id.clone(),
                                recipient_id: runtime_worker_recipient_for_state(&state, node_id),
                                sender_id: "rust-scheduler".to_owned(),
                                msg_type: "workflow.step.execute".to_owned(),
                                status: "pending".to_owned(),
                                created_at: now,
                                available_at: now,
                                run_id: Some(state.run_id.clone()),
                                step_id: Some(node_id.clone()),
                                correlation_id: Some(state.run_id.clone()),
                                payload_json: Some(serde_json::to_string(&payload)?),
                                error_json: None,
                            })
                            .await?;
                        }
                        applied
                    } else {
                        uow.apply_recorded_runtime_transition(
                            RecordedRuntimeTransition {
                                contract_version: RECORDED_RUNTIME_CONTRACT_VERSION,
                                transition_id: format!("complete-{effect_id}"),
                                expected_event_seq: applied.event_seq,
                                kind: RecordedTransitionKind::Complete,
                                run_id: state.run_id,
                                workflow_id: state.workflow_id,
                                conversation_id: state.conversation_id,
                                user_id: None,
                                user_turn_node_id: None,
                                step_seq: state.last_step_seq,
                                node_id: state.last_node_id,
                                token_id: state.last_token_id,
                                parent_token_id: state.last_parent_token_id,
                                initial_state: None,
                                state_update: Vec::new(),
                                update: None,
                                state_schema: serde_json::Map::new(),
                                frontier: Some(RuntimeFrontier::default()),
                                result: terminal_result,
                                wait_reason: None,
                                resume_payload: None,
                                errors: Vec::new(),
                            },
                            false,
                        )
                        .await?
                    };
                    if !applied_idempotent {
                        if let Some(usage) = usage {
                            uow.append_server_run_event(
                                &result.reduced.state.run_id,
                                "workflow.usage.v1",
                                serde_json::to_string(&json!({
                                    "run_id": result.reduced.state.run_id,
                                    "step_id": result.reduced.state.last_node_id,
                                    "token_id": result.reduced.state.last_token_id,
                                    "effect_id": effect_id,
                                    "usage": usage,
                                }))?,
                            )
                            .await?;
                        }
                        for (trace_index, trace) in trace_events.into_iter().enumerate() {
                            uow.append_server_run_event(
                                &result.reduced.state.run_id,
                                "workflow.trace.v1",
                                serde_json::to_string(&json!({
                                    "run_id": result.reduced.state.run_id,
                                    "step_id": result.reduced.state.last_node_id,
                                    "token_id": result.reduced.state.last_token_id,
                                    "effect_id": effect_id,
                                    "trace_index": trace_index,
                                    "trace": trace,
                                }))?,
                            )
                            .await?;
                        }
                    }
                    Ok(Ok(result))
                })
            })
            .await;
        match outcome {
            Ok(Err(RuntimeGraphCommitDenial::Owner)) => effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_RUN_OWNER_FORBIDDEN",
                "Authenticated principal does not own the durable run",
            ),
            Ok(Err(RuntimeGraphCommitDenial::Capability)) => effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "write_graph capability was revoked before graph commit",
            ),
            Ok(Err(RuntimeGraphCommitDenial::Scope)) => effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_SCOPE_FORBIDDEN",
                "Worker graph effect scope is outside the principal storage scope at commit",
            ),
            Ok(Ok(result)) => json_effect(
                StatusCode::OK,
                serde_json::to_value(result).expect("recorded runtime result serializes"),
            ),
            Err(error) => effect_error(
                StatusCode::CONFLICT,
                "KOGWISTAR_RUNTIME_RESULT_CONFLICT",
                error.to_string(),
            ),
        }
    }

    async fn resume_runtime_run(
        &self,
        run_id: &str,
        request: &ApiEffectRequest,
    ) -> ApiEffectResponse {
        let input: RuntimeResumeRequest = match serde_json::from_slice(&request.body) {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error.to_string(),
                );
            }
        };
        match self
            .store
            .resume_recorded_runtime_token(
                run_id,
                &input.workflow_id,
                &input.conversation_id,
                &input.suspended_node_id,
                &input.suspended_token_id,
                Some(input.client_result),
            )
            .await
        {
            Ok(result) => json_effect(
                StatusCode::OK,
                serde_json::to_value(result).expect("recorded runtime result serializes"),
            ),
            Err(error) => effect_error(
                StatusCode::CONFLICT,
                "KOGWISTAR_RUNTIME_RESUME_CONFLICT",
                error.to_string(),
            ),
        }
    }

    async fn apply_internal_graph_mutations(
        &self,
        request: &ApiEffectRequest,
    ) -> ApiEffectResponse {
        if !self
            .store
            .graph_authority_capability()
            .supports_atomic_index_jobs()
        {
            return effect_error(
                StatusCode::NOT_IMPLEMENTED,
                "KOGWISTAR_GRAPH_AUTHORITY_UNSUPPORTED",
                "This service does not provide atomic graph authority with index jobs",
            );
        }
        if !capability_allowed(request, &self.capabilities, "write_graph", &["write_graph"]) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Graph mutation requires write_graph capability",
            );
        }
        let mut input: GraphMutationBatchInput = match serde_json::from_slice(&request.body) {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error.to_string(),
                );
            }
        };
        if let Err(error) = input.validate() {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                error.to_string(),
            );
        }
        let mutations = match graph_mutation_batch_value(input.clone()) {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error,
                );
            }
        };
        if let Err(error) = validate_graph_mutation_envelopes(&input, &mutations) {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                error,
            );
        }
        let security_scope = security_scope_value(&request.principal);
        if mutations
            .iter()
            .any(|mutation| !graph_mutation_scope_allowed(&mutation.scope, &security_scope))
        {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_SCOPE_FORBIDDEN",
                "Graph mutation scope is outside the principal storage scope",
            );
        }
        if let Err(error) = self.embed_missing_graph_vectors(&mut input).await {
            return effect_error(
                StatusCode::BAD_GATEWAY,
                "KOGWISTAR_EMBEDDING_FAILED",
                error,
            );
        }
        let mutations = match graph_mutation_batch_value(input.clone()) {
            Ok(value) => value,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error,
                );
            }
        };
        if let Err(error) = validate_graph_mutation_envelopes(&input, &mutations) {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                error,
            );
        }
        let applied = if input.index_jobs.is_empty() {
            self.store.apply_graph_mutations_atomic(mutations).await
        } else {
            let index_jobs = match graph_index_jobs_value(&input, &mutations) {
                Ok(value) => value,
                Err(error) => {
                    return effect_error(
                        StatusCode::UNPROCESSABLE_ENTITY,
                        "KOGWISTAR_INVALID_REQUEST",
                        error,
                    );
                }
            };
            self.store
                .apply_graph_mutations_with_index_jobs_atomic(GraphMutationBatch {
                    mutations,
                    index_jobs,
                })
                .await
        };
        match applied {
            Ok(results) => json_effect(
                StatusCode::OK,
                rust_native_attested(json!({
                    "contract_version": 1,
                    "atomic": true,
                    "results": results.into_iter().map(|result| json!({
                        "event": {
                            "namespace": result.event.namespace,
                            "seq": result.event.seq,
                            "event_id": result.event.event_id,
                            "entity_kind": result.event.entity_kind,
                            "entity_id": result.event.entity_id,
                            "op": result.event.op,
                            "payload": result.event.payload,
                        },
                        "inserted": result.inserted,
                        "mutated": result.mutated,
                    })).collect::<Vec<_>>(),
                })),
            ),
            Err(error) => effect_error(
                StatusCode::CONFLICT,
                "KOGWISTAR_GRAPH_MUTATION_CONFLICT",
                error.to_string(),
            ),
        }
    }

    async fn embed_missing_graph_vectors(
        &self,
        input: &mut GraphMutationBatchInput,
    ) -> Result<(), String> {
        // A batch with index intents intentionally remains Stage 1 and is
        // embedded by the durable worker, never synchronously promoted here.
        if !input.index_jobs.is_empty() {
            return Ok(());
        }
        let Some(provider) = self.embedding_provider.as_ref() else {
            return Ok(());
        };
        let profile = provider.profile();
        profile.validate().map_err(|error| error.to_string())?;

        let mut positions = Vec::new();
        let mut texts = Vec::new();
        for (position, mutation) in input.mutations.iter_mut().enumerate() {
            if !matches!(mutation.op.as_str(), "ADD" | "REPLACE")
                || mutation.record.embedding.is_some()
                || mutation.record.metadata.get("entity_type").and_then(Value::as_str)
                    == Some("acl_record")
            {
                continue;
            }
            if mutation.embedding_dim != 0 && mutation.embedding_dim != profile.dimension {
                return Err("embedding profile dimension conflicts with graph mutation".to_owned());
            }
            let text = mutation
                .record
                .document
                .as_ref()
                .ok_or_else(|| "missing graph document for embedding generation".to_owned())?;
            positions.push(position);
            texts.push(text.clone());
        }
        if texts.is_empty() {
            return Ok(());
        }

        let vectors = provider
            .embed(&texts)
            .await
            .map_err(|_| "embedding provider failed".to_owned())?;
        if vectors.len() != positions.len() {
            return Err("embedding provider returned an unexpected vector count".to_owned());
        }
        for (position, vector) in positions.into_iter().zip(vectors) {
            if vector.len() != profile.dimension || vector.iter().any(|value| !value.is_finite()) {
                return Err("embedding provider returned an invalid vector".to_owned());
            }
            let vector = vector.into_iter().map(|value| value as f32).collect::<Vec<_>>();
            if vector.iter().any(|value| !value.is_finite()) {
                return Err("embedding provider vector exceeds graph storage range".to_owned());
            }
            input.mutations[position].record.embedding = Some(vector);
            input.mutations[position].embedding_dim = profile.dimension;
        }
        input.validate().map_err(|error| error.to_string())
    }

    async fn document_projection_effect(&self, request: &ApiEffectRequest) -> ApiEffectResponse {
        let write = match document_projection_write(request, &self.capabilities) {
            Ok(write) => write,
            Err(response) => return response,
        };
        if !self
            .document_authority_capability()
            .supports_projection_upsert()
        {
            return effect_error(
                StatusCode::NOT_IMPLEMENTED,
                "KOGWISTAR_DOCUMENT_AUTHORITY_UNSUPPORTED",
                "The selected Rust store does not own document projections",
            );
        }
        match self.store.upsert_document_projection(write).await {
            Ok(()) => json_effect(StatusCode::OK, json!({"ok": true})),
            Err(error) => effect_error(
                StatusCode::CONFLICT,
                "KOGWISTAR_DOCUMENT_PROJECTION_FAILED",
                error.to_string(),
            ),
        }
    }

    async fn execute_async(&self, request: ApiEffectRequest) -> ApiEffectResponse {
        if is_pending_server_cutover_route(&request.method, &request.path_and_query) {
            return effect_error(
                StatusCode::NOT_IMPLEMENTED,
                "KOGWISTAR_ROUTE_DEFERRED",
                "This route remains deferred to the Python application service",
            );
        }
        if request.method == "GET"
            && request.path_and_query.split('?').next() == Some("/api/workflow/tools/audit")
        {
            return self.tool_audit_effect(&request).await;
        }
        if request.method == "GET" && is_conversation_list_route(&request.path_and_query) {
            return self.conversation_list_effect(&request).await;
        }
        if request.method == "GET" && conversation_transcript_id(&request.path_and_query).is_some()
        {
            return self.conversation_transcript_effect(&request).await;
        }
        if request.method == "GET" && conversation_detail_id(&request.path_and_query).is_some() {
            return self.conversation_detail_effect(&request).await;
        }
        if request.method == "GET"
            && latest_snapshot_conversation_id(&request.path_and_query).is_some()
        {
            return self.latest_snapshot_effect(&request).await;
        }
        if request.method == "POST" && request.path_and_query == INTERNAL_DOCUMENT_PROJECTION_ROUTE
        {
            return self.document_projection_effect(&request).await;
        }
        if let Some(response) = visualization_shell_effect(&request) {
            return response;
        }
        if let Some(response) = document_graph_validation_effect(&request) {
            return response;
        }
        if let Some(response) = capability_effect(&request, &self.capabilities) {
            return response;
        }
        if let Some(response) = syscall_read_effect(&request, &self.capabilities) {
            return response;
        }
        if request.method == "POST"
            && let Some(op) = syscall_operation(&request.path_and_query)
        {
            return self.syscall_effect(&request, &op).await;
        }
        if let Some(response) = designer_capabilities_effect(&request, &self.capabilities) {
            return response;
        }
        if let Some((workflow_id, route)) = workflow_design_route(&request.path_and_query) {
            return self
                .workflow_design_effect(&request, &workflow_id, route)
                .await;
        }
        if let Some(response) = self.service_read(&request).await {
            return response;
        }
        if request.method == "POST"
            && request.path_and_query.split('?').next() == Some("/api/workflow/services")
        {
            return self.declare_service(&request).await;
        }
        if request.method == "POST"
            && let Some(service_id) = service_trigger_route(&request.path_and_query)
        {
            return self.trigger_service(&request, &service_id).await;
        }
        if request.method == "POST"
            && let Some(service_id) = service_repair_route(&request.path_and_query)
        {
            return self
                .repair_service_projections(&request, service_id.as_deref())
                .await;
        }
        if request.method == "POST"
            && let Some((service_id, enabled)) = service_enabled_route(&request.path_and_query)
        {
            return self
                .set_service_enabled(&request, &service_id, enabled)
                .await;
        }
        if request.method == "POST"
            && let Some((service_id, action)) = request
                .path_and_query
                .split('?')
                .next()
                .and_then(|path| path.strip_prefix("/api/workflow/services/"))
                .and_then(|rest| rest.split_once('/'))
            && action == "heartbeat"
        {
            return self.record_service_heartbeat(&request, service_id).await;
        }
        if request.method == "POST"
            && let Some(run_id) = dead_letter_replay_run_id(&request.path_and_query)
        {
            return self.replay_dead_letter(&request, &run_id).await;
        }
        if request.method == "POST"
            && request.path_and_query.split('?').next()
                == Some("/api/workflow/messages/repair-orphans")
        {
            return self.repair_orphaned_messages(&request).await;
        }
        if request.method == "POST" && request.path_and_query == INTERNAL_RUNTIME_CLAIM_ROUTE {
            return self.claim_runtime_work(&request).await;
        }
        if request.method == "POST" && request.path_and_query == INTERNAL_RUNTIME_RESULTS_ROUTE {
            return self.apply_runtime_result(&request).await;
        }
        if request.method == "POST"
            && request.path_and_query == INTERNAL_RUNTIME_NATIVE_EXECUTE_ROUTE
        {
            return self.execute_native_runtime_step(&request).await;
        }
        if request.method == "POST"
            && request.path_and_query == INTERNAL_CONVERSATION_TRANSCRIPT_ROUTE
        {
            return self.internal_conversation_transcript_effect(&request).await;
        }
        if request.method == "POST"
            && request.path_and_query == INTERNAL_CONVERSATION_CONTEXT_SNAPSHOT_ROUTE
        {
            return self.internal_context_snapshot_effect(&request).await;
        }
        if request.method == "POST" && request.path_and_query == INTERNAL_GRAPH_MUTATIONS_ROUTE {
            return self.apply_internal_graph_mutations(&request).await;
        }
        if request.method == "POST" && request.path_and_query == INTERNAL_GRAPH_NEIGHBORS_ROUTE {
            return self.graph_neighbors_effect(&request).await;
        }
        if request.method == "POST" && request.path_and_query == INTERNAL_GRAPH_VECTOR_QUERY_ROUTE {
            return self.graph_vector_query_effect(&request).await;
        }
        if request.method == "POST" && request.path_and_query == INTERNAL_GRAPH_FIND_EDGES_ROUTE {
            return self.graph_find_edges_effect(&request).await;
        }
        if request.method == "POST" && request.path_and_query == INTERNAL_GRAPH_SHORTEST_PATH_ROUTE {
            return self.graph_shortest_path_effect(&request).await;
        }
        if request.method == "POST" && request.path_and_query == INTERNAL_GRAPH_K_HOP_ROUTE {
            return self.graph_k_hop_effect(&request).await;
        }
        if request.method == "POST"
            && request.path_and_query.split('?').next() == Some("/api/workflow/runs")
        {
            return self.submit_run(&request).await;
        }
        if request.method == "GET"
            && let Some(path) = operational_path(&request.path_and_query)
        {
            let limit =
                query_integer(&request.path_and_query, "limit", 200).clamp(1, 10_000) as usize;
            let runs = match self.store.list_server_runs(None, None, None, 10_000).await {
                Ok(runs) => runs,
                Err(error) => {
                    return effect_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "KOGWISTAR_STORE_ERROR",
                        error.to_string(),
                    );
                }
            };
            let lanes = match self
                .store
                .list_projected_lane_messages(LaneMessageFilter {
                    namespace: Some("workflow".to_owned()),
                    limit: 10_000,
                    ..LaneMessageFilter::default()
                })
                .await
            {
                Ok(lanes) => lanes,
                Err(error) => {
                    return effect_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "KOGWISTAR_STORE_ERROR",
                        error.to_string(),
                    );
                }
            };
            let mut events = Vec::new();
            for run in &runs {
                match self
                    .store
                    .list_server_run_events(&run.run_id, 0, limit)
                    .await
                {
                    Ok(mut run_events) => events.append(&mut run_events),
                    Err(error) => {
                        return effect_error(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            "KOGWISTAR_STORE_ERROR",
                            error.to_string(),
                        );
                    }
                }
            }
            return operational_effect(&request, path, runs, lanes, events);
        }
        let Some((run_id, tail)) = runtime_run_route(&request.path_and_query) else {
            return unavailable_effect(request);
        };
        let run = match self.store.get_server_run(&run_id).await {
            Ok(Some(run)) => run,
            Ok(None) => {
                return effect_error(
                    StatusCode::NOT_FOUND,
                    "KOGWISTAR_RUN_NOT_FOUND",
                    format!("Unknown run_id: {run_id}"),
                );
            }
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error.to_string(),
                );
            }
        };
        let tail = tail.iter().map(String::as_str).collect::<Vec<_>>();
        if request.method == "GET" && is_runtime_inspection_tail(&tail) {
            return match self.store.list_server_run_events(&run_id, 0, 200_000).await {
                Ok(events) => runtime_inspection_effect(&request, &run_id, &tail, events),
                Err(error) => effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error.to_string(),
                ),
            };
        }
        match (request.method.as_str(), tail.as_slice()) {
            ("GET", []) => json_effect(StatusCode::OK, server_run_value(run)),
            ("GET", ["events", "poll"]) => {
                let after_seq = query_integer(&request.path_and_query, "after_seq", 0);
                let limit = query_integer(&request.path_and_query, "limit", 500).clamp(1, 10_000);
                match self
                    .store
                    .list_server_run_events(&run_id, after_seq, limit as usize)
                    .await
                {
                    Ok(events) => json_effect(
                        StatusCode::OK,
                        json!({
                            "run_id": run_id,
                            "events": events.into_iter().map(server_run_event_value).collect::<Vec<_>>(),
                        }),
                    ),
                    Err(error) => effect_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "KOGWISTAR_STORE_ERROR",
                        error.to_string(),
                    ),
                }
            }
            ("GET", ["events"]) => {
                let after_seq = query_integer(&request.path_and_query, "after_seq", 0);
                match self
                    .store
                    .list_server_run_events(&run_id, after_seq, 10_000)
                    .await
                {
                    Ok(events) => {
                        let mut body = String::new();
                        for event in events {
                            let payload = serde_json::from_str::<Value>(&event.payload_json)
                                .unwrap_or_else(|_| json!({}));
                            let mut event_payload = json!({
                                "run_id": event.run_id,
                                "event_type": event.event_type,
                                "created_at_ms": event.created_at_ms,
                            });
                            if let (Some(target), Some(source)) =
                                (event_payload.as_object_mut(), payload.as_object())
                            {
                                target.extend(source.clone());
                            }
                            let event_type =
                                event_payload["event_type"].as_str().unwrap_or("run.event");
                            body.push_str(&sse_frame(
                                event_type,
                                &event_payload,
                                Some(&event.seq.to_string()),
                            ));
                        }
                        ApiEffectResponse {
                            status: StatusCode::OK.as_u16(),
                            content_type: "text/event-stream; charset=utf-8".to_owned(),
                            body: body.into_bytes(),
                        }
                    }
                    Err(error) => effect_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "KOGWISTAR_STORE_ERROR",
                        error.to_string(),
                    ),
                }
            }
            ("GET", ["resume-contract"]) => match self
                .store
                .read_recorded_runtime_state(&run_id, &run.workflow_id, &run.conversation_id)
                .await
            {
                Ok(Some(state)) if !state.frontier.suspended.is_empty() => {
                    json_effect(StatusCode::OK, resume_contract_value(state))
                }
                Ok(Some(_)) => effect_error(
                    StatusCode::CONFLICT,
                    "KOGWISTAR_RUNTIME_NOT_SUSPENDED",
                    "run has no suspended token",
                ),
                Ok(None) => effect_error(
                    StatusCode::NOT_FOUND,
                    "KOGWISTAR_RUNTIME_STATE_NOT_FOUND",
                    format!("No recorded runtime state for {run_id}"),
                ),
                Err(error) => effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error.to_string(),
                ),
            },
            ("POST", ["resume"]) => self.resume_runtime_run(&run_id, &request).await,
            ("POST", ["cancel"]) => match self.store.request_server_run_cancel(&run_id).await {
                Ok(()) => match self.store.get_server_run(&run_id).await {
                    Ok(Some(updated)) => {
                        json_effect(StatusCode::ACCEPTED, server_run_value(updated))
                    }
                    Ok(None) => effect_error(
                        StatusCode::NOT_FOUND,
                        "KOGWISTAR_RUN_NOT_FOUND",
                        format!("Unknown run_id: {run_id}"),
                    ),
                    Err(error) => effect_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "KOGWISTAR_STORE_ERROR",
                        error.to_string(),
                    ),
                },
                Err(error) => effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error.to_string(),
                ),
            },
            _ => unavailable_effect(request),
        }
    }
}
