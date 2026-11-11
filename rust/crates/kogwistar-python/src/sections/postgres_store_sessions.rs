// PostgreSQL session lifecycle and managed-store cache. Python parity: kogwistar.engine_core.engine_postgres transaction facade.
fn postgres_store_error_code(error: &PostgresStoreError) -> &'static str {
    match error {
        PostgresStoreError::EventIdNamespaceCollision { .. } => STORE_EVENT_ID_NAMESPACE_COLLISION,
        PostgresStoreError::Store(kogwistar_store::StoreError::CursorOutOfRange { .. }) => {
            STORE_CURSOR_OUT_OF_RANGE
        }
        PostgresStoreError::Store(kogwistar_store::StoreError::CursorRegresses { .. }) => {
            STORE_CURSOR_REGRESSES
        }
        PostgresStoreError::Store(
            kogwistar_store::StoreError::InvalidIdentifier { .. }
            | kogwistar_store::StoreError::InvalidRecoveryBatchLimit
            | kogwistar_store::StoreError::InvalidEntityEventPayload { .. }
            | kogwistar_store::StoreError::UnsupportedEntityEventOperation { .. },
        ) => STORE_INVALID_ENTITY_EVENT,
        PostgresStoreError::TransactionAborted(message)
            if message == "unsupported PostgreSQL batch operation" =>
        {
            STORE_OPERATION_INVALID
        }
        PostgresStoreError::TransactionAborted(_) => STORE_TRANSACTION_ABORTED,
        // Backend failures deliberately remain persistence failures: they are
        // never idempotency/collision outcomes.
        PostgresStoreError::InvalidSchema { .. }
        | PostgresStoreError::EmptyNamespace
        | PostgresStoreError::Backend(_)
        | PostgresStoreError::InvalidPayload(_)
        | PostgresStoreError::RecordedRuntime(_)
        | PostgresStoreError::RecordedRuntimeConflict(_)
        | PostgresStoreError::Store(_) => STORE_PERSISTENCE_FAILED,
    }
}

fn postgres_runtime() -> Result<TokioRuntime, (&'static str, String)> {
    TokioRuntimeBuilder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|error| {
            (
                STORE_PERSISTENCE_FAILED,
                format!("cannot start Tokio runtime: {error}"),
            )
        })
}

const POSTGRES_EXTERNAL_ROLLBACK: &str = "external PostgreSQL transaction rollback requested";

enum PostgresSessionCommand {
    Operation {
        operation: Box<PostgresStoreOperation>,
        response: mpsc::Sender<Result<Value, PostgresStoreError>>,
    },
    Finish {
        commit: bool,
    },
}

struct PostgresSession {
    dsn: String,
    schema: String,
    sender: mpsc::Sender<PostgresSessionCommand>,
    worker: Option<JoinHandle<Result<(), PostgresStoreError>>>,
    finishing: bool,
}

fn postgres_sessions() -> &'static Mutex<BTreeMap<String, PostgresSession>> {
    static SESSIONS: OnceLock<Mutex<BTreeMap<String, PostgresSession>>> = OnceLock::new();
    SESSIONS.get_or_init(|| Mutex::new(BTreeMap::new()))
}

fn postgres_session_error(message: impl Into<String>) -> PostgresStoreError {
    PostgresStoreError::TransactionAborted(message.into())
}

const MAX_CACHED_POSTGRES_STORES: usize = 64;

enum CachedPostgresCommand {
    Operation {
        transaction_id: Option<String>,
        operation: Box<PostgresStoreOperation>,
        response: mpsc::Sender<Result<Value, PostgresStoreError>>,
    },
    Begin {
        transaction_id: String,
        started: mpsc::Sender<Result<(), PostgresStoreError>>,
    },
    Finish {
        transaction_id: String,
        commit: bool,
        response: mpsc::Sender<Result<(), PostgresStoreError>>,
    },
    Close {
        response: mpsc::Sender<Result<(), PostgresStoreError>>,
    },
}

type CachedPostgresReceiver = Arc<Mutex<mpsc::Receiver<CachedPostgresCommand>>>;

fn receive_cached_postgres_command(
    receiver: &CachedPostgresReceiver,
) -> Result<CachedPostgresCommand, mpsc::RecvError> {
    receiver
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .recv()
}

struct CachedPostgresStore {
    process_id: u32,
    dsn: String,
    schema: String,
    sender: mpsc::Sender<CachedPostgresCommand>,
    worker: Option<JoinHandle<()>>,
}

type SharedCachedPostgresStore = Arc<Mutex<CachedPostgresStore>>;
type ActivePostgresTransactionKey = (u32, String, String);
type ActivePostgresTransactions = BTreeMap<ActivePostgresTransactionKey, String>;
type PostgresTransactionFinishSender = mpsc::Sender<Result<(), PostgresStoreError>>;
type SharedPostgresTransactionFinish = Arc<Mutex<Option<PostgresTransactionFinishSender>>>;

fn postgres_store_cache() -> &'static Mutex<BTreeMap<String, SharedCachedPostgresStore>> {
    static STORES: OnceLock<Mutex<BTreeMap<String, SharedCachedPostgresStore>>> = OnceLock::new();
    STORES.get_or_init(|| Mutex::new(BTreeMap::new()))
}

fn active_postgres_store_transactions() -> &'static Mutex<ActivePostgresTransactions> {
    static ACTIVE: OnceLock<Mutex<ActivePostgresTransactions>> = OnceLock::new();
    ACTIVE.get_or_init(|| Mutex::new(BTreeMap::new()))
}

fn clear_active_postgres_store_transaction(key: &(u32, String, String)) {
    active_postgres_store_transactions()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .remove(key);
}

fn run_cached_postgres_worker(
    dsn: String,
    schema: String,
    receiver: mpsc::Receiver<CachedPostgresCommand>,
    startup: mpsc::SyncSender<Result<(), PostgresStoreError>>,
) {
    let store = match PostgresStore::from_dsn(&dsn, &schema) {
        Ok(store) => store,
        Err(error) => {
            let _ = startup.send(Err(error));
            return;
        }
    };
    let runtime = match postgres_runtime() {
        Ok(runtime) => runtime,
        Err((_, message)) => {
            let _ = startup.send(Err(postgres_session_error(message)));
            return;
        }
    };
    let _ = startup.send(Ok(()));
    let receiver = Arc::new(Mutex::new(receiver));

    loop {
        let command = receive_cached_postgres_command(&receiver);
        let Ok(command) = command else {
            break;
        };
        match command {
            CachedPostgresCommand::Operation {
                transaction_id: None,
                operation,
                response,
            } => {
                let result = runtime.block_on(postgres_store_operation_json(&store, *operation));
                let _ = response.send(result);
            }
            CachedPostgresCommand::Operation {
                transaction_id: Some(_),
                response,
                ..
            } => {
                let _ = response.send(Err(postgres_session_error(
                    "PostgreSQL transaction operation arrived outside its transaction",
                )));
            }
            CachedPostgresCommand::Begin {
                transaction_id,
                started,
            } => {
                let started_flag = Arc::new(AtomicBool::new(false));
                let future_started = started_flag.clone();
                let startup_response = started.clone();
                let transaction_receiver = receiver.clone();
                let finish_response: SharedPostgresTransactionFinish = Arc::new(Mutex::new(None));
                let future_finish_response = finish_response.clone();
                let result = runtime.block_on(store.transaction(move |uow| {
                    Box::pin(async move {
                        future_started.store(true, Ordering::Release);
                        let _ = started.send(Ok(()));
                        loop {
                            let command = receive_cached_postgres_command(&transaction_receiver)
                                .map_err(|_| {
                                    postgres_session_error(
                                        "PostgreSQL store worker disconnected during transaction",
                                    )
                                })?;
                            match command {
                                CachedPostgresCommand::Operation {
                                    transaction_id: Some(request_transaction_id),
                                    operation,
                                    response,
                                } if request_transaction_id == transaction_id => {
                                    let _ = response
                                        .send(postgres_uow_operation_json(uow, *operation).await);
                                }
                                CachedPostgresCommand::Operation { response, .. } => {
                                    let _ = response.send(Err(postgres_session_error(
                                        "PostgreSQL operation does not match active transaction",
                                    )));
                                }
                                CachedPostgresCommand::Finish {
                                    transaction_id: request_transaction_id,
                                    commit,
                                    response,
                                } if request_transaction_id == transaction_id => {
                                    *future_finish_response
                                        .lock()
                                        .unwrap_or_else(|poisoned| poisoned.into_inner()) =
                                        Some(response);
                                    return if commit {
                                        Ok(())
                                    } else {
                                        Err(postgres_session_error(POSTGRES_EXTERNAL_ROLLBACK))
                                    };
                                }
                                CachedPostgresCommand::Begin { started, .. } => {
                                    let _ = started.send(Err(postgres_session_error(
                                        "PostgreSQL transaction is already active",
                                    )));
                                }
                                CachedPostgresCommand::Finish { response, .. } => {
                                    let _ = response.send(Err(postgres_session_error(
                                        "PostgreSQL finish does not match active transaction",
                                    )));
                                }
                                CachedPostgresCommand::Close { response } => {
                                    let _ = response.send(Err(postgres_session_error(
                                        "cannot close PostgreSQL store while transaction is active",
                                    )));
                                }
                            }
                        }
                    })
                }));
                if !started_flag.load(Ordering::Acquire) {
                    let error = result
                        .err()
                        .unwrap_or_else(|| postgres_session_error("transaction did not start"));
                    let _ = startup_response.send(Err(error));
                    continue;
                }
                let finish_result = match result {
                    Ok(()) => Ok(()),
                    Err(PostgresStoreError::TransactionAborted(message))
                        if message == POSTGRES_EXTERNAL_ROLLBACK =>
                    {
                        Ok(())
                    }
                    Err(error) => Err(error),
                };
                if let Some(response) = finish_response
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .take()
                {
                    let _ = response.send(finish_result);
                }
            }
            CachedPostgresCommand::Finish { response, .. } => {
                let _ = response.send(Err(postgres_session_error(
                    "PostgreSQL finish requested without an active transaction",
                )));
            }
            CachedPostgresCommand::Close { response } => {
                let _ = response.send(Ok(()));
                break;
            }
        }
    }
}

fn cached_postgres_store(
    dsn: &str,
    schema: &str,
    context_id: &str,
) -> Result<SharedCachedPostgresStore, PostgresStoreError> {
    let process_id = std::process::id();
    let mut stores = postgres_store_cache()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    // Forked children inherit the cache map but not its worker threads. Prune
    // all stale handles so abandoned parent contexts cannot consume the cap.
    stores.retain(|_, entry| {
        entry
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .process_id
            == process_id
    });
    if let Some(entry) = stores.get(context_id) {
        let entry_guard = entry
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if entry_guard.process_id == process_id {
            if entry_guard.dsn != dsn || entry_guard.schema != schema {
                return Err(postgres_session_error(
                    "PostgreSQL store context does not match its DSN and schema",
                ));
            }
            return Ok(entry.clone());
        }
        drop(entry_guard);
        unreachable!("stale PostgreSQL store entries are pruned above");
    }
    if stores.len() >= MAX_CACHED_POSTGRES_STORES {
        return Err(postgres_session_error(
            "PostgreSQL native store cache is full; close an unused store before opening another",
        ));
    }
    let (sender, receiver) = mpsc::channel();
    let (startup_sender, startup_receiver) = mpsc::sync_channel(1);
    let worker_dsn = dsn.to_owned();
    let worker_schema = schema.to_owned();
    let worker = std::thread::spawn(move || {
        run_cached_postgres_worker(worker_dsn, worker_schema, receiver, startup_sender)
    });
    startup_receiver
        .recv()
        .map_err(|_| postgres_session_error("PostgreSQL store worker failed to start"))??;
    let entry = Arc::new(Mutex::new(CachedPostgresStore {
        process_id,
        dsn: dsn.to_owned(),
        schema: schema.to_owned(),
        sender,
        worker: Some(worker),
    }));
    stores.insert(context_id.to_owned(), entry.clone());
    Ok(entry)
}

fn active_postgres_store_key(dsn: &str, schema: &str) -> (u32, String, String) {
    (std::process::id(), dsn.to_owned(), schema.to_owned())
}

fn cached_postgres_operation(
    dsn: &str,
    schema: &str,
    context_id: &str,
    transaction_id: Option<String>,
    operation: PostgresStoreOperation,
) -> Result<Value, PostgresStoreError> {
    if matches!(&operation, PostgresStoreOperation::CloseStore) {
        if transaction_id.is_some() {
            return Err(postgres_session_error(
                "PostgreSQL store close cannot run inside a transaction",
            ));
        }
        close_cached_postgres_store(dsn, schema, context_id)?;
        return Ok(Value::Null);
    }
    let entry = cached_postgres_store(dsn, schema, context_id)?;
    let entry = entry
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let key = active_postgres_store_key(dsn, schema);
    match operation {
        PostgresStoreOperation::BeginTransaction => {
            require_no_active_postgres_session(dsn, schema)?;
            let transaction_id = require_postgres_transaction_id(transaction_id)?;
            {
                let mut active = active_postgres_store_transactions()
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                if active.contains_key(&key) {
                    return Err(postgres_session_error(
                        "PostgreSQL external transaction is already active for this store",
                    ));
                }
                active.insert(key.clone(), context_id.to_owned());
            }
            let (response_sender, response_receiver) = mpsc::channel();
            if entry
                .sender
                .send(CachedPostgresCommand::Begin {
                    transaction_id,
                    started: response_sender,
                })
                .is_err()
            {
                active_postgres_store_transactions()
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .remove(&key);
                return Err(postgres_session_error(
                    "PostgreSQL store worker is unavailable",
                ));
            }
            let result = response_receiver
                .recv()
                .map_err(|_| postgres_session_error("PostgreSQL store worker dropped startup"));
            if !matches!(&result, Ok(Ok(()))) {
                clear_active_postgres_store_transaction(&key);
            }
            result??;
            Ok(Value::Null)
        }
        PostgresStoreOperation::CommitTransaction | PostgresStoreOperation::RollbackTransaction => {
            let transaction_id = require_postgres_transaction_id(transaction_id)?;
            {
                let active = active_postgres_store_transactions()
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                if active.get(&key).map(String::as_str) != Some(context_id) {
                    return Err(postgres_session_error(
                        "PostgreSQL transaction belongs to another or stale store context",
                    ));
                }
            }
            let (response_sender, response_receiver) = mpsc::channel();
            let result = entry
                .sender
                .send(CachedPostgresCommand::Finish {
                    transaction_id,
                    commit: matches!(operation, PostgresStoreOperation::CommitTransaction),
                    response: response_sender,
                })
                .map_err(|_| postgres_session_error("PostgreSQL store worker is unavailable"))
                .and_then(|()| {
                    response_receiver.recv().map_err(|_| {
                        postgres_session_error("PostgreSQL store worker dropped finish")
                    })
                })
                .and_then(|result| result);
            clear_active_postgres_store_transaction(&key);
            result.map(|()| Value::Null)
        }
        PostgresStoreOperation::CloseStore => unreachable!("handled above"),
        operation => {
            let active = active_postgres_store_transactions()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            match (active.get(&key), transaction_id.as_deref()) {
                (Some(owner), Some(_)) if owner == context_id => {}
                (Some(_), _) => {
                    return Err(postgres_session_error(
                        "PostgreSQL store has an active transaction",
                    ));
                }
                (None, Some(_)) => {
                    return Err(postgres_session_error("PostgreSQL transaction_id is stale"));
                }
                (None, None) => {}
            }
            drop(active);
            let (response_sender, response_receiver) = mpsc::channel();
            entry
                .sender
                .send(CachedPostgresCommand::Operation {
                    transaction_id,
                    operation: Box::new(operation),
                    response: response_sender,
                })
                .map_err(|_| postgres_session_error("PostgreSQL store worker is unavailable"))?;
            response_receiver.recv().map_err(|_| {
                postgres_session_error("PostgreSQL store worker dropped its response")
            })?
        }
    }
}

fn close_cached_postgres_store(
    dsn: &str,
    schema: &str,
    context_id: &str,
) -> Result<(), PostgresStoreError> {
    let key = active_postgres_store_key(dsn, schema);
    if active_postgres_store_transactions()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .contains_key(&key)
    {
        return Err(postgres_session_error(
            "cannot close PostgreSQL store while a transaction is active",
        ));
    }
    if postgres_sessions()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .values()
        .any(|session| session.dsn == dsn && session.schema == schema)
    {
        return Err(postgres_session_error(
            "cannot close PostgreSQL store while a transaction is active",
        ));
    }
    let entry = postgres_store_cache()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .get(context_id)
        .cloned();
    let Some(entry) = entry else {
        return Ok(());
    };
    let mut entry_guard = entry
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if entry_guard.process_id != std::process::id() {
        drop(entry_guard);
        postgres_store_cache()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(context_id);
        return Ok(());
    }
    if entry_guard.dsn != dsn || entry_guard.schema != schema {
        return Err(postgres_session_error(
            "PostgreSQL store context does not match its DSN and schema",
        ));
    }
    let (response_sender, response_receiver) = mpsc::channel();
    entry_guard
        .sender
        .send(CachedPostgresCommand::Close {
            response: response_sender,
        })
        .map_err(|_| postgres_session_error("PostgreSQL store worker is unavailable"))?;
    response_receiver
        .recv()
        .map_err(|_| postgres_session_error("PostgreSQL store worker dropped close"))??;
    if let Some(worker) = entry_guard.worker.take() {
        worker
            .join()
            .map_err(|_| postgres_session_error("PostgreSQL store worker panicked on close"))?;
    }
    drop(entry_guard);
    postgres_store_cache()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .remove(context_id);
    Ok(())
}

fn require_postgres_transaction_id(
    transaction_id: Option<String>,
) -> Result<String, PostgresStoreError> {
    transaction_id
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            postgres_session_error("PostgreSQL transaction control requires transaction_id")
        })
}

fn begin_postgres_session(
    dsn: String,
    schema: String,
    transaction_id: String,
) -> Result<(), PostgresStoreError> {
    require_no_active_postgres_session(&dsn, &schema)?;
    let mut sessions = postgres_sessions()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if sessions.contains_key(&transaction_id) {
        return Err(postgres_session_error(
            "PostgreSQL transaction_id is already active",
        ));
    }
    if sessions
        .values()
        .any(|session| session.dsn == dsn && session.schema == schema)
    {
        return Err(postgres_session_error(
            "PostgreSQL external transaction is already active for this store",
        ));
    }

    let (sender, receiver) = mpsc::channel();
    let (started_sender, started_receiver) =
        mpsc::sync_channel::<Result<(), PostgresStoreError>>(1);
    let worker_dsn = dsn.clone();
    let worker_schema = schema.clone();
    let worker = std::thread::spawn(move || {
        let store = PostgresStore::from_dsn(&worker_dsn, &worker_schema)?;
        let runtime = postgres_runtime().map_err(|(_, message)| postgres_session_error(message))?;
        let result = runtime.block_on(store.transaction(move |uow| {
            Box::pin(async move {
                let _ = started_sender.send(Ok(()));
                loop {
                    let command = receiver.recv().map_err(|_| {
                        postgres_session_error(
                            "PostgreSQL transaction owner disconnected before commit or rollback",
                        )
                    })?;
                    match command {
                        PostgresSessionCommand::Operation {
                            operation,
                            response,
                        } => {
                            let result = postgres_uow_operation_json(uow, *operation).await;
                            let _ = response.send(result);
                        }
                        PostgresSessionCommand::Finish { commit: true } => return Ok(()),
                        PostgresSessionCommand::Finish { commit: false } => {
                            return Err(postgres_session_error(POSTGRES_EXTERNAL_ROLLBACK));
                        }
                    }
                }
            })
        }));
        match result {
            Err(PostgresStoreError::TransactionAborted(message))
                if message == POSTGRES_EXTERNAL_ROLLBACK =>
            {
                Ok(())
            }
            other => other,
        }
    });
    match started_receiver.recv() {
        Ok(Ok(())) => {
            sessions.insert(
                transaction_id,
                PostgresSession {
                    dsn,
                    schema,
                    sender,
                    worker: Some(worker),
                    finishing: false,
                },
            );
            Ok(())
        }
        Ok(Err(error)) => Err(error),
        Err(_) => match worker.join() {
            Ok(Err(error)) => Err(error),
            Ok(Ok(())) => Err(postgres_session_error(
                "PostgreSQL transaction worker exited before startup",
            )),
            Err(_) => Err(postgres_session_error(
                "PostgreSQL transaction worker panicked during startup",
            )),
        },
    }
}

fn postgres_session_operation(
    transaction_id: &str,
    operation: PostgresStoreOperation,
) -> Result<Value, PostgresStoreError> {
    let sender = {
        let sessions = postgres_sessions()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        sessions
            .get(transaction_id)
            .map(|session| session.sender.clone())
            .ok_or_else(|| postgres_session_error("PostgreSQL transaction_id is stale"))?
    };
    let (response_sender, response_receiver) = mpsc::channel();
    sender
        .send(PostgresSessionCommand::Operation {
            operation: Box::new(operation),
            response: response_sender,
        })
        .map_err(|_| postgres_session_error("PostgreSQL transaction worker is unavailable"))?;
    response_receiver
        .recv()
        .map_err(|_| postgres_session_error("PostgreSQL transaction worker dropped its response"))?
}

fn finish_postgres_session(transaction_id: &str, commit: bool) -> Result<(), PostgresStoreError> {
    let (sender, worker) = {
        let mut sessions = postgres_sessions()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let session = sessions
            .get_mut(transaction_id)
            .ok_or_else(|| postgres_session_error("PostgreSQL transaction_id is stale"))?;
        if session.finishing {
            return Err(postgres_session_error(
                "PostgreSQL transaction is already finishing",
            ));
        }
        session.finishing = true;
        let worker = session.worker.take().ok_or_else(|| {
            postgres_session_error("PostgreSQL transaction worker is unavailable")
        })?;
        (session.sender.clone(), worker)
    };
    let finish_sent = sender
        .send(PostgresSessionCommand::Finish { commit })
        .is_ok();
    let result = match worker.join() {
        Ok(Ok(())) => Ok(()),
        Ok(Err(error)) => Err(error),
        Err(_) => Err(postgres_session_error(
            "PostgreSQL transaction worker panicked",
        )),
    };
    postgres_sessions()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .remove(transaction_id);
    if !finish_sent {
        return Err(postgres_session_error(
            "PostgreSQL transaction worker is unavailable",
        ));
    }
    result
}

fn require_no_active_postgres_session(dsn: &str, schema: &str) -> Result<(), PostgresStoreError> {
    let sessions = postgres_sessions()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if sessions
        .values()
        .any(|session| session.dsn == dsn && session.schema == schema)
    {
        return Err(postgres_session_error(
            "PostgreSQL operation does not own the active transaction",
        ));
    }
    drop(sessions);
    if active_postgres_store_transactions()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .contains_key(&active_postgres_store_key(dsn, schema))
    {
        return Err(postgres_session_error(
            "PostgreSQL operation does not own the active transaction",
        ));
    }
    Ok(())
}

fn require_postgres_fields(value: &Value, allowed: &[&str]) -> Result<(), (&'static str, String)> {
    let object = value.as_object().ok_or_else(|| {
        (
            STORE_INVALID_PAYLOAD,
            "invalid PostgreSQL store payload: operation must be an object".to_owned(),
        )
    })?;
    if let Some(field) = object
        .keys()
        .find(|field| !allowed.contains(&field.as_str()))
    {
        return Err((
            STORE_INVALID_PAYLOAD,
            format!("invalid PostgreSQL store payload: unknown field {field:?}"),
        ));
    }
    Ok(())
}

fn validate_postgres_operation(value: &Value) -> Result<(), (&'static str, String)> {
    let operation = value.as_object().ok_or_else(|| {
        (
            STORE_INVALID_PAYLOAD,
            "invalid PostgreSQL store payload: operation must be an object".to_owned(),
        )
    })?;
    let kind = operation
        .get("kind")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            (
                STORE_INVALID_PAYLOAD,
                "invalid PostgreSQL store payload: operation kind must be a string".to_owned(),
            )
        })?;
    if kind == "enqueue_index_job" {
        require_postgres_fields(
            value,
            &[
                "kind",
                "job_id",
                "namespace",
                "entity_kind",
                "entity_id",
                "index_kind",
                "op",
                "payload_json",
                "max_retries",
            ],
        )?;
        let max_retries = match operation.get("max_retries") {
            Some(value) => value.as_i64().ok_or_else(|| {
                (
                    STORE_INVALID_PAYLOAD,
                    "invalid PostgreSQL store payload: index job max_retries must be an integer"
                        .to_owned(),
                )
            })?,
            None => default_max_retries(),
        };
        if !(0..=i32::MAX as i64).contains(&max_retries) {
            return Err((
                STORE_INVALID_PAYLOAD,
                "invalid PostgreSQL store payload: index job max_retries must fit a non-negative PostgreSQL INTEGER".to_owned(),
            ));
        }
        return Ok(());
    }
    let allowed = match kind {
        "close_store"
        | "begin_transaction"
        | "commit_transaction"
        | "rollback_transaction"
        | "ensure_schema"
        | "graph_authority_capability"
        | "next_global_seq"
        | "current_global_seq" => &["kind"][..],
        "lock_entity_identity" => &["kind", "namespace", "entity_kind", "entity_id"][..],
        "next_user_seq" | "current_user_seq" => &["kind", "user_id"][..],
        "set_user_seq" => &["kind", "user_id", "value"][..],
        "get_index_applied_fingerprint" => &["kind", "namespace", "coalesce_key"][..],
        "set_index_applied_fingerprint" => &[
            "kind",
            "namespace",
            "coalesce_key",
            "applied_fingerprint",
            "last_job_id",
        ][..],
        "create_graph_schema" => &[
            "kind",
            "embedding_dim",
            "nodes_table",
            "edges_table",
            "documents_table",
            "domains_table",
            "node_index_table",
        ][..],
        "node_index_upsert" => &["kind", "table", "records"][..],
        "node_index_get" => &["kind", "table", "ids"][..],
        "node_index_query" => &["kind", "table", "embedding", "limit", "metadata", "metric"][..],
        "alloc_event_seq" => &["kind", "namespace"][..],
        "raw_append" => &[
            "kind",
            "namespace",
            "event_id",
            "entity_kind",
            "entity_id",
            "op",
            "payload_json",
        ][..],
        "raw_restore" => &[
            "kind",
            "namespace",
            "seq",
            "event_id",
            "entity_kind",
            "entity_id",
            "op",
            "payload_json",
            "created_at",
        ][..],
        "graph_mutation" => &[
            "kind",
            "namespace",
            "workspace_id",
            "graph_space",
            "table",
            "entity_kind",
            "event_id",
            "op",
            "record",
            "payload",
            "embedding_dim",
            "model_envelope",
        ][..],
        "graph_metadata_patch_mutation" => &[
            "kind",
            "namespace",
            "workspace_id",
            "graph_space",
            "table",
            "entity_kind",
            "event_id",
            "op",
            "entity_id",
            "document",
            "metadata_patch",
            "payload",
        ][..],
        "graph_delete_mutation" => &[
            "kind",
            "namespace",
            "workspace_id",
            "graph_space",
            "table",
            "entity_kind",
            "event_id",
            "entity_id",
            "payload",
        ][..],
        "upsert_graph_projection" => &[
            "kind",
            "namespace",
            "workspace_id",
            "graph_space",
            "table",
            "record",
            "embedding_dim",
        ][..],
        "patch_graph_projection_metadata" => &[
            "kind",
            "namespace",
            "workspace_id",
            "graph_space",
            "table",
            "entity_id",
            "document",
            "metadata_patch",
            "patch_document_metadata",
        ][..],
        "graph_projection_records" => &[
            "kind",
            "namespace",
            "workspace_id",
            "graph_space",
            "table",
            "ids",
            "metadata",
            "after_id",
            "limit",
            "for_update",
        ][..],
        "graph_projection_vector_query" => &[
            "kind",
            "namespace",
            "workspace_id",
            "graph_space",
            "table",
            "embedding",
            "limit",
            "metadata",
            "metadata_not_equals",
            "metric",
            "embedding_dim",
        ][..],
        "graph_projection_vector_query_batch" => &[
            "kind",
            "namespace",
            "workspace_id",
            "graph_space",
            "table",
            "embeddings",
            "limit",
            "metadata",
            "metadata_not_equals",
            "metric",
            "embedding_dim",
        ][..],
        "exclusive_raw_replay" => &["kind", "namespace", "after_seq", "limit"][..],
        "latest_retained_event_seq" => &["kind", "namespace"][..],
        "prune_entity_events_after" => &["kind", "namespace", "to_seq"][..],
        "cursor_get" => &["kind", "namespace", "consumer"][..],
        "legacy_cursor_set" | "strict_cursor_advance" => {
            &["kind", "namespace", "consumer", "last_seq"][..]
        }
        "recover_entity_projection" => &[
            "kind",
            "namespace",
            "consumer",
            "projection_namespace",
            "projection_key",
            "batch_limit",
            "abort_after_projection",
        ][..],
        "rebuild_entity_projection" => &[
            "kind",
            "namespace",
            "consumer",
            "projection_namespace",
            "projection_key",
            "abort_after_projection",
        ][..],
        "get_named_projection"
        | "clear_named_projection"
        | "get_stage1_node_projection"
        | "clear_stage1_node_projection" => &["kind", "namespace", "key"][..],
        "clear_stage1_node_projection_if_current" => &[
            "kind",
            "namespace",
            "key",
            "expected_last_authoritative_seq",
        ][..],
        "list_named_projections"
        | "clear_projection_namespace"
        | "list_stage1_node_projections" => &["kind", "namespace"][..],
        "replace_stage1_node_projection" => &[
            "kind",
            "namespace",
            "key",
            "payload",
            "last_authoritative_seq",
            "last_materialized_seq",
            "projection_schema_version",
            "materialization_status",
        ][..],
        "put_workflow_design_snapshot" => &[
            "kind",
            "workflow_id",
            "version",
            "seq",
            "payload_json",
            "schema_version",
        ][..],
        "get_workflow_design_snapshot" => {
            &["kind", "workflow_id", "max_version", "schema_version"][..]
        }
        "clear_workflow_design_snapshots" => &["kind", "workflow_id"][..],
        "put_workflow_design_delta" => &[
            "kind",
            "workflow_id",
            "version",
            "prev_version",
            "target_seq",
            "forward_json",
            "inverse_json",
            "schema_version",
        ][..],
        "get_workflow_design_delta" => &["kind", "workflow_id", "version", "schema_version"][..],
        "clear_workflow_design_deltas" => &["kind", "workflow_id"][..],
        "create_server_run" => &[
            "kind",
            "run_id",
            "conversation_id",
            "workflow_id",
            "user_id",
            "user_turn_node_id",
            "status",
        ][..],
        "get_server_run" | "request_server_run_cancel" => &["kind", "run_id"][..],
        "list_server_runs" => &["kind", "status", "workflow_id", "conversation_id", "limit"][..],
        "append_server_run_event" => &["kind", "run_id", "event_type", "payload_json"][..],
        "list_server_run_events" => &["kind", "run_id", "after_seq", "limit"][..],
        "update_server_run" => &[
            "kind",
            "run_id",
            "status",
            "assistant_turn_node_id",
            "result_json",
            "error_json",
            "started_at_ms",
            "finished_at_ms",
            "cancel_requested",
        ][..],
        "apply_recorded_runtime_transition" => &["kind", "transition", "abort_after_writes"][..],
        "apply_claimed_recorded_runtime_transition" => {
            &["kind", "handoff", "transition", "abort_after_writes"][..]
        }
        "read_recorded_runtime_state" => &["kind", "run_id", "workflow_id", "conversation_id"][..],
        "claim_index_jobs" => &["kind", "limit", "lease_seconds", "namespace"][..],
        "reconcile_join_index_jobs" => &[
            "kind",
            "limit",
            "lease_seconds",
            "namespace",
            "nodes_table",
            "edges_table",
            "node_docs_table",
            "node_refs_table",
            "edge_refs_table",
            "edge_endpoints_table",
        ][..],
        "mark_index_job_done" => &["kind", "job_id", "claim_token"][..],
        "accept_index_job_result" => &[
            "kind",
            "job_id",
            "claim_token",
            "result_json",
            "result_sha256",
        ][..],
        "get_index_job_result" => &["kind", "job_id"][..],
        "mark_index_job_failed" => &["kind", "job_id", "error", "final", "claim_token"][..],
        "bump_retry_and_requeue" => &[
            "kind",
            "job_id",
            "error",
            "next_run_at_seconds",
            "claim_token",
        ][..],
        "renew_index_job_lease" => &["kind", "job_id", "claim_token", "lease_seconds"][..],
        "requeue_index_job_at_tail" => &[
            "kind",
            "job_id",
            "payload_json",
            "delay_seconds",
            "claim_token",
        ][..],
        "list_index_jobs" => &[
            "kind",
            "namespace",
            "status",
            "entity_kind",
            "entity_id",
            "index_kind",
            "limit",
        ][..],
        "project_lane_message" => &[
            "kind",
            "message_id",
            "namespace",
            "purpose",
            "inbox_id",
            "conversation_id",
            "recipient_id",
            "sender_id",
            "msg_type",
            "status",
            "created_at",
            "available_at",
            "run_id",
            "step_id",
            "correlation_id",
            "payload_json",
            "error_json",
        ][..],
        "get_projected_lane_message" => &["kind", "message_id"][..],
        "update_projected_lane_message_status" => {
            &["kind", "message_id", "status", "error_json"][..]
        }
        "update_projected_lane_message_links" => &[
            "kind",
            "message_id",
            "prev_message_id",
            "next_message_id",
            "inbox_tail_message_id",
            "conversation_tail_message_id",
        ][..],
        "list_projected_lane_messages" => &[
            "kind",
            "namespace",
            "purpose",
            "inbox_id",
            "conversation_id",
            "run_id",
            "status",
            "msg_type",
            "sender_id",
            "recipient_id",
            "correlation_id",
            "reply_to_message_id",
            "created_at_gte",
            "created_at_lte",
            "available_at_gte",
            "available_at_lte",
            "limit",
            "newest_first",
        ][..],
        "clear_projected_lane_messages" => &["kind", "namespace"][..],
        "claim_projected_lane_messages" => &[
            "kind",
            "namespace",
            "inbox_id",
            "claimed_by",
            "limit",
            "lease_seconds",
            "message_ids",
            "run_id",
            "msg_type",
            "recipient_id",
        ][..],
        "ack_projected_lane_message" => &["kind", "message_id", "claimed_by"][..],
        "requeue_projected_lane_message" => &[
            "kind",
            "message_id",
            "claimed_by",
            "error_json",
            "delay_seconds",
        ][..],
        "dead_letter_projected_lane_message" => {
            &["kind", "message_id", "claimed_by", "error_json"][..]
        }
        "replace_named_projection" => &[
            "kind",
            "namespace",
            "key",
            "payload",
            "last_authoritative_seq",
            "last_materialized_seq",
            "projection_schema_version",
            "materialization_status",
        ][..],
        "compare_and_swap_named_projection" => &[
            "kind",
            "namespace",
            "key",
            "payload",
            "expected_last_authoritative_seq",
            "expected_last_materialized_seq",
            "last_authoritative_seq",
            "last_materialized_seq",
            "projection_schema_version",
            "materialization_status",
        ][..],
        "compare_and_swap_named_projections" => &["kind", "updates"][..],
        "batch" => {
            require_postgres_fields(value, &["kind", "operations", "abort"])?;
            let operations = operation
                .get("operations")
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    (
                        STORE_INVALID_PAYLOAD,
                        "invalid PostgreSQL store payload: batch operations must be an array"
                            .to_owned(),
                    )
                })?;
            for nested in operations {
                validate_postgres_operation(nested)?;
            }
            return Ok(());
        }
        _ => return Ok(()), // serde supplies stable invalid-operation diagnostics below.
    };
    require_postgres_fields(value, allowed)
}

fn postgres_store_json_impl(payload_json: &str) -> Result<String, (&'static str, String)> {
    let value: Value = serde_json::from_str(payload_json)
        .map_err(|error| (STORE_INVALID_JSON, format!("invalid JSON: {error}")))?;
    require_postgres_fields(
        &value,
        &["dsn", "schema", "context_id", "transaction_id", "operation"],
    )?;
    let operation = value.get("operation").ok_or_else(|| {
        (
            STORE_INVALID_PAYLOAD,
            "invalid PostgreSQL store payload: missing operation".to_owned(),
        )
    })?;
    validate_postgres_operation(operation)?;
    let request: PostgresStoreRequest = serde_json::from_value(value).map_err(|error| {
        (
            STORE_INVALID_PAYLOAD,
            format!("invalid PostgreSQL store payload: {error}"),
        )
    })?;
    let PostgresStoreRequest {
        dsn,
        schema,
        context_id,
        transaction_id,
        operation,
    } = request;
    let result: Result<Value, PostgresStoreError> = match context_id.as_deref() {
        Some(context_id) => {
            cached_postgres_operation(&dsn, &schema, context_id, transaction_id, operation)
        }
        None => (|| match operation {
            PostgresStoreOperation::CloseStore => Err(postgres_session_error(
                "PostgreSQL store close requires context_id",
            )),
            PostgresStoreOperation::BeginTransaction => {
                let transaction_id = require_postgres_transaction_id(transaction_id)?;
                begin_postgres_session(dsn, schema, transaction_id).map(|()| Value::Null)
            }
            PostgresStoreOperation::CommitTransaction => {
                let transaction_id = require_postgres_transaction_id(transaction_id)?;
                finish_postgres_session(&transaction_id, true).map(|()| Value::Null)
            }
            PostgresStoreOperation::RollbackTransaction => {
                let transaction_id = require_postgres_transaction_id(transaction_id)?;
                finish_postgres_session(&transaction_id, false).map(|()| Value::Null)
            }
            operation => {
                if let Some(transaction_id) = transaction_id {
                    postgres_session_operation(&transaction_id, operation)
                } else {
                    require_no_active_postgres_session(&dsn, &schema)?;
                    let store = PostgresStore::from_dsn(&dsn, &schema)?;
                    let runtime = postgres_runtime()
                        .map_err(|(_, message)| postgres_session_error(message))?;
                    runtime.block_on(postgres_store_operation_json(&store, operation))
                }
            }
        })(),
    };
    result
        .and_then(|value| {
            serde_json::to_string(&value).map_err(|error| {
                PostgresStoreError::TransactionAborted(format!("cannot encode result: {error}"))
            })
        })
        .map_err(|error| (postgres_store_error_code(&error), error.to_string()))
}
