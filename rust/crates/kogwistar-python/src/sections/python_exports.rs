// PyO3-exposed graph, runtime, API, and backend entry points. Python parity: kogwistar._rust extension API and its Python facades.
#[pyfunction]
fn stable_id(kind: &str, parts: Vec<String>) -> String {
    contracts::stable_id(kind, &parts).to_string()
}

#[pyfunction]
fn stable_id_json(payload_json: &str) -> PyResult<String> {
    contracts::stable_id_from_json(payload_json)
        .map(|value| value.to_string())
        .map_err(contract_error)
}

#[pyfunction]
fn canonical_json(payload_json: &str) -> PyResult<String> {
    contracts::canonical_json_from_str(payload_json).map_err(contract_error)
}

#[pyfunction]
fn validate_graph_model_family(payload_json: &str) -> PyResult<()> {
    contracts::validate_graph_model_family_from_str(payload_json)
        .map_err(|error| RustContractValueError::new_err(error.to_string()))
}

#[pyfunction]
fn round_trip_raw_graph_model_envelope(payload_json: &str) -> PyResult<String> {
    contracts::round_trip_raw_graph_model_envelope_from_str(payload_json)
        .map_err(|error| RustContractValueError::new_err(error.to_string()))
}

#[pyfunction]
fn graph_model_level_from_root(payload_json: &str) -> PyResult<Option<String>> {
    contracts::python_level_from_root_from_str(payload_json)
        .map_err(|error| RustContractValueError::new_err(error.to_string()))
}

#[pyfunction]
fn decode_persistent_graph_model_v2(payload_json: &str) -> PyResult<String> {
    contracts::decode_persistent_graph_model_v2_from_str(payload_json)
        .map_err(|error| RustContractValueError::new_err(error.to_string()))
}

#[pyfunction]
fn project_graph_model_slice(payload_json: &str) -> PyResult<String> {
    contracts::project_graph_model_slice_from_str(payload_json)
        .map_err(|error| RustContractValueError::new_err(error.to_string()))
}

#[pyfunction]
fn project_graph_model_slices(payload_json: &str) -> PyResult<String> {
    contracts::project_graph_model_slices_from_str(payload_json)
        .map_err(|error| RustContractValueError::new_err(error.to_string()))
}

#[pyfunction]
fn supported_graph_model_classes() -> Vec<String> {
    contracts::supported_graph_model_classes()
        .into_iter()
        .map(str::to_owned)
        .collect()
}

#[pyfunction]
fn graph_model_capability_profiles() -> PyResult<String> {
    serde_json::to_string(&contracts::graph_model_capability_profiles())
        .map_err(|error| RustContractValueError::new_err(error.to_string()))
}

#[pyfunction]
fn evidence_pack_digest_hash(payload_json: &str) -> PyResult<String> {
    contracts::evidence_pack_digest_hash_from_str(payload_json).map_err(contract_error)
}

#[pyfunction]
fn metadata_filter_matches(payload_json: &str) -> PyResult<bool> {
    contracts::metadata_filter_matches_from_str(payload_json).map_err(contract_error)
}

#[pyfunction]
fn normalize_metadata_filter(payload_json: &str) -> PyResult<String> {
    contracts::normalize_metadata_filter_from_str(payload_json).map_err(contract_error)
}

#[pyfunction]
fn short_id_transform(payload_json: &str) -> PyResult<String> {
    contracts::short_id_transform_from_str(payload_json).map_err(contract_error)
}

#[pyfunction]
fn apply_state_update(payload_json: &str) -> PyResult<String> {
    contracts::apply_state_update_from_str(payload_json).map_err(contract_error)
}

#[pyfunction]
fn canonical_entity_event(payload_json: &str) -> PyResult<String> {
    contracts::canonical_entity_event_from_str(payload_json).map_err(contract_error)
}

#[pyfunction]
fn replay_entity_events(payload_json: &str) -> PyResult<String> {
    contracts::replay_entity_events_from_str(payload_json).map_err(contract_error)
}

#[pyfunction]
fn workflow_may_reach_join(payload_json: &str) -> PyResult<String> {
    contracts::workflow_may_reach_join_from_str(payload_json).map_err(contract_error)
}

#[pyfunction]
fn workflow_terminal_reachable(payload_json: &str) -> PyResult<bool> {
    contracts::workflow_terminal_reachable_from_str(payload_json).map_err(contract_error)
}

#[pyfunction]
fn runtime_select_route(payload_json: &str) -> PyResult<String> {
    kogwistar_runtime::select_runtime_route_from_str(payload_json)
        .map_err(|error| RustContractValueError::new_err(error.to_string()))
}

#[pyfunction]
fn runtime_plan_successors(payload_json: &str) -> PyResult<String> {
    kogwistar_runtime::plan_runtime_successors_from_str(payload_json)
        .map_err(|error| RustContractValueError::new_err(error.to_string()))
}

#[pyfunction]
fn runtime_apply_join_arrival(payload_json: &str) -> PyResult<String> {
    kogwistar_runtime::apply_runtime_join_arrival_from_str(payload_json)
        .map_err(|error| RustContractValueError::new_err(error.to_string()))
}

#[pyfunction]
fn runtime_decide_retry(payload_json: &str) -> PyResult<String> {
    kogwistar_runtime::decide_runtime_retry_from_str(payload_json)
        .map_err(|error| RustContractValueError::new_err(error.to_string()))
}

#[pyfunction]
fn runtime_plan_nested_invocation(payload_json: &str) -> PyResult<String> {
    kogwistar_runtime::plan_runtime_nested_invocation_from_str(payload_json)
        .map_err(|error| RustContractValueError::new_err(error.to_string()))
}

#[pyfunction]
fn runtime_decide_dispatch(payload_json: &str) -> PyResult<String> {
    kogwistar_runtime::decide_runtime_dispatch_from_str(payload_json)
        .map_err(|error| RustContractValueError::new_err(error.to_string()))
}

#[pyfunction]
fn runtime_decide_budget_suspend(payload_json: &str) -> PyResult<String> {
    kogwistar_runtime::decide_runtime_budget_suspend_from_str(payload_json)
        .map_err(|error| RustContractValueError::new_err(error.to_string()))
}

#[pyfunction]
fn runtime_scheduler_tick(payload_json: &str) -> PyResult<String> {
    kogwistar_runtime::tick_runtime_scheduler_from_str(payload_json)
        .map_err(|error| RustContractValueError::new_err(error.to_string()))
}

#[pyfunction]
fn api_health(payload_json: &str) -> PyResult<String> {
    kogwistar_api::health_response_from_str(payload_json)
        .map_err(|error| RustContractValueError::new_err(error.to_string()))
}

#[pyfunction]
fn api_authorize(payload_json: &str) -> PyResult<String> {
    kogwistar_api::authorize_from_str(payload_json)
        .map_err(|error| RustContractValueError::new_err(error.to_string()))
}

#[pyfunction]
fn api_sse_frame(payload_json: &str) -> PyResult<String> {
    kogwistar_api::sse_frame_from_str(payload_json)
        .map_err(|error| RustContractValueError::new_err(error.to_string()))
}

#[pyfunction]
fn api_mcp_result(payload_json: &str) -> PyResult<String> {
    kogwistar_api::mcp_result_from_str(payload_json)
        .map_err(|error| RustContractValueError::new_err(error.to_string()))
}

#[pyfunction]
fn api_cli_health(payload_json: &str) -> PyResult<String> {
    kogwistar_api::cli_health_from_str(payload_json)
        .map_err(|error| RustContractValueError::new_err(error.to_string()))
}

#[pyfunction]
fn api_run_server(py: Python<'_>) -> PyResult<()> {
    assert_transaction_entry_allowed_from_callback(py)?;
    py.detach(|| {
        TokioRuntimeBuilder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?
            .block_on(kogwistar_api::run_server_from_environment())
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    })
}

/// Build an isolated Rust in-memory store from a JSON snapshot, then inspect it.
/// This boundary has no handle to Python-owned backend or meta-store state.
#[pyfunction]
fn store_memory_read_json(py: Python<'_>, payload_json: &str) -> PyResult<String> {
    store_memory_read_json_impl(payload_json)
        .map_err(|(code, message)| store_error(py, code, message))
}

/// Keep the callback invariant at the native ABI, not only in Python facades.
/// Callers can import the extension directly and bypass `_rust_bridge` guards.
fn assert_transaction_entry_allowed_from_callback(py: Python<'_>) -> PyResult<()> {
    py.import("kogwistar.implementation_selection")?
        .call_method0("assert_transaction_entry_allowed")?;
    Ok(())
}

#[pyfunction]
fn store_sqlite_json(py: Python<'_>, payload_json: &str) -> PyResult<String> {
    assert_transaction_entry_allowed_from_callback(py)?;
    let mut request: Value = serde_json::from_str(payload_json)
        .map_err(|error| store_error(py, STORE_INVALID_JSON, format!("invalid JSON: {error}")))?;
    let database = request
        .get("path")
        .and_then(Value::as_str)
        .ok_or_else(|| store_error(py, STORE_INVALID_PAYLOAD, "missing SQLite path"))?
        .to_owned();
    let operation = request
        .get("operation")
        .and_then(|value| value.get("kind"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let reuse_session = request
        .get("reuse_session")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let guard = py
        .import("kogwistar.engine_core.sqlite_context")?
        .call_method1(
            "native_sqlite_enter",
            (&database, &operation, reuse_session),
        )?;
    let context_id: String = guard.get_item(0)?.extract()?;
    let _connection_lease = guard.get_item(1)?;
    if let Some(object) = request.as_object_mut() {
        object.insert("context_id".to_owned(), Value::String(context_id.clone()));
    }
    let payload_json = serde_json::to_string(&request)
        .map_err(|error| store_error(py, STORE_INVALID_JSON, error.to_string()))?;
    if std::env::var_os("KOGWISTAR_SQLITE_DIAGNOSTICS").as_deref()
        == Some(std::ffi::OsStr::new("1"))
    {
        eprintln!(
            "[sqlite-native-call] implementation=rust context={} operation={} reuse_session={} database={}",
            context_id, operation, reuse_session, database
        );
    }
    // SQLite calls can block on another reader or writer. Keep the Python-side
    // ownership checks above and cleanup below under the GIL, but let concurrent
    // Python callers reach the Rust store while one operation waits on SQLite.
    let result = py.detach(move || sqlite_store_json_impl(&payload_json));
    py.import("kogwistar.engine_core.sqlite_context")?
        .call_method1(
            "native_sqlite_exit",
            (
                database,
                context_id,
                operation,
                reuse_session,
                result.is_ok(),
            ),
        )?;
    result.map_err(|(code, message)| store_error(py, code, message))
}

#[pyfunction]
fn store_postgres_json(py: Python<'_>, payload_json: &str) -> PyResult<String> {
    assert_transaction_entry_allowed_from_callback(py)?;
    let payload_json = payload_json.to_owned();
    py.detach(move || postgres_store_json_impl(&payload_json))
        .map_err(|(code, message)| store_error(py, code, message))
}

#[pymodule]
fn _rust(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add("CONTRACT_VERSION", contracts::CONTRACT_VERSION)?;
    module.add(
        "RustContractTypeError",
        py.get_type::<RustContractTypeError>(),
    )?;
    module.add(
        "RustContractValueError",
        py.get_type::<RustContractValueError>(),
    )?;
    module.add("RustStoreValueError", py.get_type::<RustStoreValueError>())?;
    module.add_function(wrap_pyfunction!(stable_id, module)?)?;
    module.add_function(wrap_pyfunction!(stable_id_json, module)?)?;
    module.add_function(wrap_pyfunction!(canonical_json, module)?)?;
    module.add_function(wrap_pyfunction!(validate_graph_model_family, module)?)?;
    module.add_function(wrap_pyfunction!(
        round_trip_raw_graph_model_envelope,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(graph_model_level_from_root, module)?)?;
    module.add_function(wrap_pyfunction!(decode_persistent_graph_model_v2, module)?)?;
    module.add_function(wrap_pyfunction!(project_graph_model_slice, module)?)?;
    module.add_function(wrap_pyfunction!(project_graph_model_slices, module)?)?;
    module.add_function(wrap_pyfunction!(supported_graph_model_classes, module)?)?;
    module.add_function(wrap_pyfunction!(graph_model_capability_profiles, module)?)?;
    module.add_function(wrap_pyfunction!(evidence_pack_digest_hash, module)?)?;
    module.add_function(wrap_pyfunction!(metadata_filter_matches, module)?)?;
    module.add_function(wrap_pyfunction!(normalize_metadata_filter, module)?)?;
    module.add_function(wrap_pyfunction!(short_id_transform, module)?)?;
    module.add_function(wrap_pyfunction!(apply_state_update, module)?)?;
    module.add_function(wrap_pyfunction!(canonical_entity_event, module)?)?;
    module.add_function(wrap_pyfunction!(replay_entity_events, module)?)?;
    module.add_function(wrap_pyfunction!(workflow_may_reach_join, module)?)?;
    module.add_function(wrap_pyfunction!(workflow_terminal_reachable, module)?)?;
    module.add_function(wrap_pyfunction!(runtime_select_route, module)?)?;
    module.add_function(wrap_pyfunction!(runtime_plan_successors, module)?)?;
    module.add_function(wrap_pyfunction!(runtime_apply_join_arrival, module)?)?;
    module.add_function(wrap_pyfunction!(runtime_decide_retry, module)?)?;
    module.add_function(wrap_pyfunction!(runtime_plan_nested_invocation, module)?)?;
    module.add_function(wrap_pyfunction!(runtime_decide_dispatch, module)?)?;
    module.add_function(wrap_pyfunction!(runtime_decide_budget_suspend, module)?)?;
    module.add_function(wrap_pyfunction!(runtime_scheduler_tick, module)?)?;
    module.add_function(wrap_pyfunction!(api_health, module)?)?;
    module.add_function(wrap_pyfunction!(api_authorize, module)?)?;
    module.add_function(wrap_pyfunction!(api_sse_frame, module)?)?;
    module.add_function(wrap_pyfunction!(api_mcp_result, module)?)?;
    module.add_function(wrap_pyfunction!(api_cli_health, module)?)?;
    module.add_function(wrap_pyfunction!(api_run_server, module)?)?;
    module.add_function(wrap_pyfunction!(store_memory_read_json, module)?)?;
    module.add_function(wrap_pyfunction!(store_sqlite_json, module)?)?;
    module.add_function(wrap_pyfunction!(store_postgres_json, module)?)?;
    Ok(())
}
