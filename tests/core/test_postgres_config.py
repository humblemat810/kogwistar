from kogwistar.engine_core.postgres_backend import PgVectorConfig, postgres_connect_args


def test_postgres_connect_args_bound_sql_without_lock_timeout():
    args = postgres_connect_args(
        PgVectorConfig(
            dsn="postgresql+psycopg://user:pass@localhost/db",
            embedding_dim=2,
            statement_timeout_ms=1234,
            idle_transaction_timeout_ms=5678,
            application_name="test-app",
        )
    )

    assert args["application_name"] == "test-app"
    assert "statement_timeout=1234" in args["options"]
    assert "idle_in_transaction_session_timeout=5678" in args["options"]
    assert "lock_timeout" not in args["options"]
