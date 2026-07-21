from __future__ import annotations

import pytest

from kogwistar import rust_server


pytestmark = [pytest.mark.ci, pytest.mark.core]


def test_launcher_refuses_when_python_owns_server(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KOGWISTAR_IMPL_SERVER", "python")

    with pytest.raises(SystemExit, match="requires KOGWISTAR_IMPL_SERVER=rust"):
        rust_server.main()


def test_launcher_calls_packaged_native_server(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[bool] = []

    class Extension:
        @staticmethod
        def api_run_server() -> None:
            called.append(True)

    monkeypatch.setenv("KOGWISTAR_IMPL_SERVER", "rust")
    monkeypatch.setattr(rust_server, "_load_extension", lambda: Extension())

    rust_server.main()

    assert called == [True]
