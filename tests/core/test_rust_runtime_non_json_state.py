from __future__ import annotations

import threading

from kogwistar.runtime.base_runtime import apply_state_update_inplace


def test_rust_runtime_falls_back_for_uncopyable_process_local_state(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KOGWISTAR_IMPL_RUNTIME", "rust")
    lock = threading.RLock()
    deps = {"lock": lock}
    assigned: list[str] = []
    state = {"_deps": deps, "items": []}

    apply_state_update_inplace(
        state,
        [("u", {"assigned": assigned}), ("a", {"items": "value"})],
    )

    assert state["_deps"] is deps
    assert state["_deps"]["lock"] is lock
    assert state["assigned"] is assigned
    assert state["items"] == ["value"]


def test_json_compatible_state_still_dispatches_to_rust(monkeypatch) -> None:
    monkeypatch.setenv("KOGWISTAR_IMPL_RUNTIME", "rust")
    calls: list[dict] = []

    def native(*, payload: dict) -> dict:
        calls.append(payload)
        return {"items": ["seed", "next"], "answer": 42}

    monkeypatch.setattr("kogwistar.runtime.base_runtime.runtime_apply_state_update", native)
    state = {"items": ["seed"]}

    apply_state_update_inplace(
        state,
        [("a", {"items": "next"}), ("u", {"answer": 42})],
    )

    assert len(calls) == 1
    assert calls[0]["state"] == {"items": ["seed"]}
    assert state == {"items": ["seed", "next"], "answer": 42}
