from __future__ import annotations

import json
from typing import Any

from ..typing_interfaces import EngineLike
from .runtime import apply_state_update_inplace

State = dict[str, Any]


def _int_value(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, str)):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
    return default


def _load_json_text(value: object) -> State:
    if not isinstance(value, str):
        raise ValueError("checkpoint JSON payload must be a string")
    loaded = json.loads(value)
    if not isinstance(loaded, dict):
        raise ValueError("checkpoint JSON payload must be an object")
    return loaded


def load_checkpoint(*, conversation_engine: EngineLike, run_id: str, step_seq: int) -> State:
    """
    Load a workflow checkpoint state snapshot from conversation_engine.
    """
    ckpt_id = f"wf_ckpt|{run_id}|{step_seq}"
    metadatas = conversation_engine.read.get_node_metadatas(ids=[ckpt_id], limit=1)
    if not metadatas:
        raise KeyError(f"Checkpoint not found: {ckpt_id}")
    md = metadatas[0] or {}
    if md.get("entity_type") != "workflow_checkpoint":
        raise ValueError("Node is not a workflow_checkpoint")
    schema_version = _int_value(md.get("checkpoint_schema_version"))
    if schema_version not in {0, 1}:
        raise ValueError(
            f"Cannot load checkpoint {ckpt_id}: incompatible checkpoint_schema_version={schema_version}"
        )
    return _load_json_text(md.get("state_json"))


def _apply_state_update(state: State, state_update: list[Any]) -> None:
    """
    Replay-side reducer. Must match WorkflowRuntime.apply_state_update semantics.

    Supported ops:
      ('a', {k: v}) -> append v into list at state[k]
      ('u', {k: v}) -> overwrite state[k] = v
      ('e', {k: [..]}) -> extend list at state[k] with iterable
    """
    apply_state_update_inplace(state, state_update, None)


def replay_to(*, conversation_engine: EngineLike, run_id: str, target_step_seq: int) -> State:
    """
    Reconstruct state by:
      - finding the nearest checkpoint <= target_step_seq
      - applying persisted step exec state_updates after that checkpoint up to target_step_seq
    """
    effective_target = _int_value(target_step_seq)
    cancelled = conversation_engine.read.get_nodes(
        where={"$and": [{"entity_type": "workflow_cancelled"}, {"run_id": run_id}]},
        limit=10_000,
    )
    if cancelled:
        accepted = [
            _int_value((n.metadata or {}).get("accepted_step_seq"), -1)
            for n in cancelled
            if _int_value((n.metadata or {}).get("accepted_step_seq"), -1) >= 0
        ]
        if accepted:
            effective_target = min(effective_target, min(accepted))

    ckpts = conversation_engine.read.get_nodes(
        where={"$and": [{"entity_type": "workflow_checkpoint"}, {"run_id": run_id}]},
        limit=10000,
    )

    best = None
    best_seq = -1
    for n in ckpts:
        seq = _int_value((n.metadata or {}).get("step_seq"), -1)
        if seq <= effective_target and seq > best_seq:
            best = n
            best_seq = seq
    if best is None:
        raise ValueError(f"No checkpoint <= {effective_target} for run_id={run_id}")

    best_md = best.metadata or {}
    schema_version = _int_value(best_md.get("checkpoint_schema_version"))
    if schema_version not in {0, 1}:
        raise ValueError(
            f"Cannot replay run {run_id}: incompatible checkpoint_schema_version={schema_version}"
        )
    state = _load_json_text(best_md.get("state_json"))

    steps = conversation_engine.read.get_nodes(
        where={"$and": [{"entity_type": "workflow_step_exec"}, {"run_id": run_id}]},
        limit=200000,
    )
    steps_sorted = sorted(
        steps, key=lambda n: _int_value((n.metadata or {}).get("step_seq"))
    )

    for n in steps_sorted:
        seq = _int_value((n.metadata or {}).get("step_seq"))
        if seq <= best_seq:
            continue
        if seq > effective_target:
            break

        md = n.metadata or {}
        raw = md.get("result_json")
        if not raw:
            continue

        res = _load_json_text(raw)  # this is RunSuccess/RunFailure model_dump() JSON
        state_update = res.get("state_update") or []

        # Apply the same reducer as runtime
        _apply_state_update(state, state_update)

        # Optional: if you want to keep the envelope for debugging, store separately:
        # op = md.get("op")
        # if op:
        #     state.setdefault("_rt_step_exec", {})[str(seq)] = {"op": op, "result": res}

    return state
