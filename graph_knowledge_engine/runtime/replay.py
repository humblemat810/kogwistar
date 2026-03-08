from __future__ import annotations

import json
from typing import Any, Dict, Tuple, Optional, List

State = Dict[str, Any]


def load_checkpoint(*, conversation_engine: Any, run_id: str, step_seq: int) -> State:
    """
    Load a workflow checkpoint state snapshot from conversation_engine.
    """
    ckpt_id = f"wf_ckpt|{run_id}|{step_seq}"
    nodes = conversation_engine.get_nodes(ids=[ckpt_id], limit=1)
    if not nodes:
        raise KeyError(f"Checkpoint not found: {ckpt_id}")
    md = nodes[0].metadata or {}
    if md.get("entity_type") != "workflow_checkpoint":
        raise ValueError("Node is not a workflow_checkpoint")
    return json.loads(md["state_json"])

def _apply_state_update(state: State, state_update: List[Any]) -> None:
    """
    Replay-side reducer. Must match WorkflowRuntime.apply_state_update semantics.

    Supported ops:
      ('a', {k: v}) -> append v into list at state[k]
      ('u', {k: v}) -> overwrite state[k] = v
      ('e', {k: [..]}) -> extend list at state[k] with iterable
    """
    for item in state_update or []:
        if not (isinstance(item, (list, tuple)) and len(item) == 2):
            continue
        op, payload = item[0], item[1]
        if not isinstance(payload, dict):
            continue

        if op == "a":
            for k, v in payload.items():
                state.setdefault(k, []).append(v)

        elif op == "u":
            for k, v in payload.items():
                state[k] = v

        elif op == "e":
            for k, v in payload.items():
                state.setdefault(k, []).extend(v if isinstance(v, list) else list(v))


def replay_to(*, conversation_engine: Any, run_id: str, target_step_seq: int) -> State:
    """
    Reconstruct state by:
      - finding the nearest checkpoint <= target_step_seq
      - applying persisted step exec state_updates after that checkpoint up to target_step_seq
    """
    effective_target = int(target_step_seq)
    cancelled = conversation_engine.get_nodes(
        where={"$and": [{"entity_type": "workflow_cancelled"}, {"run_id": run_id}]},
        limit=10_000,
    )
    if cancelled:
        accepted = [
            int((n.metadata or {}).get("accepted_step_seq", -1))
            for n in cancelled
            if int((n.metadata or {}).get("accepted_step_seq", -1)) >= 0
        ]
        if accepted:
            effective_target = min(effective_target, min(accepted))

    ckpts = conversation_engine.get_nodes(
        where={"$and": [{"entity_type": "workflow_checkpoint"}, {"run_id": run_id}]},
        limit=10000,
    )

    best = None
    best_seq = -1
    for n in ckpts:
        seq = int((n.metadata or {}).get("step_seq", -1))
        if seq <= effective_target and seq > best_seq:
            best = n
            best_seq = seq
    if best is None:
        raise ValueError(f"No checkpoint <= {effective_target} for run_id={run_id}")

    state: State = json.loads((best.metadata or {})["state_json"])

    steps = conversation_engine.get_nodes(
        where={"$and": [{"entity_type": "workflow_step_exec"}, {"run_id": run_id}]},
        limit=200000,
    )
    steps_sorted = sorted(steps, key=lambda n: int((n.metadata or {}).get("step_seq", 0)))

    for n in steps_sorted:
        seq = int((n.metadata or {}).get("step_seq", 0))
        if seq <= best_seq:
            continue
        if seq > effective_target:
            break

        md = n.metadata or {}
        raw = md.get("result_json")
        if not raw:
            continue

        res = json.loads(raw)  # this is RunSuccess/RunFailure model_dump() JSON
        state_update = res.get("state_update") or []

        # Apply the same reducer as runtime
        _apply_state_update(state, state_update)

        # Optional: if you want to keep the envelope for debugging, store separately:
        # op = md.get("op")
        # if op:
        #     state.setdefault("_rt_step_exec", {})[str(seq)] = {"op": op, "result": res}

    return state

