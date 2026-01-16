from __future__ import annotations

import json
from typing import Any, Dict, Tuple, Optional

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


def replay_to(*, conversation_engine: Any, run_id: str, target_step_seq: int) -> State:
    """
    Reconstruct state by:
      - finding the nearest checkpoint <= target_step_seq
      - applying step exec results after that checkpoint up to target_step_seq

    Default replay apply:
      state["result.<op>"] = result
    """
    # find all checkpoints for run_id (simple scan by where; assumes your engine supports it)
    ckpts = conversation_engine.get_nodes(
        where={"entity_type": "workflow_checkpoint", "run_id": run_id},
        limit=10000,
    )
    best = None
    best_seq = -1
    for n in ckpts:
        seq = int((n.metadata or {}).get("step_seq", -1))
        if seq <= target_step_seq and seq > best_seq:
            best = n
            best_seq = seq
    if best is None:
        raise ValueError(f"No checkpoint <= {target_step_seq} for run_id={run_id}")

    import json as _json
    state: State = _json.loads((best.metadata or {})["state_json"])

    steps = conversation_engine.get_nodes(
        where={"entity_type": "workflow_step_exec", "run_id": run_id},
        limit=200000,
    )
    steps_sorted = sorted(steps, key=lambda n: int((n.metadata or {}).get("step_seq", 0)))

    for n in steps_sorted:
        seq = int((n.metadata or {}).get("step_seq", 0))
        if seq <= best_seq:
            continue
        if seq > target_step_seq:
            break
        md = n.metadata or {}
        op = md.get("op")
        res = _json.loads(md["result_json"])
        state[f"result.{op}"] = res

    return state
