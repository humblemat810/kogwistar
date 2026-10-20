"""Static join reachability analysis for workflow graph designs."""

from __future__ import annotations

from collections.abc import Iterator

from kogwistar.runtime.models import WorkflowEdge


def _tarjan_scc(n: int, succ: list[list[int]]) -> tuple[list[int], list[list[int]]]:
    """Return component IDs and members for a directed graph."""
    index = 0
    stack: list[int] = []
    onstack = [False] * n
    idxs = [-1] * n
    low = [0] * n
    comp_of = [-1] * n
    comps: list[list[int]] = []

    def strongconnect(v: int) -> None:
        nonlocal index
        idxs[v] = index
        low[v] = index
        index += 1
        stack.append(v)
        onstack[v] = True

        for w in succ[v]:
            if idxs[w] == -1:
                strongconnect(w)
                low[v] = low[v] if low[v] < low[w] else low[w]
            elif onstack[w]:
                low[v] = low[v] if low[v] < idxs[w] else idxs[w]

        if low[v] == idxs[v]:
            comp_id = len(comps)
            comp: list[int] = []
            while True:
                w = stack.pop()
                onstack[w] = False
                comp_of[w] = comp_id
                comp.append(w)
                if w == v:
                    break
            comps.append(comp)

    for vertex in range(n):
        if idxs[vertex] == -1:
            strongconnect(vertex)

    return comp_of, comps


def _compute_may_reach_join_bitsets_python(
    *,
    node_ids: list[str],
    adj: dict[str, list[WorkflowEdge]],
    join_ids: list[str],
) -> dict[str, int]:
    """Over-approximate the joins each node may reach, ignoring predicates."""
    n = len(node_ids)
    id_to_i = {node_id: index for index, node_id in enumerate(node_ids)}
    succ: list[list[int]] = [[] for _ in range(n)]
    for source_id, edges in adj.items():
        source_index = id_to_i.get(source_id)
        if source_index is None:
            continue
        for edge in edges:
            for target_id in edge.target_ids:
                target_index = id_to_i.get(target_id)
                if target_index is not None:
                    succ[source_index].append(target_index)

    comp_of, components = _tarjan_scc(n, succ)
    component_count = len(components)

    join_pos = {join_id: position for position, join_id in enumerate(join_ids)}
    comp_join_mask = [0] * component_count
    for node_id in node_ids:
        position = join_pos.get(node_id)
        if position is not None:
            node_index = id_to_i[node_id]
            comp_join_mask[comp_of[node_index]] |= 1 << position

    comp_succ: list[set[int]] = [set() for _ in range(component_count)]
    indeg = [0] * component_count
    for vertex in range(n):
        component = comp_of[vertex]
        for target in succ[vertex]:
            target_component = comp_of[target]
            if component != target_component and target_component not in comp_succ[component]:
                comp_succ[component].add(target_component)
                indeg[target_component] += 1

    ready = [component for component in range(component_count) if indeg[component] == 0]
    topo: list[int] = []
    while ready:
        component = ready.pop()
        topo.append(component)
        for successor in comp_succ[component]:
            indeg[successor] -= 1
            if indeg[successor] == 0:
                ready.append(successor)

    may_reach = comp_join_mask[:]
    for component in reversed(topo):
        mask = may_reach[component]
        for successor in comp_succ[component]:
            mask |= may_reach[successor]
        may_reach[component] = mask

    return {
        node_id: may_reach[comp_of[id_to_i[node_id]]]
        for node_id in node_ids
    }


def _compute_may_reach_join_bitsets(
    *,
    node_ids: list[str],
    adj: dict[str, list[WorkflowEdge]],
    join_ids: list[str],
) -> dict[str, int]:
    """Select the runtime-authoritative static join-lineage implementation."""
    from kogwistar._rust_bridge import (
        runtime_implementation_mode,
        runtime_workflow_may_reach_join,
    )

    mode = runtime_implementation_mode()
    edges = [
        (str(source_id), str(target_id))
        for source_id, workflow_edges in adj.items()
        for edge in workflow_edges
        for target_id in edge.target_ids
    ]
    if mode == "rust":
        rust_value = runtime_workflow_may_reach_join(
            node_ids=node_ids,
            edges=edges,
            join_ids=join_ids,
        )
        return rust_value if rust_value is not None else {}

    python_value = _compute_may_reach_join_bitsets_python(
        node_ids=node_ids,
        adj=adj,
        join_ids=join_ids,
    )
    selected = runtime_workflow_may_reach_join(
        node_ids=node_ids,
        edges=edges,
        join_ids=join_ids,
        python_value=python_value,
    )
    return selected if selected is not None else python_value


def _iter_bits(mask: int) -> Iterator[int]:
    """Yield bit positions (zero-based) from least significant to most."""
    while mask:
        least_significant_bit = mask & -mask
        yield least_significant_bit.bit_length() - 1
        mask ^= least_significant_bit
