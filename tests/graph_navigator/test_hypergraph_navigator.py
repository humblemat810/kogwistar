"""Contract tests for Hypergraph Navigator ACL semantics.

These tests are intentionally small and fake-backed. They are not proof that a
production navigator is wired to Kogwistar's ACL graph. They pin the semantic
contract that future integration tests and implementation must satisfy:

- traversal is ACL-first, not raw-then-filter
- pointer visibility never grants target visibility
- answer/ranking inputs come only from visible material
- derived artifacts require source-closure safety or an explicit sanitizer proof
- ACL changes stale only bounded downstream dependents
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass
class PrincipalContext:
    principal_id: str
    security_scope: str = ""
    groups: Set[str] = field(default_factory=set)
    visible_ids: Set[str] = field(default_factory=set)


@dataclass
class Node:
    node_id: str
    required_groundings: Set[str] = field(default_factory=set)
    required_spans: Set[str] = field(default_factory=set)


@dataclass
class Edge:
    edge_id: str
    source_id: str
    target_id: str


@dataclass
class Pointer:
    pointer_id: str
    target_id: str


@dataclass
class DerivedArtifact:
    artifact_id: str
    source_closure: Set[str]
    stale: bool = False
    sanitizer_proof_id: Optional[str] = None


class ContractVisibilityOracle:
    """Tiny stand-in for ACL-aware graph reads.

    Production code must replace this with persisted ACL graph decisions.
    Mutating ``visible_ids`` in tests simulates a later ACL change.
    """

    def __init__(self, context: PrincipalContext):
        self.context = context

    def can_see(self, object_id: str) -> bool:
        return object_id in self.context.visible_ids


class ContractNavigator:
    """Executable model of the navigator contract, not implementation code."""

    def __init__(self, visibility: ContractVisibilityOracle):
        self.visibility = visibility

    def can_traverse(self, edge: Edge) -> bool:
        return (
            self.visibility.can_see(edge.edge_id)
            and self.visibility.can_see(edge.source_id)
            and self.visibility.can_see(edge.target_id)
        )

    def dereference(self, pointer: Pointer) -> Optional[str]:
        if not self.visibility.can_see(pointer.pointer_id):
            return None
        if not self.visibility.can_see(pointer.target_id):
            return None
        return pointer.target_id

    def can_use_node_for_answer(self, node: Node) -> bool:
        if not self.visibility.can_see(node.node_id):
            return False
        required = set(node.required_groundings) | set(node.required_spans)
        return all(self.visibility.can_see(src) for src in required)

    def can_use_artifact_for_answer(self, artifact: DerivedArtifact) -> bool:
        if not self.visibility.can_see(artifact.artifact_id):
            return False
        if artifact.stale:
            return False
        if artifact.sanitizer_proof_id and self.visibility.can_see(
            artifact.sanitizer_proof_id
        ):
            return True
        return all(self.visibility.can_see(src) for src in artifact.source_closure)


class ReverseProvenanceIndex:
    def __init__(self, mapping: Dict[str, List[str]]):
        self.mapping = mapping

    def direct_dependents(self, source_id: str) -> List[str]:
        return list(self.mapping.get(source_id, []))

    def dependency_closure(self, source_id: str, *, max_items: int = 100) -> Set[str]:
        """Return bounded downstream closure from one changed source."""
        seen: Set[str] = set()
        frontier = [source_id]
        while frontier and len(seen) < max_items:
            current = frontier.pop(0)
            for dependent in self.mapping.get(current, []):
                if dependent in seen:
                    continue
                seen.add(dependent)
                frontier.append(dependent)
                if len(seen) >= max_items:
                    break
        return seen


def _nav_for(visible_ids: Set[str]) -> ContractNavigator:
    context = PrincipalContext(
        principal_id="agent-a",
        security_scope="tenant-a",
        visible_ids=set(visible_ids),
    )
    return ContractNavigator(ContractVisibilityOracle(context))


def test_traversal_requires_visible_edge_and_both_endpoints():
    nav = _nav_for({"e1", "n1", "n2"})

    assert nav.can_traverse(Edge("e1", "n1", "n2")) is True
    assert nav.can_traverse(Edge("e2", "n1", "n2")) is False
    assert nav.can_traverse(Edge("e1", "n1", "n3")) is False


def test_pointer_visibility_does_not_grant_target_visibility():
    nav = _nav_for({"p1"})

    assert nav.dereference(Pointer("p1", "kg-node-1")) is None


def test_pointer_dereference_uses_current_acl_not_historic_visibility():
    context = PrincipalContext(
        principal_id="agent-a",
        security_scope="tenant-a",
        visible_ids={"p1", "kg-node-1"},
    )
    nav = ContractNavigator(ContractVisibilityOracle(context))
    assert nav.dereference(Pointer("p1", "kg-node-1")) == "kg-node-1"

    context.visible_ids.remove("kg-node-1")
    assert nav.dereference(Pointer("p1", "kg-node-1")) is None


def test_node_cannot_answer_if_required_grounding_or_span_is_hidden():
    nav = _nav_for({"answer-node", "grounding-visible"})

    node = Node(
        "answer-node",
        required_groundings={"grounding-visible"},
        required_spans={"hidden-span"},
    )
    assert nav.can_use_node_for_answer(node) is False


def test_ranking_pool_must_be_acl_visible_only():
    context = PrincipalContext(
        principal_id="agent-a",
        security_scope="tenant-a",
        visible_ids={"visible-a", "visible-b"},
    )
    visibility = ContractVisibilityOracle(context)
    pool = ["visible-a", "hidden-c", "visible-b"]

    ranked_pool = [obj_id for obj_id in pool if visibility.can_see(obj_id)]
    assert ranked_pool == ["visible-a", "visible-b"]


def test_derived_artifact_requires_recursive_source_visibility_by_default():
    nav = _nav_for({"artifact-1", "src-a"})

    artifact = DerivedArtifact(
        artifact_id="artifact-1",
        source_closure={"src-a", "src-b"},
        stale=False,
    )
    assert nav.can_use_artifact_for_answer(artifact) is False


def test_sanitized_artifact_may_be_reused_without_full_source_visibility():
    nav = _nav_for({"artifact-1", "proof:safe-summary"})

    artifact = DerivedArtifact(
        artifact_id="artifact-1",
        source_closure={"private-src"},
        stale=False,
        sanitizer_proof_id="proof:safe-summary",
    )
    assert nav.can_use_artifact_for_answer(artifact) is True


def test_sanitizer_flag_without_visible_proof_does_not_bypass_source_acl():
    nav = _nav_for({"artifact-1"})

    artifact = DerivedArtifact(
        artifact_id="artifact-1",
        source_closure={"private-src"},
        stale=False,
        sanitizer_proof_id="proof:hidden",
    )
    assert nav.can_use_artifact_for_answer(artifact) is False


def test_stale_derived_artifact_is_not_user_facing_until_repaired():
    nav = _nav_for({"artifact-1", "src-a"})

    artifact = DerivedArtifact(
        artifact_id="artifact-1",
        source_closure={"src-a"},
        stale=True,
    )
    assert nav.can_use_artifact_for_answer(artifact) is False


def test_acl_change_marks_bounded_downstream_dependents_not_whole_graph():
    index = ReverseProvenanceIndex(
        {
            "src-a": ["artifact-1", "artifact-2"],
            "artifact-1": ["wisdom-1"],
            "src-b": ["artifact-3"],
        }
    )

    affected = index.dependency_closure("src-a")
    assert affected == {"artifact-1", "artifact-2", "wisdom-1"}
    assert "artifact-3" not in affected


def test_conversation_graph_stores_pointer_or_snapshot_not_full_truth_copy():
    conversation_graph_objects = {
        "run-1",
        "pointer:kg-node-1",
        "snapshot:evidence-1",
        "answer:artifact-1",
    }

    assert "kg-node-1" not in conversation_graph_objects
    assert "pointer:kg-node-1" in conversation_graph_objects
    assert "snapshot:evidence-1" in conversation_graph_objects

