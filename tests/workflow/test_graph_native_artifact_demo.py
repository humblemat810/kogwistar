# ruff: noqa: E402
from __future__ import annotations

import pytest

pytestmark = pytest.mark.ci

from kogwistar.demo import (
    run_conversation_workflow_demo,
    run_execution_memory_demo,
    run_graph_native_artifact_demo,
    run_provenance_reasoning_demo,
    run_unified_substrate_demo_suite,
)


def test_graph_native_artifact_demo_commits_notes_into_graph_snapshot():
    result = run_graph_native_artifact_demo()

    assert result["framework_name"] == "GraphArtifactPipelineFramework"
    assert result["framework_step_order"] == [
        "ingest",
        "validate",
        "normalize",
        "link",
        "commit",
        "end",
    ]
    assert result["transition_map"]["validate"] == {"valid": "normalize", "default": "end"}
    assert result["run_status"] == "succeeded"
    assert result["final_state"]["final_status"] == "committed"
    assert result["final_state"]["completed"] is True
    assert result["graph_snapshot"]["nodes"][0]["topic"] == "finance"
    assert result["graph_snapshot"]["edges"] == [
        {
            "source": "note-1",
            "target": "note-2",
            "relation": "same_topic",
            "topic": "finance",
        }
    ]
    assert [event["step"] for event in result["execution_history"]] == [
        "ingest",
        "validate",
        "normalize",
        "link",
        "commit",
        "end",
    ]
    assert [event["step"] for event in result["provenance_log"]] == [
        "ingest",
        "validate",
        "normalize",
        "link",
        "commit",
    ]


def test_graph_native_artifact_demo_blocks_invalid_input_before_commit():
    result = run_graph_native_artifact_demo(
        raw_notes=[
            {
                "id": "note-1",
                "title": "",
                "text": "This note is missing a title",
            }
        ]
    )

    assert result["run_status"] == "succeeded"
    assert result["final_state"]["final_status"] == "blocked"
    assert result["final_state"]["completed"] is False
    assert result["final_state"]["validation_passed"] is False
    assert result["final_state"]["validation_errors"] == [
        {"id": "note-1", "error": "missing title"}
    ]
    assert result["graph_snapshot"] == {"nodes": [], "edges": []}
    assert [event["step"] for event in result["execution_history"]] == [
        "ingest",
        "validate",
        "end",
    ]


def test_execution_memory_demo_reuses_graph_history():
    result = run_execution_memory_demo()

    assert result["summary"]["first_run_processed"] == ["note-1", "note-2", "note-3"]
    assert result["summary"]["second_run_skipped_as_known"] == ["note-2", "note-3"]
    assert result["summary"]["second_run_processed"] == ["note-4"]
    assert result["details"]["step_ops"]["first_run"] == [
        "ingest",
        "validate",
        "normalize",
        "link",
        "commit",
        "end",
    ]
    assert result["details"]["step_ops"]["second_run"] == [
        "ingest",
        "validate",
        "normalize",
        "link",
        "commit",
        "end",
    ]


def test_conversation_workflow_demo_links_turns_and_run():
    result = run_conversation_workflow_demo()

    assert result["summary"]["question"] == "Organize my notes"
    assert result["summary"]["linked_entities"] == {
        "conversation_nodes": 2,
        "workflow_run_nodes": 1,
        "workflow_step_nodes": 6,
        "artifact_nodes": 3,
    }
    assert result["details"]["conversation_node_ids"] == [
        "turn|conversation-workflow|user-1",
        "turn|conversation-workflow|assistant-1",
    ]
    assert result["details"]["workflow_step_ops"][:3] == [
        "ingest",
        "validate",
        "normalize",
    ]
    assert result["details"]["conversation_edge_ids"]


def test_provenance_reasoning_demo_uses_stored_step_history():
    result = run_provenance_reasoning_demo()

    assert set(result) == {"summary", "details"}
    assert result["summary"]["question"] == "Why did we move note-1 to finance?"
    assert result["summary"]["answer"].startswith("I would move note-1 to finance/")
    assert result["summary"]["evidence_steps"] == ["normalize", "link", "commit"]
    assert result["summary"]["grounding_excerpt"] == "Vendor invoice"
    assert result["details"]["scenario"] == "provenance_reasoning"
    assert result["details"]["workflow_run_id"]
    assert result["details"]["workflow_step_ids"]
    assert result["details"]["answer"]["run_id"] == result["details"]["workflow_run_id"]
    assert result["details"]["answer"]["explanation"].startswith("I would move note-1 to finance/")
    assert result["details"]["artifact_node_id"].startswith("artifact|graph_native_artifact_demo|note-1")
    assert result["details"]["grounding_trace"]["source_document_id"] == "doc|graph_native_artifact_demo|note-1"
    assert result["details"]["grounding_trace"]["span"] == {
        "doc_id": "doc|graph_native_artifact_demo|note-1",
        "document_page_url": "demo://graph-native-artifact/doc|graph_native_artifact_demo|note-1",
        "page_number": 1,
        "start_char": 7,
        "end_char": 21,
        "excerpt": "Vendor invoice",
        "context_before": "Title: ",
        "context_after": "\nBody: Invoice for subsc",
    }
    assert result["details"]["answer"]["evidence"] == [
        "grounding span excerpt='Vendor invoice' at chars 7:21 in doc|graph_native_artifact_demo|note-1",
        "normalize step stored topic='finance' for note-1",
        "commit step stored artifact note note-1 in the graph",
        "link step stored same-topic edges for the run",
    ]


def test_unified_substrate_demo_suite_contains_all_scenarios():
    result = run_unified_substrate_demo_suite()

    assert set(result) == {"summary", "details"}
    assert set(result["summary"]) == {
        "execution_memory",
        "conversation_workflow",
        "provenance_reasoning",
    }
    assert set(result["details"]) == {
        "execution_memory",
        "conversation_workflow",
        "provenance_reasoning",
    }
    assert result["details"]["conversation_workflow"]["workflow_run_id"]
