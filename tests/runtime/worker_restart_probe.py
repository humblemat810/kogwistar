from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import httpx

from kogwistar.engine_core.graph_mutation_contract import GraphMutationBatchPayload
from kogwistar.runtime.models import RunSuccess
from kogwistar.runtime.resolvers import MappingStepResolver
from kogwistar.runtime.rust_worker import RustRuntimeWorker, RustWorkerError


class _LoseResultBeforeDelivery(httpx.BaseTransport):
    def __init__(self) -> None:
        self._inner = httpx.HTTPTransport()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/internal/runtime/results":
            raise httpx.ReadError("simulated result delivery loss", request=request)
        return self._inner.handle_request(request)

    def close(self) -> None:
        self._inner.close()


class _LoseResultAcknowledgementAfterCommit(httpx.BaseTransport):
    def __init__(self) -> None:
        self._inner = httpx.HTTPTransport()
        self._result_calls = 0
        self.event_seq: int | None = None

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        response = self._inner.handle_request(request)
        if request.url.path == "/internal/runtime/results":
            self._result_calls += 1
            if self._result_calls == 1:
                body = response.read()
                result = json.loads(body)
                event_seq = result.get("event_seq") if isinstance(result, dict) else None
                if not isinstance(event_seq, int) or event_seq <= 0:
                    raise AssertionError("committed result lacks a positive event_seq")
                self.event_seq = event_seq
                response.close()
                raise httpx.ReadError(
                    "simulated result acknowledgement loss after commit", request=request
                )
        return response

    def close(self) -> None:
        self._inner.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--journal", required=True)
    parser.add_argument("--marker", required=True)
    parser.add_argument("--work-record", required=True)
    parser.add_argument("--graph-namespace")
    parser.add_argument("--graph-entity-id")
    parser.add_argument("--auth-token")
    parser.add_argument("--snapshot-effect", action="store_true")
    parser.add_argument(
        "--mode", choices=("prepare", "replay", "recover"), required=True
    )
    args = parser.parse_args()

    graph_values = (args.graph_namespace, args.graph_entity_id)
    if args.snapshot_effect and not args.graph_namespace:
        raise SystemExit("snapshot effect mode requires its graph namespace")
    if (
        not args.snapshot_effect
        and any(value is not None for value in graph_values)
        and not all(graph_values)
    ):
        raise SystemExit("graph replay probe requires both namespace and entity ID")
    if (args.graph_namespace is not None or args.snapshot_effect) and not args.auth_token:
        raise SystemExit("graph replay probe requires its auth token")
    if args.snapshot_effect and args.graph_entity_id is not None:
        raise SystemExit("snapshot effect mode derives its own graph entity ID")

    dependencies = None
    if args.snapshot_effect:
        from kogwistar.conversation.conversation_context import (
            ContextItem,
            ContextMessage,
            PromptContext,
        )
        from kogwistar.conversation.resolvers import default_resolver
        from kogwistar.conversation.service import ConversationService

        class _SnapshotEngine:
            def __init__(self, namespace: str) -> None:
                self.namespace = namespace
                self.persistence_mode = "single_stage"
                self._phase1_enable_index_jobs = True
                self._disable_event_log = False
                self.backend = SimpleNamespace(embedding_dim=3)
                self.embed = SimpleNamespace(
                    iterative_defensive_emb=lambda _text: [1.0, 0.0, 0.0]
                )
                self.indexing = SimpleNamespace(
                    canonical_source_payload_for_projection=lambda **_kwargs: "{}"
                )
                self.write = SimpleNamespace(
                    node_doc_and_meta=lambda node: (
                        node.model_dump_json(
                            field_mode="backend", exclude=["embedding"]
                        ),
                        {**node.metadata, "namespace": namespace},
                    ),
                    edge_doc_and_meta=lambda edge: (
                        edge.model_dump_json(
                            field_mode="backend", exclude=["embedding"]
                        ),
                        {**edge.metadata, "namespace": namespace},
                    ),
                    add_node=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                        AssertionError("worker snapshot must not write through Python")
                    ),
                    add_edge=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                        AssertionError("worker snapshot must not write through Python")
                    ),
                )
                self.pre_add_node_hooks = []
                self.pre_add_edge_hooks = []

        engine = _SnapshotEngine(args.graph_namespace or "conversation")
        service = ConversationService.__new__(ConversationService)
        service.conversation_engine = engine

        class _Agent:
            config = SimpleNamespace(evidence_selector="llm", max_used=3)
            cache_dir = None

            @staticmethod
            def _persist_context_snapshot(**_kwargs):
                raise AssertionError("worker snapshot must not persist in Python")

            @staticmethod
            def select_used_evidence_cached(**_kwargs):
                return SimpleNamespace(
                    model_dump=lambda: {
                        "used_node_ids": [],
                        "used_edge_ids": [],
                        "reasoning": "deterministic worker smoke",
                    }
                )

        dependencies = {
            "agent": _Agent(),
            "conversation_engine": engine,
            "conversation_service": service,
            "llm_tasks": SimpleNamespace(provider_hints=SimpleNamespace()),
        }
        import kogwistar.conversation.resolvers as resolver_module

        resolver_module._chat_service = lambda deps: deps["conversation_service"]

    resolver = MappingStepResolver()

    @resolver.register("application.before")
    def before(ctx):
        with Path(args.marker).open("a", encoding="utf-8") as marker:
            marker.write("callback\n")
        if args.snapshot_effect:
            with ctx.state_write as state:
                state["_rt"] = {
                    "view": PromptContext(
                        conversation_id=str(state["conversation_id"]),
                        purpose="answer",
                        token_budget=16,
                        tokens_used=1,
                        items=(
                            ContextItem(
                                kind="system_prompt",
                                role="system",
                                text="snapshot smoke context",
                                source="system",
                                token_cost=1,
                            ),
                        ),
                        messages=(
                            ContextMessage(
                                role="system",
                                content="snapshot smoke context",
                                source="system",
                            ),
                        ),
                    )
                }
            result = default_resolver.resolve("aa_select_used_evidence")(ctx)
            result.state_update.append(("u", {"python_before_seen": True}))
            return result
        if args.graph_namespace is not None:
            assert args.graph_namespace is not None
            assert args.graph_entity_id is not None
            assert args.auth_token is not None
            assert ctx.idempotency_key
            batch = GraphMutationBatchPayload.model_validate(
                {
                    "contract_version": 1,
                    "atomic": True,
                    "mutations": [
                        {
                            "scope": {"namespace": args.graph_namespace},
                            "table": "gke_nodes",
                            "entity_kind": "node",
                            "event_id": f"graph-effect-{ctx.idempotency_key}",
                            "op": "ADD",
                            "payload": {
                                "id": args.graph_entity_id,
                                "text": "worker effect",
                            },
                            "record": {
                                "id": args.graph_entity_id,
                                "document": "worker effect",
                            },
                            "embedding_dim": 3,
                        }
                    ],
                    "index_jobs": [
                        {
                            "job_id": f"graph-index-{ctx.idempotency_key}",
                            "namespace": args.graph_namespace,
                            "entity_kind": "node",
                            "entity_id": args.graph_entity_id,
                            "index_kind": "node_docs",
                            "op": "UPSERT",
                        }
                    ],
                }
            )
            return RunSuccess(
                conversation_node_id=None,
                state_update=[("u", {"python_before_seen": True})],
                graph_mutation_batch=batch.model_dump(mode="json"),
            )
        return RunSuccess(
            conversation_node_id=None,
            state_update=[("u", {"python_before_seen": True})],
        )

    worker_headers = (
        {"authorization": f"Bearer {args.auth_token}"}
        if args.auth_token
        else {"x-kogwistar-roles": "reader"}
    )
    commit_ack_transport = None
    if args.mode == "prepare":
        client = httpx.Client(
            transport=_LoseResultBeforeDelivery(),
            base_url=args.base_url,
            headers=worker_headers,
        )
    elif args.mode == "replay":
        commit_ack_transport = _LoseResultAcknowledgementAfterCommit()
        client = httpx.Client(
            transport=commit_ack_transport,
            base_url=args.base_url,
            headers=worker_headers,
        )
    else:
        client = None
    with RustRuntimeWorker.from_step_resolver(
        base_url=args.base_url,
        worker_id=args.worker_id,
        journal_path=args.journal,
        step_resolver=resolver,
        dependency_provider=(lambda _work: dependencies) if dependencies else None,
        headers=worker_headers,
        client=client,
    ) as worker:
        if args.mode == "recover":
            record = json.loads(Path(args.work_record).read_text(encoding="utf-8"))
            if not isinstance(record, dict) or not isinstance(record.get("work"), dict):
                raise AssertionError("restart record must contain a work object")
            work_item = record["work"]
            expected_event_seq = record.get("event_seq")
            if not isinstance(expected_event_seq, int) or expected_event_seq <= 0:
                raise AssertionError("restart record must contain a positive event_seq")
            recovered = worker.process(work_item)
            if recovered.get("event_seq") != expected_event_seq:
                raise AssertionError("process restart changed the committed event")
        else:
            work = worker.claim(run_id="mixed-live-run", lease_seconds=1)
            if len(work) != 1 or work[0]["step_id"] != "python-before":
                raise AssertionError(f"unexpected reclaimed work: {work!r}")
            work_item = work[0]
        if args.mode == "prepare":
            try:
                worker.process(work_item)
            except RustWorkerError as exc:
                if "simulated result delivery loss" not in str(exc):
                    raise
            else:
                raise AssertionError("expected the first result delivery to fail")
        elif args.mode == "replay":
            Path(args.work_record).write_text(
                json.dumps({"work": work_item}), encoding="utf-8"
            )
            try:
                worker.process(work_item)
            except RustWorkerError as exc:
                if "simulated result acknowledgement loss after commit" not in str(exc):
                    raise
                if commit_ack_transport is None or commit_ack_transport.event_seq is None:
                    raise AssertionError("result commit was not observed by the transport")
                Path(args.work_record).write_text(
                    json.dumps(
                        {"work": work_item, "event_seq": commit_ack_transport.event_seq}
                    ),
                    encoding="utf-8",
                )
            else:
                raise AssertionError("expected committed result acknowledgement loss")


if __name__ == "__main__":
    main()
