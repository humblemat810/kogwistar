from __future__ import annotations

import argparse
from pathlib import Path

import httpx

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


class _LoseDuplicateAckAfterCommit(httpx.BaseTransport):
    def __init__(self) -> None:
        self._inner = httpx.HTTPTransport()
        self._result_calls = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        response = self._inner.handle_request(request)
        if request.url.path == "/internal/runtime/results":
            self._result_calls += 1
            if self._result_calls == 2:
                response.read()
                response.close()
                raise httpx.ReadError(
                    "simulated duplicate acknowledgement loss", request=request
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
    parser.add_argument("--mode", choices=("prepare", "replay"), required=True)
    args = parser.parse_args()

    resolver = MappingStepResolver()

    @resolver.register("application.before")
    def before(_ctx):
        with Path(args.marker).open("a", encoding="utf-8") as marker:
            marker.write("callback\n")
        return RunSuccess(
            conversation_node_id=None,
            state_update=[("u", {"python_before_seen": True})],
        )

    client = (
        httpx.Client(
            transport=_LoseResultBeforeDelivery(),
            base_url=args.base_url,
            headers={"x-kogwistar-roles": "reader"},
        )
        if args.mode == "prepare"
        else httpx.Client(
            transport=_LoseDuplicateAckAfterCommit(),
            base_url=args.base_url,
            headers={"x-kogwistar-roles": "reader"},
        )
        if args.mode == "replay"
        else None
    )
    with RustRuntimeWorker.from_step_resolver(
        base_url=args.base_url,
        worker_id=args.worker_id,
        journal_path=args.journal,
        step_resolver=resolver,
        headers={"x-kogwistar-roles": "reader"},
        client=client,
    ) as worker:
        work = worker.claim(run_id="mixed-live-run", lease_seconds=1)
        if len(work) != 1 or work[0]["step_id"] != "python-before":
            raise AssertionError(f"unexpected reclaimed work: {work!r}")
        if args.mode == "prepare":
            try:
                worker.process(work[0])
            except RustWorkerError as exc:
                if "simulated result delivery loss" not in str(exc):
                    raise
            else:
                raise AssertionError("expected the first result delivery to fail")
        else:
            result = worker.process(work[0])
            if result.get("event_seq", 0) <= 0:
                raise AssertionError(f"replayed result was not admitted: {result!r}")
            try:
                worker.process(work[0])
            except RustWorkerError as exc:
                if "simulated duplicate acknowledgement loss" not in str(exc):
                    raise
            else:
                raise AssertionError("expected duplicate acknowledgement loss")
            duplicate_result = worker.process(work[0])
            if duplicate_result.get("event_seq") != result["event_seq"]:
                raise AssertionError("duplicate result changed the committed event")


if __name__ == "__main__":
    main()
