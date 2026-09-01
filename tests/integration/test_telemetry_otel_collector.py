from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from kogwistar.runtime.telemetry import EventEmitter, TraceContext


pytestmark = [pytest.mark.integration, pytest.mark.e2e, pytest.mark.slow]


def test_otel_otlp_receiver_smoke() -> None:
    """Send real SDK spans to an externally managed OTLP collector.

    Set ``KOGWISTAR_OTEL_COLLECTOR_ENDPOINT`` to an OTLP HTTP traces endpoint
    backed by an OTel Collector debug exporter.  The test is opt-in because a
    collector is not part of the normal unit-test process.
    """
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    except ImportError as exc:
        pytest.skip(f"OTLP HTTP exporter is not installed: {exc}")

    from kogwistar.runtime.telemetry_otel import OpenTelemetrySink

    container_id = None
    provider = None
    sink = None
    endpoint = os.getenv("KOGWISTAR_OTEL_COLLECTOR_ENDPOINT")
    checker = os.getenv("KOGWISTAR_OTEL_CHECK_URL")
    if not endpoint and os.getenv("KOGWISTAR_RUN_DOCKER_OTEL") == "1":
        docker = shutil.which("docker")
        if docker is None:
            pytest.skip("Docker is required for the managed OTel smoke test")
        config = Path(__file__).with_name("otel-collector-config.yaml")
        image = os.getenv(
            "KOGWISTAR_OTEL_COLLECTOR_IMAGE",
            "otel/opentelemetry-collector-contrib:0.128.0",
        )
        started = subprocess.run(
            [
                docker,
                "run",
                "--rm",
                "-d",
                "-p",
                "127.0.0.1::4318",
                "-v",
                f"{config}:/etc/otelcol-contrib/config.yaml:ro",
                image,
                "--config=/etc/otelcol-contrib/config.yaml",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        container_id = started.stdout.strip()
        try:
            port = subprocess.check_output(
                [docker, "port", container_id, "4318/tcp"], text=True
            )
        except Exception:
            subprocess.run([docker, "rm", "-f", container_id], check=False)
            container_id = None
            raise
        match = re.search(r":(\d+)\s*$", port.strip())
        if match is None:
            pytest.fail(f"cannot determine managed collector port: {port!r}")
        endpoint = f"http://127.0.0.1:{match.group(1)}/v1/traces"
    if not endpoint:
        pytest.skip(
            "configure KOGWISTAR_OTEL_COLLECTOR_ENDPOINT or "
            "KOGWISTAR_RUN_DOCKER_OTEL=1"
        )

    try:
        provider = TracerProvider()
        provider.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporter(endpoint=endpoint), max_export_batch_size=1
            )
        )
        tracer = provider.get_tracer("kogwistar.integration")
        sink = OpenTelemetrySink(tracer, queue_max=32)
        emitter = EventEmitter(sink=sink)
        ctx = TraceContext.new_root(
            run_id="otel-smoke", token_id="token", step_seq=0, node_id="start"
        )
        emitter.emit(
            type="workflow_run_started", ctx=ctx, payload={"workflow_id": "otel-smoke"}
        )
        emitter.emit(type="workflow_run_completed", ctx=ctx)
        assert sink.flush(5.0)
        provider.force_flush(timeout_millis=5_000)

        if checker:
            import urllib.request

            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                with urllib.request.urlopen(checker, timeout=2.0) as response:
                    if response.status == 200 and "otel-smoke" in response.read().decode(
                        "utf-8"
                    ):
                        return
                time.sleep(0.25)
            pytest.fail("OTel collector checker did not observe kogwistar smoke span")
        assert container_id is not None
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            logs = subprocess.run(
                [shutil.which("docker") or "docker", "logs", container_id],
                check=False,
                capture_output=True,
                text=True,
            )
            if "otel-smoke" in (logs.stdout + logs.stderr):
                return
            time.sleep(0.25)
        pytest.fail("managed OTel Collector debug exporter did not observe smoke span")
    finally:
        if sink is not None:
            sink.close(timeout=5.0)
        if provider is not None:
            provider.shutdown()
        if container_id:
            subprocess.run(
                [shutil.which("docker") or "docker", "rm", "-f", container_id],
                check=False,
                capture_output=True,
                text=True,
            )
