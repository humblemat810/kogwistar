import sys
import types

posthog_stub = types.SimpleNamespace(
    capture=lambda *a, **k: None,
    flush=lambda *a, **k: None,
    shutdown=lambda *a, **k: None,
    identify=lambda *a, **k: None,
    alias=lambda *a, **k: None,
)

sys.modules["posthog"] = posthog_stub