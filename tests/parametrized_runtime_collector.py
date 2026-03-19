import json


def pytest_collection_modifyitems(config, items):
    # Collect runtime-parameterized test information from pytest's collected items.
    # This captures the concrete parameter values as they appear in item.callspec.params
    # (if present). Writes a lightweight JSON file mapping nodeid -> params for parametrized tests.
    out = []
    for item in items:
        callspec = getattr(item, "callspec", None)
        if not callspec:
            continue
        params = getattr(callspec, "params", None)
        if not params:
            continue
        out.append({
            "nodeid": item.nodeid,
            "params": dict(params),
        })

    if out:
        with open("parametrized_runtime_export.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, sort_keys=True, default=str)
