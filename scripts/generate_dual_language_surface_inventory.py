"""Generate the versioned Python/Rust public-surface operation inventory."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "contracts" / "dual-language-surface-inventory-v1.json"
OPENAPI = ROOT / "contracts" / "golden" / "openapi.json"
MCP = ROOT / "contracts" / "golden" / "mcp-tools.json"
RUST_API = ROOT / "rust" / "crates" / "kogwistar-api" / "src"
RUST_CUTOVER = RUST_API / "cutover.rs"
RUST_MCP = RUST_API / "sections" / "contracts_and_runtime.rs"
RUST_PORT_MANIFEST = ROOT / "contracts" / "rust-port-v1.json"


def _string_pairs(source: str, constant: str) -> list[list[str]]:
    match = re.search(
        rf"pub const {constant}:.*?= &\[(.*?)\];", source, flags=re.DOTALL
    )
    if match is None:
        raise ValueError(f"could not find Rust constant {constant}")
    return [list(pair) for pair in re.findall(r'"([A-Z]+)"\s*,\s*"([^"]+)"', match.group(1))]


def _rust_mcp_allowlist(source: str) -> dict[str, set[str]]:
    start = source.index("fn rust_mcp_tool_supported(")
    end = source.index("#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]", start)
    function = source[start:end]
    result: dict[str, set[str]] = {}
    for surface in ("workflow", "conversation", "root"):
        match = re.search(
            rf'"{surface}"\s*=>\s*matches!\(\s*name,(.*?)\n\s*\),',
            function,
            flags=re.DOTALL,
        )
        if match is None:
            raise ValueError(f"could not parse Rust MCP surface {surface}")
        result[surface] = set(re.findall(r'"([a-z][a-z0-9_.]*)"', match.group(1)))
    return result


def build_inventory() -> dict[str, Any]:
    openapi = json.loads(OPENAPI.read_text(encoding="utf-8"))
    mcp = json.loads(MCP.read_text(encoding="utf-8"))
    deferrals = json.loads(RUST_PORT_MANIFEST.read_text(encoding="utf-8"))[
        "server_operation_deferrals"
    ]
    rust_mcp = _rust_mcp_allowlist(RUST_MCP.read_text(encoding="utf-8"))
    pending_routes = {
        (operation["method"], operation["path"])
        for group in deferrals["route_groups"]
        for operation in group["operations"]
    }
    transport_routes = {
        tuple(pair)
        for pair in _string_pairs(
            RUST_CUTOVER.read_text(encoding="utf-8"), "TRANSPORT_OWNED_HTTP_ROUTES"
        )
    }
    pending_syscalls = deferrals["syscall_group"]["operations"]

    http_operations: list[dict[str, str]] = []
    for path, path_item in sorted(openapi["paths"].items()):
        for method in sorted(path_item):
            if method.lower() not in {"get", "post", "put", "patch", "delete", "head", "options", "trace"}:
                continue
            operation = (method.upper(), path)
            rust_status = (
                "transport-owned"
                if operation in transport_routes
                else "pending"
                if operation in pending_routes
                else "application-route-in-cutover-inventory"
            )
            http_operations.append(
                {"method": method.upper(), "path": path, "python": "openapi", "rust": rust_status}
            )

    mcp_operations: list[dict[str, str]] = []
    for surface in sorted(mcp["surfaces"]):
        for tool in sorted(mcp["surfaces"][surface], key=lambda item: item["name"]):
            name = tool["name"]
            mcp_operations.append(
                {
                    "surface": surface,
                    "name": name,
                    "python": "frozen-catalog",
                    "rust": "allowlisted-subject-to-runtime-gates"
                    if name in rust_mcp.get(surface, set())
                    else "not-allowlisted",
                }
            )

    return {
        "schema_version": 1,
        "contract_id": "kogwistar-dual-language-surface-inventory",
        "status": "inventory-in-progress",
        "interpretation": {
            "python": "Presence in the frozen OpenAPI or MCP catalog records the public Python contract, not independent proof of every backend configuration.",
            "rust_http": "Application routes are classified by the Rust cutover inventory; transport-owned routes and pending routes are explicit. This is not behavioral parity evidence.",
            "rust_mcp": "Allowlist status is necessary but not sufficient: backend capability, ACL, namespace, and runtime checks still apply.",
            "syscalls": "Pending syscall names are the explicit Rust cutover gaps; non-pending names are not thereby proven behaviorally equivalent.",
        },
        "counts": {
            "http_operations": len(http_operations),
            "mcp_operations": len(mcp_operations),
            "pending_http_operations": sum(item["rust"] == "pending" for item in http_operations),
            "allowlisted_mcp_operations": sum(item["rust"] == "allowlisted-subject-to-runtime-gates" for item in mcp_operations),
            "pending_syscalls": len(pending_syscalls),
        },
        "http_operations": http_operations,
        "mcp_operations": mcp_operations,
        "pending_syscalls": pending_syscalls,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if the checked-in inventory is stale")
    args = parser.parse_args()
    rendered = json.dumps(build_inventory(), indent=2, sort_keys=False) + "\n"
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text(encoding="utf-8") != rendered:
            raise SystemExit("dual-language surface inventory is stale; regenerate it")
        return 0
    OUTPUT.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
