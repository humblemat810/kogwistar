"""Generate the versioned Python/Rust public-surface operation inventory."""

from __future__ import annotations

import argparse
import ast
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
PYTHON_SYSCALLS = ROOT / "kogwistar" / "server" / "syscall_api.py"
RUST_SYSCALLS = RUST_API / "sections" / "response_policy.rs"


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


def _python_syscall_operations(source: str) -> list[str]:
    module = ast.parse(source)
    router = next(
        node
        for node in ast.walk(module)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "create_syscall_router"
    )
    listing = next(
        node
        for node in ast.walk(router)
        if isinstance(node, ast.FunctionDef) and node.name == "list_syscalls"
    )
    returned = next(
        node.value
        for node in ast.walk(listing)
        if isinstance(node, ast.Return) and node.value is not None
    )
    if not isinstance(returned, ast.Dict):
        raise ValueError("Python syscall catalog must return a dictionary")
    for key, value in zip(returned.keys, returned.values, strict=True):
        if isinstance(key, ast.Constant) and key.value == "ops":
            if not isinstance(value, ast.List) or not all(
                isinstance(item, ast.Constant) and isinstance(item.value, str)
                for item in value.elts
            ):
                raise ValueError("Python syscall catalog ops must be string literals")
            return [item.value for item in value.elts]
    raise ValueError("could not find Python syscall catalog ops")


def _rust_syscall_operations(source: str) -> list[str]:
    match = re.search(r"const SYSCALL_OPS:.*?= &\[(.*?)\];", source, flags=re.DOTALL)
    if match is None:
        raise ValueError("could not find Rust syscall catalog SYSCALL_OPS")
    return re.findall(r'"([a-z][a-z0-9_]*)"', match.group(1))


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
    python_syscalls = _python_syscall_operations(
        PYTHON_SYSCALLS.read_text(encoding="utf-8")
    )
    rust_syscalls = _rust_syscall_operations(
        RUST_SYSCALLS.read_text(encoding="utf-8")
    )
    python_syscall_set = set(python_syscalls)
    rust_syscall_set = set(rust_syscalls)

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
            "syscalls": "Advertised Python and Rust syscall catalogs are extracted from their public declarations. Backend support is a separate inventory; pending names do not imply that no partial implementation exists.",
        },
        "counts": {
            "http_operations": len(http_operations),
            "mcp_operations": len(mcp_operations),
            "pending_http_operations": sum(item["rust"] == "pending" for item in http_operations),
            "allowlisted_mcp_operations": sum(item["rust"] == "allowlisted-subject-to-runtime-gates" for item in mcp_operations),
            "pending_syscalls": len(pending_syscalls),
            "python_syscalls": len(python_syscalls),
            "rust_syscalls": len(rust_syscalls),
        },
        "http_operations": http_operations,
        "mcp_operations": mcp_operations,
        "pending_syscalls": pending_syscalls,
        "syscall_catalogs": {
            "python": {
                "source": "kogwistar/server/syscall_api.py#create_syscall_router.list_syscalls",
                "operations": python_syscalls,
            },
            "rust": {
                "source": "rust/crates/kogwistar-api/src/sections/response_policy.rs#SYSCALL_OPS",
                "operations": rust_syscalls,
            },
            "python_only": sorted(python_syscall_set - rust_syscall_set),
            "rust_only": sorted(rust_syscall_set - python_syscall_set),
        },
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
