"""Normalize pytest JUnit cases into portable gate evidence."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


def read_junit_test_results(path: Path) -> list[dict[str, str]]:
    """Return one named outcome per pytest case, including parameterized cases."""
    if not path.is_file():
        return []

    results: list[dict[str, str]] = []
    for case in ET.parse(path).iter("testcase"):
        status = "passed"
        if case.find("failure") is not None or case.find("error") is not None:
            status = "failed"
        elif case.find("skipped") is not None:
            status = "skipped"
        results.append(
            {
                "source": "pytest",
                "classname": case.attrib.get("classname", ""),
                "name": case.attrib.get("name", ""),
                "function": case.attrib.get("name", "").split("[", 1)[0],
                "status": status,
            }
        )
    return results


def junit_result_counts(results: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "tests": len(results),
        "passed": sum(row.get("status") == "passed" for row in results),
        "failures": sum(row.get("status") == "failed" for row in results),
        "skipped": sum(row.get("status") == "skipped" for row in results),
    }
