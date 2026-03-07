from __future__ import annotations

from typing import Callable, Literal

# Keep this broad in core; domain-specific kind tokens live outside engine_core.
EngineType = str
ExtractionSchemaMode = Literal["auto", "full", "lean", "flattened_lean", "flattened_full"]
ResolvedExtractionSchemaMode = Literal["full", "lean", "flattened_lean", "flattened_full"]
OffsetMismatchPolicy = Literal["strict", "exact", "exact_fuzzy"]
OffsetRepairScorer = Callable[[str, str], float]
