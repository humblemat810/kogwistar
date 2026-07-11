from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class TokenPricing:
    """Per-1K-token rates used for an explicitly labelled cost estimate."""

    input_per_1k: float | None = None
    output_per_1k: float | None = None
    cached_input_per_1k: float | None = None
    source: str = "unavailable"

    def __post_init__(self) -> None:
        for name, rate in (
            ("input_per_1k", self.input_per_1k),
            ("cached_input_per_1k", self.cached_input_per_1k),
            ("output_per_1k", self.output_per_1k),
        ):
            if rate is not None and rate < 0:
                raise ValueError(f"{name} must be non-negative")

    @property
    def configured(self) -> bool:
        return any(
            rate is not None
            for rate in (
                self.input_per_1k,
                self.output_per_1k,
                self.cached_input_per_1k,
            )
        )


def estimate_token_cost_usd(
    usage: Mapping[str, float],
    pricing: TokenPricing,
) -> tuple[float, str] | None:
    """Estimate cost from token dimensions without pretending it is provider billing.

    ``cached_input_tokens`` are charged at the cached-input rate and removed
    from ordinary input tokens. Missing rates produce ``estimated_partial``.
    """

    input_tokens = max(0.0, float(usage.get("input_tokens") or 0.0))
    cached_tokens = min(
        input_tokens,
        max(0.0, float(usage.get("cached_input_tokens") or 0.0)),
    )
    components: list[float] = []
    has_missing_rate = False
    for tokens, rate in (
        (input_tokens - cached_tokens, pricing.input_per_1k),
        (cached_tokens, pricing.cached_input_per_1k or pricing.input_per_1k),
        (max(0.0, float(usage.get("output_tokens") or 0.0)), pricing.output_per_1k),
    ):
        if tokens <= 0:
            continue
        if rate is None:
            has_missing_rate = True
            continue
        components.append(tokens / 1000.0 * rate)
    if not components:
        return None
    status = "estimated_partial" if has_missing_rate else "estimated"
    return round(sum(components), 8), status
