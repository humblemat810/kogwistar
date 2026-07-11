from kogwistar.runtime.pricing import TokenPricing, estimate_token_cost_usd


def test_estimate_token_cost_uses_cached_input_rate() -> None:
    amount, status = estimate_token_cost_usd(
        {
            "input_tokens": 1000,
            "cached_input_tokens": 400,
            "output_tokens": 100,
        },
        TokenPricing(
            input_per_1k=0.001,
            cached_input_per_1k=0.0001,
            output_per_1k=0.002,
            source="test",
        ),
    )

    assert amount == 0.00084
    assert status == "estimated"


def test_estimate_token_cost_marks_missing_rate_as_partial() -> None:
    result = estimate_token_cost_usd(
        {"input_tokens": 100, "output_tokens": 20},
        TokenPricing(input_per_1k=0.001, source="test"),
    )

    assert result == (0.0001, "estimated_partial")
