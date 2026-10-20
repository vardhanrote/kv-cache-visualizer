
from simulator import (
    DEFAULT_REQUESTS,
    build_memory_grid,
    simulate_strategy,
    summarize_results,
    strategy_labels,
)


def test_strategies_return_request_outcomes():
    for strategy in strategy_labels():
        result = simulate_strategy(
            DEFAULT_REQUESTS,
            budget=64,
            block_size=8,
            strategy=strategy,
        )

        assert len(result["requests"]) == len(DEFAULT_REQUESTS)
        assert all(
            request["tokens_served"] >= 0
            for request in result["requests"]
        )
        assert all(
            request["tokens_served"] <= request["tokens_requested"]
            for request in result["requests"]
        )


def test_utilization_is_bounded():
    result = simulate_strategy(
        DEFAULT_REQUESTS,
        budget=64,
        block_size=8,
    )

    summary = summarize_results(result, 64)

    assert 0 <= summary["utilization"] <= 100
    assert 0 <= summary["occupied"] <= 64


def test_small_budget_does_not_crash():
    result = simulate_strategy(
        DEFAULT_REQUESTS,
        budget=16,
        block_size=8,
        strategy="Adaptive selective eviction",
    )

    assert len(result["requests"]) == 3


def test_default_workload_exceeds_memory_capacity():
    budget = 64
    block_size = 8

    requested_tokens = sum(
        request["tokens"] for request in DEFAULT_REQUESTS
    )

    required_slots = [
        (
            (request["tokens"] + block_size - 1) // block_size
        ) * block_size
        for request in DEFAULT_REQUESTS
    ]

    assert requested_tokens == 92
    assert sum(required_slots) == 96
    assert sum(required_slots) > budget


def test_strategies_have_distinct_default_outcomes():
    summaries = {}

    for strategy in strategy_labels():
        result = simulate_strategy(
            DEFAULT_REQUESTS,
            budget=64,
            block_size=8,
            strategy=strategy,
        )
        summaries[strategy] = summarize_results(result, 64)

    assert (
        summaries["Traditional reservation"]["tokens_served"]
        != summaries["PagedAttention-style blocks"]["tokens_served"]
    )

    assert (
        summaries["Adaptive selective eviction"]["evicted_blocks"] > 0
    )


def test_evicted_requests_are_not_counted_as_completed():
    result = simulate_strategy(
        DEFAULT_REQUESTS,
        budget=64,
        block_size=8,
        strategy="Adaptive selective eviction",
    )

    for request in result["requests"]:
        if request["status"] == "Evicted":
            assert request["tokens_served"] == 0

    summary = summarize_results(result, 64)

    assert summary["completed"] == sum(
        request["status"] == "Completed"
        for request in result["requests"]
    )


def test_memory_grid_matches_budget():
    result = simulate_strategy(
        DEFAULT_REQUESTS,
        budget=64,
        block_size=8,
    )

    grid = build_memory_grid(result, 64)

    assert len(grid) == 64
    assert [slot["slot"] for slot in grid] == list(range(64))
