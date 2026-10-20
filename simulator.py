
"""
KVCache Lab — Educational KV-cache allocation simulator.

Important:
- This is a simplified teaching model, not a real LLM inference engine.
- Each memory slot represents one illustrative token-capacity unit.
- Block-based policies allocate complete blocks.
- Traditional reservation uses power-of-two reservation sizes to
  demonstrate internal fragmentation.
- Evicting a request discards its retained simulated context.
"""

from copy import deepcopy
from math import ceil, log2


DEFAULT_REQUESTS = [
    {"name": "Short query", "workload": "short", "tokens": 12},
    {"name": "Long document", "workload": "long", "tokens": 48},
    {"name": "Reasoning task", "workload": "reasoning", "tokens": 32},
]

STRATEGIES = [
    "Traditional reservation",
    "PagedAttention-style blocks",
    "Adaptive selective eviction",
]


def strategy_labels():
    """Return strategy names in dashboard display order."""
    return STRATEGIES.copy()


def _validate_inputs(requests, budget, block_size, strategy):
    if not isinstance(budget, int) or budget <= 0:
        raise ValueError("budget must be a positive integer")

    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("block_size must be a positive integer")

    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}")

    for request in requests:
        tokens = request.get("tokens")

        if not isinstance(tokens, int) or tokens < 0:
            raise ValueError(
                "Every request must have a non-negative integer 'tokens' value"
            )

        if not request.get("name"):
            raise ValueError("Every request must have a name")


def _round_to_blocks(tokens, block_size):
    """Return the number of slots required by fixed-size blocks."""
    if tokens == 0:
        return 0
    return ceil(tokens / block_size) * block_size


def _power_of_two_capacity(tokens):
    """Round a request up to a power-of-two reservation size."""
    if tokens <= 1:
        return tokens
    return 2 ** ceil(log2(tokens))


def _empty_slots(budget):
    return [
        {"slot": i, "kind": "free", "owner": None}
        for i in range(budget)
    ]


def _free_indices(slots):
    return [
        i for i, slot in enumerate(slots)
        if slot["kind"] == "free"
    ]


def _allocate(slots, request, capacity, block_size):
    """Allocate complete blocks from the currently free slots."""
    free = _free_indices(slots)
    blocks_needed = capacity // block_size
    blocks_available = len(free) // block_size
    blocks_to_allocate = min(blocks_needed, blocks_available)

    for block_number in range(blocks_to_allocate):
        start = block_number * block_size
        selected = free[start : start + block_size]

        for index in selected:
            slots[index]["kind"] = request["workload"]
            slots[index]["owner"] = request["name"]

    return blocks_to_allocate * block_size


def _release_owner(slots, owner):
    """Release all slots owned by a request."""
    released = 0

    for slot in slots:
        if slot["owner"] == owner:
            slot["kind"] = "free"
            slot["owner"] = None
            released += 1

    return released


def simulate_strategy(
    requests,
    budget=64,
    block_size=8,
    strategy="PagedAttention-style blocks",
    eviction_threshold=0.8,
):
    """
    Run a deterministic memory-allocation simulation.

    Requests compete for a shared memory snapshot and remain allocated
    unless the adaptive policy evicts them. Completed means the request
    received its full requested token capacity and has not subsequently
    been evicted. A partial allocation is not counted as completed.
    """

    _validate_inputs(requests, budget, block_size, strategy)

    if not isinstance(eviction_threshold, (int, float)):
        raise ValueError("eviction_threshold must be numeric")

    if not 0 <= eviction_threshold <= 1:
        raise ValueError("eviction_threshold must be between 0 and 1")

    slots = _empty_slots(budget)
    request_results = []
    explanations = []
    evicted_blocks = 0

    for original in deepcopy(requests):
        request_results.append({
            "name": original["name"],
            "workload": original.get("workload", "short"),
            "tokens_requested": original["tokens"],
            "tokens_served": 0,
            "blocks_allocated": 0,
            "status": "Pending",
            "evicted_tokens": 0,
            "_capacity": (
                max(
                    block_size,
                    _power_of_two_capacity(original["tokens"]),
                )
                if (
                    strategy == "Traditional reservation"
                    and original["tokens"] > 0
                )
                else (
                    _power_of_two_capacity(original["tokens"])
                    if strategy == "Traditional reservation"
                    else _round_to_blocks(
                        original["tokens"], block_size
                    )
                )
            ),
        })

    for current_index, current in enumerate(request_results):
        requested = current["tokens_requested"]
        capacity = current["_capacity"]

        if requested == 0:
            current["status"] = "Completed"
            explanations.append(
                f"{current['name']}: zero-token request; no allocation needed."
            )
            continue

        # Adaptive policy: reclaim the oldest allocations when the
        # configured memory-pressure threshold has been reached.
        if strategy == "Adaptive selective eviction":
            occupied_before = budget - len(_free_indices(slots))
            pressure = occupied_before / budget

            if pressure >= eviction_threshold:
                for previous in request_results[:current_index]:
                    if len(_free_indices(slots)) >= capacity:
                        break

                    if previous["blocks_allocated"] == 0:
                        continue

                    released = _release_owner(slots, previous["name"])
                    released_blocks = released // block_size

                    if released_blocks == 0:
                        continue

                    evicted_blocks += released_blocks
                    previous["evicted_tokens"] = previous["tokens_served"]
                    previous["tokens_served"] = 0
                    previous["blocks_allocated"] = 0
                    previous["status"] = "Evicted"

                    explanations.append(
                        f"{previous['name']}: evicted {released_blocks} "
                        "block(s); its simulated context was discarded."
                    )

        free_count = len(_free_indices(slots))

        if strategy == "Traditional reservation":
            # All-or-nothing reservation: no allocation if the full
            # reservation does not fit.
            if free_count >= capacity:
                allocated = _allocate(
                    slots, current, capacity, block_size
                )
            else:
                allocated = 0
        else:
            # Block policies allocate only complete blocks. Partial
            # allocation is possible when fewer blocks remain.
            allocated = _allocate(
                slots, current, capacity, block_size
            )

        current["blocks_allocated"] = allocated // block_size
        current["tokens_served"] = min(requested, allocated)

        if allocated >= capacity:
            current["status"] = "Completed"
            explanations.append(
                f"{current['name']}: allocated "
                f"{current['blocks_allocated']} block(s)."
            )
        elif allocated > 0:
            current["status"] = "Partial"
            explanations.append(
                f"{current['name']}: partial allocation; "
                f"{current['tokens_served']} of {requested} tokens served."
            )
        else:
            current["status"] = "Rejected"
            explanations.append(
                f"{current['name']}: insufficient available memory."
            )

    # Explain the final retained state of every request.
    for request in request_results:
        if request["status"] == "Evicted":
            explanations.append(
                f"Final state — {request['name']}: Evicted; "
                "0 tokens retained in the simulated cache."
            )
        else:
            explanations.append(
                f"Final state — {request['name']}: "
                f"{request['status']}; "
                f"{request['tokens_served']} tokens currently retained."
            )

    for request in request_results:
        request.pop("_capacity", None)

    return {
        "requests": request_results,
        "slots": slots,
        "evicted_blocks": evicted_blocks,
        "explanations": explanations,
        "block_size": block_size,
        "budget": budget,
        "strategy": strategy,
    }


def summarize_results(result, budget):
    """Calculate metrics using the final retained memory state."""
    slots = result["slots"]
    requests = result["requests"]

    occupied = sum(slot["kind"] != "free" for slot in slots)

    tokens_requested = sum(
        request["tokens_requested"] for request in requests
    )

    tokens_served = sum(
        request["tokens_served"] for request in requests
    )

    completed = sum(
        request["status"] == "Completed" for request in requests
    )

    utilization = occupied / budget * 100 if budget > 0 else 0

    return {
        "utilization": round(utilization, 2),
        "occupied": occupied,
        "completed": completed,
        "evicted_blocks": result["evicted_blocks"],
        "tokens_served": tokens_served,
        "tokens_requested": tokens_requested,
    }


def build_memory_grid(result, budget):
    """Return slot numbers and categories for the dashboard."""
    slots = result["slots"]

    return [
        {"slot": i, "kind": slots[i]["kind"]}
        for i in range(budget)
    ]
