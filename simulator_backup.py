"""Transparent, simplified KV-cache allocation simulator.

This is an educational model, not a reproduction of vLLM/PagedAttention or a GPU benchmark.
"""
from copy import deepcopy

DEFAULT_REQUESTS = [
    {"name": "Short query", "workload": "short", "tokens": 12},
    {"name": "Long document", "workload": "long", "tokens": 48},
    {"name": "Reasoning", "workload": "reasoning", "tokens": 32},
]

def strategy_labels():
    return ["Traditional reservation", "PagedAttention-style blocks", "Adaptive selective eviction"]

def simulate_strategy(requests, budget=64, block_size=8,
                      strategy="PagedAttention-style blocks", eviction_threshold=0.8):
    """Simulate requests in order using a finite pool of slots."""
    requests = deepcopy(requests)
    total_blocks = max(1, budget // block_size)
    slots = [{"kind": "free", "owner": None, "slot": i} for i in range(budget)]
    results, evicted_blocks = [], 0
    occupied = 0

    for req in requests:
        requested = int(req["tokens"])
        blocks_needed = (requested + block_size - 1) // block_size
        available = [s for s in slots if s["kind"] == "free"]
        evicted_tokens = 0

        if strategy == "Traditional reservation":
            # Reserve an estimated per-request capacity up front, capped to the pool.
            reserve = min(max(blocks_needed, 1) * block_size, budget)
            if len(available) < reserve:
                status = "Could not reserve"
                served = 0
                allocated = 0
            else:
                status = "Completed" if reserve >= requested else "Partial"
                served = min(requested, reserve)
                allocated = reserve
                for s in available[:allocated]:
                    s.update(kind="reserved", owner=req["workload"])
                occupied += allocated
        else:
            needed_slots = min(blocks_needed * block_size, budget)
            available_slots = len(available)
            if available_slots < needed_slots and strategy == "Adaptive selective eviction":
                target_free = max(0, int(budget * (1 - eviction_threshold)))
                # Evict blocks from previously allocated requests, in whole block-sized groups.
                owners = ["reasoning", "long", "short"]
                for owner in owners:
                    owned = [s for s in slots if s["owner"] == owner and s["kind"] != "free"]
                    while owned and len([s for s in slots if s["kind"] == "free"]) < max(needed_slots, target_free):
                        batch = owned[:block_size]
                        if not batch:
                            break
                        for s in batch:
                            s.update(kind="free", owner=None)
                        evicted_blocks += 1
                        evicted_tokens += len(batch)
                        owned = [s for s in slots if s["owner"] == owner and s["kind"] != "free"]
                    if len([s for s in slots if s["kind"] == "free"]) >= needed_slots:
                        break
                available = [s for s in slots if s["kind"] == "free"]

            allocated = min(needed_slots, len(available))
            served = min(requested, (allocated // block_size) * block_size)
            # Preserve exact token count when the final block is only partially filled.
            if allocated >= blocks_needed * block_size:
                served = requested
            status = "Completed" if served >= requested else ("Partial" if served > 0 else "Could not allocate")
            for s in available[:allocated]:
                s.update(kind=req["workload"], owner=req["workload"])
            occupied = len([s for s in slots if s["kind"] != "free"])

        results.append({
            "name": req["name"], "workload": req["workload"],
            "tokens_requested": requested, "tokens_served": served,
            "blocks_allocated": (allocated + block_size - 1) // block_size,
            "status": status, "evicted_tokens": evicted_tokens
        })

    explanations = []
    if strategy == "Traditional reservation":
        explanations.append("Traditional reservation assigns estimated capacity before generation; unused reserved space can remain unavailable to other requests.")
    else:
        explanations.append("Block allocation grows in fixed-size units rather than reserving the entire uncertain future request at once.")
    if strategy == "Adaptive selective eviction":
        explanations.append("The optional eviction policy frees selected older blocks when the pool is under pressure; this simplified model does not preserve exact model context semantics.")
    explanations.append("A partial or rejected request indicates the configured pool could not accommodate its simulated token demand.")
    return {"requests": results, "slots": slots, "evicted_blocks": evicted_blocks,
            "explanations": explanations, "block_size": block_size, "budget": budget}

def build_memory_grid(result, budget):
    slots = result["slots"][:budget]
    return [{"slot": i, "kind": s["kind"]} for i, s in enumerate(slots)]

def summarize_results(result, budget):
    occupied = len([s for s in result["slots"][:budget] if s["kind"] != "free"])
    requests = result["requests"]
    return {
        "utilization": (occupied / budget * 100) if budget else 0,
        "occupied": occupied,
        "completed": sum(r["status"] == "Completed" for r in requests),
        "evicted_blocks": result["evicted_blocks"],
        "tokens_served": sum(r["tokens_served"] for r in requests),
        "tokens_requested": sum(r["tokens_requested"] for r in requests),
    }
