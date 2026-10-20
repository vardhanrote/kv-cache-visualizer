# KVCache Lab — Experiment Log

## Experiment 1: Baseline

**Objective:** Observe memory allocation under a fixed memory budget.

- Memory budget: 64 slots
- Block size: 8 tokens
- Strategy: PagedAttention-style blocks
- Workloads: Short query, long document, reasoning

### Observations

- Memory utilization:
- Occupied slots:
- Requests completed:
- Tokens served:
- Evicted blocks:

### Interpretation

Describe how the configured memory budget compares with the total cache demand. Record any unexpected results and distinguish simulated behavior from real GPU performance.

### Limitations

The simulator models allocation behavior. It does not run an LLM or measure actual GPU latency, throughput or answer quality.
