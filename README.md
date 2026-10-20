# KVCache Lab — Interactive LLM Memory Simulator

KVCache Lab is an interactive educational application built with Streamlit to demonstrate simplified KV-cache memory allocation during Large Language Model (LLM) inference.

The application simulates multiple NLP-style requests competing for a limited memory budget. It visualizes memory allocation, displays request-level outcomes, explains allocation and eviction events, and compares three allocation strategies.

The project aims to make concepts related to KV-cache management and efficient LLM serving easier to understand through an interactive visual demonstration.

## 1. Project Objectives

- Understand the role of KV-cache memory during LLM inference.
- Visualize memory allocation using simplified memory slots and blocks.
- Explore how request sizes affect memory usage.
- Compare traditional reservation, PagedAttention-style block allocation, and adaptive selective eviction.
- Observe how memory budgets, block sizes, and eviction thresholds affect simulated outcomes.
- Understand the trade-offs involved in allocating limited memory among multiple requests.

## 2. Features

- **Interactive controls:** Configure the memory budget, tokens per block, allocation strategy, and eviction threshold.
- **Multiple workloads:** Simulate short user queries, long document summaries, and multi-step reasoning tasks.
- **Memory block map:** Visualize occupied and available memory slots.
- **Request-level outcomes:** Inspect requested tokens, served tokens, allocated blocks, request status, and evicted tokens.
- **Performance-oriented metrics:** View simulated memory utilization, occupied slots, completed requests, and evicted blocks.
- **Memory usage chart:** Compare requested and served tokens across requests.
- **Strategy comparison:** Compare the simulated outcomes of all three allocation strategies under the same workload settings.
- **Diagnostic explanations:** Read explanations of allocation, eviction, and final request states.
- **Automated tests:** Run the test suite to validate important simulator behaviors.

## 3. Allocation Strategies

### 3.1 Traditional Reservation

Models a simplified reservation-based allocation approach in which memory capacity is reserved according to the simulator's allocation rules.

This strategy provides a baseline for exploring how reservation policies can affect memory usage.

### 3.2 PagedAttention-Style Block Allocation

Models allocation using fixed-size memory blocks rather than treating each request as requiring one contiguous reservation.

The simplified implementation helps demonstrate block-based allocation and memory utilization concepts associated with PagedAttention.

It is an educational abstraction, not a reproduction of the complete PagedAttention algorithm or vLLM runtime.

### 3.3 Adaptive Selective Eviction

Adds an educational eviction policy that can release existing allocations when simulated memory pressure reaches a configurable threshold.

The policy illustrates the trade-off between retaining request context and making memory available for subsequent requests.

It does not model exact attention semantics, guarantee answer quality, or reproduce a published production eviction policy.

## 4. System Architecture

The application uses a simple separation between the user interface and simulation logic.

```text
+--------------------------------------+
|          Streamlit Interface         |
|                                      |
| Controls, Metrics, Tables and Charts |
+-------------------+------------------+
                    |
                    v
+--------------------------------------+
|            app.py                    |
|                                      |
| Reads controls and displays results  |
+-------------------+------------------+
                    |
                    v
+--------------------------------------+
|          simulator.py                |
|                                      |
| Allocation strategies                |
| Memory-slot management               |
| Request outcomes and explanations    |
| Result summaries and memory grid     |
+-------------------+------------------+
                    |
                    v
+--------------------------------------+
|       Simulation Results             |
|                                      |
| Occupancy, utilization, outcomes,    |
| eviction information and explanations|
+--------------------------------------+
```

### Main Components

- **`app.py`** — Provides the Streamlit interface, interactive controls, metrics, memory visualization, tables, charts, and strategy comparison.
- **`simulator.py`** — Implements the simplified allocation strategies, request processing, memory-slot representation, eviction behavior, and result summaries.
- **`tests/test_simulator.py`** — Contains automated tests for request outcomes, utilization bounds, small-budget behavior, capacity constraints, strategy differences, eviction accounting, and memory-grid size.
- **`requirements.txt`** — Lists the Python dependencies required to run the application.

## 5. Technology Stack

- **Python** — Simulation logic and application development.
- **Streamlit** — Interactive web application interface.
- **Pandas** — Tabular presentation and transformation of simulation results.
- **Plotly Express** — Interactive memory-usage and strategy-comparison charts.
- **Pytest** — Automated testing of simulator behavior.

## 6. Project Structure

```text
KVCache_Lab_Starter/
|
|-- app.py
|-- simulator.py
|-- requirements.txt
|-- README.md
|
|-- tests/
|   |-- test_simulator.py
|
|-- .venv/
```

The `.venv` directory is created locally when setting up the Python environment. It does not need to be committed to version control.

## 7. Requirements

- Python 3.10 or newer.
- Visual Studio Code or another Python-compatible editor.
- A terminal with access to Python and pip.
- The dependencies listed in `requirements.txt`.

## 8. Installation and Execution

### Step 1: Open the Project

Open the `KVCache_Lab_Starter` folder in Visual Studio Code.

### Step 2: Create a Virtual Environment

In the VS Code terminal, run:

```powershell
py -m venv .venv
```

### Step 3: Activate the Environment

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If the environment is already activated, you can skip this step.

### Step 4: Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

### Step 5: Launch the Application

```powershell
python -m streamlit run app.py
```

Streamlit will provide a local URL, usually `http://localhost:8501`, which you can open in your browser.

## 9. Running the Tests

With the virtual environment activated, execute:

```powershell
python -m pytest -v
```

The current project test suite contains seven tests covering:

1. Request outcomes across allocation strategies.
2. Bounded memory utilization.
3. Behavior with a small memory budget.
4. Default workload capacity constraints.
5. Differences between allocation strategies.
6. Correct accounting for evicted requests.
7. Consistency between the configured budget and memory grid.

**Recorded validation result:** All seven tests passed in 0.78 seconds during the development checks.

Run the command again after making code changes to verify the current version.

## 10. Simulation Parameters and Metrics

### Configurable Parameters

- **Memory budget:** Total number of simulated memory slots.
- **Tokens per block:** Number of tokens represented by a block in the simplified model.
- **Allocation strategy:** Selects the allocation policy to simulate.
- **Eviction trigger:** Sets the memory-pressure threshold used by the adaptive strategy.
- **Workload selection:** Determines which request types participate in the simulation.

### Displayed Metrics

- **Memory utilization:** Percentage of the configured budget currently occupied, according to the simulator's accounting.
- **Occupied slots:** Number of occupied slots relative to the configured budget.
- **Requests completed:** Number of requests classified as completed by the simulator.
- **Evicted blocks:** Number of blocks released through the simulated eviction process.
- **Request-level outcomes:** Per-request allocation, served-token, status, and eviction information.

These metrics describe the simulated model. They are not measurements of real GPU performance.

## 11. Simulation Assumptions and Limitations

- A memory slot is an abstract allocation unit, not a byte count or a physical GPU memory measurement.
- The application does not load or execute an LLM.
- The simulator does not reproduce vLLM internals or the complete PagedAttention implementation.
- Allocation strategies are simplified for educational comparison.
- Adaptive selective eviction is an educational extension and does not model exact attention semantics or guarantee preservation of answer quality.
- Simulated token counts and memory utilization do not establish real-world latency, throughput, memory savings, or model-quality improvements.
- Results depend on the assumptions and rules implemented in `simulator.py`.

## 12. Research Background

The project is conceptually motivated by research on memory-efficient LLM serving, KV-cache management, and PagedAttention.

A relevant starting point is the PagedAttention and vLLM paper:

Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J., Zhang, H., and Stoica, I. (2023). _Efficient Memory Management for Large Language Model Serving with PagedAttention._ Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles (SOSP 2023).

Paper: https://arxiv.org/abs/2309.06180

The paper should be cited in the academic report when discussing the original PagedAttention design. Additional research on KV-cache eviction, compression, prefix caching, and offloading may inform future extensions.

KVCache Lab is an educational simulator inspired by these concepts, not a reproduction or benchmark of the published system.

## 13. Future Enhancements

Potential extensions include:

- More detailed fragmentation and memory-allocation analysis.
- Additional workload patterns and request-arrival models.
- Comparison of cache eviction and compression policies.
- Prefix-sharing and cache-reuse demonstrations.
- Request timelines and allocation-event visualization.
- Optional integration with an actual LLM-serving backend for measured experiments.

These are potential future improvements and are not claimed as implemented features.

## 14. Conclusion

KVCache Lab provides an interactive way to explore simplified KV-cache allocation concepts during LLM inference. By combining configurable workloads, visual memory allocation, request-level outcomes, and strategy comparisons, it helps learners understand the challenges of managing limited memory in LLM-serving systems.

Its results are intended for conceptual learning and experimentation rather than production performance evaluation.
