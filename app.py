
import streamlit as st
import pandas as pd
import plotly.express as px
from simulator import (
    DEFAULT_REQUESTS,
    simulate_strategy,
    build_memory_grid,
    strategy_labels,
    summarize_results,
)

st.set_page_config(
    page_title="KVCache Lab",
    page_icon="▦",
    layout="wide",
)

st.markdown("""
<style>
.stApp {
    background: #0b1220;
    color: #e6edf7;
}
[data-testid="stSidebar"] {
    background: #111c2e;
}
h1, h2, h3 {
    color: #f4f7fb;
}
.kicker {
    color: #65d6c5;
    font-size: .8rem;
    font-weight: 700;
    letter-spacing: .12em;
}
.subtle {
    color: #a8b7cc;
}
.metric-card {
    background: #132239;
    border: 1px solid #263b56;
    padding: 1rem;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="kicker">NLP SYSTEMS · MEMORY MANAGEMENT</div>',
    unsafe_allow_html=True,
)
st.title("KVCache Lab")
st.markdown(
    '<p class="subtle">An interactive simulator for understanding '
    'KV-cache allocation during LLM inference.</p>',
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Simulation controls
# --------------------------------------------------

with st.sidebar:
    st.header("Simulation controls")

    budget = st.slider(
        "GPU memory budget (slots)",
        16, 160, 64, step=16,
    )

    block_size = st.select_slider(
        "Tokens per block",
        options=[4, 8, 16],
        value=8,
    )

    strategy = st.selectbox(
        "Allocation strategy",
        strategy_labels(),
    )

    eviction_threshold = st.slider(
        "Eviction trigger (% full)",
        50, 95, 80, step=5,
    )

    st.caption(
        "All values are illustrative simulation units, "
        "not actual GPU measurements."
    )

    st.divider()
    st.subheader("Workload")

    use_short = st.checkbox("Short user query", True)
    use_long = st.checkbox("Long document summary", True)
    use_reason = st.checkbox("Multi-step reasoning", True)

    requests = [
        request
        for request, enabled in zip(
            DEFAULT_REQUESTS,
            [use_short, use_long, use_reason],
        )
        if enabled
    ]

    if not requests:
        st.warning("Select at least one workload.")
        st.stop()

# --------------------------------------------------
# Run simulation
# --------------------------------------------------

result = simulate_strategy(
    requests=requests,
    budget=budget,
    block_size=block_size,
    strategy=strategy,
    eviction_threshold=eviction_threshold / 100,
)

summary = summarize_results(result, budget)

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Memory utilization",
    f"{summary['utilization']:.1f}%",
)
m2.metric(
    "Occupied slots",
    f"{summary['occupied']}/{budget}",
)
m3.metric(
    "Requests completed",
    f"{summary['completed']}/{len(requests)}",
)
m4.metric(
    "Evicted blocks",
    summary["evicted_blocks"],
)

# --------------------------------------------------
# Memory block map
# --------------------------------------------------

st.subheader("Memory block map")
st.caption(
    f"Strategy: {strategy} · Each slot represents "
    "a simplified allocation unit."
)

grid = build_memory_grid(result, budget)
cols = st.columns(8)

for i, slot in enumerate(grid):
    with cols[i % 8]:
        color = {
            "free": "#1b2b42",
            "short": "#367fa0",
            "long": "#278a7b",
            "reasoning": "#c58c35",
            "evicted": "#7a526d",
            "reserved": "#56647a",
        }.get(slot["kind"], "#1b2b42")

        label = (
            "·"
            if slot["kind"] == "free"
            else str(slot["slot"] + 1)
        )

        st.markdown(
            f'<div title="{slot["kind"]}" '
            f'style="background:{color};'
            f'border:1px solid #344b66;'
            f'border-radius:6px;height:38px;'
            f'display:flex;align-items:center;'
            f'justify-content:center;font-size:11px;'
            f'color:#f6f8fc;margin-bottom:6px">'
            f'{label}</div>',
            unsafe_allow_html=True,
        )

legend = st.columns(5)

for col, (label, color) in zip(
    legend,
    [
        ("Free", "#1b2b42"),
        ("Short query", "#367fa0"),
        ("Long document", "#278a7b"),
        ("Reasoning", "#c58c35"),
        ("Evicted / released", "#7a526d"),
    ],
):
    col.markdown(
        f'<span style="color:{color}">■</span> {label}',
        unsafe_allow_html=True,
    )

# --------------------------------------------------

# --------------------------------------------------
# Request table and memory usage chart
# --------------------------------------------------

df = pd.DataFrame(result["requests"])

st.subheader("Request-level outcome")

if not df.empty:
    display_columns = [
        "name",
        "workload",
        "tokens_requested",
        "tokens_served",
        "blocks_allocated",
        "status",
        "evicted_tokens",
    ]

    display_df = df[display_columns].rename(
        columns={
            "name": "Request",
            "workload": "Workload",
            "tokens_requested": "Tokens requested",
            "tokens_served": "Tokens served",
            "blocks_allocated": "Blocks allocated",
            "status": "Outcome",
            "evicted_tokens": "Tokens evicted",
        }
    )

    def style_outcome(value):
        styles = {
            "Completed": (
                "color: #4cc9b0; font-weight: bold"
            ),
            "Partial": (
                "color: #f4c76b; font-weight: bold"
            ),
            "Rejected": (
                "color: #ff7b72; font-weight: bold"
            ),
            "Evicted": (
                "color: #d8a0c2; font-weight: bold"
            ),
            "Pending": (
                "color: #a8b7cc; font-weight: bold"
            ),
        }
        return styles.get(value, "")

    styled_df = display_df.style.map(
        style_outcome,
        subset=["Outcome"],
    )

    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True,
        height=190,
    )

st.subheader("Memory usage by request")

if not df.empty:
    chart_df = df[
        ["name", "tokens_served", "tokens_requested"]
    ].melt(
        id_vars="name",
        var_name="Measure",
        value_name="Tokens",
    )

    fig = px.bar(
        chart_df,
        x="name",
        y="Tokens",
        color="Measure",
        barmode="group",
        color_discrete_map={
            "tokens_served": "#4cc9b0",
            "tokens_requested": "#6887b5",
        },
        labels={
            "name": "Request",
            "Tokens": "Tokens",
            "Measure": "Measurement",
        },
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e6edf7",
        margin=dict(l=20, r=20, t=30, b=20),
        legend_title_text="",
        xaxis_title="",
        yaxis_title="Tokens",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# --------------------------------------------------
# Simulation explanations
# --------------------------------------------------

st.subheader("What this run tells you")

for note in result["explanations"]:
    st.markdown(f"- {note}")

# --------------------------------------------------
# Compare all strategies
# --------------------------------------------------

with st.expander(
    "Compare all strategies on the same workload"
):
    comparison = []

    for s in strategy_labels():
        r = simulate_strategy(
            requests,
            budget,
            block_size,
            s,
            eviction_threshold / 100,
        )

        comparison.append({
            "Strategy": s,
            **summarize_results(r, budget),
        })

    comp = pd.DataFrame(comparison)

    st.dataframe(
        comp,
        use_container_width=True,
        hide_index=True,
    )

    fig2 = px.bar(
        comp,
        x="Strategy",
        y="utilization",
        color="Strategy",
        title="Memory utilization by strategy",
    )

    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e6edf7",
        showlegend=False,
        yaxis_title="Utilization (%)",
    )

    st.plotly_chart(
        fig2,
        use_container_width=True,
    )

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Educational simulator only. It models allocation behavior; "
    "it does not run a language model, measure real GPU latency, "
    "or establish answer quality."
)
