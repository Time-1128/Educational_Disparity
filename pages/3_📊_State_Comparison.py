import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📍 State-Level Comparison and Inequality Analysis")

df = pd.read_csv("data/district_summary.csv")

# ------------------------------
# 📊 Average EQI by State
# ------------------------------
st.write("### 🏅 Average Education Quality Index by State")

state_df = df.groupby("state_name", as_index=False).agg({
    "avg_student_teacher_ratio": "mean",
    "avg_infra_score": "mean",
    "edu_quality_index": "mean"
}).sort_values("edu_quality_index", ascending=False)

fig_state = px.bar(
    state_df,
    x="state_name", y="edu_quality_index",
    title="Average Education Quality Index (EQI) by State",
    labels={"state_name": "State", "edu_quality_index": "Education Quality Index"},
)
st.plotly_chart(fig_state, config={"responsive": True}, use_container_width=True)

# ------------------------------
# ⚖️ Inequality Within States
# ------------------------------
st.write("### ⚖️ Intra-State Inequality")

ineq_df = df.groupby("state_name", as_index=False).agg({
    "edu_quality_index": "std"
}).rename(columns={"edu_quality_index": "EQI Variation"})

fig_ineq = px.bar(
    ineq_df.sort_values("EQI Variation", ascending=False),
    x="state_name", y="EQI Variation",
    title="Variation in EQI (Standard Deviation) Within States",
    labels={"state_name": "State", "EQI Variation": "Standard Deviation"},
)
st.plotly_chart(fig_ineq, config={"responsive": True}, use_container_width=True)

# ------------------------------
# 🔍 Compare Two States
# ------------------------------
st.write("### 🔍 Compare Two States by Education Quality Index")

states = sorted(df["state_name"].unique())
col1, col2 = st.columns(2)
state_a = col1.selectbox("Select State A", states, index=0)
state_b = col2.selectbox("Select State B", states, index=1)

comp_df = df[df["state_name"].isin([state_a, state_b])]
fig_compare = px.box(
    comp_df,
    x="state_name", y="edu_quality_index",
    color="state_name", points="all",
    title=f"Education Quality Index Comparison — {state_a} vs {state_b}",
)
st.plotly_chart(fig_compare, config={"responsive": True}, use_container_width=True)

st.markdown("""
✅ **Insights**:
- The median EQI shows overall state performance.  
- Wider box = more inequality among districts.  
- Outliers indicate unusually high or low performing districts.
""")
