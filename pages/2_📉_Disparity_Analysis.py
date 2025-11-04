import streamlit as st
import pandas as pd
import plotly.express as px

# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------
st.set_page_config(page_title="Educational Disparity Analysis", layout="wide")
st.title("📉 Mapping Educational Disparities (District & State Level)")

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/district_summary.csv")
    df["district_name"] = df["district_name"].astype(str).str.strip().str.title()
    df["state_name"] = df["state_name"].astype(str).str.strip().str.title()
    return df

df = load_data()

# Compute missing ratios if needed
if "avg_student_teacher_ratio" not in df.columns and "class_students" in df.columns:
    df["avg_student_teacher_ratio"] = df["class_students"] / df["total_teachers"]

# -------------------------------------------------
# INTRODUCTION
# -------------------------------------------------
st.markdown("""
### 🧠 About this Page
This dashboard visualizes **educational disparities** across Indian districts using:
- 👩‍🏫 *Student–Teacher Ratio*  
- 🏫 *Infrastructure Score*  
- 🌍 *Education Quality Index (EQI)*  

Each analysis helps identify **resource gaps** and **performance variations** among states and districts.
""")

# -------------------------------------------------
# NATIONAL METRICS
# -------------------------------------------------
st.subheader("📈 National Education Metrics Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Avg Student–Teacher Ratio", f"{df['avg_student_teacher_ratio'].mean():.1f}")
col2.metric("Avg Infrastructure Score", f"{df['avg_infra_score'].mean():.2f}")
col3.metric("Avg Education Quality Index (EQI)", f"{df['edu_quality_index'].mean():.2f}")
st.divider()

# -------------------------------------------------
# 🎯 DISTRIBUTION OF STUDENT–TEACHER RATIOS
# -------------------------------------------------
st.subheader("🎯 Distribution of Student–Teacher Ratios (All Districts)")
fig_dist = px.histogram(
    df,
    x="avg_student_teacher_ratio",
    nbins=50,
    title="Distribution of Student–Teacher Ratio Across Districts",
    labels={"avg_student_teacher_ratio": "Student–Teacher Ratio"},
    color_discrete_sequence=["#2E91E5"]
)
st.plotly_chart(fig_dist, config={"responsive": True}, width='stretch')

# -------------------------------------------------
# 🏫 INFRASTRUCTURE SCORE BY STATE
# -------------------------------------------------
st.subheader("🏫 Infrastructure Score Comparison by State")
fig_box = px.box(
    df,
    x="state_name",
    y="avg_infra_score",
    points="all",
    title="Infrastructure Score Distribution by State",
    labels={"avg_infra_score": "Infrastructure Score", "state_name": "State"},
    color_discrete_sequence=["#E15F99"]
)
st.plotly_chart(fig_box, config={"responsive": True}, width='stretch')
st.divider()

# -------------------------------------------------
# 📊 DISTRICT PERFORMANCE VISUALIZATION (Dynamic Bubble Chart)
# -------------------------------------------------
st.subheader("📊 District-Level Educational Performance")

metric_options = {
    "Student–Teacher Ratio": "avg_student_teacher_ratio",
    "Infrastructure Score": "avg_infra_score",
    "Education Quality Index (EQI)": "edu_quality_index"
}
metric_label = st.selectbox("Select Primary Metric for Comparison", options=list(metric_options.keys()))
metric = metric_options[metric_label]

st.markdown("""
This chart dynamically compares **districts** across major education indicators:
- 🧱 *Infrastructure Score*  
- 🧑‍🏫 *Student–Teacher Ratio*  
- 🌍 *Education Quality Index (EQI)*  
- ⚪ *Bubble Size:* Number of Schools  
""")

# Dynamic configuration based on metric selected
if metric_label == "Student–Teacher Ratio":
    x_col, y_col, color_col = "avg_infra_score", "edu_quality_index", "avg_student_teacher_ratio"
    title = "📘 Student–Teacher Ratio Focus: Relation with Infrastructure & EQI"
elif metric_label == "Infrastructure Score":
    x_col, y_col, color_col = "avg_student_teacher_ratio", "edu_quality_index", "avg_infra_score"
    title = "🏫 Infrastructure Focus: Relation with Student–Teacher Ratio & EQI"
else:  # EQI
    x_col, y_col, color_col = "avg_infra_score", "avg_student_teacher_ratio", "edu_quality_index"
    title = "🌍 Education Quality Index Focus: Relation with Infrastructure & Ratio"

# Bubble chart
fig_bubble = px.scatter(
    df,
    x=x_col,
    y=y_col,
    size="num_schools",
    color=color_col,
    hover_name="district_name",
    hover_data=["state_name"],
    color_continuous_scale="Viridis",
    title=title,
    labels={
        x_col: x_col.replace("_", " ").title(),
        y_col: y_col.replace("_", " ").title(),
        color_col: color_col.replace("_", " ").title(),
    }
)
st.plotly_chart(fig_bubble, config={"responsive": True}, width='stretch')
st.caption("🟢 Larger bubbles = More schools | 💡 Color intensity = Higher value of the selected metric")
st.divider()

# -------------------------------------------------
# 🏆 TOP AND BOTTOM 10 DISTRICTS BY METRIC
# -------------------------------------------------
st.subheader(f"🏆 Top & Bottom 10 Districts by {metric_label}")

top10 = df.nlargest(10, metric)
bottom10 = df.nsmallest(10, metric)

col1, col2 = st.columns(2)

with col1:
    st.write(f"### 🌟 Top 10 Districts (Best in {metric_label})")
    fig_top = px.bar(
        top10,
        x="district_name",
        y=metric,
        color="state_name",
        title=f"Top 10 Districts by {metric_label}",
        labels={"district_name": "District", metric: metric_label, "state_name": "State"},
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    st.plotly_chart(fig_top, config={"responsive": True}, width='stretch')

with col2:
    st.write(f"### ⚠️ Bottom 10 Districts (Needs Improvement)")
    fig_bottom = px.bar(
        bottom10,
        x="district_name",
        y=metric,
        color="state_name",
        title=f"Bottom 10 Districts by {metric_label}",
        labels={"district_name": "District", metric: metric_label, "state_name": "State"},
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    st.plotly_chart(fig_bottom, config={"responsive": True}, width='stretch')
st.divider()

# -------------------------------------------------
# ⚖️ INEQUALITY ANALYSIS (Intra-State)
# -------------------------------------------------
st.subheader("⚖️ Intra-State Variation in Education Quality")

ineq_df = df.groupby("state_name", as_index=False).agg({
    "edu_quality_index": "std"
}).rename(columns={"edu_quality_index": "EQI Variation"})

fig_ineq = px.bar(
    ineq_df.sort_values("EQI Variation", ascending=False),
    x="state_name",
    y="EQI Variation",
    title="Variation in Education Quality Index Within States",
    labels={"state_name": "State", "EQI Variation": "Standard Deviation"},
    color_discrete_sequence=["#00C49F"]
)
st.plotly_chart(fig_ineq, config={"responsive": True}, width='stretch')

st.markdown("""
🔹 **Higher bars** → greater disparity among districts  
🔹 **Lower bars** → more uniform educational standards across districts  
---
""")

# -------------------------------------------------
# 🔍 STATE COMPARISON TOOL
# -------------------------------------------------
st.subheader("🔍 Compare Two States by Education Quality Index")

states = sorted(df["state_name"].unique())
col1, col2 = st.columns(2)
state_a = col1.selectbox("Select State A", states, index=0)
state_b = col2.selectbox("Select State B", states, index=1)

comp_df = df[df["state_name"].isin([state_a, state_b])]

fig_compare = px.box(
    comp_df,
    x="state_name",
    y="edu_quality_index",
    color="state_name",
    points="all",
    title=f"Education Quality Index Distribution: {state_a} vs {state_b}",
    labels={"edu_quality_index": "Education Quality Index", "state_name": "State"},
    color_discrete_sequence=px.colors.qualitative.Set2
)
st.plotly_chart(fig_compare, config={"responsive": True}, width='stretch')



st.markdown("""
✅ **Insights:**
- The **median line** represents the typical EQI in each state.  
- **Wider boxes** indicate higher intra-state variation.  
- **Outliers** represent exceptionally high or low-performing districts.
---
""")

st.success("✅ Disparity Analysis Complete — Explore and Compare Educational Patterns Across India!")
