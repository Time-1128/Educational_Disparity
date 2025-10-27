import streamlit as st
import pandas as pd
import plotly.express as px

# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------
st.set_page_config(page_title="District Rankings", layout="wide")
st.title("🏅 District Rankings by Education Quality Index (EQI)")

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/district_summary.csv")

df = load_data()

# -------------------------------------------------
# DISPLAY TOP AND BOTTOM DISTRICTS
# -------------------------------------------------
st.markdown("### 🌟 Top and Bottom Districts Based on Education Quality Index (EQI)")

top_districts = df.nlargest(10, "edu_quality_index")
bottom_districts = df.nsmallest(10, "edu_quality_index")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🌟 Top 10 Districts (Highest EQI)")
    st.dataframe(
        top_districts[["state_name", "district_name", "edu_quality_index"]]
        .reset_index(drop=True)
        .style.format({"edu_quality_index": "{:.3f}"})
    )

with col2:
    st.subheader("⚠️ Bottom 10 Districts (Lowest EQI)")
    st.dataframe(
        bottom_districts[["state_name", "district_name", "edu_quality_index"]]
        .reset_index(drop=True)
        .style.format({"edu_quality_index": "{:.3f}"})
    )

# -------------------------------------------------
# VISUALIZATION: TOP DISTRICTS
# -------------------------------------------------
st.markdown("---")
st.subheader("📊 Visualization: Top 10 Districts by EQI")

fig = px.bar(
    top_districts.sort_values("edu_quality_index"),
    x="edu_quality_index",
    y="district_name",
    color="state_name",
    orientation="h",
    title="Top 10 Districts by Education Quality Index",
    labels={
        "edu_quality_index": "Education Quality Index (EQI)",
        "district_name": "District",
        "state_name": "State",
    },
)
st.plotly_chart(fig, config={"responsive": True}, use_container_width=True)

# -------------------------------------------------
# VISUALIZATION: BOTTOM DISTRICTS
# -------------------------------------------------
st.subheader("📉 Visualization: Bottom 10 Districts by EQI")

fig2 = px.bar(
    bottom_districts.sort_values("edu_quality_index", ascending=True),
    x="edu_quality_index",
    y="district_name",
    color="state_name",
    orientation="h",
    title="Bottom 10 Districts by Education Quality Index",
    labels={
        "edu_quality_index": "Education Quality Index (EQI)",
        "district_name": "District",
        "state_name": "State",
    },
)
st.plotly_chart(fig2, config={"responsive": True}, use_container_width=True)

# -------------------------------------------------
# 📘 EDUCATION QUALITY INDEX (EQI) FORMULA SECTION
# -------------------------------------------------

# -------------------------------------------------
# 📈 ADDITIONAL INSIGHT SECTION
# -------------------------------------------------
st.info("""
**Interpretation:**
- Districts with higher EQI scores usually have *lower student–teacher ratios* and *better infrastructure*.
- Lower-ranked districts often face *resource shortages* or *teacher imbalances*.
- This ranking helps identify which regions need *policy attention and resource allocation*.
""")
