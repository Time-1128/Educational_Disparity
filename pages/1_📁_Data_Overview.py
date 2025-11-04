import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Data Overview", layout="wide")
st.title("📊 Data Overview")

@st.cache_data
def load_data():
    return pd.read_csv("data/district_summary.csv")

df = load_data()

st.markdown("### 🗺️ Select State(s) and District(s) to Explore")

states = st.multiselect("Select State(s):", sorted(df["state_name"].unique()))

if states:
    filtered_df = df[df["state_name"].isin(states)]
    districts = sorted(filtered_df["district_name"].unique())
    selected_districts = st.multiselect("Select District(s):", districts)
    if selected_districts:
        filtered_df = filtered_df[filtered_df["district_name"].isin(selected_districts)]
else:
    filtered_df = df.copy()

st.write(f"### 📄 Showing {len(filtered_df):,} Records")
st.dataframe(filtered_df, width='stretch')

# -------------------------------------------------
# 📥 DOWNLOAD FILTERED DATA BUTTON
# -------------------------------------------------
csv = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Download Filtered Data as CSV",
    data=csv,
    file_name="filtered_district_data.csv",
    mime="text/csv",
)

# -------------------------------------------------
# VISUALIZATION — NUMBER OF SCHOOLS BY STATE / DISTRICT
# -------------------------------------------------
st.markdown("### 🏫 Number of Schools by State")

if len(states) > 1 or not states:
    state_counts = (
        filtered_df.groupby("state_name")["num_schools"]
        .sum()
        .reset_index()
        .sort_values("num_schools", ascending=False)
    )
    fig = px.bar(
        state_counts,
        x="state_name",
        y="num_schools",
        color="state_name",
        title="Total Schools by State",
    )
else:
    district_counts = (
        filtered_df.groupby("district_name")["num_schools"]
        .sum()
        .reset_index()
        .sort_values("num_schools", ascending=False)
    )
    fig = px.bar(
        district_counts,
        x="district_name",
        y="num_schools",
        color="district_name",
        title=f"Total Schools by District in {states[0]}",
    )

st.plotly_chart(fig, config={"responsive": True}, width='stretch')
