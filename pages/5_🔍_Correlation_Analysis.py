import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📈 Correlation Analysis")

df = pd.read_csv("data/district_summary.csv")
num_cols = df.select_dtypes("number").columns.tolist()

selected_cols = st.multiselect("Select columns to analyze", num_cols,
                               default=["avg_student_teacher_ratio", "avg_infra_score", "pre_primary_ratio", "edu_quality_index"])

if len(selected_cols) >= 2:
    corr = df[selected_cols].corr()
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", title="Feature Correlation Heatmap")
    st.plotly_chart(fig, config={"responsive": True}, width='stretch')
else:
    st.warning("Please select at least two columns.")
