import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import plotly.express as px

# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------
st.set_page_config(page_title="EQI Prediction (Machine Learning)", layout="wide")
st.title("🤖 Education Quality Index Prediction Using Machine Learning")

st.markdown("""
This page uses a **Random Forest Regressor** model to predict the **Education Quality Index (EQI)**  
based on key educational and infrastructure indicators at the **district level**.

The goal is to understand:
- Which features most influence EQI  
- How well we can predict EQI from school-level indicators  
""")

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/district_summary.csv")
    return df

df = load_data()

# Features for prediction
features = ["avg_student_teacher_ratio", "avg_infra_score", "pre_primary_ratio"]
target = "edu_quality_index"

X = df[features]
y = df[target]

# -------------------------------------------------
# TRAIN–TEST SPLIT
# -------------------------------------------------
test_size = st.slider("Select Test Data Size", 0.1, 0.5, 0.2)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

# -------------------------------------------------
# TRAIN MODEL
# -------------------------------------------------
n_estimators = st.slider("Number of Trees in Random Forest", 50, 300, 100, 50)
max_depth = st.slider("Max Depth of Trees", 2, 20, 8)

model = RandomForestRegressor(
    n_estimators=n_estimators,
    max_depth=max_depth,
    random_state=42
)
model.fit(X_train, y_train)

# -------------------------------------------------
# EVALUATION
# -------------------------------------------------
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

st.write("### 📊 Model Performance")
col1, col2 = st.columns(2)
col1.metric("R² Score (Accuracy)", f"{r2:.2f}")
col2.metric("Mean Absolute Error", f"{mae:.3f}")

# -------------------------------------------------
# FEATURE IMPORTANCE
# -------------------------------------------------
st.write("### 🧠 Feature Importance in Predicting EQI")
feat_imp = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=True)

fig_imp = px.bar(
    feat_imp,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Feature Importance",
    color="Importance",
    color_continuous_scale="YlGn",
)
st.plotly_chart(fig_imp, config={"responsive": True}, use_container_width=True)

st.markdown("""
✅ **Interpretation:**
- Higher importance means the feature strongly influences the EQI.
- Typically, *Infrastructure Score* and *Student–Teacher Ratio* have high influence.
""")

# -------------------------------------------------
# SCATTER COMPARISON
# -------------------------------------------------
st.write("### 🔍 Predicted vs Actual EQI")

scatter_df = pd.DataFrame({
    "Actual EQI": y_test,
    "Predicted EQI": y_pred
})
fig_pred = px.scatter(
    scatter_df,
    x="Actual EQI",
    y="Predicted EQI",
    trendline="ols",
    title="Predicted vs Actual EQI (Model Fit)",
    labels={"x": "Actual EQI", "y": "Predicted EQI"},
)
st.plotly_chart(fig_pred, config={"responsive": True}, use_container_width=True)

# -------------------------------------------------
# USER INPUT PREDICTION TOOL
# -------------------------------------------------
st.write("### 🧮 Try Predicting EQI for a Custom District")

col1, col2, col3 = st.columns(3)
user_ratio = col1.number_input("Student–Teacher Ratio", 5.0, 100.0, 30.0)
user_infra = col2.number_input("Infrastructure Score", 0.0, 2.0, 0.6)
user_preprimary = col3.number_input("Pre-Primary Ratio", 0.0, 1.0, 0.3)

user_df = pd.DataFrame({
    "avg_student_teacher_ratio": [user_ratio],
    "avg_infra_score": [user_infra],
    "pre_primary_ratio": [user_preprimary]
})

user_pred = model.predict(user_df)[0]
st.success(f"Predicted Education Quality Index (EQI): **{user_pred:.3f}**")

st.caption("🔍 Lower EQI = lower quality, closer to 1 = higher quality district")

# -------------------------------------------------
# INSIGHTS SECTION
# -------------------------------------------------
st.markdown("""
---
### 💡 Key Insights:
- The model demonstrates how **infrastructure and staffing quality** drive EQI.
- The **R² score** indicates how much of the EQI variation is explained by these features.
- You can **tune model parameters** (trees, depth, test split) to optimize performance.
- Future versions could include **temporal analysis** or **state-level feature engineering**.
---
""")
