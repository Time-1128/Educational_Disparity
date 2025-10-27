import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import plotly.express as px

# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------
st.set_page_config(page_title="Model Comparison", layout="wide")
st.title("🤖 Model Comparison: Predicting Education Quality Index (EQI)")

st.markdown("""
This page compares **Linear Regression** and **Random Forest** models  
to predict the **Education Quality Index (EQI)** using district-level indicators.
""")

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/district_summary.csv")
    df = df.dropna(subset=["edu_quality_index", "avg_student_teacher_ratio", "avg_infra_score", "pre_primary_ratio"])
    return df

df = load_data()

# -------------------------------------------------
# FEATURE SELECTION
# -------------------------------------------------
X = df[["avg_student_teacher_ratio", "avg_infra_score", "pre_primary_ratio"]]
y = df["edu_quality_index"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------------------------
# MODEL TRAINING
# -------------------------------------------------
lr = LinearRegression()
rf = RandomForestRegressor(random_state=42, n_estimators=200)

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Predictions
y_pred_lr = lr.predict(X_test)
y_pred_rf = rf.predict(X_test)

# Evaluation
r2_lr = r2_score(y_test, y_pred_lr)
r2_rf = r2_score(y_test, y_pred_rf)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mae_rf = mean_absolute_error(y_test, y_pred_rf)

# Results summary
results_df = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "R2": [r2_lr, r2_rf],
    "MAE": [mae_lr, mae_rf]
})

# -------------------------------------------------
# VISUALIZATION
# -------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Model Accuracy (R²)")
    fig1 = px.bar(
        results_df,
        x="Model",
        y="R2",
        color="Model",
        text="R2",
        title="Model Accuracy (R²)",
    )
    fig1.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig1.update_yaxes(range=[min(results_df["R2"]) - 0.02, 1.0])  # Zoomed range
    st.plotly_chart(fig1, config={"responsive": True}, use_container_width=True)

with col2:
    st.subheader("📉 Prediction Error (Mean Absolute Error)")
    fig2 = px.bar(
        results_df,
        x="Model",
        y="MAE",
        color="Model",
        text="MAE",
        title="Prediction Error (Mean Absolute Error)",
    )
    fig2.update_traces(texttemplate='%{text:.4f}', textposition='outside')
    fig2.update_yaxes(range=[0, max(results_df["MAE"]) + 0.002])
    st.plotly_chart(fig2, config={"responsive": True}, use_container_width=True)

# -------------------------------------------------
# INSIGHTS
# -------------------------------------------------
better_r2_model = results_df.loc[results_df["R2"].idxmax(), "Model"]
better_mae_model = results_df.loc[results_df["MAE"].idxmin(), "Model"]

st.markdown("---")
st.subheader("🧠 Model Insights")
st.markdown(f"""
- ✅ **{better_r2_model}** achieved a higher R² score — indicating a better fit and stronger correlation between predicted and actual EQI values.
- 📉 **{better_mae_model}** had a lower MAE — meaning smaller prediction errors on average.
- 🧩 Linear Regression provides interpretability (clear relationships between variables).
- 🌲 Random Forest captures non-linear relationships and handles data complexity more effectively.
""")

st.markdown("---")
st.markdown("""
### 💡 Conclusion
Both models show strong performance in predicting the **Education Quality Index (EQI)**.  
However, if precision and capturing complex patterns are important, **Random Forest** may be the better choice.  
For explainability and simplicity, **Linear Regression** remains valuable.
""")
