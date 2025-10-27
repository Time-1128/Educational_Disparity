import streamlit as st
import pandas as pd

# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------
st.set_page_config(page_title="Mapping Educational Disparities", layout="wide")

# -------------------------------------------------
# PAGE HEADER
# -------------------------------------------------
st.title("📘 Mapping Educational Disparities in Indian Districts")

st.markdown("""
### 🎓 Overview
This Streamlit dashboard explores **educational disparities across Indian districts**,  
analyzing key indicators like teacher availability, infrastructure quality, and pre-primary education.  
It also predicts an **Education Quality Index (EQI)** using data science and machine learning techniques.
""")

# -------------------------------------------------
# LOAD DATA AND SHOW METRICS
# -------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/district_summary.csv")

df = load_data()

col1, col2, col3, col4 = st.columns(4)
col1.metric("🏫 Total Districts", f"{len(df):,}")
col2.metric("🏠 Total Schools", f"{int(df['num_schools'].sum()):,}")
col3.metric("👩‍🏫 Total Teachers", f"{int(df['total_teachers'].sum()):,}")
col4.metric("📚 Avg Student–Teacher Ratio", round(df['avg_student_teacher_ratio'].mean(), 1))

# -------------------------------------------------
# EQI FORMULA EXPLANATION
# -------------------------------------------------
st.markdown("""
---
### 📘 Education Quality Index (EQI) Formula

The **Education Quality Index (EQI)** is a composite measure that evaluates the overall educational condition of each district.  
It is calculated as a **weighted combination of key educational indicators**:

> **EQI = (0.5 × (1 / Student–Teacher Ratio)) + (0.3 × Infrastructure Score) + (0.2 × Pre-Primary Ratio)**

To make values comparable across districts, the EQI is **normalized between 0 and 1**:

> **EQI (Normalized) = (EQI - Minimum EQI) / (Maximum EQI - Minimum EQI)**

#### 📊 Explanation of Indicators:
- 🧑‍🏫 **Student–Teacher Ratio (STR):** Average number of students per teacher (lower is better)  
- 🏫 **Infrastructure Score:** Based on classrooms and facilities per student (higher is better)  
- 👶 **Pre-Primary Ratio:** Proportion of schools offering pre-primary education (higher is better)  

✅ **Higher EQI → Better educational quality and infrastructure**
---
""")

# -------------------------------------------------
# PROJECT WORKFLOW
# -------------------------------------------------
st.markdown("""
### 🚀 Project Workflow
1️⃣ **Data Preparation** — Cleaning, aggregation, and feature generation from 1.3M+ school records  
2️⃣ **Exploratory Analysis** — Visualizing district disparities and correlations  
3️⃣ **Clustering** — Grouping districts using K-Means based on educational conditions  
4️⃣ **Machine Learning** — Predicting EQI using Random Forest and XGBoost  
5️⃣ **Interpretability & Insights** — Understanding feature impact and suggesting policies

---
### 🧭 Navigation
Use the **sidebar** to explore different stages of the analysis:
- 📊 Disparity and State Comparisons  
- 🧮 Clustering Insights  
- 🤖 EQI Prediction and Model Comparison  
- 💡 Final Policy Recommendations and Insights  

---
  
**Project Title:** *Mapping Educational Disparities in Indian Districts*  
""")
