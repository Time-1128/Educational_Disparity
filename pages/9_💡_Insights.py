import streamlit as st
import pandas as pd

# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------
st.set_page_config(page_title="Key Insights", layout="wide")
st.title("💡 Key Insights & Observations")

st.markdown("""
This page provides a **summary of the overall analysis and findings** from the project  
*“Mapping Educational Disparities in Indian Districts.”*  

It brings together the patterns discovered in data exploration, clustering,  
and model comparisons — to understand how educational quality varies across India.
---
""")

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/district_summary.csv")

df = load_data()

# -------------------------------------------------
# KEY INSIGHTS SECTION
# -------------------------------------------------
st.header("📊 Major Insights from the Analysis")

st.markdown("""
### 🏫 1. Uneven Distribution of Educational Resources
- Several districts show **high student–teacher ratios**, indicating overcrowded classrooms.  
- Districts with more balanced teacher allocation consistently achieve a **higher Education Quality Index (EQI)**.  
- The data reflects **urban–rural disparity**, where urban regions have better infrastructure and teacher availability.

---

### 🧱 2. Infrastructure Quality Strongly Influences EQI
- Districts with more **classrooms and supporting infrastructure** score higher in overall quality.  
- Poor infrastructure is a **common factor in underperforming districts**, even if teacher counts are adequate.  
- Investment in school facilities directly impacts **learning outcomes and EQI**.

---

### 🎓 3. Pre-primary Education Enhances Long-term Learning
- Districts with a higher number of **pre-primary schools** tend to have better EQI values.  
- Early education access improves student retention and performance in higher grades.  
- Expanding pre-primary programs is a **key lever for improving national education standards**.

---

### 🌍 4. State-Level Variations and Regional Gaps
- Southern and Western states show **consistently higher EQI averages**.  
- Some Northern and Eastern states have **large intra-state disparities**,  
  meaning certain districts perform well while neighboring ones lag behind.  
- These regional differences highlight where **policy intervention is most needed**.

---

### 🤖 5. Model Insights
- Both **Linear Regression** and **Random Forest** models accurately predicted EQI using district indicators.  
- Random Forest performed slightly better (lower prediction error),  
  indicating that **non-linear relationships** exist among factors.  
- Teacher–student ratio, infrastructure, and pre-primary ratio emerged as the **strongest predictors** of educational quality.

---

### 🧭 6. Key Takeaways for Policy and Planning
- Focus on **teacher distribution** — reduce student–teacher ratio in overloaded districts.  
- Prioritize **infrastructure upgrades** in low-performing areas.  
- Expand **pre-primary programs** to improve foundational learning.  
- Encourage **data-driven education policy** — using EQI metrics to identify and track improvement zones.
---
""")

# -------------------------------------------------
# CONCLUSION
# -------------------------------------------------
st.header("🎯 Final Conclusion")

st.markdown("""
This project demonstrates how **data science can uncover actionable insights**  
from large-scale educational data.

By analyzing over a million school records, we derived meaningful district-level trends,  
identified performance disparities, and developed predictive models for the **Education Quality Index (EQI)**.

> 📘 **In summary:**  
> Education quality in India is shaped by a combination of **teacher availability**,  
> **infrastructure strength**, and **access to early education**.  
> Bridging these gaps is essential for ensuring **equal learning opportunities across all districts**.
""")

st.success("✅ Project Summary Completed — 'Mapping Educational Disparities in Indian Districts'")
