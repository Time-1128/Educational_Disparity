import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px

# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------
st.set_page_config(page_title="District Clustering", layout="wide")
st.title("🧮 Clustering Analysis of Indian Districts")

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/district_summary.csv")
    return df

df = load_data()

st.markdown("""
This section applies **K-Means clustering** to group districts with similar educational conditions  
based on the following indicators:
- 👩‍🏫 Student–Teacher Ratio  
- 🏫 Infrastructure Score  
- 🎓 Pre-Primary School Ratio  
- ⭐ Education Quality Index (EQI)
""")

# -------------------------------------------------
# SELECT NUMBER OF CLUSTERS
# -------------------------------------------------
k = st.slider("Select number of clusters (K)", 2, 8, 4)

# Features used for clustering
features = ["avg_student_teacher_ratio", "avg_infra_score", "pre_primary_ratio", "edu_quality_index"]
X = df[features].dropna()

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means
kmeans = KMeans(n_clusters=k, random_state=42)
df["cluster"] = kmeans.fit_predict(X_scaled)

# -------------------------------------------------
# PCA for 2D Visualization
# -------------------------------------------------
pca = PCA(n_components=2)
pca_comp = pca.fit_transform(X_scaled)
df["PC1"], df["PC2"] = pca_comp[:, 0], pca_comp[:, 1]

st.write(f"🧭 Explained variance by PC1 + PC2: {sum(pca.explained_variance_ratio_)*100:.2f}%")

fig = px.scatter(
    df,
    x="PC1",
    y="PC2",
    color=df["cluster"].astype(str),
    hover_data=["district_name", "state_name", "edu_quality_index"],
    title="District Clusters Based on Educational Indicators (PCA Visualization)",
)
st.plotly_chart(fig, config={"responsive": True}, width='stretch')

# ... existing imports, code, and 2D PCA plot ...

# -------------------------------------------------
# 🧭 OPTIONAL: 3D PCA VISUALIZATION
# -------------------------------------------------
st.markdown("### 🧩 3D PCA Visualization of District Clusters")

from sklearn.decomposition import PCA
import plotly.express as px

pca3 = PCA(n_components=3)
pca_3d = pca3.fit_transform(X_scaled)
df["PC1_3D"], df["PC2_3D"], df["PC3_3D"] = pca_3d[:, 0], pca_3d[:, 1], pca_3d[:, 2]

fig_3d = px.scatter_3d(
    df,
    x="PC1_3D",
    y="PC2_3D",
    z="PC3_3D",
    color=df["cluster"].astype(str),
    hover_data=["district_name", "state_name", "edu_quality_index"],
    title="3D PCA Visualization of Educational Clusters",
)
st.plotly_chart(fig_3d, config={"responsive": True}, width='stretch')

# -------------------------------------------------
# CLUSTER SUMMARY
# -------------------------------------------------
st.write("### 📊 Cluster Summary Statistics")

cluster_summary = df.groupby("cluster").agg({
    "avg_student_teacher_ratio": "mean",
    "avg_infra_score": "mean",
    "pre_primary_ratio": "mean",
    "edu_quality_index": "mean",
    "district_name": "count"
}).rename(columns={"district_name": "num_districts"}).reset_index()

# Sort by EQI (so higher EQI clusters come first)
cluster_summary = cluster_summary.sort_values("edu_quality_index", ascending=False).reset_index(drop=True)

# -------------------------------------------------
# DYNAMIC CLUSTER LABELING
# -------------------------------------------------
# Adaptive cluster labels based on number of clusters (k)
if k == 2:
    cluster_labels = ["🌟 High Performing", "⚠️ Under-Resourced"]
elif k == 3:
    cluster_labels = ["🌟 High Performing", "🙂 Average", "⚠️ Under-Resourced"]
elif k == 4:
    cluster_labels = ["🌟 High Performing", "🙂 Moderate", "⚠️ Under-Resourced", "🚧 Poor Infrastructure"]
elif k == 5:
    cluster_labels = ["🌟 Very High Performing", "🌟 High Performing", "🙂 Average", "⚠️ Under-Resourced", "🚧 Poor Infrastructure"]
else:
    # For k > 5, assign neutral labels (Cluster A, Cluster B, ...)
    cluster_labels = [f"Cluster {chr(65+i)}" for i in range(k)]

cluster_summary["Cluster Type"] = cluster_labels[:len(cluster_summary)]

# -------------------------------------------------
# DISPLAY CLUSTER SUMMARY TABLE
# -------------------------------------------------
st.dataframe(
    cluster_summary.style.format({
        "avg_student_teacher_ratio": "{:.1f}",
        "avg_infra_score": "{:.2f}",
        "pre_primary_ratio": "{:.2f}",
        "edu_quality_index": "{:.2f}"
    }),
    width='stretch'
)

# -------------------------------------------------
# STATE DISTRIBUTION PER CLUSTER
# -------------------------------------------------
st.write("### 🗺️ Which States Belong to Which Clusters")

state_cluster = df.groupby(["state_name", "cluster"], as_index=False)["district_name"].count()
fig_state = px.bar(
    state_cluster,
    x="state_name",
    y="district_name",
    color=state_cluster["cluster"].astype(str),
    title="State-wise Distribution of Districts by Cluster",
    labels={"district_name": "Number of Districts", "state_name": "State", "color": "Cluster"},
)
st.plotly_chart(fig_state, config={"responsive": True}, width='stretch')

# -------------------------------------------------
# INTERPRETATION & INSIGHTS
# -------------------------------------------------
st.write("### 🧠 Cluster Interpretation")

for idx, row in cluster_summary.iterrows():
    st.markdown(f"""
    **Cluster {idx} — {row['Cluster Type']}**
    - Average Student–Teacher Ratio: `{row['avg_student_teacher_ratio']:.1f}`
    - Average Infrastructure Score: `{row['avg_infra_score']:.2f}`
    - Pre-Primary Ratio: `{row['pre_primary_ratio']:.2f}`
    - Education Quality Index: `{row['edu_quality_index']:.2f}`
    """)

st.markdown("""
---
**Key Observations:**
- 🌟 *High-performing clusters* have low student–teacher ratios and strong infrastructure.  
- ⚠️ *Under-resourced clusters* show high ratios and weaker facilities.  
- Larger states often appear in multiple clusters, revealing **intra-state disparities**.  
--- 
""")
