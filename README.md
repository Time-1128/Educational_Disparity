# 🎓 Educational Disparity Analysis

A **Streamlit dashboard** designed to analyze, visualize, and model **educational disparity** across Indian districts.
It provides rich insights through interactive data exploration, clustering, and machine learning–based EQI (Education Quality Index) prediction.

---

## 📊 Key Features

* 🏠 **Home** – Overview and introduction to the project.
* 📂 **Data Overview** – Explore raw and cleaned school datasets.
* 📉 **Disparity Analysis** – Identify regional gaps in education quality.
* 🗺️ **State Comparison** – Compare metrics across different Indian states.
* 🧩 **Clustering** – Group districts based on similar education characteristics.
* 🔍 **Correlation Analysis** – Study relationships among education indicators.
* 🏅 **District Rankings** – Rank districts based on EQI and infrastructure.
* 🤖 **EQI Prediction** – Predict Education Quality Index using regression models.
* ⚖️ **Model Comparison** – Compare multiple ML models (Linear, Ridge, Lasso, Random Forest).
* 💡 **Insights** – Visual summaries and actionable findings.

---

## ⚙️ Setup Instructions

Follow these steps to run the project locally 👇

### 🧭 1️⃣ Clone the Repository and Load Data

> ⚠️ Make sure **Git LFS** is installed before cloning.
> Download from [https://git-lfs.github.com](https://git-lfs.github.com)

```bash
git lfs install
git clone https://github.com/Time-1128/Educational_Disparity.git
cd Educational_Disparity
git lfs pull
```

---

### 🧱 2️⃣ Create and Activate Virtual Environment

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 📦 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 🧹 4️⃣ Preprocess the Data

```bash
python utils/data_cleaning.py
```

This generates cleaned files such as `data/district_summary.csv` used by the app.

---

### ▶️ 5️⃣ Launch the Streamlit Dashboard

```bash
streamlit run app.py
```

After launching, open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📁 Project Structure

```
Educational_Disparity/
├── app.py                     # Main Streamlit dashboard
├── data/
│   ├── basic-details-of-schools.csv
│   └── district_summary.csv   # Generated after preprocessing
├── utils/
│   └── data_cleaning.py       # Data cleaning and preprocessing logic
├── pages/
│   ├── 1_📂_Data_Overview.py
│   ├── 2_📉_Disparity_Analysis.py
│   ├── 3_🗺️_State_Comparison.py
│   ├── 4_🧩_Clustering.py
│   ├── 5_🔍_Correlation_Analysis.py
│   ├── 6_🏅_District_Rankings.py
│   ├── 7_🤖_EQI_Prediction.py
│   ├── 8_⚖️_Model_Comparison.py
│   └── 9_💡_Insights.py
├── run_data_preprocessing.py
├── debug_data.py
├── requirements.txt
└── README.md
```

