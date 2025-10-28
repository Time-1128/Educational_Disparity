# ⚙️ Setup Instructions

Follow these steps to clone and run the project locally 👇  

---

### 🧭 1️⃣ Clone the Repository and Load Data

> ⚠️ Make sure **Git LFS** is installed before cloning.  
> Install from: [https://git-lfs.github.com](https://git-lfs.github.com)

```bash
git lfs install
git clone https://github.com/Time-1128/Educational_Disparity.git
cd Educational_Disparity

# Pull large data files (if not automatically downloaded)
git lfs pull
```

---

### 🧱 2️⃣ Create and Activate Virtual Environment

> 💡 Use a virtual environment to isolate dependencies.

**For Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**For macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 📦 3️⃣ Install Dependencies

> 📋 Install all required Python libraries listed in `requirements.txt`.

```bash
pip install -r requirements/requirements.txt
```

---

### 🧹 4️⃣ Clean and Prepare Data (Important Step)

> 🧠 This script processes raw data and generates cleaned files used by the dashboard.

```bash
python utils/data_cleaning.py
```

This ensures that `data/district_summary.csv` and other cleaned datasets are ready.

---

### ▶️ 5️⃣ Run the Streamlit Dashboard

> 🚀 Launch the dashboard locally.

```bash
streamlit run app.py
```

After running, open the URL shown in the terminal:  
👉 [http://localhost:8501](http://localhost:8501)
