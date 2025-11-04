# Educational Disparity Analysis - Setup Guide

## 🚀 How to Run the Application

### Prerequisites
Make sure you have the following installed:
- Python 3.7 or higher
- Required Python packages (see installation below)

### 📦 Installation Steps

1. **Install Required Packages**
   ```bash
   pip install streamlit pandas numpy scikit-learn plotly scipy
   ```

2. **Verify Data Files**
   Make sure you have the following file in the `data/` directory:
   - `data/basic-details-of-schools.csv`

### 🔧 Data Preprocessing (First Time Setup)

Before running the Streamlit app, you need to preprocess the data:

#### Option 1: Run the preprocessing script
```bash
python run_data_preprocessing.py
```

#### Option 2: Debug data loading (if you encounter issues)
```bash
python debug_data.py
```

#### Option 3: Manual preprocessing
```bash
python utils/data_cleaning.py
```

### 🌐 Running the Streamlit Application

Once data preprocessing is complete, run the main application:

```bash
streamlit run Home.py
```

The application will open in your web browser at `http://localhost:8501`

### 📊 Available Pages

1. **Home** - Overview and introduction
2. **Data Overview** - Basic data exploration
3. **EQI Prediction** - Linear regression model for predicting Education Quality Index
4. **Model Comparison** - Compare multiple ML models (Linear, Ridge, Lasso, Random Forest)

### 🔍 Troubleshooting

#### Error: "ValueError: Found array with 0 sample(s)"
This means the data preprocessing resulted in an empty dataset. Try:

1. **Check if data file exists:**
   ```bash
   ls -la data/basic-details-of-schools.csv
   ```

2. **Run the debug script:**
   ```bash
   python debug_data.py
   ```

3. **Check data file format:**
   - Ensure the CSV file has the required columns: `state_name`, `district_name`, `total_teachers`, `class_students`, `class_rooms`
   - Ensure the file is not corrupted

#### Error: "ModuleNotFoundError"
Install missing packages:
```bash
pip install [missing_package_name]
```

#### Error: "FileNotFoundError"
Make sure the data file is in the correct location:
```
project_root/
├── data/
│   └── basic-details-of-schools.csv
├── utils/
│   └── data_cleaning.py
├── pages/
│   ├── 7_🤖_EQI_Prediction.py
│   └── 8_⚖️_Model_Comparison.py
└── Home.py
```

### 📈 Expected Results

After successful preprocessing, you should see:
- **EQI Prediction**: R² scores typically between 0.3-0.7 (realistic accuracy)
- **Model Comparison**: Multiple models with varying performance
- **Interactive Tools**: Prediction tools for custom district characteristics

### 🛠️ Development Notes

- **Data Cleaning**: Handled automatically in `utils/data_cleaning.py`
- **No Districts Removed**: All districts and states are preserved through outlier capping
- **Realistic Accuracy**: Complex EQI calculation reduces overfitting without artificial noise
- **Automatic Preprocessing**: Pages will attempt to create cleaned data if not found

### 📞 Support

If you encounter issues:
1. Run `python debug_data.py` to identify the problem
2. Check that all required columns exist in your data file
3. Ensure you have sufficient data (at least 10 districts recommended)
4. Verify all Python packages are installed correctly

### 🎯 Quick Start Commands

```bash
# 1. Install dependencies
pip install streamlit pandas numpy scikit-learn plotly scipy

# 2. Preprocess data
python run_data_preprocessing.py

# 3. Run application
streamlit run Home.py
```

That's it! Your Educational Disparity Analysis application should now be running successfully.