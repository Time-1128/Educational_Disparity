import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Import data cleaning utility
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.data_cleaning import process_and_save_data

# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------
st.set_page_config(page_title="EQI Prediction with Linear Regression", layout="wide")
st.title("🤖 Education Quality Index Prediction Using Linear Regression")

st.markdown("""
This page uses **Linear Regression** to predict Education Quality Index (EQI) from district-level data:
- 📊 **Load Preprocessed Data**
- ⚙️ **Feature Preprocessing**
- 🤖 **Model Training & Evaluation**
- 🧮 **Interactive Prediction Tool**
""")

# -------------------------------------------------
# DATA LOADING
# -------------------------------------------------
st.header("📊 Data Loading & Preprocessing")

@st.cache_data
def load_processed_data():
    """Load or create processed district summary data"""
    try:
        # Try to load cleaned data first
        df = pd.read_csv("data/district_summary_cleaned.csv")
        st.success("✅ Loaded preprocessed district summary data")
        return df, "preprocessed"
    except FileNotFoundError:
        try:
            # Try to create cleaned data
            st.info("🔄 Creating preprocessed data from raw files...")
            success = process_and_save_data()
            if success:
                df = pd.read_csv("data/district_summary_cleaned.csv")
                st.success("✅ Created and loaded preprocessed data")
                return df, "preprocessed"
            else:
                raise Exception("Failed to process data")
        except:
            try:
                # Fallback to original district summary
                df = pd.read_csv("data/district_summary.csv")
                st.warning("⚠️ Using original district summary. Some features may not be optimal.")
                return df, "original"
            except FileNotFoundError:
                st.error("❌ No data files found. Please ensure data files are available.")
                return None, None

df, data_source = load_processed_data()

if df is not None:
    st.write(f"**Dataset:** {df.shape[0]} districts, {df.shape[1]} features ({data_source} data)")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Districts", len(df))
    with col2:
        st.metric("States", df['state_name'].nunique())
    with col3:
        st.metric("Features", len(df.columns))
    with col4:
        eqi_range = f"{df['edu_quality_index'].min():.3f} - {df['edu_quality_index'].max():.3f}"
        st.metric("EQI Range", eqi_range)

# -------------------------------------------------
# FEATURE PREPROCESSING
# -------------------------------------------------
if df is not None:
    st.header("⚙️ Feature Selection & Preprocessing")
    
    # Feature selection based on data source
    if data_source == "preprocessed" and 'student_teacher_ratio_mean' in df.columns:
        feature_cols = ['student_teacher_ratio_mean', 'infra_score_mean', 'pre_primary_ratio']
        display_names = ['Student-Teacher Ratio', 'Infrastructure Score', 'Pre-Primary Ratio']
    else:
        feature_cols = ['avg_student_teacher_ratio', 'avg_infra_score', 'pre_primary_ratio']
        display_names = ['Avg Student-Teacher Ratio', 'Avg Infrastructure Score', 'Pre-Primary Ratio']
    
    target_col = 'edu_quality_index'
    
    # Clean data - remove any remaining missing values
    df_model = df[feature_cols + [target_col]].dropna()
    
    st.write(f"**Data for modeling:** {df_model.shape[0]} districts with complete data")
    
    # Check if we have enough data
    if len(df_model) == 0:
        st.error("❌ No complete data available for modeling. Please check the data preprocessing.")
        st.stop()
    elif len(df_model) < 10:
        st.warning(f"⚠️ Very limited data ({len(df_model)} districts). Results may not be reliable.")
    elif len(df_model) < 50:
        st.info(f"ℹ️ Limited data ({len(df_model)} districts). Consider gathering more data for better results.")
    
    # Show basic statistics
    st.subheader("📈 Feature Statistics")
    
    stats_df = df_model[feature_cols + [target_col]].describe().round(4)
    st.dataframe(stats_df, width='stretch')
    
    # Feature correlation with target
    correlations = df_model[feature_cols].corrwith(df_model[target_col]).sort_values(key=abs, ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_corr = px.bar(
            x=correlations.values, 
            y=[display_names[feature_cols.index(col)] for col in correlations.index],
            orientation='h', 
            title='Feature Correlation with EQI',
            color=correlations.values, 
            color_continuous_scale='RdBu_r'
        )
        st.plotly_chart(fig_corr, width='stretch')
    
    with col2:
        # Target distribution
        fig_target = px.histogram(
            df_model, x=target_col, 
            title='Education Quality Index Distribution',
            nbins=30, marginal='box'
        )
        st.plotly_chart(fig_target, width='stretch')
    
    # Preprocessing options
    st.subheader("🔧 Preprocessing Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        scaling_method = st.selectbox("Feature Scaling Method:", 
                                     ["StandardScaler", "RobustScaler", "No Scaling"])
    
    with col2:
        test_size = st.slider("Test Set Size", 0.1, 0.5, 0.25, 0.05)
    
    # Prepare features and target
    X = df_model[feature_cols]
    y = df_model[target_col]
    
    # Apply scaling
    if len(X) == 0:
        st.error("❌ No data available for scaling. Please check the data preprocessing.")
        st.stop()
    
    if scaling_method == "StandardScaler":
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
        st.info("✅ Applied Standard Scaling (mean=0, std=1)")
    elif scaling_method == "RobustScaler":
        scaler = RobustScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
        st.info("✅ Applied Robust Scaling (median=0, IQR=1)")
    else:
        X_scaled = X.copy()
        scaler = None
        st.info("✅ No scaling applied")
    
    # Train-test split
    random_state = 42
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )
    
    st.write(f"**Training set:** {len(X_train)} districts | **Test set:** {len(X_test)} districts")

# -------------------------------------------------
# MODEL TRAINING & EVALUATION
# -------------------------------------------------
if df is not None and 'X_train' in locals():
    st.header("🤖 Model Training & Evaluation")
    
    # Model configuration
    st.subheader("🔧 Model Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        fit_intercept = st.checkbox("Fit Intercept", value=True)
    with col2:
        cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
    
    # Train the model
    model = LinearRegression(fit_intercept=fit_intercept)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_scaled, y, cv=cv_folds, scoring='r2')
    
    # Display performance metrics
    st.subheader("📊 Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Train R²", f"{train_r2:.3f}")
        st.metric("Test R²", f"{test_r2:.3f}")
    
    with col2:
        st.metric("Train MAE", f"{train_mae:.4f}")
        st.metric("Test MAE", f"{test_mae:.4f}")
    
    with col3:
        st.metric("Train RMSE", f"{train_rmse:.4f}")
        st.metric("Test RMSE", f"{test_rmse:.4f}")
    
    with col4:
        st.metric("CV R² Mean", f"{cv_scores.mean():.3f}")
        st.metric("CV R² Std", f"{cv_scores.std():.3f}")
    
    # Performance interpretation (adjusted for realistic educational data)
    if test_r2 > 0.4:
        performance_level = "🟢 Excellent (for educational data)"
    elif test_r2 > 0.25:
        performance_level = "🟡 Good"
    elif test_r2 > 0.15:
        performance_level = "🟠 Moderate"
    else:
        performance_level = "🔴 Poor"
    
    overfitting_diff = train_r2 - test_r2
    if overfitting_diff < 0.05:
        overfitting_status = "✅ Good generalization"
    elif overfitting_diff < 0.1:
        overfitting_status = "⚠️ Slight overfitting"
    else:
        overfitting_status = "❌ Significant overfitting"
    
    st.info(f"**Performance:** {performance_level} | **Generalization:** {overfitting_status}")
    
    # Model coefficients
    st.subheader("🧠 Model Coefficients & Feature Importance")
    
    coef_df = pd.DataFrame({
        'Feature': display_names,
        'Coefficient': model.coef_,
        'Abs_Coefficient': np.abs(model.coef_)
    }).sort_values('Abs_Coefficient', ascending=True)
    
    fig_coef = px.bar(
        coef_df, x='Coefficient', y='Feature', 
        orientation='h', title='Linear Regression Coefficients',
        color='Coefficient', color_continuous_scale='RdBu_r'
    )
    st.plotly_chart(fig_coef, width='stretch')
    
    if fit_intercept:
        st.write(f"**Model Intercept:** {model.intercept_:.4f}")
    
    # Model equation
    equation_parts = []
    for coef, name in zip(model.coef_, display_names):
        sign = "+" if coef >= 0 else ""
        equation_parts.append(f"{sign}{coef:.3f}×{name}")
    
    equation = f"EQI = {model.intercept_:.3f} {' '.join(equation_parts)}"
    st.code(equation)
    
    # Prediction analysis
    st.subheader("🎯 Prediction Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Predicted vs Actual scatter plot
        fig_scatter = go.Figure()
        
        fig_scatter.add_trace(go.Scatter(
            x=y_test, y=y_pred_test, 
            mode='markers', name='Test Predictions',
            opacity=0.7, marker=dict(color='blue', size=8)
        ))
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred_test.min())
        max_val = max(y_test.max(), y_pred_test.max())
        fig_scatter.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode='lines', name='Perfect Prediction',
            line=dict(dash='dash', color='red', width=2)
        ))
        
        fig_scatter.update_layout(
            title='Predicted vs Actual EQI',
            xaxis_title='Actual EQI',
            yaxis_title='Predicted EQI',
            showlegend=True
        )
        st.plotly_chart(fig_scatter, width='stretch')
    
    with col2:
        # Residuals plot
        residuals = y_test - y_pred_test
        fig_residuals = px.scatter(
            x=y_pred_test, y=residuals,
            title='Residuals Plot (Error Analysis)',
            labels={'x': 'Predicted EQI', 'y': 'Residuals (Actual - Predicted)'},
            opacity=0.7
        )
        fig_residuals.add_hline(y=0, line_dash="dash", line_color="red", line_width=2)
        st.plotly_chart(fig_residuals, width='stretch')
    
    # Cross-validation scores visualization
    fig_cv = px.box(y=cv_scores, title=f'{cv_folds}-Fold Cross-Validation R² Scores')
    fig_cv.update_layout(yaxis_title='R² Score')
    st.plotly_chart(fig_cv, width='stretch')

# -------------------------------------------------
# INTERACTIVE PREDICTION TOOL
# -------------------------------------------------
if df is not None and 'model' in locals():
    st.header("🧮 Interactive EQI Prediction Tool")
    
    st.write("**Enter district characteristics to predict Education Quality Index:**")
    
    cols = st.columns(len(feature_cols))
    user_inputs = {}
    
    for i, (col, name) in enumerate(zip(feature_cols, display_names)):
        with cols[i]:
            min_val = float(X[col].min())
            max_val = float(X[col].max())
            mean_val = float(X[col].mean())
            
            user_inputs[col] = st.number_input(
                name, 
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                step=(max_val - min_val) / 100,
                help=f"Range: {min_val:.3f} - {max_val:.3f}"
            )
    
    # Make prediction
    user_input_df = pd.DataFrame([user_inputs])
    
    if scaler is not None:
        user_input_scaled = pd.DataFrame(
            scaler.transform(user_input_df), 
            columns=user_input_df.columns
        )
    else:
        user_input_scaled = user_input_df
    
    prediction = model.predict(user_input_scaled)[0]
    
    # Display prediction with interpretation
    col1, col2 = st.columns(2)
    
    with col1:
        st.success(f"**Predicted EQI: {prediction:.4f}**")
        
        # Interpretation
        if prediction > 0.7:
            st.info("🟢 **High Quality** - Excellent educational indicators")
        elif prediction > 0.4:
            st.info("🟡 **Moderate Quality** - Average educational performance")
        else:
            st.info("🔴 **Needs Improvement** - Requires attention for educational quality")
    
    with col2:
        # Feature contribution analysis
        if scaler is not None:
            contributions = model.coef_ * user_input_scaled.values[0]
        else:
            contributions = model.coef_ * user_input_df.values[0]
        
        contrib_df = pd.DataFrame({
            'Feature': display_names,
            'Contribution': contributions
        }).sort_values('Contribution', key=abs, ascending=True)
        
        fig_contrib = px.bar(
            contrib_df, x='Contribution', y='Feature',
            orientation='h', title='Feature Contributions to Prediction',
            color='Contribution', color_continuous_scale='RdBu_r'
        )
        st.plotly_chart(fig_contrib, width='stretch')
    
    # Model insights
    st.subheader("💡 Model Insights & Interpretation")
    
    most_important_idx = np.argmax(np.abs(model.coef_))
    most_important_feature = display_names[most_important_idx]
    most_important_coef = model.coef_[most_important_idx]
    
    st.markdown(f"""
    **Model Performance Summary:**
    - **R² Score:** {test_r2:.3f} - Explains {test_r2*100:.1f}% of EQI variation
    - **Prediction Error:** ±{test_mae:.4f} EQI units on average
    - **Cross-validation:** {cv_scores.mean():.3f} ± {cv_scores.std():.3f}
    
    **Key Findings:**
    - **Most Important Feature:** {most_important_feature} (coefficient: {most_important_coef:.3f})
    - **Model Type:** Linear relationship between features and EQI
    - **Generalization:** {overfitting_status.split(' ', 1)[1]}
    
    **Feature Interpretation:**
    - **Positive coefficients** increase EQI (better quality)
    - **Negative coefficients** decrease EQI (lower quality)
    - **Larger absolute values** have stronger influence on EQI
    """)

else:
    st.error("❌ Unable to load or process data. Please ensure data files are available.")