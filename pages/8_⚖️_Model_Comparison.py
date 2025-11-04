import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
st.set_page_config(page_title="Model Comparison for EQI Prediction", layout="wide")
st.title("⚖️ Model Comparison: Education Quality Index Prediction")

st.markdown("""
This page compares multiple machine learning models for predicting Education Quality Index (EQI):
- **Linear Regression** - Simple linear relationships
- **Ridge Regression** - Linear with L2 regularization  
- **Lasso Regression** - Linear with L1 regularization
- **Random Forest** - Non-linear ensemble method
""")

# -------------------------------------------------
# DATA LOADING
# -------------------------------------------------
st.header("📊 Data Loading")

@st.cache_data
def load_data_for_comparison():
    """Load preprocessed district summary data"""
    try:
        # Try to load preprocessed data first
        df = pd.read_csv("data/district_summary_cleaned.csv")
        st.success("✅ Loaded preprocessed district summary data")
        return df, "preprocessed"
    except FileNotFoundError:
        try:
            # Try to create preprocessed data
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

df, data_source = load_data_for_comparison()

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
# DATA PREPROCESSING
# -------------------------------------------------
if df is not None:
    st.header("⚙️ Data Preprocessing")
    
    # Feature selection based on data source
    if data_source == "preprocessed" and 'student_teacher_ratio_mean' in df.columns:
        feature_cols = ['student_teacher_ratio_mean', 'infra_score_mean', 'pre_primary_ratio']
        display_names = ['Student-Teacher Ratio', 'Infrastructure Score', 'Pre-Primary Ratio']
    else:
        feature_cols = ['avg_student_teacher_ratio', 'avg_infra_score', 'pre_primary_ratio']
        display_names = ['Avg Student-Teacher Ratio', 'Avg Infrastructure Score', 'Pre-Primary Ratio']
    
    target_col = 'edu_quality_index'
    
    # Clean data
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
    
    # Preprocessing options
    col1, col2 = st.columns(2)
    
    with col1:
        outlier_method = st.selectbox("Outlier Handling:", 
                                     ["No Removal", "IQR Method", "Z-Score Method"])
    
    with col2:
        scaling_method = st.selectbox("Feature Scaling:", 
                                     ["StandardScaler", "RobustScaler", "No Scaling"])
    
    # Apply outlier handling (cap instead of remove to preserve all districts)
    if outlier_method == "IQR Method":
        def cap_outliers_iqr(df, columns):
            df_clean = df.copy()
            for col in columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
            return df_clean
        
        df_processed = cap_outliers_iqr(df_model, feature_cols)
        st.info("Applied IQR-based outlier capping (no districts removed)")
        
    elif outlier_method == "Z-Score Method":
        def cap_outliers_zscore(df, columns, threshold=3):
            df_clean = df.copy()
            for col in columns:
                mean_val = df[col].mean()
                std_val = df[col].std()
                lower_bound = mean_val - threshold * std_val
                upper_bound = mean_val + threshold * std_val
                df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
            return df_clean
        
        df_processed = cap_outliers_zscore(df_model, feature_cols)
        st.info("Applied Z-Score-based outlier capping (no districts removed)")
    else:
        df_processed = df_model.copy()
        st.info("No outlier handling applied")
    
    # Prepare features and target
    X = df_processed[feature_cols]
    y = df_processed[target_col]
    
    # Apply scaling
    if len(X) == 0:
        st.error("❌ No data available for scaling. Please check the data preprocessing.")
        st.stop()
    
    if scaling_method == "StandardScaler":
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    elif scaling_method == "RobustScaler":
        scaler = RobustScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    else:
        X_scaled = X.copy()
        scaler = None
    
    # Train-test split
    test_size = st.slider("Test Set Size", 0.1, 0.5, 0.25, 0.05)
    random_state = 42
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )
    
    st.write(f"**Training set:** {len(X_train)} samples | **Test set:** {len(X_test)} samples")

# -------------------------------------------------
# MODEL TRAINING & COMPARISON
# -------------------------------------------------
if df is not None and 'X_train' in locals():
    st.header("🤖 Model Training & Comparison")
    
    # Model configuration
    st.subheader("🔧 Model Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ridge_alpha = st.slider("Ridge Alpha", 0.01, 10.0, 1.0, 0.01)
        lasso_alpha = st.slider("Lasso Alpha", 0.001, 1.0, 0.01, 0.001)
    
    with col2:
        rf_n_estimators = st.slider("RF Trees", 50, 300, 100, 10)
        rf_max_depth = st.slider("RF Max Depth", 3, 20, 10)
    
    with col3:
        cv_folds = st.slider("CV Folds", 3, 10, 5)
    
    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=ridge_alpha, random_state=random_state),
        'Lasso Regression': Lasso(alpha=lasso_alpha, random_state=random_state, max_iter=2000),
        'Random Forest': RandomForestRegressor(
            n_estimators=rf_n_estimators, 
            max_depth=rf_max_depth, 
            random_state=random_state
        )
    }
    
    # Train models and collect results
    results = {}
    predictions = {}
    
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        predictions[name] = y_pred_test
        
        # Calculate metrics
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_scaled, y, cv=cv_folds, scoring='r2')
        
        results[name] = {
            'Train R²': train_r2,
            'Test R²': test_r2,
            'Train MAE': train_mae,
            'Test MAE': test_mae,
            'Train RMSE': train_rmse,
            'Test RMSE': test_rmse,
            'CV R² Mean': cv_scores.mean(),
            'CV R² Std': cv_scores.std(),
            'Overfitting': train_r2 - test_r2
        }
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results).T.round(4)
    
    # Display results
    st.subheader("📊 Model Performance Comparison")
    
    # Performance table
    st.dataframe(results_df, width='stretch')
    
    # Performance visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # R² comparison
        fig_r2 = px.bar(
            x=results_df.index,
            y=results_df['Test R²'],
            title='Test R² Score Comparison',
            color=results_df['Test R²'],
            color_continuous_scale='Viridis'
        )
        fig_r2.update_traces(text=results_df['Test R²'].round(3), textposition='outside')
        st.plotly_chart(fig_r2, width='stretch')
    
    with col2:
        # MAE comparison
        fig_mae = px.bar(
            x=results_df.index,
            y=results_df['Test MAE'],
            title='Test MAE Comparison',
            color=results_df['Test MAE'],
            color_continuous_scale='Reds_r'
        )
        fig_mae.update_traces(text=results_df['Test MAE'].round(4), textposition='outside')
        st.plotly_chart(fig_mae, width='stretch')
    
    # Cross-validation comparison
    cv_data = []
    for name in models.keys():
        cv_data.append({
            'Model': name,
            'CV R² Mean': results_df.loc[name, 'CV R² Mean'],
            'CV R² Std': results_df.loc[name, 'CV R² Std']
        })
    
    cv_df = pd.DataFrame(cv_data)
    
    fig_cv = px.bar(
        cv_df, x='Model', y='CV R² Mean',
        error_y='CV R² Std',
        title='Cross-Validation R² Scores',
        color='CV R² Mean',
        color_continuous_scale='Blues'
    )
    fig_cv.update_traces(text=cv_df['CV R² Mean'].round(3), textposition='outside')
    st.plotly_chart(fig_cv, width='stretch')
    
    # Overfitting analysis
    st.subheader("🔍 Overfitting Analysis")
    
    overfitting_df = pd.DataFrame({
        'Model': results_df.index,
        'Train R²': results_df['Train R²'],
        'Test R²': results_df['Test R²'],
        'Overfitting Gap': results_df['Overfitting']
    })
    
    fig_overfit = go.Figure()
    
    fig_overfit.add_trace(go.Bar(
        name='Train R²',
        x=overfitting_df['Model'],
        y=overfitting_df['Train R²'],
        marker_color='lightblue'
    ))
    
    fig_overfit.add_trace(go.Bar(
        name='Test R²',
        x=overfitting_df['Model'],
        y=overfitting_df['Test R²'],
        marker_color='darkblue'
    ))
    
    fig_overfit.update_layout(
        title='Train vs Test R² (Overfitting Check)',
        barmode='group',
        yaxis_title='R² Score'
    )
    
    st.plotly_chart(fig_overfit, width='stretch')
    
    # Prediction comparison
    st.subheader("🎯 Prediction Comparison")
    
    # Create subplot for all models
    fig_pred = make_subplots(
        rows=2, cols=2,
        subplot_titles=list(models.keys()),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    colors = ['blue', 'red', 'green', 'orange']
    
    for i, (name, color) in enumerate(zip(models.keys(), colors)):
        row = (i // 2) + 1
        col = (i % 2) + 1
        
        # Scatter plot of predictions vs actual
        fig_pred.add_trace(
            go.Scatter(
                x=y_test,
                y=predictions[name],
                mode='markers',
                name=name,
                marker=dict(color=color, opacity=0.6),
                showlegend=False
            ),
            row=row, col=col
        )
        
        # Perfect prediction line
        min_val = min(y_test.min(), predictions[name].min())
        max_val = max(y_test.max(), predictions[name].max())
        fig_pred.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(dash='dash', color='gray'),
                showlegend=False
            ),
            row=row, col=col
        )
    
    fig_pred.update_layout(
        title_text="Predicted vs Actual EQI for All Models",
        height=600
    )
    
    # Update axes labels
    for i in range(1, 3):
        for j in range(1, 3):
            fig_pred.update_xaxes(title_text="Actual EQI", row=i, col=j)
            fig_pred.update_yaxes(title_text="Predicted EQI", row=i, col=j)
    
    st.plotly_chart(fig_pred, width='stretch')
    
    # Feature importance comparison (for applicable models)
    st.subheader("🧠 Feature Importance Comparison")
    
    importance_data = []
    
    for name, model in models.items():
        if hasattr(model, 'coef_'):
            # Linear models
            importances = np.abs(model.coef_)
        elif hasattr(model, 'feature_importances_'):
            # Tree-based models
            importances = model.feature_importances_
        else:
            continue
        
        for feature, importance in zip(display_names, importances):
            importance_data.append({
                'Model': name,
                'Feature': feature,
                'Importance': importance
            })
    
    if importance_data:
        importance_df = pd.DataFrame(importance_data)
        
        fig_importance = px.bar(
            importance_df, x='Importance', y='Feature',
            color='Model', orientation='h',
            title='Feature Importance by Model',
            barmode='group'
        )
        st.plotly_chart(fig_importance, width='stretch')
    
    # Model ranking and recommendations
    st.subheader("🏆 Model Ranking & Recommendations")
    
    # Rank models by different criteria
    ranking_criteria = {
        'Best Test R²': results_df['Test R²'].idxmax(),
        'Lowest Test MAE': results_df['Test MAE'].idxmin(),
        'Best CV Performance': results_df['CV R² Mean'].idxmax(),
        'Least Overfitting': results_df['Overfitting'].idxmin(),
        'Most Stable (Low CV Std)': results_df['CV R² Std'].idxmin()
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**🏅 Model Rankings:**")
        for criterion, best_model in ranking_criteria.items():
            st.write(f"- **{criterion}:** {best_model}")
    
    with col2:
        # Overall recommendation based on multiple criteria
        scores = {}
        for model in models.keys():
            score = 0
            # Test R² (40% weight)
            score += 0.4 * (results_df.loc[model, 'Test R²'] / results_df['Test R²'].max())
            # CV Mean (30% weight)
            score += 0.3 * (results_df.loc[model, 'CV R² Mean'] / results_df['CV R² Mean'].max())
            # Low overfitting (20% weight) - inverse scoring
            score += 0.2 * (1 - results_df.loc[model, 'Overfitting'] / results_df['Overfitting'].max())
            # Low CV std (10% weight) - inverse scoring
            score += 0.1 * (1 - results_df.loc[model, 'CV R² Std'] / results_df['CV R² Std'].max())
            scores[model] = score
        
        best_overall = max(scores, key=scores.get)
        
        st.write("**🎯 Overall Recommendation:**")
        st.success(f"**{best_overall}** (Score: {scores[best_overall]:.3f})")
        
        st.write("**📋 Summary:**")
        st.write(f"- **R² Score:** {results_df.loc[best_overall, 'Test R²']:.3f}")
        st.write(f"- **MAE:** {results_df.loc[best_overall, 'Test MAE']:.4f}")
        st.write(f"- **CV Score:** {results_df.loc[best_overall, 'CV R² Mean']:.3f} ± {results_df.loc[best_overall, 'CV R² Std']:.3f}")
    
    # Insights and conclusions
    st.subheader("💡 Key Insights")
    
    best_r2_model = results_df['Test R²'].idxmax()
    best_r2_score = results_df.loc[best_r2_model, 'Test R²']
    
    st.markdown(f"""
    **Performance Analysis:**
    - **Best performing model:** {best_r2_model} with R² = {best_r2_score:.3f}
    - **Model complexity:** {'Linear models show similar performance' if results_df.loc[['Linear Regression', 'Ridge Regression'], 'Test R²'].std() < 0.05 else 'Significant performance differences between models'}
    - **Overfitting:** {'Minimal overfitting detected' if results_df['Overfitting'].max() < 0.1 else 'Some models show overfitting'}
    
    **Recommendations:**
    - **For interpretability:** Choose Linear or Ridge Regression
    - **For performance:** Choose {best_overall}
    - **For robustness:** Consider cross-validation results
    - **For production:** Balance performance, interpretability, and computational cost
    """)

else:
    st.error("❌ Unable to load data. Please run the Data Cleaning page first to generate cleaned data.")