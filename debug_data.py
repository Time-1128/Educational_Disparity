#!/usr/bin/env python3
"""
Debug script to test data loading and processing
"""

import pandas as pd
import numpy as np

def debug_data_loading():
    """Debug the data loading process step by step"""
    
    print("=" * 60)
    print("DEBUG: Data Loading Process")
    print("=" * 60)
    
    try:
        # Step 1: Load raw data
        print("Step 1: Loading raw data...")
        df = pd.read_csv("data/basic-details-of-schools.csv")
        print(f"✅ Loaded {len(df):,} schools with {len(df.columns)} columns")
        
        # Step 2: Check critical columns
        print("\nStep 2: Checking critical columns...")
        critical_cols = ['state_name', 'district_name', 'total_teachers', 'class_students', 'class_rooms']
        
        for col in critical_cols:
            if col in df.columns:
                missing = df[col].isnull().sum()
                print(f"✅ {col}: {missing:,} missing values ({missing/len(df)*100:.1f}%)")
            else:
                print(f"❌ {col}: Column not found!")
                return False
        
        # Step 3: Check data after cleaning
        print("\nStep 3: Applying basic cleaning...")
        df_clean = df.copy()
        
        # Remove missing values
        initial_count = len(df_clean)
        df_clean = df_clean.dropna(subset=critical_cols)
        print(f"After removing missing data: {len(df_clean):,} schools ({initial_count - len(df_clean):,} removed)")
        
        # Remove invalid values
        df_clean = df_clean[df_clean['total_teachers'] > 0]
        df_clean = df_clean[df_clean['class_students'] > 0]
        df_clean = df_clean[df_clean['class_rooms'] > 0]
        print(f"After removing invalid values: {len(df_clean):,} schools")
        
        if len(df_clean) == 0:
            print("❌ ERROR: No schools remain after cleaning!")
            return False
        
        # Step 4: Test feature engineering
        print("\nStep 4: Testing feature engineering...")
        df_clean['student_teacher_ratio'] = df_clean['class_students'] / df_clean['total_teachers']
        df_clean['infra_score'] = (df_clean['other_rooms'].fillna(0) / (df_clean['class_rooms'] + df_clean['other_rooms'].fillna(0))).fillna(0)
        df_clean['has_pre_primary'] = (df_clean['pre_primary'] == 'Yes').astype(int)
        print("✅ Feature engineering successful")
        
        # Step 5: Test district aggregation
        print("\nStep 5: Testing district aggregation...")
        district_stats = df_clean.groupby(['state_name', 'district_name']).agg({
            'student_teacher_ratio': 'mean',
            'infra_score': 'mean',
            'has_pre_primary': 'mean',
            'school_name': 'count'
        }).reset_index()
        
        print(f"✅ Created {len(district_stats)} district summaries")
        print(f"States: {district_stats['state_name'].nunique()}")
        print(f"Districts: {len(district_stats)}")
        
        # Step 6: Show sample data
        print("\nStep 6: Sample district data:")
        print(district_stats.head())
        
        print("\n" + "=" * 60)
        print("✅ DEBUG COMPLETED SUCCESSFULLY")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = debug_data_loading()
    if not success:
        print("\n❌ Debug failed. Please check the error messages above.")
    else:
        print("\n✅ Data loading process works correctly!")