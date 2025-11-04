#!/usr/bin/env python3
"""
Data preprocessing script for Educational Disparity Analysis
Run this script to generate cleaned and preprocessed data files
"""

from utils.data_cleaning import process_and_save_data

if __name__ == "__main__":
    print("=" * 60)
    print("Educational Disparity Analysis - Data Preprocessing")
    print("=" * 60)
    
    success = process_and_save_data()
    
    if success:
        print("\n" + "=" * 60)
        print("✅ Data preprocessing completed successfully!")
        print("Generated files:")
        print("  - data/schools_cleaned.csv")
        print("  - data/district_summary_cleaned.csv")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ Data preprocessing failed!")
        print("Please check the error messages above.")
        print("=" * 60)