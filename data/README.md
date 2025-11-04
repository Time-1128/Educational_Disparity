# Data Directory

This directory contains the data files for the Educational Disparity Analysis project.

## Required Data Files

To run this project, you need to place the following file in this directory:

### Raw Data
- `basic-details-of-schools.csv` - The main dataset containing school information
  - **Source**: [Provide your data source here]
  - **Size**: ~1.3M schools across India
  - **Note**: This file is not included in the repository due to size constraints

## Generated Files

The following files will be automatically generated when you run the data cleaning process:

- `district_summary.csv` - Original district-level summary
- `district_summary_cleaned.csv` - Cleaned district-level summary with realistic EQI
- `schools_cleaned.csv` - Cleaned school-level data

## Data Setup Instructions

1. Download the `basic-details-of-schools.csv` file
2. Place it in this `data/` directory
3. Run the data cleaning process:
   ```bash
   python utils/data_cleaning.py
   ```

## Data Structure

### School-level data should contain:
- `state_name` - State name
- `district_name` - District name  
- `school_name` - School name
- `total_teachers` - Number of teachers
- `class_students` - Number of students
- `class_rooms` - Number of classrooms
- `other_rooms` - Number of other rooms
- `pre_primary` - Pre-primary availability (Yes/No)

### Generated district-level features:
- `student_teacher_ratio_mean` - Average student-teacher ratio
- `infra_score_mean` - Average infrastructure score
- `pre_primary_ratio` - Proportion of schools with pre-primary
- `edu_quality_index` - Calculated education quality index (0-1 scale)