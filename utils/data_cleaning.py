import pandas as pd
import numpy as np

def clean_and_aggregate(input_path="data/basic-details-of-schools.csv",
                        output_path="data/district_summary.csv",
                        chunksize=100_000):
    print("🔹 Loading large dataset in chunks...")

    dtype_dict = {
        "udise_school_code": "string",
        "school_name": "string",
        "state_name": "string",
        "district_name": "string",
        "school_category": "string",
        "school_type": "string",
        "management": "string",
        "status": "string",
        "location_type": "string"
    }

    all_chunks = []
    i = 0
    for chunk in pd.read_csv(
        input_path,
        dtype=dtype_dict,
        low_memory=False,
        chunksize=chunksize
    ):
        i += 1
        print(f"🔹 Processing chunk {i} ({len(chunk):,} rows)")

        # --- Basic cleaning per chunk ---
        chunk = chunk.drop_duplicates(subset=["udise_school_code"])
        chunk = chunk.dropna(subset=["district_name", "state_name", "total_teachers", "class_students"])
        chunk = chunk[chunk["total_teachers"] > 0]

        for col in ["class_rooms", "other_rooms", "pre_primary"]:
            if col in chunk.columns:
                chunk[col] = chunk[col].fillna(0)

        # --- Derived metrics ---
        chunk["student_teacher_ratio"] = chunk["class_students"] / chunk["total_teachers"]
        chunk["infra_score"] = (chunk["class_rooms"] + chunk["other_rooms"]) / (chunk["class_students"] + 1)
        chunk["school_density"] = 1

        # --- Keep only useful columns to save memory ---
        cols_needed = ["state_name", "district_name", "total_teachers", "class_students",
                       "class_rooms", "other_rooms", "student_teacher_ratio", "infra_score", "pre_primary", "school_density"]
        all_chunks.append(chunk[cols_needed])

    print("✅ Finished reading all chunks, combining results...")
    df = pd.concat(all_chunks, ignore_index=True)
    del all_chunks  # free memory

    # --- Aggregate to district level ---
    district_df = df.groupby(["state_name", "district_name"], as_index=False).agg({
        "school_density": "count",
        "total_teachers": "sum",
        "class_students": "sum",
        "class_rooms": "sum",
        "other_rooms": "sum",
        "student_teacher_ratio": "mean",
        "infra_score": "mean",
        "pre_primary": "sum"
    }).rename(columns={"school_density": "num_schools", "pre_primary": "num_pre_primary"})

    # --- Derived ratios per district ---
    district_df["avg_student_teacher_ratio"] = district_df["class_students"] / district_df["total_teachers"]
    district_df["avg_infra_score"] = (district_df["class_rooms"] + district_df["other_rooms"]) / (district_df["class_students"] + 1)
    district_df["pre_primary_ratio"] = district_df["num_pre_primary"] / district_df["num_schools"]

    # --- Education Quality Index ---
    eqi = (1 / district_df["avg_student_teacher_ratio"]) * 0.5 \
          + (district_df["avg_infra_score"]) * 0.3 \
          + (district_df["pre_primary_ratio"]) * 0.2
    district_df["edu_quality_index"] = (eqi - eqi.min()) / (eqi.max() - eqi.min())

    district_df.to_csv(output_path, index=False)
    print(f"✅ Saved cleaned dataset to: {output_path}")
    print(f"✅ Total districts processed: {len(district_df):,}")

    return district_df


if __name__ == "__main__":
    clean_and_aggregate()
