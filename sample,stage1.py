import os
import pandas as pd
import numpy as np


# ============================================================
# CS-13 PROJECT
# ADAPTIVE AI NETWORK SECURITY ANALYST
# DATA PREPROCESSING — STAGE 1
# ============================================================

INPUT_FILE = r"D:\CS13 PROJECT\DATA\RAW\NF-UNSW-NB15-v3.csv"
OUTPUT_FILE = r"D:\CS13 PROJECT\DATA\PROCESSED\NF-UNSW-NB15-v3-cleaned2.csv"


print("=" * 70)
print("        CS-13 NETWORK INTRUSION DETECTION PROJECT")
print("             DATA PREPROCESSING — STAGE 1")
print("=" * 70)


# ============================================================
# 1. FILE CHECK
# ============================================================

print("\n[1] FILE CHECK")
print("-" * 70)

if not os.path.exists(INPUT_FILE):
    print("❌ Input file not found!")
    print(f"Expected location: {INPUT_FILE}")
    raise FileNotFoundError(INPUT_FILE)

print("✓ Input file found")
print(f"File: {INPUT_FILE}")


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(INPUT_FILE)

print("✓ Dataset loaded successfully")


# ============================================================
# 2. DATASET SIZE
# ============================================================

print("\n[2] DATASET SIZE")
print("-" * 70)

rows, columns = df.shape

print(f"Total Rows    : {rows:,}")
print(f"Total Columns : {columns:,}")


# ============================================================
# 3. COLUMN INFORMATION
# ============================================================

print("\n[3] COLUMN INFORMATION")
print("-" * 70)

print(f"Total number of columns: {columns}")

print("\nColumn names:")

for i, column in enumerate(df.columns, start=1):
    print(f"{i:3}. {column}")


# ============================================================
# 4. DATA PREVIEW
# ============================================================
# Removed intentionally.
# Large datasets do not need rows printed in the terminal.
# If required later, use:
# print(df.head())
# ============================================================

print("\n[4] DATA PREVIEW")
print("-" * 70)

print("✓ Data preview skipped to keep output clean for large datasets.")


# ============================================================
# 5. DATA TYPES
# ============================================================

print("\n[5] DATA TYPES")
print("-" * 70)

data_type_counts = df.dtypes.value_counts()

print("Data type summary:")

for dtype, count in data_type_counts.items():
    print(f"{str(dtype):12} : {count} columns")

print(f"\nNumeric columns   : {df.select_dtypes(include=np.number).shape[1]}")
print(f"Text/Categorical  : {df.select_dtypes(exclude=np.number).shape[1]}")


# ============================================================
# 6. MISSING VALUES
# ============================================================

print("\n[6] MISSING VALUES")
print("-" * 70)

missing_counts = df.isnull().sum()

total_missing = missing_counts.sum()

if total_missing == 0:

    print("✓ No missing values found.")

else:

    print(f"Total missing values: {total_missing:,}")

    print("\nColumns containing missing values:")

    missing_columns = missing_counts[missing_counts > 0]

    for column, count in missing_columns.items():

        percentage = (count / rows) * 100

        print(
            f"{column:30} : "
            f"{count:,} "
            f"({percentage:.2f}%)"
        )


# ============================================================
# 7. DUPLICATE VALUES
# ============================================================

print("\n[7] DUPLICATE VALUES")
print("-" * 70)

duplicate_count = df.duplicated().sum()

print(f"Duplicate rows found: {duplicate_count:,}")

if duplicate_count > 0:

    df = df.drop_duplicates().reset_index(drop=True)

    print(f"✓ Duplicate rows removed: {duplicate_count:,}")

else:

    print("✓ No duplicate rows found.")


# ============================================================
# 8. INFINITE VALUES
# ============================================================

print("\n[8] INFINITE VALUES")
print("-" * 70)

numeric_columns = df.select_dtypes(include=np.number).columns

if len(numeric_columns) > 0:

    infinite_mask = np.isinf(df[numeric_columns])

    total_infinite = infinite_mask.sum().sum()

else:

    total_infinite = 0


print(f"Infinite values found: {total_infinite:,}")

if total_infinite > 0:

    df[numeric_columns] = df[numeric_columns].replace(
        [np.inf, -np.inf],
        np.nan
    )

    print("✓ Infinite values converted to NaN.")

else:

    print("✓ No infinite values found.")


# ============================================================
# 9. TARGET / LABEL CANDIDATES
# ============================================================

print("\n[9] TARGET / LABEL CANDIDATES")
print("-" * 70)

print("Searching for possible target/label columns...")

# Common names used for labels/targets
possible_label_names = [
    "label",
    "Label",
    "LABEL",
    "attack",
    "Attack",
    "target",
    "Target",
    "class",
    "Class",
    "category",
    "Category"
]

found_labels = [
    column
    for column in df.columns
    if column in possible_label_names
]

if found_labels:

    print("\nPossible label columns found:")

    for column in found_labels:

        unique_count = df[column].nunique(dropna=False)

        print(
            f"• {column} "
            f"→ {unique_count:,} unique values"
        )

else:

    print("No obvious label column found using common names.")

    print("\nLow-cardinality columns that may be labels:")

    candidates_found = False

    for column in df.columns:

        unique_count = df[column].nunique(dropna=False)

        if unique_count <= 20:

            candidates_found = True

            print(
                f"• {column} "
                f"→ {unique_count:,} unique values"
            )

    if not candidates_found:

        print("No low-cardinality columns found.")


# ============================================================
# 10. CLEANING SUMMARY
# ============================================================

print("\n[10] CLEANING SUMMARY")
print("-" * 70)

final_rows, final_columns = df.shape

print(f"Original rows       : {rows:,}")
print(f"Final rows          : {final_rows:,}")

print(f"Original columns    : {columns:,}")
print(f"Final columns       : {final_columns:,}")

print(f"Duplicates removed  : {duplicate_count:,}")
print(f"Infinite values     : {total_infinite:,}")
print(f"Missing values      : {df.isnull().sum().sum():,}")

print("\n✓ Stage-1 cleaning completed.")


# ============================================================
# 11. OUTPUT FILE LOCATION
# ============================================================

print("\n[11] OUTPUT FILE LOCATION")
print("-" * 70)

# Create output directory if it does not exist
output_directory = os.path.dirname(OUTPUT_FILE)

os.makedirs(output_directory, exist_ok=True)

# Save cleaned dataset
df.to_csv(OUTPUT_FILE, index=False)

print("✓ Cleaned dataset saved successfully.")
print(f"\nOutput file:")
print(OUTPUT_FILE)


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 70)
print("             STAGE-1 PREPROCESSING COMPLETED")
print("=" * 70)

print(f"Rows processed : {final_rows:,}")
print(f"Columns        : {final_columns:,}")
print("Status         : SUCCESS")
print("=" * 70)