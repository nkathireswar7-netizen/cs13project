import os
import pandas as pd
import numpy as np


# ============================================================
# CS-13 PROJECT
# STAGE 2A — DATASET DIAGNOSIS
# ============================================================

INPUT_FILE = (
    r"D:\CS13 PROJECT\DATA\PROCESSED"
    r"\NF-UNSW-NB15-v3-cleaned2.csv"
)


print("=" * 70)
print("        CS-13 NETWORK INTRUSION DETECTION PROJECT")
print("             STAGE 2A — DATASET DIAGNOSIS")
print("=" * 70)


# ============================================================
# 1. FILE CHECK AND DATASET LOADING
# ============================================================

print("\n[1] FILE CHECK")
print("-" * 70)

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"File not found: {INPUT_FILE}"
    )

print("✓ Cleaned dataset found.")

print("\nLoading dataset...")

df = pd.read_csv(INPUT_FILE)

print("✓ Dataset loaded successfully.")


# ============================================================
# 2. DATASET SIZE
# ============================================================

print("\n[2] DATASET SIZE")
print("-" * 70)

rows, columns = df.shape

print(f"Rows    : {rows:,}")
print(f"Columns : {columns:,}")


# ============================================================
# 3. TARGET COLUMN CHECK
# ============================================================

print("\n[3] TARGET COLUMN CHECK")
print("-" * 70)

target_columns = ["Label", "Attack"]

for column in target_columns:

    if column in df.columns:

        print(f"\nTarget column: {column}")

        print(f"Data type: {df[column].dtype}")

        print(
            f"Missing values: "
            f"{df[column].isna().sum():,}"
        )

        print(
            f"Unique values: "
            f"{df[column].nunique(dropna=False):,}"
        )

    else:

        print(f"❌ Column not found: {column}")


# ============================================================
# 4. LABEL DISTRIBUTION
# ============================================================

print("\n[4] LABEL DISTRIBUTION")
print("-" * 70)

if "Label" in df.columns:

    label_counts = df["Label"].value_counts(
        dropna=False
    )

    label_percentages = (
        df["Label"]
        .value_counts(normalize=True, dropna=False)
        .mul(100)
    )

    print("Label distribution:")

    for value, count in label_counts.items():

        percentage = label_percentages[value]

        print(
            f"Value: {str(value):15} | "
            f"Count: {count:12,} | "
            f"Percentage: {percentage:7.2f}%"
        )

else:

    print("Label column not found.")


# ============================================================
# 5. ATTACK CATEGORY DISTRIBUTION
# ============================================================

print("\n[5] ATTACK CATEGORY DISTRIBUTION")
print("-" * 70)

if "Attack" in df.columns:

    attack_counts = df["Attack"].value_counts(
        dropna=False
    )

    attack_percentages = (
        df["Attack"]
        .value_counts(normalize=True, dropna=False)
        .mul(100)
    )

    print("Attack category distribution:")

    for value, count in attack_counts.items():

        percentage = attack_percentages[value]

        print(
            f"Category: {str(value):20} | "
            f"Count: {count:12,} | "
            f"Percentage: {percentage:7.2f}%"
        )

else:

    print("Attack column not found.")


# ============================================================
# 6. MISSING VALUES BY COLUMN
# ============================================================

print("\n[6] MISSING VALUES BY COLUMN")
print("-" * 70)

missing_counts = df.isna().sum()

missing_columns = missing_counts[
    missing_counts > 0
].sort_values(ascending=False)

if len(missing_columns) == 0:

    print("✓ No missing values found.")

else:

    print(
        f"Columns containing missing values: "
        f"{len(missing_columns)}"
    )

    for column, count in missing_columns.items():

        percentage = (count / rows) * 100

        print(
            f"{column:35} | "
            f"{count:12,} | "
            f"{percentage:7.2f}%"
        )


# ============================================================
# 7. DATA TYPES
# ============================================================

print("\n[7] TEXT / CATEGORICAL COLUMNS")
print("-" * 70)

text_columns = df.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()

if len(text_columns) == 0:

    print("No text or categorical columns found.")

else:

    print(f"Total text/categorical columns: {len(text_columns)}")

    for column in text_columns:

        print(
            f"• {column:35} | "
            f"Unique values: {df[column].nunique(dropna=False):,}"
        )


# ============================================================
# 8. NUMERIC FEATURE SUMMARY
# ============================================================

print("\n[8] NUMERIC FEATURE SUMMARY")
print("-" * 70)

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

print(f"Total numeric columns: {len(numeric_columns)}")

print(
    "✓ Numeric columns identified for further ML preparation."
)


# ============================================================
# 9. POTENTIAL IDENTIFIER COLUMNS
# ============================================================

print("\n[9] POTENTIAL IDENTIFIER COLUMNS")
print("-" * 70)

identifier_keywords = [
    "IPV4_SRC_ADDR",
    "IPV4_DST_ADDR",
    "FLOW_START_MILLISECONDS",
    "FLOW_END_MILLISECONDS"
]

for column in identifier_keywords:

    if column in df.columns:

        print(
            f"• {column} "
            f"→ present; review before model training"
        )


# ============================================================
# 10. DIAGNOSIS SUMMARY
# ============================================================

print("\n[10] DIAGNOSIS SUMMARY")
print("-" * 70)

print(f"Rows examined       : {rows:,}")
print(f"Columns examined    : {columns:,}")
print(f"Total missing cells : {df.isna().sum().sum():,}")

if "Label" in df.columns:

    print(
        f"Label unique values : "
        f"{df['Label'].nunique(dropna=False):,}"
    )

if "Attack" in df.columns:

    print(
        f"Attack unique values: "
        f"{df['Attack'].nunique(dropna=False):,}"
    )

print("\n✓ Dataset diagnosis completed.")
print("✓ No additional rows were deleted.")
print("✓ No feature encoding was performed yet.")


print("\n" + "=" * 70)
print("             STAGE 2A COMPLETED")
print("=" * 70)