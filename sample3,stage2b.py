import os
import pandas as pd
import numpy as np


# ============================================================
# CS-13 PROJECT
# ADAPTIVE AI NETWORK SECURITY ANALYST
# STAGE 2B — FEATURE & TARGET PREPARATION
# ============================================================

INPUT_FILE = (
    r"D:\CS13 PROJECT\DATA\PROCESSED"
    r"\NF-UNSW-NB15-v3-cleaned2.csv"
)

OUTPUT_FILE = (
    r"D:\CS13 PROJECT\DATA\PROCESSED"
    r"\NF-UNSW-NB15-v3-ml-ready.csv"
)


print("=" * 70)
print("        CS-13 NETWORK INTRUSION DETECTION PROJECT")
print("       STAGE 2B — FEATURE & TARGET PREPARATION")
print("=" * 70)


# ============================================================
# 1. FILE CHECK
# ============================================================

print("\n[1] FILE CHECK")
print("-" * 70)

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )

print("✓ Cleaned dataset found.")


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\n[2] LOADING DATASET")
print("-" * 70)

df = pd.read_csv(INPUT_FILE)

print("✓ Dataset loaded successfully.")
print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns):,}")


# ============================================================
# 3. TARGET SELECTION
# ============================================================

print("\n[3] TARGET SELECTION")
print("-" * 70)

TARGET = "Label"

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' not found."
    )

print(f"✓ Selected target: {TARGET}")
print("Purpose: Binary network intrusion detection")


# ============================================================
# 4. CHECK TARGET VALUES
# ============================================================

print("\n[4] TARGET VALUE CHECK")
print("-" * 70)

print("Target values:")

target_values = df[TARGET].value_counts(
    dropna=False
)

for value, count in target_values.items():

    percentage = (count / len(df)) * 100

    print(
        f"Value: {str(value):15} | "
        f"Count: {count:12,} | "
        f"Percentage: {percentage:7.2f}%"
    )


# ============================================================
# 5. REMOVE TARGET LEAKAGE
# ============================================================

print("\n[5] TARGET LEAKAGE REMOVAL")
print("-" * 70)

# Attack contains information about attack category.
# It must not be given to the binary classifier.

leakage_columns = [
    "Label",
    "Attack"
]

existing_leakage_columns = [
    column
    for column in leakage_columns
    if column in df.columns
]

print("Columns excluded from ML features:")

for column in existing_leakage_columns:
    print(f"✓ {column}")


# ============================================================
# 6. REMOVE IDENTIFIER / DIRECT ADDRESS COLUMNS
# ============================================================

print("\n[6] IDENTIFIER COLUMN HANDLING")
print("-" * 70)

# These fields can identify a particular flow/session
# rather than provide stable behavioural features.

identifier_columns = [
    "IPV4_SRC_ADDR",
    "IPV4_DST_ADDR"
]

existing_identifier_columns = [
    column
    for column in identifier_columns
    if column in df.columns
]

if existing_identifier_columns:

    print("Columns excluded:")

    for column in existing_identifier_columns:
        print(f"✓ {column}")

else:

    print("No identifier columns found.")


# ============================================================
# 7. CREATE FEATURE DATASET
# ============================================================

print("\n[7] FEATURE CREATION")
print("-" * 70)

columns_to_remove = list(
    set(existing_leakage_columns + existing_identifier_columns)
)

X = df.drop(
    columns=columns_to_remove
)

y = df[TARGET].copy()

print(f"Initial feature count: {X.shape[1]}")


# ============================================================
# 8. REMOVE TIMESTAMP COLUMNS
# ============================================================

print("\n[8] TIMESTAMP HANDLING")
print("-" * 70)

timestamp_columns = [
    "FLOW_START_MILLISECONDS",
    "FLOW_END_MILLISECONDS"
]

existing_timestamp_columns = [
    column
    for column in timestamp_columns
    if column in X.columns
]

if existing_timestamp_columns:

    X = X.drop(
        columns=existing_timestamp_columns
    )

    print("Timestamp columns excluded:")

    for column in existing_timestamp_columns:
        print(f"✓ {column}")

else:

    print("No timestamp columns found.")


# ============================================================
# 9. HANDLE CATEGORICAL FEATURES
# ============================================================

print("\n[9] CATEGORICAL FEATURE HANDLING")
print("-" * 70)

categorical_columns = X.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()

if categorical_columns:

    print("Categorical columns found:")

    for column in categorical_columns:
        print(
            f"• {column} "
            f"({X[column].nunique(dropna=False):,} unique values)"
        )

    # One-hot encoding
    X = pd.get_dummies(
        X,
        columns=categorical_columns,
        dummy_na=True,
        dtype=np.int8
    )

    print("✓ Categorical columns encoded.")

else:

    print("✓ No categorical feature encoding required.")


# ============================================================
# 10. HANDLE INFINITE VALUES
# ============================================================

print("\n[10] INFINITE VALUE HANDLING")
print("-" * 70)

numeric_columns = X.select_dtypes(
    include=np.number
).columns

infinite_count = np.isinf(
    X[numeric_columns]
).sum().sum()

print(f"Infinite values found: {infinite_count:,}")

if infinite_count > 0:

    X[numeric_columns] = X[numeric_columns].replace(
        [np.inf, -np.inf],
        np.nan
    )

    print("✓ Infinite values converted to NaN.")

else:

    print("✓ No infinite values found.")


# ============================================================
# 11. HANDLE MISSING VALUES
# ============================================================

print("\n[11] MISSING VALUE HANDLING")
print("-" * 70)

missing_before = X.isna().sum().sum()

print(
    f"Missing feature values before handling: "
    f"{missing_before:,}"
)

numeric_columns = X.select_dtypes(
    include=np.number
).columns

if len(numeric_columns) > 0:

    # Median is used because network-flow features
    # can contain extreme values.
    X[numeric_columns] = X[numeric_columns].fillna(
        X[numeric_columns].median()
    )

print("✓ Missing numeric values filled using median.")

remaining_missing = X.isna().sum().sum()

print(
    f"Remaining missing feature values: "
    f"{remaining_missing:,}"
)


# ============================================================
# 12. TARGET MISSING VALUE HANDLING
# ============================================================

print("\n[12] TARGET VALIDATION")
print("-" * 70)

target_missing = y.isna().sum()

print(f"Missing target values: {target_missing:,}")

if target_missing > 0:

    valid_rows = y.notna()

    X = X.loc[valid_rows].reset_index(drop=True)
    y = y.loc[valid_rows].reset_index(drop=True)

    print(
        f"✓ Removed {target_missing:,} rows "
        "with missing target values."
    )

else:

    print("✓ No missing target values.")


# ============================================================
# 13. FINAL FEATURE / TARGET STRUCTURE
# ============================================================

print("\n[13] FINAL ML STRUCTURE")
print("-" * 70)

print(f"Final rows    : {len(X):,}")
print(f"Final features: {X.shape[1]:,}")

print("\nFeature matrix : X")
print("Target vector  : y")
print("Target         : Label")


# ============================================================
# 14. COMBINE FEATURES + TARGET
# ============================================================

print("\n[14] CREATING ML-READY DATASET")
print("-" * 70)

ml_ready = X.copy()

ml_ready[TARGET] = y.values

print("✓ Features and target combined.")


# ============================================================
# 15. FINAL VALIDATION
# ============================================================

print("\n[15] FINAL VALIDATION")
print("-" * 70)

final_missing = ml_ready.isna().sum().sum()

numeric_columns = ml_ready.select_dtypes(
    include=np.number
).columns

final_infinite = np.isinf(
    ml_ready[numeric_columns]
).sum().sum()

print(f"Remaining missing values : {final_missing:,}")
print(f"Remaining infinite values: {final_infinite:,}")

if final_missing == 0 and final_infinite == 0:

    print("✓ Dataset is ready for ML.")

else:

    print("⚠ Dataset still requires additional cleaning.")


# ============================================================
# 16. SAVE ML-READY DATASET
# ============================================================

print("\n[16] SAVING ML-READY DATASET")
print("-" * 70)

output_directory = os.path.dirname(OUTPUT_FILE)

os.makedirs(
    output_directory,
    exist_ok=True
)

ml_ready.to_csv(
    OUTPUT_FILE,
    index=False
)

print("✓ ML-ready dataset saved.")

print("\nOutput file:")
print(OUTPUT_FILE)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("          STAGE 2B COMPLETED SUCCESSFULLY")
print("=" * 70)

print(f"Final rows       : {len(ml_ready):,}")
print(f"Final features   : {X.shape[1]:,}")
print(f"Target           : {TARGET}")
print("Task             : Binary Intrusion Detection")
print("Status           : ML-READY")

print("=" * 70)