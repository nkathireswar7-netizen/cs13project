import os
import pandas as pd

from sklearn.model_selection import train_test_split


# ============================================================
# CS-13 PROJECT
# STAGE 3 — TRAIN / TEST SPLIT
# ============================================================

INPUT_FILE = r"D:\CS13 PROJECT\DATA\PROCESSED\NF-UNSW-NB15-v3-ml-ready.csv"

OUTPUT_DIR = r"D:\CS13 PROJECT\DATA\ML"

TRAIN_FILE = os.path.join(OUTPUT_DIR, "train.csv")
TEST_FILE = os.path.join(OUTPUT_DIR, "test.csv")

TARGET = "Label"


print("=" * 70)
print("       CS-13 NETWORK INTRUSION DETECTION PROJECT")
print("              STAGE 3 — TRAIN / TEST SPLIT")
print("=" * 70)


# ============================================================
# 1. CHECK INPUT FILE
# ============================================================

print("\n[1] CHECKING INPUT FILE")
print("-" * 70)

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"ML-ready dataset not found:\n{INPUT_FILE}"
    )

print("✓ ML-ready dataset found.")


# ============================================================
# 2. CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"✓ Output directory ready:")
print(OUTPUT_DIR)


# ============================================================
# 3. LOAD ML-READY DATASET
# ============================================================

print("\n[2] LOADING ML-READY DATASET")
print("-" * 70)

df = pd.read_csv(INPUT_FILE)

print("✓ Dataset loaded.")
print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns):,}")


# ============================================================
# 4. CHECK TARGET
# ============================================================

print("\n[3] CHECKING TARGET")
print("-" * 70)

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' not found."
    )

print(f"Target column: {TARGET}")

print("\nTarget distribution:")

print(
    df[TARGET]
    .value_counts(dropna=False)
)


# ============================================================
# 5. SEPARATE FEATURES AND TARGET
# ============================================================

print("\n[4] SEPARATING FEATURES AND TARGET")
print("-" * 70)

X = df.drop(columns=[TARGET])
y = df[TARGET]

print(f"Features : {X.shape}")
print(f"Target   : {y.shape}")


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

print("\n[5] TRAIN / TEST SPLIT")
print("-" * 70)

print("Using:")
print("Training data : 80%")
print("Testing data  : 20%")
print("Random state  : 42")
print("Stratify      : Yes")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n✓ Split completed.")


# ============================================================
# 7. RECREATE DATAFRAMES
# ============================================================

print("\n[6] CREATING TRAINING AND TESTING DATASETS")
print("-" * 70)

train_df = X_train.copy()
train_df[TARGET] = y_train.values

test_df = X_test.copy()
test_df[TARGET] = y_test.values

print(f"Training rows : {len(train_df):,}")
print(f"Testing rows  : {len(test_df):,}")


# ============================================================
# 8. SAVE TRAINING DATA
# ============================================================

print("\n[7] SAVING TRAINING DATA")
print("-" * 70)

train_df.to_csv(
    TRAIN_FILE,
    index=False
)

print("✓ train.csv saved.")
print(TRAIN_FILE)


# ============================================================
# 9. SAVE TESTING DATA
# ============================================================

print("\n[8] SAVING TESTING DATA")
print("-" * 70)

test_df.to_csv(
    TEST_FILE,
    index=False
)

print("✓ test.csv saved.")
print(TEST_FILE)


# ============================================================
# 10. VERIFY FILES
# ============================================================

print("\n[9] VERIFYING OUTPUT FILES")
print("-" * 70)

if os.path.exists(TRAIN_FILE):
    train_size = os.path.getsize(TRAIN_FILE) / (1024 ** 3)
    print(f"✓ train.csv exists")
    print(f"  Size: {train_size:.2f} GB")

if os.path.exists(TEST_FILE):
    test_size = os.path.getsize(TEST_FILE) / (1024 ** 3)
    print(f"✓ test.csv exists")
    print(f"  Size: {test_size:.2f} GB")


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 70)
print("             STAGE 3 COMPLETED")
print("=" * 70)

print("\nTRAIN FILE:")
print(TRAIN_FILE)

print("\nTEST FILE:")
print(TEST_FILE)

print("\nTrain rows :", f"{len(train_df):,}")
print("Test rows  :", f"{len(test_df):,}")

print("\n✓ Ready for Stage 4 — Random Forest")
print("=" * 70)
