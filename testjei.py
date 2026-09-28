import pandas as pd
import numpy as np
import joblib

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_FILE = r"D:\CS13 PROJECT\DATA\ML\random_forest_nids.joblib"

# IMPORTANT:
# Change this to the NEW dataset where Label and Attack are EMPTY.
INPUT_FILE = r"D:\CS13 PROJECT\DATA\TEST\NF-UNSW-NB15-v3-unlabeled2.csv"

OUTPUT_FILE = r"D:\CS13 PROJECT\DATA\ML\label_predictions.csv"


# ============================================================
# 1. LOAD TRAINED MODEL
# ============================================================

print("=" * 60)
print("STEP 1 - LOADING TRAINED MODEL")
print("=" * 60)

model = joblib.load(MODEL_FILE)

print("Model loaded successfully.")
print("Model type:", type(model).__name__)

if hasattr(model, "feature_names_in_"):
    print("Expected model features:", len(model.feature_names_in_))
else:
    raise ValueError(
        "The saved model does not contain feature_names_in_."
    )


# ============================================================
# 2. LOAD NEW DATASET
# ============================================================

print("\n" + "=" * 60)
print("STEP 2 - LOADING NEW UNLABELED DATASET")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print("Dataset loaded successfully.")
print("Rows:", len(df))
print("Columns:", len(df))


# ============================================================
# 3. CHECK LABEL AND ATTACK COLUMNS
# ============================================================

print("\n" + "=" * 60)
print("STEP 3 - CHECKING LABEL AND ATTACK")
print("=" * 60)

if "Label" not in df.columns:
    raise ValueError("Label column is missing from the dataset.")

if "Attack" not in df.columns:
    raise ValueError("Attack column is missing from the dataset.")

print("Label column found.")
print("Attack column found.")

print("\nLabel non-empty values:",
      df["Label"].notna().sum())

print("Attack non-empty values:",
      df["Attack"].notna().sum())

if df["Label"].notna().sum() > 0:
    print("\nWARNING: Label contains values.")
    print("Make sure this is really your unlabeled dataset.")

if df["Attack"].notna().sum() > 0:
    print("\nWARNING: Attack contains values.")
    print("Make sure this is really your unlabeled dataset.")


# ============================================================
# 4. KEEP ORIGINAL DATA FOR FINAL OUTPUT
# ============================================================

original_df = df.copy()


# ============================================================
# 5. REMOVE COLUMNS NOT USED BY THE MODEL
# ============================================================

print("\n" + "=" * 60)
print("STEP 4 - REMOVING NON-MODEL COLUMNS")
print("=" * 60)

columns_to_remove = [
    "Label",
    "Attack",
    "IPV4_SRC_ADDR",
    "IPV4_DST_ADDR",
    "FLOW_START_MILLISECONDS",
    "FLOW_END_MILLISECONDS"
]

existing_remove_columns = [
    column
    for column in columns_to_remove
    if column in df.columns
]

print("Removing:")
for column in existing_remove_columns:
    print(" -", column)

X = df.drop(
    columns=existing_remove_columns
)


# ============================================================
# 6. CONVERT CATEGORICAL DATA
# ============================================================

print("\n" + "=" * 60)
print("STEP 5 - ENCODING CATEGORICAL FEATURES")
print("=" * 60)

X = pd.get_dummies(
    X,
    dummy_na=True,
    dtype=np.int8
)

print("Features after encoding:", X.shape[1])


# ============================================================
# 7. CONVERT INFINITE VALUES TO NaN
# ============================================================

print("\n" + "=" * 60)
print("STEP 6 - HANDLING INFINITE VALUES")
print("=" * 60)

numeric_X = X.select_dtypes(
    include=[np.number]
)

infinite_count = np.isinf(
    numeric_X.to_numpy()
).sum()

print("Infinite values found:", infinite_count)

if infinite_count > 0:

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    print("Infinite values converted to NaN.")

else:

    print("No infinite values found.")


# ============================================================
# 8. HANDLE NaN / MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("STEP 7 - HANDLING MISSING VALUES")
print("=" * 60)

missing_before = X.isna().sum().sum()

print("Missing values before filling:",
      missing_before)

numeric_columns = X.select_dtypes(
    include=[np.number]
).columns

# Fill each numeric feature with its median
for column in numeric_columns:

    median_value = X[column].median()

    # Safety for completely empty columns
    if pd.isna(median_value):
        median_value = 0

    X[column] = X[column].fillna(
        median_value
    )


# ============================================================
# 9. FINAL NaN / INFINITY CHECK
# ============================================================

remaining_nan = X.isna().sum().sum()

remaining_inf = np.isinf(
    X.select_dtypes(
        include=[np.number]
    ).to_numpy()
).sum()

print("Missing values after filling:",
      remaining_nan)

print("Infinite values after filling:",
      remaining_inf)

if remaining_nan > 0:
    raise ValueError(
        "NaN values still exist. Prediction stopped."
    )

if remaining_inf > 0:
    raise ValueError(
        "Infinite values still exist. Prediction stopped."
    )


# ============================================================
# 10. MATCH EXACT TRAINING FEATURES
# ============================================================

print("\n" + "=" * 60)
print("STEP 8 - MATCHING TRAINING FEATURES")
print("=" * 60)

expected_features = list(
    model.feature_names_in_
)

print("Model expects:",
      len(expected_features),
      "features")

# ------------------------------------------------------------
# Add missing features
# ------------------------------------------------------------

missing_features = [
    column
    for column in expected_features
    if column not in X.columns
]

if missing_features:

    print(
        "Missing training features:",
        len(missing_features)
    )

    for column in missing_features:
        X[column] = 0

else:

    print("No training features are missing.")


# ------------------------------------------------------------
# Find extra features
# ------------------------------------------------------------

extra_features = [
    column
    for column in X.columns
    if column not in expected_features
]

if extra_features:

    print(
        "Extra features ignored:",
        len(extra_features)
    )

    X = X.drop(
        columns=extra_features
    )

else:

    print("No extra features.")


# ------------------------------------------------------------
# Exact feature order
# ------------------------------------------------------------

X = X[
    expected_features
]


# ============================================================
# 11. FINAL FEATURE CHECK
# ============================================================

print("\n" + "=" * 60)
print("STEP 9 - FINAL MODEL INPUT CHECK")
print("=" * 60)

print("Final rows:", X.shape[0])
print("Final features:", X.shape[1])

if X.shape[1] != len(expected_features):

    raise ValueError(
        f"Feature mismatch! "
        f"Model expects {len(expected_features)}, "
        f"but received {X.shape[1]}."
    )

if list(X.columns) != expected_features:

    raise ValueError(
        "Feature order does not match the trained model."
    )

if X.isna().sum().sum() != 0:

    raise ValueError(
        "NaN values remain in model input."
    )

if np.isinf(
    X.to_numpy()
).sum() != 0:

    raise ValueError(
        "Infinite values remain in model input."
    )

print("Feature check PASSED.")
print("Model input is ready.")


# ============================================================
# 12. MAKE LABEL PREDICTIONS
# ============================================================

print("\n" + "=" * 60)
print("STEP 10 - MAKING LABEL PREDICTIONS")
print("=" * 60)

predictions = model.predict(X)

probabilities = model.predict_proba(X)

confidence = probabilities.max(
    axis=1
)

print("Prediction completed.")


# ============================================================
# 13. ADD RESULTS TO ORIGINAL DATASET
# ============================================================

print("\n" + "=" * 60)
print("STEP 11 - ADDING PREDICTIONS")
print("=" * 60)

original_df["Predicted_Label"] = predictions

original_df["Confidence"] = confidence

original_df["Confidence_Percent"] = (
    confidence * 100
)


# ============================================================
# 14. SAVE OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("STEP 12 - SAVING RESULTS")
print("=" * 60)

original_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("Output saved successfully:")
print(OUTPUT_FILE)


# ============================================================
# 15. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FINAL RESULT")
print("=" * 60)

print("Total rows:",
      len(original_df))

print("\nPredicted Label distribution:")

print(
    original_df[
        "Predicted_Label"
    ].value_counts()
)

print(
    "\nAverage confidence:",
    round(
        original_df[
            "Confidence"
        ].mean() * 100,
        2
    ),
    "%"
)

print(
    "\nMinimum confidence:",
    round(
        original_df[
            "Confidence"
        ].min() * 100,
        2
    ),
    "%"
)

print(
    "Maximum confidence:",
    round(
        original_df[
            "Confidence"
        ].max() * 100,
        2
    ),
    "%"
)

print("\nOutput file:")
print(OUTPUT_FILE)

print("\nPrediction pipeline completed successfully.")
print("=" * 60)