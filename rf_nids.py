import os
import time
import joblib

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# CS-13 PROJECT
# STAGE 4 — RANDOM FOREST NETWORK INTRUSION DETECTION
# ============================================================

TRAIN_FILE = r"D:\CS13 PROJECT\DATA\ML\train.csv"
TEST_FILE = r"D:\CS13 PROJECT\DATA\ML\test.csv"

MODEL_FILE = r"D:\CS13 PROJECT\DATA\ML\random_forest_nids.joblib"


print("=" * 70)
print("       CS-13 NETWORK INTRUSION DETECTION PROJECT")
print("              STAGE 4 — RANDOM FOREST")
print("=" * 70)


# ============================================================
# 1. CHECK FILES
# ============================================================

print("\n[1] CHECKING FILES")
print("-" * 70)

if not os.path.exists(TRAIN_FILE):
    raise FileNotFoundError(f"Training file not found:\n{TRAIN_FILE}")

if not os.path.exists(TEST_FILE):
    raise FileNotFoundError(f"Testing file not found:\n{TEST_FILE}")

print("✓ train.csv found")
print("✓ test.csv found")


# ============================================================
# 2. LOAD TRAINING DATA
# ============================================================

print("\n[2] LOADING TRAINING DATA")
print("-" * 70)

train_df = pd.read_csv(TRAIN_FILE)

print("✓ Training dataset loaded")
print(f"Rows    : {len(train_df):,}")
print(f"Columns : {len(train_df.columns):,}")


# ============================================================
# 3. LOAD TESTING DATA
# ============================================================

print("\n[3] LOADING TESTING DATA")
print("-" * 70)

test_df = pd.read_csv(TEST_FILE)

print("✓ Testing dataset loaded")
print(f"Rows    : {len(test_df):,}")
print(f"Columns : {len(test_df.columns):,}")


# ============================================================
# 4. SEPARATE FEATURES AND TARGET
# ============================================================

print("\n[4] SEPARATING FEATURES AND TARGET")
print("-" * 70)

TARGET = "Label"

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]

print(f"Number of features: {X_train.shape[1]}")
print(f"Target column     : {TARGET}")

print("\nTraining class distribution:")
print(y_train.value_counts())


# ============================================================
# 5. CREATE RANDOM FOREST
# ============================================================

print("\n[5] CREATING RANDOM FOREST")
print("-" * 70)

model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("Algorithm       : Random Forest")
print("Trees           : 100")
print("Class weighting : Balanced")
print("CPU             : All available cores")


# ============================================================
# 6. TRAIN
# ============================================================

print("\n[6] TRAINING MODEL")
print("-" * 70)

print("Training started...")
print("This may take some time because there are 1.88 million rows.")

start_time = time.time()

model.fit(X_train, y_train)

training_time = time.time() - start_time

print("\n✓ TRAINING COMPLETED")
print(f"Training time: {training_time:.2f} seconds")


# ============================================================
# 7. PREDICT
# ============================================================

print("\n[7] MAKING TEST PREDICTIONS")
print("-" * 70)

start_time = time.time()

y_pred = model.predict(X_test)

prediction_time = time.time() - start_time

print("✓ Predictions completed")
print(f"Prediction time: {prediction_time:.2f} seconds")


# ============================================================
# 8. CALCULATE CONFIDENCE
# ============================================================

print("\n[8] CALCULATING CONFIDENCE")
print("-" * 70)

probabilities = model.predict_proba(X_test)

confidence = probabilities.max(axis=1)

average_confidence = confidence.mean()

print(
    f"Average model confidence: "
    f"{average_confidence * 100:.2f}%"
)


# ============================================================
# 9. EVALUATION
# ============================================================

print("\n[9] MODEL EVALUATION")
print("-" * 70)

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="binary",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="binary",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="binary",
    zero_division=0
)

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1-Score  : {f1 * 100:.2f}%")


# ============================================================
# 10. CLASSIFICATION REPORT
# ============================================================

print("\n[10] CLASSIFICATION REPORT")
print("-" * 70)

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# 11. CONFUSION MATRIX
# ============================================================

print("\n[11] CONFUSION MATRIX")
print("-" * 70)

cm = confusion_matrix(y_test, y_pred)

print(cm)


# ============================================================
# 12. SAVE MODEL
# ============================================================

print("\n[12] SAVING MODEL")
print("-" * 70)

joblib.dump(model, MODEL_FILE)

print("✓ Model saved successfully")
print(MODEL_FILE)


# ============================================================
# 13. SAMPLE ALERT
# ============================================================

print("\n[13] SAMPLE PREDICTION")
print("-" * 70)

sample_prediction = y_pred[0]
sample_confidence = confidence[0]

print(f"Prediction : {sample_prediction}")
print(f"Confidence : {sample_confidence * 100:.2f}%")


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("              STAGE 4 COMPLETED")
print("=" * 70)

print(f"Accuracy           : {accuracy * 100:.2f}%")
print(f"Precision          : {precision * 100:.2f}%")
print(f"Recall             : {recall * 100:.2f}%")
print(f"F1-Score           : {f1 * 100:.2f}%")
print(f"Average Confidence : {average_confidence * 100:.2f}%")

print("\nSaved model:")
print(MODEL_FILE)

print("=" * 70)