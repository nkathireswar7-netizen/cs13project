import os
import joblib
import pandas as pd


# ============================================================
# CS-13 PROJECT
# STAGE 5 — GENERATE SECURITY ALERTS
# ============================================================

TEST_FILE = r"D:\CS13 PROJECT\DATA\ML\test.csv"

MODEL_FILE = r"D:\CS13 PROJECT\DATA\ML\random_forest_nids.joblib"

OUTPUT_FILE = r"D:\CS13 PROJECT\DATA\ML\security_alerts.csv"

TARGET = "Label"


print("=" * 70)
print("       CS-13 NETWORK INTRUSION DETECTION PROJECT")
print("              STAGE 5 — SECURITY ALERTS")
print("=" * 70)


# ============================================================
# 1. CHECK FILES
# ============================================================

print("\n[1] CHECKING FILES")
print("-" * 70)

if not os.path.exists(TEST_FILE):
    raise FileNotFoundError(f"Test file not found:\n{TEST_FILE}")

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(f"Model file not found:\n{MODEL_FILE}")

print("✓ test.csv found")
print("✓ Random Forest model found")


# ============================================================
# 2. LOAD TEST DATA
# ============================================================

print("\n[2] LOADING TEST DATA")
print("-" * 70)

df = pd.read_csv(TEST_FILE)

print("✓ Test data loaded")
print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns):,}")


# ============================================================
# 3. SEPARATE FEATURES
# ============================================================

print("\n[3] PREPARING FEATURES")
print("-" * 70)

X = df.drop(columns=[TARGET])

y_actual = df[TARGET]

print(f"Features: {X.shape[1]}")


# ============================================================
# 4. LOAD MODEL
# ============================================================

print("\n[4] LOADING RANDOM FOREST")
print("-" * 70)

model = joblib.load(MODEL_FILE)

print("✓ Random Forest loaded")


# ============================================================
# 5. PREDICT
# ============================================================

print("\n[5] DETECTING NETWORK FLOWS")
print("-" * 70)

predictions = model.predict(X)

probabilities = model.predict_proba(X)

confidence = probabilities.max(axis=1)

print("✓ Prediction completed")


# ============================================================
# 6. CREATE ALERT DATA
# ============================================================

print("\n[6] CREATING SECURITY ALERTS")
print("-" * 70)

alerts = df.copy()

alerts["Predicted_Label"] = predictions

alerts["Confidence"] = confidence

alerts["Confidence_Percent"] = confidence * 100


# ============================================================
# 7. KEEP ONLY ATTACKS
# ============================================================

attack_alerts = alerts[
    alerts["Predicted_Label"] == 1
].copy()

print(f"Total test flows : {len(alerts):,}")
print(f"Attack alerts    : {len(attack_alerts):,}")


# ============================================================
# 8. SORT BY CONFIDENCE
# ============================================================

attack_alerts = attack_alerts.sort_values(
    by="Confidence",
    ascending=False
)


# ============================================================
# 9. SAVE ALERTS
# ============================================================

print("\n[7] SAVING SECURITY ALERTS")
print("-" * 70)

attack_alerts.to_csv(
    OUTPUT_FILE,
    index=False
)

print("✓ Security alerts saved")
print(OUTPUT_FILE)


# ============================================================
# 10. SHOW SAMPLE ALERTS
# ============================================================

print("\n[8] SAMPLE ATTACK ALERTS")
print("-" * 70)

display_columns = [
    "Predicted_Label",
    "Confidence_Percent"
]

print(
    attack_alerts[display_columns]
    .head(10)
    .to_string(index=False)
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("             STAGE 5 COMPLETED")
print("=" * 70)

print(f"Total flows checked : {len(alerts):,}")
print(f"Attack alerts       : {len(attack_alerts):,}")

print("\nOutput:")
print(OUTPUT_FILE)

print("=" * 70)