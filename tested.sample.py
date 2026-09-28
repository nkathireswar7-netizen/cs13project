import joblib
import pandas as pd

MODEL_FILE = r"D:\CS13 PROJECT\DATA\ML\random_forest_nids.joblib"
NEW_DATA_FILE = r"C:\Users\CHAKRAVARHTY\Downloads\cs13_new_validation_dataset.csv"

# Load model
model = joblib.load(MODEL_FILE)

# Load new dataset
df = pd.read_csv(NEW_DATA_FILE)

print("New dataset loaded")
print("Rows:", len(df))
print("Columns:", len(df.columns))

# Make sure feature order matches the trained model
expected_features = list(model.feature_names_in_)
X_new = df[expected_features]

# Predict
predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)
confidence = probabilities.max(axis=1)

# Display results
print("\n========== NEW DATASET PREDICTIONS ==========")

for i in range(len(predictions)):
    result = "ATTACK" if predictions[i] == 1 else "BENIGN"

    print(
        f"Record {i + 1:02d} → "
        f"{result:<7} | "
        f"Confidence: {confidence[i] * 100:.2f}%"
    )

# Summary
attack_count = sum(predictions == 1)
benign_count = sum(predictions == 0)

print("\n========== SUMMARY ==========")
print("Total records :", len(predictions))
print("Benign        :", benign_count)
print("Attack        :", attack_count)