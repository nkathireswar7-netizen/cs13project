import json
import pandas as pd


INPUT_FILE = r"D:\CS13 PROJECT\DATA\ML\security_alerts.csv"

OUTPUT_FILE = r"D:\CS13 PROJECT\DATA\ML\ai_investigation_alerts.json"

print("=" * 70)
print("       CS-13 — STAGE 6")
print("       PREPARING ALERTS FOR AI INVESTIGATION AGENT")
print("=" * 70)


# 1. Load alerts
print("\n[1] Loading security alerts...")

df = pd.read_csv(INPUT_FILE)

print(f"✓ Alerts loaded: {len(df):,}")


# 2. Select high-confidence alerts
print("\n[2] Selecting high-confidence alerts...")

df = df[df["Confidence"] >= 0.90].copy()

print(f"✓ High-confidence alerts: {len(df):,}")


# 3. Take top 20 for the AI demo
df = df.head(20)


# 4. Convert to JSON-friendly records
print("\n[3] Creating AI investigation input...")

records = []

for index, row in df.iterrows():

    alert = {
        "alert_id": int(index),

        "prediction": "ATTACK",

        "confidence": round(
            float(row["Confidence"]),
            4
        ),

        "confidence_percent": round(
            float(row["Confidence_Percent"]),
            2
        ),

        "network_evidence": {}
    }

    # Add available network features
    for column in df.columns:

        if column not in [
            "Predicted_Label",
            "Confidence",
            "Confidence_Percent"
        ]:

            value = row[column]

            if pd.notna(value):
                alert["network_evidence"][column] = value.item() \
                    if hasattr(value, "item") else value

    records.append(alert)


# 5. Save JSON
print("\n[4] Saving AI investigation file...")

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        records,
        file,
        indent=2,
        default=str
    )


print("✓ JSON created")

print("\nOutput:")
print(OUTPUT_FILE)


# 6. Show first alert
if records:

    print("\n[5] FIRST ALERT")
    print("-" * 70)

    print(
        json.dumps(
            records[0],
            indent=2
        )
    )

else:

    print("\nNo alerts available.")


print("\n" + "=" * 70)
print("STAGE 6 COMPLETED")
print("=" * 70)