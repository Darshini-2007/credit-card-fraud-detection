# ============================================================
# CREDIT CARD FRAUD DETECTION - PREDICTION FILE
# ============================================================

import pandas as pd
import joblib


# -----------------------------
# 1. Load Saved Files
# -----------------------------

model = joblib.load("fraud_detection_model.pkl")
threshold = joblib.load("fraud_threshold.pkl")
feature_names = joblib.load("fraud_features.pkl")

print("Saved model loaded successfully!")
print("Selected threshold:", threshold)


# -----------------------------
# 2. Load Dataset for Demo
# -----------------------------

df = pd.read_csv("creditcard.csv", encoding="latin1")

# Remove duplicates in the same way as training
df = df.drop_duplicates()

# Separate features and target
X = df.drop("Class", axis=1)
y = df["Class"]


# -----------------------------
# 3. Select One Transaction
# -----------------------------

# Select the first transaction as a demo
transaction = X.iloc[0]

# Convert it into a DataFrame
transaction_df = pd.DataFrame(
    [transaction],
    columns=feature_names
)


# -----------------------------
# 4. Predict Fraud Probability
# -----------------------------

fraud_probability = model.predict_proba(
    transaction_df
)[0][1]


# -----------------------------
# 5. Apply Selected Threshold
# -----------------------------

if fraud_probability >= threshold:
    prediction = "Fraudulent Transaction"
else:
    prediction = "Normal Transaction"


# -----------------------------
# 6. Display Result
# -----------------------------

print("\n========================================")
print("TRANSACTION PREDICTION")
print("========================================")

print("Prediction:", prediction)
print(
    f"Fraud Probability: {fraud_probability * 100:.2f}%"
)