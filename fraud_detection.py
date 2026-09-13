# ============================================================
# CREDIT CARD FRAUD DETECTION
# Random Forest Classification Project
# ============================================================

# -----------------------------
# 1. Import Libraries
# -----------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay
)


# -----------------------------
# 2. Load Dataset
# -----------------------------

df = pd.read_csv("creditcard.csv", encoding="latin1")

print("\nDataset loaded successfully!")

print("\nFirst five rows:")
print(df.head())

print("\nDataset shape before preprocessing:")
print(df.shape)


# -----------------------------
# 3. Basic Dataset Information
# -----------------------------

print("\nColumn names:")
print(df.columns)

print("\nDataset information:")
print(df.info())

print("\nMissing values in each column:")
print(df.isnull().sum())

print("\nClass distribution before preprocessing:")
print(df["Class"].value_counts())


# -----------------------------
# 4. Remove Duplicate Rows
# -----------------------------

duplicate_count = df.duplicated().sum()

print("\nNumber of duplicate rows:", duplicate_count)

df = df.drop_duplicates()

print("Dataset shape after removing duplicates:")
print(df.shape)


# -----------------------------
# 5. Analyze Class Distribution
# -----------------------------

class_counts = df["Class"].value_counts()

print("\nClass distribution after preprocessing:")
print(class_counts)

print("\nClass distribution in percentage:")
print(df["Class"].value_counts(normalize=True) * 100)


# Plot class distribution
plt.figure(figsize=(6, 4))
class_counts.plot(kind="bar")

plt.title("Normal vs Fraudulent Transactions")
plt.xlabel("Class")
plt.ylabel("Number of Transactions")
plt.xticks(
    ticks=[0, 1],
    labels=["Normal", "Fraud"],
    rotation=0
)

plt.tight_layout()
plt.show()


# -----------------------------
# 6. Separate Features and Target
# -----------------------------

X = df.drop("Class", axis=1)
y = df["Class"]

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)


# -----------------------------
# 7. Split Dataset
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())


# -----------------------------
# 8. Train Random Forest Model
# -----------------------------

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

print("\nTraining Random Forest model...")

rf_model.fit(X_train, y_train)

print("Model training completed!")


# -----------------------------
# 9. Predict Probabilities
# -----------------------------

y_prob = rf_model.predict_proba(X_test)[:, 1]

# Default threshold
default_threshold = 0.50

y_pred_default = (y_prob >= default_threshold).astype(int)


# -----------------------------
# 10. Evaluate Using Default Threshold
# -----------------------------

print("\n========================================")
print("EVALUATION AT DEFAULT THRESHOLD 0.50")
print("========================================")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_default))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_default))

roc_auc = roc_auc_score(y_test, y_prob)
pr_auc = average_precision_score(y_test, y_prob)

print("ROC-AUC Score:", roc_auc)
print("PR-AUC Score:", pr_auc)


# -----------------------------
# 11. Test Different Thresholds
# -----------------------------

thresholds = [0.50, 0.30, 0.20, 0.10, 0.05]

print("\n========================================")
print("THRESHOLD COMPARISON")
print("========================================")

for threshold in thresholds:
    y_pred = (y_prob >= threshold).astype(int)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    print(f"\nThreshold: {threshold}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")


# -----------------------------
# 12. Select Final Threshold
# -----------------------------

FINAL_THRESHOLD = 0.20

y_pred_final = (
    y_prob >= FINAL_THRESHOLD
).astype(int)


# -----------------------------
# 13. Final Model Evaluation
# -----------------------------

print("\n========================================")
print("FINAL MODEL EVALUATION")
print("========================================")

print("Selected threshold:", FINAL_THRESHOLD)

final_cm = confusion_matrix(y_test, y_pred_final)

print("\nFinal Confusion Matrix:")
print(final_cm)

print("\nFinal Classification Report:")
print(classification_report(y_test, y_pred_final))

final_precision = precision_score(
    y_test,
    y_pred_final,
    zero_division=0
)

final_recall = recall_score(
    y_test,
    y_pred_final,
    zero_division=0
)

final_f1 = f1_score(
    y_test,
    y_pred_final,
    zero_division=0
)

print("Final Precision:", final_precision)
print("Final Recall:", final_recall)
print("Final F1 Score:", final_f1)
print("Final ROC-AUC:", roc_auc)
print("Final PR-AUC:", pr_auc)


# -----------------------------
# 14. Display Final Confusion Matrix
# -----------------------------

disp = ConfusionMatrixDisplay(
    confusion_matrix=final_cm,
    display_labels=["Normal", "Fraud"]
)

disp.plot()

plt.title("Final Confusion Matrix")
plt.tight_layout()
plt.show()


# -----------------------------
# 15. Save Model and Supporting Files
# -----------------------------

joblib.dump(
    rf_model,
    "fraud_detection_model.pkl"
)

joblib.dump(
    FINAL_THRESHOLD,
    "fraud_threshold.pkl"
)

joblib.dump(
    list(X.columns),
    "fraud_features.pkl"
)

print("\n========================================")
print("FILES SAVED SUCCESSFULLY")
print("========================================")

print("fraud_detection_model.pkl")
print("fraud_threshold.pkl")
print("fraud_features.pkl")


# -----------------------------
# 16. Prediction Function
# -----------------------------

def predict_transaction(transaction_data):
    """
    Predict whether a transaction is normal or fraudulent.

    transaction_data must contain the same 30 features
    used during model training.
    """

    model = joblib.load("fraud_detection_model.pkl")
    threshold = joblib.load("fraud_threshold.pkl")
    feature_names = joblib.load("fraud_features.pkl")

    transaction_df = pd.DataFrame(
        [transaction_data],
        columns=feature_names
    )

    probability = model.predict_proba(
        transaction_df
    )[0][1]

    if probability >= threshold:
        result = "Fraudulent Transaction"
    else:
        result = "Normal Transaction"

    return result, probability


# -----------------------------
# 17. Test One Transaction
# -----------------------------

sample_transaction = X_test.iloc[0].to_dict()

result, probability = predict_transaction(
    sample_transaction
)

print("\n========================================")
print("SAMPLE TRANSACTION PREDICTION")
print("========================================")

print("Prediction:", result)
print(f"Fraud Probability: {probability * 100:.2f}%")