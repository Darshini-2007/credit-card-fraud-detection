# ============================================================
# CREDITGUARD AI - CREDIT CARD FRAUD DETECTION WEB APP
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------

st.set_page_config(
    page_title="CreditGuard AI",
    page_icon="💳",
    layout="wide"
)


# ------------------------------------------------------------
# Custom CSS - Increase Font Sizes and Improve UI
# ------------------------------------------------------------

st.markdown(
    """
    <style>

    /* Main application title */
    h1 {
        font-size: 42px !important;
        font-weight: 700 !important;
    }

    /* Main section headings */
    h2 {
        font-size: 32px !important;
        font-weight: 650 !important;
    }

    /* Smaller headings */
    h3 {
        font-size: 26px !important;
        font-weight: 600 !important;
    }

    /* Normal paragraph text */
    p {
        font-size: 18px !important;
        line-height: 1.6 !important;
    }

    /* Sidebar heading */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        font-size: 24px !important;
    }

    /* Sidebar text */
    [data-testid="stSidebar"] p {
        font-size: 17px !important;
        line-height: 1.6 !important;
    }

    /* Buttons */
    .stButton > button {
        font-size: 18px !important;
        font-weight: 600 !important;
        padding: 12px 22px !important;
        min-height: 45px !important;
        border-radius: 8px !important;
    }

    /* Radio button labels */
    [data-testid="stRadio"] label {
        font-size: 18px !important;
    }

    /* File uploader text */
    [data-testid="stFileUploader"] {
        font-size: 18px !important;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        font-size: 18px !important;
        font-weight: 600 !important;
    }

    /* Metric values */
    [data-testid="stMetricValue"] {
        font-size: 30px !important;
        font-weight: 700 !important;
    }

    /* Metric labels */
    [data-testid="stMetricLabel"] {
        font-size: 18px !important;
    }

    /* Success, warning, and info messages */
    [data-testid="stAlert"] {
        font-size: 17px !important;
    }

    /* Expanders */
    [data-testid="stExpander"] {
        font-size: 17px !important;
    }

    /* Dataframe text */
    [data-testid="stDataFrame"] {
        font-size: 16px !important;
    }

    /* Footer */
    .footer-text {
        text-align: center;
        font-size: 15px !important;
        margin-top: 30px;
        padding: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# Load Saved Model
# ------------------------------------------------------------

@st.cache_resource
def load_model_files():
    model = joblib.load("fraud_detection_model.pkl")
    threshold = joblib.load("fraud_threshold.pkl")
    feature_names = joblib.load("fraud_features.pkl")

    return model, threshold, feature_names


# ------------------------------------------------------------
# Application Header
# ------------------------------------------------------------

st.title("💳 CreditGuard AI")

st.subheader("Credit Card Fraud Detection System")

st.write(
    """
    CreditGuard AI is a machine learning application that analyzes
    credit card transactions and identifies potentially fraudulent activity.
    """
)

st.info(
    "This is an educational demonstration and should not be used "
    "as the only decision-making system for real financial transactions."
)


# ------------------------------------------------------------
# Load Model with Error Handling
# ------------------------------------------------------------

try:
    model, threshold, feature_names = load_model_files()

except FileNotFoundError:
    st.error(
        """
        Saved model files were not found.

        Please run `fraud_detection.py` first to generate:

        - fraud_detection_model.pkl
        - fraud_threshold.pkl
        - fraud_features.pkl
        """
    )

    st.stop()


# ------------------------------------------------------------
# Sidebar Information
# ------------------------------------------------------------

st.sidebar.header("📊 Model Information")

st.sidebar.write("**Model:** Random Forest Classifier")
st.sidebar.write(f"**Decision Threshold:** {threshold}")
st.sidebar.write(f"**Number of Features:** {len(feature_names)}")

st.sidebar.markdown("---")

st.sidebar.write(
    """
    **Purpose**

    This application predicts whether a credit card
    transaction is normal or potentially fraudulent.
    """
)


# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------

tab1, tab2, tab3 = st.tabs(
    [
        "📄 Upload CSV",
        "🧪 Test Sample",
        "ℹ️ About Project"
    ]
)


# ============================================================
# TAB 1: UPLOAD CSV
# ============================================================

with tab1:

    st.header("📄 Upload Transaction CSV")

    st.write(
        """
        Upload a CSV file containing transaction data.
        The file must contain the same 30 feature columns
        used during model training.
        """
    )

    st.warning(
        """
        Required columns: Time, V1 to V28, and Amount.
        The Class column is not required for prediction.
        """
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            uploaded_df = pd.read_csv(uploaded_file)

            st.subheader("Uploaded Data Preview")

            st.dataframe(
                uploaded_df.head(),
                use_container_width=True
            )

            missing_features = [
                feature
                for feature in feature_names
                if feature not in uploaded_df.columns
            ]

            if missing_features:

                st.error(
                    "The uploaded file is missing these columns: "
                    + ", ".join(missing_features)
                )

            else:

                prediction_data = uploaded_df[feature_names]

                fraud_probabilities = model.predict_proba(
                    prediction_data
                )[:, 1]

                predictions = (
                    fraud_probabilities >= threshold
                ).astype(int)

                results_df = uploaded_df.copy()

                results_df["Fraud Probability (%)"] = (
                    fraud_probabilities * 100
                ).round(2)

                results_df["Prediction"] = np.where(
                    predictions == 1,
                    "Fraudulent Transaction",
                    "Normal Transaction"
                )

                st.success(
                    "Prediction completed successfully!"
                )

                st.subheader("Prediction Results")

                st.dataframe(
                    results_df,
                    use_container_width=True
                )

                normal_count = int(
                    (predictions == 0).sum()
                )

                fraud_count = int(
                    (predictions == 1).sum()
                )

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Total Transactions",
                        len(results_df)
                    )

                with col2:
                    st.metric(
                        "Normal Transactions",
                        normal_count
                    )

                with col3:
                    st.metric(
                        "Fraudulent Transactions",
                        fraud_count
                    )

                output_csv = results_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="⬇️ Download Prediction Results",
                    data=output_csv,
                    file_name="fraud_predictions.csv",
                    mime="text/csv"
                )

        except Exception as error:

            st.error(
                f"Error while processing the uploaded file: {error}"
            )


# ============================================================
# TAB 2: TEST SAMPLE
# ============================================================

with tab2:

    st.header("🧪 Test a Sample Transaction")

    st.write(
        """
        Select a transaction from the original dataset
        and use the trained model to predict its class.
        """
    )

    sample_type = st.radio(
        "Choose transaction type for testing:",
        [
            "First transaction",
            "Actual fraud transaction",
            "Random transaction"
        ]
    )

    if st.button(
        "🔍 Predict Sample Transaction",
        key="sample_prediction_button"
    ):

        try:

            dataset = pd.read_csv(
                "creditcard.csv",
                encoding="latin1"
            )

            dataset = dataset.drop_duplicates()

            X = dataset.drop("Class", axis=1)
            y = dataset["Class"]

            if sample_type == "First transaction":

                transaction = X.iloc[0]

            elif sample_type == "Actual fraud transaction":

                fraud_transactions = X[y == 1]

                if len(fraud_transactions) == 0:

                    st.error(
                        "No fraudulent transaction found."
                    )

                    st.stop()

                transaction = fraud_transactions.iloc[0]

            else:

                transaction = X.sample(
                    n=1,
                    random_state=None
                ).iloc[0]

            transaction_df = pd.DataFrame(
                [transaction],
                columns=feature_names
            )

            fraud_probability = model.predict_proba(
                transaction_df
            )[0][1]

            prediction = (
                "Fraudulent Transaction"
                if fraud_probability >= threshold
                else "Normal Transaction"
            )

            st.subheader("Prediction Result")

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Prediction",
                    prediction
                )

            with col2:

                st.metric(
                    "Fraud Probability",
                    f"{fraud_probability * 100:.2f}%"
                )

            if prediction == "Fraudulent Transaction":

                st.error(
                    "⚠️ The model classified this transaction "
                    "as potentially fraudulent."
                )

            else:

                st.success(
                    "✅ The model classified this transaction "
                    "as normal."
                )

            with st.expander("👁️ View Transaction Features"):

                st.dataframe(
                    transaction_df,
                    use_container_width=True
                )

        except Exception as error:

            st.error(
                f"Error while predicting the sample: {error}"
            )


# ============================================================
# TAB 3: ABOUT PROJECT
# ============================================================

with tab3:

    st.header("ℹ️ About This Project")

    st.markdown(
        """
        ### 🎯 Objective

        The objective of this project is to detect fraudulent
        credit card transactions using machine learning.

        ### 📊 Dataset

        The dataset contains anonymized transaction features:

        - Time
        - V1 to V28
        - Amount
        - Class

        ### 🤖 Machine Learning Model

        A Random Forest Classifier is used for classification.

        ### ⚙️ Preprocessing

        - Missing-value checking
        - Duplicate removal
        - Feature-target separation
        - Stratified train-test split
        - Class imbalance handling

        ### 📈 Evaluation Metrics

        - Precision
        - Recall
        - F1-score
        - ROC-AUC
        - PR-AUC
        - Confusion Matrix

        ### ⚠️ Important Note

        Fraud detection is an imbalanced classification problem.
        Therefore, precision and recall are more useful than
        accuracy alone.
        """
    )


# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------

st.markdown("---")

st.markdown(
    """
    <div class="footer-text">
        💳 CreditGuard AI | Credit Card Fraud Detection |
        Machine Learning Project
    </div>
    """,
    unsafe_allow_html=True
)