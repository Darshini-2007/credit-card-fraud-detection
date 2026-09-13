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
# Custom CSS
# ------------------------------------------------------------

st.markdown(
    """
    <style>

    h1 {
        font-size: 42px !important;
        font-weight: 700 !important;
    }

    h2 {
        font-size: 32px !important;
        font-weight: 650 !important;
    }

    h3 {
        font-size: 26px !important;
        font-weight: 600 !important;
    }

    p {
        font-size: 18px !important;
        line-height: 1.6 !important;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        font-size: 24px !important;
    }

    [data-testid="stSidebar"] p {
        font-size: 17px !important;
        line-height: 1.6 !important;
    }

    .stButton > button {
        font-size: 18px !important;
        font-weight: 600 !important;
        padding: 12px 22px !important;
        min-height: 45px !important;
        border-radius: 8px !important;
    }

    [data-testid="stRadio"] label {
        font-size: 18px !important;
    }

    [data-testid="stFileUploader"] {
        font-size: 18px !important;
    }

    button[data-baseweb="tab"] {
        font-size: 18px !important;
        font-weight: 600 !important;
    }

    [data-testid="stMetricValue"] {
        font-size: 30px !important;
        font-weight: 700 !important;
    }

    [data-testid="stMetricLabel"] {
        font-size: 18px !important;
    }

    [data-testid="stAlert"] {
        font-size: 17px !important;
    }

    [data-testid="stExpander"] {
        font-size: 17px !important;
    }

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
# Load Saved Model Files
# ------------------------------------------------------------

@st.cache_resource
def load_model_files():

    model = joblib.load("fraud_detection_model.pkl")
    threshold = joblib.load("fraud_threshold.pkl")
    feature_names = joblib.load("fraud_features.pkl")

    return model, threshold, feature_names


# ------------------------------------------------------------
# Load Model with Error Handling
# ------------------------------------------------------------

try:

    model, threshold, feature_names = load_model_files()

except FileNotFoundError:

    st.error(
        """
        Model files were not found.

        Please make sure these files are in the same folder as app.py:

        - fraud_detection_model.pkl
        - fraud_threshold.pkl
        - fraud_features.pkl
        """
    )

    st.stop()

except Exception as error:

    st.error(
        f"Error while loading the model files: {error}"
    )

    st.stop()


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
# Sidebar Information
# ------------------------------------------------------------

st.sidebar.header("📊 Model Information")

st.sidebar.write("**Model:** Random Forest Classifier")
st.sidebar.write(
    f"**Decision Threshold:** {threshold}"
)
st.sidebar.write(
    f"**Number of Features:** {len(feature_names)}"
)

st.sidebar.markdown("---")

st.sidebar.success("Model Status: Ready")
st.sidebar.info("Application Status: Online")

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

                non_numeric_columns = prediction_data.select_dtypes(
                    exclude=[np.number]
                ).columns.tolist()

                if non_numeric_columns:

                    st.error(
                        "These columns must contain numeric values: "
                        + ", ".join(non_numeric_columns)
                    )

                else:

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
        Test a sample transaction using the trained fraud detection model.
        These samples are provided for educational demonstration purposes.
        """
    )

    sample_type = st.radio(
        "Choose transaction type for testing:",
        [
            "Normal transaction",
            "Actual fraud transaction",
            "Random transaction"
        ]
    )

    if st.button(
        "🔍 Predict Sample Transaction",
        key="sample_prediction_button"
    ):

        try:

            # ------------------------------------------------
            # Create an empty transaction
            # ------------------------------------------------

            sample_transaction = pd.DataFrame(
                np.zeros(
                    (1, len(feature_names))
                ),
                columns=feature_names
            )

            # ------------------------------------------------
            # Normal Transaction
            # ------------------------------------------------

            if sample_type == "Normal transaction":

                if "Time" in feature_names:
                    sample_transaction["Time"] = 50000

                if "Amount" in feature_names:
                    sample_transaction["Amount"] = 100

                for feature in feature_names:

                    if feature not in ["Time", "Amount"]:
                        sample_transaction[feature] = 0.0

            # ------------------------------------------------
            # Actual Fraud Transaction Example
            # ------------------------------------------------

            elif sample_type == "Actual fraud transaction":

                fraud_values = {
                    "Time": 406,
                    "V1": -2.312227,
                    "V2": 1.951992,
                    "V3": -1.609851,
                    "V4": 3.997906,
                    "V5": -0.522188,
                    "V6": -1.426545,
                    "V7": -2.537387,
                    "V8": 1.391657,
                    "V9": -2.770089,
                    "V10": -2.772272,
                    "V11": 3.202033,
                    "V12": -2.899907,
                    "V13": -0.595222,
                    "V14": -4.289254,
                    "V15": 0.389724,
                    "V16": -1.140747,
                    "V17": -2.830056,
                    "V18": -0.016822,
                    "V19": 0.416956,
                    "V20": 0.126911,
                    "V21": 0.517232,
                    "V22": -0.035049,
                    "V23": -0.465211,
                    "V24": 0.320198,
                    "V25": 0.044519,
                    "V26": 0.177840,
                    "V27": 0.261145,
                    "V28": -0.143276,
                    "Amount": 0.00
                }

                for feature, value in fraud_values.items():

                    if feature in feature_names:
                        sample_transaction[feature] = value

            # ------------------------------------------------
            # Random Transaction
            # ------------------------------------------------

            else:

                rng = np.random.default_rng(42)

                for feature in feature_names:

                    if feature == "Time":

                        sample_transaction[feature] = rng.uniform(
                            0,
                            172800
                        )

                    elif feature == "Amount":

                        sample_transaction[feature] = rng.uniform(
                            1,
                            5000
                        )

                    else:

                        sample_transaction[feature] = rng.uniform(
                            -3,
                            3
                        )

            # ------------------------------------------------
            # Make Prediction
            # ------------------------------------------------

            fraud_probability = model.predict_proba(
                sample_transaction
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
                    sample_transaction,
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
