"""Improved Streamlit UI with API integration and enhanced threshold control.

Key improvements:
- Threshold control in prominent boxed container
- Confusion matrix with row-percentages (responsive)
- UI calls API (no local model inference)
"""

import os
from datetime import datetime

import httpx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score

# Load environment variables
load_dotenv()

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")
API_TOKEN = os.getenv("API_TOKEN", "your_super_secret_api_token")

# Risk thresholds
RISK_THRESHOLDS = {"Low": (0.0, 0.3), "Medium": (0.3, 0.7), "High": (0.7, 1.0)}

# Page config
st.set_page_config(
    page_title="HR Attrition Prediction",
    page_icon="📊",
    layout="wide",
)

# Session state initialization
if "predictions" not in st.session_state:
    st.session_state.predictions = None
if "current_threshold" not in st.session_state:
    st.session_state.current_threshold = 0.5


def render_threshold_control():
    """Render prominent threshold control in a highlighted box."""
    st.markdown(
        """
        <style>
        .threshold-container {
            background-color: #f0f2f6;
            border: 2px solid #4CAF50;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
        }
        .threshold-title {
            font-size: 18px;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="threshold-container">', unsafe_allow_html=True)
    st.markdown(
        '<div class="threshold-title">⚙️ High Risk Threshold Control</div>',
        unsafe_allow_html=True,
    )

    threshold = st.slider(
        "Set 'High Risk' Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Adjust this slider to change the probability score at which an employee is flagged as 'High Risk' (predicted to leave).",
        key="threshold_slider",
    )
    st.session_state.current_threshold = threshold

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Low Risk", f"< {threshold - 0.1:.0%}")
    with col2:
        st.metric("Medium Risk", f"{threshold - 0.1:.0%} - {threshold + 0.1:.0%}")
    with col3:
        st.metric("High Risk", f"≥ {threshold:.0%}")

    st.markdown("</div>", unsafe_allow_html=True)

    return threshold


def display_confusion_matrix_and_metrics(y_true, y_proba, threshold, title):
    """Display confusion matrix with row-normalized percentages.

    Args:
        y_true: Actual labels (0/1)
        y_proba: Predicted probabilities
        threshold: Classification threshold
        title: Chart title

    Returns:
        accuracy, recall, flagged_count
    """
    y_pred = (y_proba >= threshold).astype(int)

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Row-normalized percentages (rows sum to 100%)
    cm_normalized = np.zeros_like(cm, dtype=float)
    row_sums = cm.sum(axis=1)
    for i in range(cm.shape[0]):
        if row_sums[i] > 0:
            cm_normalized[i, :] = cm[i, :] / row_sums[i]

    # Create bordered container for confusion matrix
    with st.container():
        st.markdown(
            f"""
            <div style="border: 1px solid #ddd; border-radius: 5px; padding: 15px; background-color: #fafafa;">
                <h4 style="margin-top: 0;">{title}</h4>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Plot confusion matrix with reduced figure size
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".1%",
            cmap="Greens",
            cbar=False,
            xticklabels=["Predicted Stay", "Predicted Leave"],
            yticklabels=["Actual Stay", "Actual Leave"],
            ax=ax,
            annot_kws={"size": 11},
        )
        ax.set_ylabel("Actual Outcome", fontsize=12)
        ax.set_xlabel("Predicted Outcome", fontsize=12)
        plt.tight_layout()

        # Responsive display
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, zero_division=0)
    flagged_count = tp + fp

    return accuracy, recall, flagged_count


def call_prediction_api(employee_data: list[dict]) -> dict | None:
    """Call the FastAPI prediction endpoint.

    Args:
        employee_data: List of employee dictionaries (raw format)

    Returns:
        API response dict or None if error
    """
    headers = {"X-API-Key": API_TOKEN, "Content-Type": "application/json"}
    payload = {"employees": employee_data}

    try:
        with st.spinner("Calling prediction API..."):
            response = httpx.post(
                f"{API_URL}/predict",
                json=payload,
                headers=headers,
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        st.error(f"API Error ({e.response.status_code}): {e.response.text}")
        return None
    except httpx.RequestError as e:
        st.error(f"Connection Error: {e}")
        st.info(f"Make sure the API is running at {API_URL}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None


def load_sample_data():
    """Load sample employee data from the data directory."""
    try:
        eval_df = pd.read_csv("data/extrait_eval.csv")
        sirh_df = pd.read_csv("data/extrait_sirh.csv")
        sondage_df = pd.read_csv("data/extrait_sondage.csv")

        # Merge data (simplified - in production use proper cleaning)
        # This is just for demo purposes
        merged = eval_df.merge(
            sirh_df,
            left_on="eval_number",
            right_on="id_employee",
            how="inner",
            suffixes=("_eval", "_sirh"),
        )
        merged = merged.merge(sondage_df, left_on="id_employee", right_on="code_sondage", how="inner")

        return merged
    except FileNotFoundError:
        st.error("Sample data files not found in 'data/' directory.")
        return None


def main():
    """Main application."""
    st.title("📊 HR Attrition Prediction System")
    st.markdown("**Predict employee attrition risk using machine learning**")

    # Threshold control (prominent boxed design)
    st.header("1. Configure Prediction Threshold")
    threshold = render_threshold_control()

    # Data section
    st.header("2. Load Employee Data")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.info(
            "Upload employee CSV files or use sample data for testing. "
            "The system will call the prediction API to analyze attrition risk."
        )

    with col2:
        use_sample = st.button("Load Sample Data", type="primary")

    if use_sample:
        with st.spinner("Loading sample data..."):
            data = load_sample_data()
            if data is not None:
                st.session_state.employee_data = data
                st.success(f"Loaded {len(data)} employee records")

    # Show data preview
    if "employee_data" in st.session_state:
        with st.expander("📋 View Employee Data"):
            st.dataframe(
                st.session_state.employee_data.head(10),
                use_container_width=True,
            )

        # Prediction section
        st.header("3. Get Predictions")

        if st.button("Predict Attrition", type="primary"):
            # Prepare data for API call (convert to raw format)
            # NOTE: This is simplified - in production, ensure proper schema mapping
            employee_records = st.session_state.employee_data.head(5).to_dict("records")

            # Call API
            result = call_prediction_api(employee_records)

            if result:
                st.session_state.predictions = result
                st.success(
                    f"Received predictions for {len(result['predictions'])} employees"
                )

    # Display results
    if st.session_state.predictions:
        st.header("4. Prediction Results")

        predictions = st.session_state.predictions["predictions"]

        # Create DataFrame for display
        results_df = pd.DataFrame(predictions)

        # Display results table
        st.subheader("Employee Risk Assessment")
        st.dataframe(
            results_df[
                [
                    "id_employee",
                    "prediction",
                    "probability",
                    "risk_category",
                ]
            ],
            use_container_width=True,
        )

        # Risk distribution
        st.subheader("Risk Distribution")
        risk_counts = results_df["risk_category"].value_counts()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Low Risk",
                risk_counts.get("Low", 0),
                delta="Safe",
                delta_color="normal",
            )
        with col2:
            st.metric(
                "Medium Risk",
                risk_counts.get("Medium", 0),
                delta="Monitor",
                delta_color="off",
            )
        with col3:
            st.metric(
                "High Risk",
                risk_counts.get("High", 0),
                delta="Action needed",
                delta_color="inverse",
            )

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #7f8c8d;'>"
        f"API Status: Connected to {API_URL} | "
        f"Threshold: {threshold:.0%}"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
