import json # Import json for debugging
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for matplotlib
import pandas as pd
import streamlit as st

pd.set_option("future.no_silent_downcasting", True)  # This was inserted here
import base64  # Import base64 for embedding images
import io
import os
import sys  # Import sys
import httpx # Import httpx for API calls
from datetime import UTC, datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # Import seaborn for enhanced plotting
import shap
import streamlit.components.v1 as components
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))


def _logit(p: float) -> float:
    """Inverse of sigmoid function."""
    if p <= 0:
        return -np.inf
    if p >= 1:
        return np.inf
    return np.log(p / (1 - p))


# --- Configuration ---
# Define risk categories for Excel/HTML reports (probability-based, for confusion matrix)
RISK_THRESHOLDS = {"Low": (0.0, 0.3), "Medium": (0.3, 0.7), "High": (0.7, 1.0)}

# Define risk categories for log-odds (f(x))
# Corresponding log-odds for probabilities 0.3 and 0.7
# logit(0.3) approx -0.847
# logit(0.7) approx 0.847
LOG_ODDS_RISK_THRESHOLDS = {
    "Low": (-np.inf, _logit(0.3)),  # f(x) < -0.847
    "Medium": (_logit(0.3), _logit(0.7)),  # -0.847 <= f(x) < 0.847
    "High": (_logit(0.7), np.inf),  # f(x) >= 0.847
}

REQUIRED_FILES = ["extrait_eval.csv", "extrait_sirh.csv", "extrait_sondage.csv"]

API_BASE_URL = "http://localhost:8001" # FastAPI endpoint

# --- Session State Initialization ---
if "prediction_triggered" not in st.session_state:
    st.session_state.prediction_triggered = False
if "report_data" not in st.session_state:
    st.session_state.report_data = None
if "shap_html_content" not in st.session_state:
    st.session_state.shap_html_content = None
if "excel_report_data" not in st.session_state:
    st.session_state.excel_report_data = None
# The following session state variables are no longer needed as SHAP values come from the API
# if "processed_data_for_shap" not in st.session_state:
#     st.session_state.processed_data_for_shap = None
# if "explainer" not in st.session_state:
#     st.session_state.explainer = None
# if "all_features" not in st.session_state:
#     st.session_state.all_features = None


def get_project_root():
    """Returns the absolute path to the project root."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def _load_local_csv_files():
    """Loads the required CSV files from the local 'data' directory."""
    try:
        project_root = get_project_root()
        eval_file_path = os.path.join(project_root, "data", "extrait_eval.csv")
        sirh_file_path = os.path.join(project_root, "data", "extrait_sirh.csv")
        sondage_file_path = os.path.join(project_root, "data", "extrait_sondage.csv")

        # Read files into BytesIO objects to mimic uploaded files
        with open(eval_file_path, "rb") as f:
            eval_file = io.BytesIO(f.read())
            eval_file.name = "extrait_eval.csv"
        with open(sirh_file_path, "rb") as f:
            sirh_file = io.BytesIO(f.read())
            sirh_file.name = "extrait_sirh.csv"
        with open(sondage_file_path, "rb") as f:
            sondage_file = io.BytesIO(f.read())
            sondage_file.name = "extrait_sondage.csv"

        return eval_file, sirh_file, sondage_file
    except FileNotFoundError as e:
        st.error(f"Required data file not found: {e}. Please ensure 'data' directory "
                 "contains 'extrait_eval.csv', 'extrait_sirh.csv', and 'extrait_sondage.csv'.")
        return None, None, None


def _call_prediction_api(eval_data: list[dict], sirh_data: list[dict], sondage_data: list[dict]) -> dict:
    """Calls the FastAPI /predict endpoint with raw employee data."""
    try:
        payload = {
            "eval_data": eval_data,
            "sirh_data": sirh_data,
            "sondage_data": sondage_data,
        }
        # Set a reasonable timeout, e.g., 60 seconds
        response = httpx.post(f"{API_BASE_URL}/predict", json=payload, timeout=60.0)
        response.raise_for_status()
        return response.json()
    except httpx.TimeoutException as e:
        print(f"API request timed out: {e}")
        st.error(f"The prediction API took too long to respond (timeout). Please try again or check the API server status.")
        return {"predictions": []}
    except httpx.RequestError as e:
        print(f"Network error while connecting to API: {e}")
        st.error(f"Network error while connecting to API: {e}. Please ensure the API server is running and accessible.")
        return {"predictions": []}
    except httpx.HTTPStatusError as e:
        print(f"API returned an error: {e.response.status_code} - {e.response.text}")
        st.error(f"API returned an error: {e.response.status_code} - {e.response.text}. Please check the API logs for details.")
        return {"predictions": []}
    except Exception as e:
        print(f"An unexpected error occurred during API call: {e}")
        st.error(f"An unexpected error occurred during API call: {e}")
        return {"predictions": []}


def _handle_file_uploads_and_predict(main_threshold: float) -> None:
    st.subheader("Upload Employee Data for Prediction")
    uploaded_files = st.file_uploader(
        "Upload three CSV files: `extrait_eval.csv`, "
        "`extrait_sirh.csv`, `extrait_sondage.csv`",
        type=["csv"],
        accept_multiple_files=True,
    )

    eval_file, sirh_file, sondage_file = None, None, None
    files_source = "uploaded" # To track if files came from uploader or local

    if uploaded_files:
        if len(uploaded_files) != len(REQUIRED_FILES):
            st.warning("Please upload all three required CSV files.")
            return

        uploaded_file_names = [file.name for file in uploaded_files]
        if not all(name in uploaded_file_names for name in REQUIRED_FILES):
            st.warning(
                "Please make sure to upload the three required files: "
                "`extrait_eval.csv`, `extrait_sirh.csv`, and `extrait_sondage.csv`."
            )
            return

        file_map = {file.name: file for file in uploaded_files}
        eval_file = file_map.get("extrait_eval.csv")
        sirh_file = file_map.get("extrait_sirh.csv")
        sondage_file = file_map.get("extrait_sondage.csv")
    else:
        # If no files uploaded, try to load from local data directory for testing
        st.info("No files uploaded. Attempting to load from local 'data' directory for testing.")
        eval_file, sirh_file, sondage_file = _load_local_csv_files()
        files_source = "local"

    if eval_file and sirh_file and sondage_file:
        predict_button = st.button("Predict Attrition", key="Predict Attrition")

        if predict_button:
            with st.spinner("Processing data and getting predictions..."):
                try:
                    eval_df = pd.read_csv(eval_file)
                    sirh_df = pd.read_csv(sirh_file)
                    sondage_df = pd.read_csv(sondage_file)

                    # Convert raw DataFrames to list of dicts for API
                    eval_data_for_api = eval_df.to_dict(orient="records")
                    sirh_data_for_api = sirh_df.to_dict(orient="records")
                    sondage_data_for_api = sondage_df.to_dict(orient="records")

                    api_response = _call_prediction_api(
                        eval_data_for_api, sirh_data_for_api, sondage_data_for_api
                    )
                    
                    # Save API response to a temporary JSON file for debugging
                    temp_api_response_path = os.path.join(get_project_root(), "temp_api_response.json")
                    with open(temp_api_response_path, "w") as f:
                        json.dump(api_response, f, indent=4)
                    st.info(f"For debugging, API response saved to: {temp_api_response_path}")

                    if api_response and api_response["predictions"]:
                        predictions_data = api_response["predictions"]
                        
                        # Prepare report_data
                        report_data = pd.DataFrame([
                            {
                                "id_employee": p["id_employee"],
                                "Attrition_Risk_Percentage": p["probability"],
                                "Prediction": p["prediction"],
                                "Risk_Attrition": p["risk_category"],
                                "shap_values": p.get("shap_values"),
                                "base_value": p.get("base_value"),
                                "feature_names": p.get("feature_names"),
                            }
                            for p in predictions_data
                        ])
                        st.session_state.report_data = report_data
                        st.session_state.prediction_triggered = True
                        st.success("Predictions received successfully!")
                    else:
                        st.error("No predictions received from the API.")

                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
        elif files_source == "uploaded":
            st.info("Please upload the CSV files to get started.")
        elif files_source == "local":
            st.info("Local files loaded. Click 'Predict Attrition' to proceed.")
    else:
        st.error("Could not load required CSV files.")


def clear_prediction_results() -> None:
    """Clear all prediction results and reset session state."""
    st.session_state.prediction_triggered = False
    st.session_state.report_data = None
    st.session_state.shap_report_data = None
    st.session_state.excel_report_data = None


# --- Helper Functions (from train.py) ---
def generate_shap_report_data(
    report_data: pd.DataFrame,
) -> list[dict]:
    """Generate SHAP waterfall plot data as base64 encoded images.

    Returns a list of dictionaries, each containing employee details and a
    base64 encoded PNG image of the SHAP waterfall plot.
    """
    shap_report_items = []

    for i, (_, row) in enumerate(report_data.iterrows()):
        employee_id = row.get("id_employee", f"Employee {i+1}")
        risk_category = row["Risk_Attrition"]
        attrition_prob = row["Attrition_Risk_Percentage"]
        prediction_type = row["Prediction"]
        shap_values_row = row["shap_values"]
        base_value_row = row["base_value"]
        feature_names_row = row.get("feature_names")

        if shap_values_row is None or base_value_row is None:
            # Skip if SHAP values are not available for this prediction
            continue

        # Use feature names from API if available, otherwise use generic names
        if feature_names_row and len(feature_names_row) == len(shap_values_row):
            feature_names = feature_names_row
        else:
            feature_names = [f"Feature {j}" for j in range(len(shap_values_row))]

        # Create a dummy Explanation object
        # The data parameter is optional for plotting, but can be useful
        # For now, we'll just pass the shap_values and base_value
        explanation = shap.Explanation(
            values=np.array(shap_values_row),
            base_values=base_value_row,
            data=np.zeros(len(shap_values_row)), # Dummy data
            feature_names=feature_names
        )

        shap.plots.waterfall(
            explanation,
            max_display=10,
            show=False,
        )
        fig = plt.gcf()
        fig.set_size_inches(8, 6)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        plt.close(fig)
        img_str = base64.b64encode(buf.getvalue()).decode("utf-8")

        shap_report_items.append(
            {
                "employee_id": employee_id,
                "risk_category": risk_category,
                "attrition_prob": attrition_prob,
                "prediction_type": prediction_type,
                "img_str": img_str,
            }
        )
    return shap_report_items
def main() -> None:
    """Run the Streamlit application."""
    # --- Streamlit App Layout ---
    st.set_page_config(layout="wide")
    st.title("Employee Attrition Risk")

    # --- Threshold Slider ---
    st.subheader("Adjust Prediction Threshold")
    main_threshold = st.slider(
        "Select Probability Threshold",
        0.0,
        1.0,
        0.5,
        0.01,
        help="Adjust this threshold to see how it impacts the model's "
        "classification on the training data.",
    )

    st.markdown("---")

    _handle_file_uploads_and_predict(main_threshold)

    # --- Display Results (if triggered) ---
    if st.session_state.prediction_triggered:
        st.markdown("---")
        st.subheader("Prediction Results and Reports")

        report_data = st.session_state.report_data
        # Removed: x_transformed_for_shap, explainer, all_features
        excel_tab2_data = st.session_state.excel_report_data

        # --- Generate Excel Report ---
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            # Tab 1: Summary (no employee name, no extra columns)
            tab1_df = report_data[
                [
                    "id_employee",
                    "Risk_Attrition",
                    "Attrition_Risk_Percentage",
                    "Prediction",
                ]
            ].copy()
            tab1_df.rename(columns={"id_employee": "Employee_ID"}, inplace=True)
            tab1_df.to_excel(writer, sheet_name="Summary", index=False)

            # Tab 2: Features (all features with coefficients; no employee name)
            excel_tab2_data = []
            for idx, row in report_data.iterrows():
                employee_id = row["id_employee"]
                shap_values_row = row.get("shap_values")
                base_value_row = row.get("base_value")
                feature_names_row = row.get("feature_names")
                prediction_label = row["Prediction"]

                if shap_values_row is not None:
                    # Use feature names from API if available, otherwise use generic names
                    if feature_names_row and len(feature_names_row) == len(shap_values_row):
                        feature_names = feature_names_row
                    else:
                        feature_names = [f"Feature {j}" for j in range(len(shap_values_row))]

                    employee_shap_df = pd.DataFrame(
                        {
                            "Feature": feature_names,
                            "Coefficient": shap_values_row,
                        }
                    )
                    employee_shap_df["Employee_ID"] = employee_id
                    employee_shap_df["Prediction"] = prediction_label
                    excel_tab2_data.append(employee_shap_df)
            
            if excel_tab2_data:
                st.session_state.excel_report_data = pd.concat(excel_tab2_data)
            else:
                st.session_state.excel_report_data = pd.DataFrame() # Empty if no SHAP data

            tab2_df = st.session_state.excel_report_data.copy()
            # Ensure column names are exactly as required
            tab2_df.rename(
                columns={
                    "Employee_ID": "Employee_ID",
                    "Feature": "Feature",
                    "Coefficient": "Coefficient",
                },
                inplace=True,
            )
            tab2_df[["Employee_ID", "Feature", "Coefficient", "Prediction"]].to_excel(
                writer, sheet_name="Features", index=False
            )

            # Tab 3: Metrics (optional)
            summary_metrics_df = pd.DataFrame(
                {
                    "Metric": [
                        "Total Employees Processed",
                        "Predicted to Leave",
                        "Predicted to Stay",
                    ],
                    "Value": [
                        len(report_data),
                        report_data["Prediction"].value_counts().get("Leave", 0),
                        report_data["Prediction"].value_counts().get("Stay", 0),
                    ],
                }
            )
            summary_metrics_df.to_excel(writer, sheet_name="Metrics", index=False)

        excel_buffer.seek(0)
        st.download_button(
            label="Download Excel Report",
            data=excel_buffer,
            file_name="employee_attrition_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        # Generate SHAP report data
        shap_report_data = generate_shap_report_data(report_data)
        st.session_state.shap_report_data = shap_report_data

        # --- Display SHAP Visualization Report ---
        st.subheader("Employee Attrition SHAP Explanations")
        shap_report_items = st.session_state.shap_report_data
        if shap_report_items:
            for item in shap_report_items:
                st.markdown(f"### Employee ID: {item['employee_id']}")
                st.markdown(
                    f"Predicted Attrition Risk: **{item['risk_category']}** "
                    f"({item['attrition_prob']:.1%}) · Prediction: "
                    f"**{item['prediction_type']}**"
                )
                st.image(
                    f"data:image/png;base64,{item['img_str']}",
                    caption=f"SHAP Waterfall Plot for Employee {item['employee_id']}",
                    use_container_width=True,
                )
                st.markdown("---")
        else:
            st.info("No SHAP reports generated yet.")

        st.success("Reports generated successfully!")


if __name__ == "__main__":
    main()