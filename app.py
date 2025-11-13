import matplotlib

matplotlib.use('Agg') # Use non-interactive backend for matplotlib
import pandas as pd
import httpx
import streamlit as st
from ui_config import UI_TEXTS

pd.set_option("future.no_silent_downcasting", True)  # This was inserted here
import base64  # Import base64 for embedding images
import io
import sys  # Import sys
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

import os

API_URL = os.getenv("API_BASE_URL", "http://localhost:8000") # Assuming FastAPI is running on this address
API_TOKEN = os.getenv("API_TOKEN") # Load API token from environment variable

from core.data_processing import clean_raw_input, engineer_features, clean_and_engineer_features


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

# --- Session State Initialization ---
if "prediction_triggered" not in st.session_state:
    st.session_state.prediction_triggered = False
if "report_data" not in st.session_state:
    st.session_state.report_data = None
if "shap_html_content" not in st.session_state:
    st.session_state.shap_html_content = None
if "excel_report_data" not in st.session_state:
    st.session_state.excel_report_data = None
if "processed_data_for_shap" not in st.session_state:
    st.session_state.processed_data_for_shap = None
if "explainer" not in st.session_state:
    st.session_state.explainer = None
if "all_features" not in st.session_state:
    st.session_state.all_features = None
if "shap_values" not in st.session_state:
    st.session_state.shap_values = None
if "expected_value" not in st.session_state:
    st.session_state.expected_value = None





def _render_file_uploader_and_validate():
    st.subheader(UI_TEXTS["upload_subheader"])
    uploaded_files = st.file_uploader(
        UI_TEXTS["upload_file_uploader_label"],
        type=["csv"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info(UI_TEXTS["upload_info_no_files"])
        return None

    if len(uploaded_files) != len(REQUIRED_FILES):
        st.warning(UI_TEXTS["upload_warning_all_files"])
        return None

    uploaded_file_names = [file.name for file in uploaded_files]
    if not all(name in uploaded_file_names for name in REQUIRED_FILES):
        st.warning(UI_TEXTS["upload_warning_correct_names"])
        return None

    file_map = {file.name: file for file in uploaded_files}
    eval_file = file_map.get("extrait_eval.csv")
    sirh_file = file_map.get("extrait_sirh.csv")
    sondage_file = file_map.get("extrait_sondage.csv")

    if not (eval_file and sirh_file and sondage_file):
        # This case should ideally be caught by the 'not all(name in uploaded_file_names...)' check,
        # but as a safeguard.
        st.warning(UI_TEXTS["upload_warning_correct_names"])
        return None
    
    return file_map


def _process_uploaded_files(file_map: dict, main_threshold: float) -> dict:
    """Loads raw dataframes, converts them to lists of dictionaries, and prepares the API payload."""
    eval_df = pd.read_csv(file_map.get("extrait_eval.csv"))
    sirh_df = pd.read_csv(file_map.get("extrait_sirh.csv"))
    sondage_df = pd.read_csv(file_map.get("extrait_sondage.csv"))

    # --- Preprocessing for FastAPI Pydantic Schema Compatibility ---

    # 1. Handle 'augementation_salaire_precedente' in eval_df
    if "augementation_salaire_precedente" in eval_df.columns:
        eval_df["augementation_salaire_precedente"] = (
            eval_df["augementation_salaire_precedente"]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(",", ".", regex=False)
            .str.strip()
        )
        eval_df["augementation_salaire_precedente"] = pd.to_numeric(
            eval_df["augementation_salaire_precedente"], errors="coerce"
        )
        # Convert to float, as the Pydantic schema expects float
        eval_df["augementation_salaire_precedente"] = eval_df["augementation_salaire_precedente"].astype(float)

    # 2. Handle 'code_sondage' in sondage_df
    if "code_sondage" in sondage_df.columns:
        sondage_df["code_sondage"] = sondage_df["code_sondage"].astype(str)

    # --- End Preprocessing ---

    eval_data_list = eval_df.to_dict(orient="records")
    sirh_data_list = sirh_df.to_dict(orient="records")
    sondage_data_list = sondage_df.to_dict(orient="records")

    payload = {
        "eval_data": eval_data_list,
        "sirh_data": sirh_data_list,
        "sondage_data": sondage_data_list,
        "threshold": main_threshold,
    }
    return payload


def _call_prediction_api(payload: dict):
    """Makes the API call and handles the response."""
    headers = {
        "X-API-Key": API_TOKEN,
        "Content-Type": "application/json",
    }
    try:
        with st.spinner("Making predictions..."): # Add spinner here
            response = httpx.post(f"{API_URL}/predict", json=payload, headers=headers)
            response.raise_for_status()
            response_data = response.json()
            return response_data["predictions"], response_data["processed_data"]
    except httpx.RequestError as e: # Use httpx's specific exception
        st.error(f"{UI_TEXTS['api_error']} {e}. Please ensure the API server is running and accessible at {API_URL}.")
        return None, None
    except httpx.HTTPStatusError as e: # Handle HTTP status errors specifically
        st.error(f"{UI_TEXTS['api_error']} HTTP Error {e.response.status_code}: {e.response.text}. Please check your input data or API logs.")
        return None, None
    except Exception as e:
        st.error(f"{UI_TEXTS['unexpected_error']}{e}")
        return None, None


def _display_prediction_results(api_predictions: list[dict], main_threshold: float, processed_data: list[dict]) -> None:
    """Processes API predictions, updates session state, and displays the success message."""
    report_data = pd.DataFrame(api_predictions)
    report_data["Attrition_Risk_Percentage"] = report_data["probability"]
    report_data["Risk_Attrition"] = report_data["probability"].apply(
        lambda x: get_risk_category(x, main_threshold)
    )
    st.session_state.report_data = report_data
    st.session_state.prediction_triggered = True
    st.session_state.processed_data_for_shap = pd.DataFrame(processed_data) # Store processed data
    
    # Extract SHAP values and expected value from the first prediction (assuming they are consistent across batch)
    # Or, if each prediction has its own SHAP values, store them per employee
    if api_predictions and "shap_values" in api_predictions[0] and "expected_value" in api_predictions[0]:
        st.session_state.shap_values = [p["shap_values"] for p in api_predictions]
        st.session_state.expected_value = api_predictions[0]["expected_value"]
    else:
        st.session_state.shap_values = None
        st.session_state.expected_value = None

    st.success(UI_TEXTS["prediction_success"])


def _handle_file_uploads_and_predict(main_threshold: float) -> None:
    file_map = _render_file_uploader_and_validate()
    if file_map is None:
        return

    predict_button = st.button(UI_TEXTS["predict_button_label"])

    if predict_button:
        payload = _process_uploaded_files(file_map, main_threshold)
        api_predictions, processed_data = _call_prediction_api(payload)

        if api_predictions and processed_data is not None:
            _display_prediction_results(api_predictions, main_threshold, processed_data)








def clear_prediction_results() -> None:
    """Clear all prediction results and reset session state."""
    st.session_state.prediction_triggered = False
    st.session_state.report_data = None
    st.session_state.shap_report_data = None
    st.session_state.excel_report_data = None
    st.session_state.processed_data_for_shap = None
    st.session_state.explainer = None
    st.session_state.all_features = None


# --- Load Model and Data ---
@st.cache_resource
def load_training_data_for_confusion_matrix():
    """Load the training data for confusion matrix display."""
    # Only load what's necessary for the confusion matrix
    y_train_loaded = pd.read_parquet("outputs/y_train.parquet").squeeze()
    # We need the model to calculate train_prediction_proba
    model = joblib.load("outputs/employee_attrition_pipeline.pkl")
    x_train_loaded = pd.read_parquet("outputs/X_train.parquet")
    train_prediction_proba = model.predict_proba(x_train_loaded)[:, 1]
    return y_train_loaded, train_prediction_proba

# Function to set up the app's global resources
def _setup_app() -> None:
    if "y_train_loaded" not in st.session_state:
        st.session_state.y_train_loaded, st.session_state.train_prediction_proba = load_training_data_for_confusion_matrix()


# Call setup function only if not in a test environment (or if Streamlit is running)
# This check is a common pattern to prevent global execution during imports in tests
if "streamlit" in sys.modules and "pytest" not in sys.modules:
    _setup_app()


# --- Helper Functions (from train.py) ---


















def _get_risk_category_from_log_odds(log_odds: float) -> str:
    """Categorize attrition risk based on log-odds (f(x)) using predefined thresholds."""
    low_threshold = LOG_ODDS_RISK_THRESHOLDS["Low"][1]  # Upper bound of Low
    high_threshold = LOG_ODDS_RISK_THRESHOLDS["High"][0]  # Lower bound of High

    if log_odds < low_threshold:
        return "Low"
    if log_odds >= high_threshold:
        return "High"
    return "Medium"


def get_risk_category(probability: float, threshold: float) -> str:
    """Categorize attrition risk based on probability and a dynamic threshold."""
    if probability >= threshold + 0.1:
        return "High"
    if probability <= threshold - 0.1:
        return "Low"
    return "Medium"


def display_confusion_matrix_and_metrics(y_true, y_proba, threshold, title):
    """
    Displays a confusion matrix with row-normalized percentages.
    Returns accuracy, recall, and the count of flagged employees.
    """
    y_pred = (y_proba >= threshold).astype(int)
    
    # Calculate confusion matrix components
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Calculate row-normalized percentages
    cm_normalized = np.zeros_like(cm, dtype=float)
    row_sums = cm.sum(axis=1)
    for i in range(cm.shape[0]):
        if row_sums[i] > 0:
            cm_normalized[i, :] = cm[i, :] / row_sums[i]

    # st.subheader(title)
    fig, ax = plt.subplots(figsize=(4, 3)) # Revert to original figure size
    sns.heatmap(cm_normalized, annot=True, fmt=".1%", cmap='Greens', cbar=False,
                xticklabels=['Predicted Stay', 'Predicted Leave'],
                yticklabels=['Actual Stay', 'Actual Leave'], ax=ax,
                annot_kws={"size": 10}) # Smaller annotation font size
    ax.set_ylabel('Actual Outcome')
    ax.set_xlabel('Predicted Outcome')
    st.pyplot(fig, width='stretch') # Ensure responsiveness
    plt.close(fig) # Close the plot to prevent it from displaying twice

    # Calculate Accuracy and Recall for external display
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    # Count of employees flagged as 'Leave' (True Positives + False Positives)
    flagged_count = tp + fp

    return accuracy, recall, flagged_count







def _render_shap_explanation(employee_id: int) -> None:
    """Renders the SHAP force plot for a selected employee."""
    if st.session_state.shap_values is None or st.session_state.expected_value is None:
        st.warning("SHAP values not available for explanation.")
        return

    # Find the index of the selected employee
    employee_ids = st.session_state.report_data["id_employee"].tolist()
    try:
        idx = employee_ids.index(employee_id)
    except ValueError:
        st.error(f"Employee ID {employee_id} not found in prediction results.")
        return

    # Get SHAP values and feature values for the selected employee
    # Select SHAP values for the positive class (index 1) for the current employee
    shap_values_for_instance = st.session_state.shap_values[0][idx][1]
    expected_value_for_class = st.session_state.expected_value

    feature_values = st.session_state.processed_data_for_shap.iloc[idx]
    feature_names = st.session_state.processed_data_for_shap.columns.tolist()

    # Create a SHAP Explanation object for the single instance and class
    explanation = shap.Explanation(
        values=shap_values_for_instance,
        base_values=expected_value_for_class,
        data=feature_values, # Pass Pandas Series directly
        feature_names=feature_names
    )

    st.subheader(UI_TEXTS["shap_plot_title"].format(Employee_ID=employee_id))
    st.write("This plot shows how each feature contributes to the employee's attrition risk prediction.")

    # Render the SHAP force plot
    shap.initjs()
    html_plot = shap.force_plot(
        explanation, # Pass the Explanation object directly
        matplotlib=False,
        show=False
    )
    shap_html = f"<head>{html_plot.html.split('<head>')[1].split('</head>')[0]}</head><body>{html_plot.html.split('<body>')[1].split('</body>')[0]}</body>"
    components.html(shap_html, height=300, scrolling=True)


def render_employee_overview_and_shap():
    """
    Renders a sortable employee table and includes a download button for a comprehensive Excel report.
    """
    st.header(UI_TEXTS["overview_header"])

    with st.container(border=True):
        # Ensure report_data is available
        if 'report_data' not in st.session_state or st.session_state.report_data.empty:
            st.warning(UI_TEXTS["overview_warning_no_data"])
            return

        df_display = st.session_state.report_data.copy()
        df_display = df_display.rename(columns={
            "id_employee": "Employee ID", # Explicitly rename to "Employee ID"
            "Attrition_Risk_Percentage": UI_TEXTS["employee_table_col_probability"],
            "prediction": UI_TEXTS["employee_table_col_prediction"],
            "Risk_Attrition": "Risk Level" # This is not in MICROCOPY.md, keep as is
        })
        
        # Format Risk Score as percentage
        df_display["Attrition Risk (%)"] = df_display["Attrition Risk (%)"].apply(lambda x: f"{x:.1%}")

        # Display sortable table
        st.dataframe(df_display[["Employee ID", "Attrition Risk (%)", "Model Decision", "Risk Level"]].sort_values(by="Attrition Risk (%)", ascending=False), use_container_width=True)

        # --- SHAP Explanation Section ---
        if st.session_state.prediction_triggered and st.session_state.report_data is not None:
            st.subheader("Individual Employee Explanation (SHAP)")
            employee_ids = df_display["Employee ID"].tolist() # Use df_display after rename
            selected_employee_id = st.selectbox(
                UI_TEXTS["employee_select_box_label"],
                options=employee_ids,
                key="shap_employee_selector"
            )
            if selected_employee_id:
                _render_shap_explanation(selected_employee_id)

        # --- Download Excel Report Button ---
        if st.session_state.prediction_triggered and st.session_state.report_data is not None:
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                # Tab 1: Summary
                tab1_df = st.session_state.report_data[
                    [
                        "id_employee",
                        "Risk_Attrition",
                        "Attrition_Risk_Percentage",
                        "Prediction",
                    ]
                ].copy()
                tab1_df.rename(columns={"id_employee": "Employee_ID"}, inplace=True)
                tab1_df.to_excel(writer, sheet_name="Summary", index=False)

                # Tab 2: Features (all features with coefficients) - Placeholder for now
                pd.DataFrame({"Message": [UI_TEXTS["excel_shap_message"]]}).to_excel(writer, sheet_name="Features", index=False)

                # Tab 3: Metrics (optional)
                summary_metrics_df = pd.DataFrame(
                    {
                        "Metric": [
                            "Total Employees Processed",
                            "Predicted to Leave",
                            "Predicted to Stay",
                        ],
                        "Value": [
                            len(st.session_state.report_data),
                            st.session_state.report_data["Prediction"].value_counts().get("Leave", 0),
                            st.session_state.report_data["Prediction"].value_counts().get("Stay", 0),
                        ],
                    }
                )
                summary_metrics_df.to_excel(writer, sheet_name="Metrics", index=False)

            excel_buffer.seek(0)
            st.download_button(
                label=UI_TEXTS["download_excel_report_label"],
                data=excel_buffer,
                file_name="employee_attrition_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help=UI_TEXTS["download_excel_report_help"]
            )

# --- End of STREAMLIT CODE ---

def main() -> None:
    """Run the Streamlit application."""
    _setup_app()  # Ensure global variables are initialized
    # --- Streamlit App Layout ---
    st.set_page_config(layout="wide")
    st.title(UI_TEXTS["app_title"])

    # Inject custom CSS
    st.markdown("""
    <style>
    /* Container for CM and Slider */
    .bordered-container {
        border: 2px solid black;
        border-image: linear-gradient(to right, black 30%, gold 70%) 1;
        padding: 20px;
        margin-bottom: 20px;
        background-color: #F0F0F0; /* Light grey background */
    }

    /* Slider styling */
    /* Target the slider track background - Light Green */
    /* This targets the filled portion */
    [data-testid*="stSlider"] > div > div:nth-child(2) > div:nth-child(1) {
        background-color: #C8E6C9 !important; /* Light green */
        opacity: 1 !important;
    }
    /* This targets the unfilled portion */
    [data-testid*="stSlider"] > div > div:nth-child(2) > div:nth-child(2) {
        background-color: #E0E0E0 !important; /* Light grey for unfilled part */
        opacity: 1 !important;
    }
    /* Make the entire track thicker */
    [data-testid*="stSlider"] > div > div:nth-child(2) {
        height: 12px !important; /* Increased track thickness */
    }

    /* Remove slider prefix/suffix text */
    .stSlider .st-bs { /* This targets the prefix/suffix text */
        visibility: hidden;
        height: 0px;
    }

    /* Slider value display (the current threshold number) - Twice as big */
    .stSlider .st-bh { /* This targets the value label above the slider */
        font-size: 2em !important; /* Twice the default size */
        visibility: visible !important; /* Make it visible */
        height: auto !important; /* Adjust height */
    }

    /* Slider thumb (bullet) - Even bigger and with shadow */
    .stSlider .st-be { /* This targets the slider thumb */
        width: 2.5em !important; /* Even bigger */
        height: 2.5em !important; /* Even bigger */
        margin-top: -1em !important; /* Adjust position to center it on the track */
        box-shadow: 0px 0px 5px rgba(0, 0, 0, 0.3); /* Add a subtle shadow */
    }

    /* Button styling */
    .stButton > button {
        border: 1px solid black;
    }

    /* Adjust slider position - this might need refinement */
    .stSlider {
        margin-top: 20px; /* Slightly lower */
        margin-left: auto;
        margin-right: auto;
        width: 80%; /* Slightly toward the center */
    }

    /* Ensure titles are aligned horizontally */
    /* This targets the st.markdown for titles, assuming they are rendered as p tags or similar */
    .title-align {
        display: flex;
        align-items: center;
        height: 100%;
        font-size: 1.5em; /* Adjust as needed for subheader size */
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

    # --- Main Content Area: Confusion Matrix and Threshold Slider ---
    with st.container():
        st.markdown('<div class="bordered-container">', unsafe_allow_html=True)
        col_left_margin, col_cm, col_slider, col_right_margin = st.columns([1, 5, 5, 1])

        with col_cm:
            st.markdown(f'<p class="title-align">{UI_TEXTS["prediction_accuracy_overview_title"]}</p>', unsafe_allow_html=True)
            # Call render_threshold_block content here, but only the CM part
            y_true = st.session_state.y_train_loaded
            y_proba = st.session_state.train_prediction_proba
            accuracy, recall, flagged_count = display_confusion_matrix_and_metrics(
                y_true, y_proba, st.session_state.get('current_threshold', 0.5), title=""
            )
            st.markdown(f"{UI_TEXTS['overall_correct_predictions']} {accuracy:.1%}")
            st.markdown(f"{UI_TEXTS['correctly_identified_leave_cases']} {recall:.1%}")
            st.markdown(UI_TEXTS["estimated_review_workload"].format(flagged_count))


        with col_slider:
            st.markdown(f'<p class="title-align">{UI_TEXTS["threshold_header"]}</p>', unsafe_allow_html=True)
            # Place the slider slightly toward the center and slightly lower than the top edge of its column.
            # This is handled by the CSS for .stSlider
            main_threshold = st.slider(
                label=UI_TEXTS["threshold_slider_label"], # Use microcopy for label
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.01,
                help=UI_TEXTS["threshold_slider_help"], # Use microcopy for help
                label_visibility="hidden" # Hide the label visually
            )
            st.session_state.current_threshold = main_threshold # Store threshold in session state
        st.markdown('</div>', unsafe_allow_html=True) # Close bordered-container div


    st.markdown(UI_TEXTS["understanding_predictions_intro"])
    

    st.subheader(UI_TEXTS["threshold_examples_subheader"]) # Add subheader for examples
    

    for example in UI_TEXTS["threshold_examples_list"]:
        st.markdown(example) # Display each example on a new line

    st.markdown("---")

    _handle_file_uploads_and_predict(st.session_state.get('current_threshold', 0.5)) # Use current_threshold from slider

    # --- Display Results (if triggered) ---
    if st.session_state.prediction_triggered:
        st.markdown("---")
        render_employee_overview_and_shap()

        st.success(UI_TEXTS["reports_generated_success"])


if __name__ == "__main__":
    main()