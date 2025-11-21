import json
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import numpy as np

pd.set_option("future.no_silent_downcasting", True)
import io
import os
import httpx
import base64
import zipfile
from typing import Any, Dict
import time
from dotenv import load_dotenv

# Load environment variables - .env.local overrides .env for local development
load_dotenv()  # Load .env
load_dotenv(".env.local", override=True)  # Load .env.local and override

# --- Configuration ---
REQUIRED_FILES = ["extrait_eval.csv", "extrait_sirh.csv", "extrait_sondage.csv"]

# Get API URL from environment variable, default to localhost for development
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8001")

# Get API key from environment
API_KEY = os.environ.get("API_KEY", "demo_api_key_change_in_production")

# Optimal threshold from model training
OPTIMAL_THRESHOLD = 0.2876


def _check_api_health(base_url: str) -> tuple[bool, str]:
    """Quick health probe to show connection status in the UI."""
    try:
        resp = httpx.get(f"{base_url}/health", timeout=3.0)
        if resp.status_code == 200:
            return True, "API is healthy"
        return False, f"API responded with status {resp.status_code}"
    except httpx.RequestError as e:
        return False, f"Network error: {e}"
    except Exception as e:
        return False, f"Unexpected error: {e}"


# --- Session State Initialization (minimal) ---
if "prediction_triggered" not in st.session_state:
    st.session_state.prediction_triggered = False
if "report_data" not in st.session_state:
    st.session_state.report_data = None
if "last_payload" not in st.session_state:
    st.session_state.last_payload = None
if "excel_report_bytes" not in st.session_state:
    st.session_state.excel_report_bytes = None
if "shap_zip_bytes" not in st.session_state:
    st.session_state.shap_zip_bytes = None
if "shap_html_bytes" not in st.session_state:
    st.session_state.shap_html_bytes = None
if "job_id" not in st.session_state:
    st.session_state.job_id = None
if "job_status" not in st.session_state:
    st.session_state.job_status = None
if "job_error" not in st.session_state:
    st.session_state.job_error = None
if "job_excel_report_bytes" not in st.session_state:
    st.session_state.job_excel_report_bytes = None
if "job_shap_zip_bytes" not in st.session_state:
    st.session_state.job_shap_zip_bytes = None
if "user_id" not in st.session_state:
    st.session_state.user_id = "demo1"  # Default user ID
if "threshold" not in st.session_state:
    st.session_state.threshold = OPTIMAL_THRESHOLD
if "predictions_raw" not in st.session_state:
    st.session_state.predictions_raw = None


def get_project_root():
    """Returns the absolute path to the project root."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


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
        st.error(
            f"Required data file not found: {e}. Please ensure 'data' directory "
            "contains 'extrait_eval.csv', 'extrait_sirh.csv', and 'extrait_sondage.csv'."
        )
        return None, None, None


def _call_prediction_api(
    eval_data: Any,
    sirh_data: Any,
    sondage_data: Any,
    user_id: str = "demo1",
) -> Dict[str, Any]:
    """Calls the FastAPI /predict endpoint with raw employee data."""
    try:
        payload = {
            "eval_data": eval_data,
            "sirh_data": sirh_data,
            "sondage_data": sondage_data,
        }
        headers = {
            "X-User-ID": user_id,
            "X-API-Key": API_KEY,
        }
        # Set a reasonable timeout (increased for DB operations and SHAP calculations)
        response = httpx.post(
            f"{API_BASE_URL}/predict", json=payload, headers=headers, timeout=180.0
        )
        response.raise_for_status()
        return response.json()
    except httpx.TimeoutException as e:
        print(f"API request timed out: {e}")
        st.error(
            f"The prediction API took too long to respond (timeout). Please try again or check the API server status."
        )
        return {"predictions": []}
    except httpx.RequestError as e:
        print(f"Network error while connecting to API: {e}")
        st.error(
            f"Network error while connecting to API: {e}. Please ensure the API server is running and accessible."
        )
        return {"predictions": []}
    except httpx.HTTPStatusError as e:
        print(f"API returned an error: {e.response.status_code} - {e.response.text}")
        st.error(
            f"API returned an error: {e.response.status_code} - {e.response.text}. Please check the API logs for details."
        )
        return {"predictions": []}
    except Exception as e:
        print(f"An unexpected error occurred during API call: {e}")
        st.error(f"An unexpected error occurred during API call: {e}")
        return {"predictions": []}


def _call_predict_excel_api(
    payload: Dict[str, Any], user_id: str = "demo1"
) -> bytes | None:
    try:
        headers = {
            "X-User-ID": user_id,
            "X-API-Key": API_KEY,
        }
        resp = httpx.post(
            f"{API_BASE_URL}/predict_excel",
            json=payload,
            headers=headers,
            timeout=180.0,
        )
        resp.raise_for_status()
        excel_b64 = resp.json().get("excel_base64")
        if not excel_b64:
            st.error("API did not return an Excel report.")
            return None
        return base64.b64decode(excel_b64)
    except Exception as e:
        st.error(f"Failed to generate Excel report: {e}")
        return None


def _call_predict_shap_images_api(
    payload: Dict[str, Any], user_id: str = "demo1"
) -> bytes | None:
    try:
        headers = {
            "X-User-ID": user_id,
            "X-API-Key": API_KEY,
        }
        resp = httpx.post(
            f"{API_BASE_URL}/predict_shap_images",
            json=payload,
            headers=headers,
            timeout=180.0,
        )
        resp.raise_for_status()
        items = resp.json().get("shap_images", [])
        if not items:
            st.warning("No SHAP images returned by API.")
            return None
        # Build a zip in-memory
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for item in items:
                emp_id = item.get("employee_id")
                img_b64 = item.get("img_base64")
                if img_b64 and emp_id is not None:
                    zf.writestr(f"employee_{emp_id}.png", base64.b64decode(img_b64))
        buf.seek(0)
        return buf.read()
    except Exception as e:
        st.error(f"Failed to generate SHAP images: {e}")
        return None


def _call_predict_shap_html_api(
    payload: Dict[str, Any], user_id: str = "demo1"
) -> bytes | None:
    """Call the API to generate SHAP HTML report and return HTML bytes."""
    try:
        headers = {
            "X-User-ID": user_id,
            "X-API-Key": API_KEY,
        }
        resp = httpx.post(
            f"{API_BASE_URL}/predict_shap_html",
            json=payload,
            headers=headers,
            timeout=180.0,
        )
        resp.raise_for_status()
        # The response is HTML text
        return resp.content
    except Exception as e:
        st.error(f"Failed to generate SHAP HTML report: {e}")
        return None


def _enqueue_report_job(payload: Dict[str, Any], user_id: str = "demo1") -> str | None:
    try:
        headers = {
            "X-User-ID": user_id,
            "X-API-Key": API_KEY,
        }
        resp = httpx.post(
            f"{API_BASE_URL}/jobs/report", json=payload, headers=headers, timeout=30.0
        )
        resp.raise_for_status()
        job_id = resp.json().get("job_id")
        if not job_id:
            st.error("API did not return a job_id.")
            return None
        return job_id
    except Exception as e:
        st.error(f"Failed to enqueue report job: {e}")
        return None


def _get_job_status(job_id: str) -> Dict[str, Any] | None:
    try:
        resp = httpx.get(f"{API_BASE_URL}/jobs/{job_id}", timeout=15.0)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"Failed to check job status: {e}")
        return None


def _fetch_job_result(job_id: str) -> tuple[bytes | None, bytes | None]:
    try:
        resp = httpx.get(f"{API_BASE_URL}/jobs/{job_id}/result", timeout=180.0)
        resp.raise_for_status()
        data = resp.json()
        excel_b64 = data.get("excel_base64")
        shap_images = data.get("shap_images", [])

        excel_bytes = base64.b64decode(excel_b64) if excel_b64 else None

        # Build a zip with SHAP images
        shap_zip_bytes = None
        if shap_images:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                for item in shap_images:
                    emp_id = item.get("employee_id")
                    img_b64 = item.get("img_base64")
                    if img_b64 and emp_id is not None:
                        zf.writestr(f"employee_{emp_id}.png", base64.b64decode(img_b64))
            buf.seek(0)
            shap_zip_bytes = buf.read()

        return excel_bytes, shap_zip_bytes
    except Exception as e:
        st.error(f"Failed to fetch job result: {e}")
        return None, None


def apply_threshold(predictions_raw: list, threshold: float) -> pd.DataFrame:
    """Apply threshold to raw predictions and return formatted DataFrame."""
    report_data = pd.DataFrame(
        [
            {
                "id_employee": p.get("id_employee"),
                "probability": p.get("probability"),
                "prediction": 1 if p.get("probability") >= threshold else 0,
                "prediction_label": (
                    "Leave" if p.get("probability") >= threshold else "Stay"
                ),
            }
            for p in predictions_raw
        ]
    )
    if "probability" in report_data.columns:
        report_data["probability"] = (
            report_data["probability"].astype(float) * 100
        ).round(1)
    return report_data


def load_test_set_confusion_matrix(threshold: float) -> tuple[np.ndarray, int, int]:
    """Load test set and compute confusion matrix using saved test data."""
    try:
        project_root = get_project_root()
        x_test_path = os.path.join(project_root, "outputs", "X_test.parquet")
        y_test_path = os.path.join(project_root, "outputs", "y_test.parquet")
        model_path = os.path.join(
            project_root, "outputs", "employee_attrition_pipeline.pkl"
        )

        # Check if files exist
        if not all(os.path.exists(p) for p in [x_test_path, y_test_path, model_path]):
            return None, 0, 0

        # Load test data
        X_test = pd.read_parquet(x_test_path)
        y_test = pd.read_parquet(y_test_path)

        # Load model and predict
        import joblib

        model = joblib.load(model_path)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)

        # Calculate confusion matrix
        y_actual = y_test.values.flatten()
        tp = ((y_actual == 1) & (y_pred == 1)).sum()
        tn = ((y_actual == 0) & (y_pred == 0)).sum()
        fp = ((y_actual == 0) & (y_pred == 1)).sum()
        fn = ((y_actual == 1) & (y_pred == 0)).sum()

        cm = np.array([[tn, fp], [fn, tp]])
        return cm, len(y_test), (y_actual == 1).sum()

    except Exception as e:
        st.error(f"Error loading test set: {e}")
        return None, 0, 0


def create_confusion_matrix(threshold: float) -> go.Figure:
    """Create a Plotly confusion matrix visualization with row percentages only."""
    cm, total_samples, total_attrition = load_test_set_confusion_matrix(threshold)

    if cm is None:
        # Return error message if test set not available
        st.error("Test set not found. Cannot display confusion matrix.")
        return None

    # Calculate metrics
    accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum() if cm.sum() > 0 else 0
    precision = cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else 0
    recall = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    # Calculate ROW percentages (each cell as percentage of its row total)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_row_percentages = (cm / row_sums * 100).round(1)

    # Create text with only percentages (no absolute numbers)
    text_display = [[f"{cm_row_percentages[i, j]:.1f}%" for j in range(2)] for i in range(2)]

    # Create heatmap with row percentages for coloring
    fig = go.Figure(
        data=go.Heatmap(
            z=cm_row_percentages,
            x=["<b>Predicted: Stay</b>", "<b>Predicted: Leave</b>"],
            y=["<b>Actual: Stay</b>", "<b>Actual: Leave</b>"],
            colorscale="Blues",
            showscale=True,
            text=text_display,
            texttemplate="%{text}",
            textfont={"size": 20, "color": "#000000", "family": "Arial Black"},
            hovertemplate="<b>%{x}</b><br><b>%{y}</b><br>Percentage: %{z:.1f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"<b>Confusion Matrix (Row Percentages)</b><br><sub>Accuracy: {accuracy:.2%} | Precision: {precision:.2%} | Recall: {recall:.2%} | F1: {f1:.2%}</sub>",
        xaxis_title="<b>Predicted Label</b>",
        yaxis_title="<b>Actual Label</b>",
        height=450,
        font=dict(size=12, color="#000000"),
        xaxis=dict(tickfont=dict(size=12, color="#000000", family="Arial Black")),
        yaxis=dict(tickfont=dict(size=12, color="#000000", family="Arial Black")),
    )

    return fig


def _load_template_files() -> dict[str, str]:
    """Load first 10 lines of each template CSV file."""
    try:
        project_root = get_project_root()
        templates = {}

        for filename in REQUIRED_FILES:
            filepath = os.path.join(project_root, "data", filename)
            with open(filepath, "r", encoding="utf-8") as f:
                lines = [next(f) for _ in range(11)]  # Header + 10 data lines
                templates[filename] = "".join(lines)

        return templates
    except Exception as e:
        st.error(f"Error loading template files: {e}")
        return {}


def _handle_file_uploads_and_predict() -> None:
    st.subheader("Upload Data")

    # Show downloadable template files
    st.markdown("### Template Files")
    st.markdown("Download these template files to see the required format (first 10 lines):")

    templates = _load_template_files()
    if templates:
        cols = st.columns(3)
        for idx, (filename, content) in enumerate(templates.items()):
            with cols[idx]:
                st.download_button(
                    label=f"📥 {filename}",
                    data=content,
                    file_name=filename,
                    mime="text/csv",
                    use_container_width=True,
                )

    st.markdown("---")

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload three CSV files with the exact names: `extrait_eval.csv`, "
        "`extrait_sirh.csv`, `extrait_sondage.csv`",
        type=["csv"],
        accept_multiple_files=True,
        key="file_uploader",
    )

    eval_file, sirh_file, sondage_file = None, None, None

    if uploaded_files:
        if len(uploaded_files) != len(REQUIRED_FILES):
            st.warning(f"Please upload exactly {len(REQUIRED_FILES)} CSV files.")
            return

        uploaded_file_names = [file.name for file in uploaded_files]
        if not all(name in uploaded_file_names for name in REQUIRED_FILES):
            st.warning(
                "Uploaded files must have these exact names: "
                "`extrait_eval.csv`, `extrait_sirh.csv`, and `extrait_sondage.csv`."
            )
            return

        file_map = {file.name: file for file in uploaded_files}
        eval_file = file_map.get("extrait_eval.csv")
        sirh_file = file_map.get("extrait_sirh.csv")
        sondage_file = file_map.get("extrait_sondage.csv")

    if eval_file and sirh_file and sondage_file:
        predict_button = st.button("Predict Attrition")

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
                        eval_data_for_api,
                        sirh_data_for_api,
                        sondage_data_for_api,
                        str(st.session_state.user_id),
                    )

                    # Optionally save API response for debugging
                    try:
                        temp_api_response_path = os.path.join(
                            get_project_root(), "temp_api_response.json"
                        )
                        with open(temp_api_response_path, "w") as f:
                            json.dump(api_response, f, indent=4)
                        st.caption(f"Saved API response to: {temp_api_response_path}")
                    except Exception:
                        pass

                    if api_response and api_response["predictions"]:
                        predictions_data = api_response["predictions"]
                        st.session_state.predictions_raw = predictions_data
                        st.session_state.last_payload = {
                            "eval_data": eval_data_for_api,
                            "sirh_data": sirh_data_for_api,
                            "sondage_data": sondage_data_for_api,
                        }

                        # Apply threshold and create report
                        report_data = apply_threshold(
                            predictions_data, st.session_state.threshold
                        )
                        st.session_state.report_data = report_data
                        st.session_state.prediction_triggered = True

                        # Auto-generate Excel and HTML reports
                        with st.spinner("Generating Excel report and SHAP analysis..."):
                            # Generate Excel report
                            excel_bytes = _call_predict_excel_api(
                                st.session_state.last_payload, str(st.session_state.user_id)
                            )
                            if excel_bytes:
                                st.session_state.excel_report_bytes = excel_bytes

                            # Generate HTML SHAP report
                            html_bytes = _call_predict_shap_html_api(
                                st.session_state.last_payload, str(st.session_state.user_id)
                            )
                            if html_bytes:
                                st.session_state.shap_html_bytes = html_bytes

                        st.success(
                            f"Predictions received successfully! ({len(predictions_data)} employees)"
                        )
                    else:
                        st.error("No predictions received from the API.")

                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
        else:
            st.info("Files uploaded. Click 'Predict Attrition' to call the API.")
    else:
        st.info("Please upload all three required CSV files to enable predictions.")


def clear_prediction_results() -> None:
    """Clear prediction results and reset session state."""
    st.session_state.prediction_triggered = False
    st.session_state.report_data = None
    st.session_state.last_payload = None
    st.session_state.excel_report_bytes = None
    st.session_state.shap_zip_bytes = None
    st.session_state.shap_html_bytes = None


# No local SHAP or Excel processing in the UI — API handles processing.
def main() -> None:
    """Run the Streamlit application."""
    # --- Streamlit App Layout ---
    st.set_page_config(
        page_title="HR Attrition Risk", layout="wide", initial_sidebar_state="collapsed"
    )
    st.title("Employee Attrition Risk Prediction")

    # Sidebar: API status
    with st.sidebar:
        st.header("API Status")
        st.caption(f"Endpoint: {API_BASE_URL}")
        ok, msg = _check_api_health(API_BASE_URL)
        if ok:
            st.success(msg)
        else:
            st.error(msg)
        if st.button("Refresh API Status", type="secondary"):
            st.rerun()

    # Two-column layout: Left = Analysis, Right = Upload & Results
    st.markdown("---")
    col_left, col_right = st.columns([1, 1])

    with col_left:
        # Threshold slider and guidance
        st.subheader("Risk Classification Threshold")
        new_threshold = st.slider(
            "Set Risk Threshold (τ)",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.threshold,
            step=0.01,
            help=f"Optimal threshold from evaluation: {OPTIMAL_THRESHOLD:.4f}",
        )
        if new_threshold != st.session_state.threshold:
            st.session_state.threshold = new_threshold
            if st.session_state.predictions_raw:
                st.session_state.report_data = apply_threshold(
                    st.session_state.predictions_raw, new_threshold
                )
            st.rerun()

        st.metric(
            "Current Threshold",
            f"{st.session_state.threshold:.4f}",
            delta=f"{(st.session_state.threshold - OPTIMAL_THRESHOLD):.4f} from optimal",
        )

        # Dynamic threshold guidance
        st.markdown("---")
        st.markdown("#### 💡 How to Choose Your Threshold")
        if st.session_state.threshold < 0.3:
            st.info("""
**Low threshold (< 0.3)**: Maximizes detection of potential leavers.

✅ **Pros**: Catches almost all at-risk employees

⚠️ **Cons**: May flag many false positives (employees who will actually stay)

🎯 **Best for**: Comprehensive retention programs with resources to engage more employees
            """)
        elif st.session_state.threshold < 0.5:
            st.info("""
**Moderate threshold (0.3 - 0.5)**: Balanced approach.

✅ **Pros**: Good balance between catching at-risk employees and precision

⚠️ **Cons**: May miss some lower-risk leavers

🎯 **Best for**: Standard retention programs with moderate resources
            """)
        else:
            st.info("""
**High threshold (≥ 0.5)**: Focuses on highest-risk employees.

✅ **Pros**: High confidence in predicted leavers (fewer false alarms)

⚠️ **Cons**: May miss employees with moderate attrition risk

🎯 **Best for**: Targeted retention programs focusing on critical talent
            """)

    with col_right:
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        fig = create_confusion_matrix(st.session_state.threshold)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        # Understanding the Confusion Matrix
        st.markdown("---")
        st.markdown("### 📊 Understanding the Confusion Matrix")
        st.markdown("""
        <div style='font-size: 1.05em; line-height: 1.7;'>

        The confusion matrix shows how well the model's predictions match actual outcomes:

        <ul style='margin-top: 1em;'>
        <li><strong style='color: #0e8f4f;'>True Positives (Predicted Leaves / Actual Leaves)</strong>: Employees correctly predicted to leave</li>
        <li><strong style='color: #0e8f4f;'>True Negatives (Predicted Stays / Actual Stays)</strong>: Employees correctly predicted to stay</li>
        <li><strong style='color: #d63f3f;'>False Positives (Predicted Leaves / Actual Stays)</strong>: Employees predicted to leave but actually stayed</li>
        <li><strong style='color: #d63f3f;'>False Negatives (Predicted Stays / Actual Leaves)</strong>: Employees predicted to stay but actually left</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    # Upload and Prediction Section (Full Width)
    st.markdown("---")
    st.subheader("Upload Data")

    # Show downloadable template files
    st.markdown("### Template Files")
    st.markdown("Download these template files to see the required format (first 10 lines):")

    templates = _load_template_files()
    if templates:
        cols = st.columns(3)
        for idx, (filename, content) in enumerate(templates.items()):
            with cols[idx]:
                st.download_button(
                    label=f"📥 {filename}",
                    data=content,
                    file_name=filename,
                    mime="text/csv",
                    use_container_width=True,
                )

    st.markdown("---")

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload three CSV files with the exact names: `extrait_eval.csv`, "
        "`extrait_sirh.csv`, `extrait_sondage.csv`",
        type=["csv"],
        accept_multiple_files=True,
        key="file_uploader",
    )

    eval_file, sirh_file, sondage_file = None, None, None

    if uploaded_files:
        if len(uploaded_files) != len(REQUIRED_FILES):
            st.warning(f"Please upload exactly {len(REQUIRED_FILES)} CSV files.")
        else:
            uploaded_file_names = [file.name for file in uploaded_files]
            if not all(name in uploaded_file_names for name in REQUIRED_FILES):
                st.warning(
                    "Uploaded files must have these exact names: "
                    "`extrait_eval.csv`, `extrait_sirh.csv`, and `extrait_sondage.csv`."
                )
            else:
                file_map = {file.name: file for file in uploaded_files}
                eval_file = file_map.get("extrait_eval.csv")
                sirh_file = file_map.get("extrait_sirh.csv")
                sondage_file = file_map.get("extrait_sondage.csv")

    if eval_file and sirh_file and sondage_file:
        predict_button = st.button("Predict Attrition")

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
                        eval_data_for_api,
                        sirh_data_for_api,
                        sondage_data_for_api,
                        str(st.session_state.user_id),
                    )

                    # Optionally save API response for debugging
                    try:
                        temp_api_response_path = os.path.join(
                            get_project_root(), "temp_api_response.json"
                        )
                        with open(temp_api_response_path, "w") as f:
                            json.dump(api_response, f, indent=4)
                        st.caption(f"Saved API response to: {temp_api_response_path}")
                    except Exception:
                        pass

                    if api_response and api_response["predictions"]:
                        predictions_data = api_response["predictions"]
                        st.session_state.predictions_raw = predictions_data
                        st.session_state.last_payload = {
                            "eval_data": eval_data_for_api,
                            "sirh_data": sirh_data_for_api,
                            "sondage_data": sondage_data_for_api,
                        }

                        # Apply threshold and create report
                        report_data = apply_threshold(
                            predictions_data, st.session_state.threshold
                        )
                        st.session_state.report_data = report_data
                        st.session_state.prediction_triggered = True

                        # Auto-generate Excel and HTML reports
                        with st.spinner("Generating Excel report and SHAP analysis..."):
                            # Generate Excel report
                            excel_bytes = _call_predict_excel_api(
                                st.session_state.last_payload, str(st.session_state.user_id)
                            )
                            if excel_bytes:
                                st.session_state.excel_report_bytes = excel_bytes

                            # Generate HTML SHAP report
                            html_bytes = _call_predict_shap_html_api(
                                st.session_state.last_payload, str(st.session_state.user_id)
                            )
                            if html_bytes:
                                st.session_state.shap_html_bytes = html_bytes

                        st.success(
                            f"Predictions received successfully! ({len(predictions_data)} employees)"
                        )
                    else:
                        st.error("No predictions received from the API.")

                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
        else:
            st.info("Files uploaded. Click 'Predict Attrition' to call the API.")
    else:
        st.info("Please upload all three required CSV files to enable predictions.")

    # Results Section (shown when predictions are available)
    if st.session_state.prediction_triggered:
        st.markdown("---")
        report_data = st.session_state.report_data

        # Employees at Risk metric
        total = len(report_data)
        at_risk = (report_data["prediction"] == 1).sum()
        st.metric(
            "Employees at Risk",
            f"{at_risk} / {total}",
            delta=f"{(at_risk/total*100):.1f}%",
        )

        # Prediction Table
        st.markdown("---")
        st.subheader("Prediction Results Table (Chosen Threshold)")
        st.dataframe(
            report_data.rename(
                columns={
                    "id_employee": "Employee ID",
                    "probability": "Probability (%)",
                    "prediction_label": "Prediction",
                }
            )[["Employee ID", "Prediction", "Probability (%)"]],
            use_container_width=True,
        )

        # Downloads section
        st.markdown("---")
        st.subheader("Downloads (With Fixed Threshold = 0.5)")
        col1, col2 = st.columns(2)

        with col1:
            if st.session_state.excel_report_bytes:
                st.download_button(
                    label="📊 Download Excel Report",
                    data=st.session_state.excel_report_bytes,
                    file_name="employee_attrition_report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
            else:
                st.info("Excel report is being generated...")

        with col2:
            if st.session_state.shap_html_bytes:
                # Create a base64 data URL for opening HTML in new tab
                html_b64 = base64.b64encode(st.session_state.shap_html_bytes).decode()
                html_data_url = f"data:text/html;base64,{html_b64}"

                st.markdown(
                    f'<a href="{html_data_url}" target="_blank" style="'
                    'display: inline-block; padding: 0.5rem 1rem; '
                    'background-color: #ff4b4b; color: white; '
                    'text-decoration: none; border-radius: 0.25rem; '
                    'font-weight: 500; text-align: center; width: 100%;'
                    '">📈 Employee Risk Analysis in Detail</a>',
                    unsafe_allow_html=True
                )
                st.info("💡 Tip: Copy the URL from the opened tab and paste it in a new browser tab to view the full report")
            else:
                st.info("SHAP analysis is being generated...")


if __name__ == "__main__":
    main()
