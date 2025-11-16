import json
import pandas as pd
import streamlit as st

pd.set_option("future.no_silent_downcasting", True)
import io
import os
import httpx
import base64
import zipfile
from typing import Any, Dict
import time
# --- Configuration ---
REQUIRED_FILES = ["extrait_eval.csv", "extrait_sirh.csv", "extrait_sondage.csv"]

# Get API URL from environment variable, default to localhost for development
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8001")


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


def _call_prediction_api(
    eval_data: Any,
    sirh_data: Any,
    sondage_data: Any,
) -> Dict[str, Any]:
    """Calls the FastAPI /predict endpoint with raw employee data."""
    try:
        payload = {
            "eval_data": eval_data,
            "sirh_data": sirh_data,
            "sondage_data": sondage_data,
        }
        # Set a reasonable timeout (increased for DB operations and SHAP calculations)
        response = httpx.post(f"{API_BASE_URL}/predict", json=payload, timeout=180.0)
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


def _call_predict_excel_api(payload: Dict[str, Any]) -> bytes | None:
    try:
        resp = httpx.post(f"{API_BASE_URL}/predict_excel", json=payload, timeout=180.0)
        resp.raise_for_status()
        excel_b64 = resp.json().get("excel_base64")
        if not excel_b64:
            st.error("API did not return an Excel report.")
            return None
        return base64.b64decode(excel_b64)
    except Exception as e:
        st.error(f"Failed to generate Excel report: {e}")
        return None


def _call_predict_shap_images_api(payload: Dict[str, Any]) -> bytes | None:
    try:
        resp = httpx.post(f"{API_BASE_URL}/predict_shap_images", json=payload, timeout=180.0)
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


def _enqueue_report_job(payload: Dict[str, Any]) -> str | None:
    try:
        resp = httpx.post(f"{API_BASE_URL}/jobs/report", json=payload, timeout=30.0)
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


def _handle_file_uploads_and_predict() -> None:
    st.subheader("Upload Data (UI is thin; API does processing)")
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
                        eval_data_for_api, sirh_data_for_api, sondage_data_for_api
                    )
                    
                    # Optionally save API response for debugging
                    try:
                        temp_api_response_path = os.path.join(get_project_root(), "temp_api_response.json")
                        with open(temp_api_response_path, "w") as f:
                            json.dump(api_response, f, indent=4)
                        st.caption(f"Saved API response to: {temp_api_response_path}")
                    except Exception:
                        pass

                    if api_response and api_response["predictions"]:
                        predictions_data = api_response["predictions"]
                        st.session_state.last_payload = {
                            "eval_data": eval_data_for_api,
                            "sirh_data": sirh_data_for_api,
                            "sondage_data": sondage_data_for_api,
                        }
                        
                        # Minimal table for display only (no local processing)
                        report_data = pd.DataFrame([
                            {
                                "id_employee": p.get("id_employee"),
                                "prediction": p.get("prediction"),
                                "probability": p.get("probability"),
                                "risk_category": p.get("risk_category"),
                            }
                            for p in predictions_data
                        ])
                        # Format probability as percentage for display
                        if "probability" in report_data.columns:
                            report_data["probability"] = (report_data["probability"].astype(float) * 100).round(1)
                        st.session_state.report_data = report_data
                        st.session_state.prediction_triggered = True
                        st.success("Predictions received successfully!")
                    else:
                        st.error("No predictions received from the API.")

                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
        elif files_source == "uploaded":
            st.info("Files uploaded. Click 'Predict Attrition' to call the API.")
        elif files_source == "local":
            st.info("Using local data/. Click 'Predict Attrition' to call the API.")
    else:
        st.error("Could not load required CSV files.")


def clear_prediction_results() -> None:
    """Clear prediction results and reset session state."""
    st.session_state.prediction_triggered = False
    st.session_state.report_data = None
    st.session_state.last_payload = None
    st.session_state.excel_report_bytes = None
    st.session_state.shap_zip_bytes = None


# No local SHAP or Excel processing in the UI — API handles processing.
def main() -> None:
    """Run the Streamlit application."""
    # --- Streamlit App Layout ---
    st.set_page_config(layout="wide")
    st.title("Employee Attrition Risk")

    with st.container():
        st.caption(f"API endpoint: {API_BASE_URL}")
        ok, msg = _check_api_health(API_BASE_URL)
        if ok:
            st.success(msg, icon="✅")
        else:
            st.error(f"{msg}. Make sure the API is running (port 8001).", icon="⚠️")
        if st.button("Re-check API status", type="secondary"):
            ok, msg = _check_api_health(API_BASE_URL)
            if ok:
                st.success(msg, icon="✅")
            else:
                st.error(f"{msg}. Make sure the API is running (port 8001).", icon="⚠️")

    _handle_file_uploads_and_predict()

    # --- Display Results (thin UI) ---
    if st.session_state.prediction_triggered:
        st.markdown("---")
        st.subheader("Prediction Results")

        report_data = st.session_state.report_data
        st.dataframe(
            report_data.rename(columns={
                "id_employee": "Employee ID",
                "prediction": "Prediction",
                "probability": "Probability (%)",
                "risk_category": "Risk Category",
            }),
            use_container_width=True,
        )
        st.success("Predictions retrieved from API.")

        # Downloads section
        st.markdown("---")
        st.subheader("Downloads")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Generate Excel Report"):
                if st.session_state.last_payload:
                    with st.spinner("Generating Excel report via API..."):
                        excel_bytes = _call_predict_excel_api(st.session_state.last_payload)
                        st.session_state.excel_report_bytes = excel_bytes
                else:
                    st.warning("No input payload available. Please run a prediction first.")
            if st.session_state.excel_report_bytes:
                st.download_button(
                    label="Download Excel Report",
                    data=st.session_state.excel_report_bytes,
                    file_name="employee_attrition_report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
        with col2:
            if st.button("Generate SHAP Images (ZIP)"):
                if st.session_state.last_payload:
                    with st.spinner("Generating SHAP images via API..."):
                        zip_bytes = _call_predict_shap_images_api(st.session_state.last_payload)
                        st.session_state.shap_zip_bytes = zip_bytes
                else:
                    st.warning("No input payload available. Please run a prediction first.")
            if st.session_state.shap_zip_bytes:
                st.download_button(
                    label="Download SHAP Images (ZIP)",
                    data=st.session_state.shap_zip_bytes,
                    file_name="shap_images.zip",
                    mime="application/zip",
                )

        st.markdown("---")
        st.subheader("Async Job (Excel + SHAP)")
        if st.button("Generate Full Report (Async)"):
            if not st.session_state.last_payload:
                st.warning("No input payload available. Please run a prediction first.")
            else:
                with st.spinner("Enqueuing job..."):
                    jid = _enqueue_report_job(st.session_state.last_payload)
                    if jid:
                        st.session_state.job_id = jid
                        st.session_state.job_status = "queued"
                        st.session_state.job_error = None

                        # Poll status with a simple loop (max ~2 minutes)
                        max_checks = 60
                        status_placeholder = st.empty()
                        progress = st.progress(0)
                        for i in range(max_checks):
                            status = _get_job_status(jid)
                            if status:
                                st.session_state.job_status = status.get("status")
                                st.session_state.job_error = status.get("error")
                                status_placeholder.info(f"Job {jid} status: {st.session_state.job_status}")
                                if st.session_state.job_status == "completed":
                                    with st.spinner("Fetching job result..."):
                                        excel_b, shap_zip_b = _fetch_job_result(jid)
                                        st.session_state.job_excel_report_bytes = excel_b
                                        st.session_state.job_shap_zip_bytes = shap_zip_b
                                    break
                                if st.session_state.job_status == "failed":
                                    break
                            progress.progress(min(int((i + 1) / max_checks * 100), 100))
                            time.sleep(2)

        # If a job has completed, expose downloads
        if st.session_state.job_status == "completed":
            st.success("Async report ready. Download below.")
            d1, d2 = st.columns(2)
            with d1:
                if st.session_state.job_excel_report_bytes:
                    st.download_button(
                        label="Download Async Excel Report",
                        data=st.session_state.job_excel_report_bytes,
                        file_name="employee_attrition_report_async.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
            with d2:
                if st.session_state.job_shap_zip_bytes:
                    st.download_button(
                        label="Download Async SHAP Images (ZIP)",
                        data=st.session_state.job_shap_zip_bytes,
                        file_name="shap_images_async.zip",
                        mime="application/zip",
                    )
        elif st.session_state.job_status == "failed":
            st.error(f"Job failed: {st.session_state.job_error}")


if __name__ == "__main__":
    main()