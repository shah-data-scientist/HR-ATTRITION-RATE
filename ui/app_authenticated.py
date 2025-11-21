"""
Authenticated wrapper for the HR Attrition Risk Streamlit application.
This module adds authentication layer before allowing access to the main app.
"""
import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import authentication functions
from ui.auth import require_authentication, show_login_page, show_user_info

# Optimal threshold from model training
OPTIMAL_THRESHOLD = 0.2876

# Initialize session state for authentication
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None
if "user_role" not in st.session_state:
    st.session_state.user_role = None
if "user_id" not in st.session_state:
    st.session_state.user_id = None

# Check if user is authenticated
if not require_authentication():
    # Show login page if not authenticated
    show_login_page()
else:
    # Initialize all required session state variables for the main app
    # These must be initialized BEFORE importing/calling app.main()
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
    if "threshold" not in st.session_state:
        st.session_state.threshold = OPTIMAL_THRESHOLD
    if "predictions_raw" not in st.session_state:
        st.session_state.predictions_raw = None

    # User is authenticated, show the main application
    # Import main app after authentication
    from ui import app

    # Show user info in sidebar with logout button
    show_user_info()

    # Call the main app function
    app.main()
