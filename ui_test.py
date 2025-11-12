
import io
import pandas as pd
import streamlit as st
from streamlit_app import (
    _load_local_csv_files,
    _process_predictions_and_reports,
    _load_and_process_data,
    load_model_and_data,
    display_confusion_matrix_and_metrics,
    generate_shap_report_data,
)
import traceback

def run_ui_test():
    print("Starting UI test...")

    # Manually initialize session state for testing
    (
        st.session_state.model,
        st.session_state.x_train_loaded,
        st.session_state.y_train_loaded,
        st.session_state.x_test_loaded,
        st.session_state.y_test_loaded,
    ) = load_model_and_data()
    st.session_state.all_features = st.session_state.model.named_steps[
        "preprocessor"
    ].get_feature_names_out()
    st.session_state.train_prediction_proba = st.session_state.model.predict_proba(
        st.session_state.x_train_loaded
    )[:, 1]
    st.session_state.prediction_triggered = False
    st.session_state.report_data = None
    st.session_state.shap_html_content = None
    st.session_state.excel_report_data = None
    st.session_state.processed_data_for_shap = None
    st.session_state.explainer = None


    # Load test data
    eval_file, sirh_file, sondage_file = _load_local_csv_files()
    if not (eval_file and sirh_file and sondage_file):
        print("Failed to load local CSV files.")
        return

    # Simulate button click
    try:
        processed_df, merged_df = _load_and_process_data(
            eval_file, sirh_file, sondage_file
        )
        _process_predictions_and_reports(
            processed_df, merged_df, 0.5
        )
        
        # Test confusion matrix
        metrics = display_confusion_matrix_and_metrics(
            st.session_state.y_test_loaded,
            st.session_state.model.predict_proba(st.session_state.x_test_loaded)[:, 1],
            0.5,
            "Test Confusion Matrix"
        )
        
        assert metrics is not None
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1_score"] <= 1
        
        # Test SHAP data
        shap_report_data = generate_shap_report_data(
            st.session_state.report_data,
            st.session_state.processed_data_for_shap,
            st.session_state.explainer,
            st.session_state.all_features,
        )
        
        assert shap_report_data is not None
        assert len(shap_report_data) > 0
        assert "employee_id" in shap_report_data[0].columns
        
        print("UI test passed!")
    except Exception as e:
        print(f"UI test failed with error: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    run_ui_test()
