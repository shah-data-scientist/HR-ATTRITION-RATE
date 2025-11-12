import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
import streamlit as st
import sys
import os

# Add the project root to the sys.path to allow importing app.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import (
    display_confusion_matrix_and_metrics,
    render_threshold_block,
    render_employee_overview_and_shap,
    get_risk_category
)

class TestStreamlitUI(unittest.TestCase):

    def setUp(self):
        # Mock st.session_state as a MagicMock
        self.patcher_session_state = patch('streamlit.session_state', new_callable=MagicMock)
        self.mock_session_state = self.patcher_session_state.start()

        # Configure the MagicMock to handle 'in' operator by checking its attributes
        def mock_contains(key):
            # Check if the key exists as an attribute on the mock object
            # This works because setting attributes on MagicMock stores them internally
            return hasattr(self.mock_session_state, key)
        self.mock_session_state.__contains__.side_effect = mock_contains

        # Mock Streamlit functions
        self.patcher_st_subheader = patch('streamlit.subheader')
        self.mock_st_subheader = self.patcher_st_subheader.start()
        self.patcher_st_markdown = patch('streamlit.markdown')
        self.mock_st_markdown = self.patcher_st_markdown.start()
        self.patcher_st_pyplot = patch('streamlit.pyplot')
        self.mock_st_pyplot = self.patcher_st_pyplot.start()
        self.patcher_st_slider = patch('streamlit.slider')
        self.mock_st_slider = self.patcher_st_slider.start()
        self.patcher_st_dataframe = patch('streamlit.dataframe')
        self.mock_st_dataframe = self.patcher_st_dataframe.start()
        self.patcher_st_selectbox = patch('streamlit.selectbox')
        self.mock_st_selectbox = self.patcher_st_selectbox.start()
        self.patcher_st_error = patch('streamlit.error')
        self.mock_st_error = self.patcher_st_error.start()
        self.patcher_st_exception = patch('streamlit.exception')
        self.mock_st_exception = self.patcher_st_exception.start()
        self.patcher_st_warning = patch('streamlit.warning')
        self.mock_st_warning = self.patcher_st_warning.start()
        self.patcher_st_download_button = patch('streamlit.download_button')
        self.mock_st_download_button = self.patcher_st_download_button.start()

        # Mock shap.waterfall_plot
        self.patcher_shap_waterfall_plot = patch('shap.waterfall_plot')
        self.mock_shap_waterfall_plot = self.patcher_shap_waterfall_plot.start()

        # Mock seaborn.heatmap
        self.patcher_sns_heatmap = patch('seaborn.heatmap')
        self.mock_sns_heatmap = self.patcher_sns_heatmap.start()

        # Mock matplotlib.pyplot functions
        self.patcher_plt_subplots = patch('matplotlib.pyplot.subplots')
        self.mock_plt_subplots = self.patcher_plt_subplots.start()
        self.mock_plt_subplots.return_value = (MagicMock(), MagicMock()) # Return a mock figure and axes
        self.patcher_plt_gcf = patch('matplotlib.pyplot.gcf')
        self.mock_plt_gcf = self.patcher_plt_gcf.start()
        self.mock_plt_gcf.return_value = MagicMock() # Return a mock figure
        self.patcher_plt_close = patch('matplotlib.pyplot.close')
        self.mock_plt_close = self.patcher_plt_close.start()


        # Synthetic data for testing
        self.y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        self.y_proba = np.array([0.1, 0.8, 0.2, 0.9, 0.3, 0.7, 0.4, 0.6, 0.5, 0.95])
        self.threshold = 0.5

        # Populate the mocked session_state by setting attributes directly
        self.mock_session_state.y_train_loaded = self.y_true
        self.mock_session_state.train_prediction_proba = self.y_proba
        self.mock_session_state.prediction_triggered = True
        self.mock_session_state.current_threshold = self.threshold

        # Mock model and explainer for SHAP
        mock_model = MagicMock()
        mock_model.named_steps = {'preprocessor': MagicMock(), 'model': MagicMock()}
        mock_model.named_steps['preprocessor'].get_feature_names_out.return_value = [f'feature_{i}' for i in range(5)]
        self.mock_session_state.model = mock_model
        self.mock_session_state.all_features = [f'feature_{i}' for i in range(5)]

        # Create dummy processed_data_for_shap
        self.mock_session_state.processed_data_for_shap = pd.DataFrame(
            np.random.rand(len(self.y_true), len(self.mock_session_state.all_features)),
            columns=self.mock_session_state.all_features
        )

        # Mock SHAP explainer
        mock_explainer = MagicMock()
        mock_explainer.expected_value = np.array([0.1, 0.5]) # For binary classification
        mock_explainer.shap_values.return_value = np.random.rand(len(self.mock_session_state.all_features))
        self.mock_session_state.explainer = mock_explainer

        # Mock report_data for employee overview
        self.mock_session_state.report_data = pd.DataFrame({
            "id_employee": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            "Attrition_Risk_Percentage": self.y_proba,
            "Prediction": np.where(self.y_proba >= self.threshold, "Leave", "Stay"),
            "Risk_Attrition": [get_risk_category(p, self.threshold) for p in self.y_proba]
        })

        # Mock excel_report_data for Excel download
        self.mock_session_state.excel_report_data = pd.DataFrame({
            "Employee_ID": [101, 102, 103],
            "Feature": ["feature_0", "feature_1", "feature_2"],
            "Coefficient": [0.1, 0.2, -0.1],
            "Prediction": ["Leave", "Stay", "Leave"]
        })


    def tearDown(self):
        self.patcher_session_state.stop()
        self.patcher_st_subheader.stop()
        self.patcher_st_markdown.stop()
        self.patcher_st_pyplot.stop()
        self.patcher_st_slider.stop()
        self.patcher_st_dataframe.stop()
        self.patcher_st_selectbox.stop()
        self.patcher_st_error.stop()
        self.patcher_st_exception.stop()
        self.patcher_st_warning.stop()
        self.patcher_st_download_button.stop()
        self.patcher_shap_waterfall_plot.stop()
        self.patcher_sns_heatmap.stop() # Stop the seaborn.heatmap patcher
        self.patcher_plt_subplots.stop()
        self.patcher_plt_gcf.stop()
        self.patcher_plt_close.stop()

    def test_confusion_matrix_percentages(self):
        # Test CM row sums to ~100%
        accuracy, recall, flagged_count = display_confusion_matrix_and_metrics(
            self.y_true, self.y_proba, self.threshold, "Test CM"
        )
        
        y_pred = (self.y_proba >= self.threshold).astype(int)
        cm = np.array(pd.crosstab(pd.Series(self.y_true), pd.Series(y_pred), rownames=['Actual'], colnames=['Predicted']))
        
        cm_normalized = np.zeros_like(cm, dtype=float)
        row_sums = cm.sum(axis=1)
        for i in range(cm.shape[0]):
            if row_sums[i] > 0:
                cm_normalized[i, :] = cm[i, :] / row_sums[i]
        
        # Assert that each row sums to approximately 1 (100%)
        for row in cm_normalized:
            self.assertAlmostEqual(np.sum(row), 1.0, delta=0.001) # Allow ±0.1%

        # Assert that st.pyplot was called with use_container_width=True
        self.mock_st_pyplot.assert_called_with(unittest.mock.ANY, use_container_width=True)

    def test_vertical_threshold_slider_updates(self):
        # Mock the slider to return a specific value
        self.mock_st_slider.return_value = 0.6

        render_threshold_block()

        # Assert that st.slider was called with orientation="vertical"
        self.mock_st_slider.assert_called_with(
            "Set 'High Risk' Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            orientation="vertical",
            height=200,
            help=unittest.mock.ANY
        )
        # Assert that the current_threshold in session state was updated
        self.assertEqual(self.mock_session_state.current_threshold, 0.6)

        # Assert that display_confusion_matrix_and_metrics was called with the new threshold
        self.mock_st_subheader.assert_any_call("Prediction Accuracy Overview")

    def test_individual_employee_analysis_success(self):
        # Mock selectbox to return a specific employee ID
        self.mock_st_selectbox.return_value = 101

        render_employee_overview_and_shap()

        # Assert that SHAP waterfall plot was generated (st.pyplot called)
        self.mock_st_pyplot.assert_called()
        # Assert no error messages were displayed
        self.mock_st_error.assert_not_called()
        self.mock_st_exception.assert_not_called()

    def test_individual_employee_analysis_error_no_explainer(self):
        # Mock selectbox to return a specific employee ID
        self.mock_st_selectbox.return_value = 101
        # Simulate missing explainer
        del self.mock_session_state.explainer # Remove explainer from session state

        render_employee_overview_and_shap()

        # Assert error message was displayed
        self.mock_st_error.assert_called_with("SHAP explainer not found in session state. Please ensure the model is loaded and predictions are made.")
        self.mock_st_pyplot.assert_not_called() # No plot should be generated

    def test_individual_employee_analysis_error_no_processed_data(self):
        # Mock selectbox to return a specific employee ID
        self.mock_st_selectbox.return_value = 101
        # Simulate missing processed_data_for_shap
        self.mock_session_state.processed_data_for_shap = pd.DataFrame() # Empty DataFrame

        render_employee_overview_and_shap()

        # Assert error message was displayed
        self.mock_st_error.assert_called_with("Processed data for SHAP not found in session state. Cannot display SHAP plot.")
        self.mock_st_pyplot.assert_not_called() # No plot should be generated

    def test_excel_export_button_and_content(self):
        render_employee_overview_and_shap()

        # Assert download button was rendered
        self.mock_st_download_button.assert_called_once()
        
        # Get the data passed to the download button
        call_args, call_kwargs = self.mock_st_download_button.call_args
        download_data_buffer = call_kwargs['data']
        
        # Read the Excel file from the buffer
        download_data_buffer.seek(0)
        xls = pd.ExcelFile(download_data_buffer)

        # Check sheet names
        self.assertIn("Summary", xls.sheet_names)
        self.assertIn("Features", xls.sheet_names)
        self.assertIn("Metrics", xls.sheet_names)

        # Check content of "Summary" sheet
        summary_df = pd.read_excel(xls, sheet_name="Summary")
        self.assertIn("Employee_ID", summary_df.columns)
        self.assertIn("Risk_Attrition", summary_df.columns)
        self.assertIn("Attrition_Risk_Percentage", summary_df.columns)
        self.assertIn("Prediction", summary_df.columns)
        self.assertFalse(summary_df.empty)
        self.assertEqual(len(summary_df), len(self.mock_session_state.report_data))

        # Check content of "Features" sheet
        features_df = pd.read_excel(xls, sheet_name="Features")
        self.assertIn("Employee_ID", features_df.columns)
        self.assertIn("Feature", features_df.columns)
        self.assertIn("Coefficient", features_df.columns)
        self.assertIn("Prediction", features_df.columns)
        self.assertFalse(features_df.empty)
        self.assertEqual(len(features_df), len(self.mock_session_state.excel_report_data))

        # Check content of "Metrics" sheet
        metrics_df = pd.read_excel(xls, sheet_name="Metrics")
        self.assertIn("Metric", metrics_df.columns)
        self.assertIn("Value", metrics_df.columns)
        self.assertFalse(metrics_df.empty)
        self.assertGreater(len(metrics_df), 0)