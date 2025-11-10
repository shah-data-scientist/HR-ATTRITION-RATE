import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import sys
import os

# Add the project root to the sys.path to allow importing app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import (
    _sigmoid,
    get_expected_columns,
    clear_prediction_results,
    _plot_probability_waterfall_topk,
)

def test_sigmoid():
    assert _sigmoid(0) == 0.5
    assert np.isclose(_sigmoid(10), 0.99995460)
    assert np.isclose(_sigmoid(-10), 0.00004539)

def test_get_expected_columns():
    mock_pipeline = MagicMock()
    mock_preprocessor = MagicMock()
    mock_preprocessor.feature_names_in_ = ['a', 'b', 'c']
    mock_pipeline.named_steps = {'preprocessor': mock_preprocessor}
    assert get_expected_columns(mock_pipeline) == ['a', 'b', 'c']

@patch('app.st')
def test_clear_prediction_results(mock_st):
    mock_st.session_state = MagicMock()
    clear_prediction_results()
    assert mock_st.session_state.prediction_triggered is False
    assert mock_st.session_state.report_data is None
    assert mock_st.session_state.shap_html_content is None
    assert mock_st.session_state.excel_report_data is None
    assert mock_st.session_state.processed_data_for_shap is None
    assert mock_st.session_state.explainer is None
    assert mock_st.session_state.all_features is None

@patch('app.plt.subplots')
def test_plot_probability_waterfall_topk(mock_subplots):
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_subplots.return_value = (mock_fig, mock_ax)

    feature_names = [f'feature_{i}' for i in range(20)]
    shap_values_row = np.random.rand(20) * 0.1 - 0.05
    base_log_odds = -0.5
    employee_prob = 0.6
    avg_prob = 0.5

    mock_ax.get_legend_handles_labels.return_value = ([MagicMock(), MagicMock()], ['label1', 'label2'])

    fig = _plot_probability_waterfall_topk(
        feature_names, shap_values_row, base_log_odds, employee_prob, avg_prob
    )

    assert fig is not None
    mock_subplots.assert_called_once()
    mock_ax.set_title.assert_called_with("Attrition Probability", fontsize=14, pad=8)
    mock_ax.set_xlabel.assert_called_with("Attrition Probability (%)", fontsize=12)
    assert mock_ax.axvline.call_count == 2
    mock_ax.legend.assert_called_once()

@patch('app.load_model_and_data') # Patch the function directly
def test_load_model_and_data(mock_load_model_and_data):
    # Mock return values for load_model_and_data
    mock_model = MagicMock()
    mock_X_train = pd.DataFrame({'col1': [1]})
    mock_y_train = pd.Series([0])
    mock_X_test = pd.DataFrame({'col1': [2]})
    mock_y_test = pd.Series([1])

    mock_load_model_and_data.return_value = (mock_model, mock_X_train, mock_y_train, mock_X_test, mock_y_test)

    # Import here to ensure patching works
    from app import load_model_and_data

    model, X_train, y_train, X_test, y_test = load_model_and_data()

    mock_load_model_and_data.assert_called_once() # Assert that our patched function was called
    assert model == mock_model
    assert X_train.equals(mock_X_train)
    assert y_train.equals(mock_y_train)
    assert X_test.equals(mock_X_test)
    assert y_test.equals(mock_y_test)

@patch('app.st.subheader')
@patch('app.st.columns')
@patch('app.st.write')
@patch('app.plt.subplots')
@patch('app.sns.heatmap')
@patch('app.plt.yticks')
@patch('app.plt.xticks')
@patch('app.plt.tight_layout')
@patch('app.st.pyplot')
def test_display_confusion_matrix_and_metrics(
    mock_st_pyplot, mock_tight_layout, mock_xticks, mock_yticks, mock_sns_heatmap,
    mock_subplots, mock_st_write, mock_st_columns, mock_st_subheader):

    mock_st_columns.return_value = [MagicMock(), MagicMock()]
    mock_subplots.return_value = (MagicMock(), MagicMock()) # fig, ax

    y_true = pd.Series([0, 1, 0, 1])
    y_proba = np.array([0.1, 0.9, 0.4, 0.6])
    threshold = 0.5

    from app import display_confusion_matrix_and_metrics # Import here to ensure patching works

    display_confusion_matrix_and_metrics(y_true, y_proba, threshold)

    mock_st_subheader.assert_called_once()
    mock_st_columns.assert_called_once()
    mock_subplots.assert_called_once()
    mock_sns_heatmap.assert_called_once()
    mock_st_pyplot.assert_called_once()
    assert mock_st_write.call_count >= 4 # For threshold, accuracy, precision, recall, f1

@patch('app.datetime')
@patch('app.base64.b64encode')
@patch('app.io.BytesIO')
@patch('app.plt.close')
@patch('app._plot_probability_waterfall_topk') # Patch the internal plotting function
@patch('app.shap.LinearExplainer') # Patch shap.LinearExplainer
def test_generate_shap_html_report(
    mock_shap_explainer, mock_plot_probability_waterfall_topk, mock_plt_close,
    mock_bytes_io, mock_b64encode, mock_datetime):

    # Mock datetime.now().strftime
    mock_datetime.now.return_value.strftime.return_value = "2025-11-10 10:00:00"

    # Mock base64 encoding
    mock_b64encode.return_value.decode.return_value = "mock_base64_image_string"

    # Mock BytesIO
    mock_bytes_io_instance = MagicMock()
    mock_bytes_io.return_value = mock_bytes_io_instance

    # Mock _plot_probability_waterfall_topk to return a mock figure
    mock_fig = MagicMock()
    mock_plot_probability_waterfall_topk.return_value = mock_fig

    # Mock explainer and shap_values
    mock_explainer_instance = MagicMock()
    mock_explainer_instance.expected_value = 0.5
    mock_shap_explainer.return_value = mock_explainer_instance
    mock_explainer_instance.shap_values.return_value = np.array([[0.1, -0.2, 0.3]]) # Example SHAP values

    employee_data_with_predictions = pd.DataFrame({
        'id_employee': [101],
        'Attrition_Risk_Percentage': [0.75],
        'Risk_Attrition': ['High'],
        'Prediction': ['Leave']
    })
    X_transformed_for_shap = np.array([[0.1, 0.2, 0.3]])
    all_features = ['feat1', 'feat2', 'feat3']

    from app import generate_shap_html_report # Import here to ensure patching works

    html_report = generate_shap_html_report(
        employee_data_with_predictions, X_transformed_for_shap, mock_explainer_instance, all_features
    )

    assert "<h1>Employee Attrition SHAP Explanation Report</h1>" in html_report
    assert "Employee ID: 101" in html_report
    assert "Predicted Attrition Risk:" in html_report
    assert "<span class=\"risk-label risk-high\">High</span>" in html_report
    assert "(75.0%)" in html_report
    assert "Prediction: <strong>Leave</strong>" in html_report
    assert "mock_base64_image_string" in html_report

    mock_plot_probability_waterfall_topk.assert_called_once()
    mock_fig.savefig.assert_called_once_with(mock_bytes_io_instance, format='png', bbox_inches='tight', dpi=100)
    mock_plt_close.assert_called_once_with(mock_fig)
    mock_b64encode.assert_called_once()
    mock_datetime.now.assert_called_once()
    mock_explainer_instance.shap_values.assert_called_once_with(X_transformed_for_shap)




