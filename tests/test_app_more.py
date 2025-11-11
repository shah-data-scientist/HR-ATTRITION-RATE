import os
import sys
from unittest.mock import ANY, MagicMock, patch

import numpy as np
import pandas as pd

# Add the project root to the sys.path to allow importing app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import (
    LOG_ODDS_RISK_THRESHOLDS,  # Added LOG_ODDS_RISK_THRESHOLDS
    _get_risk_category_from_log_odds,  # Added _get_risk_category_from_log_odds
    _logit,  # Added _logit
    _sigmoid,
    clear_prediction_results,
    get_expected_columns,
)


def test_sigmoid():
    assert _sigmoid(0) == 0.5
    assert np.isclose(_sigmoid(10), 0.99995460)
    assert np.isclose(_sigmoid(-10), 0.00004539)


def test_logit():
    assert _logit(0.5) == 0
    assert np.isclose(_logit(0.3), -0.8472978603875038)
    assert np.isclose(_logit(0.7), 0.8472978603875038)
    assert _logit(0) == -np.inf
    assert _logit(1) == np.inf


def test_get_risk_category_from_log_odds():
    # Test cases based on LOG_ODDS_RISK_THRESHOLDS
    low_threshold = LOG_ODDS_RISK_THRESHOLDS["Low"][1]  # approx -0.847
    high_threshold = LOG_ODDS_RISK_THRESHOLDS["High"][0]  # approx 0.847

    assert (
        _get_risk_category_from_log_odds(low_threshold - 0.1) == "Low"
    )  # Below low threshold
    assert (
        _get_risk_category_from_log_odds(low_threshold) == "Medium"
    )  # At low threshold
    assert (
        _get_risk_category_from_log_odds(low_threshold + 0.1) == "Medium"
    )  # Between thresholds
    assert (
        _get_risk_category_from_log_odds(high_threshold - 0.1) == "Medium"
    )  # Between thresholds
    assert (
        _get_risk_category_from_log_odds(high_threshold) == "High"
    )  # At high threshold
    assert (
        _get_risk_category_from_log_odds(high_threshold + 0.1) == "High"
    )  # Above high threshold
    assert _get_risk_category_from_log_odds(-np.inf) == "Low"
    assert _get_risk_category_from_log_odds(np.inf) == "High"


def test_get_expected_columns():
    mock_pipeline = MagicMock()
    mock_preprocessor = MagicMock()
    mock_preprocessor.feature_names_in_ = ["a", "b", "c"]
    mock_pipeline.named_steps = {"preprocessor": mock_preprocessor}
    assert get_expected_columns(mock_pipeline) == ["a", "b", "c"]


@patch("app.st")
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


@patch("app.datetime")
@patch("app.base64.b64encode")
@patch("app.io.BytesIO")
@patch("app.plt.close")
@patch("matplotlib.pyplot.gcf")  # Patch matplotlib.pyplot.gcf
@patch("PIL.Image.Image.save")  # Re-add patch for PIL.Image.Image.save
@patch("shap.plots.waterfall")  # Patch shap.plots.waterfall
@patch("app.shap.LinearExplainer")  # Patch app.shap.LinearExplainer
@patch("shap.Explanation")  # Patch shap.Explanation
def test_generate_shap_html_report(
    mock_shap_explanation,
    mock_shap_explainer,
    mock_shap_plots_waterfall,
    mock_pil_image_save,
    mock_plt_gcf,
    mock_plt_close,
    mock_bytes_io,
    mock_b64encode,
    mock_datetime,
):

    # Mock datetime.now().strftime
    mock_datetime.now.return_value.strftime.return_value = "2025-11-10 10:00:00"

    # Mock base64 encoding
    mock_b64encode.return_value.decode.return_value = "mock_base64_image_string"

    # Mock BytesIO
    mock_bytes_io_instance = MagicMock()
    mock_bytes_io.return_value = mock_bytes_io_instance

    # Mock shap.plots.waterfall to return a mock figure
    mock_fig = MagicMock()
    mock_shap_plots_waterfall.return_value = mock_fig
    mock_plt_gcf.return_value = mock_fig  # Ensure plt.gcf returns our mock figure

    # Mock explainer and shap_values
    mock_explainer_instance = MagicMock()
    mock_explainer_instance.expected_value = 0.5
    mock_shap_explainer.return_value = mock_explainer_instance
    mock_explainer_instance.shap_values.return_value = np.array(
        [[0.1, -0.2, 0.3]]
    )  # Example SHAP values

    # Mock shap.Explanation
    mock_explanation_instance = MagicMock()
    mock_shap_explanation.return_value = mock_explanation_instance

    employee_data_with_predictions = pd.DataFrame(
        {
            "id_employee": [101],
            "Attrition_Risk_Percentage": [0.75],
            "Risk_Attrition": ["High"],
            "Prediction": ["Leave"],
        }
    )
    X_transformed_for_shap = np.array([[0.1, 0.2, 0.3]])
    all_features = ["feat1", "feat2", "feat3"]

    from app import generate_shap_html_report  # Import here to ensure patching works

    html_report = generate_shap_html_report(
        employee_data_with_predictions,
        X_transformed_for_shap,
        mock_explainer_instance,
        all_features,
    )

    assert "<h1>Employee Attrition SHAP Explanation Report</h1>" in html_report
    assert "<h2>Employee ID: 101</h2>" in html_report  # New assertion for prominent ID
    assert "Predicted Attrition Risk:" in html_report
    assert '<span class="risk-label risk-high">High</span>' in html_report
    assert "(75.0%)" in html_report
    assert "Prediction: <strong>Leave</strong>" in html_report
    assert "mock_base64_image_string" in html_report

    mock_shap_explanation.assert_called_once_with(
        values=ANY,  # Use ANY for NumPy array
        base_values=0.5,
        data=ANY,  # Use ANY for NumPy array
        feature_names=all_features,
    )
    mock_shap_plots_waterfall.assert_called_once_with(
        mock_explanation_instance, max_display=10, show=False
    )
    mock_fig.savefig.assert_called_once_with(
        mock_bytes_io_instance, format="png", bbox_inches="tight", dpi=100
    )
    mock_plt_close.assert_called_once_with(mock_fig)
    mock_b64encode.assert_called_once()
    mock_datetime.now.assert_called_once()
    mock_explainer_instance.shap_values.assert_called_once_with(ANY)
