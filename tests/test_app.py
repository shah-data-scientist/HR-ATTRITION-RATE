import sys
from pathlib import Path
from typing import Any, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import shap
from shap import Explanation

# Add the project root to the sys.path to allow importing app
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import (
    _clean_extrait_eval,
    _clean_extrait_sirh,
    _clean_extrait_sondage,
    clean_and_engineer_features,
    display_confusion_matrix_and_metrics,
    generate_shap_html_report,
    get_risk_category,
    load_and_merge_data,
    load_model_and_data,
)


# Mock data for testing
@pytest.fixture
def sample_eval_df() -> pd.DataFrame:
    """Provide a sample DataFrame for evaluation data."""
    return pd.DataFrame(
        {
            "eval_number": ["E_1", "E_2", "E_3", "E_4"],
            "augmentation_salaire_precedente": ["10%", "5,5%", "0%", "invalid"],
            "heures_supplementaires": ["Oui", "Non", True, False],
            "note_evaluation_actuelle": [4, 3, 5, 2],
            "note_evaluation_precedente": [3, 3, 4, 3],
        }
    )


@pytest.fixture
def sample_sirh_df() -> pd.DataFrame:
    """Provide a sample DataFrame for SIRH data."""
    return pd.DataFrame(
        {
            "id_employee": [1, 2, 3, 5],
            "genre": ["M", "F", "m", "f"],
            "nombre_heures_travailless": [
                160,
                150,
                170,
                180,
            ],  # This column should be dropped
        }
    )


@pytest.fixture
def sample_sondage_df() -> pd.DataFrame:
    """Provide a sample DataFrame for survey data."""
    return pd.DataFrame(
        {
            "code_sondage": [1, 2, 4, 5],
            "satisfaction_employee_nature_travail": [4, 3, 5, 2],
            "satisfaction_employee_equipe": [5, 4, 3, 3],
            "satisfaction_employee_equilibre_pro_perso": [3, 2, 4, 5],
        }
    )


@pytest.fixture
def merged_df_for_feature_engineering() -> pd.DataFrame:
    """Provide a sample DataFrame for feature engineering tests."""
    return pd.DataFrame(
        {
            "id_employee": [1, 2, 3, 4, 5],
            "note_evaluation_actuelle": [4, 3, 5, 2, 4],
            "note_evaluation_precedente": [3, 3, 4, 3, 2],
            "satisfaction_employee_nature_travail": [4, 3, 5, 2, 4],
            "satisfaction_employee_equipe": [5, 4, 3, 3, 5],
            "satisfaction_employee_equilibre_pro_perso": [3, 2, 4, 5, 3],
            "annees_dans_le_poste_actuel": [2, 1, 3, 0, 2],
            "annees_dans_l_entreprise": [5, 3, 3, 0, 4],
        }
    )


@pytest.fixture
def mock_model_pipeline() -> MagicMock:
    """Provide a mock ML pipeline for testing."""
    # Create a mock preprocessor with a get_feature_names_out method
    mock_preprocessor = MagicMock()
    mock_preprocessor.feature_names_in_ = ["feature_A", "feature_B", "feature_C"]
    mock_preprocessor.transform.return_value = np.array([[0.1, 0.2, 0.3]])

    # Mock the named_transformers_ attribute for categorical features
    mock_ohe = MagicMock()
    mock_ohe.get_feature_names_out.return_value = ["cat_feature_X", "cat_feature_Y"]
    mock_preprocessor.named_transformers_ = {"cat": mock_ohe}

    # Create a mock logistic regression model
    mock_logreg_model = MagicMock()
    mock_logreg_model.predict_proba.return_value = np.array(
        [[0.2, 0.8]]
    )  # Example probability

    # Create a mock pipeline
    mock_pipeline = MagicMock()
    mock_pipeline.named_steps = {
        "preprocessor": mock_preprocessor,
        "model": mock_logreg_model,
    }
    mock_pipeline.predict_proba.return_value = np.array([[0.2, 0.8]])
    return mock_pipeline


# --- Test _clean_extrait_eval ---
def test_clean_extrait_eval_salary_augmentation(sample_eval_df: pd.DataFrame) -> None:
    """Test cleaning of salary augmentation column in eval data."""
    df = _clean_extrait_eval(sample_eval_df.copy())
    assert df["augmentation_salaire_precedente"].iloc[0] == 0.10  # 10% as float
    assert df["augmentation_salaire_precedente"].iloc[1] == 0.055  # 5.5% as float
    assert df["augmentation_salaire_precedente"].iloc[2] == 0.00
    assert pd.isna(df["augmentation_salaire_precedente"].iloc[3])


def test_clean_extrait_eval_heures_supplementaires(
    sample_eval_df: pd.DataFrame,
) -> None:
    """Test cleaning of 'heures_supplementaires' column in eval data."""
    df = _clean_extrait_eval(sample_eval_df.copy())
    assert df["heures_supplementaires"].iloc[0] == 1
    assert df["heures_supplementaires"].iloc[1] == 0
    assert df["heures_supplementaires"].iloc[2] == 1
    assert df["heures_supplementaires"].iloc[3] == 0


def test_clean_extrait_eval_id_employee(sample_eval_df: pd.DataFrame) -> None:
    """Test extraction and cleaning of 'id_employee' from eval data."""
    df = _clean_extrait_eval(sample_eval_df.copy())
    assert "eval_number" not in df.columns
    assert df["id_employee"].iloc[0] == 1
    assert df["id_employee"].iloc[1] == 2  # Expected id_employee for the second entry


def test_clean_extrait_eval_missing_columns() -> None:
    """Test cleaning of eval data with missing columns."""
    df = pd.DataFrame({"col_a": [1, 2]})
    cleaned_df = _clean_extrait_eval(df.copy())
    assert "augmentation_salaire_precedente" not in cleaned_df.columns
    assert "heures_supplementaires" not in cleaned_df.columns
    assert "id_employee" not in cleaned_df.columns


# --- Test _clean_extrait_sirh ---
def test_clean_extrait_sirh_genre(sample_sirh_df: pd.DataFrame) -> None:
    """Test cleaning of 'genre' column in SIRH data."""
    df = _clean_extrait_sirh(sample_sirh_df.copy())
    assert df["genre"].iloc[0] == 1
    assert df["genre"].iloc[1] == 0
    assert df["genre"].iloc[2] == 1
    assert df["genre"].iloc[3] == 0


def test_clean_extrait_sirh_drop_columns(sample_sirh_df: pd.DataFrame) -> None:
    """Test dropping of unnecessary columns in SIRH data."""
    df = _clean_extrait_sirh(sample_sirh_df.copy())
    assert "nombre_heures_travailless" not in df.columns


def test_clean_extrait_sirh_missing_columns() -> None:
    """Test cleaning of SIRH data with missing columns."""
    df = pd.DataFrame({"col_a": [1, 2]})
    cleaned_df = _clean_extrait_sirh(df.copy())
    assert "genre" not in cleaned_df.columns


# --- Test _clean_extrait_sondage ---
def test_clean_extrait_sondage_id_employee(sample_sondage_df: pd.DataFrame) -> None:
    """Test cleaning of 'id_employee' in survey data."""
    df = _clean_extrait_sondage(sample_sondage_df.copy())
    assert "code_sondage" not in df.columns
    assert df["id_employee"].iloc[0] == 1
    assert df["id_employee"].iloc[2] == 4  # Expected id_employee for the third entry


def test_clean_extrait_sondage_missing_columns() -> None:
    """Test cleaning of survey data with missing columns."""
    df = pd.DataFrame({"col_a": [1, 2]})
    cleaned_df = _clean_extrait_sondage(df.copy())
    assert "id_employee" not in cleaned_df.columns


# --- Test load_and_merge_data ---
def test_load_and_merge_data_merge_logic(
    sample_eval_df: pd.DataFrame,
    sample_sirh_df: pd.DataFrame,
    sample_sondage_df: pd.DataFrame,
) -> None:
    """Test the merging logic of load_and_merge_data."""
    merged_df = load_and_merge_data(
        sample_eval_df.copy(), sample_sirh_df.copy(), sample_sondage_df.copy()
    )
    assert len(merged_df) == 5  # Total expected employees after outer merge
    assert "augmentation_salaire_precedente" in merged_df.columns
    assert "genre" in merged_df.columns
    assert "satisfaction_employee_nature_travail" in merged_df.columns
    assert merged_df.loc[merged_df["id_employee"] == 1, "genre"].iloc[0] == 1
    assert (
        merged_df.loc[
            merged_df["id_employee"] == 4, "satisfaction_employee_equipe"
        ].iloc[0]
        == 3  # Expected satisfaction for employee 4
    )
    assert pd.isna(
        merged_df.loc[
            merged_df["id_employee"] == 3, "satisfaction_employee_equipe"
        ].iloc[0]
    )  # Expected NaN for employee 3


def test_load_and_merge_data_empty_dfs() -> None:
    """Test load_and_merge_data with empty DataFrames."""
    eval_df = pd.DataFrame()
    sirh_df = pd.DataFrame()
    sond_df = pd.DataFrame()
    merged_df = load_and_merge_data(eval_df, sirh_df, sond_df)
    assert merged_df.empty


# --- Test clean_and_engineer_features ---
def test_clean_and_engineer_features_improvement_evaluation(
    merged_df_for_feature_engineering: pd.DataFrame,
) -> None:
    """Test 'improvement_evaluation' feature engineering."""
    df = clean_and_engineer_features(merged_df_for_feature_engineering.copy())
    assert df["improvement_evaluation"].iloc[0] == 1
    assert df["improvement_evaluation"].iloc[1] == 0
    assert df["improvement_evaluation"].iloc[2] == 1
    assert df["improvement_evaluation"].iloc[3] == -1


def test_clean_and_engineer_features_total_satisfaction(
    merged_df_for_feature_engineering: pd.DataFrame,
) -> None:
    """Test 'total_satisfaction' feature engineering."""
    df = clean_and_engineer_features(merged_df_for_feature_engineering.copy())
    assert df["total_satisfaction"].iloc[0] == 4 * 5 * 3
    assert df["total_satisfaction"].iloc[1] == 3 * 4 * 2


def test_clean_and_engineer_features_work_mobility(
    merged_df_for_feature_engineering: pd.DataFrame,
) -> None:
    """Test 'work_mobility' feature engineering."""
    df = clean_and_engineer_features(merged_df_for_feature_engineering.copy())
    assert df["work_mobility"].iloc[0] == 2 / 5
    assert df["work_mobility"].iloc[1] == 1 / 3
    assert df["work_mobility"].iloc[3] == 0  # Division by zero handled


def test_clean_and_engineer_features_missing_columns() -> None:
    """Test feature engineering with missing columns."""
    df = pd.DataFrame({"id_employee": [1]})
    cleaned_df = clean_and_engineer_features(df.copy())
    assert "improvement_evaluation" not in cleaned_df.columns
    assert "total_satisfaction" not in cleaned_df.columns
    assert "work_mobility" not in cleaned_df.columns


# --- Test get_risk_category ---
def test_get_risk_category_low() -> None:
    """Test get_risk_category for low risk probabilities."""
    assert get_risk_category(0.1, 0.5) == "Low"
    assert get_risk_category(0.4, 0.5) == "Low"  # 0.4 < 0.45 (threshold - buffer)
    assert get_risk_category(0.29, 0.5) == "Low"  # 0.29 < 0.45 (threshold - buffer)


def test_get_risk_category_medium() -> None:
    """Test get_risk_category for medium risk probabilities."""
    assert (
        get_risk_category(0.45, 0.5) == "Medium"
    )  # 0.45 is threshold - buffer, and >= min_medium_prob
    assert (
        get_risk_category(0.5, 0.5) == "Medium"
    )  # 0.5 is between threshold - buffer and threshold + buffer, and >= min_medium_prob
    assert (
        get_risk_category(0.54, 0.5) == "Medium"
    )  # 0.54 is between threshold - buffer and threshold + buffer, and >= min_medium_prob
    assert get_risk_category(0.25, 0.5) == "Low"  # 0.25 < 0.45 (threshold - buffer)


def test_get_risk_category_high() -> None:
    """Test get_risk_category for high risk probabilities."""
    assert get_risk_category(0.55, 0.5) == "High"  # 0.55 is threshold + buffer
    assert get_risk_category(0.7, 0.5) == "High"
    assert get_risk_category(0.9, 0.5) == "High"
    assert get_risk_category(1.0, 0.5) == "High"  # Edge case for upper bound


def test_get_risk_category_edge_cases() -> None:
    """Test get_risk_category for edge cases."""
    assert get_risk_category(0.0, 0.5) == "Low"
    assert get_risk_category(0.9999999999999999, 0.5) == "High"  # Just under 1.0
    assert get_risk_category(1.0, 0.5) == "High"
    assert get_risk_category(0.44, 0.5) == "Low"  # Just below threshold - buffer
    assert (
        get_risk_category(0.49, 0.5) == "Medium"
    )  # Between threshold - buffer and threshold + buffer, and >= min_medium_prob
    assert (
        get_risk_category(0.51, 0.5) == "Medium"
    )  # Between threshold - buffer and threshold + buffer, and >= min_medium_prob
    assert get_risk_category(0.55, 0.5) == "High"  # At threshold + buffer
    assert (
        get_risk_category(0.19, 0.5) == "Low"
    )  # Below min_medium_prob, so defaults to Low


# --- New Tests for expanded coverage ---


# Test load_model_and_data
@patch("app.joblib.load")  # Patch app.joblib.load directly
@patch("app.pd.read_parquet")
def test_load_model_and_data(
    mock_read_parquet: MagicMock,
    mock_joblib_load: MagicMock,
    mock_model_pipeline: MagicMock,
) -> None:
    """Test loading of the ML model and data."""
    mock_joblib_load.return_value = mock_model_pipeline
    mock_read_parquet.side_effect = [
        pd.DataFrame({"col1": [1]}),  # x_train
        pd.Series([0]),  # y_train
        pd.DataFrame({"col1": [2]}),  # x_test
        pd.Series([1]),  # y_test
    ]

    model, x_train, y_train, x_test, y_test = load_model_and_data()

    mock_joblib_load.assert_called_once_with("outputs/employee_attrition_pipeline.pkl")
    assert mock_read_parquet.call_count == 4  # Called for x_train, y_train, x_test, y_test
    assert isinstance(model, MagicMock)
    assert not x_train.empty
    assert y_train is not None
    assert not x_test.empty
    assert y_test is not None


# Test display_confusion_matrix_and_metrics
@pytest.fixture
def mock_st_elements():
    """Fixture to provide a mock Streamlit object with necessary elements."""
    mock_st = MagicMock()
    mock_st.columns.return_value = [MagicMock(), MagicMock()]
    mock_st.pyplot = MagicMock()
    mock_st.write = MagicMock()
    mock_st.subheader = MagicMock()
    return mock_st

@pytest.fixture
def mock_plt_elements():
    """Fixture to provide mock matplotlib elements."""
    mock_plt = MagicMock()
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)
    mock_plt.gca.return_value = mock_ax
    mock_plt.xticks = MagicMock()
    mock_plt.yticks = MagicMock()
    mock_plt.tight_layout = MagicMock()
    mock_sns_heatmap = MagicMock()
    return mock_plt, mock_fig, mock_ax, mock_sns_heatmap

def test_display_confusion_matrix_and_metrics(
    mock_st_elements: MagicMock,
    mock_plt_elements: tuple[MagicMock, MagicMock, MagicMock, MagicMock],
) -> None:
    """Test the display of confusion matrix and metrics in Streamlit."""
    mock_plt, mock_fig, mock_ax, mock_sns_heatmap = mock_plt_elements

    y_true = pd.Series([0, 1, 0, 1])
    y_proba = np.array([0.1, 0.9, 0.4, 0.6])
    threshold = 0.5

    with patch("app.st", mock_st_elements), \
         patch("app.plt", mock_plt), \
         patch("app.sns.heatmap", mock_sns_heatmap):
        display_confusion_matrix_and_metrics(y_true, y_proba, threshold)

    mock_st_elements.subheader.assert_called_once()
    mock_st_elements.columns.assert_called_once()
    mock_plt.subplots.assert_called_once()
    mock_sns_heatmap.assert_called_once()
    mock_st_elements.pyplot.assert_called_once()
    assert mock_st_elements.write.call_count >= 4  # For threshold, accuracy, precision, recall, f1

@pytest.fixture
def mock_shap_elements() -> Tuple[Any, ...]:
    """Provide mock SHAP and matplotlib elements for SHAP report generation."""
    mock_datetime = MagicMock()
    mock_datetime.now.return_value.strftime.return_value = "2025-11-10 10:00:00"
    mock_b64encode = MagicMock()
    mock_b64encode.return_value = b"mock_base64_image_string_bytes" # Return bytes object
    mock_plt_close = MagicMock()
    mock_plt_savefig = MagicMock()
    mock_shap_waterfall_plot = MagicMock()
    # mock_plt_subplots = MagicMock() # Removed
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_ax.axvline = MagicMock()
    mock_ax.get_legend_handles_labels.return_value = ([], [])
    mock_ax.legend = MagicMock()
    mock_ax.get_xlim.return_value = (-2, 2)
    mock_ax.set_xticks.return_value = None
    mock_ax.set_xticklabels.return_value = None
    mock_ax.set_xlabel.return_value = None
    mock_ax.set_title.return_value = None
    mock_ax.grid.return_value = None
    mock_ax.set_yticks.return_value = None
    mock_ax.set_yticklabels.return_value = None
    # mock_plt_subplots.return_value = (mock_fig, mock_ax) # Removed
    mock_tight_layout = MagicMock()
    mock_tight_layout.return_value = None
    mock_shap_explainer = MagicMock()
    mock_shap_explainer.expected_value = 0.5 # Set an expected_value for the explainer
    mock_shap_explainer.shap_values.return_value = np.array([[0.1, 0.2, 0.3]]) # Return dummy SHAP values
    mock_model_pipeline = MagicMock()

    # Create a real shap.Explanation object with dummy data
    # The actual values, base_values, and data don't matter for the type check
    # but they need to be present.
    real_explanation_instance = Explanation(
        values=np.array([0.1, 0.2, 0.3]),
        base_values=0.0,
        data=np.array([0.5, 0.6, 0.7]),
        feature_names=["f1", "f2", "f3"],
    )

    return (
        mock_datetime,
        mock_b64encode,
        mock_plt_close,
        mock_plt_savefig,
        mock_shap_waterfall_plot,
        # mock_plt_subplots, # Removed
        mock_fig,
        mock_ax,
        mock_tight_layout,
        real_explanation_instance, # Return the real Explanation instance
        mock_shap_explainer,
        mock_model_pipeline,
    )

def test_generate_shap_html_report(
    mock_shap_elements: tuple[Any, ...],
) -> None:
    """Test the generation of SHAP HTML report."""
    (
        mock_datetime,
        mock_b64encode,
        mock_plt_close,
        mock_plt_savefig,
        mock_shap_waterfall_plot,
        # mock_plt_subplots, # Removed
        mock_fig,
        mock_ax,
        mock_tight_layout,
        real_explanation_instance, # Unpack the real Explanation instance
        mock_shap_explainer,
        mock_model_pipeline,
    ) = mock_shap_elements

    mock_plt_gcf = MagicMock(return_value=mock_fig) # Mock plt.gcf to return mock_fig
    mock_plt_gca = MagicMock(return_value=mock_ax) # Mock plt.gca to return mock_ax

    with (
        patch("app.plt.subplots", MagicMock()), # Patch with a generic mock, as it's not called in generate_shap_html_report
        patch("shap.plots.waterfall", mock_shap_waterfall_plot), # Patch shap.plots.waterfall directly
        patch("app.plt.savefig", mock_plt_savefig),
        patch("app.plt.close", mock_plt_close),
        patch("app.base64.b64encode", mock_b64encode),
        patch("app.datetime", mock_datetime),
        patch("app.plt.tight_layout", mock_tight_layout),
        patch("app.shap.Explanation", return_value=real_explanation_instance),
        patch("app.shap.LinearExplainer", mock_shap_explainer),
        patch("app.plt.gcf", mock_plt_gcf), # Patch plt.gcf
        patch("app.plt.gca", mock_plt_gca), # Patch plt.gca
    ):
        employee_data_with_predictions = pd.DataFrame(
            {
                "id_employee": [101],
                "Attrition_Risk_Percentage": [0.75],
                "Risk_Attrition": ["High"],
                "Prediction": ["Leave"],
            }
        )
        x_transformed_for_shap = np.array([[0.1, 0.2, 0.3]])
        all_features = ["feat1", "feat2", "feat3"]

        html_report = generate_shap_html_report(
            employee_data_with_predictions,
            x_transformed_for_shap,
            mock_shap_explainer,
            all_features,
        )

        assert "<h1>Employee Attrition SHAP Explanation Report</h1>" in html_report
        assert "Employee ID: 101" in html_report
        expected_risk_html = (
            'Predicted Attrition Risk: <span class="risk-label risk-high">High</span> (75.0%)'
        )
        # Remove all whitespace and newlines for robust comparison
        assert "".join(expected_risk_html.split()) in "".join(html_report.split())
        assert "mock_base64_image_string_bytes" in html_report
        mock_b64encode.assert_called_once()

        mock_shap_waterfall_plot.assert_called_once_with(
            real_explanation_instance,
            max_display=10,
            show=False,
        )
        mock_plt_gcf.assert_called_once() # Assert plt.gcf was called
        # mock_plt_subplots.assert_called_once() # Removed
        mock_fig.savefig.assert_called_once()
        mock_plt_close.assert_called_once()
        mock_ax.set_xticks.assert_called_once()
        mock_ax.set_xticklabels.assert_called_once()
        mock_ax.set_xlabel.assert_called_once()
        mock_ax.set_title.assert_called_once()
        assert mock_ax.axvline.call_count == 2  # For baseline and predicted probability lines
        mock_ax.legend.assert_called_once()


# --- Additional Error Scenarios for Cleaning Functions ---
def test_clean_extrait_eval_empty_df() -> None:
    """Test _clean_extrait_eval with an empty DataFrame."""
    df = pd.DataFrame()
    cleaned_df = _clean_extrait_eval(df.copy())
    assert cleaned_df.empty


def test_clean_extrait_sirh_empty_df() -> None:
    """Test _clean_extrait_sirh with an empty DataFrame."""
    df = pd.DataFrame()
    cleaned_df = _clean_extrait_sirh(df.copy())
    assert cleaned_df.empty


def test_clean_extrait_sondage_empty_df() -> None:
    """Test _clean_extrait_sondage with an empty DataFrame."""
    df = pd.DataFrame()
    cleaned_df = _clean_extrait_sondage(df.copy())
    assert cleaned_df.empty


def test_clean_extrait_eval_non_numeric_id_employee() -> None:
    """Test _clean_extrait_eval with non-numeric 'eval_number'."""
    df = pd.DataFrame({"eval_number": ["E_abc"]})
    cleaned_df = _clean_extrait_eval(df.copy())
    assert pd.isna(cleaned_df["id_employee"].iloc[0])


def test_clean_extrait_sondage_non_numeric_id_employee() -> None:
    """Test _clean_extrait_sondage with non-numeric 'code_sondage'."""
    df = pd.DataFrame({"code_sondage": ["abc"]})
    cleaned_df = _clean_extrait_sondage(df.copy())
    assert pd.isna(cleaned_df["id_employee"].iloc[0])
