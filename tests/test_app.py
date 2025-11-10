import pandas as pd
import numpy as np
import pytest
import sys
import os
from unittest.mock import patch, MagicMock

# Add the project root to the sys.path to allow importing app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import (
    _clean_extrait_eval,
    _clean_extrait_sirh,
    _clean_extrait_sondage,
    load_and_merge_data,
    clean_and_engineer_features,
    get_risk_category,
    RISK_THRESHOLDS,
    load_model_and_data,
    display_confusion_matrix_and_metrics,
    generate_shap_html_report,
)


# Mock data for testing
@pytest.fixture
def sample_eval_df():
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
def sample_sirh_df():
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
def sample_sondage_df():
    return pd.DataFrame(
        {
            "code_sondage": [1, 2, 4, 5],
            "satisfaction_employee_nature_travail": [4, 3, 5, 2],
            "satisfaction_employee_equipe": [5, 4, 3, 3],
            "satisfaction_employee_equilibre_pro_perso": [3, 2, 4, 5],
        }
    )


@pytest.fixture
def merged_df_for_feature_engineering():
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
def mock_model_pipeline():
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
def test_clean_extrait_eval_salary_augmentation(sample_eval_df):
    df = _clean_extrait_eval(sample_eval_df.copy())
    assert df["augmentation_salaire_precedente"].iloc[0] == 0.10
    assert df["augmentation_salaire_precedente"].iloc[1] == 0.055
    assert df["augmentation_salaire_precedente"].iloc[2] == 0.00
    assert pd.isna(df["augmentation_salaire_precedente"].iloc[3])


def test_clean_extrait_eval_heures_supplementaires(sample_eval_df):
    df = _clean_extrait_eval(sample_eval_df.copy())
    assert df["heures_supplementaires"].iloc[0] == 1
    assert df["heures_supplementaires"].iloc[1] == 0
    assert df["heures_supplementaires"].iloc[2] == 1
    assert df["heures_supplementaires"].iloc[3] == 0


def test_clean_extrait_eval_id_employee(sample_eval_df):
    df = _clean_extrait_eval(sample_eval_df.copy())
    assert "eval_number" not in df.columns
    assert df["id_employee"].iloc[0] == 1
    assert df["id_employee"].iloc[1] == 2


def test_clean_extrait_eval_missing_columns():
    df = pd.DataFrame({"col_a": [1, 2]})
    cleaned_df = _clean_extrait_eval(df.copy())
    assert "augmentation_salaire_precedente" not in cleaned_df.columns
    assert "heures_supplementaires" not in cleaned_df.columns
    assert "id_employee" not in cleaned_df.columns


# --- Test _clean_extrait_sirh ---
def test_clean_extrait_sirh_genre(sample_sirh_df):
    df = _clean_extrait_sirh(sample_sirh_df.copy())
    assert df["genre"].iloc[0] == 1
    assert df["genre"].iloc[1] == 0
    assert df["genre"].iloc[2] == 1
    assert df["genre"].iloc[3] == 0


def test_clean_extrait_sirh_drop_columns(sample_sirh_df):
    df = _clean_extrait_sirh(sample_sirh_df.copy())
    assert "nombre_heures_travailless" not in df.columns


def test_clean_extrait_sirh_missing_columns():
    df = pd.DataFrame({"col_a": [1, 2]})
    cleaned_df = _clean_extrait_sirh(df.copy())
    assert "genre" not in cleaned_df.columns


# --- Test _clean_extrait_sondage ---
def test_clean_extrait_sondage_id_employee(sample_sondage_df):
    df = _clean_extrait_sondage(sample_sondage_df.copy())
    assert "code_sondage" not in df.columns
    assert df["id_employee"].iloc[0] == 1
    assert df["id_employee"].iloc[2] == 4


def test_clean_extrait_sondage_missing_columns():
    df = pd.DataFrame({"col_a": [1, 2]})
    cleaned_df = _clean_extrait_sondage(df.copy())
    assert "id_employee" not in cleaned_df.columns


# --- Test load_and_merge_data ---
def test_load_and_merge_data_merge_logic(
    sample_eval_df, sample_sirh_df, sample_sondage_df
):
    merged_df = load_and_merge_data(
        sample_eval_df.copy(), sample_sirh_df.copy(), sample_sondage_df.copy()
    )
    assert len(merged_df) == 5  # id_employee 1, 2, 3, 4, 5 (outer merge)
    assert "augmentation_salaire_precedente" in merged_df.columns
    assert "genre" in merged_df.columns
    assert "satisfaction_employee_nature_travail" in merged_df.columns
    assert merged_df.loc[merged_df["id_employee"] == 1, "genre"].iloc[0] == 1
    assert (
        merged_df.loc[
            merged_df["id_employee"] == 4, "satisfaction_employee_equipe"
        ].iloc[0]
        == 3
    )
    assert pd.isna(
        merged_df.loc[
            merged_df["id_employee"] == 3, "satisfaction_employee_equipe"
        ].iloc[0]
    )


def test_load_and_merge_data_empty_dfs():
    eval_df = pd.DataFrame()
    sirh_df = pd.DataFrame()
    sond_df = pd.DataFrame()
    merged_df = load_and_merge_data(eval_df, sirh_df, sond_df)
    assert merged_df.empty


# --- Test clean_and_engineer_features ---
def test_clean_and_engineer_features_improvement_evaluation(
    merged_df_for_feature_engineering,
):
    df = clean_and_engineer_features(merged_df_for_feature_engineering.copy())
    assert df["improvement_evaluation"].iloc[0] == 1
    assert df["improvement_evaluation"].iloc[1] == 0
    assert df["improvement_evaluation"].iloc[2] == 1
    assert df["improvement_evaluation"].iloc[3] == -1


def test_clean_and_engineer_features_total_satisfaction(
    merged_df_for_feature_engineering,
):
    df = clean_and_engineer_features(merged_df_for_feature_engineering.copy())
    assert df["total_satisfaction"].iloc[0] == 4 * 5 * 3
    assert df["total_satisfaction"].iloc[1] == 3 * 4 * 2


def test_clean_and_engineer_features_work_mobility(merged_df_for_feature_engineering):
    df = clean_and_engineer_features(merged_df_for_feature_engineering.copy())
    assert df["work_mobility"].iloc[0] == 2 / 5
    assert df["work_mobility"].iloc[1] == 1 / 3
    assert df["work_mobility"].iloc[3] == 0  # Division by zero handled


def test_clean_and_engineer_features_missing_columns():
    df = pd.DataFrame({"id_employee": [1]})
    cleaned_df = clean_and_engineer_features(df.copy())
    assert "improvement_evaluation" not in cleaned_df.columns
    assert "total_satisfaction" not in cleaned_df.columns
    assert "work_mobility" not in cleaned_df.columns


# --- Test get_risk_category ---
def test_get_risk_category_low():
    assert get_risk_category(0.1, 0.5) == "Low"
    assert get_risk_category(0.4, 0.5) == "Low"  # 0.4 < 0.45 (threshold - buffer)
    assert get_risk_category(0.29, 0.5) == "Low"  # 0.29 < 0.45 (threshold - buffer)


def test_get_risk_category_medium():
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


def test_get_risk_category_high():
    assert get_risk_category(0.55, 0.5) == "High"  # 0.55 is threshold + buffer
    assert get_risk_category(0.7, 0.5) == "High"
    assert get_risk_category(0.9, 0.5) == "High"
    assert get_risk_category(1.0, 0.5) == "High"  # Edge case for upper bound


def test_get_risk_category_edge_cases():
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
def test_load_model_and_data(mock_read_parquet, mock_joblib_load, mock_model_pipeline):
    # Mock the return values for the internal calls within load_model_and_data
    mock_joblib_load.return_value = mock_model_pipeline
    mock_read_parquet.side_effect = [
        pd.DataFrame({"col1": [1]}),  # X_train
        pd.Series([0]),  # y_train
        pd.DataFrame({"col1": [2]}),  # X_test
        pd.Series([1]),  # y_test
    ]

    # Call the actual function
    model, X_train, y_train, X_test, y_test = load_model_and_data()

    # Assert that the internal calls were made
    mock_joblib_load.assert_called_once_with("outputs/employee_attrition_pipeline.pkl")
    assert mock_read_parquet.call_count == 4
    assert isinstance(model, MagicMock)
    assert not X_train.empty
    assert y_train is not None
    assert not X_test.empty
    assert y_test is not None


# Test display_confusion_matrix_and_metrics
@patch("app.st.subheader")
@patch("app.st.columns")
@patch("app.st.write")
@patch("app.plt.subplots")
@patch("app.sns.heatmap")
@patch("app.plt.gca")
@patch("app.plt.yticks")
@patch("app.plt.xticks")
@patch("app.plt.tight_layout")
@patch("app.st.pyplot")
def test_display_confusion_matrix_and_metrics(
    mock_st_pyplot,
    mock_tight_layout,
    mock_xticks,
    mock_yticks,
    mock_gca,
    mock_sns_heatmap,
    mock_subplots,
    mock_st_write,
    mock_st_columns,
    mock_st_subheader,
):

    mock_st_columns.return_value = [MagicMock(), MagicMock()]
    mock_subplots.return_value = (MagicMock(), MagicMock())  # fig, ax
    mock_gca.return_value = MagicMock()  # ax

    y_true = pd.Series([0, 1, 0, 1])
    y_proba = np.array([0.1, 0.9, 0.4, 0.6])
    threshold = 0.5

    display_confusion_matrix_and_metrics(y_true, y_proba, threshold)

    mock_st_subheader.assert_called_once()
    mock_st_columns.assert_called_once()
    mock_subplots.assert_called_once()
    mock_sns_heatmap.assert_called_once()
    mock_st_pyplot.assert_called_once()
    assert (
        mock_st_write.call_count >= 4
    )  # For threshold, accuracy, precision, recall, f1

    # Test generate_shap_html_report
    @patch("app.plt.subplots")  # Patch plt.subplots instead of plt.figure and plt.gca
    @patch("app.shap.waterfall_plot")
    @patch("app.plt.savefig")
    @patch("app.plt.close")
    @patch("app.base64.b64encode")
    @patch("app.datetime")
    def test_generate_shap_html_report(
        mock_datetime,
        mock_b64encode,
        mock_plt_close,
        mock_plt_savefig,
        mock_shap_waterfall_plot,
        mock_plt_subplots,
        mock_model_pipeline,
    ):  # Updated arguments

        # Mock datetime.now().strftime
        mock_datetime.now.return_value.strftime.return_value = "2025-11-10 10:00:00"

        # Mock base64 encoding
        mock_b64encode.return_value.decode.return_value = "mock_base64_image_string"

        # Mock matplotlib figure and axes returned by plt.subplots
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_ax.axvline = MagicMock()  # Explicitly mock axvline on mock_ax
        mock_ax.get_legend_handles_labels.return_value = (
            [],
            [],
        )  # Mock to prevent ValueError
        mock_ax.legend = MagicMock()  # Mock the legend call itself
        mock_ax.get_xlim.return_value = (-2, 2)  # Example log-odds limits
        mock_ax.set_xticks.return_value = None
        mock_ax.set_xticklabels.return_value = None
        mock_ax.set_xlabel.return_value = None
        mock_ax.set_title.return_value = None
        mock_ax.grid.return_value = None  # Mock grid call
        mock_ax.set_yticks.return_value = None  # Mock set_yticks call
        mock_ax.set_yticklabels.return_value = None  # Mock set_yticklabels call
        mock_plt_subplots.return_value = (
            mock_fig,
            mock_ax,
        )  # plt.subplots returns (fig, ax)

        # Mock plt.tight_layout
        with patch("app.plt.tight_layout") as mock_tight_layout:
            mock_tight_layout.return_value = None

            # Mock explainer and shap_values
            mock_explainer = MagicMock()
            mock_explainer.expected_value = (
                0.5  # Example log-odds expected value (scalar)
            )
            mock_shap_values = [np.array([0.1, -0.2, 0.3])]  # Example SHAP values
            mock_explainer.shap_values.return_value = (
                mock_shap_values  # Mock shap_values method
            )

            # Let's ensure the mock_explainer.expected_value is a simple float
            mock_explainer.expected_value = 0.5  # A scalar float

            # Mock the Explanation object creation
            with patch("app.shap.Explanation") as mock_shap_explanation:
                mock_shap_explanation.return_value = MagicMock()

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

                html_report = generate_shap_html_report(
                    employee_data_with_predictions,
                    X_transformed_for_shap,
                    mock_explainer,
                    all_features,
                )

                assert (
                    "<h1>Employee Attrition SHAP Explanation Report</h1>" in html_report
                )
                assert "Employee ID: 101" in html_report
                assert (
                    'Predicted Attrition Risk: <span class="risk-label risk-high">High</span> (75.0%)'
                    in html_report
                )
                assert "mock_base64_image_string" in html_report

                mock_shap_waterfall_plot.assert_not_called()  # We are no longer calling shap.waterfall_plot
                mock_plt_subplots.assert_called_once()  # Ensure our custom plot function is called
                mock_fig.savefig.assert_called_once()
                mock_plt_close.assert_called_once()
                mock_ax.set_xticks.assert_called_once()
                mock_ax.set_xticklabels.assert_called_once()
                mock_ax.set_xlabel.assert_called_once()
                mock_ax.set_title.assert_called_once()
                mock_ax.axvline.call_count == 2  # For baseline and predicted prob
                mock_ax.legend.assert_called_once()


# --- Additional Error Scenarios for Cleaning Functions ---
def test_clean_extrait_eval_empty_df():
    df = pd.DataFrame()
    cleaned_df = _clean_extrait_eval(df.copy())
    assert cleaned_df.empty


def test_clean_extrait_sirh_empty_df():
    df = pd.DataFrame()
    cleaned_df = _clean_extrait_sirh(df.copy())
    assert cleaned_df.empty


def test_clean_extrait_sondage_empty_df():
    df = pd.DataFrame()
    cleaned_df = _clean_extrait_sondage(df.copy())
    assert cleaned_df.empty


def test_clean_extrait_eval_non_numeric_id_employee():
    df = pd.DataFrame({"eval_number": ["E_abc"]})
    cleaned_df = _clean_extrait_eval(df.copy())
    assert pd.isna(cleaned_df["id_employee"].iloc[0])


def test_clean_extrait_sondage_non_numeric_id_employee():
    df = pd.DataFrame({"code_sondage": ["abc"]})
    cleaned_df = _clean_extrait_sondage(df.copy())
    assert pd.isna(cleaned_df["id_employee"].iloc[0])
