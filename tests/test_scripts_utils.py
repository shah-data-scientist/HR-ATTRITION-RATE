"""Tests for scripts/utils.py module.

This test file targets the data cleaning and merging utilities
used by the database initialization scripts.
"""

import pandas as pd
import numpy as np
import pytest


class TestCleanExtraitEval:
    """Test _clean_extrait_eval function."""

    def test_clean_augmentation_salaire_precedente(self):
        """Test cleaning of augmentation_salaire_precedente column."""
        from scripts.utils import _clean_extrait_eval

        df = pd.DataFrame(
            {
                "augmentation_salaire_precedente": ["5%", "10%", "15,5%"],
                "id_employee": [1, 2, 3],
            }
        )

        result = _clean_extrait_eval(df)

        assert result["augmentation_salaire_precedente"].iloc[0] == 0.05
        assert result["augmentation_salaire_precedente"].iloc[1] == 0.10
        assert result["augmentation_salaire_precedente"].iloc[2] == 0.155

    def test_clean_heures_supplementaires_oui_non(self):
        """Test cleaning of heures_supplementaires with Oui/Non values."""
        from scripts.utils import _clean_extrait_eval

        df = pd.DataFrame(
            {
                "heures_supplementaires": ["Oui", "Non", "oui", "non"],
                "id_employee": [1, 2, 3, 4],
            }
        )

        result = _clean_extrait_eval(df)

        assert result["heures_supplementaires"].iloc[0] == 1
        assert result["heures_supplementaires"].iloc[1] == 0
        assert result["heures_supplementaires"].iloc[2] == 1
        assert result["heures_supplementaires"].iloc[3] == 0

    def test_clean_heures_supplementaires_boolean(self):
        """Test cleaning of heures_supplementaires with boolean values."""
        from scripts.utils import _clean_extrait_eval

        df = pd.DataFrame(
            {"heures_supplementaires": [True, False, True], "id_employee": [1, 2, 3]}
        )

        result = _clean_extrait_eval(df)

        assert result["heures_supplementaires"].iloc[0] == 1
        assert result["heures_supplementaires"].iloc[1] == 0

    def test_clean_heure_supplementaires_variant_spelling(self):
        """Test renaming of heure_supplementaires variant."""
        from scripts.utils import _clean_extrait_eval

        df = pd.DataFrame(
            {"heure_supplementaires": ["Oui", "Non"], "id_employee": [1, 2]}
        )

        result = _clean_extrait_eval(df)

        assert "heures_supplementaires" in result.columns
        assert "heure_supplementaires" not in result.columns

    def test_clean_eval_number_conversion(self):
        """Test conversion of eval_number to id_employee."""
        from scripts.utils import _clean_extrait_eval

        df = pd.DataFrame({"eval_number": ["E_100", "E_200", "E_300"]})

        result = _clean_extrait_eval(df)

        assert "id_employee" in result.columns
        assert "eval_number" not in result.columns
        assert result["id_employee"].iloc[0] == 100
        assert result["id_employee"].iloc[1] == 200


class TestCleanExtraitSirh:
    """Test _clean_extrait_sirh function."""

    def test_clean_genre_male(self):
        """Test cleaning of genre column with male values."""
        from scripts.utils import _clean_extrait_sirh

        df = pd.DataFrame({"genre": ["M", "m"], "id_employee": [1, 2]})

        result = _clean_extrait_sirh(df)

        assert result["genre"].iloc[0] == 1
        assert result["genre"].iloc[1] == 1

    def test_clean_genre_female(self):
        """Test cleaning of genre column with female values."""
        from scripts.utils import _clean_extrait_sirh

        df = pd.DataFrame({"genre": ["F", "f"], "id_employee": [1, 2]})

        result = _clean_extrait_sirh(df)

        assert result["genre"].iloc[0] == 0
        assert result["genre"].iloc[1] == 0

    def test_drop_unnecessary_columns(self):
        """Test dropping of unnecessary columns."""
        from scripts.utils import _clean_extrait_sirh

        df = pd.DataFrame(
            {
                "genre": ["M", "F"],
                "nombre_heures_travailless": [40, 35],
                "...": ["test", "data"],
                "id_employee": [1, 2],
            }
        )

        result = _clean_extrait_sirh(df)

        assert "nombre_heures_travailless" not in result.columns
        assert "..." not in result.columns


class TestCleanExtraitSondage:
    """Test _clean_extrait_sondage function."""

    def test_rename_code_sondage(self):
        """Test renaming of code_sondage to id_employee."""
        from scripts.utils import _clean_extrait_sondage

        df = pd.DataFrame({"code_sondage": [100, 200, 300], "satisfaction": [3, 4, 2]})

        result = _clean_extrait_sondage(df)

        assert "id_employee" in result.columns
        assert "code_sondage" not in result.columns
        assert result["id_employee"].iloc[0] == 100

    def test_convert_id_employee_to_numeric(self):
        """Test conversion of id_employee to numeric."""
        from scripts.utils import _clean_extrait_sondage

        df = pd.DataFrame(
            {"id_employee": ["100", "200", "300"], "satisfaction": [3, 4, 2]}
        )

        result = _clean_extrait_sondage(df)

        assert result["id_employee"].dtype == pd.Int64Dtype()
        assert result["id_employee"].iloc[0] == 100


class TestLoadAndMergeData:
    """Test load_and_merge_data function."""

    def test_basic_merge(self):
        """Test basic merging of three dataframes."""
        from scripts.utils import load_and_merge_data

        eval_df = pd.DataFrame(
            {
                "eval_number": ["E_100", "E_200"],
                "note_evaluation_actuelle": [3.5, 4.0],
                "augmentation_salaire_precedente": ["5%", "10%"],
            }
        )

        sirh_df = pd.DataFrame(
            {"id_employee": [100, 200], "genre": ["M", "F"], "age": [30, 35]}
        )

        sond_df = pd.DataFrame({"code_sondage": [100, 200], "satisfaction": [3, 4]})

        result = load_and_merge_data(eval_df, sirh_df, sond_df)

        assert "id_employee" in result.columns
        assert len(result) == 2
        assert result["genre"].iloc[0] == 1  # M -> 1
        assert result["genre"].iloc[1] == 0  # F -> 0

    def test_merge_with_ayant_enfants(self):
        """Test merging with ayant_enfants column."""
        from scripts.utils import load_and_merge_data

        eval_df = pd.DataFrame(
            {"id_employee": [100, 200], "note_evaluation_actuelle": [3.5, 4.0]}
        )

        sirh_df = pd.DataFrame({"id_employee": [100, 200], "ayant_enfants": ["Y", "N"]})

        sond_df = pd.DataFrame({"id_employee": [100, 200], "satisfaction": [3, 4]})

        result = load_and_merge_data(eval_df, sirh_df, sond_df)

        assert result["ayant_enfants"].iloc[0] == 1  # Y -> 1
        assert result["ayant_enfants"].iloc[1] == 0  # N -> 0

    def test_merge_with_augementation_typo(self):
        """Test handling of augementation_salaire_precedente (typo variant)."""
        from scripts.utils import load_and_merge_data

        eval_df = pd.DataFrame(
            {
                "id_employee": [100, 200],
                "augementation_salaire_precedente": ["5%", "10%"],
            }
        )

        sirh_df = pd.DataFrame({"id_employee": [100, 200], "age": [30, 35]})

        sond_df = pd.DataFrame({"id_employee": [100, 200], "satisfaction": [3, 4]})

        result = load_and_merge_data(eval_df, sirh_df, sond_df)

        assert "augementation_salaire_precedente" in result.columns
        assert result["augementation_salaire_precedente"].iloc[0] == 0.05

    def test_drop_duplicates(self):
        """Test that duplicates are dropped."""
        from scripts.utils import load_and_merge_data

        eval_df = pd.DataFrame(
            {"id_employee": [100, 100], "note_evaluation_actuelle": [3.5, 3.5]}
        )

        sirh_df = pd.DataFrame({"id_employee": [100, 100], "age": [30, 30]})

        sond_df = pd.DataFrame({"id_employee": [100, 100], "satisfaction": [3, 3]})

        result = load_and_merge_data(eval_df, sirh_df, sond_df)

        # Should have only 1 row after dropping duplicates
        assert len(result) == 1


class TestCleanAndEngineerFeatures:
    """Test clean_and_engineer_features function."""

    def test_improvement_evaluation(self):
        """Test calculation of improvement_evaluation feature."""
        from scripts.utils import clean_and_engineer_features

        df = pd.DataFrame(
            {
                "id_employee": [100, 200],
                "note_evaluation_actuelle": [4.0, 3.5],
                "note_evaluation_precedente": [3.5, 4.0],
            }
        )

        result = clean_and_engineer_features(df)

        assert "improvement_evaluation" in result.columns
        assert result["improvement_evaluation"].iloc[0] == 0.5
        assert result["improvement_evaluation"].iloc[1] == -0.5

    def test_total_satisfaction(self):
        """Test calculation of total_satisfaction feature."""
        from scripts.utils import clean_and_engineer_features

        df = pd.DataFrame(
            {
                "id_employee": [100, 200],
                "satisfaction_employee_nature_travail": [3, 4],
                "satisfaction_employee_equipe": [2, 3],
                "satisfaction_employee_equilibre_pro_perso": [4, 2],
            }
        )

        result = clean_and_engineer_features(df)

        assert "total_satisfaction" in result.columns
        assert result["total_satisfaction"].iloc[0] == 24  # 3 * 2 * 4
        assert result["total_satisfaction"].iloc[1] == 24  # 4 * 3 * 2

    def test_work_mobility(self):
        """Test calculation of work_mobility feature."""
        from scripts.utils import clean_and_engineer_features

        df = pd.DataFrame(
            {
                "id_employee": [100, 200, 300],
                "annees_dans_le_poste_actuel": [2, 5, 0],
                "annees_dans_l_entreprise": [4, 10, 0],
            }
        )

        result = clean_and_engineer_features(df)

        assert "work_mobility" in result.columns
        assert result["work_mobility"].iloc[0] == 0.5  # 2 / 4
        assert result["work_mobility"].iloc[1] == 0.5  # 5 / 10
        assert result["work_mobility"].iloc[2] == 0.0  # 0 / 0 -> fillna(0)

    def test_no_features_to_engineer(self):
        """Test when DataFrame doesn't have columns for feature engineering."""
        from scripts.utils import clean_and_engineer_features

        df = pd.DataFrame({"id_employee": [100, 200], "age": [30, 35]})

        result = clean_and_engineer_features(df)

        # Should return copy without errors
        assert len(result) == 2
        assert "improvement_evaluation" not in result.columns


class TestGetExpectedColumns:
    """Test get_expected_columns function."""

    def test_get_expected_columns(self):
        """Test getting expected columns from pipeline."""
        from scripts.utils import get_expected_columns
        from unittest.mock import MagicMock

        # Mock pipeline
        mock_preprocessor = MagicMock()
        mock_preprocessor.feature_names_in_ = ["age", "genre", "revenu_mensuel"]

        mock_pipeline = MagicMock()
        mock_pipeline.named_steps = {"preprocessor": mock_preprocessor}

        columns = get_expected_columns(mock_pipeline)

        assert columns == ["age", "genre", "revenu_mensuel"]
        assert isinstance(columns, list)
