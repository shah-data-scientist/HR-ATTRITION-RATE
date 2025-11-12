"""Tests for core data processing modules."""

import pandas as pd
import pytest

from core.data_processing import (
    clean_and_engineer_features,
    clean_raw_input,
    engineer_features,
)
from core.preprocess import enforce_schema, validate_input_ranges
from core.schema import EmployeeInputSchema, BatchPredictionInput
from core.validation import (
    ALL_FEATURE_COLS,
    CATEGORICAL_COLS,
    ENGINEERED_FEATURES,
    NUMERIC_COLS,
)


class TestDataCleaning:
    """Test raw data cleaning functions."""

    def test_clean_genre_homme(self, sample_raw_employee_df):
        """Test genre cleaning: 'Homme' -> 1."""
        df = sample_raw_employee_df.copy()
        cleaned = clean_raw_input(df)
        assert cleaned["genre"].iloc[0] == 1

    def test_clean_genre_femme(self):
        """Test genre cleaning: 'Femme' -> 0."""
        df = pd.DataFrame([{"genre": "Femme"}])
        cleaned = clean_raw_input(df)
        assert cleaned["genre"].iloc[0] == 0

    def test_clean_heures_supplementaires(self, sample_raw_employee_df):
        """Test overtime cleaning: 'Non' -> 0."""
        df = sample_raw_employee_df.copy()
        cleaned = clean_raw_input(df)
        assert cleaned["heures_supplementaires"].iloc[0] == 0

    def test_clean_heures_supplementaires_oui(self):
        """Test overtime cleaning: 'Oui' -> 1."""
        df = pd.DataFrame([{"heure_supplementaires": "Oui"}])
        cleaned = clean_raw_input(df)
        assert cleaned["heures_supplementaires"].iloc[0] == 1

    def test_extract_id_from_eval_number(self):
        """Test extracting id_employee from eval_number."""
        df = pd.DataFrame([{"eval_number": "E_12345"}])
        cleaned = clean_raw_input(df)
        assert cleaned["id_employee"].iloc[0] == 12345
        assert "eval_number" not in cleaned.columns

    def test_drop_unnecessary_columns(self):
        """Test dropping code_sondage and nombre_heures_travailless."""
        df = pd.DataFrame(
            [
                {
                    "id_employee": 1,
                    "code_sondage": "80",
                    "nombre_heures_travailless": 190,
                }
            ]
        )
        cleaned = clean_raw_input(df)
        assert "code_sondage" not in cleaned.columns
        assert "nombre_heures_travailless" not in cleaned.columns


class TestFeatureEngineering:
    """Test feature engineering functions."""

    def test_improvement_evaluation(self):
        """Test improvement_evaluation = current - previous."""
        df = pd.DataFrame(
            [{"note_evaluation_actuelle": 4, "note_evaluation_precedente": 3}]
        )
        engineered = engineer_features(df)
        assert engineered["improvement_evaluation"].iloc[0] == 1

    def test_total_satisfaction(self):
        """Test total_satisfaction = product of 3 satisfaction scores."""
        df = pd.DataFrame(
            [
                {
                    "satisfaction_employee_nature_travail": 4,
                    "satisfaction_employee_equipe": 3,
                    "satisfaction_employee_equilibre_pro_perso": 3,
                }
            ]
        )
        engineered = engineer_features(df)
        assert engineered["total_satisfaction"].iloc[0] == 36  # 4 * 3 * 3

    def test_work_mobility(self):
        """Test work_mobility = years in role / years in company."""
        df = pd.DataFrame(
            [{"annees_dans_le_poste_actuel": 3, "annees_dans_l_entreprise": 5}]
        )
        engineered = engineer_features(df)
        assert engineered["work_mobility"].iloc[0] == 0.6  # 3/5

    def test_work_mobility_zero_denominator(self):
        """Test work_mobility with zero years in company returns 0."""
        df = pd.DataFrame(
            [{"annees_dans_le_poste_actuel": 3, "annees_dans_l_entreprise": 0}]
        )
        engineered = engineer_features(df)
        assert engineered["work_mobility"].iloc[0] == 0.0


class TestFullPipeline:
    """Test the complete cleaning + engineering pipeline."""

    def test_clean_and_engineer_pipeline(self, sample_raw_employee_df):
        """Test full pipeline: clean -> engineer."""
        result = clean_and_engineer_features(sample_raw_employee_df)

        # Check cleaned values
        assert result["genre"].iloc[0] == 1  # 'Homme' -> 1
        assert result["heures_supplementaires"].iloc[0] == 0  # 'Non' -> 0

        # Check engineered features exist
        assert "improvement_evaluation" in result.columns
        assert "total_satisfaction" in result.columns
        assert "work_mobility" in result.columns

        # Check engineered values
        assert result["improvement_evaluation"].iloc[0] == 1  # 4 - 3
        assert result["total_satisfaction"].iloc[0] == 36  # 4 * 3 * 3
        assert result["work_mobility"].iloc[0] == 0.6  # 3/5


class TestSchemaEnforcement:
    """Test schema enforcement and validation."""

    def test_enforce_schema_column_order(self, sample_raw_employee_df):
        """Test that enforce_schema returns columns in correct order."""
        df = clean_and_engineer_features(sample_raw_employee_df)
        enforced = enforce_schema(df, NUMERIC_COLS + CATEGORICAL_COLS)

        assert list(enforced.columns) == ALL_FEATURE_COLS

    def test_enforce_schema_adds_missing_columns(self):
        """Test that missing columns are added with defaults."""
        df = pd.DataFrame([{"id_employee": 1, "age": 30}])
        enforced = enforce_schema(df, NUMERIC_COLS + CATEGORICAL_COLS)

        assert len(enforced.columns) == len(ALL_FEATURE_COLS)

    def test_validate_input_ranges_valid(self):
        """Test validation with all values in range."""
        df = pd.DataFrame(
            [
                {
                    "age": 35,
                    "satisfaction_employee_environnement": 3,
                    "niveau_education": 3,
                }
            ]
        )
        warnings = validate_input_ranges(df)
        assert len(warnings) == 0

    def test_validate_input_ranges_out_of_range(self):
        """Test validation detects out-of-range values."""
        df = pd.DataFrame([{"age": 100}])  # Age > 70
        warnings = validate_input_ranges(df)
        assert len(warnings) > 0
        assert "age" in warnings[0].lower()


class TestPydanticSchemas:
    """Test Pydantic schema validation."""

    def test_employee_input_schema_valid(self, sample_raw_employee_data):
        """Test that valid raw data passes Pydantic validation."""
        employee = EmployeeInputSchema(**sample_raw_employee_data)
        assert employee.id_employee == 99999
        assert employee.genre == "Homme"
        assert employee.heure_supplementaires == "Non"

    def test_employee_input_schema_invalid_age(self):
        """Test that invalid age is rejected."""
        data = {
            "age": 10,  # Too young
            "genre": "Homme",
            "heure_supplementaires": "Non",
            "augementation_salaire_precedente": 5.0,
            "id_employee": 1,
            "revenu_mensuel": 5000,
            "statut_marital": "Célibataire",
            "departement": "IT",
            "poste": "Dev",
            "domaine_etude": "Info",
            "frequence_deplacement": "Rare",
            "nombre_experiences_precedentes": 1,
            "nombre_heures_travailless": 180,
            "annee_experience_totale": 5,
            "annees_dans_l_entreprise": 2,
            "annees_dans_le_poste_actuel": 1,
            "nombre_participation_pee": 0,
            "nb_formations_suivies": 1,
            "nombre_employee_sous_responsabilite": 0,
            "distance_domicile_travail": 10,
            "niveau_education": 2,
            "ayant_enfants": "Non",
            "annees_depuis_la_derniere_promotion": 1,
            "annes_sous_responsable_actuel": 1,
            "satisfaction_employee_environnement": 3,
            "note_evaluation_precedente": 3,
            "niveau_hierarchique_poste": 2,
            "satisfaction_employee_nature_travail": 3,
            "satisfaction_employee_equipe": 3,
            "satisfaction_employee_equilibre_pro_perso": 3,
            "note_evaluation_actuelle": 3,
        }
        with pytest.raises(Exception):  # Pydantic ValidationError
            EmployeeInputSchema(**data)

    def test_batch_prediction_input(self, sample_raw_employee_data):
        """Test batch prediction input schema."""
        employee = EmployeeInputSchema(**sample_raw_employee_data)
        batch = BatchPredictionInput(employees=[employee])
        assert len(batch.employees) == 1
