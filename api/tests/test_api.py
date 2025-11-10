import pytest
from httpx import AsyncClient
from unittest.mock import patch, MagicMock  # Import patch and MagicMock

from fastapi.testclient import TestClient  # Import TestClient

import sys
import os

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from api.app.main import (
    app,
    model,
    expected_model_columns,
    get_expected_columns_from_pipeline,
)
from api.app.schemas import EmployeeFeatures, BatchPredictionInput, PredictionOutput
import pandas as pd
import numpy as np


@pytest.fixture  # No longer async
def client():
    # Define the expected columns for the mock
    mock_expected_columns = [
        "id_employee",
        "age",
        "genre",
        "revenu_mensuel",
        "statut_marital",
        "departement",
        "poste",
        "nombre_experiences_precedentes",
        "annee_experience_totale",
        "annees_dans_l_entreprise",
        "annees_dans_le_poste_actuel",
        "nombre_participation_pee",
        "nb_formations_suivies",
        "nombre_employee_sous_responsabilite",
        "distance_domicile_travail",
        "niveau_education",
        "domaine_etude",
        "ayant_enfants",
        "frequence_deplacement",
        "annees_depuis_la_derniere_promotion",
        "annes_sous_responsable_actuel",
        "satisfaction_employee_environnement",
        "note_evaluation_precedente",
        "niveau_hierarchique_poste",
        "satisfaction_employee_nature_travail",
        "satisfaction_employee_equipe",
        "satisfaction_employee_equilibre_pro_perso",
        "note_evaluation_actuelle",
        "heures_supplementaires",
        "augementation_salaire_precedente",
        "improvement_evaluation",
        "total_satisfaction",
        "work_mobility",
    ]

    # Create a mock preprocessor for the pipeline
    mock_preprocessor = MagicMock()
    mock_preprocessor.feature_names_in_ = mock_expected_columns

    # Create a mock model for the pipeline
    mock_logreg_model = MagicMock()
    mock_logreg_model.predict_proba.return_value = np.array([[0.2, 0.8]])

    # Create a mock pipeline
    mock_pipeline = MagicMock()
    mock_pipeline.named_steps = {
        "preprocessor": mock_preprocessor,
        "model": mock_logreg_model,
    }
    mock_pipeline.predict_proba.return_value = np.array(
        [[0.2, 0.8], [0.3, 0.7]]
    )  # For multiple predictions

    # Patch joblib.load to return our mock pipeline
    with (
        patch("joblib.load", return_value=mock_pipeline),
        patch("api.app.main.expected_model_columns", new=mock_expected_columns),
    ):
        with TestClient(app) as client:
            yield client


# --- Test Endpoints ---


def test_read_root(client: TestClient):  # Changed to synchronous and TestClient
    response = client.get("/")  # Removed await
    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to the Employee Attrition Prediction API!",
        "version": app.version,
        "documentation_url": "/docs",
    }


def test_predict_attrition_success(
    client: TestClient,
):  # Changed to synchronous and TestClient
    # Example valid input data
    employee_data = EmployeeFeatures(
        id_employee=1,
        age=30,
        genre=1,
        revenu_mensuel=5000.0,
        statut_marital="Célibataire",
        departement="R&D",
        poste="Développeur",
        nombre_experiences_precedentes=2,
        annee_experience_totale=8,
        annees_dans_l_entreprise=4,
        annees_dans_le_poste_actuel=2,
        nombre_participation_pee=1,
        nb_formations_suivies=2,
        nombre_employee_sous_responsabilite=0,
        distance_domicile_travail=10,
        niveau_education=3,
        domaine_etude="Informatique",
        ayant_enfants=0,
        frequence_deplacement="Rarement",
        annees_depuis_la_derniere_promotion=1,
        annes_sous_responsable_actuel=1,
        satisfaction_employee_environnement=3,
        note_evaluation_precedente=3.0,
        niveau_hierarchique_poste=2,
        satisfaction_employee_nature_travail=3,
        satisfaction_employee_equipe=3,
        satisfaction_employee_equilibre_pro_perso=3,
        note_evaluation_actuelle=3.5,
        heures_supplementaires=0,
        augementation_salaire_precedente=0.05,
    )
    batch_input = BatchPredictionInput(employees=[employee_data])

    response = client.post(
        "/predict", json=batch_input.model_dump()
    )  # Removed await, changed .dict() to .model_dump()
    assert response.status_code == 200

    predictions = response.json()["predictions"]
    assert len(predictions) == 1
    assert predictions[0]["id_employee"] == 1
    assert (
        predictions[0]["prediction"] == "Leave"
    )  # Based on mock_model_and_columns returning 0.8 prob
    assert abs(predictions[0]["probability"] - 0.8) < 0.001
    assert predictions[0]["risk_category"] == "High"  # 0.8 > 0.5 + 0.05 (buffer)


def test_predict_attrition_multiple_employees(
    client: TestClient,
):  # Changed to synchronous and TestClient
    employee_data_1 = EmployeeFeatures(
        id_employee=1,
        age=30,
        genre=1,
        revenu_mensuel=5000.0,
        statut_marital="Célibataire",
        departement="R&D",
        poste="Développeur",
        nombre_experiences_precedentes=2,
        annee_experience_totale=8,
        annees_dans_l_entreprise=4,
        annees_dans_le_poste_actuel=2,
        nombre_participation_pee=1,
        nb_formations_suivies=2,
        nombre_employee_sous_responsabilite=0,
        distance_domicile_travail=10,
        niveau_education=3,
        domaine_etude="Informatique",
        ayant_enfants=0,
        frequence_deplacement="Rarement",
        annees_depuis_la_derniere_promotion=1,
        annes_sous_responsable_actuel=1,
        satisfaction_employee_environnement=3,
        note_evaluation_precedente=3.0,
        niveau_hierarchique_poste=2,
        satisfaction_employee_nature_travail=3,
        satisfaction_employee_equipe=3,
        satisfaction_employee_equilibre_pro_perso=3,
        note_evaluation_actuelle=3.5,
        heures_supplementaires=0,
        augementation_salaire_precedente=0.05,
    )
    employee_data_2 = EmployeeFeatures(
        id_employee=2,
        age=45,
        genre=0,
        revenu_mensuel=7000.0,
        statut_marital="Marié",
        departement="Ventes",
        poste="Manager",
        nombre_experiences_precedentes=5,
        annee_experience_totale=20,
        annees_dans_l_entreprise=10,
        annees_dans_le_poste_actuel=5,
        nombre_participation_pee=2,
        nb_formations_suivies=3,
        nombre_employee_sous_responsabilite=5,
        distance_domicile_travail=20,
        niveau_education=4,
        domaine_etude="Commerce",
        ayant_enfants=1,
        frequence_deplacement="Fréquemment",
        annees_depuis_la_derniere_promotion=3,
        annes_sous_responsable_actuel=2,
        satisfaction_employee_environnement=4,
        note_evaluation_precedente=4.0,
        niveau_hierarchique_poste=4,
        satisfaction_employee_nature_travail=4,
        satisfaction_employee_equipe=4,
        satisfaction_employee_equilibre_pro_perso=4,
        note_evaluation_actuelle=4.5,
        heures_supplementaires=1,
        augementation_salaire_precedente=0.10,
    )
    batch_input = BatchPredictionInput(employees=[employee_data_1, employee_data_2])

    response = client.post(
        "/predict", json=batch_input.model_dump()
    )  # Removed await, changed .dict() to .model_dump()
    assert response.status_code == 200

    predictions = response.json()["predictions"]
    assert len(predictions) == 2
    assert predictions[0]["id_employee"] == 1
    assert predictions[1]["id_employee"] == 2


def test_predict_attrition_model_not_loaded(
    client: TestClient,
):  # Changed to synchronous and TestClient
    # Temporarily set model to None to simulate not loaded state
    with patch("api.app.main.model", new=None):
        employee_data = EmployeeFeatures(
            id_employee=1,
            age=30,
            genre=1,
            revenu_mensuel=5000.0,
            statut_marital="Célibataire",
            departement="R&D",
            poste="Développeur",
            nombre_experiences_precedentes=2,
            annee_experience_totale=8,
            annees_dans_l_entreprise=4,
            annees_dans_le_poste_actuel=2,
            nombre_participation_pee=1,
            nb_formations_suivies=2,
            nombre_employee_sous_responsabilite=0,
            distance_domicile_travail=10,
            niveau_education=3,
            domaine_etude="Informatique",
            ayant_enfants=0,
            frequence_deplacement="Rarement",
            annees_depuis_la_derniere_promotion=1,
            annes_sous_responsable_actuel=1,
            satisfaction_employee_environnement=3,
            note_evaluation_precedente=3.0,
            niveau_hierarchique_poste=2,
            satisfaction_employee_nature_travail=3,
            satisfaction_employee_equipe=3,
            satisfaction_employee_equilibre_pro_perso=3,
            note_evaluation_actuelle=3.5,
            heures_supplementaires=0,
            augementation_salaire_precedente=0.05,
        )
        batch_input = BatchPredictionInput(employees=[employee_data])
        response = client.post(
            "/predict", json=batch_input.model_dump()
        )  # Removed await, changed .dict() to .model_dump()
        assert response.status_code == 503
        assert "Model not loaded yet." in response.json()["detail"]


def test_predict_attrition_invalid_input(
    client: TestClient,
):  # Changed to synchronous and TestClient
    # Missing required field 'age'
    invalid_employee_data = {
        "id_employee": 1,
        "genre": 1,
        "revenu_mensuel": 5000.0,
        "statut_marital": "Célibataire",
        "departement": "R&D",
        "poste": "Développeur",
        "nombre_experiences_precedentes": 2,
        "annee_experience_totale": 8,
        "annees_dans_l_entreprise": 4,
        "annees_dans_le_poste_actuel": 2,
        "nombre_participation_pee": 1,
        "nb_formations_suivies": 2,
        "nombre_employee_sous_responsabilite": 0,
        "distance_domicile_travail": 10,
        "niveau_education": 3,
        "domaine_etude": "Informatique",
        "ayant_enfants": 0,
        "frequence_deplacement": "Rarement",
        "annees_depuis_la_derniere_promotion": 1,
        "annes_sous_responsable_actuel": 1,
        "satisfaction_employee_environnement": 3,
        "note_evaluation_precedente": 3.0,
        "niveau_hierarchique_poste": 2,
        "satisfaction_employee_nature_travail": 3,
        "satisfaction_employee_equipe": 3,
        "satisfaction_employee_equilibre_pro_perso": 3,
        "note_evaluation_actuelle": 3.5,
        "heures_supplementaires": 0,
        "augementation_salaire_precedente": 0.05,
    }
    batch_input = {"employees": [invalid_employee_data]}

    response = client.post("/predict", json=batch_input)  # Removed await
    assert (
        response.status_code == 422
    )  # Unprocessable Entity for Pydantic validation errors
    assert "Field required" in response.json()["detail"][0]["msg"]


@pytest.mark.asyncio
@patch("api.app.main.os.path.exists")
@patch("api.app.main.joblib.load")
@patch("api.app.main.get_expected_columns_from_pipeline")
async def test_lifespan_model_loaded_successfully(
    mock_get_expected_columns, mock_joblib_load, mock_path_exists
):

    mock_path_exists.return_value = True
    mock_joblib_load.return_value = MagicMock()
    mock_get_expected_columns.return_value = ["col1", "col2"]

    from api.app.main import lifespan  # Import the lifespan context manager
    from fastapi import FastAPI  # Import FastAPI to create a mock app

    mock_app = FastAPI()  # Create a mock FastAPI app

    async with lifespan(mock_app):  # Call the lifespan context manager directly
        from api.app.main import (
            model,
            expected_model_columns,
        )  # Import after lifespan context

        assert model is not None
        assert expected_model_columns == ["col1", "col2"]
        mock_path_exists.assert_called_once()
        mock_joblib_load.assert_called_once()
        mock_get_expected_columns.assert_called_once()


@pytest.mark.asyncio
@patch("api.app.main.os.path.exists")
async def test_lifespan_model_not_found(mock_path_exists):
    mock_path_exists.return_value = False

    from api.app.main import lifespan  # Import the lifespan context manager
    from fastapi import FastAPI  # Import FastAPI to create a mock app

    mock_app = FastAPI()  # Create a mock FastAPI app

    with pytest.raises(RuntimeError, match="Model file not found"):
        async with lifespan(mock_app):  # Call the lifespan context manager directly
            pass


# --- Helper Functions Tests (adapted from tests/test_app.py) ---
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


def test_clean_extrait_eval_salary_augmentation(sample_eval_df):
    from api.app.main import _clean_extrait_eval

    df = _clean_extrait_eval(sample_eval_df.copy())
    assert df["augmentation_salaire_precedente"].iloc[0] == 0.10
    assert df["augmentation_salaire_precedente"].iloc[1] == 0.055
    assert df["augmentation_salaire_precedente"].iloc[2] == 0.00
    assert pd.isna(df["augmentation_salaire_precedente"].iloc[3])


def test_clean_extrait_eval_heures_supplementaires(sample_eval_df):
    from api.app.main import _clean_extrait_eval

    df = _clean_extrait_eval(sample_eval_df.copy())
    assert df["heures_supplementaires"].iloc[0] == 1
    assert df["heures_supplementaires"].iloc[1] == 0
    assert df["heures_supplementaires"].iloc[2] == 1
    assert df["heures_supplementaires"].iloc[3] == 0


def test_clean_extrait_eval_id_employee(sample_eval_df):
    from api.app.main import _clean_extrait_eval

    df = _clean_extrait_eval(sample_eval_df.copy())
    assert "eval_number" not in df.columns
    assert df["id_employee"].iloc[0] == 1
    assert df["id_employee"].iloc[1] == 2


def test_clean_extrait_eval_missing_columns():
    from api.app.main import _clean_extrait_eval

    df = pd.DataFrame({"col_a": [1, 2]})
    cleaned_df = _clean_extrait_eval(df.copy())
    assert "augmentation_salaire_precedente" not in cleaned_df.columns
    assert "heures_supplementaires" not in cleaned_df.columns
    assert "id_employee" not in cleaned_df.columns


def test_clean_extrait_eval_empty_df():
    from api.app.main import _clean_extrait_eval

    df = pd.DataFrame()
    cleaned_df = _clean_extrait_eval(df.copy())
    assert cleaned_df.empty


def test_clean_extrait_eval_non_numeric_id_employee():
    from api.app.main import _clean_extrait_eval

    df = pd.DataFrame({"eval_number": ["E_abc"]})
    cleaned_df = _clean_extrait_eval(df.copy())
    assert pd.isna(cleaned_df["id_employee"].iloc[0])


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


def test_clean_extrait_sirh_genre(sample_sirh_df):
    from api.app.main import _clean_extrait_sirh

    df = _clean_extrait_sirh(sample_sirh_df.copy())
    assert df["genre"].iloc[0] == 1
    assert df["genre"].iloc[1] == 0
    assert df["genre"].iloc[2] == 1
    assert df["genre"].iloc[3] == 0


def test_clean_extrait_sirh_drop_columns(sample_sirh_df):
    from api.app.main import _clean_extrait_sirh

    df = _clean_extrait_sirh(sample_sirh_df.copy())
    assert "nombre_heures_travailless" not in df.columns


def test_clean_extrait_sirh_missing_columns():
    from api.app.main import _clean_extrait_sirh

    df = pd.DataFrame({"col_a": [1, 2]})
    cleaned_df = _clean_extrait_sirh(df.copy())
    assert "genre" not in cleaned_df.columns


def test_clean_extrait_sirh_empty_df():
    from api.app.main import _clean_extrait_sirh

    df = pd.DataFrame()
    cleaned_df = _clean_extrait_sirh(df.copy())
    assert cleaned_df.empty


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


def test_clean_extrait_sondage_id_employee(sample_sondage_df):
    from api.app.main import _clean_extrait_sondage

    df = _clean_extrait_sondage(sample_sondage_df.copy())
    assert "code_sondage" not in df.columns
    assert df["id_employee"].iloc[0] == 1
    assert df["id_employee"].iloc[2] == 4


def test_clean_extrait_sondage_missing_columns():
    from api.app.main import _clean_extrait_sondage

    df = pd.DataFrame({"col_a": [1, 2]})
    cleaned_df = _clean_extrait_sondage(df.copy())
    assert "id_employee" not in cleaned_df.columns


def test_clean_extrait_sondage_empty_df():
    from api.app.main import _clean_extrait_sondage

    df = pd.DataFrame()
    cleaned_df = _clean_extrait_sondage(df.copy())
    assert cleaned_df.empty


def test_clean_extrait_sondage_non_numeric_id_employee():
    from api.app.main import _clean_extrait_sondage

    df = pd.DataFrame({"code_sondage": ["abc"]})
    cleaned_df = _clean_extrait_sondage(df.copy())
    assert pd.isna(cleaned_df["id_employee"].iloc[0])


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


def test_load_and_merge_data_merge_logic(
    sample_eval_df, sample_sirh_df, sample_sondage_df
):
    from api.app.main import load_and_merge_data

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
    from api.app.main import load_and_merge_data

    eval_df = pd.DataFrame()
    sirh_df = pd.DataFrame()
    sond_df = pd.DataFrame()
    merged_df = load_and_merge_data(eval_df, sirh_df, sond_df)
    assert merged_df.empty


def test_clean_and_engineer_features_improvement_evaluation(
    merged_df_for_feature_engineering,
):
    from api.app.main import clean_and_engineer_features

    df = clean_and_engineer_features(merged_df_for_feature_engineering.copy())
    assert df["improvement_evaluation"].iloc[0] == 1
    assert df["improvement_evaluation"].iloc[1] == 0
    assert df["improvement_evaluation"].iloc[2] == 1
    assert df["improvement_evaluation"].iloc[3] == -1


def test_clean_and_engineer_features_total_satisfaction(
    merged_df_for_feature_engineering,
):
    from api.app.main import clean_and_engineer_features

    df = clean_and_engineer_features(merged_df_for_feature_engineering.copy())
    assert df["total_satisfaction"].iloc[0] == 4 * 5 * 3
    assert df["total_satisfaction"].iloc[1] == 3 * 4 * 2


def test_clean_and_engineer_features_work_mobility(merged_df_for_feature_engineering):
    from api.app.main import clean_and_engineer_features

    df = clean_and_engineer_features(merged_df_for_feature_engineering.copy())
    assert df["work_mobility"].iloc[0] == 2 / 5
    assert df["work_mobility"].iloc[1] == 1 / 3
    assert df["work_mobility"].iloc[3] == 0  # Division by zero handled


def test_clean_and_engineer_features_missing_columns():
    from api.app.main import clean_and_engineer_features

    df = pd.DataFrame({"id_employee": [1]})
    cleaned_df = clean_and_engineer_features(df.copy())
    assert "improvement_evaluation" not in cleaned_df.columns
    assert "total_satisfaction" not in cleaned_df.columns
    assert "work_mobility" not in cleaned_df.columns


def test_get_risk_category_low():
    from api.app.main import get_risk_category

    assert get_risk_category(0.1, 0.5) == "Low"
    assert get_risk_category(0.4, 0.5) == "Low"  # 0.4 < 0.45 (threshold - buffer)
    assert get_risk_category(0.29, 0.5) == "Low"  # 0.29 < 0.45 (threshold - buffer)


def test_get_risk_category_medium():
    from api.app.main import get_risk_category

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
    from api.app.main import get_risk_category

    assert get_risk_category(0.55, 0.5) == "High"  # 0.55 is threshold + buffer
    assert get_risk_category(0.7, 0.5) == "High"
    assert get_risk_category(0.9, 0.5) == "High"
    assert get_risk_category(1.0, 0.5) == "High"  # Edge case for upper bound


def test_get_risk_category_edge_cases():
    from api.app.main import get_risk_category

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


def test_predict_attrition_no_id_employee(client: TestClient):
    # Create a dictionary representing employee data without 'id_employee'
    employee_data_dict = {
        "age": 30,
        "genre": 1,
        "revenu_mensuel": 5000.0,
        "statut_marital": "Célibataire",
        "departement": "R&D",
        "poste": "Développeur",
        "nombre_experiences_precedentes": 2,
        "annee_experience_totale": 8,
        "annees_dans_l_entreprise": 4,
        "annees_dans_le_poste_actuel": 2,
        "nombre_participation_pee": 1,
        "nb_formations_suivies": 2,
        "nombre_employee_sous_responsabilite": 0,
        "distance_domicile_travail": 10,
        "niveau_education": 3,
        "domaine_etude": "Informatique",
        "ayant_enfants": 0,
        "frequence_deplacement": "Rarement",
        "annees_depuis_la_derniere_promotion": 1,
        "annes_sous_responsable_actuel": 1,
        "satisfaction_employee_environnement": 3,
        "note_evaluation_precedente": 3.0,
        "niveau_hierarchique_poste": 2,
        "satisfaction_employee_nature_travail": 3,
        "satisfaction_employee_equipe": 3,
        "satisfaction_employee_equilibre_pro_perso": 3,
        "note_evaluation_actuelle": 3.5,
        "heures_supplementaires": 0,
        "augementation_salaire_precedente": 0.05,
    }
    batch_input = {"employees": [employee_data_dict]}

    response = client.post("/predict", json=batch_input)
    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert len(predictions) == 1
    assert predictions[0]["id_employee"] == 1  # Expecting temporary ID 1


def test_predict_attrition_missing_feature_in_input(client: TestClient):
    # Create an employee data with a missing feature that is expected by the model
    # The mock expected_model_columns includes 'improvement_evaluation', 'total_satisfaction', 'work_mobility'
    # but the input EmployeeFeatures does not directly provide them.
    # The clean_and_engineer_features function should create them.
    # Let's test a scenario where a core feature is missing from the input that
    # is NOT engineered, and thus should be filled with 0.
    employee_data = EmployeeFeatures(
        id_employee=1,
        age=30,
        genre=1,
        revenu_mensuel=5000.0,
        statut_marital="Célibataire",
        departement="R&D",
        poste="Développeur",
        nombre_experiences_precedentes=2,
        annee_experience_totale=8,
        annees_dans_l_entreprise=4,
        annees_dans_le_poste_actuel=2,
        nombre_participation_pee=1,
        nb_formations_suivies=2,
        nombre_employee_sous_responsabilite=0,
        distance_domicile_travail=10,
        niveau_education=3,
        domaine_etude="Informatique",
        ayant_enfants=0,
        frequence_deplacement="Rarement",
        annees_depuis_la_derniere_promotion=1,
        annes_sous_responsable_actuel=1,
        satisfaction_employee_environnement=3,
        note_evaluation_precedente=3.0,
        niveau_hierarchique_poste=2,
        satisfaction_employee_nature_travail=3,
        satisfaction_employee_equipe=3,
        satisfaction_employee_equilibre_pro_perso=3,
        note_evaluation_actuelle=3.5,
        heures_supplementaires=0,
        augementation_salaire_precedente=0.05,
    )
    batch_input = BatchPredictionInput(employees=[employee_data])

    # To simulate a missing feature that would be filled with 0,
    # we need to ensure 'expected_model_columns' contains a column
    # that is neither in EmployeeFeatures nor created by feature engineering.
    # For this test, we'll temporarily patch expected_model_columns to include a dummy.
    with patch(
        "api.app.main.expected_model_columns",
        new=["id_employee", "age", "dummy_feature"],
    ):
        response = client.post("/predict", json=batch_input.model_dump())
        assert response.status_code == 200
        predictions = response.json()["predictions"]
        assert len(predictions) == 1
        # The actual prediction values are mocked, so we just check for success.


@patch("api.app.main.model")
def test_predict_attrition_internal_server_error(mock_model, client: TestClient):
    # Simulate an error during model prediction
    mock_model.predict_proba.side_effect = Exception("Model prediction failed")

    employee_data = EmployeeFeatures(
        id_employee=1,
        age=30,
        genre=1,
        revenu_mensuel=5000.0,
        statut_marital="Célibataire",
        departement="R&D",
        poste="Développeur",
        nombre_experiences_precedentes=2,
        annee_experience_totale=8,
        annees_dans_l_entreprise=4,
        annees_dans_le_poste_actuel=2,
        nombre_participation_pee=1,
        nb_formations_suivies=2,
        nombre_employee_sous_responsabilite=0,
        distance_domicile_travail=10,
        niveau_education=3,
        domaine_etude="Informatique",
        ayant_enfants=0,
        frequence_deplacement="Rarement",
        annees_depuis_la_derniere_promotion=1,
        annes_sous_responsable_actuel=1,
        satisfaction_employee_environnement=3,
        note_evaluation_precedente=3.0,
        niveau_hierarchique_poste=2,
        satisfaction_employee_nature_travail=3,
        satisfaction_employee_equipe=3,
        satisfaction_employee_equilibre_pro_perso=3,
        note_evaluation_actuelle=3.5,
        heures_supplementaires=0,
        augementation_salaire_precedente=0.05,
    )
    batch_input = BatchPredictionInput(employees=[employee_data])

    response = client.post("/predict", json=batch_input.model_dump())
    assert response.status_code == 500
    assert "Prediction failed: Model prediction failed" in response.json()["detail"]
