import pytest
from httpx import AsyncClient
from unittest.mock import patch, MagicMock # Import patch and MagicMock

from fastapi.testclient import TestClient # Import TestClient

import sys
import os

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from api.app.main import app, model, expected_model_columns, get_expected_columns_from_pipeline
from api.app.schemas import EmployeeFeatures, BatchPredictionInput, PredictionOutput
import pandas as pd
import numpy as np



@pytest.fixture # No longer async
def client():
    # Define the expected columns for the mock
    mock_expected_columns = [
        "id_employee", "age", "genre", "revenu_mensuel", "statut_marital",
        "departement", "poste", "nombre_experiences_precedentes",
        "annee_experience_totale", "annees_dans_l_entreprise",
        "annees_dans_le_poste_actuel", "nombre_participation_pee",
        "nb_formations_suivies", "nombre_employee_sous_responsabilite",
        "distance_domicile_travail", "niveau_education", "domaine_etude",
        "ayant_enfants", "frequence_deplacement",
        "annees_depuis_la_derniere_promotion", "annes_sous_responsable_actuel",
        "satisfaction_employee_environnement", "note_evaluation_precedente",
        "niveau_hierarchique_poste", "satisfaction_employee_nature_travail",
        "satisfaction_employee_equipe", "satisfaction_employee_equilibre_pro_perso",
        "note_evaluation_actuelle", "heures_supplementaires",
        "augementation_salaire_precedente", "improvement_evaluation",
        "total_satisfaction", "work_mobility"
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
        'preprocessor': mock_preprocessor,
        'model': mock_logreg_model
    }
    mock_pipeline.predict_proba.return_value = np.array([[0.2, 0.8], [0.3, 0.7]]) # For multiple predictions

    # Patch joblib.load to return our mock pipeline
    with patch('joblib.load', return_value=mock_pipeline), \
         patch('api.app.main.expected_model_columns', new=mock_expected_columns):
        with TestClient(app) as client:
            yield client

# --- Test Endpoints ---

def test_read_root(client: TestClient): # Changed to synchronous and TestClient
    response = client.get("/") # Removed await
    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to the Employee Attrition Prediction API!",
        "version": app.version,
        "documentation_url": "/docs"
    }

def test_predict_attrition_success(client: TestClient): # Changed to synchronous and TestClient
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
        augementation_salaire_precedente=0.05
    )
    batch_input = BatchPredictionInput(employees=[employee_data])

    response = client.post("/predict", json=batch_input.model_dump()) # Removed await, changed .dict() to .model_dump()
    assert response.status_code == 200
    
    predictions = response.json()["predictions"]
    assert len(predictions) == 1
    assert predictions[0]["id_employee"] == 1
    assert predictions[0]["prediction"] == "Leave" # Based on mock_model_and_columns returning 0.8 prob
    assert abs(predictions[0]["probability"] - 0.8) < 0.001
    assert predictions[0]["risk_category"] == "High" # 0.8 > 0.5 + 0.05 (buffer)

def test_predict_attrition_multiple_employees(client: TestClient): # Changed to synchronous and TestClient
    employee_data_1 = EmployeeFeatures(
        id_employee=1, age=30, genre=1, revenu_mensuel=5000.0, statut_marital="Célibataire",
        departement="R&D", poste="Développeur", nombre_experiences_precedentes=2,
        annee_experience_totale=8, annees_dans_l_entreprise=4, annees_dans_le_poste_actuel=2,
        nombre_participation_pee=1, nb_formations_suivies=2, nombre_employee_sous_responsabilite=0,
        distance_domicile_travail=10, niveau_education=3, domaine_etude="Informatique",
        ayant_enfants=0, frequence_deplacement="Rarement", annees_depuis_la_derniere_promotion=1,
        annes_sous_responsable_actuel=1, satisfaction_employee_environnement=3,
        note_evaluation_precedente=3.0, niveau_hierarchique_poste=2,
        satisfaction_employee_nature_travail=3, satisfaction_employee_equipe=3,
        satisfaction_employee_equilibre_pro_perso=3, note_evaluation_actuelle=3.5,
        heures_supplementaires=0, augementation_salaire_precedente=0.05
    )
    employee_data_2 = EmployeeFeatures(
        id_employee=2, age=45, genre=0, revenu_mensuel=7000.0, statut_marital="Marié",
        departement="Ventes", poste="Manager", nombre_experiences_precedentes=5,
        annee_experience_totale=20, annees_dans_l_entreprise=10, annees_dans_le_poste_actuel=5,
        nombre_participation_pee=2, nb_formations_suivies=3, nombre_employee_sous_responsabilite=5,
        distance_domicile_travail=20, niveau_education=4, domaine_etude="Commerce",
        ayant_enfants=1, frequence_deplacement="Fréquemment", annees_depuis_la_derniere_promotion=3,
        annes_sous_responsable_actuel=2, satisfaction_employee_environnement=4,
        note_evaluation_precedente=4.0, niveau_hierarchique_poste=4,
        satisfaction_employee_nature_travail=4, satisfaction_employee_equipe=4,
        satisfaction_employee_equilibre_pro_perso=4, note_evaluation_actuelle=4.5,
        heures_supplementaires=1, augementation_salaire_precedente=0.10
    )
    batch_input = BatchPredictionInput(employees=[employee_data_1, employee_data_2])

    response = client.post("/predict", json=batch_input.model_dump()) # Removed await, changed .dict() to .model_dump()
    assert response.status_code == 200
    
    predictions = response.json()["predictions"]
    assert len(predictions) == 2
    assert predictions[0]["id_employee"] == 1
    assert predictions[1]["id_employee"] == 2

def test_predict_attrition_model_not_loaded(client: TestClient): # Changed to synchronous and TestClient
    # Temporarily set model to None to simulate not loaded state
    with patch('api.app.main.model', new=None):
        employee_data = EmployeeFeatures(
            id_employee=1, age=30, genre=1, revenu_mensuel=5000.0, statut_marital="Célibataire",
            departement="R&D", poste="Développeur", nombre_experiences_precedentes=2,
            annee_experience_totale=8, annees_dans_l_entreprise=4, annees_dans_le_poste_actuel=2,
            nombre_participation_pee=1, nb_formations_suivies=2, nombre_employee_sous_responsabilite=0,
            distance_domicile_travail=10, niveau_education=3, domaine_etude="Informatique",
            ayant_enfants=0, frequence_deplacement="Rarement", annees_depuis_la_derniere_promotion=1,
            annes_sous_responsable_actuel=1, satisfaction_employee_environnement=3,
            note_evaluation_precedente=3.0, niveau_hierarchique_poste=2,
            satisfaction_employee_nature_travail=3, satisfaction_employee_equipe=3,
            satisfaction_employee_equilibre_pro_perso=3, note_evaluation_actuelle=3.5,
            heures_supplementaires=0, augementation_salaire_precedente=0.05
        )
        batch_input = BatchPredictionInput(employees=[employee_data])
        response = client.post("/predict", json=batch_input.model_dump()) # Removed await, changed .dict() to .model_dump()
        assert response.status_code == 503
        assert "Model not loaded yet." in response.json()["detail"]

def test_predict_attrition_invalid_input(client: TestClient): # Changed to synchronous and TestClient
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
        "augementation_salaire_precedente": 0.05
    }
    batch_input = {"employees": [invalid_employee_data]}

    response = client.post("/predict", json=batch_input) # Removed await
    assert response.status_code == 422 # Unprocessable Entity for Pydantic validation errors
    assert "Field required" in response.json()["detail"][0]["msg"]
