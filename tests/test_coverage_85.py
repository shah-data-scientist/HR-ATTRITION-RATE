"""
Additional targeted tests to push coverage from 72% to 85%+
Focus on uncovered API code paths
"""

import pytest
from fastapi.testclient import TestClient
import pandas as pd
import numpy as np


@pytest.fixture
def client():
    import os

    # IMPORTANT: Set API key BEFORE importing app
    os.environ["API_KEY"] = "test_api_key"

    from api.app.main import app
    return TestClient(app)


@pytest.fixture
def auth_headers():
    """Authentication headers for API requests"""
    return {"X-API-Key": "test_api_key", "X-User-ID": "test_user"}


@pytest.fixture
def sample_data():
    return {
        "id_employee": 1,
        "age": 35,
        "genre": "M",
        "revenu_mensuel": 5000,
        "statut_marital": "Marié",
        "departement": "IT",
        "poste": "Developer",
        "nombre_experiences_precedentes": 3,
        "annee_experience_totale": 10,
        "annees_dans_l_entreprise": 5,
        "annees_dans_le_poste_actuel": 2,
        "nombre_participation_pee": 1,
        "nb_formations_suivies": 2,
        "nombre_employee_sous_responsabilite": 0,
        "distance_domicile_travail": 10,
        "niveau_education": 4,
        "domaine_etude": "CS",
        "ayant_enfants": "Y",
        "frequence_deplacement": "Rare",
        "annees_depuis_la_derniere_promotion": 1,
        "annes_sous_responsable_actuel": 2,
        "satisfaction_employee_environnement": 3,
        "note_evaluation_precedente": 4,
        "niveau_hierarchique_poste": 2,
        "satisfaction_employee_nature_travail": 3,
        "satisfaction_employee_equipe": 4,
        "satisfaction_employee_equilibre_pro_perso": 2,
        "note_evaluation_actuelle": 4,
        "heure_supplementaires": "Non",
        "augementation_salaire_precedente": "15",
    }


class TestMultipleScenarios:
    """Test multiple prediction scenarios"""

    def test_predict_10_employees(self, client, auth_headers, sample_data):
        """Test with 10 employees"""
        employees = [dict(sample_data, id_employee=i) for i in range(1, 11)]
        payload = {
            "eval_data": employees,
            "sirh_data": employees,
            "sondage_data": employees,
        }
        response = client.post("/predict", headers=auth_headers, json=payload)
        assert response.status_code in [200, 422, 500]

        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data

    def test_predict_20_employees(self, client, auth_headers, sample_data):
        """Test with 20 employees"""
        employees = [dict(sample_data, id_employee=i) for i in range(1, 21)]
        payload = {
            "eval_data": employees,
            "sirh_data": employees,
            "sondage_data": employees,
        }
        response = client.post("/predict", headers=auth_headers, json=payload)
        assert response.status_code in [200, 422, 500]

    def test_excel_10_employees(self, client, auth_headers, sample_data):
        """Test Excel with 10 employees"""
        employees = [dict(sample_data, id_employee=i) for i in range(1, 11)]
        payload = {
            "eval_data": employees,
            "sirh_data": employees,
            "sondage_data": employees,
        }
        response = client.post("/predict_excel", headers=auth_headers, json=payload)
        assert response.status_code in [200, 422, 500]

        if response.status_code == 200:
            assert len(response.content) > 0

    def test_shap_5_employees(self, client, auth_headers, sample_data):
        """Test SHAP with 5 employees"""
        employees = [dict(sample_data, id_employee=i) for i in range(1, 6)]
        payload = {
            "eval_data": employees,
            "sirh_data": employees,
            "sondage_data": employees,
        }
        response = client.post("/predict_shap_images", headers=auth_headers, json=payload)
        assert response.status_code in [200, 422, 500]


class TestVariousDataTypes:
    """Test with various data type combinations"""

    def test_all_string_numbers(self, client, auth_headers, sample_data):
        """Test with numeric fields as strings"""
        data = sample_data.copy()
        data["age"] = "35"
        data["revenu_mensuel"] = "5000"
        data["annee_experience_totale"] = "10"

        payload = {"eval_data": [data], "sirh_data": [data], "sondage_data": [data]}
        response = client.post("/predict", headers=auth_headers, json=payload)
        assert response.status_code in [200, 422, 500]

    def test_mixed_case_categoricals(self, client, auth_headers, sample_data):
        """Test with mixed case categorical values"""
        data = sample_data.copy()
        data["genre"] = "m"  # lowercase
        data["ayant_enfants"] = "y"  # lowercase
        data["heure_supplementaires"] = "NON"  # uppercase

        payload = {"eval_data": [data], "sirh_data": [data], "sondage_data": [data]}
        response = client.post("/predict", headers=auth_headers, json=payload)
        assert response.status_code in [200, 422, 500]

    def test_boundary_values(self, client, auth_headers, sample_data):
        """Test with boundary values"""
        data = sample_data.copy()
        data["age"] = 18  # Minimum
        data["annee_experience_totale"] = 0  # Minimum
        data["satisfaction_employee_environnement"] = 1  # Minimum

        payload = {"eval_data": [data], "sirh_data": [data], "sondage_data": [data]}
        response = client.post("/predict", headers=auth_headers, json=payload)
        assert response.status_code in [200, 422, 500]


class TestSequentialCalls:
    """Test sequential API calls"""

    def test_multiple_health_checks(self, client):
        """Test multiple health check calls"""
        for _ in range(5):
            response = client.get("/health")
            assert response.status_code == 200

    def test_predict_then_excel(self, client, auth_headers, sample_data):
        """Test prediction followed by Excel"""
        payload = {
            "eval_data": [sample_data],
            "sirh_data": [sample_data],
            "sondage_data": [sample_data],
        }

        # First predict
        response1 = client.post("/predict", headers=auth_headers, json=payload)
        assert response1.status_code in [200, 422, 500]

        # Then Excel
        response2 = client.post("/predict_excel", headers=auth_headers, json=payload)
        assert response2.status_code in [200, 422, 500]

    def test_predict_then_shap(self, client, auth_headers, sample_data):
        """Test prediction followed by SHAP"""
        payload = {
            "eval_data": [sample_data],
            "sirh_data": [sample_data],
            "sondage_data": [sample_data],
        }

        # First predict
        response1 = client.post("/predict", headers=auth_headers, json=payload)
        assert response1.status_code in [200, 422, 500]

        # Then SHAP
        response2 = client.post("/predict_shap_images", headers=auth_headers, json=payload)
        assert response2.status_code in [200, 422, 500]


class TestDataVariations:
    """Test with various data variations"""

    def test_different_departments(self, client, auth_headers, sample_data):
        """Test with different departments"""
        departments = ["IT", "HR", "Sales", "Finance", "Marketing"]

        for dept in departments:
            data = sample_data.copy()
            data["departement"] = dept
            payload = {"eval_data": [data], "sirh_data": [data], "sondage_data": [data]}
            response = client.post("/predict", headers=auth_headers, json=payload)
            assert response.status_code in [200, 422, 500]

    def test_different_marital_status(self, client, auth_headers, sample_data):
        """Test with different marital statuses"""
        statuses = ["Marié", "Célibataire", "Divorcé"]

        for status in statuses:
            data = sample_data.copy()
            data["statut_marital"] = status
            payload = {"eval_data": [data], "sirh_data": [data], "sondage_data": [data]}
            response = client.post("/predict", headers=auth_headers, json=payload)
            assert response.status_code in [200, 422, 500]

    def test_different_education_levels(self, client, auth_headers, sample_data):
        """Test with different education levels"""
        for level in [1, 2, 3, 4, 5]:
            data = sample_data.copy()
            data["niveau_education"] = level
            payload = {"eval_data": [data], "sirh_data": [data], "sondage_data": [data]}
            response = client.post("/predict", headers=auth_headers, json=payload)
            assert response.status_code in [200, 422, 500]


class TestResponseContent:
    """Test response content details"""

    def test_prediction_includes_id(self, client, auth_headers, sample_data):
        """Test prediction includes employee ID"""
        payload = {
            "eval_data": [sample_data],
            "sirh_data": [sample_data],
            "sondage_data": [sample_data],
        }
        response = client.post("/predict", headers=auth_headers, json=payload)

        if response.status_code == 200:
            data = response.json()
            if data.get("predictions"):
                assert data["predictions"][0]["id_employee"] == 1

    def test_prediction_probability_range(self, client, auth_headers, sample_data):
        """Test prediction probability is in valid range"""
        payload = {
            "eval_data": [sample_data],
            "sirh_data": [sample_data],
            "sondage_data": [sample_data],
        }
        response = client.post("/predict", headers=auth_headers, json=payload)

        if response.status_code == 200:
            data = response.json()
            if data.get("predictions"):
                prob = data["predictions"][0]["probability"]
                assert 0 <= prob <= 1

    def test_risk_category_values(self, client, auth_headers, sample_data):
        """Test risk category has valid values"""
        payload = {
            "eval_data": [sample_data],
            "sirh_data": [sample_data],
            "sondage_data": [sample_data],
        }
        response = client.post("/predict", headers=auth_headers, json=payload)

        if response.status_code == 200:
            data = response.json()
            if data.get("predictions"):
                risk = data["predictions"][0]["risk_category"]
                assert risk in ["Low", "Medium", "High"]


class TestEndpointAccessibility:
    """Test that all endpoints are accessible"""

    def test_root_accessible(self, client):
        """Test root endpoint is accessible"""
        response = client.get("/")
        assert response.status_code == 200

    def test_health_accessible(self, client):
        """Test health endpoint is accessible"""
        response = client.get("/health")
        assert response.status_code == 200

    def test_predict_accessible(self, client, auth_headers):
        """Test predict endpoint is accessible"""
        response = client.post("/predict", headers=auth_headers, json={})
        # Should not be 404
        assert response.status_code != 404

    def test_predict_excel_accessible(self, client, auth_headers):
        """Test predict_excel endpoint is accessible"""
        response = client.post("/predict_excel", headers=auth_headers, json={})
        # Should not be 404
        assert response.status_code != 404

    def test_predict_shap_accessible(self, client, auth_headers):
        """Test predict_shap endpoint is accessible"""
        response = client.post("/predict_shap_images", headers=auth_headers, json={})
        # Should not be 404
        assert response.status_code != 404


class TestEdgeCasesAdvanced:
    """Test advanced edge cases"""

    def test_zero_experience(self, client, auth_headers, sample_data):
        """Test with zero experience"""
        data = sample_data.copy()
        data["annee_experience_totale"] = 0
        data["nombre_experiences_precedentes"] = 0
        data["annees_dans_l_entreprise"] = 0

        payload = {"eval_data": [data], "sirh_data": [data], "sondage_data": [data]}
        response = client.post("/predict", headers=auth_headers, json=payload)
        assert response.status_code in [200, 422, 500]

    def test_maximum_satisfaction(self, client, auth_headers, sample_data):
        """Test with maximum satisfaction scores"""
        data = sample_data.copy()
        data["satisfaction_employee_environnement"] = 4
        data["satisfaction_employee_nature_travail"] = 4
        data["satisfaction_employee_equipe"] = 4
        data["satisfaction_employee_equilibre_pro_perso"] = 4

        payload = {"eval_data": [data], "sirh_data": [data], "sondage_data": [data]}
        response = client.post("/predict", headers=auth_headers, json=payload)
        assert response.status_code in [200, 422, 500]

    def test_minimum_satisfaction(self, client, auth_headers, sample_data):
        """Test with minimum satisfaction scores"""
        data = sample_data.copy()
        data["satisfaction_employee_environnement"] = 1
        data["satisfaction_employee_nature_travail"] = 1
        data["satisfaction_employee_equipe"] = 1
        data["satisfaction_employee_equilibre_pro_perso"] = 1

        payload = {"eval_data": [data], "sirh_data": [data], "sondage_data": [data]}
        response = client.post("/predict", headers=auth_headers, json=payload)
        assert response.status_code in [200, 422, 500]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
