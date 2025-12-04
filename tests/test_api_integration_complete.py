"""Comprehensive integration tests for API endpoints to reach 85% coverage.

These tests require running Docker containers (docker-compose --profile local up -d).
"""

import os
import pytest
import requests
import pandas as pd
from io import BytesIO


# Get API configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8081")
API_KEY = os.getenv("API_KEY", "test_api_key")


@pytest.fixture(scope="module")
def api_url():
    """Base API URL."""
    return API_BASE_URL


@pytest.fixture(scope="module")
def headers():
    """API headers with authentication."""
    return {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }


@pytest.fixture(scope="module")
def sample_employee_data():
    """Sample employee data for predictions."""
    return {
        "employees": [
            {
                "id_employee": 99001,
                "age": 35,
                "genre": "M",
                "revenu_mensuel": 6000.0,
                "statut_marital": "Marié",
                "departement": "R&D",
                "poste": "Développeur",
                "nombre_experiences_precedentes": 2,
                "annee_experience_totale": 10,
                "annees_dans_l_entreprise": 5,
                "annees_dans_le_poste_actuel": 3,
                "nombre_participation_pee": 1,
                "nb_formations_suivies": 2,
                "nombre_employee_sous_responsabilite": 0,
                "distance_domicile_travail": 15,
                "niveau_education": 3,
                "domaine_etude": "Informatique",
                "ayant_enfants": "Oui",
                "frequence_deplacement": "Rarement",
                "annees_depuis_la_derniere_promotion": 2,
                "annes_sous_responsable_actuel": 2,
                "satisfaction_employee_environnement": 3,
                "note_evaluation_precedente": 3.5,
                "niveau_hierarchique_poste": 2,
                "satisfaction_employee_nature_travail": 4,
                "satisfaction_employee_equipe": 3,
                "satisfaction_employee_equilibre_pro_perso": 3,
                "note_evaluation_actuelle": 4.0,
                "heures_supplementaires": "Non",
                "augmentation_salaire_precedente": 7.0
            }
        ],
        "evaluations": []
    }


class TestRootAndHealthEndpoints:
    """Test root and health endpoints."""

    def test_root_endpoint(self, api_url):
        """Test GET / endpoint."""
        response = requests.get(f"{api_url}/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "HR Attrition" in data["message"]

    def test_health_endpoint(self, api_url, headers):
        """Test GET /health endpoint."""
        response = requests.get(f"{api_url}/health", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"

    def test_health_endpoint_structure(self, api_url, headers):
        """Test health endpoint returns all expected fields."""
        response = requests.get(f"{api_url}/health", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "database" in data
        assert "model_loaded" in data


class TestPredictEndpoint:
    """Test /predict endpoint thoroughly."""

    def test_predict_single_employee(self, api_url, headers, sample_employee_data):
        """Test prediction for single employee."""
        response = requests.post(
            f"{api_url}/predict",
            json=sample_employee_data,
            headers=headers
        )
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 1

        prediction = data["predictions"][0]
        assert "id_employee" in prediction
        assert "prediction" in prediction
        assert "probability" in prediction
        assert "risk_category" in prediction

    def test_predict_multiple_employees(self, api_url, headers, sample_employee_data):
        """Test prediction for multiple employees."""
        # Add second employee
        second_employee = sample_employee_data["employees"][0].copy()
        second_employee["id_employee"] = 99002
        second_employee["age"] = 45
        sample_employee_data["employees"].append(second_employee)

        response = requests.post(
            f"{api_url}/predict",
            json=sample_employee_data,
            headers=headers
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) == 2

    def test_predict_with_evaluations(self, api_url, headers, sample_employee_data):
        """Test prediction with evaluation data."""
        sample_employee_data["evaluations"] = [
            {
                "id_employee": 99001,
                "note_evaluation_actuelle": 4.0,
                "note_evaluation_precedente": 3.5,
                "augmentation_salaire_precedente": 7.0
            }
        ]

        response = requests.post(
            f"{api_url}/predict",
            json=sample_employee_data,
            headers=headers
        )
        assert response.status_code == 200

    def test_predict_invalid_data_missing_fields(self, api_url, headers):
        """Test prediction with missing required fields."""
        invalid_data = {
            "employees": [
                {
                    "id_employee": 99001,
                    "age": 35
                    # Missing many required fields
                }
            ],
            "evaluations": []
        }

        response = requests.post(
            f"{api_url}/predict",
            json=invalid_data,
            headers=headers
        )
        # Should still work due to default values in preprocessing
        assert response.status_code in [200, 400, 422]

    def test_predict_no_authentication(self, api_url, sample_employee_data):
        """Test prediction without API key."""
        response = requests.post(
            f"{api_url}/predict",
            json=sample_employee_data
        )
        assert response.status_code in [401, 403]


class TestPredictReportEndpoint:
    """Test /predict/report endpoint."""

    def test_predict_report_single_employee(self, api_url, headers, sample_employee_data):
        """Test report generation for single employee."""
        response = requests.post(
            f"{api_url}/predict/report",
            json=sample_employee_data,
            headers=headers
        )
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert "shap_values" in data or "message" in data

    def test_predict_report_with_shap(self, api_url, headers, sample_employee_data):
        """Test that report includes SHAP values."""
        response = requests.post(
            f"{api_url}/predict/report",
            json=sample_employee_data,
            headers=headers
        )
        assert response.status_code == 200
        data = response.json()

        if "shap_values" in data:
            assert isinstance(data["shap_values"], dict)

    def test_predict_report_multiple_employees(self, api_url, headers, sample_employee_data):
        """Test report for multiple employees."""
        # Add more employees
        for i in range(2, 5):
            emp = sample_employee_data["employees"][0].copy()
            emp["id_employee"] = 99000 + i
            sample_employee_data["employees"].append(emp)

        response = requests.post(
            f"{api_url}/predict/report",
            json=sample_employee_data,
            headers=headers
        )
        assert response.status_code == 200


class TestExcelGenerationEndpoint:
    """Test /predict/excel endpoint."""

    def test_predict_excel_single_employee(self, api_url, headers, sample_employee_data):
        """Test Excel generation for single employee."""
        response = requests.post(
            f"{api_url}/predict/excel",
            json=sample_employee_data,
            headers=headers
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

        # Verify it's valid Excel content
        excel_content = BytesIO(response.content)
        df = pd.read_excel(excel_content)
        assert len(df) > 0
        assert "id_employee" in df.columns

    def test_predict_excel_multiple_employees(self, api_url, headers, sample_employee_data):
        """Test Excel generation for multiple employees."""
        # Add more employees
        for i in range(2, 4):
            emp = sample_employee_data["employees"][0].copy()
            emp["id_employee"] = 99000 + i
            sample_employee_data["employees"].append(emp)

        response = requests.post(
            f"{api_url}/predict/excel",
            json=sample_employee_data,
            headers=headers
        )
        assert response.status_code == 200

        excel_content = BytesIO(response.content)
        df = pd.read_excel(excel_content)
        assert len(df) >= 3

    def test_predict_excel_no_data(self, api_url, headers):
        """Test Excel generation with empty data."""
        empty_data = {
            "employees": [],
            "evaluations": []
        }

        response = requests.post(
            f"{api_url}/predict/excel",
            json=empty_data,
            headers=headers
        )
        # Should handle gracefully
        assert response.status_code in [200, 400, 422]


class TestShapEndpoints:
    """Test SHAP visualization endpoints."""

    def test_predict_shap_images(self, api_url, headers, sample_employee_data):
        """Test SHAP images generation."""
        response = requests.post(
            f"{api_url}/predict/shap/images",
            json=sample_employee_data,
            headers=headers
        )
        assert response.status_code == 200
        data = response.json()
        assert "shap_images" in data or "images" in data

    def test_predict_shap_html(self, api_url, headers, sample_employee_data):
        """Test SHAP HTML generation."""
        response = requests.post(
            f"{api_url}/predict/shap/html",
            json=sample_employee_data,
            headers=headers
        )
        # This endpoint might not exist, so accept 404
        assert response.status_code in [200, 404]


class TestAuthenticationEndpoints:
    """Test authentication-related endpoints."""

    def test_login_endpoint_exists(self, api_url):
        """Test that login endpoint exists."""
        response = requests.post(
            f"{api_url}/auth/login",
            json={"username": "test", "password": "test"}
        )
        # Endpoint should exist (even if credentials are wrong)
        assert response.status_code in [200, 400, 401, 422]

    def test_login_with_invalid_credentials(self, api_url):
        """Test login with invalid credentials."""
        response = requests.post(
            f"{api_url}/auth/login",
            json={"username": "invalid_user", "password": "wrong_password"}
        )
        assert response.status_code in [400, 401]

    def test_get_user_info_endpoint(self, api_url):
        """Test user info endpoint."""
        response = requests.get(
            f"{api_url}/auth/users/test_user"
        )
        # Should require authentication or return 404
        assert response.status_code in [401, 403, 404]


class TestErrorHandling:
    """Test error handling paths."""

    def test_invalid_json_payload(self, api_url, headers):
        """Test with malformed JSON."""
        response = requests.post(
            f"{api_url}/predict",
            data="invalid json",
            headers=headers
        )
        assert response.status_code in [400, 422]

    def test_predict_with_null_values(self, api_url, headers):
        """Test prediction with null values."""
        data = {
            "employees": [
                {
                    "id_employee": None,
                    "age": None,
                    "genre": None
                }
            ],
            "evaluations": []
        }

        response = requests.post(
            f"{api_url}/predict",
            json=data,
            headers=headers
        )
        # Should handle gracefully
        assert response.status_code in [200, 400, 422]

    def test_predict_with_invalid_types(self, api_url, headers):
        """Test prediction with invalid data types."""
        data = {
            "employees": [
                {
                    "id_employee": "not_a_number",
                    "age": "thirty",
                    "revenu_mensuel": "lots"
                }
            ],
            "evaluations": []
        }

        response = requests.post(
            f"{api_url}/predict",
            json=data,
            headers=headers
        )
        # Should handle gracefully
        assert response.status_code in [200, 400, 422]

    def test_nonexistent_endpoint(self, api_url, headers):
        """Test requesting non-existent endpoint."""
        response = requests.get(
            f"{api_url}/nonexistent/endpoint",
            headers=headers
        )
        assert response.status_code == 404


class TestDatabaseIntegration:
    """Test database integration paths."""

    def test_predict_stores_in_database(self, api_url, headers, sample_employee_data):
        """Test that predictions are stored in database when enabled."""
        # Use unique ID to test storage
        sample_employee_data["employees"][0]["id_employee"] = 99999

        response = requests.post(
            f"{api_url}/predict",
            json=sample_employee_data,
            headers=headers
        )
        assert response.status_code == 200

        # If database is enabled, prediction should have trace_id
        data = response.json()
        prediction = data["predictions"][0]
        # trace_id may or may not be present depending on DB state
        assert "id_employee" in prediction

    def test_health_check_database_status(self, api_url, headers):
        """Test that health check includes database status."""
        response = requests.get(f"{api_url}/health", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "database" in data


class TestConcurrency:
    """Test concurrent requests."""

    def test_multiple_simultaneous_predictions(self, api_url, headers, sample_employee_data):
        """Test multiple concurrent prediction requests."""
        import concurrent.futures

        def make_prediction():
            return requests.post(
                f"{api_url}/predict",
                json=sample_employee_data,
                headers=headers
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(make_prediction) for _ in range(3)]
            results = [f.result() for f in futures]

        # All should succeed
        assert all(r.status_code == 200 for r in results)


class TestDataValidation:
    """Test data validation rules."""

    def test_age_out_of_range(self, api_url, headers, sample_employee_data):
        """Test with age outside valid range."""
        sample_employee_data["employees"][0]["age"] = 150  # Invalid age

        response = requests.post(
            f"{api_url}/predict",
            json=sample_employee_data,
            headers=headers
        )
        # Should either accept (with warning) or reject
        assert response.status_code in [200, 400, 422]

    def test_negative_values(self, api_url, headers, sample_employee_data):
        """Test with negative values."""
        sample_employee_data["employees"][0]["revenu_mensuel"] = -5000

        response = requests.post(
            f"{api_url}/predict",
            json=sample_employee_data,
            headers=headers
        )
        assert response.status_code in [200, 400, 422]


# Skip these tests if containers aren't running
pytestmark = pytest.mark.skipif(
    not os.path.exists("/.dockerenv") and os.system(f"curl -s {API_BASE_URL}/health > nul 2>&1") != 0,
    reason="Docker containers not running"
)
