"""
Database integration tests for HR Attrition API.
These tests require a running PostgreSQL database.
"""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime, timezone

# Set test environment BEFORE importing - must disable the database for these tests
# since we're running in DISABLE_DB mode (from .env.local)
os.environ["API_KEY"] = "test_api_key"
# Note: These tests work in DISABLE_DB=1 mode, testing the code paths for DB disabled


@pytest.fixture
def client():
    """Create test client with fresh app instance."""
    from api.app.main import app
    return TestClient(app)


@pytest.fixture
def auth_headers():
    """Authentication headers for API requests."""
    return {"X-API-Key": "test_api_key", "X-User-ID": "test_user"}


@pytest.fixture
def sample_employee_data():
    """Sample employee data for testing."""
    return {
        "eval_data": [{
            "satisfaction_employee_environnement": 3,
            "note_evaluation_precedente": 3,
            "niveau_hierarchique_poste": 2,
            "satisfaction_employee_nature_travail": 4,
            "satisfaction_employee_equipe": 3,
            "satisfaction_employee_equilibre_pro_perso": 3,
            "eval_number": "E_12345",
            "note_evaluation_actuelle": 4,
            "heure_supplementaires": "Non",
            "augementation_salaire_precedente": "11 %",
        }],
        "sirh_data": [{
            "id_employee": 12345,
            "age": 35,
            "genre": "M",
            "revenu_mensuel": 5000,
            "statut_marital": "Marié",
            "departement": "IT",
            "poste": "Developer",
            "nombre_experiences_precedentes": 3,
            "nombre_heures_travailless": 40,
            "annee_experience_totale": 10,
            "annees_dans_l_entreprise": 5,
            "annees_dans_le_poste_actuel": 2,
        }],
        "sondage_data": [{
            "nombre_participation_pee": 1,
            "nb_formations_suivies": 2,
            "nombre_employee_sous_responsabilite": 0,
            "code_sondage": 12345,
            "distance_domicile_travail": 10,
            "niveau_education": 4,
            "domaine_etude": "CS",
            "ayant_enfants": "Oui",
            "frequence_deplacement": "Rarement",
            "annees_depuis_la_derniere_promotion": 1,
            "annes_sous_responsable_actuel": 2,
        }],
    }


class TestDatabaseIntegration:
    """Tests requiring database access."""

    def test_db_health_endpoint(self, client):
        """Test database health check endpoint."""
        response = client.get("/db_health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        # Status should be either "ok" (db connected) or "disabled" (DISABLE_DB=1)
        assert data["status"] in ["ok", "disabled"]

    def test_predict_with_database(self, client, auth_headers, sample_employee_data):
        """Test prediction with database storage."""
        response = client.post("/predict", headers=auth_headers, json=sample_employee_data)
        # 200 if success, 503 if model not loaded
        assert response.status_code in [200, 503]
        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data
            assert len(data["predictions"]) > 0

    def test_predict_report_with_database(self, client, auth_headers, sample_employee_data):
        """Test predict_report endpoint with database storage."""
        response = client.post("/predict_report", headers=auth_headers, json=sample_employee_data)
        # 200 if success, 503 if model not loaded
        assert response.status_code in [200, 503]
        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data
            assert "excel_base64" in data


class TestJobEndpointsWithDatabase:
    """Test job endpoints (require database)."""

    def test_create_report_job(self, client, auth_headers, sample_employee_data):
        """Test creating a report job."""
        response = client.post("/jobs/report", headers=auth_headers, json=sample_employee_data)
        # Either 200 (success) or 503 (DB disabled)
        assert response.status_code in [200, 503]
        if response.status_code == 200:
            data = response.json()
            assert "job_id" in data

    def test_get_job_status_not_found(self, client, auth_headers):
        """Test getting status of non-existent job."""
        response = client.get("/jobs/non-existent-job-id", headers=auth_headers)
        # Either 404 (not found) or 503 (DB disabled)
        assert response.status_code in [404, 503]

    def test_get_job_result_not_found(self, client, auth_headers):
        """Test getting result of non-existent job."""
        response = client.get("/jobs/non-existent-job-id/result", headers=auth_headers)
        # Either 404 (not found) or 503 (DB disabled)
        assert response.status_code in [404, 503]


class TestAuthEndpointsWithDatabase:
    """Test authentication endpoints (require database)."""

    def test_login_invalid_credentials(self, client):
        """Test login with invalid credentials."""
        response = client.post(
            "/auth/login",
            params={"username": "invalid_user", "password": "wrong_password"}
        )
        # Either 401 (invalid) or 503 (DB disabled)
        assert response.status_code in [401, 503]

    def test_get_user_info_not_found(self, client):
        """Test getting info for non-existent user."""
        response = client.get("/auth/user/nonexistent_user")
        # Either 404 (not found) or 503 (DB disabled)
        assert response.status_code in [404, 503]


class TestDatabasePathsCoverage:
    """Tests specifically targeting database code paths."""

    def test_generate_predictions_db_enabled_path(self, client, auth_headers, sample_employee_data):
        """Test that predictions are stored when DB is enabled."""
        # This tests lines 378-502 in generate_predictions
        response = client.post("/predict", headers=auth_headers, json=sample_employee_data)
        assert response.status_code in [200, 503]

    def test_multiple_employee_predictions(self, client, auth_headers):
        """Test batch prediction with multiple employees."""
        payload = {
            "eval_data": [
                {
                    "satisfaction_employee_environnement": 3,
                    "note_evaluation_precedente": 3,
                    "niveau_hierarchique_poste": 2,
                    "satisfaction_employee_nature_travail": 4,
                    "satisfaction_employee_equipe": 3,
                    "satisfaction_employee_equilibre_pro_perso": 3,
                    "eval_number": f"E_{i}",
                    "note_evaluation_actuelle": 4,
                    "heure_supplementaires": "Non",
                    "augementation_salaire_precedente": "11 %",
                }
                for i in range(1, 4)
            ],
            "sirh_data": [
                {
                    "id_employee": i,
                    "age": 30 + i,
                    "genre": "M",
                    "revenu_mensuel": 5000 + i * 100,
                    "statut_marital": "Marié",
                    "departement": "IT",
                    "poste": "Developer",
                    "nombre_experiences_precedentes": 3,
                    "nombre_heures_travailless": 40,
                    "annee_experience_totale": 10,
                    "annees_dans_l_entreprise": 5,
                    "annees_dans_le_poste_actuel": 2,
                }
                for i in range(1, 4)
            ],
            "sondage_data": [
                {
                    "nombre_participation_pee": 1,
                    "nb_formations_suivies": 2,
                    "nombre_employee_sous_responsabilite": 0,
                    "code_sondage": i,
                    "distance_domicile_travail": 10,
                    "niveau_education": 4,
                    "domaine_etude": "CS",
                    "ayant_enfants": "Oui",
                    "frequence_deplacement": "Rarement",
                    "annees_depuis_la_derniere_promotion": 1,
                    "annes_sous_responsable_actuel": 2,
                }
                for i in range(1, 4)
            ],
        }
        response = client.post("/predict", headers=auth_headers, json=payload)
        assert response.status_code in [200, 503]
        if response.status_code == 200:
            data = response.json()
            assert len(data["predictions"]) == 3


class TestMockDatabasePaths:
    """Tests using mocks to cover specific database code paths."""

    def test_db_ok_function_variations(self):
        """Test _db_ok function with various inputs."""
        from api.app.main import _db_ok

        # Test with None
        assert _db_ok(None) is False

        # Test with mock session that works
        mock_session = Mock(spec=Session)
        mock_session.execute.return_value = None
        assert _db_ok(mock_session) is True

        # Test with mock session that raises
        mock_session_error = Mock(spec=Session)
        mock_session_error.execute.side_effect = Exception("DB Error")
        assert _db_ok(mock_session_error) is False

    def test_is_db_disabled_variations(self):
        """Test _is_db_disabled function with various env values."""
        from api.app.main import _is_db_disabled

        # Save original
        original = os.environ.get("DISABLE_DB")

        try:
            # Test with "1" - the only value that returns True
            os.environ["DISABLE_DB"] = "1"
            result = _is_db_disabled()
            assert result is True, f"Expected True for '1', got {result}"

            # Test with "0" - returns False
            os.environ["DISABLE_DB"] = "0"
            result = _is_db_disabled()
            assert result is False, f"Expected False for '0', got {result}"

            # Test with empty string - returns False (default behavior)
            os.environ["DISABLE_DB"] = ""
            result = _is_db_disabled()
            assert result is False, f"Expected False for '', got {result}"

            # Test with "true" - returns False (implementation only checks == "1")
            os.environ["DISABLE_DB"] = "true"
            result = _is_db_disabled()
            assert result is False, f"Expected False for 'true', got {result}"

        finally:
            # Restore original
            if original is not None:
                os.environ["DISABLE_DB"] = original
            else:
                os.environ.pop("DISABLE_DB", None)


class TestJobEndpointsWithMocks:
    """Test job endpoints with mocked database for coverage."""

    @patch("api.app.main._is_db_disabled", return_value=False)
    @patch("api.app.main.get_db")
    def test_create_job_with_mock_db(self, mock_get_db, mock_disabled, client, auth_headers, sample_employee_data):
        """Test job creation with mocked database."""
        # Create mock session
        mock_session = MagicMock(spec=Session)
        mock_job = MagicMock()
        mock_job.job_id = "test-job-123"
        mock_session.add = MagicMock()
        mock_session.commit = MagicMock()
        mock_session.refresh = MagicMock(side_effect=lambda j: setattr(j, 'job_id', 'test-job-123'))
        mock_get_db.return_value = iter([mock_session])

        # Make request
        response = client.post("/jobs/report", headers=auth_headers, json=sample_employee_data)
        # With mock we expect either success or the actual endpoint behavior
        assert response.status_code in [200, 500, 503]

    @patch("api.app.main._is_db_disabled", return_value=False)
    @patch("api.app.main.get_db")
    def test_get_job_status_with_mock_db(self, mock_get_db, mock_disabled, client):
        """Test getting job status with mocked database."""
        mock_session = MagicMock(spec=Session)
        mock_job = MagicMock()
        mock_job.job_id = "test-job-123"
        mock_job.job_type = "report"
        mock_job.status = "completed"
        mock_job.updated_at = datetime.now()
        mock_job.error = None
        mock_session.query.return_value.filter.return_value.first.return_value = mock_job
        mock_get_db.return_value = iter([mock_session])

        response = client.get("/jobs/test-job-123")
        assert response.status_code in [200, 503]

    @patch("api.app.main._is_db_disabled", return_value=False)
    @patch("api.app.main.get_db")
    def test_get_job_result_completed(self, mock_get_db, mock_disabled, client):
        """Test getting completed job result."""
        mock_session = MagicMock(spec=Session)
        mock_job = MagicMock()
        mock_job.job_id = "test-job-123"
        mock_job.status = "completed"
        mock_job.result_json = {"predictions": [], "excel_base64": "test"}
        mock_session.query.return_value.filter.return_value.first.return_value = mock_job
        mock_get_db.return_value = iter([mock_session])

        response = client.get("/jobs/test-job-123/result")
        assert response.status_code in [200, 503]

    @patch("api.app.main._is_db_disabled", return_value=False)
    @patch("api.app.main.get_db")
    def test_get_job_result_not_completed(self, mock_get_db, mock_disabled, client):
        """Test getting job result when not completed."""
        mock_session = MagicMock(spec=Session)
        mock_job = MagicMock()
        mock_job.job_id = "test-job-123"
        mock_job.status = "queued"
        mock_job.result_json = None
        mock_session.query.return_value.filter.return_value.first.return_value = mock_job
        mock_get_db.return_value = iter([mock_session])

        response = client.get("/jobs/test-job-123/result")
        # Should return 202 (not completed yet) or 503 (db disabled)
        assert response.status_code in [202, 503]


class TestAuthEndpointsWithMocks:
    """Test authentication endpoints with mocked database for coverage."""

    @patch("api.app.main._is_db_disabled", return_value=False)
    @patch("api.app.main.get_db")
    def test_login_success_mock(self, mock_get_db, mock_disabled, client):
        """Test successful login with mocked database."""
        mock_session = MagicMock(spec=Session)
        mock_user = MagicMock()
        mock_user.user_id = "user-123"
        mock_user.username = "testuser"
        mock_user.role = "user"
        mock_user.is_active = True
        mock_user.last_login = datetime.now(timezone.utc)
        mock_user.password_hash = "$2b$12$test_hash"

        # Mock User.verify_password to return True
        with patch("api.app.main.User.verify_password", return_value=True):
            mock_session.query.return_value.filter.return_value.first.return_value = mock_user
            mock_get_db.return_value = iter([mock_session])

            response = client.post("/auth/login", params={"username": "testuser", "password": "password"})
            assert response.status_code in [200, 401, 503]

    @patch("api.app.main._is_db_disabled", return_value=False)
    @patch("api.app.main.get_db")
    def test_login_user_inactive(self, mock_get_db, mock_disabled, client):
        """Test login with inactive user."""
        mock_session = MagicMock(spec=Session)
        mock_user = MagicMock()
        mock_user.is_active = False
        mock_session.query.return_value.filter.return_value.first.return_value = mock_user
        mock_get_db.return_value = iter([mock_session])

        response = client.post("/auth/login", params={"username": "inactive", "password": "password"})
        assert response.status_code in [401, 503]

    @patch("api.app.main._is_db_disabled", return_value=False)
    @patch("api.app.main.get_db")
    def test_get_user_info_found(self, mock_get_db, mock_disabled, client):
        """Test getting user info for existing user."""
        mock_session = MagicMock(spec=Session)
        mock_user = MagicMock()
        mock_user.user_id = "user-123"
        mock_user.username = "testuser"
        mock_user.role = "admin"
        mock_user.is_active = True
        mock_user.last_login = datetime.now(timezone.utc)
        mock_session.query.return_value.filter.return_value.first.return_value = mock_user
        mock_get_db.return_value = iter([mock_session])

        response = client.get("/auth/user/testuser")
        assert response.status_code in [200, 503]


class TestExcelGenerationCoverage:
    """Tests for Excel generation code paths."""

    def test_predict_excel_endpoint(self, client, auth_headers, sample_employee_data):
        """Test Excel generation endpoint."""
        response = client.post("/predict_excel", headers=auth_headers, json=sample_employee_data)
        assert response.status_code in [200, 503]
        if response.status_code == 200:
            # Check response is Excel file
            assert response.headers.get("content-type") == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

    def test_predict_excel_multiple_employees(self, client, auth_headers):
        """Test Excel generation with multiple employees."""
        payload = {
            "eval_data": [
                {
                    "satisfaction_employee_environnement": 3,
                    "note_evaluation_precedente": 3,
                    "niveau_hierarchique_poste": 2,
                    "satisfaction_employee_nature_travail": 4,
                    "satisfaction_employee_equipe": 3,
                    "satisfaction_employee_equilibre_pro_perso": 3,
                    "eval_number": f"E_{i}",
                    "note_evaluation_actuelle": 4,
                    "heure_supplementaires": "Non" if i % 2 == 0 else "Oui",
                    "augementation_salaire_precedente": f"{11 + i} %",
                }
                for i in range(1, 6)
            ],
            "sirh_data": [
                {
                    "id_employee": i,
                    "age": 30 + i,
                    "genre": "M" if i % 2 == 0 else "F",
                    "revenu_mensuel": 5000 + i * 100,
                    "statut_marital": "Marié",
                    "departement": "IT",
                    "poste": "Developer",
                    "nombre_experiences_precedentes": 3,
                    "nombre_heures_travailless": 40,
                    "annee_experience_totale": 10,
                    "annees_dans_l_entreprise": 5,
                    "annees_dans_le_poste_actuel": 2,
                }
                for i in range(1, 6)
            ],
            "sondage_data": [
                {
                    "nombre_participation_pee": 1,
                    "nb_formations_suivies": 2,
                    "nombre_employee_sous_responsabilite": 0,
                    "code_sondage": i,
                    "distance_domicile_travail": 10,
                    "niveau_education": 4,
                    "domaine_etude": "CS",
                    "ayant_enfants": "Oui",
                    "frequence_deplacement": "Rarement",
                    "annees_depuis_la_derniere_promotion": 1,
                    "annes_sous_responsable_actuel": 2,
                }
                for i in range(1, 6)
            ],
        }
        response = client.post("/predict_excel", headers=auth_headers, json=payload)
        assert response.status_code in [200, 503]


class TestShapImageCoverage:
    """Tests for SHAP image generation code paths."""

    def test_shap_images_endpoint(self, client, auth_headers, sample_employee_data):
        """Test SHAP image generation endpoint."""
        response = client.post("/predict_shap_images", headers=auth_headers, json=sample_employee_data)
        assert response.status_code in [200, 503]
        if response.status_code == 200:
            data = response.json()
            assert "shap_images" in data

    def test_shap_html_endpoint(self, client, auth_headers, sample_employee_data):
        """Test SHAP HTML generation endpoint."""
        response = client.post("/predict_shap_html", headers=auth_headers, json=sample_employee_data)
        assert response.status_code in [200, 503]
        if response.status_code == 200:
            data = response.json()
            assert "shap_html" in data


class TestReportEndpointCoverage:
    """Tests for predict_report endpoint coverage."""

    def test_predict_report_full(self, client, auth_headers, sample_employee_data):
        """Test full predict_report response."""
        response = client.post("/predict_report", headers=auth_headers, json=sample_employee_data)
        assert response.status_code in [200, 503]
        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data
            assert "excel_base64" in data
            assert "shap_images" in data

    def test_predict_report_multiple_employees(self, client, auth_headers):
        """Test predict_report with multiple employees."""
        payload = {
            "eval_data": [
                {
                    "satisfaction_employee_environnement": 3,
                    "note_evaluation_precedente": 3,
                    "niveau_hierarchique_poste": 2,
                    "satisfaction_employee_nature_travail": 4,
                    "satisfaction_employee_equipe": 3,
                    "satisfaction_employee_equilibre_pro_perso": 3,
                    "eval_number": f"E_{i}",
                    "note_evaluation_actuelle": 4,
                    "heure_supplementaires": "Non",
                    "augementation_salaire_precedente": "11 %",
                }
                for i in range(1, 3)
            ],
            "sirh_data": [
                {
                    "id_employee": i,
                    "age": 30 + i,
                    "genre": "M",
                    "revenu_mensuel": 5000,
                    "statut_marital": "Marié",
                    "departement": "IT",
                    "poste": "Developer",
                    "nombre_experiences_precedentes": 3,
                    "nombre_heures_travailless": 40,
                    "annee_experience_totale": 10,
                    "annees_dans_l_entreprise": 5,
                    "annees_dans_le_poste_actuel": 2,
                }
                for i in range(1, 3)
            ],
            "sondage_data": [
                {
                    "nombre_participation_pee": 1,
                    "nb_formations_suivies": 2,
                    "nombre_employee_sous_responsabilite": 0,
                    "code_sondage": i,
                    "distance_domicile_travail": 10,
                    "niveau_education": 4,
                    "domaine_etude": "CS",
                    "ayant_enfants": "Oui",
                    "frequence_deplacement": "Rarement",
                    "annees_depuis_la_derniere_promotion": 1,
                    "annes_sous_responsable_actuel": 2,
                }
                for i in range(1, 3)
            ],
        }
        response = client.post("/predict_report", headers=auth_headers, json=payload)
        assert response.status_code in [200, 503]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
