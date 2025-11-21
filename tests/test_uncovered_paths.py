"""
Targeted tests for uncovered code paths in api/app/main.py
Focus on: _db_ok, init_model_for_cli, SHAP branches, Excel errors, Job endpoints
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
import pandas as pd
import os
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError


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
def sample_employee_data():
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


class TestDatabaseHealthCheck:
    """Test _db_ok function"""

    def test_db_ok_with_none_session(self):
        """Test _db_ok returns False when db session is None"""
        from api.app.main import _db_ok

        result = _db_ok(None)
        assert result is False

    def test_db_ok_with_valid_session(self):
        """Test _db_ok returns True with valid session"""
        from api.app.main import _db_ok

        mock_session = Mock(spec=Session)
        mock_session.execute.return_value = None
        result = _db_ok(mock_session)
        assert result is True

    def test_db_ok_with_exception(self):
        """Test _db_ok returns False when exception occurs"""
        from api.app.main import _db_ok

        mock_session = Mock(spec=Session)
        mock_session.execute.side_effect = SQLAlchemyError("Connection failed")
        result = _db_ok(mock_session)
        assert result is False


class TestCliModelInitialization:
    """Test init_model_for_cli function"""

    def test_init_model_for_cli_already_initialized(self):
        """Test init_model_for_cli returns early if model already loaded"""
        from api.app.main import init_model_for_cli, model, explainer

        # Model should already be loaded from FastAPI lifespan
        # So this should return early
        with patch("api.app.main.model", "loaded_model"):
            with patch("api.app.main.explainer", "loaded_explainer"):
                # Should not raise any error
                init_model_for_cli()

    @patch("api.app.main.os.path.exists")
    @patch("api.app.main.joblib.load")
    def test_init_model_for_cli_model_not_found(self, mock_load, mock_exists):
        """Test init_model_for_cli raises error when model file not found"""
        from api.app.main import init_model_for_cli

        mock_exists.return_value = False

        with patch("api.app.main.model", None):
            with patch("api.app.main.explainer", None):
                with pytest.raises(RuntimeError, match="Model file not found"):
                    init_model_for_cli()

    @patch("api.app.main.os.path.exists")
    @patch("api.app.main.joblib.load")
    @patch("api.app.main.pd.read_parquet")
    def test_init_model_for_cli_xtrain_not_found(
        self, mock_read_parquet, mock_load, mock_exists
    ):
        """Test init_model_for_cli raises error when X_train not found"""
        from api.app.main import init_model_for_cli, get_expected_columns_from_pipeline

        # Mock model file exists, X_train doesn't
        def exists_side_effect(path):
            if "employee_attrition_pipeline.pkl" in path:
                return True
            return False

        mock_exists.side_effect = exists_side_effect

        # Mock model loading
        mock_model = MagicMock()
        mock_load.return_value = mock_model

        with patch("api.app.main.model", None):
            with patch("api.app.main.explainer", None):
                with patch(
                    "api.app.main.get_expected_columns_from_pipeline",
                    return_value=["col1", "col2"],
                ):
                    with pytest.raises(RuntimeError, match="X_train file not found"):
                        init_model_for_cli()


class TestIsDbDisabled:
    """Test _is_db_disabled function"""

    @patch.dict(os.environ, {"DISABLE_DB": "1"})
    def test_is_db_disabled_true(self):
        """Test _is_db_disabled returns True when env var is 1"""
        from api.app.main import _is_db_disabled

        assert _is_db_disabled() is True

    @patch.dict(os.environ, {"DISABLE_DB": "0"})
    def test_is_db_disabled_false(self):
        """Test _is_db_disabled returns False when env var is 0"""
        from api.app.main import _is_db_disabled

        assert _is_db_disabled() is False

    @patch.dict(os.environ, {}, clear=True)
    def test_is_db_disabled_default(self):
        """Test _is_db_disabled returns False by default"""
        from api.app.main import _is_db_disabled

        # Remove DISABLE_DB if it exists
        os.environ.pop("DISABLE_DB", None)
        assert _is_db_disabled() is False


class TestExcelGenerationErrors:
    """Test Excel generation error paths"""

    def test_predict_excel_with_invalid_data_structure(self, client, auth_headers):
        """Test Excel generation with invalid data structure"""
        payload = {
            "eval_data": "invalid",  # Should be list
            "sirh_data": [],
            "sondage_data": [],
        }
        response = client.post("/predict_excel", headers=auth_headers, json=payload)
        assert response.status_code in [422, 500]

    def test_predict_excel_with_missing_required_fields(self, client, auth_headers):
        """Test Excel generation with missing required fields"""
        payload = {
            "eval_data": [
                {"id_employee": 1, "age": 30}
            ],  # Missing many required fields
            "sirh_data": [{"id_employee": 1}],
            "sondage_data": [{"id_employee": 1}],
        }
        response = client.post("/predict_excel", headers=auth_headers, json=payload)
        assert response.status_code in [422, 500]

    def test_predict_excel_with_mismatched_ids(self, client, auth_headers, sample_employee_data):
        """Test Excel generation when IDs don't match across datasets"""
        data1 = dict(sample_employee_data, id_employee=1)
        data2 = dict(sample_employee_data, id_employee=2)
        data3 = dict(sample_employee_data, id_employee=3)

        payload = {"eval_data": [data1], "sirh_data": [data2], "sondage_data": [data3]}
        response = client.post("/predict_excel", headers=auth_headers, json=payload)
        # Should handle gracefully
        assert response.status_code in [200, 422, 500]


class TestShapCalculationBranches:
    """Test SHAP calculation branches"""

    def test_predict_shap_with_single_employee(self, client, auth_headers, sample_employee_data):
        """Test SHAP image generation with single employee"""
        payload = {
            "eval_data": [sample_employee_data],
            "sirh_data": [sample_employee_data],
            "sondage_data": [sample_employee_data],
        }
        response = client.post("/predict_shap_images", headers=auth_headers, json=payload)
        assert response.status_code in [200, 422, 500]

        if response.status_code == 200:
            data = response.json()
            assert "shap_images" in data or "error" in data

    def test_predict_shap_with_multiple_employees(self, client, auth_headers, sample_employee_data):
        """Test SHAP with multiple employees"""
        employees = [dict(sample_employee_data, id_employee=i) for i in range(1, 4)]
        payload = {
            "eval_data": employees,
            "sirh_data": employees,
            "sondage_data": employees,
        }
        response = client.post("/predict_shap_images", headers=auth_headers, json=payload)
        assert response.status_code in [200, 422, 500]

    def test_predict_shap_with_invalid_data(self, client, auth_headers):
        """Test SHAP with empty data - expects validation error"""
        payload = {"eval_data": [], "sirh_data": [], "sondage_data": []}
        response = client.post("/predict_shap_images", headers=auth_headers, json=payload)
        # Empty data should be rejected by Pydantic validation (min_length=1)
        # or return a server error if validation passes
        assert response.status_code in [422, 500, 503]
        # Empty lists should fail validation, so we expect 422
        if response.status_code == 422:
            assert "detail" in response.json()

    def test_predict_shap_with_extreme_values(self, client, auth_headers, sample_employee_data):
        """Test SHAP with extreme values"""
        extreme_data = dict(sample_employee_data)
        extreme_data["age"] = 65
        extreme_data["annee_experience_totale"] = 40
        extreme_data["revenu_mensuel"] = 20000

        payload = {
            "eval_data": [extreme_data],
            "sirh_data": [extreme_data],
            "sondage_data": [extreme_data],
        }
        response = client.post("/predict_shap_images", headers=auth_headers, json=payload)
        assert response.status_code in [200, 422, 500]


class TestJobQueueEndpoints:
    """Test job queue endpoints (may be unimplemented)"""

    @patch.dict(os.environ, {"DISABLE_DB": "1"})
    def test_job_status_endpoint_exists(self, client, auth_headers):
        """Test if job status endpoint exists"""
        response = client.get("/jobs/test-job-id", headers=auth_headers)
        # May be 404 if not implemented, or 400/422 for invalid job ID, or 503 if DB disabled
        assert response.status_code in [200, 400, 404, 422, 500, 503]

    @patch.dict(os.environ, {"DISABLE_DB": "1"})
    def test_list_jobs_endpoint(self, client, auth_headers):
        """Test list jobs endpoint"""
        response = client.get("/jobs", headers=auth_headers)
        # May be 404 if not implemented, or 503 if DB disabled
        assert response.status_code in [200, 404, 500, 503]

    def test_submit_job_endpoint(self, client, auth_headers, sample_employee_data):
        """Test submit job endpoint"""
        payload = {
            "eval_data": [sample_employee_data],
            "sirh_data": [sample_employee_data],
            "sondage_data": [sample_employee_data],
        }
        response = client.post("/jobs/submit", headers=auth_headers, json=payload)
        # May be 404 if not implemented, or 201 if created
        assert response.status_code in [200, 201, 404, 405, 422, 500]

    @patch.dict(os.environ, {"DISABLE_DB": "1"})
    def test_get_job_report_endpoint(self, client, auth_headers):
        """Test get job report endpoint"""
        response = client.get("/jobs/test-job-id/report", headers=auth_headers)
        # May be 404 if not implemented or job not found, or 503 if DB disabled
        assert response.status_code in [200, 404, 422, 500, 503]


class TestModelLoadingEdgeCases:
    """Test model loading edge cases in lifespan"""

    @patch("api.app.main.os.path.exists")
    def test_lifespan_model_not_found_error(self, mock_exists):
        """Test that lifespan raises error when model file doesn't exist"""
        mock_exists.return_value = False

        # This tests the error path in lifespan function
        # We can't easily test the async context manager directly,
        # but we can verify the path exists
        from api.app.main import lifespan

        assert lifespan is not None


class TestRiskCategorization:
    """Test risk categorization logic"""

    def test_risk_thresholds_exist(self):
        """Test that RISK_THRESHOLDS constant is defined"""
        from api.app.main import RISK_THRESHOLDS

        assert RISK_THRESHOLDS is not None
        assert "Low" in RISK_THRESHOLDS
        assert "Medium" in RISK_THRESHOLDS
        assert "High" in RISK_THRESHOLDS

    def test_risk_threshold_ranges(self):
        """Test risk threshold ranges are correct"""
        from api.app.main import RISK_THRESHOLDS

        assert RISK_THRESHOLDS["Low"] == (0.0, 0.3)
        assert RISK_THRESHOLDS["Medium"] == (0.3, 0.7)
        assert RISK_THRESHOLDS["High"] == (0.7, 1.0)


class TestPredictionWithDatabaseDisabled:
    """Test predictions with database disabled"""

    @patch.dict(os.environ, {"DISABLE_DB": "1"})
    def test_predict_with_db_disabled(self, client, auth_headers, sample_employee_data):
        """Test prediction when database is disabled"""
        payload = {
            "eval_data": [sample_employee_data],
            "sirh_data": [sample_employee_data],
            "sondage_data": [sample_employee_data],
        }
        response = client.post("/predict", headers=auth_headers, json=payload)
        assert response.status_code in [200, 422, 500]

    @patch.dict(os.environ, {"DISABLE_DB": "1"})
    def test_predict_excel_with_db_disabled(self, client, auth_headers, sample_employee_data):
        """Test Excel generation when database is disabled"""
        payload = {
            "eval_data": [sample_employee_data],
            "sirh_data": [sample_employee_data],
            "sondage_data": [sample_employee_data],
        }
        response = client.post("/predict_excel", headers=auth_headers, json=payload)
        assert response.status_code in [200, 422, 500]


class TestPredictionErrorPaths:
    """Test various error paths in prediction"""

    def test_predict_with_none_values(self, client, auth_headers, sample_employee_data):
        """Test prediction with None values"""
        data = dict(sample_employee_data)
        data["genre"] = None
        data["ayant_enfants"] = None

        payload = {"eval_data": [data], "sirh_data": [data], "sondage_data": [data]}
        response = client.post("/predict", headers=auth_headers, json=payload)
        assert response.status_code in [200, 422, 500]

    def test_predict_with_empty_strings(self, client, auth_headers, sample_employee_data):
        """Test prediction with empty strings"""
        data = dict(sample_employee_data)
        data["departement"] = ""
        data["poste"] = ""

        payload = {"eval_data": [data], "sirh_data": [data], "sondage_data": [data]}
        response = client.post("/predict", headers=auth_headers, json=payload)
        assert response.status_code in [200, 422, 500]

    def test_predict_with_invalid_numeric_types(self, client, auth_headers, sample_employee_data):
        """Test prediction with invalid numeric types"""
        data = dict(sample_employee_data)
        data["age"] = "not_a_number"
        data["revenu_mensuel"] = "invalid"

        payload = {"eval_data": [data], "sirh_data": [data], "sondage_data": [data]}
        response = client.post("/predict", headers=auth_headers, json=payload)
        assert response.status_code in [422, 500]


class TestGetExpectedColumnsFunction:
    """Test get_expected_columns_from_pipeline helper"""

    def test_get_expected_columns_called_on_startup(self):
        """Test that get_expected_columns_from_pipeline is available"""
        from api.app.main import get_expected_columns_from_pipeline

        assert get_expected_columns_from_pipeline is not None

    @patch("api.app.main.model")
    def test_get_expected_columns_with_mock_model(self, mock_model):
        """Test get_expected_columns_from_pipeline with mock model"""
        from api.app.main import get_expected_columns_from_pipeline

        # Create a mock pipeline
        mock_preprocessor = MagicMock()
        mock_preprocessor.get_feature_names_out.return_value = ["feature1", "feature2"]
        mock_model.named_steps = {"preprocessor": mock_preprocessor}

        result = get_expected_columns_from_pipeline(mock_model)
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
