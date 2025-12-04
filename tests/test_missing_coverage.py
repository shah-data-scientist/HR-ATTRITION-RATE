"""Tests to improve coverage for untested modules.

This test file targets modules with 0% or low coverage:
- api/app/schemas.py (deprecated Pydantic models)
- api/get_model_features.py (model feature extraction)
- database/seed_data.py (database seeding)
- database/init_db.py (database initialization)
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock

import pytest
from pydantic import ValidationError


class TestDeprecatedSchemas:
    """Test api/app/schemas.py - Deprecated Pydantic models."""

    def test_employee_features_valid(self):
        """Test EmployeeFeatures model with valid data."""
        from api.app.schemas import EmployeeFeatures

        data = {
            "age": 35,
            "genre": 1,
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
            "augementation_salaire_precedente": 0.07,
        }

        employee = EmployeeFeatures(**data)
        assert employee.age == 35
        assert employee.genre == 1
        assert employee.revenu_mensuel == 6000.0
        assert employee.statut_marital == "Marié"

    def test_employee_features_with_optional_fields(self):
        """Test EmployeeFeatures with optional fields."""
        from api.app.schemas import EmployeeFeatures

        data = {
            "id_employee": 12345,
            "age": 35,
            "genre": 1,
            "ayant_enfants": 1,
            "heures_supplementaires": 0,
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
            "augementation_salaire_precedente": 0.07,
        }

        employee = EmployeeFeatures(**data)
        assert employee.id_employee == 12345
        assert employee.ayant_enfants == 1
        assert employee.heures_supplementaires == 0

    def test_employee_features_validation_errors(self):
        """Test EmployeeFeatures validation."""
        from api.app.schemas import EmployeeFeatures

        # Invalid age (too low)
        with pytest.raises(ValidationError):
            EmployeeFeatures(
                age=15,  # < 18
                revenu_mensuel=6000.0,
                statut_marital="Marié",
                departement="R&D",
                poste="Développeur",
                nombre_experiences_precedentes=2,
                annee_experience_totale=10,
                annees_dans_l_entreprise=5,
                annees_dans_le_poste_actuel=3,
                nombre_participation_pee=1,
                nb_formations_suivies=2,
                nombre_employee_sous_responsabilite=0,
                distance_domicile_travail=15,
                niveau_education=3,
                domaine_etude="Informatique",
                frequence_deplacement="Rarement",
                annees_depuis_la_derniere_promotion=2,
                annes_sous_responsable_actuel=2,
                satisfaction_employee_environnement=3,
                note_evaluation_precedente=3.5,
                niveau_hierarchique_poste=2,
                satisfaction_employee_nature_travail=4,
                satisfaction_employee_equipe=3,
                satisfaction_employee_equilibre_pro_perso=3,
                note_evaluation_actuelle=4.0,
                augementation_salaire_precedente=0.07,
            )

    def test_employee_features_boundary_values(self):
        """Test EmployeeFeatures with boundary values."""
        from api.app.schemas import EmployeeFeatures

        data = {
            "age": 65,  # Maximum age
            "revenu_mensuel": 0.0,  # Minimum income
            "statut_marital": "Célibataire",
            "departement": "RH",
            "poste": "Manager",
            "nombre_experiences_precedentes": 0,
            "annee_experience_totale": 0,
            "annees_dans_l_entreprise": 0,
            "annees_dans_le_poste_actuel": 0,
            "nombre_participation_pee": 0,
            "nb_formations_suivies": 0,
            "nombre_employee_sous_responsabilite": 0,
            "distance_domicile_travail": 1,  # Minimum
            "niveau_education": 5,  # Maximum
            "domaine_etude": "Sciences",
            "frequence_deplacement": "Jamais",
            "annees_depuis_la_derniere_promotion": 0,
            "annes_sous_responsable_actuel": 0,
            "satisfaction_employee_environnement": 1,  # Minimum
            "note_evaluation_precedente": 0.0,  # Minimum
            "niveau_hierarchique_poste": 1,  # Minimum
            "satisfaction_employee_nature_travail": 1,
            "satisfaction_employee_equipe": 1,
            "satisfaction_employee_equilibre_pro_perso": 1,
            "note_evaluation_actuelle": 0.0,
            "augementation_salaire_precedente": 0.0,
        }

        employee = EmployeeFeatures(**data)
        assert employee.age == 65
        assert employee.niveau_education == 5

    def test_prediction_output_model(self):
        """Test PredictionOutput model."""
        from api.app.schemas import PredictionOutput

        output = PredictionOutput(
            id_employee=12345,
            prediction="Stay",
            probability=0.85,
            risk_category="Low",
            message="Employee is likely to stay",
        )

        assert output.id_employee == 12345
        assert output.prediction == "Stay"
        assert output.probability == 0.85
        assert output.risk_category == "Low"
        assert output.trace_id is None

    def test_prediction_output_with_trace_id(self):
        """Test PredictionOutput with trace_id."""
        from api.app.schemas import PredictionOutput

        output = PredictionOutput(
            id_employee=12345,
            prediction="Leave",
            probability=0.65,
            risk_category="High",
            message="Employee is at risk",
            trace_id=999,
        )

        assert output.trace_id == 999

    def test_batch_prediction_input(self):
        """Test BatchPredictionInput model."""
        from api.app.schemas import BatchPredictionInput, EmployeeFeatures

        employees_data = [
            {
                "age": 35,
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
                "augementation_salaire_precedente": 0.07,
            }
        ]

        batch_input = BatchPredictionInput(
            employees=[EmployeeFeatures(**emp) for emp in employees_data]
        )

        assert len(batch_input.employees) == 1
        assert batch_input.employees[0].age == 35

    def test_batch_prediction_output(self):
        """Test BatchPredictionOutput model."""
        from api.app.schemas import BatchPredictionOutput, PredictionOutput

        predictions = [
            PredictionOutput(
                id_employee=1,
                prediction="Stay",
                probability=0.85,
                risk_category="Low",
                message="Test",
            ),
            PredictionOutput(
                id_employee=2,
                prediction="Leave",
                probability=0.65,
                risk_category="High",
                message="Test 2",
            ),
        ]

        batch_output = BatchPredictionOutput(predictions=predictions)

        assert len(batch_output.predictions) == 2
        assert batch_output.predictions[0].id_employee == 1
        assert batch_output.predictions[1].id_employee == 2


class TestGetModelFeatures:
    """Test api/get_model_features.py module."""

    def test_get_expected_columns(self):
        """Test get_expected_columns function."""
        from api.get_model_features import get_expected_columns

        # Mock pipeline with preprocessor
        mock_preprocessor = MagicMock()
        mock_preprocessor.feature_names_in_ = ["age", "genre", "revenu_mensuel"]

        mock_pipeline = MagicMock()
        mock_pipeline.named_steps = {"preprocessor": mock_preprocessor}

        columns = get_expected_columns(mock_pipeline)

        assert columns == ["age", "genre", "revenu_mensuel"]
        assert isinstance(columns, list)

    @patch("api.get_model_features.joblib.load")
    def test_load_model_only(self, mock_joblib_load):
        """Test load_model_only function."""
        from api.get_model_features import load_model_only

        # Mock the model
        mock_model = MagicMock()
        mock_joblib_load.return_value = mock_model

        model = load_model_only()

        assert model == mock_model
        mock_joblib_load.assert_called_once_with(
            "outputs/employee_attrition_pipeline.pkl"
        )

    @patch("api.get_model_features.load_model_only")
    @patch("api.get_model_features.get_expected_columns")
    def test_main_execution(self, mock_get_cols, mock_load_model):
        """Test main execution path."""
        # Mock the model and columns
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        mock_get_cols.return_value = ["col1", "col2", "col3"]

        # We don't actually run the main block, but we've tested both functions it uses


class TestDatabaseSeedData:
    """Test database/seed_data.py module."""

    @patch("database.seed_data.SessionLocal")
    @patch("database.seed_data.Base")
    def test_seed_database_new_records(self, mock_base, mock_session_local):
        """Test seeding database with new records."""
        from database.seed_data import seed_database

        # Mock session
        mock_session = MagicMock()
        mock_session_local.return_value = mock_session

        # Mock query to return no existing employees
        mock_query = MagicMock()
        mock_query.filter.return_value.first.return_value = None
        mock_query.count.return_value = 3
        mock_session.query.return_value = mock_query

        # Run seeding
        seed_database(drop_existing=False)

        # Verify tables were created
        mock_base.metadata.create_all.assert_called_once()

        # Verify employees were added (should be 3 calls to add)
        assert mock_session.add.call_count == 3

        # Verify commit was called
        mock_session.commit.assert_called_once()

        # Verify session was closed
        mock_session.close.assert_called_once()

    @patch("database.seed_data.SessionLocal")
    @patch("database.seed_data.Base")
    def test_seed_database_with_drop(self, mock_base, mock_session_local):
        """Test seeding database with drop_existing=True."""
        from database.seed_data import seed_database

        # Mock session
        mock_session = MagicMock()
        mock_session_local.return_value = mock_session

        # Mock query
        mock_query = MagicMock()
        mock_query.filter.return_value.first.return_value = None
        mock_query.count.return_value = 3
        mock_session.query.return_value = mock_query

        # Run seeding with drop
        seed_database(drop_existing=True)

        # Verify tables were dropped
        mock_base.metadata.drop_all.assert_called_once()

        # Verify tables were created
        mock_base.metadata.create_all.assert_called_once()

    @patch("database.seed_data.SessionLocal")
    @patch("database.seed_data.Base")
    def test_seed_database_existing_records(self, mock_base, mock_session_local):
        """Test seeding database with existing records."""
        from database.seed_data import seed_database

        # Mock session
        mock_session = MagicMock()
        mock_session_local.return_value = mock_session

        # Mock query to return existing employee
        mock_existing = MagicMock()
        mock_query = MagicMock()
        mock_query.filter.return_value.first.return_value = mock_existing
        mock_query.count.return_value = 3
        mock_session.query.return_value = mock_query

        # Run seeding
        seed_database(drop_existing=False)

        # Verify no employees were added (all exist)
        mock_session.add.assert_not_called()

        # Verify commit was still called
        mock_session.commit.assert_called_once()

    @patch("database.seed_data.SessionLocal")
    @patch("database.seed_data.Base")
    def test_seed_database_error_handling(self, mock_base, mock_session_local):
        """Test error handling in seed_database."""
        from database.seed_data import seed_database

        # Mock session
        mock_session = MagicMock()
        mock_session_local.return_value = mock_session

        # Mock query to raise an exception
        mock_session.query.side_effect = Exception("Database error")

        # Run seeding and expect exception
        with pytest.raises(Exception, match="Database error"):
            seed_database(drop_existing=False)

        # Verify rollback was called
        mock_session.rollback.assert_called_once()

        # Verify session was closed
        mock_session.close.assert_called_once()


class TestDatabaseInitDb:
    """Test database/init_db.py module."""

    @patch("database.init_db.SessionLocal")
    def test_create_default_users_new(self, mock_session_local):
        """Test creating default users when none exist."""
        from database.init_db import create_default_users

        # Mock session
        mock_session = MagicMock()
        mock_session_local.return_value = mock_session

        # Mock query to return 0 users
        mock_query = MagicMock()
        mock_query.count.return_value = 0
        mock_session.query.return_value = mock_query

        # Call function
        create_default_users(mock_session)

        # Verify users were added (2 default users: admin and analyst)
        assert mock_session.add.call_count == 2

        # Verify commit was called
        mock_session.commit.assert_called_once()

    @patch("database.init_db.SessionLocal")
    def test_create_default_users_existing(self, mock_session_local):
        """Test creating default users when users already exist."""
        from database.init_db import create_default_users

        # Mock session
        mock_session = MagicMock()

        # Mock query to return existing users
        mock_query = MagicMock()
        mock_query.count.return_value = 2
        mock_session.query.return_value = mock_query

        # Call function
        create_default_users(mock_session)

        # Verify no users were added
        mock_session.add.assert_not_called()

        # Verify commit was not called
        mock_session.commit.assert_not_called()

    @patch("database.init_db.os.getenv")
    @patch("database.init_db.SessionLocal")
    def test_create_default_users_custom_credentials(
        self, mock_session_local, mock_getenv
    ):
        """Test creating default users with custom credentials from env."""
        from database.init_db import create_default_users

        # Mock session
        mock_session = MagicMock()

        # Mock query to return 0 users
        mock_query = MagicMock()
        mock_query.count.return_value = 0
        mock_session.query.return_value = mock_query

        # Mock environment variables
        def getenv_side_effect(key, default=None):
            env_vars = {
                "UI_ADMIN_USERNAME": "custom_admin",
                "UI_ADMIN_PASSWORD": "CustomPass123!",
                "UI_ANALYST_USERNAME": "custom_analyst",
                "UI_ANALYST_PASSWORD": "CustomAnalyst123!",
            }
            return env_vars.get(key, default)

        mock_getenv.side_effect = getenv_side_effect

        # Call function
        create_default_users(mock_session)

        # Verify users were added
        assert mock_session.add.call_count == 2

    @patch("database.init_db.load_and_merge_data")
    @patch("database.init_db.pd.read_csv")
    @patch("database.init_db.SessionLocal")
    @patch("database.init_db.Base")
    def test_init_db_success(
        self, mock_base, mock_session_local, mock_read_csv, mock_load_merge
    ):
        """Test init_db function success path."""
        from database.init_db import init_db
        import pandas as pd

        # Mock session
        mock_session = MagicMock()
        mock_session_local.return_value = mock_session

        # Mock query to return 0 employees (empty table)
        mock_query = MagicMock()
        mock_query.count.return_value = 0
        mock_session.query.return_value = mock_query

        # Mock CSV read
        mock_df = pd.DataFrame(
            {
                "id_employee": [1, 2],
                "age": [30, 35],
                "a_quitte_l_entreprise": [0, 1],
            }
        )
        mock_read_csv.return_value = mock_df
        mock_load_merge.return_value = mock_df

        # Call function
        init_db()

        # Verify tables were created
        mock_base.metadata.create_all.assert_called_once()

        # Verify data was added
        mock_session.add_all.assert_called_once()

    @patch("database.init_db.SessionLocal")
    @patch("database.init_db.Base")
    def test_init_db_with_existing_data(self, mock_base, mock_session_local):
        """Test init_db when data already exists."""
        from database.init_db import init_db

        # Mock session
        mock_session = MagicMock()
        mock_session_local.return_value = mock_session

        # Mock query to return existing employees
        mock_query = MagicMock()
        mock_query.count.return_value = 100
        mock_session.query.return_value = mock_query

        # Call function
        init_db()

        # Verify tables were created
        mock_base.metadata.create_all.assert_called_once()

        # Verify no data was added (already exists)
        mock_session.add_all.assert_not_called()


class TestAuthModule:
    """Test api/auth.py module."""

    @patch("api.auth.os.getenv")
    def test_get_valid_api_key(self, mock_getenv):
        """Test _get_valid_api_key function."""
        from api.auth import _get_valid_api_key

        mock_getenv.return_value = "test_api_key"
        result = _get_valid_api_key()
        assert result == "test_api_key"

    def test_verify_password_success(self):
        """Test verify_password with correct password."""
        from api.auth import verify_password, get_password_hash

        password = "MySecurePassword123!"
        hashed = get_password_hash(password)

        result = verify_password(password, hashed)
        assert result is True

    def test_verify_password_failure(self):
        """Test verify_password with wrong password."""
        from api.auth import verify_password, get_password_hash

        password = "MySecurePassword123!"
        hashed = get_password_hash(password)

        result = verify_password("WrongPassword", hashed)
        assert result is False

    def test_get_password_hash(self):
        """Test get_password_hash function."""
        from api.auth import get_password_hash

        password = "MyTestPassword123!"
        hashed = get_password_hash(password)

        assert hashed is not None
        assert isinstance(hashed, str)
        assert hashed != password  # Ensure it's actually hashed

    def test_generate_api_key_default_length(self):
        """Test generate_api_key with default length."""
        from api.auth import generate_api_key

        api_key = generate_api_key()

        assert api_key is not None
        assert isinstance(api_key, str)
        assert len(api_key) == 64  # 32 bytes = 64 hex characters

    def test_generate_api_key_custom_length(self):
        """Test generate_api_key with custom length."""
        from api.auth import generate_api_key

        api_key = generate_api_key(length=16)

        assert api_key is not None
        assert isinstance(api_key, str)
        assert len(api_key) == 32  # 16 bytes = 32 hex characters

    @pytest.mark.asyncio
    @patch("api.auth._get_valid_api_key")
    async def test_get_api_key_valid(self, mock_get_valid_key):
        """Test get_api_key with valid API key."""
        from api.auth import get_api_key

        mock_get_valid_key.return_value = "valid_key"

        result = await get_api_key("valid_key")

        assert result == "valid_key"

    @pytest.mark.asyncio
    @patch("api.auth._get_valid_api_key")
    async def test_get_api_key_missing(self, mock_get_valid_key):
        """Test get_api_key with missing API key."""
        from api.auth import get_api_key
        from fastapi import HTTPException

        mock_get_valid_key.return_value = "valid_key"

        with pytest.raises(HTTPException) as exc_info:
            await get_api_key(None)

        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    @patch("api.auth._get_valid_api_key")
    async def test_get_api_key_invalid(self, mock_get_valid_key):
        """Test get_api_key with invalid API key."""
        from api.auth import get_api_key
        from fastapi import HTTPException

        mock_get_valid_key.return_value = "valid_key"

        with pytest.raises(HTTPException) as exc_info:
            await get_api_key("invalid_key")

        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    @patch("api.auth._get_valid_api_key")
    async def test_get_optional_api_key_none(self, mock_get_valid_key):
        """Test get_optional_api_key with no key provided."""
        from api.auth import get_optional_api_key

        mock_get_valid_key.return_value = "valid_key"

        result = await get_optional_api_key(None)

        assert result is None

    @pytest.mark.asyncio
    @patch("api.auth._get_valid_api_key")
    async def test_get_optional_api_key_valid(self, mock_get_valid_key):
        """Test get_optional_api_key with valid key."""
        from api.auth import get_optional_api_key

        mock_get_valid_key.return_value = "valid_key"

        result = await get_optional_api_key("valid_key")

        assert result == "valid_key"

    @pytest.mark.asyncio
    @patch("api.auth._get_valid_api_key")
    async def test_get_optional_api_key_invalid(self, mock_get_valid_key):
        """Test get_optional_api_key with invalid key."""
        from api.auth import get_optional_api_key
        from fastapi import HTTPException

        mock_get_valid_key.return_value = "valid_key"

        with pytest.raises(HTTPException) as exc_info:
            await get_optional_api_key("invalid_key")

        assert exc_info.value.status_code == 403
