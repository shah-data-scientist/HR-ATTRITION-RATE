"""Shared test fixtures and configuration for pytest."""

import os
import sys
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# MOCK dotenv.load_dotenv BEFORE imports that might use it
from unittest.mock import MagicMock
import dotenv

dotenv.load_dotenv = MagicMock()

# Set test environment variables BEFORE importing application modules
# This ensures that when database.database is imported, it picks up these values
import os

os.environ["TESTING"] = "1"  # Signal to main.py to skip loading .env.local
os.environ["API_KEY"] = os.getenv("API_KEY", "test_api_key")
os.environ["SECRET_KEY"] = "test_secret_key_at_least_32_chars_long_for_testing"
# Force SQLite for ALL tests, overriding any .env files
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["DISABLE_DB"] = "0"  # Enable DB for tests

from database.database import Base
from database.models import Employee, ModelInput, ModelOutput, PredictionTraceability


@pytest.fixture(scope="session")
def sample_raw_employee_data():
    """Sample raw employee data (before cleaning)."""
    return {
        "id_employee": 99999,
        "eval_number": "E_99999",
        "age": 35,
        "genre": "Homme",  # RAW format
        "revenu_mensuel": 6000.0,
        "heure_supplementaires": "Non",  # RAW format
        "augementation_salaire_precedente": 7.0,  # 0-35 range
        "statut_marital": "Marié",
        "departement": "IT",
        "poste": "Développeur",
        "nombre_experiences_precedentes": 2,
        "nombre_heures_travailless": 190,
        "annee_experience_totale": 10,
        "annees_dans_l_entreprise": 5,
        "annees_dans_le_poste_actuel": 3,
        "nombre_participation_pee": 1,
        "nb_formations_suivies": 2,
        "nombre_employee_sous_responsabilite": 0,
        "code_sondage": "80",
        "distance_domicile_travail": 15,
        "niveau_education": 3,
        "domaine_etude": "Informatique",
        "ayant_enfants": "Oui",
        "frequence_deplacement": "Rarement",
        "annees_depuis_la_derniere_promotion": 2,
        "annes_sous_responsable_actuel": 2,
        "satisfaction_employee_environnement": 3,
        "note_evaluation_precedente": 3,
        "niveau_hierarchique_poste": 2,
        "satisfaction_employee_nature_travail": 4,
        "satisfaction_employee_equipe": 3,
        "satisfaction_employee_equilibre_pro_perso": 3,
        "note_evaluation_actuelle": 4,
    }


@pytest.fixture(scope="session")
def sample_raw_employee_df(sample_raw_employee_data):
    """Sample raw employee DataFrame."""
    return pd.DataFrame([sample_raw_employee_data])


@pytest.fixture(scope="function")
def test_db_engine():
    """Create an in-memory SQLite database for testing."""
    # Use in-memory SQLite for fast, isolated tests
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture(scope="function")
def test_db_session(test_db_engine):
    """Create a test database session."""
    TestingSessionLocal = sessionmaker(
        autocommit=False, autoflush=False, bind=test_db_engine
    )
    session = TestingSessionLocal()
    yield session
    session.close()


@pytest.fixture(scope="session")
def api_headers():
    """Headers for authenticated API requests."""
    return {
        "X-API-Key": os.getenv("API_KEY", "test_api_key"),
        "Content-Type": "application/json",
    }


@pytest.fixture(scope="session")
def api_base_url():
    """Base URL for API requests."""
    return os.getenv("API_BASE_URL", "http://localhost:8001")


@pytest.fixture(autouse=True)
def init_sqlite_tables():
    """Initialize tables if using SQLite in-memory database."""
    from database.database import engine, Base

    # Ensure models are imported so they are registered in Base.metadata
    import database.models

    # Check if using SQLite (in-memory or file)
    if str(engine.url).startswith("sqlite"):
        print(f"DEBUG: Creating tables on {engine.url}")
        print(f"DEBUG: Tables in metadata: {list(Base.metadata.tables.keys())}")
        # Create tables
        Base.metadata.create_all(bind=engine)
        yield
        # Drop tables after test
        Base.metadata.drop_all(bind=engine)
    else:
        yield


@pytest.fixture(scope="function")
def sample_employee_in_db(test_db_session):
    """Create a sample employee record in the test database."""
    from datetime import datetime

    employee = Employee(
        id_employee=10001,
        age=35,
        genre=1,
        revenu_mensuel=6000,
        departement="IT",
        poste="Développeur",
        statut_marital="Marié",
        niveau_education=3,
        domaine_etude="Informatique",
        ayant_enfants="Oui",
        frequence_deplacement="Rarement",
        annee_experience_totale=10,
        annees_dans_l_entreprise=5,
        annees_dans_le_poste_actuel=3,
        nombre_experiences_precedentes=2,
        nombre_participation_pee=1,
        nb_formations_suivies=2,
        nombre_employee_sous_responsabilite=0,
        distance_domicile_travail=15,
        annees_depuis_la_derniere_promotion=2,
        annes_sous_responsable_actuel=2,
        satisfaction_employee_environnement=3.0,
        note_evaluation_precedente=3.0,
        niveau_hierarchique_poste=2.0,
        satisfaction_employee_nature_travail=4.0,
        satisfaction_employee_equipe=3.0,
        satisfaction_employee_equilibre_pro_perso=3.0,
        note_evaluation_actuelle=4.0,
        heures_supplementaires=0,
        augmentation_salaire_precedente=0.07,
        improvement_evaluation=1.0,
        total_satisfaction=36.0,
        work_mobility=0.6,
        user_id="demo1",
        date_ingestion=datetime.now(),
    )
    test_db_session.add(employee)
    test_db_session.commit()
    test_db_session.refresh(employee)
    return employee
