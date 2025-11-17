"""Deep coverage tests for predict_report, predict_excel, predict_shap_images with DB enabled"""
import pytest
from fastapi.testclient import TestClient
from api.app.main import app
from database.database import Base, engine, DISABLE_DB
from sqlalchemy.orm import Session
import json


@pytest.fixture(scope="function", autouse=True)
def setup_database():
    """Setup database for tests"""
    if not DISABLE_DB:
        Base.metadata.create_all(bind=engine)
    yield
    if not DISABLE_DB:
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def valid_employee_data():
    """Valid employee data for predictions"""
    return {
        "sirh_data": [{
            "id_employee": 90001,
            "age": 35,
            "genre": "M",
            "revenu_mensuel": 6000,
            "statut_marital": "Marié",
            "departement": "IT",
            "poste": "Developer",
            "nombre_experiences_precedentes": 2,
            "nombre_heures_travailless": 180,
            "annee_experience_totale": 10,
            "annees_dans_l_entreprise": 5,
            "annees_dans_le_poste_actuel": 3
        }],
        "eval_data": [{
            "eval_number": "E_90001",
            "note_evaluation_precedente": 3,
            "note_evaluation_actuelle": 4,
            "augementation_salaire_precedente": "15 %",
            "niveau_hierarchique_poste": 2,
            "satisfaction_employee_environnement": 4,
            "satisfaction_employee_nature_travail": 4,
            "satisfaction_employee_equipe": 4,
            "satisfaction_employee_equilibre_pro_perso": 3,
            "heure_supplementaires": "Non",
            "annees_depuis_la_derniere_promotion": 2,
            "annes_sous_responsable_actuel": 2
        }],
        "sondage_data": [{
            "code_sondage": 90001,
            "nombre_participation_pee": 1,
            "nb_formations_suivies": 3,
            "nombre_employee_sous_responsabilite": 0,
            "distance_domicile_travail": 10,
            "niveau_education": 4,
            "domaine_etude": "Informatique",
            "ayant_enfants": "oui",
            "frequence_deplacement": "Rarement",
            "annees_depuis_la_derniere_promotion": 2,
            "annes_sous_responsable_actuel": 2
        }]
    }


@pytest.mark.skipif(DISABLE_DB, reason="Requires database enabled")
class TestPredictReportFullCoverage:
    """Test predict_report endpoint for full code coverage"""
    
    def test_predict_report_full_flow(self, client, valid_employee_data):
        """Test predict_report with all sheets generated"""
        response = client.post("/predict_report", json=valid_employee_data)
        assert response.status_code == 200
        data = response.json()
        
        # Check predictions
        assert "predictions" in data
        assert len(data["predictions"]) == 1
        
        # Check Excel generation
        assert "excel_base64" in data
        assert data["excel_base64"] is not None
        assert len(data["excel_base64"]) > 0
        
        # Check SHAP images
        assert "shap_images" in data
        assert data["shap_images"] is not None
        assert len(data["shap_images"]) > 0
    
    def test_predict_report_multiple_employees_with_shap(self, client):
        """Test predict_report with multiple employees to cover concatenation logic"""
        data = {
            "sirh_data": [
                {
                    "id_employee": 90002,
                    "age": 28,
                    "genre": "F",
                    "revenu_mensuel": 5000,
                    "statut_marital": "Célibataire",
                    "departement": "Sales",
                    "poste": "Sales Rep",
                    "nombre_experiences_precedentes": 1,
                    "nombre_heures_travailless": 160,
                    "annee_experience_totale": 5,
                    "annees_dans_l_entreprise": 2,
                    "annees_dans_le_poste_actuel": 1
                },
                {
                    "id_employee": 90003,
                    "age": 45,
                    "genre": "M",
                    "revenu_mensuel": 8000,
                    "statut_marital": "Marié",
                    "departement": "Management",
                    "poste": "Manager",
                    "nombre_experiences_precedentes": 3,
                    "nombre_heures_travailless": 200,
                    "annee_experience_totale": 20,
                    "annees_dans_l_entreprise": 10,
                    "annees_dans_le_poste_actuel": 5
                },
                {
                    "id_employee": 90004,
                    "age": 32,
                    "genre": "F",
                    "revenu_mensuel": 6500,
                    "statut_marital": "Marié",
                    "departement": "IT",
                    "poste": "Developer",
                    "nombre_experiences_precedentes": 2,
                    "nombre_heures_travailless": 175,
                    "annee_experience_totale": 8,
                    "annees_dans_l_entreprise": 4,
                    "annees_dans_le_poste_actuel": 2
                }
            ],
            "eval_data": [
                {
                    "eval_number": "E_90002",
                    "note_evaluation_precedente": 3,
                    "note_evaluation_actuelle": 3,
                    "augementation_salaire_precedente": "12 %",
                    "niveau_hierarchique_poste": 1,
                    "satisfaction_employee_environnement": 3,
                    "satisfaction_employee_nature_travail": 3,
                    "satisfaction_employee_equipe": 3,
                    "satisfaction_employee_equilibre_pro_perso": 3,
                    "heure_supplementaires": "Oui",
                    "annees_depuis_la_derniere_promotion": 1,
                    "annes_sous_responsable_actuel": 1
                },
                {
                    "eval_number": "E_90003",
                    "note_evaluation_precedente": 4,
                    "note_evaluation_actuelle": 4,
                    "augementation_salaire_precedente": "18 %",
                    "niveau_hierarchique_poste": 3,
                    "satisfaction_employee_environnement": 4,
                    "satisfaction_employee_nature_travail": 4,
                    "satisfaction_employee_equipe": 4,
                    "satisfaction_employee_equilibre_pro_perso": 4,
                    "heure_supplementaires": "Non",
                    "annees_depuis_la_derniere_promotion": 3,
                    "annes_sous_responsable_actuel": 3
                },
                {
                    "eval_number": "E_90004",
                    "note_evaluation_precedente": 4,
                    "note_evaluation_actuelle": 4,
                    "augementation_salaire_precedente": "16 %",
                    "niveau_hierarchique_poste": 2,
                    "satisfaction_employee_environnement": 4,
                    "satisfaction_employee_nature_travail": 4,
                    "satisfaction_employee_equipe": 4,
                    "satisfaction_employee_equilibre_pro_perso": 4,
                    "heure_supplementaires": "Non",
                    "annees_depuis_la_derniere_promotion": 2,
                    "annes_sous_responsable_actuel": 2
                }
            ],
            "sondage_data": [
                {
                    "code_sondage": 90002,
                    "nombre_participation_pee": 0,
                    "nb_formations_suivies": 1,
                    "nombre_employee_sous_responsabilite": 0,
                    "distance_domicile_travail": 25,
                    "niveau_education": 3,
                    "domaine_etude": "Commerce",
                    "ayant_enfants": "non",
                    "frequence_deplacement": "Frequent",
                    "annees_depuis_la_derniere_promotion": 1,
                    "annes_sous_responsable_actuel": 1
                },
                {
                    "code_sondage": 90003,
                    "nombre_participation_pee": 1,
                    "nb_formations_suivies": 5,
                    "nombre_employee_sous_responsabilite": 5,
                    "distance_domicile_travail": 5,
                    "niveau_education": 5,
                    "domaine_etude": "Management",
                    "ayant_enfants": "oui",
                    "frequence_deplacement": "Rarement",
                    "annees_depuis_la_derniere_promotion": 3,
                    "annes_sous_responsable_actuel": 3
                },
                {
                    "code_sondage": 90004,
                    "nombre_participation_pee": 1,
                    "nb_formations_suivies": 3,
                    "nombre_employee_sous_responsabilite": 0,
                    "distance_domicile_travail": 15,
                    "niveau_education": 4,
                    "domaine_etude": "Informatique",
                    "ayant_enfants": "oui",
                    "frequence_deplacement": "Rarement",
                    "annees_depuis_la_derniere_promotion": 2,
                    "annes_sous_responsable_actuel": 2
                }
            ]
        }
        response = client.post("/predict_report", json=data)
        assert response.status_code == 200
        result = response.json()
        
        # Verify multiple predictions
        assert len(result["predictions"]) == 3
        
        # Verify Excel contains data
        assert result["excel_base64"] is not None
        
        # Verify SHAP images for all employees
        assert len(result["shap_images"]) == 3
        for img in result["shap_images"]:
            assert len(img) > 0


@pytest.mark.skipif(DISABLE_DB, reason="Requires database enabled")
class TestPredictExcelFullCoverage:
    """Test predict_excel endpoint for full code coverage"""
    
    def test_predict_excel_single_employee(self, client, valid_employee_data):
        """Test Excel generation with single employee"""
        response = client.post("/predict_excel", json=valid_employee_data)
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        assert len(response.content) > 0
    
    def test_predict_excel_multiple_employees(self, client):
        """Test Excel with multiple employees for concatenation coverage"""
        data = {
            "sirh_data": [
                {"id_employee": 91001, "age": 30, "genre": "M", "revenu_mensuel": 5500,
                 "statut_marital": "Célibataire", "departement": "IT", "poste": "Dev",
                 "nombre_experiences_precedentes": 1, "nombre_heures_travailless": 170,
                 "annee_experience_totale": 7, "annees_dans_l_entreprise": 3,
                 "annees_dans_le_poste_actuel": 2},
                {"id_employee": 91002, "age": 40, "genre": "F", "revenu_mensuel": 7000,
                 "statut_marital": "Marié", "departement": "HR", "poste": "Manager",
                 "nombre_experiences_precedentes": 2, "nombre_heures_travailless": 180,
                 "annee_experience_totale": 15, "annees_dans_l_entreprise": 8,
                 "annees_dans_le_poste_actuel": 4}
            ],
            "eval_data": [
                {"eval_number": "E_91001", "note_evaluation_precedente": 3,
                 "note_evaluation_actuelle": 3, "augementation_salaire_precedente": "14 %",
                 "niveau_hierarchique_poste": 2, "satisfaction_employee_environnement": 3,
                 "satisfaction_employee_nature_travail": 3, "satisfaction_employee_equipe": 3,
                 "satisfaction_employee_equilibre_pro_perso": 3, "heure_supplementaires": "Non",
                 "annees_depuis_la_derniere_promotion": 1, "annes_sous_responsable_actuel": 1},
                {"eval_number": "E_91002", "note_evaluation_precedente": 4,
                 "note_evaluation_actuelle": 4, "augementation_salaire_precedente": "16 %",
                 "niveau_hierarchique_poste": 3, "satisfaction_employee_environnement": 4,
                 "satisfaction_employee_nature_travail": 4, "satisfaction_employee_equipe": 4,
                 "satisfaction_employee_equilibre_pro_perso": 4, "heure_supplementaires": "Non",
                 "annees_depuis_la_derniere_promotion": 2, "annes_sous_responsable_actuel": 2}
            ],
            "sondage_data": [
                {"code_sondage": 91001, "nombre_participation_pee": 1, "nb_formations_suivies": 2,
                 "nombre_employee_sous_responsabilite": 0, "distance_domicile_travail": 12,
                 "niveau_education": 4, "domaine_etude": "Informatique", "ayant_enfants": "non",
                 "frequence_deplacement": "Rarement", "annees_depuis_la_derniere_promotion": 1,
                 "annes_sous_responsable_actuel": 1},
                {"code_sondage": 91002, "nombre_participation_pee": 1, "nb_formations_suivies": 4,
                 "nombre_employee_sous_responsabilite": 3, "distance_domicile_travail": 8,
                 "niveau_education": 5, "domaine_etude": "RH", "ayant_enfants": "oui",
                 "frequence_deplacement": "Occasionnel", "annees_depuis_la_derniere_promotion": 2,
                 "annes_sous_responsable_actuel": 2}
            ]
        }
        response = client.post("/predict_excel", json=data)
        assert response.status_code == 200
        assert len(response.content) > 5000  # Should be a substantial Excel file


@pytest.mark.skipif(DISABLE_DB, reason="Requires database enabled")
class TestPredictShapImagesFullCoverage:
    """Test predict_shap_images endpoint for full code coverage"""
    
    def test_predict_shap_images_single(self, client, valid_employee_data):
        """Test SHAP images generation for single employee"""
        response = client.post("/predict_shap_images", json=valid_employee_data)
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/zip"
        assert len(response.content) > 0
    
    def test_predict_shap_images_multiple_employees(self, client):
        """Test SHAP images with multiple employees"""
        data = {
            "sirh_data": [
                {"id_employee": 92001, "age": 32, "genre": "F", "revenu_mensuel": 5800,
                 "statut_marital": "Célibataire", "departement": "Sales", "poste": "Rep",
                 "nombre_experiences_precedentes": 2, "nombre_heures_travailless": 165,
                 "annee_experience_totale": 8, "annees_dans_l_entreprise": 4,
                 "annees_dans_le_poste_actuel": 2},
                {"id_employee": 92002, "age": 38, "genre": "M", "revenu_mensuel": 6500,
                 "statut_marital": "Marié", "departement": "IT", "poste": "Lead",
                 "nombre_experiences_precedentes": 3, "nombre_heures_travailless": 190,
                 "annee_experience_totale": 12, "annees_dans_l_entreprise": 6,
                 "annees_dans_le_poste_actuel": 3}
            ],
            "eval_data": [
                {"eval_number": "E_92001", "note_evaluation_precedente": 3,
                 "note_evaluation_actuelle": 4, "augementation_salaire_precedente": "13 %",
                 "niveau_hierarchique_poste": 2, "satisfaction_employee_environnement": 3,
                 "satisfaction_employee_nature_travail": 4, "satisfaction_employee_equipe": 3,
                 "satisfaction_employee_equilibre_pro_perso": 3, "heure_supplementaires": "Oui",
                 "annees_depuis_la_derniere_promotion": 1, "annes_sous_responsable_actuel": 2},
                {"eval_number": "E_92002", "note_evaluation_precedente": 4,
                 "note_evaluation_actuelle": 4, "augementation_salaire_precedente": "17 %",
                 "niveau_hierarchique_poste": 3, "satisfaction_employee_environnement": 4,
                 "satisfaction_employee_nature_travail": 4, "satisfaction_employee_equipe": 4,
                 "satisfaction_employee_equilibre_pro_perso": 4, "heure_supplementaires": "Non",
                 "annees_depuis_la_derniere_promotion": 2, "annes_sous_responsable_actuel": 3}
            ],
            "sondage_data": [
                {"code_sondage": 92001, "nombre_participation_pee": 0, "nb_formations_suivies": 2,
                 "nombre_employee_sous_responsabilite": 0, "distance_domicile_travail": 18,
                 "niveau_education": 3, "domaine_etude": "Commerce", "ayant_enfants": "non",
                 "frequence_deplacement": "Frequent", "annees_depuis_la_derniere_promotion": 1,
                 "annes_sous_responsable_actuel": 2},
                {"code_sondage": 92002, "nombre_participation_pee": 1, "nb_formations_suivies": 4,
                 "nombre_employee_sous_responsabilite": 2, "distance_domicile_travail": 10,
                 "niveau_education": 4, "domaine_etude": "Informatique", "ayant_enfants": "oui",
                 "frequence_deplacement": "Rarement", "annees_depuis_la_derniere_promotion": 2,
                 "annes_sous_responsable_actuel": 3}
            ]
        }
        response = client.post("/predict_shap_images", json=data)
        assert response.status_code == 200
        assert len(response.content) > 5000  # ZIP file with multiple images


@pytest.mark.skipif(DISABLE_DB, reason="Requires database enabled")
class TestJobEndpointsFullCoverage:
    """Test job-related endpoints for full coverage"""
    
    def test_create_report_job(self, client, valid_employee_data):
        """Test job creation endpoint"""
        response = client.post("/jobs/report", json=valid_employee_data)
        # Should return 200 or 201 for successful job creation
        assert response.status_code in [200, 201]
        if response.status_code == 200:
            data = response.json()
            assert "job_id" in data
