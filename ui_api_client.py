"""API client for UI to call prediction endpoints instead of local inference.

This module provides functions to replace local model.predict_proba() calls
with API calls to the FastAPI backend.
"""

import os
from typing import Any

import httpx
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")
API_TOKEN = os.getenv("API_TOKEN", "your_super_secret_api_token")


def call_prediction_api(employee_dataframe: pd.DataFrame) -> dict[str, Any] | None:
    """Call the FastAPI /predict endpoint with employee data.

    Args:
        employee_dataframe: DataFrame with employee features (raw format)

    Returns:
        API response dict with predictions, or None if error

    Example response:
        {
            "predictions": [
                {
                    "id_employee": 12345,
                    "prediction": "Leave",
                    "probability": 0.75,
                    "risk_category": "High",
                    "message": "...",
                    "trace_id": 123
                }
            ]
        }
    """
    headers = {
        "X-API-Key": API_TOKEN,
        "Content-Type": "application/json",
    }

    # Convert DataFrame to list of dicts
    employees = employee_dataframe.to_dict("records")
    payload = {"employees": employees}

    try:
        response = httpx.post(
            f"{API_URL}/predict",
            json=payload,
            headers=headers,
            timeout=60.0,  # Longer timeout for batch predictions
        )
        response.raise_for_status()
        return response.json()

    except httpx.HTTPStatusError as e:
        print(f"API Error ({e.response.status_code}): {e.response.text}")
        return None
    except httpx.RequestError as e:
        print(f"Connection Error: {e}. Is the API running at {API_URL}?")
        return None
    except Exception as e:
        print(f"Unexpected error calling API: {e}")
        return None


def extract_probabilities_from_api_response(api_response: dict) -> list[float]:
    """Extract prediction probabilities from API response.

    Args:
        api_response: Response dict from call_prediction_api()

    Returns:
        List of probabilities (floats between 0 and 1)
    """
    if not api_response or "predictions" not in api_response:
        return []

    return [pred["probability"] for pred in api_response["predictions"]]


def check_api_health() -> bool:
    """Check if the API is reachable and healthy.

    Returns:
        True if API is healthy, False otherwise
    """
    try:
        response = httpx.get(f"{API_URL}/health", timeout=5.0)
        response.raise_for_status()
        data = response.json()
        return data.get("status") == "ok"
    except Exception:
        return False


# Example usage (for testing)
if __name__ == "__main__":
    import sys

    print(f"Checking API at {API_URL}...")

    if check_api_health():
        print("✓ API is healthy!")
    else:
        print("✗ API is not reachable")
        sys.exit(1)

    # Test prediction with sample data
    sample_data = pd.DataFrame(
        [
            {
                "id_employee": 99999,
                "age": 35,
                "genre": "Homme",
                "revenu_mensuel": 6000.0,
                "heure_supplementaires": "Non",
                "augementation_salaire_precedente": 7.0,
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
        ]
    )

    print("\nCalling prediction API...")
    result = call_prediction_api(sample_data)

    if result:
        print("\n✓ Prediction successful!")
        for pred in result["predictions"]:
            print(f"  Employee {pred['id_employee']}:")
            print(f"    Prediction: {pred['prediction']}")
            print(f"    Probability: {pred['probability']:.2%}")
            print(f"    Risk: {pred['risk_category']}")
    else:
        print("\n✗ Prediction failed")
