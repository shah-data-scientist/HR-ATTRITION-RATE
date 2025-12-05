"""
Test script to simulate a single employee prediction and trace through all database tables.
"""

import httpx
import json

# Single employee data (matching the schema)
payload = {
    "eval_data": [
        {
            "satisfaction_employee_environnement": 3,
            "note_evaluation_precedente": 3,
            "niveau_hierarchique_poste": 2,
            "satisfaction_employee_nature_travail": 4,
            "satisfaction_employee_equipe": 3,
            "satisfaction_employee_equilibre_pro_perso": 3,
            "eval_number": "E_88888",
            "note_evaluation_actuelle": 4,
            "heure_supplementaires": "Non",
            "augementation_salaire_precedente": "7 %",
        }
    ],
    "sirh_data": [
        {
            "id_employee": 88888,
            "age": 35,
            "genre": "M",
            "revenu_mensuel": 6000,
            "statut_marital": "Marié",
            "departement": "IT",
            "poste": "Développeur",
            "nombre_experiences_precedentes": 2,
            "nombre_heures_travailless": 190,
            "annee_experience_totale": 10,
            "annees_dans_l_entreprise": 5,
            "annees_dans_le_poste_actuel": 3,
        }
    ],
    "sondage_data": [
        {
            "nombre_participation_pee": 1,
            "nb_formations_suivies": 2,
            "nombre_employee_sous_responsabilite": 0,
            "code_sondage": 88888,
            "distance_domicile_travail": 15,
            "niveau_education": 3,
            "domaine_etude": "Informatique",
            "ayant_enfants": "Oui",
            "frequence_deplacement": "Rarement",
            "annees_depuis_la_derniere_promotion": 2,
            "annes_sous_responsable_actuel": 2,
        }
    ],
}

# Send prediction request with user_id header
headers = {"X-User-ID": "test1"}
API_URL = "http://localhost:8001"

print("=" * 80)
print("SENDING PREDICTION REQUEST FOR EMPLOYEE 88888")
print("=" * 80)
print(f"\nUser ID: {headers['X-User-ID']}")
print(f"Employee ID: 88888")
print(f"Age: 35, Gender: M, Department: IT, Position: Développeur")
print(f"\nCalling API: POST {API_URL}/predict_report")
print("-" * 80)

try:
    response = httpx.post(
        f"{API_URL}/predict_report", json=payload, headers=headers, timeout=60.0
    )
    response.raise_for_status()
    result = response.json()

    print("\n✅ PREDICTION SUCCESSFUL!")
    print("-" * 80)

    if result.get("predictions"):
        pred = result["predictions"][0]
        print(f"\nPrediction Result:")
        print(f"  Employee ID: {pred['id_employee']}")
        print(f"  Prediction: {pred['prediction']}")
        print(
            f"  Probability: {pred['probability']:.4f} ({pred['probability']*100:.2f}%)"
        )
        print(f"  Risk Category: {pred['risk_category']}")
        print(f"  Trace ID: {pred['trace_id']}")
        print(f"  SHAP values computed: {'Yes' if pred.get('shap_values') else 'No'}")
        if pred.get("shap_values"):
            print(f"  Number of features: {len(pred['shap_values'])}")
            print(f"  Base value: {pred.get('base_value', 'N/A')}")

    print("\n" + "=" * 80)
    print("Prediction completed. Now querying database for all created records...")
    print("=" * 80)

except httpx.HTTPError as e:
    print(f"\n❌ ERROR: {e}")
    if hasattr(e, "response") and e.response:
        print(f"Response: {e.response.text}")
except Exception as e:
    print(f"\n❌ UNEXPECTED ERROR: {e}")
