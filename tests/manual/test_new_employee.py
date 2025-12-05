"""Test with new employee ID"""

import httpx

payload = {
    "sirh_data": [
        {
            "id_employee": 99999,
            "age": 30,
            "genre": "M",
            "revenu_mensuel": 5000,
            "statut_marital": "Marié",
            "departement": "IT",
            "poste": "Developer",
            "nombre_experiences_precedentes": 2,
            "nombre_heures_travailless": 80,
            "annee_experience_totale": 8,
            "annees_dans_l_entreprise": 3,
            "annees_dans_le_poste_actuel": 2,
        }
    ],
    "eval_data": [
        {
            "eval_number": "E_99999",
            "note_evaluation_precedente": 3,
            "note_evaluation_actuelle": 3,
            "augementation_salaire_precedente": "Moyen",
            "niveau_hierarchique_poste": 2,
            "satisfaction_employee_environnement": 3,
            "satisfaction_employee_nature_travail": 4,
            "satisfaction_employee_equipe": 4,
            "satisfaction_employee_equilibre_pro_perso": 3,
            "heure_supplementaires": "Non",
            "annees_depuis_la_derniere_promotion": 1,
            "annes_sous_responsable_actuel": 2,
        }
    ],
    "sondage_data": [
        {
            "code_sondage": 99999,
            "nombre_participation_pee": 1,
            "nb_formations_suivies": 2,
            "nombre_employee_sous_responsabilite": 0,
            "distance_domicile_travail": 10,
            "niveau_education": 3,
            "domaine_etude": "Informatique",
            "ayant_enfants": "oui",
            "frequence_deplacement": "Rarement",
            "annees_depuis_la_derniere_promotion": 1,
            "annes_sous_responsable_actuel": 2,
        }
    ],
}

response = httpx.post(
    "http://localhost:8001/predict_report",
    json=payload,
    headers={"X-User-ID": "usr1"},  # 4 chars - fits in VARCHAR(5)
    timeout=30.0,
)

print(f"Status: {response.status_code}")
if response.status_code == 200:
    result = response.json()
    pred = result["predictions"][0]
    print(
        f"✅ SUCCESS! Trace ID: {pred['trace_id']}, Probability: {pred['probability']:.4f}"
    )
    print(f"SHAP computed: {pred.get('shap_values') is not None}")
else:
    print(f"❌ ERROR: {response.text[:200]}")
