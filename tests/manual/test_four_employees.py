"""Test SHAP storage for 4 employees with default user_id=demo1"""

import httpx
import time

# 4 different employees with varying profiles
employees_data = [
    {
        "id": 11111,
        "age": 28,
        "genre": "F",
        "revenu_mensuel": 4500,
        "statut_marital": "Célibataire",
        "departement": "Sales",
        "poste": "Sales Representative",
    },
    {
        "id": 22222,
        "age": 45,
        "genre": "M",
        "revenu_mensuel": 8500,
        "statut_marital": "Marié",
        "departement": "IT",
        "poste": "Manager",
    },
    {
        "id": 33333,
        "age": 35,
        "genre": "F",
        "revenu_mensuel": 6000,
        "statut_marital": "Divorcé",
        "departement": "HR",
        "poste": "HR Specialist",
    },
    {
        "id": 44444,
        "age": 52,
        "genre": "M",
        "revenu_mensuel": 12000,
        "statut_marital": "Marié",
        "departement": "Executive",
        "poste": "Director",
    },
]

print("=" * 60)
print("TESTING 4 EMPLOYEES - USER_ID: demo1")
print("=" * 60)

results = []

for i, emp in enumerate(employees_data, 1):
    print(f"\n[{i}/4] Testing Employee {emp['id']}...")

    payload = {
        "sirh_data": [
            {
                "id_employee": emp["id"],
                "age": emp["age"],
                "genre": emp["genre"],
                "revenu_mensuel": emp["revenu_mensuel"],
                "statut_marital": emp["statut_marital"],
                "departement": emp["departement"],
                "poste": emp["poste"],
                "nombre_experiences_precedentes": 2,
                "nombre_heures_travailless": 80,
                "annee_experience_totale": 10,
                "annees_dans_l_entreprise": 5,
                "annees_dans_le_poste_actuel": 3,
            }
        ],
        "eval_data": [
            {
                "eval_number": f"E_{emp['id']}",
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
                "code_sondage": emp["id"],
                "nombre_participation_pee": 1,
                "nb_formations_suivies": 3,
                "nombre_employee_sous_responsabilite": 0,
                "distance_domicile_travail": 15,
                "niveau_education": 3,
                "domaine_etude": "Informatique",
                "ayant_enfants": "oui",
                "frequence_deplacement": "Rarement",
                "annees_depuis_la_derniere_promotion": 1,
                "annes_sous_responsable_actuel": 2,
            }
        ],
    }

    # No X-User-ID header = defaults to "demo1"
    response = httpx.post(
        "http://localhost:8001/predict_report", json=payload, timeout=30.0
    )

    if response.status_code == 200:
        result = response.json()
        pred = result["predictions"][0]
        trace_id = pred["trace_id"]
        probability = pred["probability"]
        has_shap = pred.get("shap_values") is not None

        results.append(
            {
                "employee_id": emp["id"],
                "trace_id": trace_id,
                "probability": probability,
                "has_shap": has_shap,
                "status": "✅",
            }
        )

        print(f"  ✅ SUCCESS")
        print(f"     Trace ID: {trace_id}")
        print(f"     Attrition Probability: {probability:.4f}")
        print(f"     SHAP Computed: {has_shap}")
    else:
        results.append(
            {"employee_id": emp["id"], "status": "❌", "error": response.text[:100]}
        )
        print(f"  ❌ FAILED: {response.status_code}")
        print(f"     {response.text[:100]}")

    time.sleep(0.5)  # Small delay between requests

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
for r in results:
    if r["status"] == "✅":
        print(
            f"{r['status']} Employee {r['employee_id']}: trace_id={r['trace_id']}, "
            f"probability={r['probability']:.4f}, SHAP={r['has_shap']}"
        )
    else:
        print(
            f"{r['status']} Employee {r['employee_id']}: {r.get('error', 'Unknown error')}"
        )

# Show trace_ids for database query
successful_traces = [r["trace_id"] for r in results if r["status"] == "✅"]
print(f"\n📊 Trace IDs for database verification: {successful_traces}")
