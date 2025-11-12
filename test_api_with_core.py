"""Test API with new core schema (raw input format)."""

import json

from core.schema import EmployeeInputSchema, BatchPredictionInput

# Test 1: Validate raw input schema
print("=== Testing API Schema Validation ===\n")

raw_employee_data = {
    "id_employee": 12345,
    "eval_number": "E_12345",
    "age": 35,
    "genre": "Homme",  # STRING, not int!
    "revenu_mensuel": 6000.0,
    "heure_supplementaires": "Non",  # STRING "Oui"/"Non", not int!
    "augementation_salaire_precedente": 7.0,  # Float 0-35, not 0.0-0.35!
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

try:
    # Test single employee validation
    employee = EmployeeInputSchema(**raw_employee_data)
    print("[OK] Employee schema validation PASSED")
    print(f"  - Genre (raw): {employee.genre}")
    print(f"  - Heures supp (raw): {employee.heure_supplementaires}")
    print(f"  - Salary increase: {employee.augementation_salaire_precedente}%")

    # Test batch validation
    batch = BatchPredictionInput(employees=[employee])
    print("\n[OK] Batch schema validation PASSED")
    print(f"  - Batch size: {len(batch.employees)}")

    # Show JSON payload
    payload = batch.model_dump()
    print("\n[OK] JSON serialization PASSED")
    print(f"  - Payload keys: {list(payload.keys())}")
    print(f"  - First employee keys: {len(payload['employees'][0])} fields")

except Exception as e:
    print(f"[FAIL] Schema validation FAILED: {e}")
    raise

print("\n=== API Schema Test Complete ===")
print("\nNOTE: This test validates the RAW input schema.")
print("The API will clean and transform this data before model inference.")
print("  - genre: 'Homme' -> 1, 'Femme' -> 0")
print("  - heure_supplementaires: 'Oui' -> 1, 'Non' -> 0")
print("  - Then add engineered features: improvement_evaluation, total_satisfaction, work_mobility")
