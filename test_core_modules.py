"""Quick test script to validate core modules."""

import pandas as pd

from core import (
    clean_and_engineer_features,
    enforce_schema,
    validate_input_ranges,
    ALL_FEATURE_COLS,
    NUMERIC_COLS,
    CATEGORICAL_COLS,
)

# Create sample raw input data
sample_data = {
    "id_employee": [1],
    "age": [35],
    "genre": ["Homme"],
    "revenu_mensuel": [6000.0],
    "heure_supplementaires": ["Non"],
    "augementation_salaire_precedente": [7.0],
    "statut_marital": ["Marié"],
    "departement": ["IT"],
    "poste": ["Développeur"],
    "domaine_etude": ["Informatique"],
    "ayant_enfants": ["Oui"],
    "frequence_deplacement": ["Rarement"],
    "nombre_experiences_precedentes": [2],
    "nombre_heures_travailless": [190],
    "annee_experience_totale": [10],
    "annees_dans_l_entreprise": [5],
    "annees_dans_le_poste_actuel": [3],
    "nombre_participation_pee": [1],
    "nb_formations_suivies": [2],
    "nombre_employee_sous_responsabilite": [0],
    "distance_domicile_travail": [15],
    "niveau_education": [3],
    "annees_depuis_la_derniere_promotion": [2],
    "annes_sous_responsable_actuel": [2],
    "satisfaction_employee_environnement": [3],
    "note_evaluation_precedente": [3],
    "niveau_hierarchique_poste": [2],
    "satisfaction_employee_nature_travail": [4],
    "satisfaction_employee_equipe": [3],
    "satisfaction_employee_equilibre_pro_perso": [3],
    "note_evaluation_actuelle": [4],
}

print("=== Testing Core Modules ===\n")

# Step 1: Create DataFrame
df = pd.DataFrame(sample_data)
print(f"1. Raw input shape: {df.shape}")
print(f"   Columns: {df.columns.tolist()[:5]}...")

# Step 2: Clean and engineer features
df_processed = clean_and_engineer_features(df)
print(f"\n2. After cleaning & engineering: {df_processed.shape}")
print(f"   Engineered features added:")
print(f"   - improvement_evaluation: {df_processed['improvement_evaluation'].values[0]}")
print(f"   - total_satisfaction: {df_processed['total_satisfaction'].values[0]}")
print(f"   - work_mobility: {df_processed['work_mobility'].values[0]}")

# Step 3: Enforce schema
feature_order = NUMERIC_COLS + CATEGORICAL_COLS
df_final = enforce_schema(df_processed, feature_order)
print(f"\n3. After schema enforcement: {df_final.shape}")
print(f"   Expected columns: {len(ALL_FEATURE_COLS)}")
print(f"   Actual columns: {len(df_final.columns)}")
print(f"   Match: {df_final.columns.tolist() == ALL_FEATURE_COLS}")

# Step 4: Validate ranges
warnings = validate_input_ranges(df_final)
if warnings:
    print(f"\n4. Validation warnings: {warnings}")
else:
    print("\n4. Validation: All values within expected ranges OK")

print("\n=== Core Modules Test Complete ===")
