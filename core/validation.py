"""Canonical schema validation definitions.

This module defines:
1. RAW_INPUT_SCHEMA: What the API receives (before cleaning)
2. MODEL_INPUT_SCHEMA: What the model expects (after cleaning + engineering)
3. Validation rules and constraints
"""

from typing import Final

# ============================================================================
# RAW INPUT SCHEMA (API receives this)
# ============================================================================

RAW_NUMERIC_COLS: Final[list[str]] = [
    "satisfaction_employee_environnement",  # 1-4
    "note_evaluation_precedente",  # 1-4
    "niveau_hierarchique_poste",  # 1-5
    "satisfaction_employee_nature_travail",  # 1-4
    "satisfaction_employee_equipe",  # 1-4
    "satisfaction_employee_equilibre_pro_perso",  # 1-4
    "note_evaluation_actuelle",  # 1-4
    "augementation_salaire_precedente",  # float 0.0-35.0
    "id_employee",  # int >= 0
    "age",  # 18-70
    "revenu_mensuel",  # float >= 0
    "nombre_experiences_precedentes",  # int >= 0
    "nombre_heures_travailless",  # int >= 0
    "annee_experience_totale",  # int >= 0
    "annees_dans_l_entreprise",  # int >= 0
    "annees_dans_le_poste_actuel",  # int >= 0
    "nombre_participation_pee",  # int >= 0
    "nb_formations_suivies",  # 0-3
    "nombre_employee_sous_responsabilite",  # 0-20
    "distance_domicile_travail",  # 0-50
    "niveau_education",  # 1-5
    "annees_depuis_la_derniere_promotion",  # int >= 0
    "annes_sous_responsable_actuel",  # int >= 0
]

RAW_CATEGORICAL_COLS: Final[list[str]] = [
    "heure_supplementaires",  # "Oui"/"Non"
    "genre",  # "Homme"/"Femme"
    "statut_marital",  # string category
    "departement",  # string category
    "poste",  # string category
    "domaine_etude",  # string category
    "ayant_enfants",  # "Oui"/"Non" or 0/1
    "frequence_deplacement",  # string category
    "eval_number",  # string (e.g., "E_123")
    "code_sondage",  # string (later → int, constant 80)
]

# All raw input columns
RAW_INPUT_COLS: Final[list[str]] = RAW_NUMERIC_COLS + RAW_CATEGORICAL_COLS

# Target column (NOT in inference payload)
TARGET_COL: Final[str] = "a_quitte_l_entreprise"

# ============================================================================
# MODEL INPUT SCHEMA (after cleaning + engineering)
# ============================================================================

# After cleaning, these columns feed into the model
NUMERIC_COLS: Final[list[str]] = [
    "satisfaction_employee_environnement",
    "note_evaluation_precedente",
    "niveau_hierarchique_poste",
    "satisfaction_employee_nature_travail",
    "satisfaction_employee_equipe",
    "satisfaction_employee_equilibre_pro_perso",
    "note_evaluation_actuelle",
    "heures_supplementaires",  # cleaned to 0/1
    "id_employee",
    "age",
    "genre",  # cleaned to 0/1
    "revenu_mensuel",
    "nombre_experiences_precedentes",
    "annee_experience_totale",
    "annees_dans_l_entreprise",
    "annees_dans_le_poste_actuel",
    "nombre_participation_pee",
    "nb_formations_suivies",
    "nombre_employee_sous_responsabilite",
    "distance_domicile_travail",
    "niveau_education",
    "annees_depuis_la_derniere_promotion",
    "annes_sous_responsable_actuel",
    "improvement_evaluation",  # engineered
    "total_satisfaction",  # engineered
    "work_mobility",  # engineered (float)
]

CATEGORICAL_COLS: Final[list[str]] = [
    "augementation_salaire_precedente",
    "statut_marital",
    "departement",
    "poste",
    "domaine_etude",
    "ayant_enfants",
    "frequence_deplacement",
]

# All model input columns (in canonical order)
ALL_FEATURE_COLS: Final[list[str]] = NUMERIC_COLS + CATEGORICAL_COLS

# Engineered features (computed during processing)
ENGINEERED_FEATURES: Final[list[str]] = [
    "improvement_evaluation",
    "total_satisfaction",
    "work_mobility",
]

# ============================================================================
# EXPECTED DTYPES (after cleaning)
# ============================================================================

EXPECTED_DTYPES: Final[dict[str, str]] = {
    "satisfaction_employee_environnement": "int64",
    "note_evaluation_precedente": "int64",
    "niveau_hierarchique_poste": "int64",
    "satisfaction_employee_nature_travail": "int64",
    "satisfaction_employee_equipe": "int64",
    "satisfaction_employee_equilibre_pro_perso": "int64",
    "note_evaluation_actuelle": "int64",
    "heures_supplementaires": "Int64",  # cleaned to 0/1
    "augementation_salaire_precedente": "object",
    "id_employee": "Int64",
    "age": "int64",
    "genre": "Int64",  # cleaned to 0/1
    "revenu_mensuel": "int64",
    "statut_marital": "object",
    "departement": "object",
    "poste": "object",
    "nombre_experiences_precedentes": "int64",
    "annee_experience_totale": "int64",
    "annees_dans_l_entreprise": "int64",
    "annees_dans_le_poste_actuel": "int64",
    "nombre_participation_pee": "int64",
    "nb_formations_suivies": "int64",
    "nombre_employee_sous_responsabilite": "int64",
    "distance_domicile_travail": "int64",
    "niveau_education": "int64",
    "domaine_etude": "object",
    "ayant_enfants": "object",
    "frequence_deplacement": "object",
    "annees_depuis_la_derniere_promotion": "int64",
    "annes_sous_responsable_actuel": "int64",
    "improvement_evaluation": "int64",
    "total_satisfaction": "int64",
    "work_mobility": "float64",
}

# ============================================================================
# VALIDATION RANGES
# ============================================================================

VALIDATION_RANGES: Final[dict[str, tuple[float, float]]] = {
    "satisfaction_employee_environnement": (1, 4),
    "note_evaluation_precedente": (1, 4),
    "niveau_hierarchique_poste": (1, 5),
    "satisfaction_employee_nature_travail": (1, 4),
    "satisfaction_employee_equipe": (1, 4),
    "satisfaction_employee_equilibre_pro_perso": (1, 4),
    "note_evaluation_actuelle": (1, 4),
    "age": (18, 70),
    "augementation_salaire_precedente": (0.0, 35.0),
    "nb_formations_suivies": (0, 3),
    "nombre_employee_sous_responsabilite": (0, 20),
    "distance_domicile_travail": (0, 50),
    "niveau_education": (1, 5),
}
