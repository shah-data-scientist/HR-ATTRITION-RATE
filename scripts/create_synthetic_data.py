import pandas as pd
import numpy as np
import os
from datetime import datetime

# Ensure the data directory exists
os.makedirs('data', exist_ok=True)

# Number of synthetic employees to generate
NUM_EMPLOYEES = 10

# --- Generate a single DataFrame conforming to EmployeeInputSchema ---
# This will be a simplified version for synthetic data, focusing on required fields.

synthetic_data = {
    'satisfaction_employee_environnement': np.random.randint(1, 5, NUM_EMPLOYEES),
    'note_evaluation_precedente': np.random.randint(1, 5, NUM_EMPLOYEES),
    'satisfaction_employee_nature_travail': np.random.randint(1, 5, NUM_EMPLOYEES),
    'satisfaction_employee_equipe': np.random.randint(1, 5, NUM_EMPLOYEES),
    'satisfaction_employee_equilibre_pro_perso': np.random.randint(1, 5, NUM_EMPLOYEES),
    'note_evaluation_actuelle': np.random.randint(1, 5, NUM_EMPLOYEES),
    'niveau_hierarchique_poste': np.random.randint(1, 6, NUM_EMPLOYEES),
    'heure_supplementaires': np.random.choice(['Oui', 'Non'], NUM_EMPLOYEES),
    'augementation_salaire_precedente': np.round(np.random.uniform(0.0, 35.0, NUM_EMPLOYEES), 1),
    'id_employee': [i + 1 for i in range(NUM_EMPLOYEES)],
    'eval_number': [f'E_{i + 1}' for i in range(NUM_EMPLOYEES)],
    'age': np.random.randint(20, 60, NUM_EMPLOYEES),
    'genre': np.random.choice(['Homme', 'Femme'], NUM_EMPLOYEES),
    'revenu_mensuel': np.random.randint(2000, 10000, NUM_EMPLOYEES),
    'statut_marital': np.random.choice(['Célibataire', 'Marié', 'Divorcé'], NUM_EMPLOYEES),
    'departement': np.random.choice(['IT', 'HR', 'Sales', 'Marketing'], NUM_EMPLOYEES),
    'poste': np.random.choice(['Développeur', 'Manager', 'Analyste', 'Support'], NUM_EMPLOYEES),
    'domaine_etude': np.random.choice(['Informatique', 'RH', 'Commerce', 'Autre'], NUM_EMPLOYEES),
    'frequence_deplacement': np.random.choice(['Rarement', 'Souvent', 'Jamais'], NUM_EMPLOYEES),
    'nombre_experiences_precedentes': np.random.randint(0, 10, NUM_EMPLOYEES),
    'nombre_heures_travailless': np.random.randint(150, 200, NUM_EMPLOYEES),
    'annee_experience_totale': np.random.randint(1, 20, NUM_EMPLOYEES),
    'annees_dans_l_entreprise': np.random.randint(1, 15, NUM_EMPLOYEES),
    'annees_dans_le_poste_actuel': np.random.randint(0, 10, NUM_EMPLOYEES),
    'annees_depuis_la_derniere_promotion': np.random.randint(0, 5, NUM_EMPLOYEES),
    'annes_sous_responsable_actuel': np.random.randint(0, 10, NUM_EMPLOYEES),
    'nombre_participation_pee': np.random.randint(0, 3, NUM_EMPLOYEES),
    'nb_formations_suivies': np.random.randint(0, 4, NUM_EMPLOYEES),
    'nombre_employee_sous_responsabilite': np.random.randint(0, 10, NUM_EMPLOYEES),
    'code_sondage': [str(i + 1) for i in range(NUM_EMPLOYEES)], # Assuming string for code_sondage
    'distance_domicile_travail': np.random.randint(1, 30, NUM_EMPLOYEES),
    'niveau_education': np.random.randint(1, 6, NUM_EMPLOYEES),
    'ayant_enfants': np.random.choice(['Oui', 'Non'], NUM_EMPLOYEES),
}

synthetic_df = pd.DataFrame(synthetic_data)

# Save the single synthetic data file
synthetic_df.to_csv('data/synthetic_employees.csv', index=False)

print(f"Generated {NUM_EMPLOYEES} synthetic employee records in 'data/synthetic_employees.csv'.")