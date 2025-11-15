import pandas as pd
import numpy as np
import os

# Ensure the data directory exists
os.makedirs('data', exist_ok=True)

# --- extrait_eval.csv ---
eval_data = {
    'eval_number': [f'E_{i+1}' for i in range(10)],
    'augmentation_salaire_precedente': [f'{np.random.randint(3, 15)}%' for _ in range(10)],
    'heures_supplementaires': np.random.choice(['Oui', 'Non'], 10),
    'note_evaluation_actuelle': np.random.randint(1, 5, 10),
    'note_evaluation_precedente': np.random.randint(1, 5, 10),
    'anciennete': np.random.randint(1, 10, 10)
}
eval_df = pd.DataFrame(eval_data)
eval_df.to_csv('data/extrait_eval.csv', index=False)

# --- extrait_sirh.csv ---
sirh_data = {
    'id_employee': [i+1 for i in range(10)],
    'genre': np.random.choice(['m', 'f'], 10),
    'nombre_heures_travailless': np.random.randint(150, 200, 10),
    'departement': np.random.choice(['HR', 'IT', 'Sales'], 10),
    'salaire': np.random.randint(30000, 80000, 10)
}
sirh_df = pd.DataFrame(sirh_data)
sirh_df.to_csv('data/extrait_sirh.csv', index=False)

# --- extrait_sondage.csv ---
sondage_data = {
    'code_sondage': [i+1 for i in range(10)],
    'satisfaction_employee_nature_travail': np.random.randint(1, 5, 10),
    'satisfaction_employee_equipe': np.random.randint(1, 5, 10),
    'satisfaction_employee_equilibre_pro_perso': np.random.randint(1, 5, 10),
    'annees_dans_le_poste_actuel': np.random.randint(1, 5, 10),
    'annees_dans_l_entreprise': np.random.randint(1, 10, 10),
    'annees_sous_responsable_actuel': np.random.randint(1, 5, 10)
}
sondage_df = pd.DataFrame(sondage_data)
sondage_df.to_csv('data/extrait_sondage.csv', index=False)

print("Synthetic data files created in 'data/' directory.")
