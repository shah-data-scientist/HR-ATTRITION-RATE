# Troubleshooting 422 Validation Error

## Root Cause Identified

The 422 error is caused by **uploading the wrong CSV files** to the Streamlit app. The app expects **three separate CSV files** with specific schemas, but you're uploading files that have already been merged/preprocessed.

## The Problem

Looking at the error, the uploaded files contain fields from ALL three data sources mixed together:

**Your uploaded "eval" file contains:**
- `satisfaction_employee_environnement` (from sondage)
- `note_evaluation_precedente` (correct - from eval)
- `niveau_hierarchique_poste` (from sirh)
- `augementation_salaire_precedente` (TYPO - should be `augmentation_`)
- `heure_supplementaires` (TYPO - should be `heures_`)
- **MISSING**: `anciennete`

**Your uploaded "sirh" file contains:**
- `age`, `revenu_mensuel`, `statut_marital`, etc. (mixed data)
- **MISSING**: `salaire`

**Your uploaded "sondage" file contains:**
- `nombre_participation_pee`, `code_sondage`, etc.
- **MISSING**: `satisfaction_employee_nature_travail` and other required fields

## Expected CSV Schemas

### extrait_eval.csv
```csv
eval_number,augmentation_salaire_precedente,heures_supplementaires,note_evaluation_actuelle,note_evaluation_precedente,anciennete
E_1,11%,Oui,2,4,3
E_2,12%,Oui,4,3,2
```

### extrait_sirh.csv
```csv
id_employee,genre,nombre_heures_travailless,departement,salaire
1,m,186,IT,76106
2,f,163,Sales,63096
```

### extrait_sondage.csv
```csv
code_sondage,satisfaction_employee_nature_travail,satisfaction_employee_equipe,satisfaction_employee_equilibre_pro_perso,annees_dans_le_poste_actuel,annees_dans_l_entreprise,annees_sous_responsable_actuel
1,2,3,2,1,5,4
2,3,3,3,2,2,3
```

## Solution

### Option 1: Use the Correct CSV Files (RECOMMENDED)

Use the files from the `data/` directory:
- `data/extrait_eval.csv`
- `data/extrait_sirh.csv`
- `data/extrait_sondage.csv`

These files have the correct schema.

### Option 2: Don't Upload Any Files

If you don't upload files, the Streamlit app will automatically load the correct files from the `data/` directory for testing.

### Option 3: Fix Your CSV Files

If you have custom CSV files, ensure they match the exact schema shown above:

1. **extrait_eval.csv** - must have exactly these columns:
   - `eval_number` (format: "E_123")
   - `augmentation_salaire_precedente` (format: "11%")  ⚠️ Note the 'g'!
   - `heures_supplementaires` (values: "Oui" or "Non")  ⚠️ Note the plural!
   - `note_evaluation_actuelle` (integer 1-4)
   - `note_evaluation_precedente` (integer 1-4)
   - `anciennete` (integer)

2. **extrait_sirh.csv** - must have exactly these columns:
   - `id_employee` (integer)
   - `genre` (values: "m" or "f")
   - `nombre_heures_travailless` (integer)
   - `departement` (string)
   - `salaire` (integer)  ⚠️ Not revenu_mensuel!

3. **extrait_sondage.csv** - must have exactly these columns:
   - `code_sondage` (integer, corresponds to id_employee)
   - `satisfaction_employee_nature_travail` (integer 1-4)
   - `satisfaction_employee_equipe` (integer 1-4)
   - `satisfaction_employee_equilibre_pro_perso` (integer 1-4)
   - `annees_dans_le_poste_actuel` (integer)
   - `annees_dans_l_entreprise` (integer)
   - `annees_sous_responsable_actuel` (integer)

## Common Typos to Avoid

- ❌ `augementation_salaire_precedente` → ✅ `augmentation_salaire_precedente`
- ❌ `heure_supplementaires` → ✅ `heures_supplementaires`
- ❌ `revenu_mensuel` in SIRH → ✅ `salaire`

## Testing Your Files

Run this command to test your CSV files before uploading:

```bash
poetry run python test_api_debug.py
```

This will show you exactly what the API receives and whether it's valid.

## Quick Fix

**For now, simply DON'T upload any files in the Streamlit app.**

The app will automatically use the correct files from `data/` directory, which will work perfectly!
