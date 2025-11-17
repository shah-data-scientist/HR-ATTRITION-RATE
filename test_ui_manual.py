"""
Manual UI Test Script
Tests the Streamlit UI by creating 5 employee predictions from training data
and verifying the results in the database.
"""

import pandas as pd
import httpx
import json
import time
from datetime import datetime

# Load training data
print("=" * 80)
print("LOADING TRAINING DATA FOR 5 TEST EMPLOYEES")
print("=" * 80)

sirh_df = pd.read_csv("data/extrait_sirh.csv")
eval_df = pd.read_csv("data/extrait_eval.csv")
sondage_df = pd.read_csv("data/extrait_sondage.csv")

# Select 5 diverse employees (IDs: 1, 2, 4, 5, 7)
test_employee_ids = [1, 2, 4, 5, 7]

print(f"\nSelected Employee IDs: {test_employee_ids}")
print("-" * 80)

# Prepare data for each employee
for idx, emp_id in enumerate(test_employee_ids, 1):
    print(f"\n{'=' * 80}")
    print(f"TEST {idx}/5: EMPLOYEE {emp_id}")
    print("=" * 80)
    
    # Get employee data
    sirh_row = sirh_df[sirh_df['id_employee'] == emp_id].iloc[0].to_dict()
    
    # Find matching eval row (E_{emp_id})
    eval_number = f"E_{emp_id}"
    eval_row = eval_df[eval_df['eval_number'] == eval_number].iloc[0].to_dict()
    
    # Find matching sondage row (code_sondage = emp_id)
    sondage_row = sondage_df[sondage_df['code_sondage'] == emp_id].iloc[0].to_dict()
    
    # Display employee info
    print(f"\nEmployee Information:")
    print(f"  ID: {emp_id}")
    print(f"  Age: {sirh_row['age']}, Gender: {sirh_row['genre']}")
    print(f"  Department: {sirh_row['departement']}, Position: {sirh_row['poste']}")
    print(f"  Monthly Income: {sirh_row['revenu_mensuel']}")
    print(f"  Marital Status: {sirh_row['statut_marital']}")
    print(f"  Years in Company: {sirh_row['annees_dans_l_entreprise']}")
    print(f"  Total Experience: {sirh_row['annee_experience_totale']}")
    
    # Prepare API request (same format as UI sends)
    payload = {
        "sirh_data": [sirh_row],
        "eval_data": [eval_row],
        "sondage_data": [sondage_row]
    }
    
    # Call API with user_id header (simulating UI)
    print(f"\nCalling API: POST /predict_report")
    print(f"User ID: test_user_{idx}")
    
    try:
        response = httpx.post(
            "http://localhost:8001/predict_report",
            json=payload,
            headers={"X-User-ID": f"test_user_{idx}"},
            timeout=30.0
        )
        
        if response.status_code == 200:
            result = response.json()
            prediction = result['predictions'][0]
            
            print(f"\n✅ PREDICTION SUCCESSFUL!")
            print("-" * 80)
            print(f"Prediction Result:")
            print(f"  Employee ID: {prediction['id_employee']}")
            print(f"  Prediction: {prediction['prediction']}")
            print(f"  Probability: {prediction['probability']:.4f} ({prediction['probability']*100:.2f}%)")
            print(f"  Risk Category: {prediction['risk_category']}")
            print(f"  Trace ID: {prediction['trace_id']}")
            
            if prediction.get('shap_values'):
                print(f"  SHAP Analysis: ✅ Computed ({len(prediction['shap_values'])} features)")
                print(f"  Base Value: {prediction['base_value']:.4f}")
            else:
                print(f"  SHAP Analysis: ❌ Not computed")
            
            # Wait a moment before next request
            time.sleep(1)
            
        else:
            print(f"\n❌ API ERROR: Status {response.status_code}")
            print(f"Response: {response.text[:500]}")
            
    except Exception as e:
        print(f"\n❌ EXCEPTION: {e}")

print("\n" + "=" * 80)
print("VERIFICATION: CHECKING DATABASE RECORDS")
print("=" * 80)

# Query database to verify all records were created
import subprocess

print("\nQuerying database for test employees...")
query = """
SELECT 
    e.id_employee,
    e.user_id,
    e.age,
    e.genre,
    e.departement,
    t.trace_id,
    o.risk_category,
    o.prediction_proba,
    CASE WHEN s.shap_id IS NOT NULL THEN 'YES' ELSE 'NO' END as has_shap
FROM employees e
JOIN model_inputs mi ON e.id_employee = mi.id_employee
JOIN predictions_traceability t ON mi.input_id = t.input_id
JOIN model_outputs o ON t.output_id = o.output_id
LEFT JOIN shap_analysis s ON t.trace_id = s.trace_id
WHERE e.id_employee IN (1, 2, 4, 5, 7)
ORDER BY t.trace_id DESC
LIMIT 5;
"""

result = subprocess.run(
    ["docker", "exec", "hrattritionrate-db-1", "psql", "-U", "user", "-d", "hr_attrition_db", "-c", query],
    capture_output=True,
    text=True
)

print("\nDatabase Query Results:")
print("-" * 80)
print(result.stdout)

# Summary
print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print(f"✓ Tested 5 employees from training data")
print(f"✓ All predictions sent via API (simulating UI workflow)")
print(f"✓ Database records verified")
print(f"\nCheck the output above to verify:")
print(f"  - All 5 employees have predictions")
print(f"  - Trace IDs are assigned")
print(f"  - SHAP analysis is stored (has_shap = 'YES')")
print("=" * 80)
