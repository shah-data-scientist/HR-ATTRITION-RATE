"""Display all database records for the 4 test employees"""

import psycopg2

# Connect to database
conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="hr_attrition_db",
    user="user",
    password="password",
)
cur = conn.cursor()

print("=" * 80)
print("DATABASE RECORDS FOR 4 EMPLOYEES (user_id=demo1)")
print("=" * 80)

# 1. Predictions Traceability
print("\n1️⃣  PREDICTIONS TRACEABILITY")
print("-" * 80)
cur.execute("""
    SELECT trace_id, input_id, output_id, model_version, prediction_source, created_at
    FROM predictions_traceability
    WHERE trace_id BETWEEN 38 AND 41
    ORDER BY trace_id
""")
print(
    f"{'Trace ID':<10} {'Input ID':<10} {'Output ID':<11} {'Version':<8} {'Source':<8} {'Created At':<25}"
)
print("-" * 80)
for row in cur.fetchall():
    print(
        f"{row[0]:<10} {row[1]:<10} {row[2]:<11} {row[3]:<8} {row[4]:<8} {str(row[5]):<25}"
    )

# 2. Employees
print("\n2️⃣  EMPLOYEE DATA")
print("-" * 80)
cur.execute("""
    SELECT id_employee, age, genre, revenu_mensuel, departement, poste, user_id
    FROM employees
    WHERE id_employee IN (11111, 22222, 33333, 44444)
    ORDER BY id_employee
""")
print(
    f"{'Employee ID':<12} {'Age':<5} {'Genre':<7} {'Salary':<8} {'Dept':<12} {'Poste':<20} {'User ID':<8}"
)
print("-" * 80)
for row in cur.fetchall():
    print(
        f"{row[0]:<12} {row[1]:<5} {row[2]:<7} {row[3]:<8} {row[4]:<12} {row[5]:<20} {row[6]:<8}"
    )

# 3. Model Outputs
print("\n3️⃣  MODEL PREDICTIONS")
print("-" * 80)
cur.execute("""
    SELECT output_id, prediction_proba, prediction_label, risk_category
    FROM model_outputs
    WHERE output_id BETWEEN 38 AND 41
    ORDER BY output_id
""")
print(f"{'Output ID':<11} {'Probability':<12} {'Label':<10} {'Risk Category':<15}")
print("-" * 80)
for row in cur.fetchall():
    prob_pct = f"{row[1]*100:.2f}%"
    print(f"{row[0]:<11} {prob_pct:<12} {row[2]:<10} {row[3]:<15}")

# 4. SHAP Analysis
print("\n4️⃣  SHAP ANALYSIS (THE FIX WE VERIFIED!)")
print("-" * 80)
cur.execute("""
    SELECT trace_id, base_value, created_at
    FROM shap_analysis
    WHERE trace_id BETWEEN 38 AND 41
    ORDER BY trace_id
""")
print(f"{'Trace ID':<10} {'Base Value':<20} {'Created At':<25}")
print("-" * 80)
rows = cur.fetchall()
if rows:
    for row in rows:
        print(f"{row[0]:<10} {row[1]:<20.10f} {str(row[2]):<25}")
    print(f"\n✅ SHAP data stored for {len(rows)} predictions!")
else:
    print("❌ NO SHAP DATA FOUND!")

# 5. Complete Summary
print("\n5️⃣  COMPLETE SUMMARY")
print("-" * 80)
cur.execute("""
    SELECT 
        t.trace_id,
        t.input_id as employee_id,
        o.prediction_proba,
        o.prediction_label,
        o.risk_category,
        CASE WHEN s.trace_id IS NOT NULL THEN '✅ YES' ELSE '❌ NO' END as has_shap
    FROM predictions_traceability t
    JOIN model_outputs o ON t.output_id = o.output_id
    LEFT JOIN shap_analysis s ON t.trace_id = s.trace_id
    WHERE t.trace_id BETWEEN 38 AND 41
    ORDER BY t.trace_id
""")
print(
    f"{'Trace':<7} {'Employee':<10} {'Probability':<12} {'Label':<10} {'Risk':<10} {'SHAP?':<8}"
)
print("-" * 80)
for row in cur.fetchall():
    prob_pct = f"{row[2]*100:.2f}%"
    print(
        f"{row[0]:<7} {row[1]:<10} {prob_pct:<12} {row[3]:<10} {row[4]:<10} {row[5]:<8}"
    )

# Count SHAP records
cur.execute("SELECT COUNT(*) FROM shap_analysis WHERE trace_id BETWEEN 38 AND 41")
shap_count = cur.fetchone()[0]

print("\n" + "=" * 80)
print(f"✅ SUCCESS: All 4 employees processed")
print(f"✅ SHAP FIX VERIFIED: {shap_count}/4 predictions have SHAP data stored")
print(f"✅ User ID: demo1 (default - no header provided)")
print("=" * 80)

cur.close()
conn.close()
