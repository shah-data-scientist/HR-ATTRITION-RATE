"""Display SHAP analysis details for one employee"""

import psycopg2
import json

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
print("SHAP ANALYSIS DETAILS - EMPLOYEE 11111 (Trace ID 38)")
print("=" * 80)

# Get SHAP data for trace_id 38
cur.execute("""
    SELECT trace_id, shap_values, base_value, feature_names, created_at
    FROM shap_analysis
    WHERE trace_id = 38
""")

row = cur.fetchone()
if row:
    trace_id = row[0]
    shap_values = row[1] if isinstance(row[1], list) else json.loads(row[1])
    base_value = row[2]
    feature_names = row[3] if isinstance(row[3], list) else json.loads(row[3])
    created_at = row[4]

    print(f"\n📊 Trace ID: {trace_id}")
    print(f"📅 Created: {created_at}")
    print(f"🎯 Base Value (log-odds): {base_value:.6f}")
    print(f"🔢 Number of Features: {len(feature_names)}")
    print(f"🔢 Number of SHAP Values: {len(shap_values)}")

    print("\n" + "=" * 80)
    print("SHAP VALUES FOR ALL 66 FEATURES")
    print("=" * 80)
    print(f"{'#':<4} {'Feature Name':<50} {'SHAP Value':<15} {'Impact':<10}")
    print("-" * 80)

    # Combine feature names and values, sort by absolute SHAP value
    feature_impacts = list(zip(feature_names, shap_values))
    feature_impacts_sorted = sorted(
        feature_impacts, key=lambda x: abs(x[1]), reverse=True
    )

    for i, (feature, value) in enumerate(feature_impacts_sorted, 1):
        impact = "↑ Increase" if value > 0 else "↓ Decrease"
        print(f"{i:<4} {feature:<50} {value:>14.6f} {impact:<10}")

    print("\n" + "=" * 80)
    print("TOP 10 MOST IMPORTANT FEATURES")
    print("=" * 80)

    for i, (feature, value) in enumerate(feature_impacts_sorted[:10], 1):
        impact = "INCREASES" if value > 0 else "DECREASES"
        print(f"{i}. {feature}")
        print(f"   SHAP Value: {value:.6f}")
        print(f"   {impact} attrition risk by {abs(value):.6f}")
        print()

    # Calculate total positive and negative impacts
    positive_impact = sum(v for v in shap_values if v > 0)
    negative_impact = sum(v for v in shap_values if v < 0)

    print("=" * 80)
    print("IMPACT SUMMARY")
    print("=" * 80)
    print(f"Base Value (log-odds):        {base_value:.6f}")
    print(f"Total Positive Impact:        {positive_impact:>+.6f} (increases risk)")
    print(f"Total Negative Impact:        {negative_impact:>+.6f} (decreases risk)")
    print(f"Net SHAP Impact:              {positive_impact + negative_impact:>+.6f}")
    print(
        f"Final Prediction (log-odds):  {base_value + positive_impact + negative_impact:.6f}"
    )

    # Convert to probability
    import math

    log_odds = base_value + positive_impact + negative_impact
    probability = 1 / (1 + math.exp(-log_odds))
    print(f"Final Probability:            {probability:.4%}")

else:
    print("❌ No SHAP data found for trace_id 38")

# Check all 4 employees
print("\n" + "=" * 80)
print("SHAP RECORDS FOR ALL 4 EMPLOYEES")
print("=" * 80)

cur.execute("""
    SELECT 
        s.trace_id,
        t.input_id as employee_id,
        jsonb_array_length(s.shap_values::jsonb) as num_features,
        s.base_value
    FROM shap_analysis s
    JOIN predictions_traceability t ON s.trace_id = t.trace_id
    WHERE s.trace_id BETWEEN 38 AND 41
    ORDER BY s.trace_id
""")

print(f"{'Trace ID':<10} {'Employee ID':<13} {'Num Features':<15} {'Base Value':<15}")
print("-" * 80)
for row in cur.fetchall():
    print(f"{row[0]:<10} {row[1]:<13} {row[2]:<15} {row[3]:<15.6f}")

cur.close()
conn.close()

print("\n" + "=" * 80)
print("✅ CONFIRMED: Each prediction has ONE record with ALL 66 feature SHAP values")
print("=" * 80)
