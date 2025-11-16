"""Quick test for download endpoints"""
import httpx
import json
import base64
import csv

API_BASE_URL = "http://localhost:8001"

# Load test data as list of dicts (matching RawBatchPredictionInput schema)
def load_csv_as_dicts(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=';')
        return [row for row in reader]

eval_data = load_csv_as_dicts("data/extrait_eval.csv")
sirh_data = load_csv_as_dicts("data/extrait_sirh.csv")
sondage_data = load_csv_as_dicts("data/extrait_sondage.csv")

# Create payload
payload = {
    "eval_data": eval_data,
    "sirh_data": sirh_data,
    "sondage_data": sondage_data,
}

print("Testing /predict_excel...")
try:
    resp = httpx.post(f"{API_BASE_URL}/predict_excel", json=payload, timeout=60.0)
    resp.raise_for_status()
    result = resp.json()
    excel_b64 = result.get("excel_base64")
    if excel_b64:
        excel_bytes = base64.b64decode(excel_b64)
        print(f"✅ Excel report generated: {len(excel_bytes)} bytes")
    else:
        print("❌ No excel_base64 in response")
except Exception as e:
    print(f"❌ Excel endpoint failed: {e}")

print("\nTesting /predict_shap_images...")
try:
    resp = httpx.post(f"{API_BASE_URL}/predict_shap_images", json=payload, timeout=60.0)
    resp.raise_for_status()
    result = resp.json()
    images = result.get("shap_images", [])
    print(f"✅ SHAP images generated: {len(images)} images")
except Exception as e:
    print(f"❌ SHAP endpoint failed: {e}")
