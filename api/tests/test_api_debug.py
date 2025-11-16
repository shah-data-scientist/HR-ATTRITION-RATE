import pytest
from fastapi.testclient import TestClient
from api.app.main import app
import pandas as pd
import os
import sys
import io
import json

# Add the project root to the sys.path to allow importing modules from the project
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

# Assuming create_synthetic_data.py is in the root of the project
# from create_synthetic_data import create_synthetic_data

client = TestClient(app)

@pytest.fixture(scope="module")
def synthetic_data():
    """Loads synthetic data for testing."""
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

    eval_file_path = os.path.join(data_dir, "extrait_eval.csv")
    sirh_file_path = os.path.join(data_dir, "extrait_sirh.csv")
    sondage_file_path = os.path.join(data_dir, "extrait_sondage.csv")

    eval_df = pd.read_csv(eval_file_path)
    sirh_df = pd.read_csv(sirh_file_path)
    sondage_df = pd.read_csv(sondage_file_path)

    eval_data_for_api = eval_df.to_dict(orient="records")
    sirh_data_for_api = sirh_df.to_dict(orient="records")
    sondage_data_for_api = sondage_df.to_dict(orient="records")

    return {
        "eval_data": eval_data_for_api,
        "sirh_data": sirh_data_for_api,
        "sondage_data": sondage_data_for_api,
    }

def test_predict_attrition_with_raw_data(synthetic_data):
    """
    Tests the /predict endpoint with raw, unmerged data,
    expecting the API to perform the merging and prediction.
    """
    print("\n--- Starting API Debug Test ---")
    print(f"Synthetic data payload: {json.dumps(synthetic_data, indent=2)}")

    with TestClient(app) as client:
        response = client.post("/predict", json=synthetic_data)

    print("\n--- API Response ---")
    print(f"Status Code: {response.status_code}")
    print(f"Response Body: {json.dumps(response.json(), indent=2)}")
    print("--- API Debug Test Finished ---")

    assert response.status_code == 200
    assert "predictions" in response.json()
    assert len(response.json()["predictions"]) > 0
    assert "id_employee" in response.json()["predictions"][0]
    assert "prediction" in response.json()["predictions"][0]
    assert "probability" in response.json()["predictions"][0]
    assert "risk_category" in response.json()["predictions"][0]
    assert "shap_values" in response.json()["predictions"][0]
    assert "base_value" in response.json()["predictions"][0]
