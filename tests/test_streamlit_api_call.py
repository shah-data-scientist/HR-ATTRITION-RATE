import pytest
import pandas as pd
import os
import sys
import json

# Add the project root to the sys.path to allow importing modules from the project
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the functions from ui/app.py
from ui.app import _call_prediction_api, _load_local_csv_files, get_project_root


@pytest.fixture(scope="module")
def raw_synthetic_data():
    """Loads raw synthetic data for API call simulation."""
    eval_file, sirh_file, sondage_file = _load_local_csv_files()

    eval_df = pd.read_csv(eval_file)
    sirh_df = pd.read_csv(sirh_file)
    sondage_df = pd.read_csv(sondage_file)

    eval_data_for_api = eval_df.to_dict(orient="records")
    sirh_data_for_api = sirh_df.to_dict(orient="records")
    sondage_data_for_api = sondage_df.to_dict(orient="records")

    return {
        "eval_data": eval_data_for_api,
        "sirh_data": sirh_data_for_api,
        "sondage_data": sondage_data_for_api,
    }


def test_streamlit_api_call_success(raw_synthetic_data):
    """
    Tests the _call_prediction_api function from ui/app.py directly.
    """
    print("\n--- Starting Streamlit API Call Test ---")
    print(f"Raw synthetic data payload: {json.dumps(raw_synthetic_data, indent=2)}")

    try:
        api_response = _call_prediction_api(
            raw_synthetic_data["eval_data"],
            raw_synthetic_data["sirh_data"],
            raw_synthetic_data["sondage_data"],
        )

        print("\n--- API Response received by _call_prediction_api ---")
        print(f"Response: {json.dumps(api_response, indent=2)}")
        print("--- Streamlit API Call Test Finished ---")

        assert api_response is not None
        assert "predictions" in api_response
        assert len(api_response["predictions"]) > 0
        assert "id_employee" in api_response["predictions"][0]
        assert "prediction" in api_response["predictions"][0]
        assert "probability" in api_response["predictions"][0]
        assert "risk_category" in api_response["predictions"][0]
        assert "shap_values" in api_response["predictions"][0]
        assert "base_value" in api_response["predictions"][0]

    except Exception as e:
        pytest.fail(f"An unexpected exception occurred during API call: {e}")
