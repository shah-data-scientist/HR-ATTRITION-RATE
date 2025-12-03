"""
Simulates what the Streamlit app does when you click "Predict Attrition"
This test uses the actual data files from the data/ folder
"""

import httpx
import pandas as pd
import json
import pytest


def test_streamlit_simulation():
    """Test the full Streamlit workflow with actual data files."""
    print("=" * 70)
    print("STREAMLIT APP SIMULATION TEST")
    print("=" * 70)

    # Step 1: Load the CSV files from data/ folder (what Streamlit does)
    print("\n[Step 1] Loading CSV files from data/ folder...")
    try:
        eval_df = pd.read_csv("data/extrait_eval.csv")
        sirh_df = pd.read_csv("data/extrait_sirh.csv")
        sondage_df = pd.read_csv("data/extrait_sondage.csv")

        print(f"[OK] Loaded eval_df: {len(eval_df)} rows, columns: {list(eval_df.columns)}")
        print(f"[OK] Loaded sirh_df: {len(sirh_df)} rows, columns: {list(sirh_df.columns)}")
        print(f"[OK] Loaded sondage_df: {len(sondage_df)} rows, columns: {list(sondage_df.columns)}")
    except Exception as e:
        print(f"[ERROR] Error loading files: {e}")
        pytest.fail(f"Failed to load CSV files: {e}")

    # Step 2: Convert to API format (what Streamlit does at lines 187-189)
    print("\n[Step 2] Converting DataFrames to API format...")
    try:
        eval_data_for_api = eval_df.to_dict(orient="records")
        sirh_data_for_api = sirh_df.to_dict(orient="records")
        sondage_data_for_api = sondage_df.to_dict(orient="records")

        print(f"[OK] Converted eval data: {len(eval_data_for_api)} records")
        print(f"[OK] Converted sirh data: {len(sirh_data_for_api)} records")
        print(f"[OK] Converted sondage data: {len(sondage_data_for_api)} records")

        # Show first record of each
        print("\nFirst record of eval_data:")
        print(json.dumps(eval_data_for_api[0], indent=2))
        print("\nFirst record of sirh_data:")
        print(json.dumps(sirh_data_for_api[0], indent=2))
        print("\nFirst record of sondage_data:")
        print(json.dumps(sondage_data_for_api[0], indent=2))

    except Exception as e:
        print(f"[ERROR] Error converting data: {e}")
        pytest.fail(f"Failed to convert data: {e}")

    # Step 3: Call the API (what Streamlit does at lines 191-193)
    print("\n[Step 3] Calling the prediction API...")
    API_BASE_URL = "http://localhost:8001"

    try:
        # Use only first 5 records for faster testing
        payload = {
            "eval_data": eval_data_for_api[:5],
            "sirh_data": sirh_data_for_api[:5],
            "sondage_data": sondage_data_for_api[:5],
        }

        # Save payload for inspection
        with open("tests/fixtures/streamlit_simulation_payload.json", "w") as f:
            json.dump(payload, f, indent=2)
        print("[OK] Payload saved to tests/fixtures/streamlit_simulation_payload.json")

        response = httpx.post(f"{API_BASE_URL}/predict", json=payload, timeout=60.0)

        print(f"\nAPI Response Status: {response.status_code}")

        if response.status_code == 200:
            api_response = response.json()

            # Save response
            with open("tests/fixtures/streamlit_simulation_response.json", "w") as f:
                json.dump(api_response, f, indent=2)
            print("[OK] Response saved to tests/fixtures/streamlit_simulation_response.json")

            print(f"\n[OK] SUCCESS! Received {len(api_response['predictions'])} predictions")

            # Show summary
            for i, pred in enumerate(api_response["predictions"], 1):
                print(f"\nEmployee {i}:")
                print(f"  ID: {pred['id_employee']}")
                print(f"  Prediction: {pred['prediction']}")
                print(f"  Probability: {pred['probability']:.2%}")
                print(f"  Risk: {pred['risk_category']}")
                shap_values = pred.get("shap_values", [])
                if shap_values:
                    print(f"  SHAP values: {len(shap_values)} features")

            print("\n" + "=" * 70)
            print("TEST PASSED - The data/ folder files work correctly!")
            print("=" * 70)

        elif response.status_code == 422:
            print("\n[ERROR] VALIDATION ERROR (422)")
            error_detail = response.json()

            # Save error for analysis
            with open("tests/fixtures/streamlit_simulation_error.json", "w") as f:
                json.dump(error_detail, f, indent=2)
            print("[ERROR] Error details saved to tests/fixtures/streamlit_simulation_error.json")

            # Analyze errors
            print("\nValidation Errors:")
            if "detail" in error_detail:
                # Group errors by location
                errors_by_loc = {}
                for error in error_detail["detail"]:
                    loc = " -> ".join(str(l) for l in error.get("loc", []))
                    if loc not in errors_by_loc:
                        errors_by_loc[loc] = []
                    errors_by_loc[loc].append(error.get("msg", "unknown"))

                for loc, msgs in errors_by_loc.items():
                    print(f"\n  Location: {loc}")
                    for msg in msgs:
                        print(f"    - {msg}")

            print("\n" + "=" * 70)
            print("TEST FAILED - Schema validation error")
            print("=" * 70)
            pytest.fail(f"API validation error (422): {error_detail}")
        else:
            print(f"\n[ERROR] Unexpected status code: {response.status_code}")
            print(f"Response: {response.text}")
            pytest.fail(f"Unexpected status code: {response.status_code}")

    except httpx.TimeoutException as e:
        print(f"\n[ERROR] Timeout: {e}")
        pytest.skip(f"API timeout - is the API running? {e}")
    except httpx.RequestError as e:
        print(f"\n[ERROR] Network error: {e}")
        print("Make sure the API is running on http://localhost:8001")
        pytest.skip(f"Cannot connect to API - is it running on http://localhost:8001? {e}")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        pytest.fail(f"Unexpected error: {e}")
