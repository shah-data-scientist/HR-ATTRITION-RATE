"""
End-to-End Testing Script for HR Attrition API
This script tests the full workflow from CSV loading to prediction.
"""
import requests
import pandas as pd
import json
import sys

print("="*70)
print("HR ATTRITION API - END-TO-END TEST")
print("="*70)

# Test 1: Health Check
print("\n[TEST 1] Health Check")
print("-" * 70)
try:
    response = requests.get("http://localhost:8001/health", timeout=5)
    if response.status_code == 200:
        print("✓ API is healthy")
        print(f"  Response: {response.json()}")
    else:
        print(f"✗ Health check failed with status {response.status_code}")
        sys.exit(1)
except Exception as e:
    print(f"✗ Failed to connect to API: {e}")
    print("  Make sure the API is running on port 8001")
    sys.exit(1)

# Test 2: Load CSV files
print("\n[TEST 2] Loading CSV Files")
print("-" * 70)
try:
    eval_df = pd.read_csv("data/extrait_eval.csv")
    sirh_df = pd.read_csv("data/extrait_sirh.csv")
    sondage_df = pd.read_csv("data/extrait_sondage.csv")

    print(f"✓ Loaded extrait_eval.csv: {len(eval_df)} rows")
    print(f"✓ Loaded extrait_sirh.csv: {len(sirh_df)} rows")
    print(f"✓ Loaded extrait_sondage.csv: {len(sondage_df)} rows")
except Exception as e:
    print(f"✗ Failed to load CSV files: {e}")
    sys.exit(1)

# Test 3: Convert to API format (simulating Streamlit app)
print("\n[TEST 3] Converting Data to API Format")
print("-" * 70)
try:
    # Use only first 5 rows for faster testing
    eval_data = eval_df.head(5).to_dict(orient="records")
    sirh_data = sirh_df.head(5).to_dict(orient="records")
    sondage_data = sondage_df.head(5).to_dict(orient="records")

    print(f"✓ Converted eval data: {len(eval_data)} records")
    print(f"✓ Converted sirh data: {len(sirh_data)} records")
    print(f"✓ Converted sondage data: {len(sondage_data)} records")

    # Show sample record
    print("\n  Sample eval record:")
    print(f"    {json.dumps(eval_data[0], indent=6)}")

except Exception as e:
    print(f"✗ Failed to convert data: {e}")
    sys.exit(1)

# Test 4: Send prediction request
print("\n[TEST 4] Sending Prediction Request")
print("-" * 70)
try:
    payload = {
        "eval_data": eval_data,
        "sirh_data": sirh_data,
        "sondage_data": sondage_data,
    }

    # Save payload for debugging
    with open("test_payload.json", "w") as f:
        json.dump(payload, f, indent=2)
    print("  Payload saved to test_payload.json")

    response = requests.post(
        "http://localhost:8001/predict",
        json=payload,
        timeout=60.0
    )

    print(f"\n  Status Code: {response.status_code}")

    if response.status_code == 200:
        print("✓ Prediction successful!")
        result = response.json()

        # Save response for debugging
        with open("test_response.json", "w") as f:
            json.dump(result, f, indent=2)
        print("  Response saved to test_response.json")

    elif response.status_code == 422:
        print("✗ Validation Error (422)")
        print("\n  Error Details:")
        error_detail = response.json()
        print(json.dumps(error_detail, indent=4))

        # Analyze the validation error
        print("\n  Analysis:")
        if "detail" in error_detail:
            for error in error_detail["detail"]:
                print(f"    - Field: {error.get('loc', 'unknown')}")
                print(f"      Error: {error.get('msg', 'unknown')}")
                print(f"      Type: {error.get('type', 'unknown')}")
        sys.exit(1)
    else:
        print(f"✗ Request failed with status {response.status_code}")
        print(f"  Response: {response.text}")
        sys.exit(1)

except requests.exceptions.Timeout:
    print("✗ Request timed out")
    sys.exit(1)
except Exception as e:
    print(f"✗ Failed to send request: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Validate predictions
print("\n[TEST 5] Validating Predictions")
print("-" * 70)
try:
    predictions = result["predictions"]
    print(f"✓ Received {len(predictions)} predictions")

    # Check each prediction
    for i, pred in enumerate(predictions, 1):
        print(f"\n  Employee {i} (ID: {pred['id_employee']}):")
        print(f"    Prediction: {pred['prediction']}")
        print(f"    Probability: {pred['probability']:.2%}")
        print(f"    Risk Category: {pred['risk_category']}")
        print(f"    Trace ID: {pred['trace_id']}")

        # Validate SHAP values
        if pred.get('shap_values'):
            print(f"    SHAP values: {len(pred['shap_values'])} features")
            print(f"    Base value: {pred.get('base_value', 'N/A')}")
        else:
            print("    ⚠ Warning: No SHAP values in response")

    print("\n✓ All predictions validated successfully!")

except Exception as e:
    print(f"✗ Failed to validate predictions: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test with full dataset
print("\n[TEST 6] Testing with Full Dataset")
print("-" * 70)
response = input("Do you want to test with the full dataset? (y/n): ")
if response.lower() == 'y':
    try:
        print("  Processing full dataset...")
        eval_data_full = eval_df.to_dict(orient="records")
        sirh_data_full = sirh_df.to_dict(orient="records")
        sondage_data_full = sondage_df.to_dict(orient="records")

        payload_full = {
            "eval_data": eval_data_full,
            "sirh_data": sirh_data_full,
            "sondage_data": sondage_data_full,
        }

        response = requests.post(
            "http://localhost:8001/predict",
            json=payload_full,
            timeout=120.0
        )

        if response.status_code == 200:
            result_full = response.json()
            print(f"✓ Full dataset prediction successful!")
            print(f"  Total predictions: {len(result_full['predictions'])}")

            # Save full response
            with open("test_response_full.json", "w") as f:
                json.dump(result_full, f, indent=2)
            print("  Full response saved to test_response_full.json")
        else:
            print(f"✗ Full dataset test failed with status {response.status_code}")

    except Exception as e:
        print(f"✗ Full dataset test error: {e}")
else:
    print("  Skipped full dataset test")

print("\n" + "="*70)
print("END-TO-END TEST COMPLETED SUCCESSFULLY!")
print("="*70)
