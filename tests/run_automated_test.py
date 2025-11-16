#!/usr/bin/env python3
"""
Simple Test Runner - Tests API with data from data/ folder
No pytest required - can be run directly: python tests/run_automated_test.py
"""
import json
import os
import sys
from pathlib import Path

import httpx
import pandas as pd


# Configuration
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8001")
DATA_DIR = Path(__file__).parent.parent / "data"
TIMEOUT = 120.0


def print_header(title):
    """Print formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def print_test(name):
    """Print test name."""
    print(f"\n[TEST] {name}")
    print("-" * 70)


def main():
    """Run automated tests."""
    print_header("HR ATTRITION - AUTOMATED TEST SUITE")
    
    # Initialize client
    client = httpx.Client(base_url=API_BASE_URL, timeout=TIMEOUT)
    
    # Test 1: Health Check
    print_test("API Health Check")
    try:
        response = client.get("/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ API is healthy: {data['message']}")
        else:
            print(f"✗ Health check failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Failed to connect to API: {e}")
        print(f"  Make sure API is running at: {API_BASE_URL}")
        return False
    
    # Test 2: Load Test Data
    print_test("Loading Test Data")
    try:
        eval_df = pd.read_csv(DATA_DIR / "extrait_eval.csv")
        sirh_df = pd.read_csv(DATA_DIR / "extrait_sirh.csv")
        sondage_df = pd.read_csv(DATA_DIR / "extrait_sondage.csv")
        
        print(f"✓ Loaded extrait_eval.csv: {len(eval_df)} rows")
        print(f"✓ Loaded extrait_sirh.csv: {len(sirh_df)} rows")
        print(f"✓ Loaded extrait_sondage.csv: {len(sondage_df)} rows")
    except Exception as e:
        print(f"✗ Failed to load test data: {e}")
        return False
    
    # Test 3: Sample Prediction (5 rows)
    print_test("Prediction with Sample Data (5 rows)")
    try:
        eval_data = eval_df.head(5).to_dict(orient="records")
        sirh_data = sirh_df.head(5).to_dict(orient="records")
        sondage_data = sondage_df.head(5).to_dict(orient="records")
        
        payload = {
            "eval_data": eval_data,
            "sirh_data": sirh_data,
            "sondage_data": sondage_data,
        }
        
        response = client.post("/predict", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            predictions = result["predictions"]
            print(f"✓ Received {len(predictions)} predictions")
            
            # Show sample prediction
            if predictions:
                pred = predictions[0]
                print(f"\n  Sample Result (Employee {pred['id_employee']}):")
                print(f"    Prediction: {pred['prediction']}")
                print(f"    Probability: {pred['probability']:.2%}")
                print(f"    Risk Category: {pred['risk_category']}")
                print(f"    Trace ID: {pred['trace_id']}")
                
                # Check SHAP values
                if pred.get('shap_values'):
                    print(f"    SHAP Values: {len(pred['shap_values'])} features")
                    print(f"    Base Value: {pred.get('base_value', 'N/A')}")
                else:
                    print("    ⚠ Warning: No SHAP values")
        else:
            print(f"✗ Prediction failed with status {response.status_code}")
            print(f"  Response: {response.text[:500]}")
            return False
            
    except Exception as e:
        print(f"✗ Sample prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Validate Prediction Structure
    print_test("Validating Prediction Structure")
    try:
        required_fields = [
            "id_employee", "prediction", "probability",
            "risk_category", "message", "trace_id",
            "shap_values", "base_value", "feature_names"
        ]
        
        pred = predictions[0]
        missing_fields = [f for f in required_fields if f not in pred]
        
        if missing_fields:
            print(f"✗ Missing fields: {missing_fields}")
            return False
        
        # Validate values
        if pred["prediction"] not in ["Stay", "Leave"]:
            print(f"✗ Invalid prediction: {pred['prediction']}")
            return False
        
        if not (0.0 <= pred["probability"] <= 1.0):
            print(f"✗ Probability out of range: {pred['probability']}")
            return False
        
        if pred["risk_category"] not in ["Low", "Medium", "High"]:
            print(f"✗ Invalid risk category: {pred['risk_category']}")
            return False
        
        print("✓ All required fields present and valid")
        print("✓ Prediction values within expected ranges")
        print("✓ SHAP explanations included")
        
    except Exception as e:
        print(f"✗ Validation failed: {e}")
        return False
    
    # Test 5: Full Dataset
    print_test("Full Dataset Prediction")
    try:
        full_eval = eval_df.to_dict(orient="records")
        full_sirh = sirh_df.to_dict(orient="records")
        full_sondage = sondage_df.to_dict(orient="records")
        
        print(f"  Processing {len(full_eval)} employees...")
        
        payload_full = {
            "eval_data": full_eval,
            "sirh_data": full_sirh,
            "sondage_data": full_sondage,
        }
        
        response = client.post("/predict", json=payload_full)
        
        if response.status_code == 200:
            result = response.json()
            predictions_full = result["predictions"]
            
            # Calculate statistics
            total = len(predictions_full)
            leave_count = sum(1 for p in predictions_full if p["prediction"] == "Leave")
            stay_count = sum(1 for p in predictions_full if p["prediction"] == "Stay")
            
            high_risk = sum(1 for p in predictions_full if p["risk_category"] == "High")
            medium_risk = sum(1 for p in predictions_full if p["risk_category"] == "Medium")
            low_risk = sum(1 for p in predictions_full if p["risk_category"] == "Low")
            
            avg_prob = sum(p["probability"] for p in predictions_full) / total
            
            print(f"✓ Successfully processed {total} employees\n")
            
            print("  Prediction Distribution:")
            print(f"    Leave: {leave_count} ({leave_count/total*100:.1f}%)")
            print(f"    Stay:  {stay_count} ({stay_count/total*100:.1f}%)")
            
            print("\n  Risk Distribution:")
            print(f"    High:   {high_risk} ({high_risk/total*100:.1f}%)")
            print(f"    Medium: {medium_risk} ({medium_risk/total*100:.1f}%)")
            print(f"    Low:    {low_risk} ({low_risk/total*100:.1f}%)")
            
            print(f"\n  Average Attrition Probability: {avg_prob:.2%}")
            
            # Save results
            output_file = Path(__file__).parent.parent / "test_results_automated.json"
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\n✓ Full results saved to: {output_file}")
            
        else:
            print(f"✗ Full dataset prediction failed: {response.status_code}")
            print(f"  Response: {response.text[:500]}")
            return False
            
    except Exception as e:
        print(f"✗ Full dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Success Summary
    print_header("ALL TESTS PASSED ✓")
    print("\nTest Summary:")
    print(f"  ✓ API Health Check")
    print(f"  ✓ Data Loading ({len(eval_df)} employees)")
    print(f"  ✓ Sample Prediction (5 employees)")
    print(f"  ✓ Structure Validation")
    print(f"  ✓ Full Dataset Prediction ({len(predictions_full)} employees)")
    print(f"\nResults saved to: test_results_automated.json")
    print("="*70 + "\n")
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
