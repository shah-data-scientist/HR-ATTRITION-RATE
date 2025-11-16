"""
Automated End-to-End Test for HR Attrition Prediction System
Tests the full workflow: data upload → API prediction → result validation
"""

import json
import os
from pathlib import Path

import pandas as pd
import pytest
import httpx


# Test configuration
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8001")
DATA_DIR = Path(__file__).parent.parent / "data"
TIMEOUT = 120.0


@pytest.fixture(scope="module")
def api_client():
    """Create HTTP client for API testing."""
    return httpx.Client(base_url=API_BASE_URL, timeout=TIMEOUT)


@pytest.fixture(scope="module")
def test_data():
    """Load test data from data/ folder."""
    eval_df = pd.read_csv(DATA_DIR / "extrait_eval.csv")
    sirh_df = pd.read_csv(DATA_DIR / "extrait_sirh.csv")
    sondage_df = pd.read_csv(DATA_DIR / "extrait_sondage.csv")

    return {
        "eval": eval_df,
        "sirh": sirh_df,
        "sondage": sondage_df,
    }


class TestAPIHealthAndConnectivity:
    """Test API availability and basic connectivity."""

    def test_api_health_check(self, api_client):
        """Test that API health endpoint responds correctly."""
        response = api_client.get("/health")
        assert response.status_code == 200, f"API health check failed: {response.text}"

        data = response.json()
        assert data["status"] == "ok", "API status is not ok"
        assert "message" in data, "Health response missing message"

        print("✓ API is healthy and responding")


class TestDataUploadAndPrediction:
    """Test data upload and prediction functionality."""

    def test_predict_with_sample_data(self, api_client, test_data):
        """Test prediction with first 5 rows of data."""
        # Use first 5 rows for quick test
        eval_data = test_data["eval"].head(5).to_dict(orient="records")
        sirh_data = test_data["sirh"].head(5).to_dict(orient="records")
        sondage_data = test_data["sondage"].head(5).to_dict(orient="records")

        payload = {
            "eval_data": eval_data,
            "sirh_data": sirh_data,
            "sondage_data": sondage_data,
        }

        # Send prediction request
        response = api_client.post("/predict", json=payload)

        # Check response
        assert (
            response.status_code == 200
        ), f"Prediction failed with status {response.status_code}: {response.text}"

        result = response.json()
        assert "predictions" in result, "Response missing predictions key"

        predictions = result["predictions"]
        assert len(predictions) > 0, "No predictions returned"
        assert len(predictions) <= 5, "More predictions than expected"

        print(f"✓ Successfully received {len(predictions)} predictions")

        return predictions

    def test_prediction_structure(self, api_client, test_data):
        """Test that predictions have correct structure and required fields."""
        # Use first row for detailed validation
        eval_data = test_data["eval"].head(1).to_dict(orient="records")
        sirh_data = test_data["sirh"].head(1).to_dict(orient="records")
        sondage_data = test_data["sondage"].head(1).to_dict(orient="records")

        payload = {
            "eval_data": eval_data,
            "sirh_data": sirh_data,
            "sondage_data": sondage_data,
        }

        response = api_client.post("/predict", json=payload)
        result = response.json()
        prediction = result["predictions"][0]

        # Validate required fields
        required_fields = [
            "id_employee",
            "prediction",
            "probability",
            "risk_category",
            "message",
            "trace_id",
        ]
        for field in required_fields:
            assert field in prediction, f"Missing required field: {field}"

        # Validate field types and values
        assert isinstance(prediction["id_employee"], int), "id_employee should be int"
        assert prediction["prediction"] in [
            "Stay",
            "Leave",
        ], f"Invalid prediction: {prediction['prediction']}"
        assert (
            0.0 <= prediction["probability"] <= 1.0
        ), f"Probability out of range: {prediction['probability']}"
        assert prediction["risk_category"] in [
            "Low",
            "Medium",
            "High",
        ], f"Invalid risk category: {prediction['risk_category']}"
        assert isinstance(prediction["trace_id"], int), "trace_id should be int"

        print(
            f"✓ Prediction structure validated for employee {prediction['id_employee']}"
        )
        print(f"  Prediction: {prediction['prediction']}")
        print(f"  Probability: {prediction['probability']:.2%}")
        print(f"  Risk: {prediction['risk_category']}")

    def test_shap_values_present(self, api_client, test_data):
        """Test that SHAP values are included in predictions."""
        eval_data = test_data["eval"].head(1).to_dict(orient="records")
        sirh_data = test_data["sirh"].head(1).to_dict(orient="records")
        sondage_data = test_data["sondage"].head(1).to_dict(orient="records")

        payload = {
            "eval_data": eval_data,
            "sirh_data": sirh_data,
            "sondage_data": sondage_data,
        }

        response = api_client.post("/predict", json=payload)
        result = response.json()
        prediction = result["predictions"][0]

        # Validate SHAP fields
        assert "shap_values" in prediction, "Missing shap_values"
        assert "base_value" in prediction, "Missing base_value"
        assert "feature_names" in prediction, "Missing feature_names"

        assert isinstance(prediction["shap_values"], list), "shap_values should be list"
        assert len(prediction["shap_values"]) > 0, "shap_values should not be empty"
        assert isinstance(
            prediction["base_value"], (int, float)
        ), "base_value should be numeric"
        assert isinstance(
            prediction["feature_names"], list
        ), "feature_names should be list"

        # Validate consistency
        assert len(prediction["shap_values"]) == len(
            prediction["feature_names"]
        ), "SHAP values and feature names length mismatch"

        print(f"✓ SHAP values validated: {len(prediction['shap_values'])} features")
        print(f"  Base value: {prediction['base_value']:.4f}")


class TestFullDataset:
    """Test with full dataset from data/ folder."""

    @pytest.mark.slow
    def test_predict_with_full_dataset(self, api_client, test_data):
        """Test prediction with complete dataset (marked as slow test)."""
        eval_data = test_data["eval"].to_dict(orient="records")
        sirh_data = test_data["sirh"].to_dict(orient="records")
        sondage_data = test_data["sondage"].to_dict(orient="records")

        print(f"\nProcessing full dataset:")
        print(f"  Eval records: {len(eval_data)}")
        print(f"  SIRH records: {len(sirh_data)}")
        print(f"  Sondage records: {len(sondage_data)}")

        payload = {
            "eval_data": eval_data,
            "sirh_data": sirh_data,
            "sondage_data": sondage_data,
        }

        # Send prediction request
        response = api_client.post("/predict", json=payload)

        assert (
            response.status_code == 200
        ), f"Full dataset prediction failed: {response.text}"

        result = response.json()
        predictions = result["predictions"]

        # Validate results
        assert len(predictions) > 0, "No predictions returned for full dataset"

        # Calculate statistics
        leave_count = sum(1 for p in predictions if p["prediction"] == "Leave")
        stay_count = sum(1 for p in predictions if p["prediction"] == "Stay")

        high_risk = sum(1 for p in predictions if p["risk_category"] == "High")
        medium_risk = sum(1 for p in predictions if p["risk_category"] == "Medium")
        low_risk = sum(1 for p in predictions if p["risk_category"] == "Low")

        avg_probability = sum(p["probability"] for p in predictions) / len(predictions)

        print(f"\n✓ Full dataset prediction successful!")
        print(f"\nResults Summary:")
        print(f"  Total predictions: {len(predictions)}")
        print(
            f"  Predicted to Leave: {leave_count} ({leave_count/len(predictions)*100:.1f}%)"
        )
        print(
            f"  Predicted to Stay: {stay_count} ({stay_count/len(predictions)*100:.1f}%)"
        )
        print(f"\nRisk Distribution:")
        print(f"  High Risk: {high_risk} ({high_risk/len(predictions)*100:.1f}%)")
        print(f"  Medium Risk: {medium_risk} ({medium_risk/len(predictions)*100:.1f}%)")
        print(f"  Low Risk: {low_risk} ({low_risk/len(predictions)*100:.1f}%)")
        print(f"\nAverage Attrition Probability: {avg_probability:.2%}")

        # Save detailed results
        output_file = Path(__file__).parent.parent / "test_results_full.json"
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n✓ Full results saved to: {output_file}")

        return predictions


class TestErrorHandling:
    """Test error handling and validation."""

    def test_invalid_data_format(self, api_client):
        """Test that API properly validates invalid data."""
        # Send invalid payload (missing required fields)
        invalid_payload = {
            "eval_data": [{"invalid": "data"}],
            "sirh_data": [],
            "sondage_data": [],
        }

        response = api_client.post("/predict", json=invalid_payload)

        # Should return validation error (422)
        assert response.status_code in [
            422,
            500,
        ], "API should return validation error for invalid data"

        print("✓ API correctly validates invalid data")

    def test_empty_dataset(self, api_client):
        """Test prediction with empty dataset."""
        empty_payload = {
            "eval_data": [],
            "sirh_data": [],
            "sondage_data": [],
        }

        response = api_client.post("/predict", json=empty_payload)

        # API should handle empty data gracefully
        # Either return empty predictions or validation error
        assert response.status_code in [
            200,
            422,
        ], f"Unexpected status for empty data: {response.status_code}"

        if response.status_code == 200:
            result = response.json()
            assert "predictions" in result
            print("✓ API handles empty dataset gracefully")


def test_summary(api_client, test_data):
    """Generate a comprehensive test summary."""
    print("\n" + "=" * 70)
    print("AUTOMATED TEST SUITE SUMMARY")
    print("=" * 70)

    # Get data counts
    eval_count = len(test_data["eval"])
    sirh_count = len(test_data["sirh"])
    sondage_count = len(test_data["sondage"])

    print(f"\nTest Data:")
    print(f"  Eval records: {eval_count}")
    print(f"  SIRH records: {sirh_count}")
    print(f"  Sondage records: {sondage_count}")

    print(f"\nAPI Configuration:")
    print(f"  Base URL: {API_BASE_URL}")
    print(f"  Timeout: {TIMEOUT}s")

    print(f"\nData Location:")
    print(f"  {DATA_DIR}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    """Run tests directly with detailed output."""
    pytest.main(
        [
            __file__,
            "-v",  # Verbose output
            "-s",  # Show print statements
            "--tb=short",  # Short traceback format
        ]
    )
