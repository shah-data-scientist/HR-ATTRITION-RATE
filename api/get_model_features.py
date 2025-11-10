import joblib
import pandas as pd
import sys
import os

# Add the project root to the sys.path to allow importing app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


# Define get_expected_columns function (copied from app.py)
def get_expected_columns(pipeline):
    """Gets the list of columns the model was trained on."""
    preprocessor = pipeline.named_steps["preprocessor"]
    # The feature_names_in_ attribute stores the names of features seen during fit
    return list(preprocessor.feature_names_in_)


# Define load_model_and_data function (copied from app.py, simplified for model loading only)
def load_model_only():
    """Loads only the trained model."""
    model = joblib.load("outputs/employee_attrition_pipeline.pkl")
    return model


if __name__ == "__main__":
    model = load_model_only()
    expected_cols = get_expected_columns(model)
    print("Expected Columns for Model Input:")
    for col in expected_cols:
        print(col)
