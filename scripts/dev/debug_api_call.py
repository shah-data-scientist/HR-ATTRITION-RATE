import pandas as pd
import httpx
import os
import sys
import io

# Add the project root to the sys.path to allow importing modules from the project
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

API_BASE_URL = "http://localhost:8001"


def get_project_root():
    """Returns the absolute path to the project root (one level up from scripts)."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_local_csv_files():
    """Loads the required CSV files from the local 'data' directory."""
    try:
        project_root = get_project_root()
        eval_file_path = os.path.join(project_root, "data", "extrait_eval.csv")
        sirh_file_path = os.path.join(project_root, "data", "extrait_sirh.csv")
        sondage_file_path = os.path.join(project_root, "data", "extrait_sondage.csv")

        eval_df = pd.read_csv(eval_file_path)
        sirh_df = pd.read_csv(sirh_file_path)
        sondage_df = pd.read_csv(sondage_file_path)

        return eval_df, sirh_df, sondage_df
    except FileNotFoundError as e:
        print(
            f"Required data file not found: {e}. Please ensure 'data' directory "
            "contains 'extrait_eval.csv', 'extrait_sirh.csv', and 'extrait_sondage.csv'."
        )
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred loading local CSV files: {e}")
        sys.exit(1)


def debug_api_call():
    print("--- Starting API Debug Call ---")

    eval_df, sirh_df, sondage_df = _load_local_csv_files()
    sample_n = int(os.environ.get("SAMPLE_N", "10"))
    eval_df = eval_df.head(sample_n)
    sirh_df = sirh_df.head(sample_n)
    sondage_df = sondage_df.head(sample_n)

    # Convert raw DataFrames to list of dicts for API
    eval_data_for_api = eval_df.to_dict(orient="records")
    sirh_data_for_api = sirh_df.to_dict(orient="records")
    sondage_data_for_api = sondage_df.to_dict(orient="records")

    print(f"Sample eval_data_for_api (first 2): {eval_data_for_api[:2]}")
    print(f"Sample sirh_data_for_api (first 2): {sirh_data_for_api[:2]}")
    print(f"Sample sondage_data_for_api (first 2): {sondage_data_for_api[:2]}")

    try:
        print(f"Calling API endpoint: {API_BASE_URL}/predict")
        payload = {
            "eval_data": eval_data_for_api,
            "sirh_data": sirh_data_for_api,
            "sondage_data": sondage_data_for_api,
        }
        response = httpx.post(f"{API_BASE_URL}/predict", json=payload, timeout=120.0)
        response.raise_for_status()  # Raise an exception for 4xx or 5xx status codes

        print("\n--- API Response ---")
        print(f"Status Code: {response.status_code}")
        print(f"Response Body: {response.json()}")
    except httpx.RequestError as e:
        print(f"Network error while connecting to API: {e}")
    except httpx.HTTPStatusError as e:
        print(f"API returned an error: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    print("--- API Debug Call Finished ---")


if __name__ == "__main__":
    debug_api_call()
