import httpx
import pandas as pd
import json

# Import the data processing functions from app.py
from app import load_and_merge_data

def run_e2e_test():
    """
    Performs an end-to-end test of the FastAPI application.
    """
    # 1. Read the three CSV files from the data directory
    try:
        eval_df = pd.read_csv("data/extrait_eval.csv")
        sirh_df = pd.read_csv("data/extrait_sirh.csv")
        sondage_df = pd.read_csv("data/extrait_sondage.csv")
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure the data files are in the 'data' directory.")
        return

    # 2. Merge the dataframes
    merged_df = load_and_merge_data(eval_df, sirh_df, sondage_df)

    # 3. Convert the merged DataFrame to a list of dictionaries
    employees_data = merged_df.to_dict(orient="records")

    # 4. Create the payload
    payload = {
        "employees": employees_data,
    }

    # 5. Send the request to the API
    api_url = "http://localhost:8000/predict?threshold=0.5"
    print(f"Sending request to {api_url}...")

    try:
        response = httpx.post(api_url, json=payload, timeout=30)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        # 6. Print the response
        print("Request successful!")
        print("Response status code:", response.status_code)
        print("Response body:")
        print(json.dumps(response.json(), indent=2))

    except httpx.RequestError as e:
        print(f"An error occurred while requesting {e.request.url!r}.")
        print(f"Error: {e}")
    except httpx.HTTPStatusError as e:
        print(f"Error response {e.response.status_code} while requesting {e.request.url!r}.")
        print("Response body:")
        try:
            print(json.dumps(e.response.json(), indent=2))
        except json.JSONDecodeError:
            print(e.response.text)

if __name__ == "__main__":
    run_e2e_test()
