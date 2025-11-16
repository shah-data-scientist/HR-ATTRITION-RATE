import requests
import pandas as pd
import json

# Load CSV files
eval_df = pd.read_csv("data/extrait_eval.csv")
sirh_df = pd.read_csv("data/extrait_sirh.csv")
sondage_df = pd.read_csv("data/extrait_sondage.csv")

# Take only first 3 rows for testing
eval_data = eval_df.head(3).to_dict(orient="records")
sirh_data = sirh_df.head(3).to_dict(orient="records")
sondage_data = sondage_df.head(3).to_dict(orient="records")

print("Eval data sample:")
print(json.dumps(eval_data[0], indent=2))
print("\nSIRH data sample:")
print(json.dumps(sirh_data[0], indent=2))
print("\nSondage data sample:")
print(json.dumps(sondage_data[0], indent=2))

# Prepare API payload
payload = {
    "eval_data": eval_data,
    "sirh_data": sirh_data,
    "sondage_data": sondage_data,
}

print("\n" + "="*50)
print("Sending request to API...")
print("="*50)

try:
    response = requests.post(
        "http://localhost:8001/predict",
        json=payload,
        timeout=60.0
    )

    print(f"\nStatus Code: {response.status_code}")

    if response.status_code == 200:
        print("\nSuccess! API Response:")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"\nError Response:")
        print(response.text)

except Exception as e:
    print(f"\nException occurred: {e}")
