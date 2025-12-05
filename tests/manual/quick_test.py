"""Quick single employee test"""

import pandas as pd
import httpx

sirh_df = pd.read_csv("data/extrait_sirh.csv")
eval_df = pd.read_csv("data/extrait_eval.csv")
sondage_df = pd.read_csv("data/extrait_sondage.csv")

emp_id = 1
sirh_row = sirh_df[sirh_df["id_employee"] == emp_id].iloc[0].to_dict()
eval_row = eval_df[eval_df["eval_number"] == f"E_{emp_id}"].iloc[0].to_dict()
sondage_row = sondage_df[sondage_df["code_sondage"] == emp_id].iloc[0].to_dict()

payload = {
    "sirh_data": [sirh_row],
    "eval_data": [eval_row],
    "sondage_data": [sondage_row],
}

response = httpx.post(
    "http://localhost:8001/predict_report",
    json=payload,
    headers={"X-User-ID": "testuser"},
    timeout=30.0,
)

print(f"Status: {response.status_code}")
print(f"Trace ID: {response.json()['predictions'][0]['trace_id']}")
