import os
import json
from pathlib import Path
import pandas as pd
import httpx

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8001")

# Go up 2 levels: scripts/dev -> scripts -> root
root = Path(__file__).resolve().parents[2]
data_dir = root / "data"

eval_df = pd.read_csv(data_dir / "extrait_eval.csv").head(5)
sirh_df = pd.read_csv(data_dir / "extrait_sirh.csv").head(5)
sondage_df = pd.read_csv(data_dir / "extrait_sondage.csv").head(5)

payload = {
    "eval_data": eval_df.to_dict(orient="records"),
    "sirh_data": sirh_df.to_dict(orient="records"),
    "sondage_data": sondage_df.to_dict(orient="records"),
}

print(f"POST {API_BASE_URL}/jobs/report ...")
resp = httpx.post(f"{API_BASE_URL}/jobs/report", json=payload, timeout=60.0)
print("Status:", resp.status_code)
print(resp.text)
