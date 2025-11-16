import os
import json
import base64
from pathlib import Path

import httpx
import pandas as pd


def main():
    base_url = os.environ.get("API_BASE_URL", "http://localhost:8001")
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"

    eval_df = pd.read_csv(data_dir / "extrait_eval.csv")
    sirh_df = pd.read_csv(data_dir / "extrait_sirh.csv")
    sondage_df = pd.read_csv(data_dir / "extrait_sondage.csv")

    # Reduce payload size for quicker end-to-end validation
    sample_n = int(os.environ.get("SAMPLE_N", "10"))
    eval_df = eval_df.head(sample_n)
    sirh_df = sirh_df.head(sample_n)
    sondage_df = sondage_df.head(sample_n)

    payload = {
        "eval_data": eval_df.to_dict(orient="records"),
        "sirh_data": sirh_df.to_dict(orient="records"),
        "sondage_data": sondage_df.to_dict(orient="records"),
    }

    print(f"POST {base_url}/predict_report ...")
    # Allow longer processing time for SHAP + Excel generation
    resp = httpx.post(f"{base_url}/predict_report", json=payload, timeout=180.0)
    print(f"Status: {resp.status_code}")
    resp.raise_for_status()

    data = resp.json()
    # Save API response (truncated) for debugging
    (root / "temp_api_response.json").write_text(json.dumps(data)[:2000], encoding="utf-8")

    # Save Excel report
    excel_b64 = data.get("excel_base64")
    if excel_b64:
        (root / "report.xlsx").write_bytes(base64.b64decode(excel_b64))
        print(f"Saved Excel report to: {root / 'report.xlsx'}")

    # Save SHAP images
    images_dir = root / "shap_images"
    images_dir.mkdir(exist_ok=True)
    for item in data.get("shap_images", []):
        emp_id = item.get("employee_id")
        img_b64 = item.get("img_base64")
        if img_b64 and emp_id is not None:
            (images_dir / f"employee_{emp_id}.png").write_bytes(base64.b64decode(img_b64))
    print(f"Saved {len(data.get('shap_images', []))} SHAP images to: {images_dir}")


if __name__ == "__main__":
    main()
