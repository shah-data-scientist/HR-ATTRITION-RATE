import os
import time
import base64
import io
import pandas as pd
import numpy as np
import shap
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from sqlalchemy import text

from database.database import SessionLocal
from database.models import Job
from api.app.main import init_model_for_cli, generate_predictions
from core.schema import RawBatchPredictionInput

POLL_INTERVAL = float(os.environ.get("WORKER_POLL_SEC", "2"))
STALE_SECONDS = int(os.environ.get("WORKER_STALE_SEC", "600"))


def mark_stale_jobs(db):
    cutoff = datetime.utcnow() - timedelta(seconds=STALE_SECONDS)
    # For SQLite, updated_at may be None; be defensive
    jobs = db.query(Job).filter(Job.status == "processing").all()
    for j in jobs:
        if not j.updated_at or (j.updated_at and j.updated_at < cutoff):
            j.status = "queued"
    db.commit()


def build_report_artifacts(predictions_output):
    # Build Summary sheet
    import pandas as pd

    summary_rows = []
    for p in predictions_output:
        summary_rows.append(
            {
                "Employee_ID": p.id_employee,
                "Risk_Attrition": p.risk_category,
                "Attrition_Risk_Percentage": round(float(p.probability) * 100, 1),
                "Prediction": p.prediction,
            }
        )
    summary_df = pd.DataFrame(summary_rows)

    # Build Features sheet (stacked SHAP values)
    features_frames = []
    for p in predictions_output:
        if p.shap_values is not None and p.feature_names is not None:
            feature_names = (
                p.feature_names
                if len(p.feature_names) == len(p.shap_values)
                else [f"Feature {i}" for i in range(len(p.shap_values))]
            )
            df = pd.DataFrame(
                {
                    "Feature": feature_names,
                    "Coefficient": p.shap_values,
                }
            )
            df["Employee_ID"] = p.id_employee
            df["Prediction"] = p.prediction
            features_frames.append(df)
    features_df = (
        pd.concat(features_frames, ignore_index=True)
        if features_frames
        else pd.DataFrame()
    )

    # Build Metrics sheet
    total = len(predictions_output)
    predicted_leave = sum(1 for p in predictions_output if p.prediction == "Leave")
    predicted_stay = sum(1 for p in predictions_output if p.prediction == "Stay")
    metrics_df = pd.DataFrame(
        {
            "Metric": [
                "Total Employees Processed",
                "Predicted to Leave",
                "Predicted to Stay",
            ],
            "Value": [total, predicted_leave, predicted_stay],
        }
    )

    # Write Excel to bytes
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        if not features_df.empty:
            features_df[
                ["Employee_ID", "Feature", "Coefficient", "Prediction"]
            ].to_excel(writer, sheet_name="Features", index=False)
        metrics_df.to_excel(writer, sheet_name="Metrics", index=False)
    excel_buffer.seek(0)
    excel_b64 = base64.b64encode(excel_buffer.read()).decode("utf-8")

    # Generate SHAP waterfall images per employee
    shap_images = []
    for p in predictions_output:
        if p.shap_values is None or p.base_value is None:
            continue
        feature_names = (
            p.feature_names
            if p.feature_names and len(p.feature_names) == len(p.shap_values)
            else [f"Feature {i}" for i in range(len(p.shap_values))]
        )
        explanation = shap.Explanation(
            values=np.array(p.shap_values),
            base_values=p.base_value,
            data=np.zeros(len(p.shap_values)),
            feature_names=feature_names,
        )
        shap.plots.waterfall(explanation, max_display=10, show=False)
        fig = plt.gcf()
        fig.set_size_inches(8, 6)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        plt.close(fig)
        img_str = base64.b64encode(buf.getvalue()).decode("utf-8")

        shap_images.append(
            {
                "employee_id": p.id_employee,
                "risk_category": p.risk_category,
                "attrition_prob": float(p.probability),
                "prediction_type": p.prediction,
                "img_base64": img_str,
            }
        )

    return {
        "excel_base64": excel_b64,
        "shap_images": shap_images,
    }


def process_job(db, job):
    payload = job.payload_json
    batch_input = RawBatchPredictionInput(**payload)

    # Init model once
    init_model_for_cli()

    # Create a mock request object for the worker
    from types import SimpleNamespace

    # Use the user_id from the job (who originally created the job request)
    user_id = getattr(job, 'user_id', 'demo1')
    mock_headers_dict = {"X-User-ID": user_id}
    
    # SimpleNamespace to mimic request.headers with dict's .get() method
    class MockHeaders:
        def __init__(self, headers_dict):
            self._headers = headers_dict
        def get(self, key, default=None):
            return self._headers.get(key, default)
    
    mock_request = SimpleNamespace(
        headers=MockHeaders(mock_headers_dict),
        client=SimpleNamespace(host="worker")
    )

    # Use DB session for traceability
    predictions_output = generate_predictions(
        batch_input=batch_input, request=mock_request, db=db, compute_shap=True
    )

    artifacts = build_report_artifacts(predictions_output)
    result = {
        "predictions": [p.model_dump() for p in predictions_output],
        **artifacts,
    }
    return result


def main_loop():
    while True:
        with SessionLocal() as db:
            mark_stale_jobs(db)
            job = (
                db.query(Job)
                .filter(Job.status == "queued")
                .order_by(Job.created_at.asc())
                .first()
            )
            if not job:
                time.sleep(POLL_INTERVAL)
                continue
            job.status = "processing"
            db.commit()
            try:
                result = process_job(db, job)
                job.result_json = result
                job.status = "completed"
                job.error = None
                db.commit()
            except Exception as e:
                job.status = "failed"
                job.error = str(e)
                db.commit()
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main_loop()
