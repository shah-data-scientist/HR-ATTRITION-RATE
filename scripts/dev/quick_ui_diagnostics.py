"""Lightweight diagnostics for the Streamlit UI logic.

Run without starting Streamlit to quickly validate:
 - Artifact presence (model, X_test.parquet, y_test.parquet)
 - Confusion matrix counts at a chosen threshold
 - Risk categorization function behavior

Usage (PowerShell):
  poetry run python scripts/dev/quick_ui_diagnostics.py --threshold 0.2876

Will exit with non‑zero code if critical artifacts are missing.
"""

from __future__ import annotations

import argparse
import os
import sys
import joblib
import pandas as pd
import numpy as np


def project_root() -> str:
    # Go up two levels: scripts/dev/ -> scripts/ -> root
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def load_artifacts(root: str):
    model_path = os.path.join(root, "outputs", "employee_attrition_pipeline.pkl")
    x_test_path = os.path.join(root, "outputs", "X_test.parquet")
    y_test_path = os.path.join(root, "outputs", "y_test.parquet")
    missing = [
        p for p in [model_path, x_test_path, y_test_path] if not os.path.exists(p)
    ]
    if missing:
        raise FileNotFoundError(f"Missing artifact(s): {', '.join(missing)}")
    model = joblib.load(model_path)
    X_test = pd.read_parquet(x_test_path)
    y_df = pd.read_parquet(y_test_path)
    if isinstance(y_df, pd.DataFrame) and y_df.shape[1] == 1:
        y_series = y_df.iloc[:, 0]
    else:
        y_series = y_df.squeeze()
    return model, X_test, y_series


def normalize_label(x):
    s = str(x).strip().lower()
    pos = {"1", "yes", "y", "true", "leave", "leaver", "oui"}
    neg = {"0", "no", "n", "false", "stay", "stayer", "non"}
    if s in pos:
        return "Leave"
    if s in neg:
        return "Stay"
    try:
        v = float(s)
        return "Leave" if v >= 0.5 else "Stay"
    except Exception:
        return None


def compute_confusion(model, X_test, y_raw, tau: float):
    y_labels = y_raw.map(normalize_label).dropna()
    # Align indices
    try:
        X_aligned = X_test.loc[y_labels.index]
    except Exception:
        min_len = min(len(X_test), len(y_labels))
        X_aligned = X_test.iloc[:min_len]
        y_labels = y_labels.iloc[:min_len]
    probs = model.predict_proba(X_aligned)[:, 1]
    preds = np.where(probs >= tau, "Leave", "Stay")
    TP = int(
        (
            (y_labels == "Leave") & (pd.Series(preds, index=y_labels.index) == "Leave")
        ).sum()
    )
    FP = int(
        (
            (y_labels == "Stay") & (pd.Series(preds, index=y_labels.index) == "Leave")
        ).sum()
    )
    TN = int(
        (
            (y_labels == "Stay") & (pd.Series(preds, index=y_labels.index) == "Stay")
        ).sum()
    )
    FN = int(
        (
            (y_labels == "Leave") & (pd.Series(preds, index=y_labels.index) == "Stay")
        ).sum()
    )
    return TP, FP, TN, FN, probs


def risk_category_dyn(prob: float, threshold: float = 0.5) -> str:
    buffer = 0.05
    min_medium_prob = 0.20
    if prob >= threshold + buffer:
        return "High"
    if prob < threshold - buffer:
        return "Low"
    if prob >= min_medium_prob:
        return "Medium"
    return "Low"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--threshold", "-t", type=float, default=0.2876, help="Decision threshold τ"
    )
    args = parser.parse_args()

    root = project_root()
    try:
        model, X_test, y_series = load_artifacts(root)
    except FileNotFoundError as e:
        print(f"[FAIL] {e}")
        return 2
    except Exception as e:
        print(f"[FAIL] Unexpected artifact load error: {e}")
        return 3

    TP, FP, TN, FN, probs = compute_confusion(model, X_test, y_series, args.threshold)
    total_leave = TP + FN
    total_stay = TN + FP
    print(f"Artifacts OK. Rows used: {len(probs)}")
    print(f"Threshold τ = {args.threshold:.4f}")
    print(f"TP={TP} FP={FP} TN={TN} FN={FN}")
    if total_leave:
        print(f"Recall (Leave) = {TP/total_leave:.3f}")
    if TP + FP:
        print(f"Precision (Leave prediction) = {TP/(TP+FP):.3f}")
    # Show sample risk categories
    sample_probs = sorted(probs[:10])  # deterministic subset
    samples = [f"{p:.3f}->{risk_category_dyn(p, args.threshold)}" for p in sample_probs]
    print("Sample risk mappings:")
    for s in samples:
        print("  ", s)
    return 0


if __name__ == "__main__":
    code = main()
    sys.exit(code)
