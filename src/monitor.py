"""Indicateurs simples de dérive des entrées et des scores."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .churn import ID_COLUMN, NUMERIC


def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Population Stability Index avec bornes quantiles de la référence."""
    ref = pd.to_numeric(reference, errors="coerce").dropna().to_numpy()
    cur = pd.to_numeric(current, errors="coerce").dropna().to_numpy()
    if not len(ref) or not len(cur):
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0 if np.all(cur == ref[0]) else float("inf")
    edges[0], edges[-1] = -np.inf, np.inf
    ref_rate = np.histogram(ref, bins=edges)[0] / len(ref)
    cur_rate = np.histogram(cur, bins=edges)[0] / len(cur)
    ref_rate = np.clip(ref_rate, 1e-6, 1)
    cur_rate = np.clip(cur_rate, 1e-6, 1)
    return float(np.sum((cur_rate - ref_rate) * np.log(cur_rate / ref_rate)))


def monitor(reference_path: str, current_path: str, model_path: str, output_path: str) -> dict:
    bundle = joblib.load(model_path)
    reference, current = pd.read_csv(reference_path), pd.read_csv(current_path)
    features = bundle["features"]
    for label, frame in [("reference", reference), ("current", current)]:
        missing = set(features) - set(frame.columns)
        if missing:
            raise ValueError(f"Colonnes absentes dans {label} : {sorted(missing)}")
        if ID_COLUMN not in frame:
            raise ValueError(f"customerID absent dans {label}")
    result = {"reference_rows": len(reference), "current_rows": len(current), "numeric_psi": {},
              "new_category_rate": {}, "missing_rate": {}}
    for col in features:
        result["missing_rate"][col] = {
            "reference": float(reference[col].isna().mean()),
            "current": float(current[col].isna().mean()),
        }
        if col in NUMERIC:
            result["numeric_psi"][col] = psi(reference[col], current[col])
        else:
            known = set(reference[col].dropna().astype(str))
            result["new_category_rate"][col] = float(
                (~current[col].dropna().astype(str).isin(known)).mean()
            ) if current[col].notna().any() else None
    current_x = current[features].copy()
    current_x["TotalCharges"] = pd.to_numeric(current_x["TotalCharges"], errors="coerce")
    probabilities = bundle["pipeline"].predict_proba(current_x)[:, 1]
    result["prediction_summary"] = {
        "mean_probability": float(np.mean(probabilities)),
        "contact_rate": float(np.mean(probabilities >= bundle["threshold"])),
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Surveillance d'un lot de scoring")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--model", default="artifacts/model.joblib")
    parser.add_argument("--output", default="artifacts/drift.json")
    args = parser.parse_args()
    print(json.dumps(monitor(args.reference, args.current, args.model, args.output),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
