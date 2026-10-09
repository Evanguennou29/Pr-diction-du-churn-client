import joblib
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.churn import EXPECTED, preprocess, score, split_features
from src.monitor import psi


def sample_data():
    rows = []
    for i in range(8):
        row = {name: "No" for name in EXPECTED}
        row.update({
            "customerID": f"id-{i}", "Churn": "Yes" if i % 2 else "No",
            "tenure": i + 1, "MonthlyCharges": 20.0 + i,
            "TotalCharges": " " if i == 0 else str(20.0 * (i + 1)),
            "Contract": "Month-to-month" if i % 2 else "Two year",
        })
        rows.append(row)
    return pd.DataFrame(rows)


def test_cleaning_and_unseen_category():
    data = sample_data()
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    x, y = split_features(data)
    pipeline = Pipeline([
        ("preprocess", preprocess()),
        ("model", LogisticRegression(max_iter=1000)),
    ]).fit(x, y)
    incoming = x.iloc[[0]].copy()
    incoming["Contract"] = "Unknown future contract"
    assert 0 <= pipeline.predict_proba(incoming)[0, 1] <= 1


def test_scoring_ignores_target_and_keeps_ids(tmp_path):
    data = sample_data()
    x, y = split_features(data)
    pipeline = Pipeline([
        ("preprocess", preprocess()),
        ("model", LogisticRegression(max_iter=1000)),
    ]).fit(x, y)
    model = tmp_path / "model.joblib"
    source = tmp_path / "input.csv"
    output = tmp_path / "scores.csv"
    joblib.dump({"pipeline": pipeline, "threshold": 0.5, "features": list(x.columns)}, model)
    data.to_csv(source, index=False)
    score(source, model, output)
    result = pd.read_csv(output)
    assert set(result["customerID"]) == set(data["customerID"])
    assert result["probabilite_churn"].between(0, 1).all()
    assert "Churn" not in result.columns


def test_duplicate_customer_id_is_rejected(tmp_path):
    data = sample_data()
    data.loc[1, "customerID"] = data.loc[0, "customerID"]
    source = tmp_path / "duplicates.csv"
    data.to_csv(source, index=False)
    with pytest.raises(ValueError, match="unique"):
        from src.churn import load_data
        load_data(source, download=False)


def test_psi_identical_samples_is_zero():
    series = pd.Series(range(100))
    assert psi(series, series) == pytest.approx(0.0, abs=1e-10)
