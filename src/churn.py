"""Pipeline reproductible pour IBM Telco Customer Churn."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import urlretrieve

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
RANDOM_STATE = 42
ID_COLUMN = "customerID"
TARGET = "Churn"
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
EXPECTED = {
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
}


def load_data(path: str | Path = "data/Telco-Customer-Churn.csv", download: bool = True) -> pd.DataFrame:
    """Charge le CSV et télécharge la copie officielle d'IBM si nécessaire."""
    path = Path(path)
    if not path.exists():
        if not download:
            raise FileNotFoundError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(DATA_URL, path)
    data = pd.read_csv(path)
    missing = (EXPECTED | {ID_COLUMN, TARGET}) - set(data.columns)
    if missing:
        raise ValueError(f"Colonnes absentes : {sorted(missing)}")
    if data[ID_COLUMN].isna().any() or data[ID_COLUMN].duplicated().any():
        raise ValueError("customerID doit être renseigné et unique.")
    if not data[TARGET].isin(["Yes", "No"]).all():
        raise ValueError("Churn doit contenir uniquement Yes/No.")
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    return data


def split_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Supprime identifiant et cible des variables explicatives."""
    features = data.drop(columns=[ID_COLUMN, TARGET]).copy()
    missing = EXPECTED - set(features.columns)
    if missing:
        raise ValueError(f"Variables manquantes : {sorted(missing)}")
    features["TotalCharges"] = pd.to_numeric(features["TotalCharges"], errors="coerce")
    target = data[TARGET].map({"No": 0, "Yes": 1})
    if target.isna().any():
        raise ValueError("La cible Churn doit contenir Yes/No.")
    return features, target.astype(int)


def preprocess(scale: bool = True, imputation: str = "median") -> ColumnTransformer:
    """Apprend imputations et encodage uniquement sur le pli d'entraînement."""
    if imputation not in {"median", "constant"}:
        raise ValueError("imputation doit être median ou constant")
    numerical_imputer = (
        SimpleImputer(strategy="median") if imputation == "median"
        else SimpleImputer(strategy="constant", fill_value=0)
    )
    numerical_steps = [("imputer", numerical_imputer)]
    if scale:
        numerical_steps.append(("scale", StandardScaler()))
    categorical = sorted(EXPECTED - set(NUMERIC))
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numerical_steps), NUMERIC),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical),
        ],
        remainder="drop",
    )


def candidate_models() -> dict[str, tuple[object, dict[str, list]]]:
    """Grilles compactes, fixées avant l'évaluation du test."""
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            {"model__C": [0.1, 1.0, 10.0], "model__class_weight": [None, "balanced"]},
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
            {"model__n_estimators": [200, 400], "model__max_depth": [5, 10, None],
             "model__min_samples_leaf": [2, 5]},
        ),
        "XGBoost": (
            XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                          tree_method="hist", random_state=RANDOM_STATE, n_jobs=1),
            {"model__n_estimators": [200, 400], "model__max_depth": [2, 4],
             "model__learning_rate": [0.03, 0.1]},
        ),
        "LightGBM": (
            LGBMClassifier(random_state=RANDOM_STATE, verbosity=-1, n_jobs=1),
            {"model__n_estimators": [200, 400], "model__num_leaves": [7, 15],
             "model__learning_rate": [0.03, 0.1]},
        ),
    }


def select_model(
    x_train: pd.DataFrame, y_train: pd.Series, folds: int = 5,
    n_jobs: int = -1,
) -> tuple[Pipeline, pd.DataFrame, dict]:
    """Compare preprocessing et modèles en CV stratifiée, sur train uniquement."""
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    best_estimator = None
    best_score = -np.inf
    best_details = {}
    for preprocessing_name, scale, imputation in [
        ("median+standard", True, "median"),
        ("median+sans_echelle", False, "median"),
        ("zero+standard", True, "constant"),
    ]:
        for model_name, (model, grid) in candidate_models().items():
            pipeline = Pipeline([("preprocess", preprocess(scale, imputation)), ("model", model)])
            search = GridSearchCV(
                pipeline, grid, scoring="average_precision", cv=cv,
                n_jobs=n_jobs, refit=True, error_score="raise",
            )
            search.fit(x_train, y_train)
            rows.append({
                "preprocessing": preprocessing_name, "modele": model_name,
                "cv_pr_auc": search.best_score_, "params": str(search.best_params_),
            })
            if search.best_score_ > best_score:
                best_score = search.best_score_
                best_estimator = search.best_estimator_
                best_details = {
                    "preprocessing": preprocessing_name, "modele": model_name,
                    "cv_pr_auc": float(best_score), "params": search.best_params_,
                }
    return best_estimator, pd.DataFrame(rows).sort_values("cv_pr_auc", ascending=False), best_details


def choose_threshold(
    estimator: Pipeline, x_train: pd.DataFrame, y_train: pd.Series,
    folds: int = 5,
) -> tuple[float, pd.DataFrame]:
    """Choisit le seuil par F1 sur prédictions hors pli du train."""
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    probabilities = cross_val_predict(
        estimator, x_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    rows = []
    for threshold in np.arange(0.15, 0.81, 0.025):
        predicted = probabilities >= threshold
        tp = int(((predicted == 1) & (y_train == 1)).sum())
        fp = int(((predicted == 1) & (y_train == 0)).sum())
        fn = int(((predicted == 0) & (y_train == 1)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"seuil": float(threshold), "precision": precision, "rappel": recall, "f1": f1})
    table = pd.DataFrame(rows)
    winner = table.sort_values(["f1", "seuil"], ascending=[False, True]).iloc[0]
    return float(winner["seuil"]), table


def evaluate(estimator: Pipeline, x_test: pd.DataFrame, y_test: pd.Series, threshold: float) -> dict:
    """Évalue une seule fois sur le jeu test réservé."""
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
    from sklearn.metrics import f1_score, precision_score, recall_score
    p = estimator.predict_proba(x_test)[:, 1]
    pred = (p >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_test, p)),
        "pr_auc": float(average_precision_score(y_test, p)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "threshold": threshold,
        "test_rows": len(y_test),
    }


def train(data_path: str | Path, output_dir: str | Path = "artifacts", folds: int = 5, n_jobs: int = -1) -> dict:
    data = load_data(data_path)
    x, y = split_features(data)
    x_train, x_test, y_train, y_test, id_train, id_test = train_test_split(
        x, y, data[ID_COLUMN], test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    estimator, comparison, details = select_model(x_train, y_train, folds=folds, n_jobs=n_jobs)
    threshold, thresholds = choose_threshold(estimator, x_train, y_train, folds=folds)
    metrics = evaluate(estimator, x_test, y_test, threshold)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": estimator, "threshold": threshold, "features": list(x.columns)}, output / "model.joblib")
    comparison.to_csv(output / "model_comparison.csv", index=False)
    thresholds.to_csv(output / "thresholds.csv", index=False)
    pd.DataFrame({
        ID_COLUMN: id_test.to_numpy(), "churn_observe": y_test.to_numpy(),
        "probabilite_churn": estimator.predict_proba(x_test)[:, 1],
    }).sort_values("probabilite_churn", ascending=False).to_csv(output / "test_predictions.csv", index=False)
    report = {"dataset_rows": len(data), "train_rows": len(x_train), "selection": details, "test": metrics}
    (output / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def score(input_path: str | Path, model_path: str | Path, output_path: str | Path) -> None:
    """Score un CSV de clients sans utiliser la cible, si elle est présente."""
    bundle = joblib.load(model_path)
    data = pd.read_csv(input_path)
    if ID_COLUMN not in data or data[ID_COLUMN].isna().any() or data[ID_COLUMN].duplicated().any():
        raise ValueError("customerID doit être renseigné et unique.")
    features = bundle["features"]
    missing = set(features) - set(data.columns)
    if missing:
        raise ValueError(f"Variables manquantes : {sorted(missing)}")
    x = data[features].copy()
    x["TotalCharges"] = pd.to_numeric(x["TotalCharges"], errors="coerce")
    probability = bundle["pipeline"].predict_proba(x)[:, 1]
    result = pd.DataFrame({
        ID_COLUMN: data[ID_COLUMN], "probabilite_churn": probability,
        "a_contacter": probability >= bundle["threshold"],
    })
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.sort_values("probabilite_churn", ascending=False).to_csv(destination, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prédiction du churn IBM Telco")
    sub = parser.add_subparsers(dest="command", required=True)
    training = sub.add_parser("train")
    training.add_argument("--data", default="data/Telco-Customer-Churn.csv")
    training.add_argument("--output", default="artifacts")
    training.add_argument("--folds", type=int, default=5)
    training.add_argument("--jobs", type=int, default=-1)
    scoring = sub.add_parser("score")
    scoring.add_argument("--input", required=True)
    scoring.add_argument("--model", default="artifacts/model.joblib")
    scoring.add_argument("--output", default="artifacts/scored_customers.csv")
    args = parser.parse_args()
    if args.command == "train":
        print(json.dumps(train(args.data, args.output, args.folds, args.jobs), ensure_ascii=False, indent=2))
    else:
        score(args.input, args.model, args.output)
        print(args.output)


if __name__ == "__main__":
    main()
