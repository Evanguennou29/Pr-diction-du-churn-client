# Prédiction du churn client

Projet Python de classification sur le jeu de données **IBM Telco Customer Churn** (7 043 clients). Le rendu principal est le [notebook d'analyse et de modélisation](notebooks/01_prediction_churn.ipynb). Il exécute l'exploration, compare les traitements et les modèles, choisit les hyperparamètres, affiche les métriques et explique les scores.

## Démarrage

Python 3.11 ou 3.12 recommandé. Depuis la racine du dépôt :

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
jupyter lab notebooks/01_prediction_churn.ipynb
```

Exécuter toutes les cellules dans l'ordre. La première exécution télécharge le CSV depuis le [dépôt officiel IBM](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv) vers `data/Telco-Customer-Churn.csv`. Si le téléchargement est interdit, placer manuellement ce fichier à cet emplacement. La [fiche Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) décrit le même jeu de données.

Le notebook a été exécuté sur le CSV IBM et contient les sorties, tableaux et graphiques. Il enregistre aussi les artefacts dans `artifacts/`. Une nouvelle exécution recalcule tous les résultats.

## Résultats mesurés

Sur le découpage stratifié fixé par la graine 42, **XGBoost** avec imputation médiane et standardisation obtient la meilleure PR-AUC moyenne en validation croisée : **0,671**. Ses paramètres retenus sont `learning_rate=0.03`, `max_depth=4` et `n_estimators=200`.

Sur les **1 409 clients du test réservé** : PR-AUC **0,659**, ROC-AUC **0,846**, précision **56,3 %**, rappel **70,6 %** et F1 **0,626** au seuil **0,35** choisi sur le train. La matrice de confusion compte 830 vrais négatifs, 205 faux positifs, 110 faux négatifs et 264 vrais positifs. Le fichier `artifacts/metrics.json` contient les valeurs non arrondies.

## Pipeline d'entraînement et de scoring

```powershell
python -m src.churn train --data data/Telco-Customer-Churn.csv --output artifacts
python -m src.churn score --input data/Telco-Customer-Churn.csv --model artifacts/model.joblib --output artifacts/scored_customers.csv
python -m src.monitor --reference data/Telco-Customer-Churn.csv --current data/Telco-Customer-Churn.csv --model artifacts/model.joblib --output artifacts/drift.json
python -m pytest -q
```

`train` sépare les données en train/test stratifiés (80/20, graine 42), compare Logistic Regression, Random Forest, XGBoost et LightGBM avec trois traitements des données et une recherche sur grille à cinq plis. La sélection utilise la PR-AUC. Un seuil de F1 est choisi sur des prédictions hors pli du train, avant l'évaluation sur le test réservé. Le pipeline enregistré contient imputation, encodage, modèle, liste des variables et seuil. `score` restitue `customerID`, la probabilité de churn et une proposition de contact ; la cible n'est pas nécessaire dans le CSV entrant.

Les fichiers `metrics.json`, `model_comparison.csv`, `thresholds.csv` et `test_predictions.csv` documentent l'expérience. `drift.json` donne des indicateurs simples de changement de distribution entre une référence et un lot récent ; les seuils d'alerte doivent être définis sur des données de production.

## Organisation

| Chemin | Rôle |
| --- | --- |
| `notebooks/01_prediction_churn.ipynb` | Rendu principal, visualisations, résultats et SHAP |
| `src/churn.py` | Chargement, validation, preprocessing, comparaison, entraînement et scoring |
| `src/monitor.py` | Contrôle de distribution et de score d'un nouveau lot |
| `tests/` | Contrôles de données et de scoring |
| `.github/workflows/ci.yml` | Tests automatiques et entraînement déclenchable manuellement |

## Interprétation et limites

Le jeu IBM est un jeu de démonstration. Le découpage aléatoire ne simule pas une évaluation dans le futur ; une validation temporelle sur des données réelles est nécessaire. SHAP explique les prédictions, sans démontrer une cause de résiliation. La priorisation du notebook et sa simulation économique reposent sur des hypothèses ; l'efficacité d'une offre de rétention doit être mesurée avec un test A/B.
