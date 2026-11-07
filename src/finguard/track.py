"""Enregistre les résultats de l'entraînement dans MLflow (sans réentraîner)."""
import json

import joblib
import mlflow

from .config import MODELS, ROOT

METRICS = ("pr_auc", "precision", "recall", "expected_cost", "cost_no_model")
GBM_PARAMS = ("n_estimators", "learning_rate", "num_leaves", "scale_pos_weight", "random_state")


def main():
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("finguard-fraud")
    results = json.loads((MODELS / "metrics.json").read_text())
    gbm = joblib.load(MODELS / "lightgbm.joblib")["model"]

    for r in results:
        with mlflow.start_run(run_name=r["model"]):
            mlflow.log_params({"model": r["model"], "split": "temporel 70/15/15",
                               "desequilibre": "class_weight", "cout_fausse_alerte": 5.0,
                               "seuil": r["threshold"]})
            mlflow.log_metrics({k: r[k] for k in METRICS})
            mlflow.set_tag("donnees", "synthétiques (Sparkov)")
            if r["model"] == "lightgbm":
                p = gbm.get_params()
                mlflow.log_params({k: p[k] for k in GBM_PARAMS})
                mlflow.log_param("best_iteration", gbm.best_iteration_)
                for f in ("lightgbm.joblib", "calibrator.joblib", "risk_levels.csv",
                          "feature_importance.csv", "metrics.json"):
                    mlflow.log_artifact(str(MODELS / f))
    print("Runs enregistrés. Lance : mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000")


if __name__ == "__main__":
    main()
