"""Baseline : régression logistique vs LightGBM, split temporel, seuil au coût."""
import json
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, precision_score, recall_score

from .config import RAW, MODELS, FALSE_ALERT_COST
from .features import build_features, NUM_FEATURES, CAT_FEATURES
from .split import temporal_split


def load_raw() -> pd.DataFrame:
    files = sorted(RAW.glob("fraud*.csv"))
    if not files:
        raise FileNotFoundError("Place fraudTrain.csv et fraudTest.csv (Sparkov, Kaggle) dans data/raw/")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def expected_cost(y, p, amt, thr):
    """Fraude manquée = montant perdu ; fausse alerte = coût fixe."""
    pred = p >= thr
    missed = amt[(y == 1) & ~pred].sum()
    false_alerts = ((y == 0) & pred).sum() * FALSE_ALERT_COST
    return missed + false_alerts


def best_threshold(y, p, amt):
    grid = np.linspace(0.01, 0.99, 99)
    return float(min(grid, key=lambda t: expected_cost(y, p, amt, t)))


def report(name, y, p, amt, thr):
    pred = (p >= thr).astype(int)
    return {
        "model": name, "pr_auc": float(average_precision_score(y, p)), "threshold": thr,
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred)),
        "expected_cost": float(expected_cost(y.values, p, amt.values, thr)),
        "cost_no_model": float(amt[y == 1].sum()),
    }


def main():
    df = build_features(load_raw())
    train, val, test = temporal_split(df)
    print({k: len(v) for k, v in dict(train=train, val=val, test=test).items()})

    feats = NUM_FEATURES + CAT_FEATURES
    for d in (train, val, test):
        for c in CAT_FEATURES:
            d[c] = pd.Categorical(d[c], categories=sorted(train[c].unique()))
    y = {k: d["is_fraud"] for k, d in dict(train=train, val=val, test=test).items()}

    # Baseline : régression logistique (class_weight, pas de SMOTE)
    def lr_matrix(d):
        return pd.get_dummies(d[feats], columns=CAT_FEATURES, dtype=float)
    scaler = StandardScaler().fit(lr_matrix(train))
    lr = LogisticRegression(class_weight="balanced", max_iter=1000).fit(scaler.transform(lr_matrix(train)), y["train"])
    p_lr = lambda d: lr.predict_proba(scaler.transform(lr_matrix(d)))[:, 1]

    # LightGBM
    spw = (y["train"] == 0).sum() / max((y["train"] == 1).sum(), 1)
    gbm = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                             scale_pos_weight=spw, random_state=42, verbose=-1)
    gbm.fit(train[feats], y["train"], eval_set=[(val[feats], y["val"])],
            callbacks=[lgb.early_stopping(30, verbose=False)])
    p_gbm = lambda d: gbm.predict_proba(d[feats])[:, 1]

    results = []
    for name, fn in [("logreg", p_lr), ("lightgbm", p_gbm)]:
        thr = best_threshold(y["val"], fn(val), val["amt"].values)  # seuil choisi sur VALIDATION
        results.append(report(name, y["test"], fn(test), test["amt"], thr))  # évalué sur TEST
    print(json.dumps(results, indent=2))

    MODELS.mkdir(exist_ok=True)
    joblib.dump({"model": gbm, "features": feats, "threshold": results[1]["threshold"]}, MODELS / "lightgbm.joblib")
    (MODELS / "metrics.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
