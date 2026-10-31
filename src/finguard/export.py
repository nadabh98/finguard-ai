"""Exporte les transactions scorées du jeu de test (console Streamlit, Power BI)."""
import joblib
import numpy as np
import pandas as pd

from .api import ACTIONS
from .config import EXPORTS, MODELS
from .explain import LABELS, level
from .features import CAT_FEATURES, build_features
from .split import temporal_split
from .train import load_raw


def main():
    art = joblib.load(MODELS / "lightgbm.joblib")
    model, feats = art["model"], art["features"]
    iso = joblib.load(MODELS / "calibrator.joblib")

    df = build_features(load_raw())
    train, _, test = temporal_split(df)
    test = test.reset_index(drop=True)
    for c in CAT_FEATURES:
        test[c] = pd.Categorical(test[c], categories=sorted(train[c].unique()))
    X = test[feats]

    proba = iso.predict(model.predict_proba(X)[:, 1])
    contrib = model.predict(X, pred_contrib=True)[:, :-1]
    niveau = np.array([level(p) for p in proba])
    top = np.argsort(-contrib, axis=1)[:, :3]

    raisons = [""] * len(test)
    for i in np.flatnonzero(niveau != "faible"):
        parts = []
        for j in top[i]:
            if contrib[i, j] > 0:
                v = X.iat[i, j]
                v = f"{v:.1f}" if isinstance(v, (float, np.floating)) else str(v)
                parts.append(f"{LABELS[feats[j]]} = {v} (+{contrib[i, j]:.1f})")
        raisons[i] = " ; ".join(parts)

    out = pd.DataFrame({
        "trans_num": test["trans_num"], "ts": test["ts"], "cc_num": test["cc_num"].astype(str),
        "merchant": test["merchant"].str.replace("fraud_", "", regex=False),
        "category": test["category"].astype(str), "amt": test["amt"], "hour": test["hour"],
        "distance_km": test["distance_km"].round(1), "tx_count_24h": test["tx_count_24h"],
        "amt_sum_24h": test["amt_sum_24h"].round(2), "is_fraud": test["is_fraud"],
        "proba": proba.round(4), "niveau": niveau,
        "action": [ACTIONS[n] for n in niveau], "raisons": raisons,
    })
    EXPORTS.mkdir(exist_ok=True)
    out.to_csv(EXPORTS / "scored_transactions.csv", index=False)
    print(f"{len(out)} transactions exportées dans exports/scored_transactions.csv")
    print(out["niveau"].value_counts().to_string())


if __name__ == "__main__":
    main()
