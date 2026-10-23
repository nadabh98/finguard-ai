"""Phase 2 : calibration, niveaux de risque, explications par alerte."""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss

from .config import MODELS
from .features import build_features, CAT_FEATURES
from .split import temporal_split
from .train import load_raw

# Niveaux sur la probabilité CALIBRÉE (seuils à ajuster selon la politique de risque)
LEVELS = [(0.05, "faible"), (0.30, "moyen"), (0.80, "élevé"), (1.01, "blocage")]

LABELS = {
    "amt": "montant", "hour": "heure", "dayofweek": "jour de la semaine", "age": "âge du client",
    "city_pop": "population de la ville", "distance_km": "distance client-marchand (km)",
    "tx_count_1h": "transactions sur la dernière heure", "tx_count_24h": "transactions sur 24 h",
    "amt_sum_24h": "somme dépensée sur 24 h", "amt_ratio_to_mean": "montant / moyenne habituelle du client",
    "secs_since_last": "secondes depuis la dernière transaction", "new_merchant": "nouveau marchand (1 = oui)",
    "category": "catégorie", "gender": "genre",
}


def level(p):
    for thr, name in LEVELS:
        if p < thr:
            return name


def main():
    art = joblib.load(MODELS / "lightgbm.joblib")
    model, feats = art["model"], art["features"]

    df = build_features(load_raw())
    train, val, test = temporal_split(df)
    for d in (train, val, test):
        for c in CAT_FEATURES:
            d[c] = pd.Categorical(d[c], categories=sorted(train[c].unique()))
    test = test.reset_index(drop=True)

    # 1) Calibration sur la VALIDATION, évaluation sur le TEST
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    iso.fit(model.predict_proba(val[feats])[:, 1], val["is_fraud"])
    raw_p = model.predict_proba(test[feats])[:, 1]
    p = iso.predict(raw_p)
    y = test["is_fraud"]
    print("\n=== CALIBRATION (test) ===")
    print(f"Taux de fraude réel          : {y.mean():.4%}")
    print(f"Proba moyenne AVANT calibr.  : {raw_p.mean():.4%}   (Brier {brier_score_loss(y, raw_p):.5f})")
    print(f"Proba moyenne APRÈS calibr.  : {p.mean():.4%}   (Brier {brier_score_loss(y, p):.5f})")
    print(f"PR-AUC après calibration     : {average_precision_score(y, p):.4f}")

    # 2) Niveaux de risque
    test["proba"] = p
    test["niveau"] = [level(x) for x in p]
    order = [n for _, n in LEVELS]
    tab = test.groupby("niveau").agg(transactions=("is_fraud", "size"), fraudes=("is_fraud", "sum"),
                                     montant_fraude=("amt", lambda s: s[test.loc[s.index, "is_fraud"] == 1].sum()))
    tab["taux_fraude"] = (tab["fraudes"] / tab["transactions"]).map("{:.2%}".format)
    print("\n=== NIVEAUX DE RISQUE (test) ===")
    print(tab.reindex(order).to_string())

    # 3) Explications (contributions de type SHAP, calculées par LightGBM)
    contrib = model.predict(test[feats], pred_contrib=True)[:, :-1]
    imp = pd.Series(np.abs(contrib).mean(axis=0), index=feats).sort_values(ascending=False)
    print("\n=== VARIABLES LES PLUS IMPORTANTES (impact moyen) ===")
    print(imp.head(10).round(3).to_string())

    print("\n=== LES 3 ALERTES LES PLUS RISQUÉES : POURQUOI ? ===")
    for i in np.argsort(p)[-3:][::-1]:
        row = test.iloc[i]
        print(f"\nFraude réelle = {int(row['is_fraud'])}  montant {row['amt']:.2f}  proba {p[i]:.1%}  niveau {row['niveau']}")
        for j in np.argsort(contrib[i])[::-1][:3]:
            v = row[feats[j]]
            v = f"{v:.1f}" if isinstance(v, (int, float, np.number)) else v
            print(f"   + {LABELS[feats[j]]} = {v}   (impact {contrib[i][j]:+.2f})")

    joblib.dump(iso, MODELS / "calibrator.joblib")
    tab.reset_index().to_csv(MODELS / "risk_levels.csv", index=False)
    imp.to_csv(MODELS / "feature_importance.csv")
    print("\nSauvegardé : calibrator.joblib, risk_levels.csv, feature_importance.csv")


if __name__ == "__main__":
    main()
