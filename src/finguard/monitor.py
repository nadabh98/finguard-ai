"""Surveillance du drift : PSI hebdomadaire par rapport aux deux premières semaines."""
import sys

import numpy as np
import pandas as pd

from .config import EXPORTS

VARIABLES = ["amt", "hour", "distance_km", "tx_count_24h", "amt_sum_24h", "proba"]
RISK_EDGES = [-np.inf, 0.05, 0.30, 0.80, np.inf]  # limites des niveaux de risque


def psi(ref, cur, edges=None, bins=10):
    """Population Stability Index : 0 = identique, plus de 0,25 = dérive forte."""
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    if edges is None:
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:
            return 0.0
        edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def main(simulate=False):
    df = pd.read_csv(EXPORTS / "scored_transactions.csv", parse_dates=["ts"],
                     usecols=["ts", "is_fraud", "niveau"] + VARIABLES)
    df["semaine"] = df["ts"].dt.to_period("W").dt.start_time
    if simulate:  # démonstration : les montants des deux dernières semaines augmentent de 80 %
        late = df["semaine"] >= df["semaine"].max() - pd.Timedelta(weeks=1)
        df.loc[late, ["amt", "amt_sum_24h"]] *= 1.8

    cutoff = df["ts"].min() + pd.Timedelta(days=14)
    ref = df[df["ts"] < cutoff]
    rows = []
    for sem, g in df[df["ts"] >= cutoff].groupby("semaine"):
        if len(g) < 500:  # semaine incomplète
            continue
        row = {"semaine": sem.date(), "transactions": len(g),
               "taux_fraude_reel": g["is_fraud"].mean(), "proba_moyenne": g["proba"].mean(),
               "taux_alertes": (g["niveau"] != "faible").mean()}
        for v in VARIABLES:
            row[f"psi_{v}"] = psi(ref[v], g[v], RISK_EDGES if v == "proba" else None)
        rows.append(row)

    out = pd.DataFrame(rows)
    name = "drift_report_simulation.csv" if simulate else "drift_report.csv"
    out.to_csv(EXPORTS / name, index=False)
    print(out.round(3).to_string(index=False))
    psi_cols = [c for c in out.columns if c.startswith("psi_")]
    worst = out[psi_cols].iloc[-1]
    print(f"\nPSI maximal (dernière semaine) : {worst.max():.3f} sur {worst.idxmax().replace('psi_', '')}")
    print("Statut :", "stable" if worst.max() < 0.10 else "à surveiller" if worst.max() < 0.25 else "DÉRIVE SIGNIFICATIVE")
    print(f"Fichier écrit : exports/{name}")


if __name__ == "__main__":
    main(simulate="--simulate" in sys.argv)
