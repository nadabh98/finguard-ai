"""Page Surveillance : dérive des données (PSI) dans la console."""
from pathlib import Path

import pandas as pd
import streamlit as st

EXPORTS = Path(__file__).resolve().parents[1] / "exports"


def statut(psi):
    if psi < 0.10:
        return "stable"
    if psi < 0.25:
        return "à surveiller"
    return "dérive significative"


def page_drift():
    st.title("Surveillance du modèle")
    st.caption("PSI (indice de stabilité de population) par semaine, comparé aux deux premières semaines du jeu de test.")
    src = st.radio("Données", ["Réelles", "Simulation de dérive"], horizontal=True)
    f = EXPORTS / ("drift_report.csv" if src == "Réelles" else "drift_report_simulation.csv")
    if not f.exists():
        st.warning("Fichier absent. Lance : PYTHONPATH=src python -m finguard.monitor"
                   + ("" if src == "Réelles" else " --simulate"))
        return
    d = pd.read_csv(f, parse_dates=["semaine"])
    psi_cols = [c for c in d.columns if c.startswith("psi_")]
    last = d.iloc[-1]
    worst = last[psi_cols].astype(float).idxmax()
    a, b, c = st.columns(3)
    a.metric("Dernière semaine", str(last["semaine"].date()))
    b.metric("PSI maximal", f"{last[worst]:.3f}", worst.replace("psi_", ""), delta_color="off")
    c.metric("Statut", statut(last[worst]))
    st.subheader("PSI par variable")
    st.line_chart(d.set_index("semaine")[psi_cols])
    st.subheader("Suivi du score")
    st.line_chart(d.set_index("semaine")[["taux_fraude_reel", "proba_moyenne", "taux_alertes"]])
    st.dataframe(d.round(4), hide_index=True)
    st.caption("PSI inférieur à 0,10 : stable. De 0,10 à 0,25 : à surveiller. "
               "Au-dessus de 0,25 : dérive significative, réentraînement à envisager.")
