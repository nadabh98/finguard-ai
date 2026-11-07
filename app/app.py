"""Console analyste FinGuard AI (Streamlit)."""
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from drift_page import page_drift

ROOT = Path(__file__).resolve().parents[1]
EXPORTS, MODELS = ROOT / "exports", ROOT / "models"
DEC = EXPORTS / "decisions.csv"
API_URL = os.getenv("API_URL", "http://localhost:8080").rstrip("/")
API_TOKEN = os.getenv("API_TOKEN", "")
COLORS = {"faible": "#2e7d32", "moyen": "#f9a825", "élevé": "#ef6c00", "blocage": "#c62828"}
CATEGORIES = ["entertainment", "food_dining", "gas_transport", "grocery_net", "grocery_pos",
              "health_fitness", "home", "kids_pets", "misc_net", "misc_pos", "personal_care",
              "shopping_net", "shopping_pos", "travel"]
SUSPECTE = dict(amt=1081.63, hour=22, dayofweek=2, age=45.0, city_pop=5000, distance_km=40.0,
                tx_count_1h=2, tx_count_24h=6, amt_sum_24h=6154.4, amt_ratio_to_mean=8.0,
                secs_since_last=600.0, new_merchant=1, category="shopping_net", gender="F")
BANALE = dict(amt=18.5, hour=14, dayofweek=3, age=45.0, city_pop=50000, distance_km=10.0,
              tx_count_1h=0, tx_count_24h=1, amt_sum_24h=25.0, amt_ratio_to_mean=0.8,
              secs_since_last=40000.0, new_merchant=0, category="food_dining", gender="F")

st.set_page_config(page_title="FinGuard AI", page_icon="🛡️", layout="wide")


def badge(niveau):
    return (f"<span style='background:{COLORS[niveau]};color:white;padding:4px 14px;"
            f"border-radius:14px;font-weight:600'>{niveau.upper()}</span>")


# ---------------------------------------------------------------- Simulateur
def page_simulateur():
    st.title("Simulateur de scoring")
    st.caption(f"Chaque envoi appelle l'API : {API_URL}")
    preset = st.radio("Exemple", ["Transaction suspecte", "Transaction banale"], horizontal=True)
    p = SUSPECTE if preset == "Transaction suspecte" else BANALE
    k = preset[:5]  # change les clés pour réinitialiser les champs à chaque exemple

    with st.form("tx"):
        c1, c2, c3 = st.columns(3)
        amt = c1.number_input("Montant", min_value=0.01, value=float(p["amt"]), key=f"amt{k}")
        hour = c1.number_input("Heure (0-23)", min_value=0, max_value=23, value=int(p["hour"]), key=f"hour{k}")
        dow = c1.number_input("Jour (0 = lundi)", min_value=0, max_value=6, value=int(p["dayofweek"]), key=f"dow{k}")
        age = c1.number_input("Âge du client", min_value=0.0, max_value=120.0, value=float(p["age"]), key=f"age{k}")
        pop = c2.number_input("Population de la ville", min_value=0, value=int(p["city_pop"]), key=f"pop{k}")
        dist = c2.number_input("Distance client-marchand (km)", min_value=0.0, value=float(p["distance_km"]), key=f"dist{k}")
        n1 = c2.number_input("Transactions sur 1 h", min_value=0, value=int(p["tx_count_1h"]), key=f"n1{k}")
        n24 = c2.number_input("Transactions sur 24 h", min_value=0, value=int(p["tx_count_24h"]), key=f"n24{k}")
        s24 = c3.number_input("Somme dépensée sur 24 h", min_value=0.0, value=float(p["amt_sum_24h"]), key=f"s24{k}")
        ratio = c3.number_input("Montant / moyenne habituelle", min_value=0.0, value=float(p["amt_ratio_to_mean"]), key=f"ratio{k}")
        secs = c3.number_input("Secondes depuis la dernière transaction", min_value=-1.0, value=float(p["secs_since_last"]), key=f"secs{k}")
        new = c3.selectbox("Nouveau marchand", [0, 1], index=int(p["new_merchant"]), key=f"new{k}")
        cat = c2.selectbox("Catégorie", CATEGORIES, index=CATEGORIES.index(p["category"]), key=f"cat{k}")
        gen = c3.selectbox("Genre", ["F", "M"], index=["F", "M"].index(p["gender"]), key=f"gen{k}")
        go = st.form_submit_button("Scorer la transaction", type="primary")

    if not go:
        return
    payload = dict(amt=amt, hour=int(hour), dayofweek=int(dow), age=age, city_pop=int(pop),
                   distance_km=dist, tx_count_1h=n1, tx_count_24h=n24, amt_sum_24h=s24,
                   amt_ratio_to_mean=ratio, secs_since_last=secs, new_merchant=int(new),
                   category=cat, gender=gen)
    headers = {"Authorization": f"Bearer {API_TOKEN}"} if API_TOKEN else {}
    try:
        r = requests.post(f"{API_URL}/score", json=payload, headers=headers, timeout=60)
    except requests.RequestException as e:
        st.error(f"API injoignable : {e}")
        st.info("Lance l'API dans un autre terminal : PYTHONPATH=src uvicorn finguard.api:app --port 8080")
        return
    if r.status_code != 200:
        st.error(f"Erreur {r.status_code} : {r.text}")
        return
    res = r.json()
    st.divider()
    a, b, c = st.columns(3)
    a.metric("Probabilité de fraude", f"{res['probabilite_fraude']:.1%}")
    b.markdown("**Niveau de risque**")
    b.markdown(badge(res["niveau"]), unsafe_allow_html=True)
    c.metric("Action recommandée", res["action"])
    st.subheader("Pourquoi ce score ?")
    if res["raisons"]:
        df = pd.DataFrame(res["raisons"])
        st.bar_chart(df.set_index("variable")["impact"])
        st.dataframe(df, hide_index=True)
    else:
        st.write("Aucun facteur de risque marquant.")


# ------------------------------------------------------------ File d'alertes
@st.cache_data
def load_scored():
    f = EXPORTS / "scored_transactions.csv"
    return pd.read_csv(f, parse_dates=["ts"], dtype={"cc_num": str}) if f.exists() else None


def load_decisions():
    if DEC.exists():
        return pd.read_csv(DEC)
    return pd.DataFrame(columns=["trans_num", "decision", "date"])


def save_decision(trans_num, decision):
    row = pd.DataFrame([{"trans_num": trans_num, "decision": decision,
                         "date": datetime.now().isoformat(timespec="seconds")}])
    row.to_csv(DEC, mode="a", header=not DEC.exists(), index=False)


def page_alertes():
    st.title("File d'alertes")
    df = load_scored()
    if df is None:
        st.warning("Fichier absent. Lance : PYTHONPATH=src python -m finguard.export")
        return
    dec = load_decisions().drop_duplicates("trans_num", keep="last")[["trans_num", "decision"]]
    df = df.merge(dec, on="trans_num", how="left")
    df["decision"] = df["decision"].fillna("à traiter")

    f1, f2, f3 = st.columns(3)
    niveaux = f1.multiselect("Niveaux", list(COLORS), default=["moyen", "élevé", "blocage"])
    cats = f2.multiselect("Catégories", sorted(df["category"].unique()))
    min_amt = f3.number_input("Montant minimum", min_value=0.0, value=0.0)
    view = df[df["niveau"].isin(niveaux) & (df["amt"] >= min_amt)]
    if cats:
        view = view[view["category"].isin(cats)]
    view = view.sort_values("proba", ascending=False).reset_index(drop=True)

    k1, k2, k3 = st.columns(3)
    k1.metric("Alertes", f"{len(view):,}")
    k2.metric("Montant exposé", f"{view['amt'].sum():,.0f} €")
    k3.metric("Déjà traitées", int((view["decision"] != "à traiter").sum()))

    shown = view[["ts", "merchant", "category", "amt", "proba", "niveau", "decision"]].copy()
    shown["proba"] = (shown["proba"] * 100).round(1)
    event = st.dataframe(
        shown, hide_index=True, on_select="rerun", selection_mode="single-row",
        column_config={"proba": st.column_config.ProgressColumn("Risque %", min_value=0, max_value=100, format="%.1f"),
                       "amt": st.column_config.NumberColumn("Montant", format="%.2f €")})
    rows = event.selection.rows
    if not rows:
        st.info("Clique sur une ligne pour voir le détail et traiter l'alerte.")
        return

    r = view.iloc[rows[0]]
    st.divider()
    st.subheader(f"Alerte : {r['merchant']}, {r['amt']:.2f} €")
    a, b, c = st.columns(3)
    a.metric("Probabilité", f"{r['proba']:.1%}")
    b.markdown("**Niveau**")
    b.markdown(badge(r["niveau"]), unsafe_allow_html=True)
    c.metric("Action recommandée", r["action"])
    st.markdown("**Raisons principales**")
    for part in str(r["raisons"]).split(" ; "):
        st.markdown(f"- {part}")
    hist = df[(df["cc_num"] == r["cc_num"]) & (df["ts"] <= r["ts"])].sort_values("ts").tail(8)
    st.markdown("**Dernières transactions de ce client**")
    st.dataframe(hist[["ts", "merchant", "category", "amt", "proba", "niveau"]], hide_index=True)

    d1, d2, _ = st.columns([1, 1, 3])
    if d1.button("Confirmer la fraude", type="primary"):
        save_decision(r["trans_num"], "fraude confirmée")
        st.rerun()
    if d2.button("Faux positif"):
        save_decision(r["trans_num"], "faux positif")
        st.rerun()


# --------------------------------------------------------------- Performance
def page_performance():
    st.title("Performance du modèle")
    mf = MODELS / "metrics.json"
    if not mf.exists():
        st.warning("Lance d'abord l'entraînement.")
        return
    m = pd.DataFrame(json.loads(mf.read_text()))
    best = m[m["model"] == "lightgbm"].iloc[0]
    a, b, c, d = st.columns(4)
    a.metric("PR-AUC", f"{best['pr_auc']:.3f}")
    b.metric("Rappel", f"{best['recall']:.1%}")
    c.metric("Précision", f"{best['precision']:.1%}")
    d.metric("Coût évité", f"{best['cost_no_model'] - best['expected_cost']:,.0f} €")
    st.subheader("Comparaison des modèles")
    st.dataframe(m.round(3), hide_index=True)
    rl = MODELS / "risk_levels.csv"
    if rl.exists():
        st.subheader("Niveaux de risque (jeu de test)")
        t = pd.read_csv(rl)
        t["taux_fraude_%"] = (t["fraudes"] / t["transactions"] * 100).round(2)
        st.dataframe(t, hide_index=True)
        st.bar_chart(t.set_index("niveau")["fraudes"])
    st.caption("Données synthétiques : ces performances ne se transposent pas telles quelles à une vraie banque.")


PAGES = {"Simulateur": page_simulateur, "File d'alertes": page_alertes, "Performance": page_performance, "Surveillance": page_drift}
st.sidebar.title("🛡️ FinGuard AI")
PAGES[st.sidebar.radio("Navigation", list(PAGES))]()
st.sidebar.caption("Console analyste")
