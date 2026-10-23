"""Phase 3 : API de scoring de fraude (FastAPI)."""
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from .config import MODELS
from .explain import LABELS, level
from .features import CAT_FEATURES

ACTIONS = {
    "faible": "approuver",
    "moyen": "revue analyste",
    "élevé": "revue prioritaire",
    "blocage": "bloquer et contacter le client",
}


class Transaction(BaseModel):
    amt: float = Field(gt=0, description="Montant de la transaction")
    hour: int = Field(ge=0, le=23)
    dayofweek: int = Field(ge=0, le=6, description="0 = lundi")
    age: float = Field(ge=0, le=120)
    city_pop: int = Field(ge=0)
    distance_km: float = Field(ge=0, description="Distance client-marchand")
    tx_count_1h: float = Field(ge=0)
    tx_count_24h: float = Field(ge=0)
    amt_sum_24h: float = Field(ge=0)
    amt_ratio_to_mean: float = Field(ge=0, description="Montant / moyenne passée du client")
    secs_since_last: float = Field(description="-1 si première transaction")
    new_merchant: int = Field(ge=0, le=1)
    category: str
    gender: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    art = joblib.load(MODELS / "lightgbm.joblib")
    app.state.model = art["model"]
    app.state.feats = art["features"]
    app.state.iso = joblib.load(MODELS / "calibrator.joblib")
    # Catégories connues du modèle, dans l'ordre des colonnes catégorielles
    app.state.cats = app.state.model.booster_.pandas_categorical
    yield


app = FastAPI(title="FinGuard AI", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score")
def score(tx: Transaction, request: Request):
    s = request.app.state
    row = pd.DataFrame([tx.model_dump()])
    for col, cats in zip(CAT_FEATURES, s.cats):
        if row.loc[0, col] not in cats:
            raise HTTPException(422, f"{col} inconnue. Valeurs possibles : {list(cats)}")
        row[col] = pd.Categorical(row[col], categories=cats)
    X = row[s.feats]

    proba = float(s.iso.predict(s.model.predict_proba(X)[:, 1])[0])
    niveau = level(proba)

    contrib = s.model.predict(X, pred_contrib=True)[0][:-1]
    reasons = []
    for j in np.argsort(contrib)[::-1][:3]:
        if contrib[j] > 0:
            val = tx.model_dump()[s.feats[j]]
            reasons.append({"variable": LABELS[s.feats[j]], "valeur": val, "impact": round(float(contrib[j]), 2)})

    return {"probabilite_fraude": round(proba, 4), "niveau": niveau,
            "action": ACTIONS[niveau], "raisons": reasons}
