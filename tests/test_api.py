import pytest
from fastapi.testclient import TestClient

from finguard.api import app
from finguard.config import MODELS

pytestmark = pytest.mark.skipif(not (MODELS / "calibrator.joblib").exists(), reason="modèles absents")

TX = dict(amt=1081.63, hour=22, dayofweek=2, age=45, city_pop=5000, distance_km=40,
          tx_count_1h=2, tx_count_24h=6, amt_sum_24h=6154.4, amt_ratio_to_mean=8,
          secs_since_last=600, new_merchant=1, category="shopping_net", gender="F")


def test_health():
    with TestClient(app) as c:
        assert c.get("/health").json() == {"status": "ok"}


def test_score_returns_expected_fields():
    with TestClient(app) as c:
        r = c.post("/score", json=TX)
        assert r.status_code == 200
        body = r.json()
        assert 0 <= body["probabilite_fraude"] <= 1
        assert body["niveau"] in {"faible", "moyen", "élevé", "blocage"}
        assert "action" in body and "raisons" in body


def test_invalid_inputs_rejected():
    with TestClient(app) as c:
        assert c.post("/score", json={**TX, "amt": -5}).status_code == 422
        assert c.post("/score", json={**TX, "category": "inconnue"}).status_code == 422
