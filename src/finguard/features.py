"""Features comportementales. RÈGLE : chaque feature n'utilise que le passé
strict de la transaction (aucune fuite de données)."""
import numpy as np
import pandas as pd

NUM_FEATURES = [
    "amt", "hour", "dayofweek", "age", "city_pop", "distance_km",
    "tx_count_1h", "tx_count_24h", "amt_sum_24h",
    "amt_ratio_to_mean", "secs_since_last", "new_merchant",
]
CAT_FEATURES = ["category", "gender"]


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ts"] = pd.to_datetime(df["trans_date_trans_time"])
    df = df.sort_values(["cc_num", "ts"], kind="mergesort").reset_index(drop=True)

    # Features sans état
    df["hour"] = df["ts"].dt.hour
    df["dayofweek"] = df["ts"].dt.dayofweek
    df["age"] = (df["ts"] - pd.to_datetime(df["dob"])).dt.days / 365.25
    df["distance_km"] = haversine_km(df["lat"], df["long"], df["merch_lat"], df["merch_long"])

    g = df.groupby("cc_num", sort=False)

    # Fenêtres glissantes : closed='left' EXCLUT la transaction courante
    # Le df est trié par (cc_num, ts) : l'ordre du résultat est identique, on aligne par position
    # (l'index du résultat dépend de la version de pandas, donc on évite l'alignement par index).
    def roll(win, agg):
        r = getattr(g.rolling(win, on="ts", closed="left")["amt"], agg)()
        return r.to_numpy()

    df["tx_count_1h"] = pd.Series(roll("1h", "count")).fillna(0).to_numpy()
    df["tx_count_24h"] = pd.Series(roll("24h", "count")).fillna(0).to_numpy()
    df["amt_sum_24h"] = pd.Series(roll("24h", "sum")).fillna(0).to_numpy()

    # Écart au comportement habituel (moyenne des montants passés)
    past_n = g.cumcount()
    past_mean = (g["amt"].cumsum() - df["amt"]) / past_n.replace(0, np.nan)
    df["amt_ratio_to_mean"] = (df["amt"] / past_mean).fillna(1.0)

    df["secs_since_last"] = g["ts"].diff().dt.total_seconds().fillna(-1)
    df["new_merchant"] = (~df.duplicated(["cc_num", "merchant"])).astype(int)
    return df
