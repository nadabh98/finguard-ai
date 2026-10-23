import pandas as pd
from finguard.features import build_features
from finguard.split import temporal_split


def make_df():
    rows = []
    base = pd.Timestamp("2020-01-01 10:00")
    for i, amt in enumerate([10, 20, 30, 1000]):
        rows.append(dict(trans_date_trans_time=str(base + pd.Timedelta(minutes=10 * i)), cc_num=1,
                         merchant="m" if i < 3 else "new", category="shopping", amt=amt, gender="F",
                         dob="1990-01-01", city_pop=1000, lat=48.0, long=2.0, merch_lat=48.1, merch_long=2.1, is_fraud=0))
    return pd.DataFrame(rows)


def test_no_leakage_in_windows():
    f = build_features(make_df())
    assert f.loc[0, "tx_count_1h"] == 0 and f.loc[0, "amt_sum_24h"] == 0   # rien dans le passé
    assert f.loc[3, "tx_count_1h"] == 3 and f.loc[3, "amt_sum_24h"] == 60   # la tx courante est exclue
    assert f.loc[3, "amt_ratio_to_mean"] == 1000 / 20                        # moyenne du passé = 20
    assert f.loc[3, "new_merchant"] == 1 and f.loc[1, "new_merchant"] == 0


def test_temporal_split_is_chronological():
    df = build_features(make_df())
    df = pd.concat([df.assign(ts=df["ts"] + pd.Timedelta(days=d)) for d in range(20)])
    tr, va, te = temporal_split(df)
    assert tr["ts"].max() < va["ts"].min() and va["ts"].max() < te["ts"].min()
