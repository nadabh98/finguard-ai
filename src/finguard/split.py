import pandas as pd
from .config import TRAIN_FRAC, VAL_FRAC


def temporal_split(df: pd.DataFrame, time_col: str = "ts"):
    """Découpe chronologique : jamais aléatoire."""
    df = df.sort_values(time_col)
    t1 = df[time_col].quantile(TRAIN_FRAC)
    t2 = df[time_col].quantile(TRAIN_FRAC + VAL_FRAC)
    train, val, test = df[df[time_col] <= t1], df[(df[time_col] > t1) & (df[time_col] <= t2)], df[df[time_col] > t2]
    assert train[time_col].max() < val[time_col].min() <= val[time_col].max() < test[time_col].min()
    return train, val, test
