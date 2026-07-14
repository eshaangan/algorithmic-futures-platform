import numpy as np
import pandas as pd


def uniqueness_weights(events: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """Average inverse event concurrency over each inclusive event interval."""
    missing = {"event_id", "t0", "t1"}.difference(events.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if events["event_id"].duplicated().any():
        raise ValueError("event_id values must be unique")
    if not index.is_monotonic_increasing or not index.is_unique:
        raise ValueError("index must be sorted and unique")
    starts = index.get_indexer(pd.to_datetime(events["t0"]))
    ends = index.get_indexer(pd.to_datetime(events["t1"]))
    if (starts < 0).any() or (ends < starts).any():
        raise ValueError("event intervals must align to index")
    delta = np.zeros(len(index) + 1)
    np.add.at(delta, starts, 1)
    np.add.at(delta, ends + 1, -1)
    concurrency = np.cumsum(delta)[:-1]
    inverse = np.divide(1, concurrency, out=np.zeros_like(concurrency), where=concurrency > 0)
    prefix = np.r_[0.0, np.cumsum(inverse)]
    weights = (prefix[ends + 1] - prefix[starts]) / (ends - starts + 1)
    return pd.Series(weights, index=events["event_id"], name="uniqueness")
