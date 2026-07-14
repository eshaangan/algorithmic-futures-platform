import numpy as np
import pandas as pd


def triple_barrier_labels(close: pd.Series, events: pd.DataFrame, target: float, stop: float) -> pd.DataFrame:
    """Label events by the first horizontal barrier touched before ``t1``."""
    if target <= 0 or stop <= 0:
        raise ValueError("target and stop must be positive")
    missing = {"event_id", "t0", "t1"}.difference(events.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if events["event_id"].duplicated().any():
        raise ValueError("event_id values must be unique")
    if not close.index.is_monotonic_increasing or not close.index.is_unique:
        raise ValueError("close index must be sorted and unique")
    if not np.isfinite(close.to_numpy(dtype=float)).all() or (close <= 0).any():
        raise ValueError("close prices must be finite and positive")
    rows = []
    for event in events.itertuples():
        if event.t1 < event.t0:
            raise ValueError("event end precedes start")
        path = close.loc[event.t0:event.t1]
        if path.empty:
            raise ValueError("event interval has no prices")
        origin = float(path.iloc[0])
        returns = path / origin - 1
        target_hits = np.flatnonzero(returns.to_numpy() >= target)
        stop_hits = np.flatnonzero(returns.to_numpy() <= -stop)
        target_at = target_hits[0] if target_hits.size else len(path)
        stop_at = stop_hits[0] if stop_hits.size else len(path)
        label = 1 if target_at < stop_at else -1 if stop_at < target_at else 0
        touch = min(target_at, stop_at, len(path) - 1)
        rows.append({"event_id": event.event_id, "label": label, "t_touch": path.index[touch]})
    return pd.DataFrame(rows, columns=["event_id", "label", "t_touch"])
