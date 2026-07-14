from itertools import combinations
import numpy as np
import pandas as pd


def _normalise(events: pd.DataFrame) -> pd.DataFrame:
    missing = {"event_id", "t0", "t1"}.difference(events.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    out = events.copy()
    out["t0"] = pd.to_datetime(out["t0"], utc=True)
    out["t1"] = pd.to_datetime(out["t1"], utc=True)
    if out[["t0", "t1"]].isna().any().any():
        raise ValueError("event timestamps must not be missing")
    if (out["t1"] < out["t0"]).any():
        raise ValueError("event end precedes start")
    if out["event_id"].duplicated().any():
        raise ValueError("event_id values must be unique")
    return out.sort_values("t0").reset_index(drop=True)


def build_purged_kfold_splits(events: pd.DataFrame, bars_index: pd.DatetimeIndex, n_splits: int, embargo_bars: int = 0) -> list[dict]:
    frame = _normalise(events)
    if n_splits < 2 or n_splits > len(frame):
        raise ValueError("n_splits must be between 2 and event count")
    if embargo_bars < 0:
        raise ValueError("embargo_bars must be non-negative")
    bars = pd.DatetimeIndex(pd.to_datetime(bars_index, utc=True))
    if not bars.is_monotonic_increasing or not bars.is_unique:
        raise ValueError("bars_index must be sorted and unique")
    splits = []
    for fold, test_positions in enumerate(np.array_split(np.arange(len(frame)), n_splits)):
        test = frame.iloc[test_positions]
        start, end = test["t0"].min(), test["t1"].max()
        overlap = (frame["t0"] <= end) & (frame["t1"] >= start)
        test_mask = frame.index.isin(test_positions)
        train_mask = ~test_mask & ~overlap
        first_after_end = bars.searchsorted(end, side="right")
        if embargo_bars and first_after_end < len(bars):
            embargo_stop = min(first_after_end + embargo_bars, len(bars))
            embargo_end = bars[embargo_stop - 1]
            embargo = (frame["t0"] > end) & (frame["t0"] <= embargo_end)
        else:
            embargo = pd.Series(False, index=frame.index)
        purged = int((~test_mask & overlap).sum())
        embargoed = int((train_mask & embargo).sum())
        train_mask &= ~embargo
        splits.append({"fold": fold, "train_event_ids": frame.loc[train_mask, "event_id"].tolist(), "test_event_ids": test["event_id"].tolist(), "n_purged": purged, "n_embargoed": embargoed})
    return splits


def build_cpcv_paths(base_folds: list[dict], test_groups: int = 2) -> list[dict]:
    if not 1 <= test_groups < len(base_folds):
        raise ValueError("test_groups must leave at least one training fold")
    required = {"train_event_ids", "test_event_ids"}
    if any(not required.issubset(fold) for fold in base_folds):
        raise ValueError("each fold must contain train_event_ids and test_event_ids")
    all_ids = set().union(*(set(f["test_event_ids"]) for f in base_folds))
    paths = []
    for path_id, held_out in enumerate(combinations(range(len(base_folds)), test_groups)):
        test_ids = set().union(*(set(base_folds[i]["test_event_ids"]) for i in held_out))
        fold_train = set.intersection(*(set(base_folds[i]["train_event_ids"]) for i in held_out))
        paths.append({"path_id": path_id, "test_folds": held_out, "test_event_ids": sorted(test_ids), "train_event_ids": sorted((all_ids - test_ids) & fold_train)})
    return paths
