from pathlib import Path

import numpy as np
import pandas as pd


def aggregate_ticks(ticks: pd.DataFrame, frequency: str = "1min", price_scale: float = 1.0) -> pd.DataFrame:
    """Aggregate trades into UTC OHLCV bars without depending on input order."""
    required = {"ts_recv", "price", "size", "side"}
    missing = required.difference(ticks.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if price_scale <= 0:
        raise ValueError("price_scale must be positive")
    if (ticks["size"] < 0).any():
        raise ValueError("trade size must be non-negative")
    invalid_sides = set(ticks["side"].dropna().unique()).difference({"A", "B"})
    if invalid_sides:
        raise ValueError(f"invalid trade sides: {sorted(invalid_sides)}")

    frame = ticks.loc[ticks["price"] > 0].sort_values("ts_recv", kind="stable").copy()
    frame["price"] = frame["price"] / price_scale
    frame["timestamp"] = pd.to_datetime(frame["ts_recv"], unit="ns", utc=True).dt.floor(frequency)
    frame["buy_volume"] = np.where(frame["side"] == "B", frame["size"], 0.0)
    frame["sell_volume"] = np.where(frame["side"] == "A", frame["size"], 0.0)
    grouped = frame.groupby("timestamp", sort=True)
    return grouped.agg(
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        volume=("size", "sum"),
        trades=("size", "size"),
        buy_volume=("buy_volume", "sum"),
        sell_volume=("sell_volume", "sum"),
    )


def read_ticks_chunked(path: Path, chunksize: int, price_scale: float = 1.0) -> pd.DataFrame:
    """Read a time-ordered tick CSV in bounded chunks and merge partial bars."""
    if chunksize < 1:
        raise ValueError("chunksize must be positive")
    parts = []
    last_timestamp = None
    for chunk in pd.read_csv(path, chunksize=chunksize):
        timestamps = chunk["ts_recv"]
        if not timestamps.is_monotonic_increasing:
            raise ValueError("tick CSV must be ordered by ts_recv")
        if last_timestamp is not None and not timestamps.empty and timestamps.iloc[0] < last_timestamp:
            raise ValueError("tick CSV must be ordered by ts_recv")
        if not timestamps.empty:
            last_timestamp = timestamps.iloc[-1]
        parts.append(aggregate_ticks(chunk, price_scale=price_scale))
    if not parts:
        return pd.DataFrame()
    combined = pd.concat(parts).reset_index()
    return combined.groupby("timestamp", sort=True).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        trades=("trades", "sum"),
        buy_volume=("buy_volume", "sum"),
        sell_volume=("sell_volume", "sum"),
    )
