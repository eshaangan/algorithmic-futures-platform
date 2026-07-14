import numpy as np
import pandas as pd


def compute_bar_features(ticks: pd.DataFrame, large_trade_threshold: float = 5.0) -> dict[str, float]:
    required = {"ts_recv", "price", "size", "side"}
    missing = required.difference(ticks.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if ticks.empty:
        return {}
    ordered = ticks.sort_values("ts_recv")
    price = ordered["price"].to_numpy(float)
    size = ordered["size"].to_numpy(float)
    side = ordered["side"].to_numpy(str)
    signed = np.where(side == "B", size, np.where(side == "A", -size, 0.0))
    large = size >= large_trade_threshold
    total = float(size.sum())
    ofi = float(signed.sum())
    changes = np.diff(price)
    roll = np.nan
    if changes.size >= 3:
        covariance = float(np.cov(changes[:-1], changes[1:])[0, 1])
        roll = float(2 * np.sqrt(max(-covariance, 0.0)))
    return {
        "buy_volume": float(size[side == "B"].sum()),
        "sell_volume": float(size[side == "A"].sum()),
        "volume": total,
        "ofi": ofi,
        "ofi_imbalance": ofi / total if total else 0.0,
        "large_trade_fraction": float(size[large].sum() / total) if total else 0.0,
        "large_ofi": float(signed[large].sum()),
        "kyle_lambda": float(abs(price[-1] - price[0]) / abs(ofi)) if ofi else np.nan,
        "roll_spread": roll,
        "vwap": float(np.average(price, weights=size)) if total else float(price[-1]),
        "open": float(price[0]), "high": float(price.max()), "low": float(price.min()), "close": float(price[-1]),
    }


def compute_vpin(buy_volume: pd.Series, sell_volume: pd.Series, window: int = 20) -> pd.Series:
    total = buy_volume + sell_volume
    return (buy_volume - sell_volume).abs().rolling(window).sum() / total.rolling(window).sum().replace(0, np.nan)
