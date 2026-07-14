from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostModel:
    commission_bps: float = 0.5
    slippage_bps: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite([self.commission_bps, self.slippage_bps]).all():
            raise ValueError("execution costs must be finite")
        if self.commission_bps < 0 or self.slippage_bps < 0:
            raise ValueError("execution costs must be non-negative")

    @property
    def one_way_rate(self) -> float:
        return (self.commission_bps + self.slippage_bps) / 10_000.0


def vector_backtest(
    close: pd.Series,
    desired_position: pd.Series,
    costs: CostModel = CostModel(),
) -> pd.DataFrame:
    """Apply one-bar-lagged positions and one-way costs per unit of turnover."""
    if not close.index.equals(desired_position.index):
        raise ValueError("close and desired_position indexes must match")
    if not close.index.is_monotonic_increasing or not close.index.is_unique:
        raise ValueError("execution index must be sorted and unique")
    if close.empty:
        return pd.DataFrame(
            columns=["position", "gross_return", "cost", "net_return", "equity"],
            index=close.index,
            dtype=float,
        )
    if not np.isfinite(close.to_numpy(dtype=float)).all() or (close <= 0).any():
        raise ValueError("close prices must be finite and positive")
    if not np.isfinite(desired_position.to_numpy(dtype=float)).all():
        raise ValueError("desired_position must contain only finite values")
    position = desired_position.shift(1).fillna(0.0).clip(-1, 1)
    returns = close.pct_change().fillna(0.0)
    turnover = position.diff().abs().fillna(position.abs())
    cost = turnover * costs.one_way_rate
    net = position * returns - cost
    return pd.DataFrame(
        {
            "position": position,
            "gross_return": position * returns,
            "cost": cost,
            "net_return": net,
            "equity": (1 + net).cumprod(),
        }
    )
