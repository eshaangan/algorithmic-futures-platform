import numpy as np
import pandas as pd
import pytest
from core.execution import CostModel, vector_backtest
from core.microstructure import compute_bar_features
from core.risk import PropRiskConfig, PropRiskEngine


def test_daily_and_trailing_risk_locks():
    engine = PropRiskEngine(PropRiskConfig(max_daily_loss=500, trailing_drawdown=1000))
    engine.mark(pd.Timestamp("2025-01-02 10:00"), -500)
    assert engine.daily_locked
    engine.mark(pd.Timestamp("2025-01-03 10:00"), 1200)
    assert not engine.daily_locked
    engine.mark(pd.Timestamp("2025-01-03 11:00"), -1000)
    assert engine.permanently_locked


def test_daily_risk_resets_at_futures_session_boundary():
    engine = PropRiskEngine(PropRiskConfig(max_daily_loss=500))
    engine.mark(pd.Timestamp("2025-01-02 16:59", tz="America/Chicago"), -500)
    assert engine.daily_locked
    engine.mark(pd.Timestamp("2025-01-02 17:00", tz="America/Chicago"), 0)
    assert engine.can_trade
    assert engine.daily_pnl == 0


def test_projected_trade_checks_daily_and_trailing_boundaries():
    engine = PropRiskEngine(
        PropRiskConfig(max_daily_loss=500, trailing_drawdown=1_000)
    )
    engine.mark(pd.Timestamp("2025-01-02 10:00"), 800)
    assert not engine.projected_trade_allowed(1_000)
    assert engine.projected_trade_allowed(999)
    daily_engine = PropRiskEngine(
        PropRiskConfig(max_daily_loss=500, trailing_drawdown=1_000)
    )
    daily_engine.mark(pd.Timestamp("2025-01-02 10:01"), -300)
    assert not daily_engine.projected_trade_allowed(200)


def test_microstructure_signed_flow_and_vwap():
    ticks = pd.DataFrame({"ts_recv": [1, 2, 3], "price": [100, 101, 100], "size": [2, 6, 1], "side": ["B", "A", "B"]})
    result = compute_bar_features(ticks)
    assert result["ofi"] == -3
    assert result["large_trade_fraction"] == 6 / 9
    assert np.isclose(result["vwap"], 906 / 9)


def test_backtest_shifts_signal_and_charges_turnover():
    close = pd.Series([100.0, 110.0, 121.0])
    signal = pd.Series([1.0, 0.0, 0.0])
    result = vector_backtest(close, signal, CostModel(1, 1))
    assert result.position.tolist() == [0, 1, 0]
    assert np.isclose(result.loc[1, "gross_return"], 0.1)
    assert result["cost"].sum() > 0


def test_backtest_charges_two_way_reversal_and_rejects_bad_inputs():
    close = pd.Series([100.0, 101.0, 102.0])
    signal = pd.Series([1.0, -1.0, 0.0])
    result = vector_backtest(close, signal, CostModel(1, 1))
    assert np.isclose(result.loc[2, "cost"], 2 * CostModel(1, 1).one_way_rate)
    with pytest.raises(ValueError, match="finite"):
        vector_backtest(close, pd.Series([1.0, np.nan, 0.0]))
    with pytest.raises(ValueError, match="non-negative"):
        CostModel(-1, 1)
    with pytest.raises(ValueError, match="finite"):
        CostModel(np.nan, 1)
    duplicate_index = pd.Index([0, 0, 1])
    with pytest.raises(ValueError, match="sorted and unique"):
        vector_backtest(
            pd.Series([100.0, 101.0, 102.0], index=duplicate_index),
            pd.Series([0.0, 1.0, 0.0], index=duplicate_index),
        )
