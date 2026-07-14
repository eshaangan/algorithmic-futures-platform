import pandas as pd
import pytest

from strategies.rule_based.engine import RuleEngine, RuleSignal
from strategies.rule_based.orb import OpeningRangeBreakout


def test_orb_requires_first_confirmed_break():
    index = pd.date_range("2025-01-02 14:30", periods=8, freq="5min", tz="UTC")
    bars = pd.DataFrame({"high": [100, 101, 101, 101, 101, 101, 101, 102], "low": [99, 99, 99, 99, 99, 99, 99, 100], "close": [100, 100, 100, 100, 100, 100, 101, 102]}, index=index)
    signal = OpeningRangeBreakout(30).evaluate(bars)
    assert signal.direction == 1
    assert signal.metadata["range_high"] == 101


def test_orb_ignores_premarket_bars():
    index = pd.to_datetime(
        [
            "2025-01-02 13:00Z",
            "2025-01-02 14:30Z",
            "2025-01-02 14:45Z",
            "2025-01-02 14:55Z",
            "2025-01-02 15:00Z",
        ]
    )
    bars = pd.DataFrame(
        {
            "high": [200, 100, 101, 101, 102],
            "low": [50, 99, 99, 99, 100],
            "close": [150, 100, 100, 101, 102],
        },
        index=index,
    )
    signal = OpeningRangeBreakout(30).evaluate(bars)
    assert signal.direction == 1
    assert signal.metadata["range_high"] == 101


def test_orb_rejects_malformed_bars():
    index = pd.date_range("2025-01-02 14:30", periods=2, freq="5min", tz="UTC")
    with pytest.raises(ValueError, match="missing columns"):
        OpeningRangeBreakout().evaluate(pd.DataFrame({"close": [1, 2]}, index=index))


def test_orb_does_not_emit_after_regular_session():
    index = pd.to_datetime(
        [
            "2025-01-02 14:30Z",
            "2025-01-02 14:45Z",
            "2025-01-02 14:55Z",
            "2025-01-02 15:00Z",
            "2025-01-02 22:00Z",
        ]
    )
    bars = pd.DataFrame(
        {
            "high": [100, 101, 101, 101, 103],
            "low": [99, 99, 99, 99, 100],
            "close": [100, 100, 101, 101, 103],
        },
        index=index,
    )
    signal = OpeningRangeBreakout(30).evaluate(bars)
    assert signal.direction == 0
    assert signal.reason == "outside regular session"


def test_rule_engine_returns_neutral_on_exact_tie():
    class FixedRule:
        def __init__(self, signal):
            self.signal = signal

        def evaluate(self, bars):
            return self.signal

    engine = RuleEngine(
        [
            FixedRule(RuleSignal(1, 0.5, "long")),
            FixedRule(RuleSignal(-1, 0.5, "short")),
        ]
    )
    assert engine.evaluate(pd.DataFrame()).direction == 0
