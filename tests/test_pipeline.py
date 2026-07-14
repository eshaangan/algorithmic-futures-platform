import numpy as np
import pandas as pd
import pytest
from pipeline.labels import triple_barrier_labels
from pipeline.weights import uniqueness_weights
from pipeline.replay import replay_decisions
from strategies.rule_based.validation import PreRegistration, evaluate_gate


def test_replay_exposes_only_point_in_time_history():
    features = pd.DataFrame({"momentum": [-1, 2, 3, -2]})
    observed_lengths = []

    def decide(history):
        observed_lengths.append(len(history))
        return int(history["momentum"].iloc[-1] > 0)

    result = replay_decisions(features, decide)
    assert observed_lengths == [1, 2, 3, 4]
    assert result.tolist() == [0, 1, 1, 0]


def test_preregistration_hash_and_gate_are_stable():
    prereg = PreRegistration("demo", {"start": "2025-01-01", "end": "2025-03-31"}, 5, {"min_trades": 3, "min_months": 3, "min_positive_month_fraction": 0.66, "min_dsr": 0.0}, {"threshold": 1})
    reordered = PreRegistration("demo", {"end": "2025-03-31", "start": "2025-01-01"}, 5, dict(reversed(list(prereg.gate.items()))), {"threshold": 1})
    assert prereg.content_hash() == reordered.content_hash()
    trades = pd.DataFrame({"exit_time": pd.to_datetime(["2025-01-10", "2025-02-10", "2025-03-10"]), "pnl": [2.0, 1.0, 3.0]})
    assert evaluate_gate(trades, prereg)["verdict"] == "GO"


def test_triple_barrier_uses_first_touch_and_rejects_nonfinite_prices():
    index = pd.date_range("2025-01-01", periods=4, freq="min", tz="UTC")
    close = pd.Series([100.0, 102.0, 97.0, 100.0], index=index)
    events = pd.DataFrame({"event_id": [1], "t0": [index[0]], "t1": [index[-1]]})
    result = triple_barrier_labels(close, events, target=0.01, stop=0.02)
    assert result.loc[0, "label"] == 1
    assert result.loc[0, "t_touch"] == index[1]
    with pytest.raises(ValueError, match="finite"):
        triple_barrier_labels(close.mask(close.index == index[1], np.nan), events, 0.01, 0.02)


def test_uniqueness_weights_handle_overlap_and_validate_ids():
    index = pd.date_range("2025-01-01", periods=4, freq="min", tz="UTC")
    events = pd.DataFrame(
        {
            "event_id": ["a", "b"],
            "t0": [index[0], index[1]],
            "t1": [index[2], index[3]],
        }
    )
    weights = uniqueness_weights(events, index)
    assert np.isclose(weights["a"], (1 + 0.5 + 0.5) / 3)
    assert np.isclose(weights["b"], (0.5 + 0.5 + 1) / 3)
    with pytest.raises(ValueError, match="unique"):
        uniqueness_weights(events.assign(event_id=["a", "a"]), index)
