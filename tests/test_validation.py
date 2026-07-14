import numpy as np
import pandas as pd
import pytest
from validation.diagnostics import compute_dsr, compute_pbo
from validation.purged_cv import build_purged_kfold_splits


def test_purging_and_embargo_remove_leakage():
    bars = pd.date_range("2025-01-01", periods=20, freq="min", tz="UTC")
    events = pd.DataFrame({"event_id": range(6), "t0": bars[[0, 3, 6, 9, 12, 15]], "t1": bars[[4, 7, 10, 13, 16, 19]]})
    splits = build_purged_kfold_splits(events, bars, 3, embargo_bars=2)
    for split in splits:
        train = events[events.event_id.isin(split["train_event_ids"])]
        test = events[events.event_id.isin(split["test_event_ids"])]
        assert not (((train.t0 <= test.t1.max()) & (train.t1 >= test.t0.min())).any())
    assert any(split["n_embargoed"] > 0 for split in splits)


def test_embargo_counts_exact_bars_after_test_end():
    bars = pd.date_range("2025-01-01", periods=8, freq="min", tz="UTC")
    events = pd.DataFrame(
        {
            "event_id": ["test_a", "test_b", "next", "later"],
            "t0": bars[[0, 1, 3, 4]],
            "t1": [bars[0], bars[2] + pd.Timedelta(seconds=30), bars[3], bars[4]],
        }
    )
    first = build_purged_kfold_splits(events, bars, 2, embargo_bars=1)[0]
    assert "next" not in first["train_event_ids"]
    assert "later" in first["train_event_ids"]
    assert first["n_embargoed"] == 1


def test_splitter_rejects_ambiguous_inputs():
    bars = pd.date_range("2025-01-01", periods=4, freq="min", tz="UTC")
    duplicate_ids = pd.DataFrame(
        {"event_id": [1, 1], "t0": bars[[0, 2]], "t1": bars[[1, 3]]}
    )
    with pytest.raises(ValueError, match="unique"):
        build_purged_kfold_splits(duplicate_ids, bars, 2)
    with pytest.raises(ValueError, match="embargo"):
        build_purged_kfold_splits(
            duplicate_ids.assign(event_id=[1, 2]), bars, 2, embargo_bars=-1
        )


def test_dsr_penalizes_more_trials():
    returns = np.tile([1.2, -0.7, 0.9, -0.3, 1.0], 30)
    assert compute_dsr(returns, 100)["dsr"] < compute_dsr(returns, 1)["dsr"]


def test_dsr_penalizes_two_trials_and_validates_trial_count():
    returns = np.tile([1.2, -0.7, 0.9, -0.3, 1.0], 30)
    assert compute_dsr(returns, 2)["sr_star"] > 0
    with pytest.raises(ValueError, match="n_trials"):
        compute_dsr(returns, 0)
    with pytest.raises(ValueError, match="finite"):
        compute_dsr([0.1, np.nan, 0.2], 2)


def test_pbo_detects_unstable_selection():
    performance = pd.DataFrame(
        np.column_stack([np.eye(4) * 10, np.ones(4)]),
        columns=["lucky_1", "lucky_2", "lucky_3", "lucky_4", "stable"],
    )
    result = compute_pbo(performance)
    assert result["pbo"] == 1.0
    assert result["n_partitions"] == 6


def test_pbo_requires_even_finite_subperiod_matrix():
    with pytest.raises(ValueError, match="even"):
        compute_pbo(pd.DataFrame(np.ones((3, 2))))
    invalid = pd.DataFrame([[1.0, np.nan], [2.0, 1.0], [3.0, 2.0], [4.0, 3.0]])
    with pytest.raises(ValueError, match="finite"):
        compute_pbo(invalid)
    with pytest.raises(ValueError, match="partitions"):
        compute_pbo(pd.DataFrame(np.ones((8, 2))), max_partitions=10)
