import pandas as pd
import pytest

from ingestion.microflow import aggregate_ticks, read_ticks_chunked


def test_chunked_ingestion_matches_single_pass(tmp_path):
    base = pd.Timestamp("2025-01-01", tz="UTC").value
    ticks = pd.DataFrame({"ts_recv": [base, base + 20_000_000_000, base + 70_000_000_000, base + 80_000_000_000], "price": [100, 101, 102, 101], "size": [1, 2, 3, 4], "side": ["B", "A", "B", "A"]})
    path = tmp_path / "ticks.csv"
    ticks.to_csv(path, index=False)
    expected = aggregate_ticks(ticks)
    actual = read_ticks_chunked(path, chunksize=2)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)


def test_chunked_ingestion_rejects_out_of_order_input(tmp_path):
    ticks = pd.DataFrame(
        {
            "ts_recv": [2, 1],
            "price": [100, 101],
            "size": [1, 1],
            "side": ["B", "A"],
        }
    )
    path = tmp_path / "ticks.csv"
    ticks.to_csv(path, index=False)
    with pytest.raises(ValueError, match="ordered"):
        read_ticks_chunked(path, chunksize=2)


def test_aggregation_validates_trade_domain():
    ticks = pd.DataFrame(
        {"ts_recv": [1], "price": [100], "size": [1], "side": ["unknown"]}
    )
    with pytest.raises(ValueError, match="trade sides"):
        aggregate_ticks(ticks)
