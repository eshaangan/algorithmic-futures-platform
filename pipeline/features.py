import pandas as pd


def build_features(bars: pd.DataFrame) -> pd.DataFrame:
    required = {"high", "low", "close", "volume"}
    missing = required.difference(bars.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    out = pd.DataFrame(index=bars.index)
    out["return_1"] = bars["close"].pct_change()
    out["range"] = (bars["high"] - bars["low"]) / bars["close"].shift(1)
    out["volume_z20"] = (bars["volume"] - bars["volume"].rolling(20).mean()) / bars["volume"].rolling(20).std()
    out["momentum_5"] = bars["close"].pct_change(5)
    return out
