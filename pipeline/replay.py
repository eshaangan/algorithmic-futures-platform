import pandas as pd


def replay_decisions(features: pd.DataFrame, decision_fn) -> pd.Series:
    """Replay a decision function with an expanding, point-in-time history."""
    decisions = []
    for i in range(len(features)):
        history = features.iloc[: i + 1]
        decisions.append(decision_fn(history))
    return pd.Series(decisions, index=features.index, name="decision")
