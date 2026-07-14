from itertools import combinations
from math import comb, e
from statistics import NormalDist

import numpy as np
import pandas as pd


def compute_dsr(returns, n_trials: int, target_sharpe: float = 0.0) -> dict:
    """Estimate the probability that Sharpe exceeds a multiple-testing benchmark.

    Returns are expected to be equally spaced and expressed in consistent units.
    The reported Sharpe is per observation; no annualization is inferred.
    """
    if isinstance(n_trials, bool) or int(n_trials) != n_trials or n_trials < 1:
        raise ValueError("n_trials must be a positive integer")
    if not np.isfinite(target_sharpe):
        raise ValueError("target_sharpe must be finite")
    values = np.asarray(list(returns), dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("returns must contain only finite values")
    if values.size < 2:
        return {"dsr": None, "reason": "insufficient_returns"}
    mean, std = float(values.mean()), float(values.std(ddof=1))
    if std == 0:
        return {"dsr": None, "reason": "zero_variance"}
    sharpe = mean / std
    z = (values - mean) / std
    skew, kurtosis = float(np.mean(z**3)), float(np.mean(z**4))
    variance = (1 - skew * sharpe + (kurtosis - 1) * sharpe**2 / 4) / (len(values) - 1)
    if variance <= 0:
        return {"dsr": None, "reason": "nonpositive_variance"}
    sr_std = float(np.sqrt(variance))
    trials = int(n_trials)
    if trials == 1:
        benchmark = target_sharpe
    else:
        normal = NormalDist()
        euler_mascheroni = 0.5772156649015329
        expected_maximum = (
            (1 - euler_mascheroni) * normal.inv_cdf(1 - 1 / trials)
            + euler_mascheroni * normal.inv_cdf(1 - 1 / (trials * e))
        )
        benchmark = target_sharpe + sr_std * expected_maximum
    dsr = float(NormalDist().cdf((sharpe - benchmark) / sr_std))
    return {"dsr": dsr, "sharpe": sharpe, "sr_star": benchmark, "n_trials": trials, "n_obs": len(values)}


def compute_pbo(
    performance: pd.DataFrame,
    *,
    max_partitions: int = 100_000,
) -> dict:
    """Compute CSCV-style probability of backtest overfitting.

    Rows are independent performance paths or time blocks and columns are
    strategy configurations. Each unique half-split selects the best in-sample
    configuration and records its out-of-sample rank.
    """
    if len(performance) % 2:
        raise ValueError("performance must contain an even number of subperiods")
    if performance.shape[0] < 4 or performance.shape[1] < 2:
        return {"pbo": None, "reason": "insufficient_matrix"}
    n_partitions = comb(len(performance), len(performance) // 2)
    if n_partitions > max_partitions:
        raise ValueError(
            f"CSCV requires {n_partitions:,} partitions; reduce subperiods or "
            "increase max_partitions explicitly"
        )
    try:
        finite = np.isfinite(performance.to_numpy(dtype=float)).all()
    except (TypeError, ValueError) as exc:
        raise ValueError("performance must contain finite numeric values") from exc
    if not finite:
        raise ValueError("performance must contain only finite values")

    n_rows = len(performance)
    train_size = n_rows // 2
    row_positions = range(n_rows)
    ranks: list[float] = []
    logits: list[float] = []
    for train_positions in combinations(row_positions, train_size):
        test_positions = sorted(set(row_positions).difference(train_positions))
        best = performance.iloc[list(train_positions)].mean().idxmax()
        test_scores = performance.iloc[test_positions].mean()
        ordinal_rank = float(test_scores.rank(method="average")[best])
        rank = ordinal_rank / (len(test_scores) + 1)
        ranks.append(rank)
        logits.append(float(np.log(rank / (1 - rank))))

    return {
        "pbo": float(np.mean(np.asarray(logits) <= 0)),
        "lambda_values": logits,
        "oos_percentile_ranks": ranks,
        "n_splits": len(ranks),
        "n_partitions": len(ranks),
    }
