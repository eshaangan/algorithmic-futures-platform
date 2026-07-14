from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from validation.diagnostics import compute_dsr, compute_pbo
from strategies.rule_based.orb import OpeningRangeBreakout

OUT = ROOT / "docs" / "figures"
SEED = 42
CAPTION = "Illustrative methodology demo on synthetic data."


def save(fig: plt.Figure, name: str) -> None:
    fig.text(0.5, 0.01, CAPTION, ha="center", fontsize=8, color="#52606d")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def purged_cv() -> None:
    fig, ax = plt.subplots(figsize=(9, 3))
    colors = ["#2563eb", "#ef4444", "#94a3b8", "#f59e0b"]
    labels = ["Train events", "Test fold", "Purged overlap", "Embargo"]
    spans = [(0, 3.8), (4, 6), (3.4, 4), (6, 6.8)]
    for y, ((start, end), color, label) in enumerate(zip(spans, colors, labels)):
        ax.broken_barh(
            [(start, end - start)],
            (y - 0.3, 0.6),
            facecolors=color,
            label=label,
        )
    ax.set(
        xlim=(0, 8),
        ylim=(-0.7, 3.7),
        xlabel="Time",
        yticks=[],
        title="Purged CV prevents interval leakage",
    )
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(.5, -.25))
    save(fig, "purged_cv_schematic.png")


def distributions(rng: np.random.Generator) -> None:
    perf = pd.DataFrame(rng.normal(0.25, 0.7, size=(8, 24)))
    pbo = compute_pbo(perf)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(pbo["lambda_values"], bins=10, color="#7c3aed")
    ax.axvline(0, color="#ef4444", ls="--")
    ax.set(
        title=f"PBO rank logits (PBO={pbo['pbo']:.2f})",
        xlabel="Logit of OOS percentile rank",
    )
    save(fig, "pbo_distribution.png")
    dsrs = [compute_dsr(rng.normal(mu, 1, 120), 25)["dsr"] for mu in np.linspace(-.05, .2, 120)]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(dsrs, bins=16, color="#0891b2")
    ax.set(title="Deflated Sharpe ratio distribution", xlabel="DSR")
    save(fig, "dsr_distribution.png")


def cost_curve(rng: np.random.Generator) -> None:
    returns = rng.normal(.00015, .0025, 500)
    position = np.sign(
        pd.Series(returns).rolling(8).mean().shift(1).fillna(0).to_numpy()
    )
    gross = position * returns
    turnover = np.abs(np.diff(position, prepend=0))
    costs = np.linspace(0, 5, 30)
    net_returns = [gross - turnover * cost / 10_000 for cost in costs]
    sharpes = [
        net.mean() / (net.std() + 1e-12) * np.sqrt(252) for net in net_returns
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(costs, sharpes, color="#2563eb", lw=2)
    ax.axhline(0, color="#64748b", lw=1)
    ax.set(
        title="Turnover-aware execution-cost sensitivity",
        xlabel="One-way cost (bps per unit turnover)",
        ylabel="Annualized Sharpe",
    )
    save(fig, "cost_curve.png")


def monte_carlo(rng: np.random.Generator) -> None:
    daily_pnl = np.maximum(rng.normal(45, 280, size=(4000, 20)), -900)
    paths = 50_000 + np.cumsum(daily_pnl, axis=1)
    peaks = np.maximum.accumulate(
        np.column_stack([np.full(len(paths), 50_000), paths]), axis=1
    )[:, 1:]
    trailing_breach = np.any(paths <= peaks - 2_000, axis=1)
    terminal_pnl = paths[:, -1] - 50_000
    passed = (terminal_pnl >= 3_000) & ~trailing_breach
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(
        terminal_pnl[~passed],
        bins=35,
        color="#94a3b8",
        alpha=.8,
        label="Did not pass",
    )
    ax.hist(
        terminal_pnl[passed],
        bins=20,
        color="#16a34a",
        alpha=.8,
        label="Passed",
    )
    ax.axvline(3_000, color="#2563eb", ls="--", label="$3,000 target")
    ax.set(
        title=(
            f"Synthetic evaluation paths: pass={passed.mean():.1%}, "
            f"trailing breach={trailing_breach.mean():.1%}"
        ),
        xlabel="Terminal P&L ($)",
    )
    ax.legend()
    save(fig, "combine_mc_histogram.png")


def orb_demo() -> None:
    index = pd.date_range(
        "2025-01-02 09:30", periods=8, freq="5min", tz="America/New_York"
    )
    bars = pd.DataFrame(
        {
            "high": [100, 101, 101, 101, 101, 101, 101, 102],
            "low": [99, 99, 99, 99, 99, 99, 99, 100],
            "close": [100, 100, 100, 100, 100, 100, 101, 102],
        },
        index=index,
    )
    signal = OpeningRangeBreakout(30).evaluate(bars)
    cutoff = index[0] + pd.Timedelta(minutes=30)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(index, bars["close"], marker="o", color="#2563eb", label="Synthetic close")
    ax.fill_between(
        index,
        signal.metadata["range_low"],
        signal.metadata["range_high"],
        where=index <= cutoff,
        color="#f59e0b",
        alpha=.25,
        label="Opening range",
    )
    ax.axvline(cutoff, color="#64748b", ls="--", label="Range complete")
    ax.scatter(
        index[-1],
        bars["close"].iloc[-1],
        color="#16a34a",
        s=70,
        zorder=3,
        label="First confirmed breakout",
    )
    ax.set(
        title="Synthetic ORB: range fixed before breakout",
        xlabel="New York time",
        ylabel="Price",
    )
    ax.legend()
    save(fig, "orb_synthetic_demo.png")


def equity(rng: np.random.Generator) -> None:
    returns = rng.normal(.0003, .006, 250)
    curve = 50_000 * np.cumprod(1 + returns)
    drawdown = curve / np.maximum.accumulate(curve) - 1
    fig, (equity_axis, drawdown_axis) = plt.subplots(
        2, 1, figsize=(8, 5), sharex=True
    )
    equity_axis.plot(curve, color="#2563eb")
    equity_axis.set_ylabel("Equity ($)")
    drawdown_axis.fill_between(
        np.arange(len(drawdown)), drawdown, 0, color="#ef4444", alpha=.6
    )
    drawdown_axis.set(ylabel="Drawdown", xlabel="Synthetic session")
    save(fig, "equity_drawdown_demo.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    purged_cv()
    distributions(rng)
    cost_curve(rng)
    monte_carlo(rng)
    orb_demo()
    equity(rng)
    print(f"generated 7 deterministic figures in {OUT}")


if __name__ == "__main__":
    main()
