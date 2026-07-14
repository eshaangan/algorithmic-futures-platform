from dataclasses import dataclass
import hashlib
import json
import numpy as np
import pandas as pd
from validation.diagnostics import compute_dsr


@dataclass(frozen=True)
class PreRegistration:
    strategy_id: str
    holdout: dict
    n_trials: int
    gate: dict
    parameters: dict

    def content_hash(self) -> str:
        rule = {"holdout": self.holdout, "n_trials": self.n_trials, "gate": self.gate, "parameters": self.parameters}
        payload = json.dumps(rule, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def evaluate_gate(trades: pd.DataFrame, preregistration: PreRegistration) -> dict:
    pnl = trades["pnl"].to_numpy(float) if len(trades) else np.array([])
    months = pd.to_datetime(trades["exit_time"]).dt.to_period("M") if len(trades) else pd.Series(dtype=str)
    monthly = trades.assign(month=months).groupby("month")["pnl"].sum() if len(trades) else pd.Series(dtype=float)
    dsr = compute_dsr(pnl, preregistration.n_trials)
    gate = preregistration.gate
    failures = []
    if len(pnl) < gate["min_trades"]:
        failures.append("insufficient trades")
    if len(monthly) < gate["min_months"]:
        failures.append("insufficient months")
    if len(monthly) and float((monthly > 0).mean()) < gate["min_positive_month_fraction"]:
        failures.append("positive month fraction")
    if dsr.get("dsr") is None or dsr["dsr"] < gate["min_dsr"]:
        failures.append("deflated Sharpe")
    return {"verdict": "GO" if not failures else "NO-GO", "failures": failures, "dsr": dsr, "config_hash": preregistration.content_hash()}
