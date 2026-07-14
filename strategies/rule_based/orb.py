import pandas as pd

from .engine import RuleSignal


class OpeningRangeBreakout:
    def __init__(
        self,
        opening_minutes: int = 30,
        timezone: str = "America/New_York",
        session_open: str = "09:30",
        session_close: str = "16:00",
    ):
        if opening_minutes <= 0:
            raise ValueError("opening_minutes must be positive")
        self.opening_minutes = opening_minutes
        self.timezone = timezone
        self.session_open = session_open
        self.session_close = session_close

    def evaluate(self, bars: pd.DataFrame) -> RuleSignal:
        if bars.empty or not isinstance(bars.index, pd.DatetimeIndex):
            return RuleSignal(0, 0.0, "missing bars")
        missing = {"high", "low", "close"}.difference(bars.columns)
        if missing:
            raise ValueError(f"missing columns: {sorted(missing)}")
        if not bars.index.is_monotonic_increasing or not bars.index.is_unique:
            raise ValueError("bar index must be sorted and unique")
        local = bars.copy()
        local.index = (
            local.index.tz_convert(self.timezone)
            if local.index.tz is not None
            else local.index.tz_localize(self.timezone)
        )
        day = local.index[-1].date()
        session_start = pd.Timestamp(
            f"{day.isoformat()} {self.session_open}", tz=self.timezone
        )
        session_end = pd.Timestamp(
            f"{day.isoformat()} {self.session_close}", tz=self.timezone
        )
        if not session_start <= local.index[-1] <= session_end:
            return RuleSignal(0, 0.0, "outside regular session")
        session = local[
            (local.index >= session_start)
            & (local.index <= session_end)
            & (local.index.date == day)
        ]
        if session.empty:
            return RuleSignal(0, 0.0, "empty session")
        cutoff = session_start + pd.Timedelta(minutes=self.opening_minutes)
        opening = session[session.index < cutoff]
        if len(opening) < 2 or session.index[-1] < cutoff:
            return RuleSignal(0, 0.0, "opening range incomplete")
        high, low = float(opening["high"].max()), float(opening["low"].min())
        current = float(session["close"].iloc[-1])
        previous = float(session["close"].iloc[-2])
        if current > high and previous <= high:
            return RuleSignal(
                1,
                min((current - high) / max(high - low, 1e-12), 1.0),
                "first close above opening range",
                {"range_high": high, "range_low": low},
            )
        if current < low and previous >= low:
            return RuleSignal(
                -1,
                min((low - current) / max(high - low, 1e-12), 1.0),
                "first close below opening range",
                {"range_high": high, "range_low": low},
            )
        return RuleSignal(0, 0.0, "inside range or continuation")
