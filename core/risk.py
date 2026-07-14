from dataclasses import dataclass
from datetime import date, datetime, timedelta
from math import isfinite
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class PropRiskConfig:
    starting_balance: float = 50_000.0
    max_daily_loss: float = 1_000.0
    trailing_drawdown: float = 2_000.0
    session_timezone: str = "America/Chicago"
    session_rollover_hour: int = 17

    def __post_init__(self) -> None:
        if self.starting_balance <= 0:
            raise ValueError("starting_balance must be positive")
        if self.max_daily_loss <= 0 or self.trailing_drawdown <= 0:
            raise ValueError("loss limits must be positive")
        if not 0 <= self.session_rollover_hour <= 23:
            raise ValueError("session_rollover_hour must be between 0 and 23")
        ZoneInfo(self.session_timezone)


class PropRiskEngine:
    def __init__(self, config: PropRiskConfig):
        self.config = config
        self.equity = config.starting_balance
        self.peak_equity = config.starting_balance
        self.daily_pnl = 0.0
        self.session_day: date | None = None
        self.daily_locked = False
        self.permanently_locked = False

    @property
    def can_trade(self) -> bool:
        return not (self.daily_locked or self.permanently_locked)

    @property
    def trailing_floor(self) -> float:
        return self.peak_equity - self.config.trailing_drawdown

    def mark(self, timestamp: datetime, pnl: float) -> None:
        """Apply incremental marked P&L and update lock state."""
        if not isfinite(pnl):
            raise ValueError("pnl must be finite")
        timezone = ZoneInfo(self.config.session_timezone)
        if timestamp.tzinfo is None:
            local_timestamp = timestamp.replace(tzinfo=timezone)
        else:
            local_timestamp = timestamp.astimezone(timezone)
        day = (
            local_timestamp - timedelta(hours=self.config.session_rollover_hour)
        ).date()
        if day != self.session_day:
            self.session_day = day
            self.daily_pnl = 0.0
            self.daily_locked = False
        self.daily_pnl += float(pnl)
        self.equity += float(pnl)
        self.peak_equity = max(self.peak_equity, self.equity)
        if self.daily_pnl <= -self.config.max_daily_loss:
            self.daily_locked = True
        if self.equity <= self.trailing_floor:
            self.permanently_locked = True

    def projected_trade_allowed(self, stop_loss_usd: float) -> bool:
        if not isfinite(stop_loss_usd) or stop_loss_usd < 0:
            raise ValueError("stop_loss_usd must be finite and non-negative")
        preserves_daily_limit = (
            self.daily_pnl - stop_loss_usd > -self.config.max_daily_loss
        )
        preserves_trailing_floor = self.equity - stop_loss_usd > self.trailing_floor
        return self.can_trade and preserves_daily_limit and preserves_trailing_floor
