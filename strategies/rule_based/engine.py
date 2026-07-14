from dataclasses import dataclass, field


@dataclass(frozen=True)
class RuleSignal:
    direction: int
    strength: float
    reason: str
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.direction not in {-1, 0, 1}:
            raise ValueError("direction must be -1, 0, or 1")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("strength must be between 0 and 1")
        if self.direction == 0 and self.strength != 0:
            raise ValueError("neutral signals must have zero strength")


class RuleEngine:
    def __init__(self, rules):
        self.rules = list(rules)

    def evaluate(self, bars):
        signals = [rule.evaluate(bars) for rule in self.rules]
        active = [signal for signal in signals if signal.direction]
        if not active:
            return RuleSignal(0, 0.0, "no active rule")
        score = sum(signal.direction * signal.strength for signal in active)
        if score == 0:
            return RuleSignal(0, 0.0, "rule consensus tied")
        direction = 1 if score > 0 else -1
        confidence = abs(score) / len(active)
        return RuleSignal(direction, min(confidence, 1.0), "weighted rule consensus")
