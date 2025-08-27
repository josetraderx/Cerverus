from typing import List

from ..entities.trade import Trade


class FraudDetector:
    def __init__(self, threshold: float = 0.9):
        self.threshold = threshold

    def score(self, trade: Trade) -> float:
        # placeholder: simple heuristic
        value = trade.value()
        if value > 1_000_000:
            return 0.99
        return min(0.1 + value / 1_000_000, 0.99)

    def detect(self, trades: List[Trade]):
        return [t for t in trades if self.score(t) >= self.threshold]
