from ..entities.trade import Trade


class RiskCalculator:
    def exposure(self, trades: list[Trade]) -> float:
        return sum(t.value() for t in trades)
