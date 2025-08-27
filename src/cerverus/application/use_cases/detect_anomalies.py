from ...domain.repositories.trade_repository import TradeRepository
from ...domain.services.fraud_detector import FraudDetector


class DetectAnomaliesUseCase:
    def __init__(self, trade_repo: TradeRepository, detector: FraudDetector):
        self.trade_repo = trade_repo
        self.detector = detector

    def execute(self):
        trades = self.trade_repo.list_recent()
        anomalies = self.detector.detect(trades)
        return anomalies
