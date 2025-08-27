from src.cerverus.application.use_cases.detect_anomalies import DetectAnomaliesUseCase
from tests.helpers.mock_services import MockTradeRepo
from src.cerverus.domain.services.fraud_detector import FraudDetector
from datetime import datetime
from src.cerverus.domain.entities.trade import Trade


def test_detect_anomalies_use_case():
    trades = [
        Trade("t1", "s1", datetime.utcnow(), 1, 10),
        Trade("t2", "s2", datetime.utcnow(), 20000, 60),
    ]
    repo = MockTradeRepo(trades)
    detector = FraudDetector(threshold=0.9)
    uc = DetectAnomaliesUseCase(trade_repo=repo, detector=detector)

    anomalies = uc.execute()
    assert len(anomalies) >= 1
