from datetime import datetime
from src.cerverus.domain.entities.trade import Trade
from src.cerverus.domain.services.fraud_detector import FraudDetector


def test_fraud_detector_scores():
    t_small = Trade("t1", "s1", datetime.utcnow(), 1, 10)
    t_large = Trade("t2", "s2", datetime.utcnow(), 20000, 60)
    detector = FraudDetector(threshold=0.9)

    assert detector.score(t_small) < 0.9
    assert detector.score(t_large) >= 0.9
