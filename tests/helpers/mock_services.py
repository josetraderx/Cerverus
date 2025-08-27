class MockTradeRepo:
    def __init__(self, trades):
        self._trades = trades

    def list_recent(self, limit: int = 100):
        return self._trades

class MockAnomalyRepo:
    def __init__(self):
        self.saved = []

    def save(self, anomaly):
        self.saved.append(anomaly)

    def list_recent(self, limit: int = 100):
        return self.saved
