from typing import List
from ..entities.anomaly import Anomaly


class AnomalyRepository:
    def list_recent(self, limit: int = 100) -> List[Anomaly]:
        raise NotImplementedError()

    def save(self, anomaly: Anomaly) -> None:
        raise NotImplementedError()
