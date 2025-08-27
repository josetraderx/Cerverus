from typing import List

from ...domain.entities.anomaly import Anomaly


class ElasticSearchStore:
    def __init__(self, host: str):
        self.host = host

    def index_anomaly(self, anomaly: Anomaly) -> None:
        print(f"Indexing anomaly {anomaly.anomaly_id} into ElasticSearch")

    def search_anomalies(self, query: dict) -> List[Anomaly]:
        raise NotImplementedError()
