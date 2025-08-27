from ...domain.repositories.anomaly_repository import AnomalyRepository


class GenerateAlertsUseCase:
    def __init__(self, anomaly_repo: AnomalyRepository):
        self.anomaly_repo = anomaly_repo

    def execute(self):
        anomalies = self.anomaly_repo.list_recent()
        # placeholder: convert anomalies to alerts and push
        return anomalies
