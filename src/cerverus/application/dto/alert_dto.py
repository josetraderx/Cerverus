from dataclasses import dataclass
from datetime import datetime


@dataclass
class AlertDTO:
    alert_id: str
    anomaly_id: str
    created_at: datetime
    message: str
