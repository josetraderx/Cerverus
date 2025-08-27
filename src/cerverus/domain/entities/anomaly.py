from dataclasses import dataclass
from datetime import datetime


@dataclass
class Anomaly:
    anomaly_id: str
    trade_id: str
    detected_at: datetime
    score: float
    reason: str
