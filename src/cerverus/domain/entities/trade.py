from dataclasses import dataclass
from datetime import datetime


@dataclass
class Trade:
    trade_id: str
    security_id: str
    timestamp: datetime
    quantity: float
    price: float

    def value(self) -> float:
        return self.quantity * self.price
