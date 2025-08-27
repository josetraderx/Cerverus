from typing import List
from ..entities.trade import Trade


class TradeRepository:
    def list_recent(self, limit: int = 100) -> List[Trade]:
        raise NotImplementedError()

    def save(self, trade: Trade) -> None:
        raise NotImplementedError()
