from typing import List

from ...domain.entities.trade import Trade


class PostgresDB:
    def __init__(self, dsn: str):
        self.dsn = dsn

    def insert_trade(self, trade: Trade) -> None:
        # placeholder: implement actual DB insert
        print(f"Inserting trade {trade.trade_id} into Postgres")

    def list_trades(self, limit: int = 100) -> List[Trade]:
        raise NotImplementedError()
