from dataclasses import dataclass


@dataclass
class Security:
    security_id: str
    symbol: str
    name: str
    market: str
