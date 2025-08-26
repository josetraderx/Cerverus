from pydantic import BaseModel

class FinancialRecord(BaseModel):
    timestamp: str
    symbol: str
    price: float
