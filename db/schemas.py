from decimal import Decimal

from pydantic import BaseModel

class JobBase(BaseModel):
    start_year: int
    start_month: int
    end_year: int
    end_month: int
    tickers: str
    safe_asset: str
    initial_investment: float
    rebalance_period: int
    lookback_period: int
    switching_cost: float


class JobCreate(JobBase):
    pass


class Job(JobBase):
    id: int

    class Config:
        orm_mode = True