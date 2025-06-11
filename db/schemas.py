from datetime import datetime

from pydantic import BaseModel, ConfigDict

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
    single_absolute_momentum: str | None = None


class JobCreate(JobBase):
    pass


class Job(JobBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
