from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


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
    max_assets: int
    exclude_prev_month: bool
    rebalance_threshold: float
    user: str = Field(min_length=1)


class JobCreate(JobBase):
    pass


class Job(JobBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime


class IndustryTrendsJobBase(BaseModel):
    initial_balance: float = 100_000
    start_date: datetime
    end_date: datetime
    tickers: list[str]
    up_period: int = 20
    down_period: int = 40
    vol_window: int = 14
    max_leverage: float = 200
    target_volatility: float = 1.5
    trade_cost_per_share: float = 0.0035
    trade_cost_min: float = 0.35
    rebalance_threshold: float = 10
    benchmark: str
    user: str = Field(min_length=1)


class IndustryTrendsJobCreate(IndustryTrendsJobBase):
    pass


class IndustryTrendsJob(IndustryTrendsJobBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime


class IndustryTrendsJobForm(IndustryTrendsJobBase):
    tickers: str
