from sqlalchemy import Column, Integer, String, DateTime, Double
from sqlalchemy.orm import declarative_base
from sqlalchemy import func



Base = declarative_base()


class Job(Base):
    __tablename__ = 'jobs'

    id = Column(Integer, primary_key=True)
    start_year = Column(Integer, nullable=False)
    start_month = Column(Integer, nullable=False)
    end_year = Column(Integer, nullable=False)
    end_month = Column(Integer, nullable=False)
    tickers = Column(String, nullable=False)
    safe_asset = Column(String, nullable=False)
    initial_investment = Column(Double, nullable=False)
    rebalance_period = Column(Integer, nullable=False)
    lookback_period = Column(Integer, nullable=False)
    switching_cost = Column(Double, nullable=False)
    single_absolute_momentum = Column(String, nullable=True)
    user = Column(String, nullable=False, server_default='*')
    created_at = Column(DateTime, nullable=False, server_default=func.now())


class IndustryTrendsJob(Base):
    __tablename__ = 'industry_trends_jobs'

    id = Column(Integer, primary_key=True)
    initial_balance = Column(Double, nullable=False)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    tickers = Column(String, nullable=False)
    up_period = Column(Integer, nullable=False)
    down_period = Column(Integer, nullable=False)
    vol_window = Column(Integer, nullable=False)
    max_leverage = Column(Double, nullable=False)
    target_volatility = Column(Double, nullable=False)
    trade_cost_per_share = Column(Double, nullable=False)
    trade_cost_min = Column(Double, nullable=False)
    rebalance_threshold = Column(Double, nullable=False)
    user = Column(String, nullable=False, server_default='*')
    created_at = Column(DateTime, nullable=False, server_default=func.now())