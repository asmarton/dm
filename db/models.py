from sqlalchemy import Column, Integer, String, DateTime, Double, Boolean
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
    max_assets = Column(Integer, nullable=False, default=1, server_default='1')
    exclude_prev_month = Column(Boolean, nullable=False, default=False, server_default='0')
    rebalance_threshold = Column(Double, nullable=False, default=0.0, server_default='0.0')
    cagr = Column(Double, nullable=False, default=0.0, server_default='0.0')
    drawdown = Column(Double, nullable=False, default=0.0, server_default='0.0')
    cagr_benchmark = Column(Double, nullable=False, default=0.0, server_default='0.0')
    drawdown_benchmark = Column(Double, nullable=False, default=0.0, server_default='0.0')
    benchmark = Column(String, nullable=False, default='SPY', server_default='SPY')
    trades = Column(Integer, nullable=False, default=0, server_default='0')
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
    benchmark = Column(String, nullable=False)
    cagr = Column(Double, nullable=False, default=0.0, server_default='0.0')
    drawdown = Column(Double, nullable=False, default=0.0, server_default='0.0')
    cagr_benchmark = Column(Double, nullable=False, default=0.0, server_default='0.0')
    drawdown_benchmark = Column(Double, nullable=False, default=0.0, server_default='0.0')
    user = Column(String, nullable=False, server_default='*')
    created_at = Column(DateTime, nullable=False, server_default=func.now())
