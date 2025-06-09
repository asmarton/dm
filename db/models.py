from sqlalchemy import Column, Integer, String, DateTime, Double
from sqlalchemy.orm import declarative_base


Base = declarative_base()


class Job(Base):
    __tablename__ = "jobs"

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
    created_at = Column(DateTime, nullable=False, server_default='current_timestamp')
