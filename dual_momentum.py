from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pandas_datareader as pdr
import requests_cache
import yfinance as yf
from dateutil.relativedelta import relativedelta

from db.schemas import JobBase


def get_tbills(job: JobBase, index: pd.DataFrame) -> pd.DataFrame:
    pdr_expire_after = timedelta(days=1)
    pdr_session = requests_cache.CachedSession(cache_name='fred_cache', backend='sqlite', expire_after=pdr_expire_after)

    t_bills = pdr.data.DataReader('TB3MS', 'fred', start='1934-01-01', session=pdr_session)
    t_bills = t_bills.rename(columns={'TB3MS': 'TBillRate'}) / 100
    t_bills = t_bills.reindex(index, method='ffill')
    t_bills['Lookback Return'] = (
        t_bills['TBillRate'].rolling(window=job.lookback_period).mean() / 12 * job.lookback_period
    )

    return t_bills


tbill_etf = 'BIL'


def rebalance(idx: int, market_closes: pd.DataFrame, rf_closes: pd.DataFrame, job: JobBase) -> str:
    market_returns = (
        market_closes.iloc[idx] / market_closes.iloc[idx - job.lookback_period]
    ) - 1
    rf_returns = (rf_closes.iloc[idx] / rf_closes.iloc[idx - job.lookback_period]) - 1
    market_returns = market_returns.fillna(0)
    rf_returns = rf_returns.fillna(0)

    # Relative momentum
    best_asset = market_returns.idxmax()

    am_asset = job.single_absolute_momentum or best_asset
    # Absolute momentum
    if market_returns[am_asset] > rf_returns[tbill_etf]:
        selected_asset = best_asset
    else:
        selected_asset = job.safe_asset
    return selected_asset


def dual_momentum(job: JobBase) -> tuple[pd.DataFrame, pd.DataFrame]:
    tickers = [t.strip() for t in job.tickers.split(',') if t.strip()]

    start = datetime.strptime(f'{job.start_year}-{job.start_month:02}-01', '%Y-%m-%d') - relativedelta(
        months=job.lookback_period + 1
    )
    start = start.strftime('%Y-%m-%d')
    end = f'{job.end_year}-{job.end_month:02}-01'

    if job.lookback_period < job.rebalance_period:
        raise RuntimeError('Lookback period cannot be less than rebalancing period')

    all_assets = tickers + [job.safe_asset, tbill_etf]
    prices = yf.download(tickers=all_assets, start=start, end=end, auto_adjust=True)['Close']
    monthly_closes = prices.groupby(pd.Grouper(freq='ME')).nth(-1)

    monthly_closes_market = monthly_closes.drop(columns=[job.safe_asset, tbill_etf])
    monthly_closes_bil = monthly_closes[[tbill_etf]]

    trades = pd.DataFrame(columns=['Trade Date', 'Sold', 'Bought'])

    # Entry
    selected_asset = rebalance(job.lookback_period, monthly_closes_market, monthly_closes_bil, job)
    trades.loc[len(trades)] = [monthly_closes.index[job.lookback_period], '', selected_asset]

    monthly_closes = monthly_closes.iloc[1:]
    monthly_closes_market = monthly_closes_market.iloc[1:]
    monthly_closes_bil = monthly_closes_bil.iloc[1:]
    monthly_returns = monthly_closes.pct_change().fillna(0)

    balance = job.initial_investment
    switched = False

    portfolio = pd.DataFrame(
        index=monthly_closes.index,
        columns=[
            'Selected Asset',
            'Dual Momentum Return',
            'Switching Cost',
            'Dual Momentum Balance',
        ],
    )

    # Rebalancing is made at the end of the month.
    for i in range(job.lookback_period, len(monthly_closes)):
        date = monthly_closes.index[i]

        asset_return = monthly_closes.at[date, selected_asset] / monthly_closes.iloc[i - 1][selected_asset] - 1
        asset_return = 0 if np.isnan(asset_return) else asset_return
        switching_cost = 0 if not switched else job.switching_cost / 100
        balance = balance * (1 - switching_cost) * (1 + asset_return)

        portfolio.iloc[i] = [selected_asset, asset_return, switching_cost, balance]

        switched = False
        if (i - job.lookback_period) % job.rebalance_period == 0:
            new_asset = rebalance(i, monthly_closes_market, monthly_closes_bil, job)
            if new_asset != selected_asset:
                trades.loc[len(trades)] = [date, selected_asset, new_asset]
                switched = True
            selected_asset = new_asset

    # Cut portfolio to real start date.
    portfolio = portfolio[job.lookback_period :]

    for ticker in all_assets:
        portfolio[f'{ticker} Return'] = monthly_returns[ticker]
        portfolio[f'{ticker} Balance'] = job.initial_investment * (1 + portfolio[f'{ticker} Return']).cumprod()

    portfolio = portfolio.infer_objects()

    trades.set_index('Trade Date', inplace=True)

    return portfolio, trades


def humanize_portfolio(portfolio: pd.DataFrame) -> pd.DataFrame:
    portfolio = portfolio.copy()

    for col in portfolio.columns:
        if col.endswith('Balance'):
            portfolio[col] = portfolio[col].round(3)
        elif col.endswith('Return') or col == 'Switching Cost':
            portfolio[col] = (portfolio[col] * 100).round(2).astype(str) + '%'

    return portfolio
