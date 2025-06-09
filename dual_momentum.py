from datetime import datetime, timedelta

import pandas as pd
import pandas_datareader as pdr
import requests_cache
import yfinance as yf
from dateutil.relativedelta import relativedelta

from db.schemas import JobBase


def dual_momentum(job: JobBase):
    tickers = [t.strip() for t in job.tickers.split(",") if t.strip()]

    start = datetime.strptime(f'{job.start_year}-{job.start_month:02}-01', '%Y-%m-%d') - relativedelta(
        months=job.lookback_period)
    start = start.strftime('%Y-%m-%d')
    end = f'{job.end_year}-{job.end_month:02}-01'

    if job.lookback_period < job.rebalance_period:
        raise RuntimeError('Lookback period cannot be less than rebalancing period')

    all_assets = tickers + [job.safe_asset]
    prices = yf.download(tickers=all_assets, start=start, end=end, auto_adjust=True)['Close']
    monthly_prices = prices.resample('ME').last()
    monthly_returns = monthly_prices.pct_change().fillna(0)

    pdr_expire_after = timedelta(days=1)
    pdr_session = requests_cache.CachedSession(cache_name='fred_cache', backend='sqlite', expire_after=pdr_expire_after)

    t_bills = pdr.data.DataReader('TB3MS', 'fred', start='1934-01-01', session=pdr_session)
    t_bills = t_bills.rename(columns={'TB3MS': 'TBillRate'}) / 100
    t_bills = t_bills.reindex(monthly_returns.index, method='ffill')

    portfolio = pd.DataFrame(index=monthly_prices.index,
                             columns=['Selected Asset', 'DM Return', 'Switching Cost', 'DM Balance'])
    selected_asset = tickers[0]
    trades = pd.DataFrame(columns=['Trade Date', 'Sold', 'Bought'])
    month_start = monthly_prices.index[job.lookback_period].replace(day=1)
    trades.loc[len(trades)] = [month_start, None, selected_asset]

    balance = job.initial_investment
    for i in range(job.lookback_period, len(monthly_prices)):
        date = monthly_prices.index[i]
        month_start = date.replace(day=1)

        switched = False
        if (i - job.lookback_period) % job.rebalance_period == 0:
            momentums = (monthly_prices.iloc[i] / monthly_prices.iloc[i - job.lookback_period]) - 1
            # Relative momentum
            best_asset = momentums.idxmax()

            # Absolute momentum
            if momentums[best_asset] > t_bills.iloc[i]['TBillRate']:
                if selected_asset != best_asset:
                    trades.loc[len(trades)] = [month_start, selected_asset, best_asset]
                    switched = True
                selected_asset = best_asset
            else:
                if selected_asset != job.safe_asset:
                    trades.loc[len(trades)] = [month_start, selected_asset, job.safe_asset]
                    switched = True
                selected_asset = job.safe_asset

        if pd.notna(monthly_prices.at[date, selected_asset]):
            asset_return = monthly_prices.at[date, selected_asset] / monthly_prices.iloc[i - 1][selected_asset] - 1
        else:
            asset_return = 0

        sc = 0 if not switched else job.switching_cost / 100
        balance = round(balance * (1 - sc) * (1 + asset_return), 4)
        portfolio.iloc[i] = [selected_asset, asset_return, sc, balance]

    # Cut portfolio to real start date.
    portfolio = portfolio[job.lookback_period:]

    for ticker in all_assets:
        portfolio[f'{ticker} Return'] = monthly_returns[ticker]
        portfolio[f'{ticker} Balance'] = job.initial_investment * (1 + portfolio[f'{ticker} Return']).cumprod()

    portfolio[f'Treasuries Return'] = t_bills['TBillRate']

    return portfolio, trades
