import pathlib
from datetime import datetime

import pandas as pd
from dateutil.relativedelta import relativedelta
from pydantic import BaseModel
import yfinance as yf

from db.schemas import JobBase


tbills_symbol = 'DGS3MO'


class TickerInfo(BaseModel):
    symbol: str
    start_date: datetime
    fallbacks: list['TickerInfo'] | None = None


def get_tbills() -> pd.DataFrame:
    dgs3mo = pd.read_csv(f'data/{tbills_symbol}.csv').dropna().set_index('Date')
    dgs3mo.index = pd.to_datetime(dgs3mo.index)
    tbill_yield = dgs3mo.groupby(pd.Grouper(freq='ME')).nth(-1)
    tbill_monthly_return = (1 + tbill_yield / 100) ** (1 / 12) - 1
    return tbill_monthly_return


def tickers_to_list(tickers: str) -> list[str]:
    return [t.strip() for t in tickers.split(',')]


def clear_tickers_list(tickers: str) -> str:
    return ','.join(tickers_to_list(tickers))


fallbacks = {
    'SPY': ['^SPX'],
    'VEU': ['MSCI ACWI ex US', 'MSCI World ex US'],
    'AGG': ['VBMFX'],
    'QQQ': ['^IXIC'],
}


def compute_monthly_returns(ticker: str) -> tuple[pd.DataFrame, TickerInfo]:
    if pathlib.Path(f'data/{ticker}.csv').exists():
        prices = pd.read_csv(f'data/{ticker}.csv', thousands=',').dropna().set_index('Date')
        prices.index = pd.to_datetime(prices.index)
    else:
        prices = yf.download(tickers=[ticker], period='max', auto_adjust=True)['Close']
        prices.to_csv(f'data/{ticker}.csv')

    monthly_closes = prices.groupby(pd.Grouper(freq='ME')).nth(-1)
    monthly_returns = monthly_closes.pct_change()
    ticker_info = TickerInfo(symbol=ticker, start_date=monthly_returns.index[0])
    return monthly_returns, ticker_info


def compute_monthly_prices_with_fallback(ticker: str) -> tuple[pd.DataFrame, TickerInfo]:
    monthly_returns, ticker_info = compute_monthly_returns(ticker)
    monthly_returns.dropna(inplace=True)
    if ticker in fallbacks:
        ticker_info.fallbacks = []
        for fallback in fallbacks[ticker]:
            fb_returns, fb_ticker_info = compute_monthly_returns(fallback)
            earliest_index = monthly_returns.index[0]
            kept_fb = fb_returns[fb_returns.index < earliest_index].rename(columns={fallback: ticker})
            monthly_returns = pd.concat([kept_fb, monthly_returns])
            ticker_info.fallbacks.append(fb_ticker_info)
    return monthly_returns, ticker_info


def rebalance(
    idx: int,
    market_lookback_returns: pd.DataFrame,
    tbills_lookback_returns: pd.DataFrame,
    sam_lookback_returns: pd.DataFrame | None,
    job: JobBase,
) -> str:
    market_returns = market_lookback_returns.iloc[idx]
    tbills_returns = tbills_lookback_returns.iloc[idx]
    # Relative momentum
    best_asset = market_returns.idxmax()

    am_asset = job.single_absolute_momentum or best_asset
    if job.single_absolute_momentum and job.single_absolute_momentum not in market_returns:
        market_returns = pd.concat([market_returns, sam_lookback_returns])

    # Absolute momentum
    if market_returns[am_asset] > tbills_returns[tbills_symbol]:
        selected_asset = best_asset
    else:
        selected_asset = job.safe_asset
    return selected_asset


def dual_momentum(job: JobBase) -> tuple[pd.DataFrame, pd.DataFrame, list[TickerInfo]]:
    tickers = tickers_to_list(job.tickers)

    start = datetime.strptime(f'{job.start_year}-{job.start_month:02}-01', '%Y-%m-%d') - relativedelta(
        months=job.lookback_period + 1
    )
    start = start.strftime('%Y-%m-%d')
    end = f'{job.end_year}-{job.end_month:02}-01'

    if job.lookback_period < job.rebalance_period:
        raise RuntimeError('Lookback period cannot be less than rebalancing period')

    monthly_returns = pd.DataFrame(columns=['Date'], index=pd.DatetimeIndex([]))
    monthly_returns.set_index('Date', inplace=True)

    all_tickers = tickers + [job.safe_asset]
    if job.single_absolute_momentum is not None:
        all_tickers = all_tickers + [job.single_absolute_momentum]
    all_tickers = list(set(all_tickers))

    ticker_info: list[TickerInfo] = []

    for ticker in all_tickers:
        ticker_monthly_returns, info = compute_monthly_prices_with_fallback(ticker)
        ticker_info.append(info)
        monthly_returns = pd.merge(monthly_returns, ticker_monthly_returns, on='Date', how='outer')

    tbills_monthly_returns = get_tbills()
    ticker_info.append(TickerInfo(symbol='DGS3MO', start_date=tbills_monthly_returns.index[0]))

    monthly_returns = pd.merge(monthly_returns, tbills_monthly_returns, on='Date', how='outer')
    monthly_returns = monthly_returns.ffill().dropna().groupby(pd.Grouper(freq='ME')).nth(-1)
    monthly_returns = monthly_returns.loc[start:end]

    lookback_returns = (1 + monthly_returns).rolling(job.lookback_period).apply(lambda x: x.prod()) - 1

    market_lookback_returns = lookback_returns[tickers]
    sam_lookback_returns = (
        lookback_returns[job.single_absolute_momentum] if job.single_absolute_momentum is not None else None
    )
    tbills_lookback_returns = lookback_returns[[tbills_symbol]]
    tbills_monthly_returns = monthly_returns[[tbills_symbol]]

    trades = pd.DataFrame(columns=['Trade Date', 'Sold', 'Bought'])

    # Entry
    selected_asset = rebalance(
        job.lookback_period, market_lookback_returns, tbills_lookback_returns, sam_lookback_returns, job
    )

    trades.loc[len(trades)] = [monthly_returns.index[job.lookback_period], '', selected_asset]

    # Remove first month to correctly align with start date. We only needed an extra month to compute first entry.
    monthly_returns = monthly_returns.iloc[1:]
    lookback_returns = lookback_returns.iloc[1:]
    market_lookback_returns = market_lookback_returns.iloc[1:]
    sam_lookback_returns = sam_lookback_returns.iloc[1:] if sam_lookback_returns is not None else None
    tbills_lookback_returns = tbills_lookback_returns.iloc[1:]

    balance = job.initial_investment
    switched = False

    portfolio = pd.DataFrame(
        index=monthly_returns.index,
        columns=[
            'Selected Asset',
            'Dual Momentum Return',
            'Switching Cost',
            'Dual Momentum Balance',
        ],
    )

    # Rebalancing is made at the end of the month.
    for i in range(job.lookback_period, len(monthly_returns.index)):
        date = monthly_returns.index[i]

        asset_return = monthly_returns.loc[date][selected_asset]
        switching_cost = 0 if not switched else job.switching_cost / 100
        balance = balance * (1 - switching_cost) * (1 + asset_return)

        portfolio.iloc[i] = [selected_asset, asset_return, switching_cost, balance]

        switched = False
        if (i - job.lookback_period) % job.rebalance_period == 0:
            new_asset = rebalance(i, market_lookback_returns, tbills_lookback_returns, sam_lookback_returns, job)
            if new_asset != selected_asset:
                trades.loc[len(trades)] = [date, selected_asset, new_asset]
                switched = True
            selected_asset = new_asset

    # Cut portfolio to real start date.
    portfolio = portfolio[job.lookback_period :]

    for ticker in all_tickers:
        portfolio[f'{ticker} Return'] = monthly_returns[ticker]
        portfolio[f'{ticker} Balance'] = job.initial_investment * (1 + portfolio[f'{ticker} Return']).cumprod()
    portfolio[f'{tbills_symbol} Return'] = tbills_monthly_returns[tbills_symbol]
    portfolio[f'{tbills_symbol} Balance'] = (
        job.initial_investment * (1 + portfolio[f'{tbills_symbol} Return']).cumprod()
    )

    portfolio = portfolio.infer_objects()

    trades.set_index('Trade Date', inplace=True)

    return portfolio, trades, ticker_info


def humanize_portfolio(portfolio: pd.DataFrame) -> pd.DataFrame:
    portfolio = portfolio.copy()

    for col in portfolio.columns:
        if col.endswith('Balance'):
            portfolio[col] = portfolio[col].round(3)
        elif col.endswith('Return') or col == 'Switching Cost':
            portfolio[col] = (portfolio[col] * 100).round(2).astype(str) + '%'

    return portfolio
