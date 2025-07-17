import json
import pathlib
import re
from datetime import datetime

import fastapi.encoders
import pandas as pd
from dateutil.relativedelta import relativedelta
from pydantic import BaseModel
from dataclasses import dataclass

import db.models
import db.schemas as schemas
import utils
from db.schemas import JobBase
from utils import ROOT_DIR, tickers_to_list

tbills_symbol = 'DGS3MO'


class TickerInfo(BaseModel):
    symbol: str
    start_date: datetime
    fallbacks: list['TickerInfo'] | None = None


def get_tbills() -> pd.DataFrame:
    path = pathlib.Path(ROOT_DIR) / 'data' / f'{tbills_symbol}.csv'
    dgs3mo = pd.read_csv(path).dropna().set_index('Date')
    dgs3mo.index = pd.to_datetime(dgs3mo.index)
    tbill_yield = dgs3mo.groupby(pd.Grouper(freq='ME')).nth(-1)
    tbill_monthly_return = (1 + tbill_yield / 100) ** (1 / 12) - 1
    return tbill_monthly_return


fallbacks = {
    'SPY': ['^SPX'],
    'VEU': ['MSCI ACWI ex US', 'MSCI World ex US'],
    'AGG': ['VBMFX'],
    'QQQ': ['^IXIC'],
}


def compute_monthly_returns(ticker: str) -> tuple[pd.DataFrame, TickerInfo]:
    prices = utils.get_closing_prices(ticker)
    monthly_closes = prices.groupby(pd.Grouper(freq='ME')).nth(-1)
    monthly_returns = monthly_closes.pct_change()
    ticker_info = TickerInfo(symbol=ticker, start_date=monthly_returns.index[0])
    return monthly_returns, ticker_info


def compute_monthly_returns_with_fallback(ticker: str) -> tuple[pd.DataFrame, TickerInfo]:
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


def compute_monthly_prices(ticker: str) -> tuple[pd.DataFrame, TickerInfo]:
    prices = utils.get_closing_prices(ticker)
    monthly_closes = prices.groupby(pd.Grouper(freq='ME')).nth(-1)
    ticker_info = TickerInfo(symbol=ticker, start_date=monthly_closes.index[0])
    return monthly_closes, ticker_info


def compute_monthly_prices_with_fallback(ticker: str) -> tuple[pd.DataFrame, TickerInfo]:
    monthly_closes, ticker_info = compute_monthly_prices(ticker)
    monthly_closes.dropna(inplace=True)
    if ticker in fallbacks:
        ticker_info.fallbacks = []
        for fallback in fallbacks[ticker]:
            fb_returns, fb_ticker_info = compute_monthly_prices(fallback)
            earliest_index = monthly_closes.index[0]
            kept_fb = fb_returns[fb_returns.index < earliest_index].rename(columns={fallback: ticker})
            monthly_closes = pd.concat([kept_fb, monthly_closes])
            ticker_info.fallbacks.append(fb_ticker_info)
    return monthly_closes, ticker_info


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


@dataclass
class DualMomentumResults:
    portfolio: pd.DataFrame
    trades: pd.DataFrame
    ticker_info: list[TickerInfo]


def dual_momentum(job: JobBase) -> DualMomentumResults:
    tickers = utils.tickers_to_list(job.tickers)

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
        ticker_monthly_returns, info = compute_monthly_returns_with_fallback(ticker)
        ticker_info.append(info)
        monthly_returns = pd.merge(monthly_returns, ticker_monthly_returns, on='Date', how='outer')

    tbills_monthly_returns = get_tbills()
    ticker_info.append(TickerInfo(symbol='DGS3MO', start_date=tbills_monthly_returns.index[0]))

    monthly_returns = pd.merge(monthly_returns, tbills_monthly_returns, on='Date', how='outer')
    monthly_returns = monthly_returns.ffill().dropna().groupby(pd.Grouper(freq='ME')).nth(-1)
    monthly_returns = monthly_returns.loc[start:end]

    if job.exclude_prev_month:
        if job.lookback_period <= 1:
            raise RuntimeError('Lookback period must be at least 2 months when excluding previous month')
        lookback_returns = (1 + monthly_returns).shift(1).rolling(job.lookback_period - 1).apply(lambda x: x.prod()) - 1
    else:
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

    return DualMomentumResults(portfolio, trades, ticker_info)


def rebalance_multi(
    job: JobBase,
    idx: int,
    market_lookback_returns: pd.DataFrame,
    tbills_lookback_returns: pd.DataFrame,
    sam_lookback_returns: pd.Series | None,
    monthly_returns: pd.DataFrame,
    holdings: pd.DataFrame,
    balance: float
) -> tuple[list[str], pd.Series, pd.Series, float, float]:
    market_returns = market_lookback_returns.iloc[idx]
    tbills_returns = tbills_lookback_returns.iloc[idx]
    # Relative momentum
    best_assets = market_returns.nlargest(job.max_assets).index.tolist()

    should_enter_sam = job.single_absolute_momentum is not None and sam_lookback_returns.iloc[idx] > tbills_returns[tbills_symbol]
    eligible_assets = market_returns.loc[best_assets]
    eligible_assets = eligible_assets[eligible_assets > tbills_returns[tbills_symbol]].index.tolist()
    should_enter = should_enter_sam if job.single_absolute_momentum is not None else len(eligible_assets) > 0

    # Absolute momentum
    if should_enter:
        selected_assets = eligible_assets
    else:
        selected_assets = []

    prev_holdings = holdings.iloc[idx]
    new_holdings = pd.Series(index=monthly_returns.columns, data=0.0)
    new_holdings[selected_assets] = 1 / job.max_assets * balance
    new_holdings[job.safe_asset] = (job.max_assets - len(selected_assets)) / job.max_assets * balance
    trades = new_holdings - prev_holdings
    bought = trades.copy()
    bought[bought < 0] = 0
    fees = bought * job.switching_cost / 100
    new_holdings = new_holdings - fees
    balance = new_holdings.sum()
    trades = new_holdings - prev_holdings

    if len(selected_assets) < job.max_assets:
        selected_assets.append(job.safe_asset)
    return selected_assets, new_holdings, trades, balance, fees.sum(),


@dataclass
class DualMomentumMultiResults:
    portfolio: pd.DataFrame
    holdings: pd.DataFrame
    trades: pd.DataFrame
    ticker_info: list[TickerInfo]


def dual_momentum_multi(job: JobBase) -> DualMomentumMultiResults:
    tickers = utils.tickers_to_list(job.tickers)

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
        ticker_monthly_returns, info = compute_monthly_returns_with_fallback(ticker)
        ticker_info.append(info)
        monthly_returns = pd.merge(monthly_returns, ticker_monthly_returns, on='Date', how='outer')

    tbills_monthly_returns = get_tbills()
    ticker_info.append(TickerInfo(symbol='DGS3MO', start_date=tbills_monthly_returns.index[0]))

    monthly_returns = pd.merge(monthly_returns, tbills_monthly_returns, on='Date', how='outer')
    monthly_returns = monthly_returns.ffill().dropna().groupby(pd.Grouper(freq='ME')).nth(-1)
    monthly_returns = monthly_returns.loc[start:end]

    if job.exclude_prev_month:
        if job.lookback_period <= 1:
            raise RuntimeError('Lookback period must be at least 2 months when excluding previous month')
        lookback_returns = (1 + monthly_returns).shift(1).rolling(job.lookback_period - 1).apply(lambda x: x.prod()) - 1
    else:
        lookback_returns = (1 + monthly_returns).rolling(job.lookback_period).apply(lambda x: x.prod()) - 1

    market_lookback_returns = lookback_returns[tickers]
    sam_lookback_returns = (
        lookback_returns[job.single_absolute_momentum] if job.single_absolute_momentum is not None else None
    )
    tbills_lookback_returns = lookback_returns[[tbills_symbol]]
    tbills_monthly_returns = monthly_returns[[tbills_symbol]]

    holdings = pd.DataFrame(index=monthly_returns.index, columns=monthly_returns.columns, data=0.0)
    trades = pd.DataFrame(index=monthly_returns.index, columns=monthly_returns.columns, data=0.0)

    # Entry
    balance = job.initial_investment
    old_balance = balance
    selected_assets, new_holdings, new_trades, balance, switching_cost = rebalance_multi(
        job, job.lookback_period, market_lookback_returns, tbills_lookback_returns, sam_lookback_returns, monthly_returns, holdings, balance
    )
    holdings.loc[monthly_returns.index[job.lookback_period + 1]] = new_holdings
    trades.loc[monthly_returns.index[job.lookback_period + 1]] = new_trades

    # Remove first month to correctly align with start date. We only needed an extra month to compute first entry.
    monthly_returns = monthly_returns.iloc[1:]
    lookback_returns = lookback_returns.iloc[1:]
    market_lookback_returns = market_lookback_returns.iloc[1:]
    sam_lookback_returns = sam_lookback_returns.iloc[1:] if sam_lookback_returns is not None else None
    tbills_lookback_returns = tbills_lookback_returns.iloc[1:]
    holdings = holdings.iloc[1:]


    portfolio = pd.DataFrame(
        index=monthly_returns.index,
        columns=[
            'Selected Assets',
            'Dual Momentum Return',
            'Switching Fees',
            'Dual Momentum Balance',
        ],
    )

    # Rebalancing is made at the end of the month.
    for i in range(job.lookback_period, len(monthly_returns.index)):
        date = monthly_returns.index[i]

        new_holdings = new_holdings * (1 + monthly_returns.loc[date])
        holdings.loc[date] = new_holdings
        balance = new_holdings.sum()
        portfolio.loc[date] = [','.join(selected_assets), (balance - old_balance) / old_balance, switching_cost, balance]
        trades.iloc[i] = new_trades

        if (i - job.lookback_period) % job.rebalance_period == 0:
            old_balance = balance
            selected_assets, new_holdings, new_trades, balance, switching_cost = rebalance_multi(job, i, market_lookback_returns, tbills_lookback_returns, sam_lookback_returns, monthly_returns, holdings, balance)

    # Cut portfolio to real start date.
    portfolio = portfolio[job.lookback_period :]
    holdings = holdings[job.lookback_period :]
    trades = trades[job.lookback_period :]
    trades = trades.loc[~(trades == 0).all(axis=1)]

    for ticker in all_tickers:
        portfolio[f'{ticker} Return'] = monthly_returns[ticker]
        portfolio[f'{ticker} Balance'] = job.initial_investment * (1 + portfolio[f'{ticker} Return']).cumprod()
    portfolio[f'{tbills_symbol} Return'] = tbills_monthly_returns[tbills_symbol]
    portfolio[f'{tbills_symbol} Balance'] = (
        job.initial_investment * (1 + portfolio[f'{tbills_symbol} Return']).cumprod()
    )

    portfolio = portfolio.infer_objects()

    return DualMomentumMultiResults(portfolio, holdings, trades, ticker_info)


def humanize_portfolio(portfolio: pd.DataFrame) -> pd.DataFrame:
    portfolio = portfolio.copy()

    for col in portfolio.columns:
        if col.endswith('Balance'):
            portfolio[col] = portfolio[col].round(3)
        elif col.endswith('Return') or col == 'Switching Cost':
            portfolio[col] = (portfolio[col] * 100).round(2).astype(str) + '%'

    return portfolio


def job_results_paths(job_id: int) -> tuple[str, str, str]:
    results_dir = ROOT_DIR / 'static' / 'results'
    portfolio_path = results_dir / f'{job_id}-portfolio.csv'
    trades_path = results_dir / f'{job_id}-trades.csv'
    ticker_info_path = results_dir/ f'{job_id}-ticker-info.json'
    return portfolio_path, trades_path, ticker_info_path


def save_results(job_id: int, results: DualMomentumResults):
    portfolio = humanize_portfolio(results.portfolio)
    portfolio_path, trades_path, ticker_info_path = job_results_paths(job_id)
    portfolio.to_csv(portfolio_path)
    results.trades.to_csv(trades_path)
    with open(ticker_info_path, 'w') as f:
        f.write(json.dumps(results.ticker_info, default=fastapi.encoders.jsonable_encoder))


@dataclass
class JobViewModel:
    job: schemas.Job
    portfolio: pd.DataFrame
    trades: pd.DataFrame
    drawdowns: pd.DataFrame
    ticker_info: list[TickerInfo]


def compute_drawdowns(portfolio: pd.DataFrame) -> pd.DataFrame:
    drawdowns = pd.DataFrame()
    for col in portfolio.columns:
        col_search = re.search(r'(.*) Balance', col)
        if col_search:
            asset = col_search.group(1)
            balance_col = f'{asset} Balance'
            running_max = portfolio[balance_col].cummax()
            drawdown_pct = (portfolio[balance_col] - running_max) / running_max * 100
            mdd = drawdown_pct.min()
            drawdowns[f'{asset} Maximum Drawdown'] = [f'{round(mdd, 2)}%']
    return drawdowns


def load_results(job: schemas.Job) -> JobViewModel:
    portfolio_path, trades_path, ticker_info_path = job_results_paths(job.id)
    portfolio = pd.read_csv(portfolio_path)
    trades = pd.read_csv(trades_path)
    if pathlib.Path(ticker_info_path).exists():
        with open(ticker_info_path) as f:
            ticker_info_json = json.load(f)
        ticker_info = [TickerInfo(**ticker_info) for ticker_info in ticker_info_json]
    else:
        ticker_info = []

    portfolio.index = portfolio.index + 1
    trades.index = trades.index + 1

    drawdowns = compute_drawdowns(portfolio)

    return JobViewModel(
        job=job,
        portfolio=portfolio,
        trades=trades,
        drawdowns=drawdowns,
        ticker_info=ticker_info,
    )


def compare_performance(job: schemas.JobBase) -> tuple[pd.DataFrame, datetime, datetime]:
    perfs = {}
    tickers = tickers_to_list(job.tickers)
    for max_assets in range(1, len(tickers) + 1):
        job.max_assets = max_assets
        if max_assets == 1:
            results = dual_momentum(job)
            trades_count = len(results.trades) * 2 - 1
        else:
            results = dual_momentum_multi(job)
            trades_count = results.trades.astype(bool).sum(axis=1).sum()
        start_date = results.portfolio.index[0]
        end_date = results.portfolio.index[-1]
        perfs[max_assets] = {'Balance': results.portfolio.iloc[-1]['Dual Momentum Balance'],
                             'Drawdown': compute_drawdowns(results.portfolio).loc[
                                 0, 'Dual Momentum Maximum Drawdown'], 'Trades': trades_count}
    df = pd.DataFrame.from_dict(perfs, orient='index')
    df.index.name = 'Max Assets'
    return df, start_date, end_date