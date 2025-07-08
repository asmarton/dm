import pathlib
from datetime import datetime

import pandas as pd
import numpy as np
import yfinance as yf
import math

from dataclasses import dataclass
from pydantic import BaseModel

import utils
from db import schemas
from utils import ROOT_DIR

etfs = ['XLF','XLK','XLE','XLV','XLI','XBI','XLU','XLP','XLY','KRE','XLB','XLC','XRT','XOP','XLRE','XHB','KBE','XME','KIE','XSD','XAR','XES','KCE','XNTK','XHE','XSW','XPH','XTN','XHS','XITK','XTL']


def get_tbills() -> pd.DataFrame:
    path = pathlib.Path(ROOT_DIR) / 'data' / 'DGS1MO.csv'
    dgs1mo = pd.read_csv(path).ffill().set_index('Date')
    dgs1mo.index = pd.to_datetime(dgs1mo.index)
    trading_days = 252
    tbill_daily_return = (1 + dgs1mo / 100) ** (1 / trading_days) - 1
    return tbill_daily_return['DGS1MO']


keltner_mult = 2
atr_approx_factor = 1.4


@dataclass
class TrendFollowingResults:
    weights: pd.DataFrame
    positions: pd.DataFrame
    trailing_stop: pd.DataFrame
    cash: pd.Series
    borrowed: pd.Series
    shares: pd.DataFrame
    holdings: pd.DataFrame
    equity: pd.Series
    balance: pd.DataFrame
    tx_costs: pd.Series
    monthly_returns: pd.DataFrame
    trades: pd.DataFrame
    drawdowns: pd.DataFrame
    channels: pd.DataFrame
    keltner_channels: pd.DataFrame
    donchian_channels: pd.DataFrame


def trend_following_strategy(job: schemas.IndustryTrendsJobBase) -> TrendFollowingResults:
    max_leverage = job.max_leverage / 100
    target_volatility = job.target_volatility / 100
    rebalance_threshold = job.rebalance_threshold / 100

    price_dfs = []
    for ticker in job.tickers:
        price_dfs.append(utils.get_closing_prices(ticker))
    prices = pd.concat(price_dfs, axis='columns')
    prices = prices[job.start_date:job.end_date]

    cash = pd.Series(index=prices.index, data=0.0)
    borrowed = pd.Series(index=prices.index, data=0.0).to_dict()
    tx_costs = pd.Series(index=prices.index, data=0.0).to_dict()

    tbills_daily_rate = get_tbills()

    returns = prices.pct_change()
    volatility = returns.rolling(job.vol_window).std()
    weights = pd.DataFrame(index=prices.index, columns=prices.columns, data=0.0).to_dict(orient='index')
    positions = pd.DataFrame(index=prices.index, columns=prices.columns, data=False).to_dict(orient='index')
    shares = pd.DataFrame(index=prices.index, columns=prices.columns, data=0.0).to_dict(orient='index')

    channels = dict()
    keltner_channels = dict()
    donchian_channels = dict()
    for j in prices.columns:
        for pos in ['up', 'price', 'down']:
            channels[(j, pos)] = dict()
            keltner_channels[(j, pos)] = dict()
            donchian_channels[(j, pos)] = dict()

    ema_up = prices.ewm(span=job.up_period, adjust=False).mean()
    ema_down = prices.ewm(span=job.down_period, adjust=False).mean()
    atr_approx_up = atr_approx_factor * returns.abs().rolling(job.up_period).mean()
    atr_approx_down = atr_approx_factor * returns.abs().rolling(job.down_period).mean()

    keltner_up = ema_up + keltner_mult * atr_approx_up
    keltner_down = ema_down + keltner_mult * atr_approx_down
    donchian_up = prices.rolling(job.up_period).max()
    donchian_down = prices.rolling(job.down_period).min()

    upper_band = np.minimum(donchian_up, keltner_up)
    lower_band = np.maximum(donchian_down, keltner_down)

    trailing_stop = [lower_band.iloc[0]]
    for t in range(1, len(lower_band.index)):
        trailing_stop_t = np.maximum(trailing_stop[t - 1], lower_band.iloc[t])
        trailing_stop.append(trailing_stop_t)
    trailing_stop = pd.DataFrame(index=lower_band.index, columns=lower_band.columns, data=trailing_stop)

    for j in prices.columns:
        channels[(j, 'down')] = lower_band[j]
        channels[(j, 'price')] = prices[j]
        channels[(j, 'up')] = upper_band[j]
        keltner_channels[(j, 'down')] = keltner_down[j]
        keltner_channels[(j, 'price')] = prices[j]
        keltner_channels[(j, 'up')] = keltner_up[j]
        donchian_channels[(j, 'down')] = donchian_down[j]
        donchian_channels[(j, 'price')] = prices[j]
        donchian_channels[(j, 'up')] = donchian_up[j]

    cash.iloc[:max(job.up_period, job.down_period, job.vol_window)] = job.initial_balance

    # EOD: prices, channels, bands, trailing stops. These are computed for the end of the day, because they are derived from close prices.
    # SOD: positions, weights. These are computed for the start of the day. We use t - 1 EOD data to get t SOD positions and weights.
    # This means that at the end of day t - 1 we figure out what trades we need to execute the following morning. After the trades are executed
    # positions and weights for day t become reality. At the end of the t day, when evaluating the portfolio, we use the t positions and weights.

    for t in range(max(job.up_period, job.down_period, job.vol_window), len(prices)):
        date = prices.index[t]
        # print(f'* Day {t} - {date}')
        prev_date = prices.index[t - 1]

        # Compute desired positions at the start of the day
        for j in prices.columns:
            prev_position = positions[prev_date][j]
            prev_price = prices[j].iloc[t - 1]

            # Entry condition
            if not prev_position and prev_price >= upper_band[j].iloc[t - 1]:
                positions[date][j] = True
            # Exit condition
            elif prev_position:
                if prev_price < trailing_stop[j].iloc[t - 1]:
                    positions[date][j] = False
                else:
                    positions[date][j] = True
            else:
                positions[date][j] = False

        # Compute desired weights at the start of the day
        active = positions[date]
        n_active = sum(active.values())
        weights_today = {}
        for j in prices.columns:
            if active[j]:
                sigma = volatility[j].iloc[t]
                if pd.notna(sigma) and sigma > 0:
                    w = (target_volatility / n_active) / sigma
                    weights_today[j] = w
                else:
                    weights_today[j] = 0
            else:
                weights_today[j] = 0
        exposure = sum(weights_today.values())
        if exposure > max_leverage:
            scaling_factor = max_leverage / exposure
            for j in weights_today:
                weights_today[j] *= scaling_factor
        weights[date] = weights_today

        # Evaluate portfolio and compute real weights before rebalancing
        holdings = prices.loc[prev_date] * pd.Series(shares[prev_date])
        borrowed[date] = borrowed[prev_date] + borrowed[prev_date] * tbills_daily_rate.loc[prev_date]
        cash[date] = cash[prev_date] + cash[prev_date] * tbills_daily_rate.loc[prev_date]
        equity = holdings.fillna(0).sum() + cash[date] - borrowed[date]
        before_rebalance_weights = holdings / equity

        trade_balance = 0
        for j in prices.columns:
            if weights_today[j] > before_rebalance_weights[j]:
                # We buy
                amount = (weights_today[j] - before_rebalance_weights[j]) * equity
                bought_shares = amount / prices.loc[prev_date, j]
                pct_change_shares = bought_shares / shares[prev_date][j] if shares[prev_date][j] > 0 else +math.inf
                if pct_change_shares >= rebalance_threshold:
                    shares[date][j] = shares[prev_date][j] + bought_shares
                    fees = max(job.trade_cost_min, job.trade_cost_per_share * bought_shares)
                    tx_costs[date] += fees
                    trade_balance += -fees - amount
                    # print(f'  Buying {bought_shares} shares of {j} for ${amount} (${prices.loc[prev_date, j]}/share) - change {pct_change_shares * 100} % (W: {before_rebalance_weights[j] * 100}% -> {weights_today[j] * 100}%)')
                else:
                    shares[date][j] = shares[prev_date][j]
            elif weights_today[j] < before_rebalance_weights[j]:
                # We sell
                amount = (before_rebalance_weights[j] - weights_today[j]) * equity
                sold_shares = amount / prices.loc[prev_date, j]
                pct_change_shares = sold_shares / shares[prev_date][j] if shares[prev_date][j] > 0 else +math.inf
                if pct_change_shares >= rebalance_threshold:
                    shares[date][j] = shares[prev_date][j] - sold_shares
                    fees = max(job.trade_cost_min, job.trade_cost_per_share * sold_shares)
                    tx_costs[date] += fees
                    trade_balance += -fees + amount
                    # print(f'  Selling {sold_shares} shares of {j} for ${amount} (${prices.loc[prev_date, j]}/share) - change {pct_change_shares * 100} % (W: {before_rebalance_weights[j] * 100}% -> {weights_today[j] * 100}%)')
                else:
                    shares[date][j] = shares[prev_date][j]
            else:
                shares[date][j] = shares[prev_date][j]
        if trade_balance >= 0:
            cash[date] += max(0, trade_balance - borrowed[date])
            borrowed[date] = max(0, borrowed[date] - trade_balance)
        else:
            borrowed[date] += max(0, -trade_balance - cash[date])
            cash[date] = max(0, cash[date] + trade_balance)
        # equity = (prices.loc[date] * pd.Series(shares[date])).fillna(0).sum() + cash[date] - borrowed[date]
        # print(f'- Cash: {cash[date]} | Borrowed: {borrowed[date]} | Holdings: {(prices.loc[date] * pd.Series(shares[date])).fillna(0).sum()} | Equity: {equity} | Exposure: {sum(weights_today.values())} | Leverage: {(equity + borrowed[date]) / equity}')
    weights = pd.DataFrame.from_dict(weights, orient='index')
    positions = pd.DataFrame.from_dict(positions, orient='index')
    shares = pd.DataFrame.from_dict(shares, orient='index')
    cash = pd.Series(cash)
    borrowed = pd.Series(borrowed)
    tx_costs = pd.Series(tx_costs)
    channels = pd.DataFrame.from_dict(channels)
    keltner_channels = pd.DataFrame.from_dict(keltner_channels)
    donchian_channels = pd.DataFrame.from_dict(donchian_channels)

    holdings = (shares * prices).fillna(0).sum(axis=1)
    equity = holdings + cash - borrowed

    benchmark_prices = utils.get_closing_prices(job.benchmark)[job.start_date:job.end_date]
    benchmark_balance = (benchmark_prices.pct_change() + 1).fillna(1).cumprod() * job.initial_balance
    balance = pd.concat([equity, holdings, cash, borrowed, benchmark_balance], axis=1).rename(columns={0: 'Equity', 1: 'Holdings', 2: 'Cash', 3: 'Borrowed'})
    balance.index.name = 'Date'

    benchmark_returns = pd.DataFrame(benchmark_prices.groupby(pd.Grouper(freq='ME')).nth(-1).pct_change() * 100).rename(
        columns={job.benchmark: 'Return'})
    print(benchmark_returns.columns)
    benchmark_returns.index.name = 'Date'
    benchmark_returns['Year'] = benchmark_returns.index.year
    benchmark_returns['Month'] = benchmark_returns.index.month
    print(benchmark_returns.columns)
    benchmark_returns = benchmark_returns.pivot(index='Year', columns='Month', values='Return')
    benchmark_returns = ((1 + benchmark_returns / 100).prod(axis=1) - 1) * 100

    monthly_returns = pd.DataFrame(equity.groupby(pd.Grouper(freq='ME')).nth(-1).pct_change() * 100).rename(
        columns={0: 'Return'})
    monthly_returns.index.name = 'Date'
    monthly_returns['Year'] = monthly_returns.index.year
    monthly_returns['Month'] = monthly_returns.index.month
    monthly_returns = monthly_returns.pivot(index='Year', columns='Month', values='Return')
    monthly_returns.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_returns['Yearly'] = ((1 + monthly_returns / 100).prod(axis=1) - 1) * 100
    monthly_returns[f'Benchmark ({job.benchmark})'] = benchmark_returns

    trades = shares.diff().fillna(0)
    trades = trades.loc[~(trades == 0).all(axis=1)]
    trades.index.name = 'Date'

    # TODO:
    drawdowns = pd.DataFrame()
    def compute_drawdown(col: str):
        running_max = balance[col].cummax()
        drawdown_pct = (balance[col] - running_max) / running_max * 100
        mdd = drawdown_pct.min()
        return f'{round(mdd, 2)}%'
    drawdowns['Max Strategy Drawdown'] = [compute_drawdown('Equity')]
    drawdowns[f'Max {job.benchmark} Drawdown'] = [compute_drawdown(job.benchmark)]

    return TrendFollowingResults(
        weights=weights,
        positions=positions,
        trailing_stop=trailing_stop,
        cash=cash,
        borrowed=borrowed,
        shares=shares,
        holdings=holdings,
        equity=equity,
        balance=balance,
        tx_costs=tx_costs,
        monthly_returns=monthly_returns,
        trades=trades,
        drawdowns=drawdowns,
        channels=channels,
        keltner_channels=keltner_channels,
        donchian_channels=donchian_channels,
    )


def save_results(id: int, results: TrendFollowingResults):
    it_results_dir = ROOT_DIR / 'static' / 'it_results'
    results.balance.to_csv(it_results_dir / f'{id}-balance.csv')
    results.monthly_returns.to_csv(it_results_dir / f'{id}-monthly_returns.csv')
    results.trades.to_csv(it_results_dir / f'{id}-trades.csv')
    results.drawdowns.to_csv(it_results_dir / f'{id}-drawdowns.csv', index=False)


@dataclass
class JobViewModel:
    job: schemas.IndustryTrendsJob
    balance: pd.DataFrame
    returns: pd.DataFrame
    trades: pd.DataFrame
    trades_count: int
    drawdowns: pd.DataFrame


def load_results(job: schemas.IndustryTrendsJob) -> JobViewModel:
    it_results_dir = ROOT_DIR / 'static' / 'it_results'
    balance = pd.read_csv(it_results_dir / f'{job.id}-balance.csv').set_index('Date')
    returns = pd.read_csv(it_results_dir / f'{job.id}-monthly_returns.csv').set_index('Year').fillna('')
    trades = pd.read_csv(it_results_dir / f'{job.id}-trades.csv').set_index('Date')
    drawdowns = pd.read_csv(it_results_dir / f'{job.id}-drawdowns.csv')

    trades_count = trades.astype(bool).sum().sum()

    return JobViewModel(job, balance, returns, trades, trades_count, drawdowns)