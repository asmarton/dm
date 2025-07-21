import pathlib
import pandas as pd
import numpy as np
from dataclasses import dataclass
import utils
from db import schemas
from utils import ROOT_DIR
import kfrench

etfs = [
    'XLF',
    'XLK',
    'XLE',
    'XLV',
    'XLI',
    'XBI',
    'XLU',
    'XLP',
    'XLY',
    'KRE',
    'XLB',
    'XLC',
    'XRT',
    'XOP',
    'XLRE',
    'XHB',
    'KBE',
    'XME',
    'KIE',
    'XSD',
    'XAR',
    'XES',
    'KCE',
    'XNTK',
    'XHE',
    'XSW',
    'XPH',
    'XTN',
    'XHS',
    'XITK',
    'XTL',
]


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
    indicators_df: pd.DataFrame
    portfolio: pd.DataFrame
    returns: pd.DataFrame
    trades: pd.DataFrame
    drawdowns: pd.DataFrame


def timing_etfs(job: schemas.IndustryTrendsJobBase) -> TrendFollowingResults:
    max_leverage = job.max_leverage / 100
    target_volatility = job.target_volatility / 100
    rebalance_threshold = job.rebalance_threshold / 100

    price_dfs = []
    for ticker in etfs:
        price_dfs.append(utils.get_closing_prices(ticker))
    benchmark_prices = utils.get_closing_prices(job.benchmark)[job.start_date:job.end_date]
    price_dfs.append(benchmark_prices)
    prices = pd.concat(price_dfs, axis='columns')
    prices = prices[job.start_date : job.end_date]
    returns = prices.pct_change()

    caldt = prices.index
    returns.rename(columns={job.benchmark: 'mkt_ret'}, inplace=True)
    returns['tbill_ret'] = kfrench.tbill_french_reconciled(caldt)

    prices.drop(job.benchmark, axis=1, inplace=True)

    # Determine the number of portfolios dynamically
    num_portfolios = returns.shape[1] - 2  # Subtracting 2 to account for 'mkt_ret' and 'tbill_ret'

    # Indicator parameters
    UP_DAY = job.up_period
    DOWN_DAY = job.down_period
    ADR_VOL_ADJ = 1.4  # ATR is usually 1.4x Vol(close2close)
    KELT_MULT = 2 * ADR_VOL_ADJ

    # Define rolling functions
    def rolling_vol(df, window):
        return df.rolling(window=window).std(ddof=0)

    def rolling_ema(df, window):
        return df.ewm(span=window, adjust=False).mean()

    def rolling_max(df, window):
        return df.rolling(window=window).max()

    def rolling_min(df, window):
        return df.rolling(window=window).min()

    def rolling_mean(df, window):
        return df.rolling(window=window, min_periods=window - 1).mean()

    # Calculate rolling volatility of daily returns
    vol = rolling_vol(returns.iloc[:, :num_portfolios], UP_DAY)

    # Technical indicators
    ema_down = rolling_ema(prices, DOWN_DAY)
    ema_up = rolling_ema(prices, UP_DAY)

    # Donchian channels
    donc_up = rolling_max(prices, UP_DAY)
    donc_down = rolling_min(prices, DOWN_DAY)

    # Keltner bands
    price_change = prices.diff(periods=1).abs()
    kelt_up = ema_up + KELT_MULT * rolling_mean(price_change, UP_DAY)
    kelt_down = ema_down - KELT_MULT * rolling_mean(price_change, DOWN_DAY)

    # Model bands
    long_band = pd.DataFrame(np.minimum(donc_up.values, kelt_up.values), index=donc_up.index, columns=donc_up.columns)
    short_band = pd.DataFrame(
        np.maximum(donc_down.values, kelt_down.values), index=donc_down.index, columns=donc_down.columns
    )

    # Model long signal
    long_band_shifted = long_band.shift(1)
    short_band_shifted = short_band.shift(1)
    long_signal = (prices >= long_band_shifted) & (long_band_shifted > short_band_shifted)

    # Create a dictionary of DataFrames for indicators
    indicator_dfs = {f'ret_{i + 1}': returns.iloc[:, i] for i in range(num_portfolios)}
    indicator_dfs.update({f'price_{i + 1}': prices.iloc[:, i] for i in range(num_portfolios)})
    indicator_dfs.update({f'vol_{i + 1}': vol.iloc[:, i] for i in range(num_portfolios)})
    indicator_dfs.update({f'ema_down_{i + 1}': ema_down.iloc[:, i] for i in range(num_portfolios)})
    indicator_dfs.update({f'ema_up_{i + 1}': ema_up.iloc[:, i] for i in range(num_portfolios)})
    indicator_dfs.update({f'donc_up_{i + 1}': donc_up.iloc[:, i] for i in range(num_portfolios)})
    indicator_dfs.update({f'donc_down_{i + 1}': donc_down.iloc[:, i] for i in range(num_portfolios)})
    indicator_dfs.update({f'kelt_up_{i + 1}': kelt_up.iloc[:, i] for i in range(num_portfolios)})
    indicator_dfs.update({f'kelt_down_{i + 1}': kelt_down.iloc[:, i] for i in range(num_portfolios)})
    indicator_dfs.update({f'long_band_{i + 1}': long_band.iloc[:, i] for i in range(num_portfolios)})
    indicator_dfs.update({f'short_band_{i + 1}': short_band.iloc[:, i] for i in range(num_portfolios)})
    indicator_dfs.update({f'long_signal_{i + 1}': long_signal.iloc[:, i] for i in range(num_portfolios)})

    # Concatenate all indicator columns into a single DataFrame
    indicators_df = pd.concat(indicator_dfs.values(), axis=1)
    indicators_df.columns = indicator_dfs.keys()

    # Add market and tbill returns
    indicators_df['mkt_ret'] = returns['mkt_ret']
    indicators_df['tbill_ret'] = returns['tbill_ret']

    AUM_0 = job.initial_balance
    invest_cash = 'YES'
    target_vol = target_volatility
    max_leverage = max_leverage
    # max_not_trade = 0.20

    N_ind = num_portfolios  # Number of industries in the database
    T = len(indicators_df[indicators_df.columns[0]])  # Length of the time series

    # Pre-allocate arrays with more specific initial values
    exposure = np.zeros((T, N_ind))
    ind_weight = np.zeros((T, N_ind))
    trail_stop_long = np.full((T, N_ind), np.nan)

    # Vectorized indicator data
    rets = indicators_df[[f'ret_{j + 1}' for j in range(N_ind)]].values
    long_signals = indicators_df[[f'long_signal_{j + 1}' for j in range(N_ind)]].values
    long_bands = indicators_df[[f'long_band_{j + 1}' for j in range(N_ind)]].values
    short_bands = indicators_df[[f'short_band_{j + 1}' for j in range(N_ind)]].values
    prices = indicators_df[[f'price_{j + 1}' for j in range(N_ind)]].values
    vols = indicators_df[[f'vol_{j + 1}' for j in range(N_ind)]].values

    for t in range(1, T):
        valid_entries = ~np.isnan(rets[t]) & ~np.isnan(long_bands[t])

        prev_exposure = exposure[t - 1]
        current_exposure = exposure[t]
        current_trail_stop = trail_stop_long[t]
        current_long_signals = long_signals[t]
        current_short_bands = short_bands[t]
        current_prices = prices[t]
        current_vols = vols[t]

        new_long_condition = (prev_exposure <= 0) & (current_long_signals == 1)
        confirm_long_condition = (prev_exposure == 1) & (
            current_prices > np.maximum(trail_stop_long[t - 1], current_short_bands)
        )
        exit_long_condition = (prev_exposure == 1) & (
            current_prices <= np.maximum(trail_stop_long[t - 1], current_short_bands)
        )

        # Process new long positions
        new_longs = valid_entries & new_long_condition
        current_exposure[new_longs] = 1
        current_trail_stop[new_longs] = current_short_bands[new_longs]

        # Process confirmed long positions
        confirm_longs = valid_entries & confirm_long_condition
        current_exposure[confirm_longs] = 1
        current_trail_stop[confirm_longs] = np.maximum(
            trail_stop_long[t - 1, confirm_longs], current_short_bands[confirm_longs]
        )

        # Process exit long positions
        exit_longs = valid_entries & exit_long_condition
        current_exposure[exit_longs] = 0
        ind_weight[t, exit_longs] = 0

        # Update leverage and weights for active long positions
        active_longs = current_exposure == 1
        lev_vol = np.divide(target_vol, current_vols, out=np.zeros_like(current_vols), where=current_vols != 0)
        ind_weight[t, active_longs] = lev_vol[active_longs]

    # Update the indicators dataframe
    # Collect new columns in a dictionary
    new_columns = {}
    for j in range(N_ind):
        new_columns[f'exposure_{j + 1}'] = exposure[:, j]
        new_columns[f'ind_weight_{j + 1}'] = ind_weight[:, j]
        new_columns[f'trail_stop_long_{j + 1}'] = trail_stop_long[:, j]

    # Convert the dictionary to a DataFrame and concatenate with the original DataFrame
    new_columns_df = pd.DataFrame(new_columns, index=indicators_df.index)
    indicators_df = pd.concat([indicators_df, new_columns_df], axis=1)

    # Initialize a DataFrame to store the results at the aggregate portfolio level
    port = pd.DataFrame(index=indicators_df.index)
    port['caldt'] = indicators_df.index
    port['available'] = (
        indicators_df.filter(like='ret_').notna().sum(axis=1)
    )  # How many industries were available each day

    ind_weight_df = indicators_df.filter(like='ind_weight_')
    port_weights = ind_weight_df.div(port['available'], axis=0)

    # Limit the exposure of each industry at "max_not_trade"
    # port_weights = port_weights.clip(upper=max_not_trade)

    port['sum_exposure'] = port_weights.sum(axis=1)
    idx_above_max_lev = port[port['sum_exposure'] > max_leverage].index

    port_weights.loc[idx_above_max_lev] = (
        port_weights.loc[idx_above_max_lev].div(port['sum_exposure'][idx_above_max_lev], axis=0).mul(max_leverage)
    )

    port['sum_exposure'] = port_weights.sum(axis=1)

    for i in range(N_ind):
        port[f'weight_{i + 1}'] = port_weights.iloc[:, i]

    ret_long_components = [
        port[f'weight_{i + 1}'].shift(1).fillna(0) * indicators_df[f'ret_{i + 1}'].fillna(0) for i in range(N_ind)
    ]
    port['ret_long'] = sum(ret_long_components)

    port['ret_tbill'] = (1 - port[[f'weight_{i + 1}' for i in range(N_ind)]].shift(1).sum(axis=1)) * indicators_df[
        'tbill_ret'
    ]

    if invest_cash == 'YES':
        port['ret_long'] += port['ret_tbill']

    port['AUM_simple'] = AUM_0 * (1 + port['ret_long']).cumprod()
    port[job.benchmark] = AUM_0 * (1 + indicators_df['mkt_ret']).cumprod()

    aum = AUM_0
    trades_count = 0
    rebalance_threshold = rebalance_threshold

    all_time_aum = []
    all_time_fees = []
    all_time_shares = []
    all_time_trades = []
    for t in range(0, T):
        date = port.index[t]
        aum = aum * (1 + port['ret_long'].iloc[t])
        shares = port_weights.iloc[t - 1].values * aum / prices[t]
        shares_ = port_weights.iloc[t].values * aum / prices[t]
        trades = shares_ - shares
        trades = np.where(np.isnan(trades), 0, trades)

        saved_trades = trades.copy()
        trades = abs(trades)

        no_trade = []
        for j in range(len(trades)):
            if trades[j] > 0 and shares[j] > 0 and trades[j] / shares[j] < rebalance_threshold:
                no_trade.append(j)
                trades[j] = 0
                saved_trades[j] = 0
                shares_[j] = shares[j]
        all_time_trades.append(saved_trades)
        trade_fees = trades * job.trade_cost_per_share
        trade_fees[(trade_fees > 0) & (trade_fees < 0.35)] = job.trade_cost_min
        trades_count += len(trade_fees[trade_fees > 0])
        trade_fees = sum(trade_fees)
        all_time_fees.append(trade_fees)
        aum = aum - trade_fees
        all_time_aum.append(aum)

        for j in no_trade:
            port_weights.iloc[t, j] = shares_[j] * prices[t][j] / aum
        if len(no_trade) > 0:
            sum_exposure = port_weights.iloc[t].sum()
            if sum_exposure > max_leverage:
                port_weights.iloc[t] = port_weights.iloc[t] / sum_exposure * max_leverage
            if t < T - 1:
                ret_long_components = (
                    port_weights.iloc[t].fillna(0).values
                    * indicators_df.iloc[t + 1].filter(like='ret_').fillna(0).values
                )
                ret_long = sum(ret_long_components)
                port.loc[date, 'ret_long'] = ret_long
                ret_tbill = (1 - port_weights.iloc[t].sum()) * indicators_df.iloc[t + 1]['tbill_ret']
                port.loc[date, 'ret_tbill'] = ret_tbill
                if invest_cash == 'YES':
                    port.loc[date, 'ret_long'] += ret_tbill
        shares = port_weights.iloc[t] * aum / prices[t]
        all_time_shares.append(shares)

    for i in range(N_ind):
        port[f'weight_{i + 1}'] = port_weights.iloc[:, i]

    port['AUM'] = all_time_aum
    port['Fees'] = pd.Series(all_time_fees, index=port.index).cumsum()
    port['Cash'] = (1 - port[[f'weight_{i + 1}' for i in range(N_ind)]].sum(axis=1)) * port['AUM']
    port['Borrowed'] = np.minimum(0, port['Cash']).abs()
    port['Cash'] = np.maximum(0, port['Cash'])

    benchmark_returns = pd.DataFrame(benchmark_prices.groupby(pd.Grouper(freq='ME')).nth(-1).pct_change() * 100).rename(
        columns={job.benchmark: 'Return'})
    benchmark_returns.index.name = 'Date'
    benchmark_returns['Year'] = benchmark_returns.index.year
    benchmark_returns['Month'] = benchmark_returns.index.month
    benchmark_returns = benchmark_returns.pivot(index='Year', columns='Month', values='Return')
    benchmark_returns = ((1 + benchmark_returns / 100).prod(axis=1) - 1) * 100

    monthly_returns = pd.DataFrame(port['AUM'].groupby(pd.Grouper(freq='ME')).nth(-1).pct_change() * 100).rename(
        columns={'AUM': 'Return'})
    monthly_returns.index.name = 'Date'
    monthly_returns['Year'] = monthly_returns.index.year
    monthly_returns['Month'] = monthly_returns.index.month
    monthly_returns = monthly_returns.pivot(index='Year', columns='Month', values='Return')
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    if len(monthly_returns.columns) < 12:
        kept_months = []
        for i in range(0, 12):
            if i in monthly_returns.columns:
                kept_months.append(month_names[i])
        monthly_returns.columns = kept_months
    else:
        monthly_returns.columns = month_names
    monthly_returns['Yearly'] = ((1 + monthly_returns / 100).prod(axis=1) - 1) * 100
    monthly_returns[f'Benchmark ({job.benchmark})'] = benchmark_returns

    weights = port.filter(like='weight_').copy()
    weights.rename(columns={f'weight_{i + 1}': job.tickers[i] for i in range(len(job.tickers))}, inplace=True)

    # shares = pd.DataFrame(all_time_shares)
    trades = pd.DataFrame(all_time_trades)

    drawdowns = pd.DataFrame()
    def compute_drawdown(series: pd.Series | np.ndarray) -> str:
        running_max = np.maximum.accumulate(series)
        drawdown_pct = (series - running_max) / running_max * 100
        mdd = drawdown_pct.min()
        return f'{round(mdd, 2)}%'

    drawdowns['Strategy Drawdown'] = [compute_drawdown(port['AUM'])]
    for j in range(len(job.tickers)):
        ticker = job.tickers[j]
        ticker_prices = prices[:, j]
        drawdowns[ticker] = compute_drawdown(np.where(np.isnan(ticker_prices), 0, ticker_prices))
    drawdowns[f'{job.benchmark} Drawdown'] = [compute_drawdown(benchmark_prices)]

    trades['Date'] = port.index
    trades.set_index('Date', inplace=True)
    trades = trades.loc[~(trades == 0).all(axis=1)]

    return TrendFollowingResults(
        indicators_df=indicators_df,
        portfolio=port,
        returns=monthly_returns,
        trades=trades,
        drawdowns=drawdowns,
    )


def save_results(id: int, results: TrendFollowingResults):
    it_results_dir = ROOT_DIR / 'static' / 'it_results'
    results.portfolio.to_csv(it_results_dir / f'{id}-portfolio.csv')
    results.indicators_df.to_csv(it_results_dir / f'{id}-indicators.csv')
    results.returns.to_csv(it_results_dir / f'{id}-returns.csv')
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
    portfolio = pd.read_csv(it_results_dir / f'{job.id}-portfolio.csv').set_index('Date')
    indicators = pd.read_csv(it_results_dir / f'{job.id}-indicators.csv').set_index('Date')
    returns = pd.read_csv(it_results_dir / f'{job.id}-returns.csv').set_index('Year').fillna('')
    trades = pd.read_csv(it_results_dir / f'{job.id}-trades.csv').set_index('Date')
    drawdowns = pd.read_csv(it_results_dir / f'{job.id}-drawdowns.csv')

    balance = portfolio[['AUM', 'Cash', 'Borrowed', job.benchmark]].copy()
    balance.rename(columns={'AUM': 'Equity'}, inplace=True)
    balance['Holdings'] = portfolio['sum_exposure'] * portfolio['AUM'].values

    trades_count = trades.astype(bool).sum().sum()

    return JobViewModel(job, balance, returns, trades, trades_count, drawdowns)
