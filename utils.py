import os
import pandas as pd
import pathlib
import yfinance as yf
from datetime import datetime

ROOT_DIR = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))

HISTORICAL_DATA_TICKERS = set(['MSCI ACWI ex US', 'MSCI World ex US', 'VBMFX'])

def get_closing_prices(ticker: str) -> pd.DataFrame:
    path = ROOT_DIR / 'data' / f'{ticker}.csv'
    should_refresh = False

    if path.exists():
        prices = pd.read_csv(path, thousands=',').dropna().set_index('Date')
        prices.index = pd.to_datetime(prices.index)
        last_price_date = prices.index[-1]
        if last_price_date < datetime.now() and last_price_date.month != datetime.now().month:
            # If different month, download more data
            should_refresh = True
    else:
        should_refresh = True

    if should_refresh and ticker not in HISTORICAL_DATA_TICKERS:
        yf_prices = yf.download(tickers=[ticker], period='max', auto_adjust=True)['Close']
        if len(yf_prices.index) > 0:
            prices = yf_prices
            prices.to_csv(path)
    return prices


def tickers_to_list(tickers: str) -> list[str]:
    return [t.strip() for t in tickers.split(',')]


def clear_tickers_list(tickers: str) -> str:
    return ','.join(tickers_to_list(tickers))
