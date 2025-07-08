import os
import pandas as pd
import pathlib
import yfinance as yf

ROOT_DIR = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))


def get_closing_prices(ticker: str) -> pd.DataFrame:
    path = ROOT_DIR / 'data' / f'{ticker}.csv'
    if path.exists():
        prices = pd.read_csv(path, thousands=',').dropna().set_index('Date')
        prices.index = pd.to_datetime(prices.index)
    else:
        prices = yf.download(tickers=[ticker], period='max', auto_adjust=True)['Close']
        prices.to_csv(path)
    return prices


def tickers_to_list(tickers: str) -> list[str]:
    return [t.strip() for t in tickers.split(',')]


def clear_tickers_list(tickers: str) -> str:
    return ','.join(tickers_to_list(tickers))