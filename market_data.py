import yfinance as yf
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

PAIRS = {
    "USD/JPY": "JPY=X",
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "AUD/USD": "AUDUSD=X",
    "NZD/USD": "NZDUSD=X",
    "USD/CHF": "CHF=X",
    "USD/CAD": "CAD=X",
    "EUR/JPY": "EURJPY=X",
    "GBP/JPY": "GBPJPY=X",
    "AUD/JPY": "AUDJPY=X",
    "EUR/GBP": "EURGBP=X",
    "XAU/USD": "GC=F",
}

TIMEFRAMES = {
    "1分足":  ("1d",  "1m"),
    "5分足":  ("5d",  "5m"),
    "15分足": ("5d",  "15m"),
    "1時間足": ("1mo", "1h"),
}


def fetch_ohlcv(pair: str, timeframe: str) -> pd.DataFrame:
    ticker_symbol = PAIRS[pair]
    period, interval = TIMEFRAMES[timeframe]
    ticker = yf.Ticker(ticker_symbol)
    df = ticker.history(period=period, interval=interval)
    if df.empty:
        raise ValueError(f"{pair} のデータ取得失敗")
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)
    return df


def get_current_price(pair: str) -> float:
    ticker_symbol = PAIRS[pair]
    ticker = yf.Ticker(ticker_symbol)
    hist = ticker.history(period="1d", interval="1m")
    if hist.empty:
        return 0.0
    return float(hist["Close"].iloc[-1])
