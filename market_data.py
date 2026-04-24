import yfinance as yf
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import pytz

logger = logging.getLogger(__name__)


def is_market_open() -> tuple[bool, str]:
    """FX市場が開いているか判定。(開いているか, 状態メッセージ) を返す"""
    jst = pytz.timezone("Asia/Tokyo")
    now = datetime.now(jst)
    weekday = now.weekday()  # 0=月〜6=日
    hour = now.hour

    # 土曜日6時以降〜日曜日23時59分 → クローズ
    # 月曜日0時〜5時59分 → クローズ（週明けオープン前）
    if weekday == 5 and hour >= 6:
        return False, "⛔ 取引不可 — 週末クローズ中（月曜 06:00 JST にオープン）"
    if weekday == 6:
        return False, "⛔ 取引不可 — 週末クローズ中（月曜 06:00 JST にオープン）"
    if weekday == 0 and hour < 6:
        return False, "⛔ 取引不可 — 週明けオープン前（月曜 06:00 JST にオープン）"

    # 平日でもNYクローズ直後（土曜0〜5時台）
    if weekday == 5 and hour < 6:
        return False, "⛔ 取引不可 — NY市場クローズ済み（月曜 06:00 JST にオープン）"

    return True, "✅ 取引可能"

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
