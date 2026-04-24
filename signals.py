import pandas as pd
import numpy as np


# ── インジケーター計算 ────────────────────────────────────────

def calc_rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])


def calc_ema(close: pd.Series, period: int) -> pd.Series:
    return close.ewm(span=period, adjust=False).mean()


def calc_macd(close: pd.Series):
    ema12  = calc_ema(close, 12)
    ema26  = calc_ema(close, 26)
    macd   = ema12 - ema26
    signal = calc_ema(macd, 9)
    hist   = macd - signal
    return float(macd.iloc[-1]), float(signal.iloc[-1]), float(hist.iloc[-1])


def calc_bollinger(close: pd.Series, period: int = 20, sigma: float = 2.0):
    ma    = close.rolling(period).mean()
    std   = close.rolling(period).std()
    upper = ma + sigma * std
    lower = ma - sigma * std
    price = float(close.iloc[-1])
    pct_b = float((price - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1]))
    return float(upper.iloc[-1]), float(ma.iloc[-1]), float(lower.iloc[-1]), pct_b


def calc_stochastic(df: pd.DataFrame, k: int = 14, d: int = 3):
    low_min  = df["Low"].rolling(k).min()
    high_max = df["High"].rolling(k).max()
    k_val = 100 * (df["Close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    d_val = k_val.rolling(d).mean()
    return float(k_val.iloc[-1]), float(d_val.iloc[-1])


def calc_atr(df: pd.DataFrame, period: int = 14) -> float:
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - df["Close"].shift()).abs()
    tr3 = (df["Low"]  - df["Close"].shift()).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])


# ── スキャルピング判定 ────────────────────────────────────────

def analyze(df: pd.DataFrame) -> dict:
    close = df["Close"]

    rsi              = calc_rsi(close)
    ema5             = float(calc_ema(close, 5).iloc[-1])
    ema20            = float(calc_ema(close, 20).iloc[-1])
    ema5_prev        = float(calc_ema(close, 5).iloc[-2])
    ema20_prev       = float(calc_ema(close, 20).iloc[-2])
    macd, sig, hist  = calc_macd(close)
    bb_up, bb_mid, bb_low, pct_b = calc_bollinger(close)
    stoch_k, stoch_d = calc_stochastic(df)
    atr              = calc_atr(df)
    price            = float(close.iloc[-1])

    signals = []
    score   = 0  # 買い+、売りー

    # ── RSI ─────────────────────────────────────────────────
    if rsi < 30:
        score += 2
        signals.append({"indicator": "RSI", "value": f"{rsi:.1f}", "signal": "BUY",     "reason": "売られすぎ（30未満）"})
    elif rsi < 40:
        score += 1
        signals.append({"indicator": "RSI", "value": f"{rsi:.1f}", "signal": "BUY",     "reason": "やや売られすぎ"})
    elif rsi > 70:
        score -= 2
        signals.append({"indicator": "RSI", "value": f"{rsi:.1f}", "signal": "SELL",    "reason": "買われすぎ（70超）"})
    elif rsi > 60:
        score -= 1
        signals.append({"indicator": "RSI", "value": f"{rsi:.1f}", "signal": "SELL",    "reason": "やや買われすぎ"})
    else:
        signals.append({"indicator": "RSI", "value": f"{rsi:.1f}", "signal": "NEUTRAL", "reason": "中立ゾーン"})

    # ── EMA クロス ──────────────────────────────────────────
    golden = ema5_prev < ema20_prev and ema5 > ema20
    dead   = ema5_prev > ema20_prev and ema5 < ema20
    if golden:
        score += 2
        signals.append({"indicator": "EMA5/20", "value": f"5:{ema5:.4f} 20:{ema20:.4f}", "signal": "BUY",  "reason": "ゴールデンクロス"})
    elif dead:
        score -= 2
        signals.append({"indicator": "EMA5/20", "value": f"5:{ema5:.4f} 20:{ema20:.4f}", "signal": "SELL", "reason": "デッドクロス"})
    elif ema5 > ema20:
        score += 1
        signals.append({"indicator": "EMA5/20", "value": f"5:{ema5:.4f} 20:{ema20:.4f}", "signal": "BUY",  "reason": "EMA5がEMA20上方"})
    else:
        score -= 1
        signals.append({"indicator": "EMA5/20", "value": f"5:{ema5:.4f} 20:{ema20:.4f}", "signal": "SELL", "reason": "EMA5がEMA20下方"})

    # ── MACD ────────────────────────────────────────────────
    if hist > 0 and macd > sig:
        score += 1
        signals.append({"indicator": "MACD", "value": f"{macd:.5f}", "signal": "BUY",     "reason": "MACDがシグナル上"})
    elif hist < 0 and macd < sig:
        score -= 1
        signals.append({"indicator": "MACD", "value": f"{macd:.5f}", "signal": "SELL",    "reason": "MACDがシグナル下"})
    else:
        signals.append({"indicator": "MACD", "value": f"{macd:.5f}", "signal": "NEUTRAL", "reason": "クロス付近"})

    # ── ボリンジャーバンド ────────────────────────────────────
    if pct_b < 0.1:
        score += 2
        signals.append({"indicator": "BB %B", "value": f"{pct_b:.2f}", "signal": "BUY",     "reason": "下限バンド付近（反発狙い）"})
    elif pct_b > 0.9:
        score -= 2
        signals.append({"indicator": "BB %B", "value": f"{pct_b:.2f}", "signal": "SELL",    "reason": "上限バンド付近（反落狙い）"})
    elif pct_b < 0.3:
        score += 1
        signals.append({"indicator": "BB %B", "value": f"{pct_b:.2f}", "signal": "BUY",     "reason": "バンド下半分"})
    elif pct_b > 0.7:
        score -= 1
        signals.append({"indicator": "BB %B", "value": f"{pct_b:.2f}", "signal": "SELL",    "reason": "バンド上半分"})
    else:
        signals.append({"indicator": "BB %B", "value": f"{pct_b:.2f}", "signal": "NEUTRAL", "reason": "バンド中央付近"})

    # ── ストキャスティクス ────────────────────────────────────
    if stoch_k < 20 and stoch_d < 20:
        score += 2
        signals.append({"indicator": "Stoch %K/%D", "value": f"{stoch_k:.1f}/{stoch_d:.1f}", "signal": "BUY",     "reason": "売られすぎゾーン（20未満）"})
    elif stoch_k > 80 and stoch_d > 80:
        score -= 2
        signals.append({"indicator": "Stoch %K/%D", "value": f"{stoch_k:.1f}/{stoch_d:.1f}", "signal": "SELL",    "reason": "買われすぎゾーン（80超）"})
    elif stoch_k > stoch_d:
        score += 1
        signals.append({"indicator": "Stoch %K/%D", "value": f"{stoch_k:.1f}/{stoch_d:.1f}", "signal": "BUY",     "reason": "%Kが%D上方"})
    elif stoch_k < stoch_d:
        score -= 1
        signals.append({"indicator": "Stoch %K/%D", "value": f"{stoch_k:.1f}/{stoch_d:.1f}", "signal": "SELL",    "reason": "%Kが%D下方"})
    else:
        signals.append({"indicator": "Stoch %K/%D", "value": f"{stoch_k:.1f}/{stoch_d:.1f}", "signal": "NEUTRAL", "reason": "クロス付近"})

    # ── 総合判定 ─────────────────────────────────────────────
    max_score = 10
    if score >= 5:
        verdict, verdict_label, color = "STRONG_BUY",  "★ 強く買い推奨",  "#00c853"
    elif score >= 2:
        verdict, verdict_label, color = "BUY",         "◎ 買い",          "#00e676"
    elif score <= -5:
        verdict, verdict_label, color = "STRONG_SELL", "★ 強く売り推奨",  "#d50000"
    elif score <= -2:
        verdict, verdict_label, color = "SELL",        "◎ 売り",          "#ff1744"
    else:
        verdict, verdict_label, color = "NEUTRAL",     "△ 様子見",        "#ffd600"

    return {
        "price":         price,
        "score":         score,
        "max_score":     max_score,
        "verdict":       verdict,
        "verdict_label": verdict_label,
        "color":         color,
        "signals":       signals,
        "indicators": {
            "rsi":     rsi,
            "ema5":    ema5,
            "ema20":   ema20,
            "macd":    macd,
            "bb_up":   bb_up,
            "bb_mid":  bb_mid,
            "bb_low":  bb_low,
            "pct_b":   pct_b,
            "stoch_k": stoch_k,
            "stoch_d": stoch_d,
            "atr":     atr,
        },
        "chart_data": df,
    }
