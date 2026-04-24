import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import pytz

from market_data import PAIRS, TIMEFRAMES, fetch_ohlcv, is_market_open
from signals import analyze, calc_ema, calc_rsi

st.set_page_config(
    page_title="FX GO - スキャルピング判定",
    page_icon="📈",
    layout="wide",
)

st.title("📈 FX GO — スキャルピング売買判定")
st.caption("リアルタイムテクニカル分析で買い・売りタイミングを判定")

# ── 市場オープン確認 ─────────────────────────────────────────
market_open, market_msg = is_market_open()
if not market_open:
    st.error(f"### {market_msg}")
    st.markdown("""
    **FX市場の営業時間（JST）**
    | 曜日 | 状態 |
    |------|------|
    | 月曜 06:00 〜 土曜 06:00 | ✅ 取引可能 |
    | 土曜 06:00 〜 月曜 06:00 | ⛔ 取引不可（週末クローズ） |

    > 下記のシグナルは参考値です。実際の取引は月曜日の市場オープン後に行ってください。
    """)
else:
    st.success(market_msg)

# ── サイドバー：設定 ─────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 設定")
    pair      = st.selectbox("通貨ペア",  list(PAIRS.keys()),     index=0)
    timeframe = st.selectbox("時間足",    list(TIMEFRAMES.keys()), index=1)
    auto_refresh = st.toggle("自動更新（30秒）", value=False)

    st.divider()
    st.caption("データソース: Yahoo Finance")
    st.caption("※ 15〜20分の遅延があります")

# ── データ取得 ───────────────────────────────────────────────
jst = pytz.timezone("Asia/Tokyo")
now = datetime.now(jst).strftime("%Y-%m-%d %H:%M:%S")

try:
    with st.spinner(f"{pair} {timeframe} のデータ取得中..."):
        df     = fetch_ohlcv(pair, timeframe)
        result = analyze(df)
except Exception as e:
    st.error(f"データ取得エラー: {e}")
    st.stop()

price         = result["price"]
score         = result["score"]
verdict       = result["verdict"]
verdict_label = result["verdict_label"]
color         = result["color"]
signals       = result["signals"]
ind           = result["indicators"]

# ── ヘッダー指標 ─────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("通貨ペア",    pair)
c2.metric("現在価格",    f"{price:.5f}" if "JPY" not in pair else f"{price:.3f}")
c3.metric("時間足",      timeframe)
c4.metric("取得時刻 JST", now)

st.divider()

# ── 判定バナー ───────────────────────────────────────────────
bar_score = max(0, min(10, score + 10))  # -10〜+10 → 0〜20
st.markdown(
    f"""
    <div style="background:{color};padding:20px 28px;border-radius:14px;text-align:center;margin-bottom:16px;">
      <span style="font-size:2.2rem;font-weight:bold;color:{'#000' if verdict=='NEUTRAL' else '#fff'}">
        {verdict_label}
      </span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── シグナル内訳 ─────────────────────────────────────────────
st.subheader("📊 シグナル内訳")

SIGNAL_COLOR = {"BUY": "🟢", "SELL": "🔴", "NEUTRAL": "🟡"}

rows = []
for s in signals:
    icon = SIGNAL_COLOR.get(s["signal"], "⚪")
    rows.append({
        "": icon,
        "インジケーター": s["indicator"],
        "値":            s["value"],
        "判定":          s["signal"],
        "根拠":          s["reason"],
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ── インジケーター数値 ────────────────────────────────────────
st.subheader("🔢 インジケーター数値")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("RSI(14)",    f"{ind['rsi']:.1f}",
            delta="売られすぎ" if ind['rsi'] < 30 else ("買われすぎ" if ind['rsi'] > 70 else "中立"))
col2.metric("EMA5",       f"{ind['ema5']:.4f}")
col3.metric("EMA20",      f"{ind['ema20']:.4f}")
col4.metric("Stoch %K",   f"{ind['stoch_k']:.1f}")
col5.metric("ATR(14)",    f"{ind['atr']:.5f}")

col6, col7, col8, col9, col10 = st.columns(5)
col6.metric("MACD",       f"{ind['macd']:.5f}")
col7.metric("BB 上限",    f"{ind['bb_up']:.4f}")
col8.metric("BB 中央",    f"{ind['bb_mid']:.4f}")
col9.metric("BB 下限",    f"{ind['bb_low']:.4f}")
col10.metric("BB %B",     f"{ind['pct_b']:.2f}")

# ── ローソク足チャート ────────────────────────────────────────
st.subheader("📉 チャート")

chart_df = result["chart_data"].copy().tail(100)
close    = chart_df["Close"]

# EMA線を追加
chart_df["EMA5"]  = close.ewm(span=5,  adjust=False).mean()
chart_df["EMA20"] = close.ewm(span=20, adjust=False).mean()
chart_df["BB_UP"] = close.rolling(20).mean() + 2 * close.rolling(20).std()
chart_df["BB_LO"] = close.rolling(20).mean() - 2 * close.rolling(20).std()

tab1, tab2 = st.tabs(["価格 + EMA", "ボリンジャーバンド"])

with tab1:
    st.line_chart(chart_df[["Close", "EMA5", "EMA20"]])

with tab2:
    st.line_chart(chart_df[["Close", "BB_UP", "BB_LO"]])

# ── 総合スコアゲージ ──────────────────────────────────────────
st.subheader("📐 総合スコア")
st.markdown(f"**買いシグナル合計: {max(0, score)} / 売りシグナル合計: {abs(min(0, score))}**")

gauge_pct = (score + 10) / 20
buy_w  = int(gauge_pct * 100)
sell_w = 100 - buy_w
st.markdown(
    f"""
    <div style="display:flex;height:28px;border-radius:8px;overflow:hidden;margin-top:4px;">
      <div style="width:{buy_w}%;background:#00c853;"></div>
      <div style="width:{sell_w}%;background:#ff1744;"></div>
    </div>
    <div style="display:flex;justify-content:space-between;font-size:0.85rem;margin-top:4px;">
      <span>← 売り</span><span>中立</span><span>買い →</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── 注意書き ──────────────────────────────────────────────────
st.divider()
st.warning("⚠️ このツールは情報提供のみを目的としています。投資判断はご自身の責任で行ってください。")

# ── 自動更新 ──────────────────────────────────────────────────
if auto_refresh:
    st.markdown('<meta http-equiv="refresh" content="30">', unsafe_allow_html=True)
    st.caption("🔄 30秒ごとに自動更新中")
