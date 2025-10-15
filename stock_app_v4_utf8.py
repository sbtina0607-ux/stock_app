# streamlit_stock_app_final_fixed.py

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import rcParams, font_manager
from math import pi
import matplotlib.colors as mcolors
import datetime

st.set_page_config(layout="wide", page_title="股票分析系統")

# -----------------------------
# 設定中文字型 (回落機制)
# -----------------------------
def set_chinese_font():
    import platform
    try:
        if platform.system() == 'Windows':
            font_path = 'C:\\Windows\\Fonts\\msjh.ttc'
        elif platform.system() == 'Darwin':
            font_path = '/System/Library/Fonts/STHeiti Medium.ttc'
        else:
            font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
        myfont = font_manager.FontProperties(fname=font_path)
        rcParams['axes.unicode_minus'] = False
        return myfont
    except Exception:
        return None

myfont = set_chinese_font()

# -----------------------------
# 安全取值函數 (防止 Series 導致 ValueError)
# -----------------------------
def safe_float(x):
    """
    將單一值或 Series 轉成 float
    """
    if isinstance(x, pd.Series):
        x_valid = x.dropna()
        if not x_valid.empty:
            return float(x_valid.iloc[-1])
        else:
            return np.nan
    elif pd.notna(x):
        return float(x)
    else:
        return np.nan

# -----------------------------
# 使用者提供的美股清單 (已整理)
# -----------------------------
us_stocks = [
"AAPL","MSFT","GOOGL","AMZN","NVDA","META","AVGO","TSLA","BRK.B","ORCL","JPM","WMT","LLY","V","UNH",
"HD","MA","NFLX","BAC","VZ","ADBE","CMCSA","KO","NKE","INTC","PFE","CSCO","PEP","MRK","COST",
"MCD","IBM","TXN","MDT","NEE","AMGN","QCOM","DHR","CRM","PM","BMY","LOW","RTX","SBUX","GS",
"AXP","AMD","NOW","UBER","ZM","MMM","GE","CAT","LMT","SCHW","BLK","T","BK","PLD","SPGI",
"ISRG","GILD","ADP","CME","CI","SYK","HUM","MDLZ","AMAT","MAR","CSX","DD","TJX","CB","FIS",
"CHTR","ZTS","MO","EL","ADI","MU","ETN","CCI","EQIX","WM","EMR","FDX","GM","LRCX"
]

# -----------------------------
# 使用者提供的台股清單 (已整理)
# -----------------------------
taiwan_stocks = [
"2330.TW","2317.TW","2382.TW","2308.TW","2454.TW","2891.TW","2881.TW","2882.TW","1101.TW","2885.TW",
"3045.TW","3481.TW","3037.TW","1301.TW","1303.TW","2301.TW","2357.TW","2327.TW","2474.TW","2408.TW",
"2603.TW","2609.TW"
]

# combine options for multiselect
all_options = ["🇺🇸 " + s for s in us_stocks] + ["🇹🇼 " + s for s in taiwan_stocks]

# -----------------------------
# UI: 選擇/新增股票
# -----------------------------
st.title("📈 股票分析系統（含技術指標、線性預測與蒙地卡羅）")

with st.sidebar:
    st.header("選擇或加入股票")
    selected = st.multiselect("下拉選擇（可多選 / 可搜尋）", all_options,
                              default=["🇺🇸 AAPL","🇹🇼 2330.TW"])
    manual_input = st.text_input("手動輸入股票代碼（例如 AAPL 或 2330.TW）")
    if st.button("加入手動代碼"):
        if manual_input:
            if 'manual_list' not in st.session_state:
                st.session_state['manual_list'] = []
            code = manual_input.strip().upper()
            if code not in st.session_state['manual_list']:
                st.session_state['manual_list'].append(code)
            st.success(f"已加入：{code}")
    manual_list = st.session_state.get('manual_list', [])
    if manual_list:
        st.write("手動加入清單：", manual_list)

# Build final tickers list to analyze
final_selection = []
final_selection.extend(selected)
for m in manual_list:
    final_selection.append("🔎 " + m)

if not final_selection:
    st.warning("請至少選擇或加入一檔股票。")
    st.stop()

# -----------------------------
# 單支股票技術分析函數
# -----------------------------
def stock_analysis(ticker, lookback=365, horizon=10, target=0.05, monte_carlo_sim=1000):
    try:
        data = yf.download(ticker, period=f"{lookback}d", auto_adjust=True, progress=False)
    except Exception as e:
        return None
    if data is None or data.empty:
        return None

    # 計算技術指標
    data['MA20'] = data['Close'].rolling(20).mean()
    data['MA60'] = data['Close'].rolling(60).mean()
    delta = data['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    RS = roll_up / roll_down
    data['RSI'] = 100.0 - (100.0 / (1.0 + RS))
    exp12 = data['Close'].ewm(span=12, adjust=False).mean()
    exp26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp12 - exp26
    data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    # ATR
    data['H-L'] = data['High'] - data['Low']
    data['H-C'] = abs(data['High'] - data['Close'].shift(1))
    data['L-C'] = abs(data['Low'] - data['Close'].shift(1))
    data['TR'] = data[['H-L', 'H-C', 'L-C']].max(axis=1)
    data['ATR'] = data['TR'].rolling(14).mean()
    # 布林通道
    data['BB_Mid'] = data['Close'].rolling(20).mean()
    data['BB_Std'] = data['Close'].rolling(20).std()
    data['BB_Upper'] = data['BB_Mid'] + 2 * data['BB_Std']
    data['BB_Lower'] = data['BB_Mid'] - 2 * data['BB_Std']
    # 歷史波動率
    data['Returns'] = data['Close'].pct_change()
    volatility = data['Returns'].std() * np.sqrt(252)

    last_row = data.iloc[-1]
    current_price = safe_float(last_row['Close'])
    ma20 = safe_float(last_row['MA20'])
    ma60 = safe_float(last_row['MA60'])
    rsi = safe_float(last_row['RSI'])
    macd = safe_float(last_row['MACD'])
    signal = safe_float(last_row['Signal'])
    atr = safe_float(last_row['ATR'])
    bb_upper = safe_float(last_row['BB_Upper'])
    bb_lower = safe_float(last_row['BB_Lower'])

    # 技術指標訊號
    data['Action'] = '持有'
    data.loc[(data['RSI'] < 30) & (data['MA20'] > data['MA60']), 'Action'] = '買進'
    data.loc[(data['RSI'] > 70) & (data['MA20'] < data['MA60']), 'Action'] = '賣出'
    macd_cross_up = (data['MACD'] > data['Signal']) & (data['MACD'].shift(1) <= data['Signal'].shift(1))
    macd_cross_down = (data['MACD'] < data['Signal']) & (data['MACD'].shift(1) >= data['Signal'].shift(1))
    data.loc[macd_cross_up, 'Action'] = '買進'
    data.loc[macd_cross_down, 'Action'] = '賣出'
    last_action = data['Action'].iloc[-1]
    trend = "上升趨勢" if (not np.isnan(ma20) and not np.isnan(ma60) and ma20 > ma60) else "下降趨勢"
    entry_price = current_price
    target_price = entry_price * (1 + target)
    stop_loss = entry_price - atr if not np.isnan(atr) else entry_price * 0.98

    # 線性回歸趨勢預測
    recent_data = data['Close'].iloc[-60:]
    x = np.arange(len(recent_data)).reshape(-1, 1)
    y = recent_data.values.reshape(-1, 1)
    model = LinearRegression()
    model.fit(x, y)
    x_future = np.arange(len(recent_data), len(recent_data) + horizon).reshape(-1, 1)
    y_future = model.predict(np.vstack([x, x_future]))
    y_future_only = model.predict(x_future).flatten()
    last_date = data.index[-1]
    future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=horizon)
    future_change = (y_future_only[-1] - y_future_only[0]) / (y_future_only[0]) * 100 if len(y_future_only) >= horizon else 0.0
    future_trend = "上漲" if future_change > 0 else "下跌"

    # 蒙地卡羅模擬
    np.random.seed(42)
    sim_paths = np.zeros((monte_carlo_sim, horizon))
    mu = data['Returns'].mean()
    sigma = data['Returns'].std()
    for i in range(monte_carlo_sim):
        daily_returns = np.random.normal(mu, sigma, horizon)
        path = current_price * np.cumprod(1 + daily_returns)
        sim_paths[i, :] = path
    sim_results_final = sim_paths[:, -1]
    prob_up = np.mean(sim_results_final > current_price)
    prob_target = np.mean(sim_results_final >= target_price)
    lower_band = np.percentile(sim_paths, 2.5, axis=0)
    upper_band = np.percentile(sim_paths, 97.5, axis=0)
    median_path = np.percentile(sim_paths, 50, axis=0)

    # 分數
    short_score = max(0, (30 - rsi)) if not np.isnan(rsi) else 0.0
    mid_score = max(0, (ma20 - ma60)) if (not np.isnan(ma20) and not np.isnan(ma60)) else 0.0
    long_score = prob_up * 100

    result = {
        'ticker': ticker,
        'data': data,
        'current_price': current_price,
        'ma20': ma20,
        'ma60': ma60,
        'rsi': rsi,
        'macd': macd,
        'signal': signal,
        'atr': atr,
        'bb_upper': bb_upper,
        'bb_lower': bb_lower,
        'volatility': volatility,
        'last_action': last_action,
        'trend': trend,
        'entry_price': entry_price,
        'target_price': target_price,
        'stop_loss': stop_loss,
        'future_trend': future_trend,
        'future_change': future_change,
        'future_dates': future_dates,
        'y_future_only': y_future_only,
        'y_future_full': y_future,
        'sim_paths': sim_paths,
        'sim_final': sim_results_final,
        'lower_band': lower_band,
        'upper_band': upper_band,
        'median_path': median_path,
        'prob_up': prob_up,
        'prob_target': prob_target,
        'short_score': short_score,
        'mid_score': mid_score,
        'long_score': long_score
    }
    return result

# -----------------------------
# 主執行區 (分析所有選擇)
# -----------------------------
st.subheader("🔍 分析進度與結果")

col1, col2 = st.columns([1, 2])
with col1:
    st.info("已選股票")
    st.write(final_selection)

progress = st.progress(0)
results = []
total = len(final_selection)
for i, item in enumerate(final_selection):
    if item.startswith("🇺🇸") or item.startswith("🇹🇼"):
        ticker = item.split(" ")[1]
    elif item.startswith("🔎"):
        ticker = item.split(" ")[1]
    else:
        ticker = item.split(" ")[-1]
    st.write(f"分析中：{ticker} ({i+1}/{total})")
    res = stock_analysis(ticker, lookback=365, horizon=20, target=0.05, monte_carlo_sim=500)
    if res is None:
        st.error(f"{ticker} 資料抓取失敗或無資料")
    else:
        results.append(res)
    progress.progress((i + 1) / total)

if not results:
    st.warning("沒有可用的分析結果。")
    st.stop()

# 顯示彙總表格
# -----------------------------
df_summary = pd.DataFrame([{
    '股票': r['ticker'],
    '現價': r['current_price'],
    '訊號': r['last_action'],
    '趨勢': r['trend'],
    '短線分數': r['short_score'],
    '中線分數': r['mid_score'],
    '長線分數': r['long_score'],
    '上漲機率(%)': round(r['prob_up']*100,2),
    '達標機率(%)': round(r['prob_target']*100,2)
} for r in results])

st.subheader("📋 分析彙總")
st.dataframe(df_summary.set_index('股票'))

# -----------------------------
# 雷達圖 & 條形圖
# -----------------------------
st.subheader("🧭 分數雷達圖 / 條形比較")
categories = ['短線分數','中線分數','長線分數']
N = len(categories)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

colors = list(mcolors.TABLEAU_COLORS.values())
fig_radar = plt.figure(figsize=(7,6))
ax = fig_radar.add_subplot(111, polar=True)
for idx, r in enumerate(results):
    vals = [r['short_score'], r['mid_score'], r['long_score']]
    vals = vals + vals[:1]
    ax.plot(angles, vals, label=r['ticker'], color=colors[idx % len(colors)], linewidth=2)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontproperties=myfont)
ax.set_title("股票分數雷達圖", fontproperties=myfont)
leg = ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.05))
if myfont:
    for txt in leg.get_texts():
        txt.set_fontproperties(myfont)
plt.tight_layout()
st.pyplot(fig_radar, clear_figure=True)

fig_bar, axb = plt.subplots(figsize=(8,4))
x = np.arange(N)
width = 0.8 / max(1, len(results))
for idx, r in enumerate(results):
    vals = [r['short_score'], r['mid_score'], r['long_score']]
    axb.bar(x + idx*width, vals, width=width, label=r['ticker'], color=colors[idx % len(colors)])
axb.set_xticks(x + width*(len(results)-1)/2)
axb.set_xticklabels(categories, fontproperties=myfont)
axb.set_ylabel('分數', fontproperties=myfont)
axb.set_title('短/中/長線分數比較', fontproperties=myfont)
axb.legend(prop=myfont, bbox_to_anchor=(1.05,1))
plt.tight_layout()
st.pyplot(fig_bar, clear_figure=True)

# -----------------------------
# 個股詳細圖
# -----------------------------
st.subheader("📊 個股詳細圖")
for r in results:
    ticker = r['ticker']
    data = r['data']
    st.markdown(f"### {ticker} — 現價: {r['current_price']:.2f}，訊號: **{r['last_action']}**，趨勢: **{r['trend']}**")
    cols = st.columns([2,1])
    with cols[1]:
        st.write(f"建議進場價: {r['entry_price']:.2f}")
        st.write(f"目標價: {r['target_price']:.2f}")
        st.write(f"停損價: {r['stop_loss']:.2f}")
        st.write(f"未來 {len(r['future_dates'])} 日預測趨勢: {r['future_trend']} ({r['future_change']:.2f}%)")
        st.write(f"蒙地卡羅上漲機率: {r['prob_up']:.2%}，達標機率: {r['prob_target']:.2%}")

    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(data.index, data['Close'], label='收盤價', color='black')
    ax.plot(data.index, data['MA20'], label='MA20', color='blue')
    ax.plot(data.index, data['MA60'], label='MA60', color='orange')
    ax.fill_between(data.index, data['BB_Lower'], data['BB_Upper'], color='gray', alpha=0.2, label='布林通道')
    ax.plot(r['future_dates'], r['y_future_only'], label='線性預測', color='green', linestyle='--')
    ax.fill_between(r['future_dates'], r['lower_band'], r['upper_band'], color='red', alpha=0.1, label='蒙地卡羅 95% CI')
    ax.set_title(f"{ticker} 技術指標與預測", fontproperties=myfont)
    ax.legend(prop=myfont)
    plt.xticks(rotation=20)
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)
