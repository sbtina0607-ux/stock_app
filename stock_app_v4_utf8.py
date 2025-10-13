import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.set_page_config(page_title="📊 股票技術分析 App", layout="wide")
st.title("📈 股票技術分析與技術指標展示")

st.write("請從下方選擇要分析的股票代碼（可多選或自行輸入）")

# ------------------------------
# 股票選取
# ------------------------------

# 常見台股
taiwan_stocks = [
    '2330.TW','2317.TW','2454.TW','0050.TW','2303.TW','2412.TW','2881.TW','2882.TW','1301.TW','1101.TW'
]

# 常見美股
us_stocks = [
    'AAPL','MSFT','GOOG','AMZN','TSLA','NVDA','META','NFLX','INTC','AMD'
]

# 將台股+美股合併成選項
all_stocks = taiwan_stocks + us_stocks

tickers = st.multiselect(
    "選擇或輸入股票代碼：",
    options=all_stocks,
    default=['2330.TW', 'AAPL'],
    help="可手動輸入其他股票代碼（例如：2454.TW 或 GOOG）"
)

# ------------------------------
# 側邊欄參數
# ------------------------------
st.sidebar.header("⚙️ 分析參數設定")
lookback = st.sidebar.slider("回溯天數", 100, 1000, 365)
horizon = st.sidebar.slider("預測天數", 5, 60, 20)
target = st.sidebar.slider("目標報酬率(%)", 1, 20, 5) / 100
monte_carlo_sim = st.sidebar.number_input("蒙地卡羅模擬次數", 100, 5000, 1000)

# ------------------------------
# 技術分析函數
# ------------------------------
def stock_analysis(ticker):
    data = yf.download(ticker, period=f"{lookback}d", auto_adjust=True)
    if data.empty:
        return None
    
    # 均線
    data['MA20'] = data['Close'].rolling(20).mean()
    data['MA60'] = data['Close'].rolling(60).mean()

    # RSI
    delta = data['Close'].diff()
    up = delta.clip(lower=0)
    down = -1*delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    RS = roll_up / roll_down
    data['RSI'] = 100 - (100 / (1 + RS))

    # MACD
    exp12 = data['Close'].ewm(span=12, adjust=False).mean()
    exp26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp12 - exp26
    data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

    # 布林通道
    data['BB_Mid'] = data['Close'].rolling(20).mean()
    data['BB_Std'] = data['Close'].rolling(20).std()
    data['BB_Upper'] = data['BB_Mid'] + 2 * data['BB_Std']
    data['BB_Lower'] = data['BB_Mid'] - 2 * data['BB_Std']

    # 收盤價 & 波動率
    current_price = float(data['Close'].iloc[-1])
    volatility = float(data['Close'].pct_change().std() * np.sqrt(252))

    # 線性趨勢預測
    recent = data['Close'].iloc[-60:]
    x = np.arange(len(recent)).reshape(-1, 1)
    model = LinearRegression().fit(x, recent)
    x_future = np.arange(len(recent)+horizon).reshape(-1, 1)
    y_future = model.predict(x_future)

    return data, y_future, current_price, volatility

# ------------------------------
# 執行分析
# ------------------------------
if st.button("開始分析"):
    if not tickers:
        st.warning("請至少選擇一檔股票！")
    else:
        for t in tickers:
            st.divider()
            st.subheader(f"📊 {t} 技術分析結果")

            result = stock_analysis(t)
            if not result:
                st.warning(f"{t} 無法取得資料")
                continue

            data, y_future, current_price, volatility = result

            # ---- 股價與均線圖 ----
            fig, ax = plt.subplots(figsize=(10,5))
            ax.plot(data['Close'], label="收盤價", color="black")
            ax.plot(data['MA20'], label="MA20", color="blue")
            ax.plot(data['MA60'], label="MA60", color="orange")
            ax.plot(data['BB_Upper'], '--', color='red', label='布林上軌')
            ax.plot(data['BB_Lower'], '--', color='green', label='布林下軌')
            ax.set_title(f"{t} 股價走勢與布林通道")
            ax.legend()
            st.pyplot(fig)

            # ---- RSI & MACD ----
            fig2, ax2 = plt.subplots(2,1, figsize=(10,6), sharex=True)
            ax2[0].plot(data['RSI'], color='purple', label='RSI')
            ax2[0].axhline(70, color='red', linestyle='--')
            ax2[0].axhline(30, color='green', linestyle='--')
            ax2[0].set_ylabel('RSI')
            ax2[0].legend()
            ax2[1].plot(data['MACD'], label='MACD', color='blue')
            ax2[1].plot(data['Signal'], label='Signal', color='orange')
            ax2[1].set_ylabel('MACD')
            ax2[1].legend()
            st.pyplot(fig2)

            # ---- 資料 ----
            st.write(f"💰 目前價格：{current_price:.2f}")
            st.write(f"📈 波動率：{volatility:.2%}")
