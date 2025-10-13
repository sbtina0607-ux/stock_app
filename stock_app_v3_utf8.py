import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# --------------------------------
# 📋 App 基本設定
# --------------------------------
st.set_page_config(page_title="📊 股票技術分析 App", layout="wide")
st.title("📈 股票技術分析與趨勢預測")

st.write("請從下方選擇要分析的股票代碼（可多選或自行輸入）")

# --------------------------------
# 📊 股票清單選擇
# --------------------------------
default_tickers = ['2330.TW', '2317.TW', '0050.TW', 'AAPL', 'MSFT', 'NVDA', 'TSLA']
tickers = st.multiselect(
    "選擇或輸入股票代碼：",
    options=default_tickers,
    default=['2330.TW', 'AAPL'],
    help="可手動輸入其他股票代碼（例如：2454.TW 或 GOOG）"
)

# --------------------------------
# ⚙️ 參數設定（側邊欄）
# --------------------------------
st.sidebar.header("⚙️ 分析參數設定")
lookback = st.sidebar.slider("回溯天數", 100, 1000, 365)
horizon = st.sidebar.slider("預測天數", 5, 60, 20)
target = st.sidebar.slider("目標報酬率(%)", 1, 20, 5) / 100
monte_carlo_sim = st.sidebar.number_input("蒙地卡羅模擬次數", 100, 5000, 1000)

# --------------------------------
# 🧮 技術分析主函數
# --------------------------------
def stock_analysis(ticker):
    data = yf.download(ticker, period=f"{lookback}d", auto_adjust=True)
    if data.empty:
        return None
    data['MA20'] = data['Close'].rolling(20).mean()
    data['MA60'] = data['Close'].rolling(60).mean()
    data['Returns'] = data['Close'].pct_change()
    current_price = data['Close'].iloc[-1]
    volatility = data['Returns'].std() * np.sqrt(252)

    # 線性趨勢預測
    recent = data['Close'].iloc[-60:]
    x = np.arange(len(recent)).reshape(-1, 1)
    model = LinearRegression().fit(x, recent)
    x_future = np.arange(len(recent) + horizon).reshape(-1, 1)
    y_future = model.predict(x_future)

    return data, y_future, current_price, volatility

# --------------------------------
# 🚀 開始分析按鈕
# --------------------------------
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

            # ---- 📉 圖表 ----
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(data['Close'], label="收盤價", color="black")
            ax.plot(data['MA20'], label="MA20", color="blue")
            ax.plot(data['MA60'], label="MA60", color="orange")
            ax.set_title(f"{t} 股價走勢")
            ax.legend()
            st.pyplot(fig)

            # ---- 📄 資料 ----
            st.write(f"💰 目前價格：{current_price:.2f}")
            st.write(f"📈 波動率：{volatility:.2%}")
