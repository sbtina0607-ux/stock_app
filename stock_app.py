import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from math import pi

st.set_page_config(page_title="?? 股票技術分析 App", layout="wide")
st.title("?? 股票技術分析與趨勢預測")

# --------------------------------
# ?? 上傳股票 CSV 檔
# --------------------------------
st.sidebar.header("?? 股票清單設定")
uploaded_file = st.sidebar.file_uploader("上傳 CSV（含股票代碼欄）", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if len(df.columns) > 0:
        tickers = df[df.columns[0]].dropna().astype(str).tolist()
        st.sidebar.success(f"已載入 {len(tickers)} 檔股票")
    else:
        st.sidebar.warning("CSV 檔無法識別，改用預設股票清單")
        tickers = ['1436.TW', '1504.TW', '3231.TW', 'AAPL', 'MSFT']
else:
    tickers = ['1436.TW', '1504.TW', '3231.TW', 'AAPL', 'MSFT']

# --------------------------------
# ?? 參數設定
# --------------------------------
lookback = st.sidebar.slider("回溯天數", 100, 1000, 365)
horizon = st.sidebar.slider("預測天數", 5, 60, 20)
target = st.sidebar.slider("目標報酬率(%)", 1, 20, 5) / 100
monte_carlo_sim = st.sidebar.number_input("蒙地卡羅模擬次數", 100, 5000, 1000)

# --------------------------------
# ?? 技術分析主函數
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

    recent = data['Close'].iloc[-60:]
    x = np.arange(len(recent)).reshape(-1, 1)
    model = LinearRegression().fit(x, recent)
    x_future = np.arange(len(recent) + horizon).reshape(-1, 1)
    y_future = model.predict(x_future)

    return data, y_future, current_price, volatility

# --------------------------------
# ?? 執行分析
# --------------------------------
if st.button("開始分析"):
    for t in tickers:
        st.divider()
        st.subheader(f"?? {t} 技術分析結果")

        result = stock_analysis(t)
        if not result:
            st.warning(f"{t} 無法取得資料")
            continue

        data, y_future, current_price, volatility = result

        # ---- ?? 圖表 ----
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(data['Close'], label="收盤價", color="black")
        ax.plot(data['MA20'], label="MA20", color="blue")
        ax.plot(data['MA60'], label="MA60", color="orange")
        ax.set_title(f"{t} 股價走勢")
        ax.legend()
        st.pyplot(fig)

        # ---- ?? 資料 ----
        st.write(f"?? 目前價格：{current_price:.2f}")
        st.write(f"?? 波動率：{volatility:.2%}")
