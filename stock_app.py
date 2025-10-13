import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from math import pi

st.set_page_config(page_title="?? �Ѳ��޳N���R App", layout="wide")
st.title("?? �Ѳ��޳N���R�P�Ͷչw��")

# --------------------------------
# ?? �W�ǪѲ� CSV ��
# --------------------------------
st.sidebar.header("?? �Ѳ��M��]�w")
uploaded_file = st.sidebar.file_uploader("�W�� CSV�]�t�Ѳ��N�X��^", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if len(df.columns) > 0:
        tickers = df[df.columns[0]].dropna().astype(str).tolist()
        st.sidebar.success(f"�w���J {len(tickers)} �ɪѲ�")
    else:
        st.sidebar.warning("CSV �ɵL�k�ѧO�A��ιw�]�Ѳ��M��")
        tickers = ['1436.TW', '1504.TW', '3231.TW', 'AAPL', 'MSFT']
else:
    tickers = ['1436.TW', '1504.TW', '3231.TW', 'AAPL', 'MSFT']

# --------------------------------
# ?? �ѼƳ]�w
# --------------------------------
lookback = st.sidebar.slider("�^���Ѽ�", 100, 1000, 365)
horizon = st.sidebar.slider("�w���Ѽ�", 5, 60, 20)
target = st.sidebar.slider("�ؼг��S�v(%)", 1, 20, 5) / 100
monte_carlo_sim = st.sidebar.number_input("�X�a�dù��������", 100, 5000, 1000)

# --------------------------------
# ?? �޳N���R�D���
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
# ?? ������R
# --------------------------------
if st.button("�}�l���R"):
    for t in tickers:
        st.divider()
        st.subheader(f"?? {t} �޳N���R���G")

        result = stock_analysis(t)
        if not result:
            st.warning(f"{t} �L�k���o���")
            continue

        data, y_future, current_price, volatility = result

        # ---- ?? �Ϫ� ----
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(data['Close'], label="���L��", color="black")
        ax.plot(data['MA20'], label="MA20", color="blue")
        ax.plot(data['MA60'], label="MA60", color="orange")
        ax.set_title(f"{t} �ѻ�����")
        ax.legend()
        st.pyplot(fig)

        # ---- ?? ��� ----
        st.write(f"?? �ثe����G{current_price:.2f}")
        st.write(f"?? �i�ʲv�G{volatility:.2%}")
