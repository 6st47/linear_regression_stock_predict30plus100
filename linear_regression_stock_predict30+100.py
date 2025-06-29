import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

def calc_technical_indicators(close):
    df = pd.DataFrame(index=close.index)
    df['Close'] = close
    df['SMA_20'] = close.rolling(window=20).mean()
    df['SMA_50'] = close.rolling(window=50).mean()
    df['RSI_14'] = RSIIndicator(close, window=14).rsi()
    macd = MACD(close)
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    return df.dropna()

def predict_stock(ticker):
    data = yf.download(ticker, period="2y", auto_adjust=True).dropna()
    close = pd.Series(data['Close'].values.flatten(), index=data.index)

    data_ind = calc_technical_indicators(close)
    data_ind['Close_next'] = data_ind['Close'].shift(-1)
    data_ind.dropna(inplace=True)

    test_size = 30
    train_data = data_ind.iloc[:-test_size]
    test_data = data_ind.iloc[-test_size:]

    features = ['Close', 'SMA_20', 'SMA_50', 'RSI_14', 'MACD', 'MACD_signal']
    X_train = train_data[features]
    y_train = train_data['Close_next']
    X_test = test_data[features]
    y_test = test_data['Close_next']

    model = LinearRegression()
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    mae = mean_absolute_error(y_test, test_pred)
    mape = np.mean(np.abs((y_test - test_pred) / y_test)) * 100

    # 未来100日予測
    future_days = 100
    last_known_data = data_ind.iloc[-1][features].values.reshape(1, -1)
    future_preds = []
    future_dates = pd.date_range(start=data_ind.index[-1] + pd.Timedelta(days=1), periods=future_days)

    close_history = list(data_ind['Close'][-50:])

    for _ in range(future_days):
        pred = model.predict(last_known_data)[0]
        future_preds.append(pred)

        close_history.append(pred)
        close_series = pd.Series(close_history[-50:])

        sma_20 = close_series.rolling(window=20).mean().iloc[-1]
        sma_50 = close_series.rolling(window=50).mean().iloc[-1]

        delta = close_series.diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean().iloc[-1]
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean().iloc[-1]
        rs = gain / loss if loss != 0 else 0
        rsi_14 = 100 - (100 / (1 + rs)) if loss != 0 else 100

        ema_12 = close_series.ewm(span=12, adjust=False).mean().iloc[-1]
        ema_26 = close_series.ewm(span=26, adjust=False).mean().iloc[-1]
        macd_val = ema_12 - ema_26
        macd_signal = pd.Series(close_history[-9:]).ewm(span=9, adjust=False).mean().iloc[-1]

        last_known_data = np.array([[pred, sma_20, sma_50, rsi_14, macd_val, macd_signal]])

    return train_data, train_pred, test_data, y_test, test_pred, rmse, mae, mape, future_dates, future_preds

st.title("Stock Price Prediction (Linear Regression) - 2 Years Training + 30 Days Test + 100 Days Future Forecast")

ticker = st.text_input("Enter ticker symbol (e.g. 5411.T)", value="5411.T")

if ticker:
    try:
        train_data, train_pred, test_data, y_test, test_pred, rmse, mae, mape, future_dates, future_preds = predict_stock(ticker.strip().upper())

        st.write(f"Performance metrics for {ticker}:")
        st.write(f"- RMSE: {rmse:.2f}")
        st.write(f"- MAE: {mae:.2f}")
        st.write(f"- MAPE: {mape:.2f}%")

        # グラフ描画
        import matplotlib.dates as mdates
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(train_data.index, train_data['Close_next'], label='Training Actual', color='blue')
        ax.plot(train_data.index, train_pred, label='Training Predicted', color='blue', linestyle='--')
        ax.plot(test_data.index, y_test, label='Test Actual', color='orange')
        ax.plot(test_data.index, test_pred, label='Test Predicted', color='orange', linestyle='--')
        ax.plot(future_dates, future_preds, label='Future 100-day Prediction', color='green', linestyle='-')

        ax.set_title(f"{ticker} - Actual vs Predicted Close Price with Future Forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Close Price (JPY)")
        ax.legend()
        ax.grid(True)

        # x軸の日付フォーマット
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        fig.autofmt_xdate()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error: {e}")
