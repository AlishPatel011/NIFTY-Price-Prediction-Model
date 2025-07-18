import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# === Load Data ===
def load_data(path):
    """Loads data from a CSV or Excel file."""
    df = pd.read_excel(path) if path.endswith('.xlsx') else pd.read_csv(path)
    df.columns = [c.strip().capitalize() for c in df.columns]
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(subset=['Date'], inplace=True)
    df.sort_values('Date', inplace=True)
    return df.reset_index(drop=True)

# === Clean & Reindex ===
def clean_data(df):
    """Cleans data by handling NaNs, duplicates, and missing business days."""
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
    df = df.drop_duplicates(subset=['Date'])
    all_days = pd.date_range(df['Date'].min(), df['Date'].max(), freq='B')
    df = df.set_index('Date').reindex(all_days).rename_axis('Date').reset_index()
    df[['Open', 'High', 'Low', 'Close']] = df[['Open', 'High', 'Low', 'Close']].ffill()
    return df

# === Feature Engineering ===
def create_features(df):
    """Creates technical indicators and time-based features."""
    for lag in range(1, 6):
        df[f'lag_{lag}'] = df['Close'].shift(lag)
    df['RSI_14'] = RSIIndicator(close=df['Close'], window=14).rsi()
    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['EMA_20'] = EMAIndicator(close=df['Close'], window=20).ema_indicator()
    df['BB_width'] = BollingerBands(close=df['Close'], window=20).bollinger_wband()
    df['Volatility_10'] = df['Close'].rolling(10).std()
    df['Daily_return'] = df['Close'].diff()
    df['Pct_change'] = df['Close'].pct_change()
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    return df

# === Weekly Aggregation ===
def create_weekly_data(df):
    """Aggregates daily data into weekly OHLC data and adds features."""
    df_weekly = df.groupby(pd.Grouper(key='Date', freq='W-MON')).agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
    }).reset_index()
    df_weekly.dropna(subset=['Close'], inplace=True)
    df_weekly = create_features(df_weekly)
    df_weekly['Target'] = df_weekly['Close'].diff().shift(-1)
    return df_weekly.dropna().reset_index(drop=True)

# === Monthly Aggregation ===
def create_monthly_data(df):
    """Aggregates daily data into monthly OHLC data and adds features."""
    df_monthly = df.groupby(pd.Grouper(key='Date', freq='MS')).agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
    }).reset_index()
    df_monthly = create_features(df_monthly)
    df_monthly['Target'] = df_monthly['Close'].diff().shift(-1)
    return df_monthly.dropna().reset_index(drop=True)

# === Train and Predict Function ===
def train_and_predict(df, title_suffix=""):
    """Trains models on price *changes* and evaluates on absolute prices."""
    if len(df) < 10:
        print(f"\n❌ Insufficient data for {title_suffix} predictions.")
        return None, None, None
    features = [
        'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'RSI_14', 'MACD',
        'MACD_signal', 'EMA_20', 'BB_width', 'Volatility_10', 'Daily_return',
        'Pct_change', 'DayOfWeek', 'Month', 'Quarter'
    ]
    X = df[features]; y = df['Target']
    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=test_size)
    xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=5)
    xgb_model.fit(X_train, y_train)
    predicted_change_xgb = xgb_model.predict(X_test)
    base_price = df.loc[X_test.index, 'Close']
    predicted_price_xgb = base_price + predicted_change_xgb
    actual_price = df.loc[y_test.index, 'Close']
    df_compare = pd.DataFrame({
        'Date': df.loc[y_test.index, 'Date'], 'Actual_Close': actual_price,
        'XGB_Predicted_Close': predicted_price_xgb,
    })
    rmse_xgb = np.sqrt(mean_squared_error(actual_price, predicted_price_xgb))
    print(f"\n{title_suffix} Predictions:"); print(f"✅ XGBoost RMSE (on absolute price): {rmse_xgb:.2f}")
    return df_compare, rmse_xgb, xgb_model

# === Future Prediction Function ===
def predict_future(model, df, periods, freq='D'):
    """Generates iterative future predictions based on price changes."""
    if model is None: return [], []
    last_row = df.iloc[-1:].copy()
    predictions, future_dates = [], []
    features = [
        'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'RSI_14', 'MACD',
        'MACD_signal', 'EMA_20', 'BB_width', 'Volatility_10', 'Daily_return',
        'Pct_change', 'DayOfWeek', 'Month', 'Quarter'
    ]
    current_date = df['Date'].iloc[-1]
    for _ in range(periods):
        X_pred = last_row[features]
        predicted_change = model.predict(X_pred)[0]
        last_known_price = last_row['Close'].iloc[0]
        new_predicted_price = last_known_price + predicted_change
        if freq == 'D': current_date += timedelta(days=1)
        elif freq == 'W': current_date += timedelta(weeks=1)
        elif freq == 'M': current_date += pd.DateOffset(months=1)
        predictions.append(new_predicted_price); future_dates.append(current_date)
        next_row = last_row.copy()
        next_row['Close'] = new_predicted_price
        for lag in range(5, 1, -1): next_row[f'lag_{lag}'] = next_row[f'lag_{lag-1}']
        next_row['lag_1'] = last_known_price; next_row['Date'] = current_date
        next_row['Daily_return'] = predicted_change
        next_row['Pct_change'] = (predicted_change / last_known_price) if last_known_price != 0 else 0
        next_row['DayOfWeek'] = next_row['Date'].dt.dayofweek
        next_row['Month'] = next_row['Date'].dt.month; next_row['Quarter'] = next_row['Date'].dt.quarter
        last_row = next_row
    return future_dates, predictions

# === Main Execution ===
# V V V --- IMPORTANT: Make sure this path is correct! --- V V V
df = load_data('sample_data/nifty_historical_data.csv')
df = clean_data(df); df = create_features(df)
df['Target'] = df['Close'].diff().shift(-1)
df = df.dropna().reset_index(drop=True)

df_weekly = create_weekly_data(df.copy())
df_monthly = create_monthly_data(df.copy())

# === Run Predictions ===
df_daily_compare, _, xgb_daily = train_and_predict(df, "Daily")
df_weekly_compare, _, xgb_weekly = train_and_predict(df_weekly, "Weekly")
df_monthly_compare, _, xgb_monthly = train_and_predict(df_monthly, "Monthly")

# === Generate Future Predictions ===
daily_future_dates, daily_future_preds = predict_future(xgb_daily, df, 5, 'D')
weekly_future_dates, weekly_future_preds = predict_future(xgb_weekly, df_weekly, 4, 'W')
monthly_future_dates, monthly_future_preds = predict_future(xgb_monthly, df_monthly, 3, 'M')

# === PLOTTING with INTERACTIVE CHARTS ===
print("\n" + "=" * 60); print("📈 PLOTTING INTERACTIVE PREDICTIONS AND FORECASTS"); print("=" * 60)
plot_dfs = [df_daily_compare, df_weekly_compare, df_monthly_compare]
num_plots = sum(x is not None for x in plot_dfs)

if num_plots > 0:
    subplot_titles = []
    if df_daily_compare is not None: subplot_titles.append("NIFTY Daily Price and Forecast")
    if df_weekly_compare is not None: subplot_titles.append("NIFTY Weekly Price and Forecast")
    if df_monthly_compare is not None: subplot_titles.append("NIFTY Monthly Price and Forecast")

    fig = make_subplots(rows=num_plots, cols=1, subplot_titles=subplot_titles, vertical_spacing=0.08)
    plot_row = 1

    # --- Daily Interactive Plot ---
    if df_daily_compare is not None:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Historical Close', legendgroup='1', line=dict(color='gray', width=1)), row=plot_row, col=1)
        fig.add_trace(go.Scatter(x=df_daily_compare['Date'], y=df_daily_compare['Actual_Close'], name='Actual Test Data', legendgroup='1', line=dict(color='black', width=2)), row=plot_row, col=1)
        fig.add_trace(go.Scatter(x=df_daily_compare['Date'], y=df_daily_compare['XGB_Predicted_Close'], name='Model Prediction', legendgroup='1', line=dict(color='orange', width=2)), row=plot_row, col=1)
        fig.add_trace(go.Scatter(x=daily_future_dates, y=daily_future_preds, name='Future Forecast', legendgroup='1', line=dict(color='red', dash='dash')), row=plot_row, col=1)
        plot_row += 1

    # --- Weekly Interactive Plot ---
    if df_weekly_compare is not None:
        fig.add_trace(go.Scatter(x=df_weekly['Date'], y=df_weekly['Close'], name='Historical Weekly', legendgroup='2', showlegend=False, line=dict(color='gray', width=1)), row=plot_row, col=1)
        fig.add_trace(go.Scatter(x=df_weekly_compare['Date'], y=df_weekly_compare['Actual_Close'], name='Actual Weekly Test', legendgroup='2', showlegend=False, line=dict(color='black', width=2)), row=plot_row, col=1)
        fig.add_trace(go.Scatter(x=df_weekly_compare['Date'], y=df_weekly_compare['XGB_Predicted_Close'], name='Weekly Prediction', legendgroup='2', showlegend=False, line=dict(color='orange', width=2)), row=plot_row, col=1)
        fig.add_trace(go.Scatter(x=weekly_future_dates, y=weekly_future_preds, name='Weekly Forecast', legendgroup='2', showlegend=False, line=dict(color='red', dash='dash')), row=plot_row, col=1)
        plot_row += 1

    # --- Monthly Interactive Plot ---
    if df_monthly_compare is not None:
        fig.add_trace(go.Scatter(x=df_monthly['Date'], y=df_monthly['Close'], name='Historical Monthly', legendgroup='3', showlegend=False, line=dict(color='gray', width=1)), row=plot_row, col=1)
        fig.add_trace(go.Scatter(x=df_monthly_compare['Date'], y=df_monthly_compare['Actual_Close'], name='Actual Monthly Test', legendgroup='3', showlegend=False, line=dict(color='black', width=2)), row=plot_row, col=1)
        fig.add_trace(go.Scatter(x=df_monthly_compare['Date'], y=df_monthly_compare['XGB_Predicted_Close'], name='Monthly Prediction', legendgroup='3', showlegend=False, line=dict(color='orange', width=2)), row=plot_row, col=1)
        fig.add_trace(go.Scatter(x=monthly_future_dates, y=monthly_future_preds, name='Monthly Forecast', legendgroup='3', showlegend=False, line=dict(color='red', dash='dash')), row=plot_row, col=1)

    fig.update_layout(height=350 * num_plots, title_text="NIFTY Price Predictions (Interactive)", template="plotly_white")
    fig.show()