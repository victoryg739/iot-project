from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
import os
import json
from datetime import datetime, timedelta
from io import BytesIO
import base64
import warnings
import traceback
warnings.filterwarnings('ignore') 
app = Flask(__name__)

np.random.seed(42)

os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True) 

# --- Constants ---
FEATURES = [
    'Close', 'Price_lag_1', 'Price_lag_2', 
    
    'Log_Return_1d', 'Log_Return_5d',
    
    'OBV_change', 'VWAP',
    
    'SMA20_50_Crossover', 'EMA_Crossover',
    
    'BB_width',
    
    'TDI_Volatility',
    
    'ADX'
]

TARGET_5D = 'Future_Price_5d'
TARGET_20D = 'Future_Price_20d'

def flatten_yf_columns(df):
    """Flattens MultiIndex columns from yfinance if present."""
    if isinstance(df.columns, pd.MultiIndex):
        print("Detected MultiIndex columns. Flattening.")
        df.columns = df.columns.get_level_values(0)
    return df

def engineer_features(data):
    """Adds optimized technical indicators to the DataFrame.
       Optimized version with removed weak indicators and added strong ones.
    """
    df = data.copy()
    
    if 'Close' not in df.columns or 'Volume' not in df.columns:
        print("Warning: Input data missing 'Close' or 'Volume'.")
        if 'Close' not in df.columns: df['Close'] = np.nan
        if 'Volume' not in df.columns: df['Volume'] = np.nan
    
    df['SMA20'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['SMA50'] = df['Close'].rolling(window=50, min_periods=1).mean()

    df['Price_lag_1'] = df['Close'].shift(1)
    df['Price_lag_2'] = df['Close'].shift(2)
    
    df['Log_Return_1d'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Log_Return_5d'] = np.log(df['Close'] / df['Close'].shift(5))
    
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    df['SMA20_50_Crossover'] = np.where(df['SMA20'] > df['SMA50'], 1, -1)
    df['EMA_Crossover'] = np.where(df['EMA12'] > df['EMA26'], 1, -1)
    
    if 'Volume' in df.columns and not df['Volume'].isnull().all():
        df['OBV'] = 0
        df['Price_direction'] = np.sign(df['Close'].diff())
        df.loc[df['Price_direction'] == 0, 'Price_direction'] = 1
        
        df['Volume_direction'] = df['Volume'] * df['Price_direction']
        df['OBV'] = df['Volume_direction'].cumsum()
        
        df['OBV_change'] = df['OBV'].pct_change()
        
        df.drop(columns=['Price_direction', 'Volume_direction'], inplace=True)
    else:
        df['OBV'] = np.nan
        df['OBV_change'] = np.nan
    
    if 'Volume' in df.columns and not df['Volume'].isnull().all() and not df['Volume'].empty:
        if 'High' in df.columns and 'Low' in df.columns:
            df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
        else:
            df['TP'] = df['Close']
            
        df['TP_Volume'] = df['TP'] * df['Volume']
        df['Volume_Sum'] = df['Volume'].expanding().sum()
        df['TP_Volume_Sum'] = df['TP_Volume'].expanding().sum()
        
        df['VWAP'] = df['TP_Volume_Sum'] / df['Volume_Sum'].replace(0, np.nan)
        
        df = df.drop(['TP', 'TP_Volume', 'Volume_Sum', 'TP_Volume_Sum'], axis=1, errors='ignore')
    else:
        df['VWAP'] = np.nan
    
    if 'High' in df.columns and 'Low' in df.columns:
        df['TR1'] = abs(df['High'] - df['Low'])
        df['TR2'] = abs(df['High'] - df['Close'].shift(1))
        df['TR3'] = abs(df['Low'] - df['Close'].shift(1))
        df['TR'] = df[['TR1', 'TR2', 'TR3']].max(axis=1)
        df['ATR'] = df['TR'].rolling(window=14, min_periods=1).mean()
        
        df = df.drop(['TR1', 'TR2', 'TR3', 'TR'], axis=1, errors='ignore')
    else:
        df['ATR'] = df['Close'].rolling(window=14, min_periods=1).std()
    
    if 'High' in df.columns and 'Low' in df.columns:
        df['High_diff'] = df['High'].diff()
        df['Low_diff'] = df['Low'].diff()
        
        df['+DM'] = np.where(
            (df['High_diff'] > 0) & (df['High_diff'] > df['Low_diff'].abs()),
            df['High_diff'],
            0
        )
        
        df['-DM'] = np.where(
            (df['Low_diff'] < 0) & (df['Low_diff'].abs() > df['High_diff']),
            df['Low_diff'].abs(),
            0
        )
        
        if 'ATR' not in df.columns:
            df['TR1'] = abs(df['High'] - df['Low'])
            df['TR2'] = abs(df['High'] - df['Close'].shift(1))
            df['TR3'] = abs(df['Low'] - df['Close'].shift(1))
            df['TR'] = df[['TR1', 'TR2', 'TR3']].max(axis=1)
            df['ATR'] = df['TR'].ewm(span=14, min_periods=1, adjust=False).mean()
        
        df['+DM14'] = df['+DM'].ewm(span=14, min_periods=1, adjust=False).mean()
        df['-DM14'] = df['-DM'].ewm(span=14, min_periods=1, adjust=False).mean()
        
        df['+DI'] = 100 * df['+DM14'] / df['ATR'].replace(0, np.nan)
        df['-DI'] = 100 * df['-DM14'] / df['ATR'].replace(0, np.nan)
        
        df['DI_diff'] = abs(df['+DI'] - df['-DI'])
        df['DI_sum'] = df['+DI'] + df['-DI']
        df['DX'] = 100 * df['DI_diff'] / df['DI_sum'].replace(0, np.nan)
        
        df['ADX'] = df['DX'].ewm(span=14, min_periods=1, adjust=False).mean()
        
        columns_to_drop = ['High_diff', 'Low_diff', '+DM', '-DM', '+DM14', '-DM14', 
                          '+DI', '-DI', 'DI_diff', 'DI_sum', 'DX']
        df = df.drop(columns_to_drop, axis=1, errors='ignore')
    else:
        df['ADX'] = df['Close'].pct_change().abs().rolling(window=14).mean() * 100
    
    rsi_period = 13
    rsi_smoothing = 2
    bb_period = 34
    
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=rsi_period, min_periods=1).mean()
    avg_loss = loss.rolling(window=rsi_period, min_periods=1).mean()
    
    avg_loss_nonzero = avg_loss.replace(0, np.nan)
    rs = avg_gain / avg_loss_nonzero
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)
    
    smoothed_rsi = rsi.rolling(window=rsi_smoothing).mean()
    tdi_std = smoothed_rsi.rolling(window=bb_period).std()
    
    df['TDI_Volatility'] = tdi_std.rolling(window=5).mean()
    
    rolling_mean = df['Close'].rolling(window=20, min_periods=1).mean()
    rolling_std = df['Close'].rolling(window=20, min_periods=1).std()
    df['BB_upper'] = rolling_mean + (rolling_std * 2)
    df['BB_lower'] = rolling_mean - (rolling_std * 2)
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / rolling_mean
    
    df['Future_Price_5d'] = df['Close'].shift(-5)
    df['Future_Price_20d'] = df['Close'].shift(-20)
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    necessary_columns = FEATURES + [TARGET_5D, TARGET_20D]
    df = df[necessary_columns]
    
    return df


def train_model(ticker):
    """Downloads data, flattens columns, engineers features, cleans data, and trains a Random Forest model."""
    print(f"Training new model for {ticker}...")

    # Download historical data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1825) # 5 years of data
    data_raw_orig = yf.download(ticker, start=start_date, end=end_date, progress=False)

    if data_raw_orig.empty:
         raise ValueError(f"No historical data downloaded for {ticker}")

    data_raw = flatten_yf_columns(data_raw_orig.copy()) # Work on a copy

    if len(data_raw) < 300:
        raise ValueError(f"Insufficient historical data points ({len(data_raw)}) for {ticker} after potential flattening")

    data_featured = engineer_features(data_raw)

    required_columns_train = FEATURES + [TARGET_5D]

    print("Columns available in data_featured (train):", data_featured.columns.tolist())
    print("Columns required for dropna (train):", required_columns_train)
    missing_cols_train = [col for col in required_columns_train if col not in data_featured.columns]
    if missing_cols_train:
        print(f"!!! CRITICAL (train): Missing columns before dropna: {missing_cols_train}")
        raise ValueError(f"Feature engineering failed to produce required columns: {missing_cols_train}")

    original_len = len(data_featured)
    data_clean = data_featured.dropna(subset=required_columns_train).copy()
    cleaned_len = len(data_clean)
    print(f"Data points before NaN drop: {original_len}, after: {cleaned_len}")

    if cleaned_len < 100: 
        raise ValueError(f"Insufficient data remaining for {ticker} after cleaning ({cleaned_len} points)")

    X = data_clean[FEATURES].values
    y_5d = data_clean[TARGET_5D].values

    print(f"Shape of X: {X.shape}, Shape of y_5d: {y_5d.shape}")
    assert X.shape[0] == y_5d.shape[0], f"Shape mismatch after cleaning: X({X.shape[0]}) != y_5d({y_5d.shape[0]})"

    price_scaler = MinMaxScaler(feature_range=(0, 1))
    price_scaler.fit(data_clean[['Close']])

    volume_scaler = MinMaxScaler(feature_range=(0, 1))
    if 'Volume' in data_clean.columns and not data_clean['Volume'].isnull().all():
        volume_scaler.fit(data_clean[['Volume']])
    else:
        volume_scaler = None
        print("Warning: Volume data not available or all NaN, Volume scaler not fitted.")

    X_train, X_test, y_train_5d, y_test_5d = train_test_split(X, y_5d, test_size=0.2, shuffle=False)

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train_5d)

    predictions_5d = model.predict(X_test)
    mse = np.mean(np.square(y_test_5d - predictions_5d))
    rmse = np.sqrt(mse)

    close_idx = FEATURES.index('Close')
    actual_direction = np.sign(y_test_5d - X_test[:, close_idx])
    predicted_direction = np.sign(predictions_5d - X_test[:, close_idx])
    valid_comparison = predicted_direction != 0
    if np.sum(valid_comparison) > 0:
         directional_accuracy = np.mean(actual_direction[valid_comparison] == predicted_direction[valid_comparison]) * 100
    else:
         directional_accuracy = 0.0

    feature_importances = dict(zip(FEATURES, model.feature_importances_))
    top_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:5]

    training_summary = {
        "mse": float(mse),
        "rmse": float(rmse),
        "directional_accuracy": float(directional_accuracy),
        "used_data_points_after_cleaning": cleaned_len,
        "top_features": dict(top_features),
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    return model, price_scaler, volume_scaler, training_summary, data_clean, data_raw


def get_latest_data_and_eval(ticker, model, price_scaler, volume_scaler):
    """Gets recent data, flattens columns, applies features, evaluates the model, returns summary and FLATTENED raw data."""
    print(f"Getting latest data and evaluating model for {ticker}...")
    # Download recent data (e.g., last year)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    data_raw_orig = yf.download(ticker, start=start_date, end=end_date, progress=False)

    if data_raw_orig.empty:
        raise ValueError(f"Could not download recent data for {ticker}")

    data_raw = flatten_yf_columns(data_raw_orig.copy()) 

    data_featured = engineer_features(data_raw)

    required_columns_eval = FEATURES + [TARGET_5D]

    print(f"Timestamp: {datetime.now()}")
    print(f"Ticker: {ticker}")
    print("Columns available in data_featured (eval):", data_featured.columns.tolist()) # Should be simple strings
    print("Columns required for dropna (eval):", required_columns_eval)

    missing_cols = [col for col in required_columns_eval if col not in data_featured.columns]

    if missing_cols:
        print(f"!!! CRITICAL (eval): Missing columns before dropna: {missing_cols}")
        print("\n--- Data Info (data_featured) ---")
        data_featured.info()
        print("\n--- Data Head (data_featured) ---")
        print(data_featured.head())
        print("\n--- Data Tail (data_featured) ---")
        print(data_featured.tail())
        print("--- End Debugging Info ---\n")
        raise KeyError(f"Evaluation failed: DataFrame is missing required columns after feature engineering: {missing_cols}")
    else:
        print("All required columns are present before dropna.")
        print("--- End Debugging Info ---\n")

    eval_data_clean = data_featured.dropna(subset=required_columns_eval).copy()

    training_summary = {
        "mse": None, "rmse": None, "directional_accuracy": None,
        "used_data_points": 0,
        "evaluation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "eval_status": "Insufficient clean data for evaluation"
    }

    if len(eval_data_clean) < 20:
        print(f"Warning: Insufficient clean data ({len(eval_data_clean)}) for evaluation of {ticker}. Only {len(eval_data_clean)} rows after dropna.")
        return training_summary, data_raw.dropna(subset=['Close']) # Ensure raw data has 'Close'

    X_eval = eval_data_clean[FEATURES].values
    y_eval_5d = eval_data_clean[TARGET_5D].values

    if X_eval.shape[0] != y_eval_5d.shape[0]:
         raise ValueError(f"Shape mismatch after cleaning in eval: X({X_eval.shape[0]}) != y({y_eval_5d.shape[0]})")

    predictions = model.predict(X_eval)

    mse = np.mean(np.square(y_eval_5d - predictions))
    rmse = np.sqrt(mse)

    close_idx = FEATURES.index('Close')
    actual_direction = np.sign(y_eval_5d - X_eval[:, close_idx])
    predicted_direction = np.sign(predictions - X_eval[:, close_idx])
    valid_comparison = predicted_direction != 0
    if np.sum(valid_comparison) > 0:
         directional_accuracy = np.mean(actual_direction[valid_comparison] == predicted_direction[valid_comparison]) * 100
    else:
         directional_accuracy = 0.0

    training_summary.update({
        "mse": float(mse),
        "rmse": float(rmse),
        "directional_accuracy": float(directional_accuracy),
        "used_data_points": len(eval_data_clean),
        "eval_status": "Evaluation successful"
    })

    return training_summary, data_raw.dropna(subset=['Close']) 


def generate_prediction(model, current_data_raw_flattened, price_scaler, volume_scaler, days=7):
    """Generates future price prediction using the model or linear regression.
       Expects FLATTENED current_data_raw.
    """
    print(f"Generating {days}-day prediction...")
    pred_data_raw = current_data_raw_flattened.copy().tail(60)

    if len(pred_data_raw) < 55:
         raise ValueError(f"Insufficient recent data ({len(pred_data_raw)} points) to calculate features for prediction.")

    pred_data_featured = engineer_features(pred_data_raw)

    print("\n--- Debugging Info in generate_prediction ---")
    print("Columns available in pred_data_featured:", pred_data_featured.columns.tolist())
    missing_pred_cols = [col for col in FEATURES if col not in pred_data_featured.columns]
    if missing_pred_cols:
        print(f"!!! WARNING (prediction): Missing base features before dropna: {missing_pred_cols}")
    else:
        print("All required features columns are present in pred_data_featured.")

    latest_row_features = pred_data_featured[FEATURES].iloc[-1]
    nans_in_latest = latest_row_features.isnull().sum()
    if nans_in_latest > 0:
        print(f"!!! WARNING (prediction): Latest feature row contains {nans_in_latest} NaN(s) BEFORE dropna:")
        print(latest_row_features[latest_row_features.isnull()])

    print("--- End Debugging Info ---\n")

    pred_data_clean = pred_data_featured.dropna(subset=FEATURES).copy()

    if pred_data_clean.empty:
        print("--- Error Details: pred_data_featured (before dropna) ---")
        print(pred_data_featured.tail(10))
        print(f"Required columns for dropna: {FEATURES}")
        raise ValueError("Could not generate features for prediction - final data point row is empty after dropping NaNs in required features.")


    latest_features = pred_data_clean[FEATURES].values[-1].reshape(1, -1)
    last_close_price = pred_data_clean['Close'].iloc[-1]

    if days <= 10:
        if np.isnan(latest_features).any():
             raise ValueError(f"NaN values detected in the latest features used for prediction: {latest_features}")
        prediction = model.predict(latest_features)[0]
        confidence = 0.75
        print(f"Using RF model for {days}-day prediction. Raw output: {prediction}")
    else:
        print(f"Using Linear Regression trend for {days}-day prediction.")
        recent_clean_data = pred_data_clean.tail(30)
        if len(recent_clean_data) < 5: 
            raise ValueError("Insufficient cleaned data points (<5) for linear regression trend.")

        X_lr = np.arange(len(recent_clean_data)).reshape(-1, 1)
        y_lr = recent_clean_data['Close'].values

        lr_model = LinearRegression()
        lr_model.fit(X_lr, y_lr)

        future_index = len(recent_clean_data) - 1 + days
        prediction = lr_model.predict(np.array([[future_index]]))[0]

        confidence = 0.6 if days <= 20 else 0.5
        print(f"Using LR model for {days}-day prediction. Output: {prediction}")

    if prediction < 0:
        print(f"Warning: Negative prediction ({prediction}) clipped to 0.")
        prediction = 0.0

    return prediction, confidence, last_close_price

def update_model_with_feedback(ticker, feedback_date, actual_price):
    """Updates the model for a ticker with new actual price data."""
    print(f"Updating model for {ticker} with feedback: date={feedback_date}, price={actual_price}")
    
    model_filename = f'models/{ticker}_rf_model.joblib'
    scaler_filename = f'models/{ticker}_scalers.joblib'
    
    if not os.path.exists(model_filename) or not os.path.exists(scaler_filename):
        raise ValueError(f"No existing model found for {ticker}. Cannot update.")
    
    model = joblib.load(model_filename)
    scalers = joblib.load(scaler_filename)
    price_scaler = scalers['price_scaler']
    volume_scaler = scalers.get('volume_scaler', None)
    
    try:
        feedback_date_dt = datetime.strptime(feedback_date, "%Y-%m-%d")
        
        start_date = feedback_date_dt - timedelta(days=60)
        end_date = feedback_date_dt + timedelta(days=1)
        
        update_data_raw_orig = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if update_data_raw_orig.empty:
            raise ValueError(f"Could not download historical data for {ticker} around feedback date")
        
        update_data_raw = flatten_yf_columns(update_data_raw_orig.copy())
        
        update_data_featured = engineer_features(update_data_raw)
        
        exact_date = feedback_date_dt.strftime("%Y-%m-%d")
        
        date_exists = False
        for idx, date in enumerate(update_data_featured.index):
            if date.strftime("%Y-%m-%d") == exact_date:
                date_exists = True
                update_idx = idx
                break
                
        if not date_exists:
            closest_date = None
            closest_diff = float('inf')
            
            for idx, date in enumerate(update_data_featured.index):
                date_diff = abs((date - feedback_date_dt).total_seconds())
                if date_diff < closest_diff:
                    closest_diff = date_diff
                    closest_date = date
                    update_idx = idx
                    
            if closest_date is None:
                raise ValueError(f"No data found on or before {feedback_date} for {ticker}")
                
            print(f"Exact date {exact_date} not found, using closest date {closest_date.strftime('%Y-%m-%d')}")
        
        update_data_featured.iloc[update_idx, update_data_featured.columns.get_loc('Close')] = actual_price
        
        modified_data = engineer_features(update_data_featured)
        
        required_columns = FEATURES + [TARGET_5D]
        train_data = modified_data.dropna(subset=required_columns).copy()
        
        if len(train_data) < 5:
            raise ValueError(f"Insufficient data points ({len(train_data)}) after cleaning for model update")
        
        X_update = train_data[FEATURES].values
        y_update_5d = train_data[TARGET_5D].values
        
        model.fit(X_update, y_update_5d)
        
        joblib.dump(model, model_filename)
        
        return {
            "status": "Model updated successfully",
            "ticker": ticker,
            "feedback_date": feedback_date,
            "data_points_used": len(train_data)
        }
        
    except Exception as e:
        print(f"Error updating model: {str(e)}")
        traceback.print_exc()
        raise ValueError(f"Failed to update model: {str(e)}")

def get_recommendation(price_change_percent, confidence):
    """Determines a trading recommendation based on predicted change and confidence."""
    if price_change_percent is None or np.isnan(price_change_percent) or \
       confidence is None or np.isnan(confidence):
        return "HOLD" 

    abs_change = abs(price_change_percent)

    if confidence < 0.55: 
        return "HOLD"

    if price_change_percent > 5 and confidence > 0.7:
        return "STRONG BUY"
    elif price_change_percent < -5 and confidence > 0.7:
        return "STRONG SELL"
    elif price_change_percent > 2 and confidence > 0.65:
        return "BUY"
    elif price_change_percent < -2 and confidence > 0.65:
        return "SELL"
    elif price_change_percent > 1 and confidence > 0.6:
        return "WEAK BUY" 
    elif price_change_percent < -1 and confidence > 0.6:
        return "WEAK SELL"
    # Default
    else:
        return "HOLD"


def generate_prediction_plot(data_raw_flattened, one_week_pred, one_month_pred, ticker):
    """Generates a base64 encoded plot image with historical data and predictions.
       Expects FLATTENED data_raw.
    """
    plt.figure(figsize=(12, 6))

    if data_raw_flattened is None or data_raw_flattened.empty or 'Close' not in data_raw_flattened.columns:
         print("Error: Cannot generate plot due to missing or invalid flattened data_raw.")
         return None 

    plot_data_base = data_raw_flattened.dropna(subset=['Close']).copy()

    if not isinstance(plot_data_base.index, pd.DatetimeIndex):
         try:
             plot_data_base.index = pd.to_datetime(plot_data_base.index)
         except Exception as e:
             print(f"Error converting index to datetime for plot: {e}. Plot might be incorrect.")
             if not plot_data_base.empty:
                 plt.plot(plot_data_base['Close'].values[-90:], label='Historical Price (index fallback)', color='blue')
                 last_price = plot_data_base['Close'].iloc[-1]
                 plt.title(f'{ticker} Stock Price History (Date Error)')
             else:
                 plt.title(f'{ticker} Stock Price (Plot Error)')
                 plt.close()
                 return None 

    else:
        plot_data = plot_data_base.last('90D')
        if plot_data.empty or len(plot_data) < 2:
             plot_data = plot_data_base.tail(90) 

        if not plot_data.empty and len(plot_data) >= 2:
             plt.plot(plot_data.index, plot_data['Close'], label='Historical Price', color='blue')
             last_date = plot_data.index[-1]
             last_price = plot_data['Close'].iloc[-1]
        elif not plot_data_base.empty: 
             plt.scatter(plot_data_base.index[-1], plot_data_base['Close'].iloc[-1], color='blue', label='Last Price')
             last_date = plot_data_base.index[-1]
             last_price = plot_data_base['Close'].iloc[-1]
             print("Warning: Plotting limited historical data (1 point).")
        else:
             print("Warning: No historical data available for plotting.")
             plt.title(f'{ticker} Stock Price Forecast (No Data)')
             plt.close() 
             return None 


    if 'last_date' in locals() and 'last_price' in locals():
        one_week_date = last_date + timedelta(days=7)
        one_month_date = last_date + timedelta(days=30)

        if one_week_pred is not None and not np.isnan(one_week_pred):
             plt.scatter([one_week_date], [one_week_pred], color='green', s=100, zorder=5, label=f'1 Week Pred: ${one_week_pred:.2f}')
             plt.plot([last_date, one_week_date], [last_price, one_week_pred], 'g--', alpha=0.6)
        else:
             print("Warning: Skipping 1-week prediction point in plot (invalid value).")

        if one_month_pred is not None and not np.isnan(one_month_pred):
             plt.scatter([one_month_date], [one_month_pred], color='red', s=100, zorder=5, label=f'1 Month Pred: ${one_month_pred:.2f}')
             plt.plot([last_date, one_month_date], [last_price, one_month_pred], 'r--', alpha=0.6)
        else:
            print("Warning: Skipping 1-month prediction point in plot (invalid value).")

        plt.scatter([last_date], [last_price], color='blue', s=50, zorder=5, label=f'Last Actual: ${last_price:.2f}')
        plt.axvline(x=last_date, color='grey', linestyle='--', alpha=0.5)

    plt.title(f'{ticker} Stock Price Forecast')
    plt.xlabel('Date')
    plt.ylabel('Stock Price ($)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close() # Close the plot figure to free memory

    plot_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return plot_base64


# --- API Routes ---
@app.route('/predict', methods=['POST'])
def predict_stock():
    """
    API endpoint to get stock predictions and recommendations.
    Request JSON format: {"ticker": "AAPL", "force_retrain": false}
    """
    model = None
    price_scaler = None
    volume_scaler = None
    training_summary = {} 
    current_data_raw_flattened = None 
    ticker = "N/A"
    request_data = None

    try:
        request_data = request.get_json()
        if not request_data:
            return jsonify({"error": "Invalid JSON payload"}), 400
        ticker = request_data.get('ticker', 'AAPL').upper()
        force_retrain = request_data.get('force_retrain', False)

        if not isinstance(ticker, str) or len(ticker) > 6 or not ticker.replace('-', '').isalnum():
             return jsonify({"error": f"Invalid ticker symbol format: {ticker}"}), 400

        print(f"\n--- Request received for {ticker} (Force Retrain: {force_retrain}) ---")

        model_filename = f'models/{ticker}_rf_model.joblib'
        scaler_filename = f'models/{ticker}_scalers.joblib'

        if force_retrain or not os.path.exists(model_filename) or not os.path.exists(scaler_filename):
            if force_retrain:
                 print(f"Forcing retrain for {ticker}...")
            else:
                 print(f"Model or scaler not found for {ticker}. Training new model...")
            try:
                # train_model now returns flattened raw data as the last element
                model, price_scaler, volume_scaler, training_summary, _, current_data_raw_flattened = train_model(ticker)
                joblib.dump(model, model_filename)
                joblib.dump({'price_scaler': price_scaler, 'volume_scaler': volume_scaler}, scaler_filename)
                print(f"Model for {ticker} trained and saved.")
                # Add training status to summary
                training_summary["status"] = "Model trained successfully"

            except Exception as train_error:
                print(f"ERROR during training for {ticker}: {train_error}")
                traceback.print_exc()
                return jsonify({"error": f"Failed to train model for {ticker}: {str(train_error)}"}), 500
        else:
            # Load existing model and scalers
            try:
                print(f"Loading existing model and scalers for {ticker}...")
                model = joblib.load(model_filename)
                scalers = joblib.load(scaler_filename)
                price_scaler = scalers['price_scaler']
                volume_scaler = scalers.get('volume_scaler', None)
                print(f"Loaded existing model for {ticker}.")

                training_summary, current_data_raw_flattened = get_latest_data_and_eval(ticker, model, price_scaler, volume_scaler)
                print(f"Evaluation summary for {ticker}: {training_summary.get('eval_status', 'N/A')}")
                training_summary["status"] = "Model loaded and evaluated successfully"


            except FileNotFoundError:
                 print(f"Error: Model or scaler file disappeared for {ticker}. Forcing retrain.")
                 try:
                      model, price_scaler, volume_scaler, training_summary, _, current_data_raw_flattened = train_model(ticker)
                      joblib.dump(model, model_filename)
                      joblib.dump({'price_scaler': price_scaler, 'volume_scaler': volume_scaler}, scaler_filename)
                      print(f"Model for {ticker} trained and saved after load failure.")
                      training_summary["status"] = "Model re-trained successfully after load failure"
                 except Exception as train_error:
                      print(f"ERROR during fallback training for {ticker}: {train_error}")
                      traceback.print_exc()
                      return jsonify({"error": f"Failed to train model for {ticker} after load failure: {str(train_error)}"}), 500
            except Exception as load_eval_error:
                 print(f"ERROR during loading/evaluation for {ticker}: {load_eval_error}")
                 traceback.print_exc()
                 return jsonify({"error": f"Failed to load/evaluate model for {ticker}: {str(load_eval_error)}"}), 500


        if model is None or price_scaler is None: 
             print(f"Error: Model or price_scaler is None for {ticker} after load/train phase.")
             return jsonify({"error": f"Failed to obtain a valid model or scaler for {ticker}."}), 500
        if current_data_raw_flattened is None or current_data_raw_flattened.empty:
             print(f"Error: Could not obtain valid flattened data for {ticker}.")
             return jsonify({"error": f"Could not obtain valid data for {ticker}."}), 500

        try:
            one_week_pred, one_week_confidence, current_price = generate_prediction(model, current_data_raw_flattened, price_scaler, volume_scaler, days=7)
            one_month_pred, one_month_confidence, _ = generate_prediction(model, current_data_raw_flattened, price_scaler, volume_scaler, days=30)
        except Exception as pred_error:
             print(f"ERROR during prediction generation for {ticker}: {pred_error}")
             traceback.print_exc()
             return jsonify({"error": f"Failed to generate predictions for {ticker}: {str(pred_error)}"}), 500

        plot_image = None 
        try:
            plot_image = generate_prediction_plot(current_data_raw_flattened, one_week_pred, one_month_pred, ticker)
            if plot_image is None:
                 print(f"Plot generation returned None for {ticker}.")
        except Exception as plot_error:
            print(f"ERROR generating plot for {ticker}: {plot_error}")
            traceback.print_exc()

        one_week_change = None
        one_month_change = None
        if current_price is not None and current_price != 0 and one_week_pred is not None:
            one_week_change = ((one_week_pred - current_price) / current_price) * 100
        if current_price is not None and current_price != 0 and one_month_pred is not None:
            one_month_change = ((one_month_pred - current_price) / current_price) * 100

        one_week_rec = get_recommendation(one_week_change, one_week_confidence)
        one_month_rec = get_recommendation(one_month_change, one_month_confidence)

        def sanitize_float(value, precision=2):
            if value is None or np.isnan(value) or np.isinf(value):
                return None
            return float(f"{value:.{precision}f}")

        response = {
            "ticker": ticker,
            "current_price": sanitize_float(current_price),
            "one_week_prediction": {
                "price": sanitize_float(one_week_pred),
                "change_percent": sanitize_float(one_week_change),
                "recommendation": one_week_rec,
                "confidence": sanitize_float(one_week_confidence)
            },
            "one_month_prediction": {
                "price": sanitize_float(one_month_pred),
                "change_percent": sanitize_float(one_month_change),
                "recommendation": one_month_rec,
                "confidence": sanitize_float(one_month_confidence)
            },
            "model_summary": training_summary, 
            "data_plot": plot_image 
        }

        print(f"--- Request for {ticker} completed successfully ---")
        return jsonify(response), 200

    except Exception as e:
        print(f"!!! UNEXPECTED ERROR in /predict for ticker '{ticker}' !!!")
        print(f"Request Data: {request_data}")
        traceback.print_exc()
        return jsonify({"error": f"An unexpected server error occurred: {str(e)}"}), 500
    
@app.route('/update_model', methods=['POST'])
def update_model_route():
    """
    API endpoint to update the model with new feedback.
    Request JSON format: {"ticker": "AAPL", "date": "YYYY-MM-DD", "actual_price": 123.45}
    """
    try:
        request_data = request.get_json()
        if not request_data:
            return jsonify({"error": "Invalid JSON payload"}), 400
            
        ticker = request_data.get('ticker', '').upper()
        feedback_date = request_data.get('date', '')
        actual_price = request_data.get('actual_price')
        
        if not ticker or not feedback_date or actual_price is None:
            return jsonify({
                "error": "Missing required parameters. Please provide ticker, date, and actual_price."
            }), 400
            
        try:
            datetime.strptime(feedback_date, "%Y-%m-%d")
        except ValueError:
            return jsonify({"error": "Invalid date format. Use YYYY-MM-DD."}), 400
            
        try:
            actual_price = float(actual_price)
            if actual_price <= 0:
                raise ValueError("Price must be positive")
        except (ValueError, TypeError):
            return jsonify({"error": "actual_price must be a positive number"}), 400
            
        result = update_model_with_feedback(ticker, feedback_date, actual_price)
        
        return jsonify(result), 200
        
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        print(f"ERROR during model update: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Server error: {str(e)}"}), 500
    
@app.route('/model_info', methods=['POST'])
def get_model_info():
    try:
        ticker = "MSFT"
        print(f"Getting model info for {ticker}")
        
        model_filename = f'models/{ticker}_rf_model.joblib'
        scaler_filename = f'models/{ticker}_scalers.joblib'
        
        if not os.path.exists(model_filename) or not os.path.exists(scaler_filename):
            return jsonify({
                "error": f"No trained model found for {ticker}"
            }), 404
        
        model = joblib.load(model_filename)
        
        model_stats = {
            "feature_count": len(FEATURES),
            "estimators_count": model.n_estimators
        }
        
        try:
            scalers = joblib.load(scaler_filename)
            price_scaler = scalers['price_scaler']
            volume_scaler = scalers.get('volume_scaler', None)
            evaluation, _ = get_latest_data_and_eval(ticker, model, price_scaler, volume_scaler)
            model_stats["evaluation"] = evaluation
        except Exception as eval_error:
            print(f"Warning: Could not evaluate model: {str(eval_error)}")
            model_stats["evaluation"] = {
                "status": "Evaluation failed",
                "error": str(eval_error)
            }
        
        return jsonify(model_stats), 200
        
    except Exception as e:
        print(f"ERROR during model info retrieval: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Server error: {str(e)}"}), 500
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001) 