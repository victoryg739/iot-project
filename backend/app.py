# app.py - Flask backend for stock prediction (scikit-learn version - Corrected + Debugging + Flattened Columns)

from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
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
import traceback # Import traceback for better error logging

warnings.filterwarnings('ignore') # Suppress warnings for cleaner output

app = Flask(__name__)

# Set random seed for reproducibility
np.random.seed(42)

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True) # Although plots aren't saved to disk here

# --- Constants ---
# Define features list globally to ensure consistency
FEATURES = ['Close', 'Volume', 'SMA20', 'SMA50', 'SMA_ratio',
            'Price_change', 'Volume_change'] + \
           [f'Price_lag_{lag}' for lag in [1, 2, 3, 5, 10]] + \
           [f'Volume_lag_{lag}' for lag in [1, 2, 3, 5, 10]]
TARGET_5D = 'Future_Price_5d'
TARGET_20D = 'Future_Price_20d' # Defined but not used in current training

# --- Helper Function: Flatten Columns ---
def flatten_yf_columns(df):
    """Flattens MultiIndex columns from yfinance if present."""
    if isinstance(df.columns, pd.MultiIndex):
        # Assumes structure like ('Close', 'AAPL'), keeps 'Close'
        print("Detected MultiIndex columns. Flattening.")
        df.columns = df.columns.get_level_values(0)
    return df

# --- Helper Function: Feature Engineering ---
def engineer_features(data):
    """Adds technical indicators and lag features to the DataFrame.
       Expects DataFrame with simple string column names (e.g., 'Close', 'Volume').
    """
    df = data.copy()
    # Ensure basic columns exist (using simple string names now)
    if 'Close' not in df.columns or 'Volume' not in df.columns:
        print("Warning: Input data to engineer_features missing 'Close' or 'Volume'.")
        # Return early or handle error appropriately
        # For now, let it proceed, but calculations might fail or add NaNs
        if 'Close' not in df.columns: df['Close'] = np.nan # Add dummy column if missing
        if 'Volume' not in df.columns: df['Volume'] = np.nan # Add dummy column if missing


    df['SMA20'] = df['Close'].rolling(window=20, min_periods=1).mean() # Use min_periods=1
    df['SMA50'] = df['Close'].rolling(window=50, min_periods=1).mean() # Use min_periods=1

    # Handle potential division by zero before calculating ratio
    df['SMA50_nonzero'] = df['SMA50'].replace(0, np.nan) # Create temp column
    df['SMA_ratio'] = df['SMA20'] / df['SMA50_nonzero']
    df.drop(columns=['SMA50_nonzero'], inplace=True) # Remove temp column

    df['Price_change'] = df['Close'].pct_change()
    # Check if Volume exists before calculating volume-based features
    if 'Volume' in df.columns and not df['Volume'].isnull().all(): # Check if column exists AND has data
        df['Volume_change'] = df['Volume'].pct_change()
    else:
         df['Volume_change'] = np.nan # Assign NaN if Volume is missing or all NaN

    for lag in [1, 2, 3, 5, 10]:
        df[f'Price_lag_{lag}'] = df['Close'].shift(lag)
        # Check if Volume exists before creating Volume lags
        if 'Volume' in df.columns and not df['Volume'].isnull().all():
            df[f'Volume_lag_{lag}'] = df['Volume'].shift(lag)
        else:
             df[f'Volume_lag_{lag}'] = np.nan # Assign NaN if Volume is missing or all NaN


    # Add future targets (will introduce NaNs at the end)
    df[TARGET_5D] = df['Close'].shift(-5)
    df[TARGET_20D] = df['Close'].shift(-20)

    # Handle potential infinities from calculations (like ratio)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df

# --- Core Functions ---
def train_model(ticker):
    """Downloads data, flattens columns, engineers features, cleans data, and trains a Random Forest model."""
    print(f"Training new model for {ticker}...")

    # Download historical data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1825) # 5 years of data
    data_raw_orig = yf.download(ticker, start=start_date, end=end_date, progress=False)

    if data_raw_orig.empty:
         raise ValueError(f"No historical data downloaded for {ticker}")

    # *** FIX: Flatten columns immediately after download ***
    data_raw = flatten_yf_columns(data_raw_orig.copy()) # Work on a copy

    if len(data_raw) < 300: # Need sufficient raw data
        raise ValueError(f"Insufficient historical data points ({len(data_raw)}) for {ticker} after potential flattening")

    # Feature Engineering (expects flattened columns)
    data_featured = engineer_features(data_raw)

    # Data Cleaning: Drop rows where any required feature OR the 5-day target is NaN
    required_columns_train = FEATURES + [TARGET_5D]

    # --- Debugging ---
    print("Columns available in data_featured (train):", data_featured.columns.tolist())
    print("Columns required for dropna (train):", required_columns_train)
    missing_cols_train = [col for col in required_columns_train if col not in data_featured.columns]
    if missing_cols_train:
        print(f"!!! CRITICAL (train): Missing columns before dropna: {missing_cols_train}")
        raise ValueError(f"Feature engineering failed to produce required columns: {missing_cols_train}")
    # --- End Debugging ---

    original_len = len(data_featured)
    # Now dropna should work as data_featured has simple string columns
    data_clean = data_featured.dropna(subset=required_columns_train).copy()
    cleaned_len = len(data_clean)
    print(f"Data points before NaN drop: {original_len}, after: {cleaned_len}")

    if cleaned_len < 100: # Check if enough data remains after cleaning
        raise ValueError(f"Insufficient data remaining for {ticker} after cleaning ({cleaned_len} points)")

    # Prepare Features (X) and Target (y) from the cleaned data
    X = data_clean[FEATURES].values
    y_5d = data_clean[TARGET_5D].values

    print(f"Shape of X: {X.shape}, Shape of y_5d: {y_5d.shape}")
    assert X.shape[0] == y_5d.shape[0], f"Shape mismatch after cleaning: X({X.shape[0]}) != y_5d({y_5d.shape[0]})"

    # Fit Scalers on the CLEANED data
    price_scaler = MinMaxScaler(feature_range=(0, 1))
    price_scaler.fit(data_clean[['Close']]) # Fit only on the 'Close' column

    volume_scaler = MinMaxScaler(feature_range=(0, 1))
    # Only fit volume scaler if Volume column exists and has non-NaN values
    if 'Volume' in data_clean.columns and not data_clean['Volume'].isnull().all():
        volume_scaler.fit(data_clean[['Volume']])
    else:
        volume_scaler = None # No volume scaler if no volume data
        print("Warning: Volume data not available or all NaN, Volume scaler not fitted.")

    # Train/Test Split (using the cleaned data)
    X_train, X_test, y_train_5d, y_test_5d = train_test_split(X, y_5d, test_size=0.2, shuffle=False)

    # Model Training
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train_5d)

    # Evaluation
    predictions_5d = model.predict(X_test)
    mse = np.mean(np.square(y_test_5d - predictions_5d))
    rmse = np.sqrt(mse)

    # Directional Accuracy
    close_idx = FEATURES.index('Close')
    actual_direction = np.sign(y_test_5d - X_test[:, close_idx])
    predicted_direction = np.sign(predictions_5d - X_test[:, close_idx])
    valid_comparison = predicted_direction != 0
    if np.sum(valid_comparison) > 0:
         directional_accuracy = np.mean(actual_direction[valid_comparison] == predicted_direction[valid_comparison]) * 100
    else:
         directional_accuracy = 0.0

    # Feature Importance
    feature_importances = dict(zip(FEATURES, model.feature_importances_))
    top_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:5]

    # Training Summary
    training_summary = {
        "mse": float(mse),
        "rmse": float(rmse),
        "directional_accuracy": float(directional_accuracy),
        "used_data_points_after_cleaning": cleaned_len,
        "top_features": dict(top_features),
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # Return model, scalers, summary, CLEANED data, and the FLATTENED raw data
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

    # *** FIX: Flatten columns immediately after download ***
    data_raw = flatten_yf_columns(data_raw_orig.copy()) # Work on a copy

    # Feature Engineering (expects flattened columns)
    data_featured = engineer_features(data_raw)

    # Define required columns for evaluation dropna
    required_columns_eval = FEATURES + [TARGET_5D]

    # <<< --- DEBUGGING BLOCK (Should show simple columns now) --- >>>
    print("\n--- Debugging Info in get_latest_data_and_eval ---")
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
    # <<< --- END DEBUGGING BLOCK --- >>>


    # Clean data specifically for evaluation (need features and 5d target)
    # This dropna should now work correctly
    eval_data_clean = data_featured.dropna(subset=required_columns_eval).copy()

    # Default summary if insufficient clean data for evaluation
    training_summary = {
        "mse": None, "rmse": None, "directional_accuracy": None,
        "used_data_points": 0,
        "evaluation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "eval_status": "Insufficient clean data for evaluation"
    }

    if len(eval_data_clean) < 20: # Need some data points to evaluate
        print(f"Warning: Insufficient clean data ({len(eval_data_clean)}) for evaluation of {ticker}. Only {len(eval_data_clean)} rows after dropna.")
        # Return default summary and the FLATTENED raw data
        return training_summary, data_raw.dropna(subset=['Close']) # Ensure raw data has 'Close'

    # Prepare features and target for evaluation from cleaned data
    X_eval = eval_data_clean[FEATURES].values
    y_eval_5d = eval_data_clean[TARGET_5D].values

    if X_eval.shape[0] != y_eval_5d.shape[0]:
         raise ValueError(f"Shape mismatch after cleaning in eval: X({X_eval.shape[0]}) != y({y_eval_5d.shape[0]})")

    # Make predictions
    predictions = model.predict(X_eval)

    # Calculate metrics
    mse = np.mean(np.square(y_eval_5d - predictions))
    rmse = np.sqrt(mse)

    # Calculate directional accuracy
    close_idx = FEATURES.index('Close')
    actual_direction = np.sign(y_eval_5d - X_eval[:, close_idx])
    predicted_direction = np.sign(predictions - X_eval[:, close_idx])
    valid_comparison = predicted_direction != 0
    if np.sum(valid_comparison) > 0:
         directional_accuracy = np.mean(actual_direction[valid_comparison] == predicted_direction[valid_comparison]) * 100
    else:
         directional_accuracy = 0.0

    # Update summary with evaluation results
    training_summary.update({
        "mse": float(mse),
        "rmse": float(rmse),
        "directional_accuracy": float(directional_accuracy),
        "used_data_points": len(eval_data_clean),
        "eval_status": "Evaluation successful"
    })

    # Return evaluation summary and the FLATTENED raw data
    return training_summary, data_raw.dropna(subset=['Close']) # Ensure raw data has 'Close'


def generate_prediction(model, current_data_raw_flattened, price_scaler, volume_scaler, days=7):
    """Generates future price prediction using the model or linear regression.
       Expects FLATTENED current_data_raw.
    """
    print(f"Generating {days}-day prediction...")
    # Need ~60 days of raw data to calculate all features (SMA50, lags) for the latest point
    # current_data_raw_flattened should already have simple columns
    pred_data_raw = current_data_raw_flattened.copy().tail(60) # Increased from 50

    if len(pred_data_raw) < 55: # Rough check if enough data exists
         raise ValueError(f"Insufficient recent data ({len(pred_data_raw)} points) to calculate features for prediction.")

    # Engineer features on this recent raw data subset (expects flattened columns)
    pred_data_featured = engineer_features(pred_data_raw)

    # --- Debugging for prediction feature generation ---
    print("\n--- Debugging Info in generate_prediction ---")
    print("Columns available in pred_data_featured:", pred_data_featured.columns.tolist())
    missing_pred_cols = [col for col in FEATURES if col not in pred_data_featured.columns]
    if missing_pred_cols:
        print(f"!!! WARNING (prediction): Missing base features before dropna: {missing_pred_cols}")
        # Handle based on which features are missing
    else:
        print("All required features columns are present in pred_data_featured.")

    # Check for NaNs in the latest row *before* dropna, specifically in FEATURES
    latest_row_features = pred_data_featured[FEATURES].iloc[-1]
    nans_in_latest = latest_row_features.isnull().sum()
    if nans_in_latest > 0:
        print(f"!!! WARNING (prediction): Latest feature row contains {nans_in_latest} NaN(s) BEFORE dropna:")
        print(latest_row_features[latest_row_features.isnull()])
        # This indicates an issue either in data source or feature engineering logic for the most recent point(s)

    print("--- End Debugging Info ---\n")
    # --- End Debugging ---

    # Clean the featured data - only need FEATURES, target is not required here
    # Drop rows in this smaller dataset if they have NaNs *in the required features*
    pred_data_clean = pred_data_featured.dropna(subset=FEATURES).copy()

    if pred_data_clean.empty:
        # Add more context to the error
        print("--- Error Details: pred_data_featured (before dropna) ---")
        print(pred_data_featured.tail(10)) # Show last few rows before dropna
        print(f"Required columns for dropna: {FEATURES}")
        raise ValueError("Could not generate features for prediction - final data point row is empty after dropping NaNs in required features.")


    # Get the latest available feature row
    latest_features = pred_data_clean[FEATURES].values[-1].reshape(1, -1)
    last_close_price = pred_data_clean['Close'].iloc[-1] # Get the actual last closing price from cleaned data

    # Short-term prediction (<= 10 days): Use the trained Random Forest model
    if days <= 10:
        # Ensure latest_features does not contain NaNs (should be handled by dropna, but double-check)
        if np.isnan(latest_features).any():
             raise ValueError(f"NaN values detected in the latest features used for prediction: {latest_features}")
        prediction = model.predict(latest_features)[0]
        # Simple confidence estimate (could be improved)
        confidence = 0.75
        print(f"Using RF model for {days}-day prediction. Raw output: {prediction}")
    # Longer-term prediction (> 10 days): Use simple linear trend on recent data
    else:
        print(f"Using Linear Regression trend for {days}-day prediction.")
        # Use last 30 days of *cleaned* data for trend calculation
        recent_clean_data = pred_data_clean.tail(30)
        if len(recent_clean_data) < 5: # Need at least a few points for LR
            raise ValueError("Insufficient cleaned data points (<5) for linear regression trend.")

        X_lr = np.arange(len(recent_clean_data)).reshape(-1, 1)
        y_lr = recent_clean_data['Close'].values

        lr_model = LinearRegression()
        lr_model.fit(X_lr, y_lr)

        # Predict 'days' steps *after* the last point used in LR fit
        future_index = len(recent_clean_data) - 1 + days
        prediction = lr_model.predict(np.array([[future_index]]))[0]

        # Lower confidence for longer-term linear extrapolation
        confidence = 0.6 if days <= 20 else 0.5
        print(f"Using LR model for {days}-day prediction. Output: {prediction}")

    # Sanity check: prediction shouldn't be wildly different or negative
    if prediction < 0:
        print(f"Warning: Negative prediction ({prediction}) clipped to 0.")
        prediction = 0.0

    return prediction, confidence, last_close_price


def get_recommendation(price_change_percent, confidence):
    """Determines a trading recommendation based on predicted change and confidence."""
    # Check for None or NaN before calculations
    if price_change_percent is None or np.isnan(price_change_percent) or \
       confidence is None or np.isnan(confidence):
        return "HOLD" # Default if inputs are invalid

    abs_change = abs(price_change_percent)

    if confidence < 0.55: # Low confidence defaults to HOLD
        return "HOLD"

    # Strong signals (require higher confidence)
    if price_change_percent > 5 and confidence > 0.7:
        return "STRONG BUY"
    elif price_change_percent < -5 and confidence > 0.7:
        return "STRONG SELL"
    # Medium signals
    elif price_change_percent > 2 and confidence > 0.65:
        return "BUY"
    elif price_change_percent < -2 and confidence > 0.65:
        return "SELL"
    # Weak signals / Consider HOLD
    elif price_change_percent > 1 and confidence > 0.6:
        return "WEAK BUY" # Consider HOLD
    elif price_change_percent < -1 and confidence > 0.6:
        return "WEAK SELL" # Consider HOLD
    # Default
    else:
        return "HOLD"


def generate_prediction_plot(data_raw_flattened, one_week_pred, one_month_pred, ticker):
    """Generates a base64 encoded plot image with historical data and predictions.
       Expects FLATTENED data_raw.
    """
    plt.figure(figsize=(12, 6))

    # Ensure data_raw_flattened is not empty and has 'Close' (simple string column)
    if data_raw_flattened is None or data_raw_flattened.empty or 'Close' not in data_raw_flattened.columns:
         print("Error: Cannot generate plot due to missing or invalid flattened data_raw.")
         return None # Return None if plot cannot be generated

    # Drop rows where Close is NaN for plotting robustness
    plot_data_base = data_raw_flattened.dropna(subset=['Close']).copy()

    # Ensure index is DatetimeIndex
    if not isinstance(plot_data_base.index, pd.DatetimeIndex):
         try:
             plot_data_base.index = pd.to_datetime(plot_data_base.index)
         except Exception as e:
             print(f"Error converting index to datetime for plot: {e}. Plot might be incorrect.")
             # Attempt basic plot without dates if conversion fails
             if not plot_data_base.empty:
                 plt.plot(plot_data_base['Close'].values[-90:], label='Historical Price (index fallback)', color='blue')
                 # Can't reliably place predictions without dates
                 last_price = plot_data_base['Close'].iloc[-1]
                 plt.title(f'{ticker} Stock Price History (Date Error)')
                 # Skip predictions if dates failed
             else:
                 plt.title(f'{ticker} Stock Price (Plot Error)')
                 plt.close()
                 return None # Cannot plot anything useful

    else:
        # Plot historical data for the last 90 calendar days
        plot_data = plot_data_base.last('90D')
        if plot_data.empty or len(plot_data) < 2:
             plot_data = plot_data_base.tail(90) # Fallback to last 90 points

        if not plot_data.empty and len(plot_data) >= 2:
             plt.plot(plot_data.index, plot_data['Close'], label='Historical Price', color='blue')
             last_date = plot_data.index[-1]
             last_price = plot_data['Close'].iloc[-1]
        elif not plot_data_base.empty: # Only 1 point available
             plt.scatter(plot_data_base.index[-1], plot_data_base['Close'].iloc[-1], color='blue', label='Last Price')
             last_date = plot_data_base.index[-1]
             last_price = plot_data_base['Close'].iloc[-1]
             print("Warning: Plotting limited historical data (1 point).")
        else:
             print("Warning: No historical data available for plotting.")
             plt.title(f'{ticker} Stock Price Forecast (No Data)')
             plt.close() # Close the empty plot
             return None # Cannot generate plot


    # Add prediction points ONLY if we have a valid last_date and last_price
    if 'last_date' in locals() and 'last_price' in locals():
        one_week_date = last_date + timedelta(days=7)
        one_month_date = last_date + timedelta(days=30)

        # Ensure predictions are valid numbers before plotting
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


        # Add marker for the last actual data point
        plt.scatter([last_date], [last_price], color='blue', s=50, zorder=5, label=f'Last Actual: ${last_price:.2f}')
        plt.axvline(x=last_date, color='grey', linestyle='--', alpha=0.5)

    plt.title(f'{ticker} Stock Price Forecast')
    plt.xlabel('Date')
    plt.ylabel('Stock Price ($)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Convert plot to base64 string
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
    training_summary = {} # Initialize as dict
    current_data_raw_flattened = None # Will store FLATTENED raw data
    ticker = "N/A"
    request_data = None

    try:
        # Get request data
        request_data = request.get_json()
        if not request_data:
            return jsonify({"error": "Invalid JSON payload"}), 400
        ticker = request_data.get('ticker', 'AAPL').upper()
        force_retrain = request_data.get('force_retrain', False)

        if not isinstance(ticker, str) or len(ticker) > 6 or not ticker.replace('-', '').isalnum():
             return jsonify({"error": f"Invalid ticker symbol format: {ticker}"}), 400

        print(f"\n--- Request received for {ticker} (Force Retrain: {force_retrain}) ---")

        # Define filenames
        model_filename = f'models/{ticker}_rf_model.joblib'
        scaler_filename = f'models/{ticker}_scalers.joblib'

        # Train or load model logic
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
                volume_scaler = scalers.get('volume_scaler', None) # Use .get for backward compatibility
                print(f"Loaded existing model for {ticker}.")

                # Get latest data, evaluate, and get FLATTENED raw data
                # get_latest_data_and_eval now returns flattened raw data
                training_summary, current_data_raw_flattened = get_latest_data_and_eval(ticker, model, price_scaler, volume_scaler)
                print(f"Evaluation summary for {ticker}: {training_summary.get('eval_status', 'N/A')}")
                # Add load status to summary
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

        # --- Post Model Load/Train ---

        if model is None or price_scaler is None: # volume_scaler can be None
             print(f"Error: Model or price_scaler is None for {ticker} after load/train phase.")
             return jsonify({"error": f"Failed to obtain a valid model or scaler for {ticker}."}), 500
        if current_data_raw_flattened is None or current_data_raw_flattened.empty:
             print(f"Error: Could not obtain valid flattened data for {ticker}.")
             return jsonify({"error": f"Could not obtain valid data for {ticker}."}), 500

        # Generate predictions using the FLATTENED data
        try:
            one_week_pred, one_week_confidence, current_price = generate_prediction(model, current_data_raw_flattened, price_scaler, volume_scaler, days=7)
            one_month_pred, one_month_confidence, _ = generate_prediction(model, current_data_raw_flattened, price_scaler, volume_scaler, days=30)
        except Exception as pred_error:
             print(f"ERROR during prediction generation for {ticker}: {pred_error}")
             traceback.print_exc()
             return jsonify({"error": f"Failed to generate predictions for {ticker}: {str(pred_error)}"}), 500

        # Generate base64 plot image using the FLATTENED raw data
        plot_image = None # Default to None
        try:
            plot_image = generate_prediction_plot(current_data_raw_flattened, one_week_pred, one_month_pred, ticker)
            if plot_image is None:
                 print(f"Plot generation returned None for {ticker}.")
        except Exception as plot_error:
            print(f"ERROR generating plot for {ticker}: {plot_error}")
            traceback.print_exc()
            # plot_image remains None

        # Calculate percentage changes safely
        one_week_change = None
        one_month_change = None
        if current_price is not None and current_price != 0 and one_week_pred is not None:
            one_week_change = ((one_week_pred - current_price) / current_price) * 100
        if current_price is not None and current_price != 0 and one_month_pred is not None:
            one_month_change = ((one_month_pred - current_price) / current_price) * 100

        # Determine recommendations
        one_week_rec = get_recommendation(one_week_change, one_week_confidence)
        one_month_rec = get_recommendation(one_month_change, one_month_confidence)

        # Ensure numerical fields are JSON serializable (handle potential NaN/inf)
        def sanitize_float(value, precision=2):
            if value is None or np.isnan(value) or np.isinf(value):
                return None
            return float(f"{value:.{precision}f}")


        # Create response JSON
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
            "model_summary": training_summary, # Contains status, metrics etc.
            "data_plot": plot_image # Base64 image string or None
        }

        print(f"--- Request for {ticker} completed successfully ---")
        return jsonify(response), 200

    except Exception as e:
        # Catch-all for unexpected errors in the route handler
        print(f"!!! UNEXPECTED ERROR in /predict for ticker '{ticker}' !!!")
        print(f"Request Data: {request_data}")
        traceback.print_exc()
        return jsonify({"error": f"An unexpected server error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    # Ensure the app runs on localhost:5000 or adjust as needed
    app.run(debug=True, host='0.0.0.0', port=5001) # Run on port 5001 if 5000 is common