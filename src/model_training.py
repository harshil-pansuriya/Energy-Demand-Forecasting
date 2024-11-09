import pandas as pd
import numpy as np
import os
import json, pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Simple settings for our model      # Number of samples to process at once
feature_cols = [
    # Main target
    'Global_active_power',
    
    # Power-related features
    'Global_reactive_power',
    'total_submetering',
    'unknown_power',
    'power_factor',
    'active_power_change',
    
    # Time-based features
    'hour',
    'is_weekend',
    
    # Statistical features
    'Global_active_power_24h_avg',
    'Global_active_power_24h_std'
]

data_file = 'data/processed/featuredd_power_consumption.csv'

def create_sequences(data, seq_length=24):
    X, y = [], []
    for i in range(len(data) - seq_length):
        # Get sequence of seq_length values
        X.append(data[i:(i + seq_length), :])
        # Get the next value after the sequence
        y.append(data[i + seq_length, :])
        
    return np.array(X), np.array(y)

def prepare_data(df, train_split=0.8):
    """
    Prepares data for training by scaling and creating sequences
    """
    # Make sure data is in time order
    df = df.sort_values('Datetime')
    
    # Scale the power consumption values between 0 and 1
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[feature_cols])
    
    # Create sequences for training
    X, y = create_sequences(scaled_data)
    
    # Split into training and testing sets
    train_size = int(len(X) * train_split)
    
    X_train = X[:train_size]
    X_test = X[train_size:]
    y_train = y[:train_size]
    y_test = y[train_size:]
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, scaler

def build_model(sequence_length=24, n_features=len(feature_cols)):
    """
    Creates a simple LSTM model
    """
    model = Sequential([
        # First LSTM layer
        LSTM(64, activation='relu', input_shape=(sequence_length, n_features), return_sequences=True),
        Dropout(0.2),
        
        # Second LSTM layer
        LSTM(32, activation='relu'),
        Dropout(0.2),
        
        # Dense layers for final prediction
        Dense(16, activation='relu'),
        Dense(n_features)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    return model

def train_model(model, X_train, y_train, X_test, y_test, epochss=20, batch_size=32):
    """
    Trains the LSTM model
    """
    # Set up early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=epochss,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )
    
    print("Model training completed!")
    return model, history

def evaluate_model(model, X_test, y_test, scaler):
    """
    Tests how well the model performs
    """
    # Make predictions
    predictions = model.predict(X_test)
    
    # Ensure correct shapes for inverse transform
    if len(predictions.shape) == 2:
        predictions_reshaped = predictions  # Already 2D
    else:
        predictions_reshaped = predictions.reshape(-1, len(feature_cols))
        
    if len(y_test.shape) == 2:
        y_test_reshaped = y_test  # Already 2D
    else:
        y_test_reshaped = y_test.reshape(-1, len(feature_cols))
    
    # Convert predictions back to original scale
    predictions = scaler.inverse_transform(predictions_reshaped)
    y_test_scaled = scaler.inverse_transform(y_test_reshaped)
    
    metrics = {}
    for i, col in enumerate(feature_cols):
        col_metrics = {
            'MSE': mean_squared_error(y_test_scaled[:, i], predictions[:, i]),
            'RMSE': np.sqrt(mean_squared_error(y_test_scaled[:, i], predictions[:, i])),
            'MAE': mean_absolute_error(y_test_scaled[:, i], predictions[:, i])
        }
        metrics[col] = col_metrics
    print("\nPrediction Results:")
    for col in feature_cols:
        print(f"\n{col}:")
        for metric, value in metrics[col].items():
            print(f"{metric}: {value:.4f}")
            
    return predictions, metrics

def save_model_artifacts(model, scaler, metrics, save_dir='models'):
    """
    Saves model and related artifacts
    """
    # Save model
    model.save(f'{save_dir}/power_prediction_model.h5')
    # Save scaler
    with open(f'{save_dir}/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    # Save metrics for each feature
    with open(f'{save_dir}/metrics.json', 'w') as f:
        json.dump({k: {mk: float(mv) for mk, mv in v.items()} 
                  for k, v in metrics.items()}, f, indent=4)
    # Save metrics
    with open(f'{save_dir}/metrics.json', 'w') as f:
        json.dump(metrics, f)

def main():
    try:
        print("Loading data...")
        df = pd.read_csv(data_file)
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        
        # Prepare the data
        X_train, X_test, y_train, y_test, scaler = prepare_data(df)
        
        # Train the model
        model=build_model()
        model,history = train_model(model, X_train, y_train, X_test, y_test)
        
        # Evaluate the model
        predictions, metrics = evaluate_model(model, X_test, y_test, scaler)
        
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
            
        # Save the model
        save_model_artifacts(model, scaler, metrics)
        print("Model saved successfully!")
        
        return model, history, metrics, predictions
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    model, history, metrics = main()