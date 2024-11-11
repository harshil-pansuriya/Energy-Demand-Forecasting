import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
import os, json, pickle
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

class PowerPredictionModel:
    def __init__(self, data_file='data/processed/featuredd_power_consumption.csv'):
        self.data_file = data_file
        # Only keep features that are most important for predictions
        self.feature_cols = [
            'Global_active_power',          # Target variable
            'Global_reactive_power',        # Important for power factor
            'total_submetering',           # Direct power consumption component
            'hour_sin',                    # Cyclical time patterns
            'hour_cos',                    # Cyclical time patterns
            'is_weekend',                  # Weekly consumption patterns
            'Global_active_power_24h_avg', # Historical trend
            'submetering_ma'              # Moving average for trend
        ]
    
    def prepare_data(self, sequence_length=24):
        """Prepare data for training with only essential features"""
        df = pd.read_csv(self.data_file)
        
        # Scale features
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[self.feature_cols].fillna(0))
        
        # Create sequences
        X, y = self._create_sequences(scaled_data, sequence_length)
        train_size = int(len(X) * 0.8)
        
        return (X[:train_size], X[train_size:], y[:train_size], y[train_size:], scaler)
    
    def _create_sequences(self, data, seq_length):
        """Create sequences for LSTM"""
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:(i + seq_length), :])
            y.append(data[i + seq_length, 0])  # Predict Global_active_power
        return np.array(X), np.array(y)
    
    def build_model(self, sequence_length=24, n_features=None):
        """Build LSTM model with optimized architecture"""
        if n_features is None:
            n_features = len(self.feature_cols)
            
        model = Sequential([
            LSTM(128, activation='relu',input_shape=(sequence_length, n_features), return_sequences=True),
            Dropout(0.3),
            LSTM(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dense(16, activation='relu'),
            Dense(1)  # Output layer for predicting Global_active_power
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        return model
    
    def train(self, epochs=50):
        print("Preparing data...")
        X_train, X_test, y_train, y_test, scaler = self.prepare_data()
        
        print("Building model...")
        model = self.build_model()
        
        print("Training model...")
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001),
            ModelCheckpoint('models/best_model.keras', monitor='val_loss', save_best_only=True)
        ]
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        self.save_model(model, scaler)
        return model, history
    
    def save_model(self, model, scaler, save_dir='models'):
        os.makedirs(save_dir, exist_ok=True)
        model.save(os.path.join(save_dir, 'power_prediction_model.keras'))
        with open(os.path.join(save_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)
        with open(os.path.join(save_dir, 'feature_cols.json'), 'w') as f:
            json.dump(self.feature_cols, f)

if __name__ == "__main__":
    predictor = PowerPredictionModel()
    model, history = predictor.train()