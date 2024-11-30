import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, Flatten, AdditiveAttention, Multiply
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import pickle

class EnergyPredictor:
    def __init__(self, data_path, look_back=168):
        self.data_path = data_path
        self.look_back = look_back
        self.scaler = MinMaxScaler(feature_range=(0,1))
        self.model = None
        os.makedirs('models', exist_ok=True)
        self.model_path = 'models/best_model.keras'
        self.scaler_path = 'models/scaler.pkl'
    
    def prepare_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - self.look_back - 1):
            X.append(data[i:(i + self.look_back), 0])
            y.append(data[i + self.look_back, 0])
        return np.array(X), np.array(y)

    def prepare_data(self):
        df = pd.read_csv(self.data_path)
        dataset = df.Global_active_power.values.reshape(-1,1)
        scaled_data = self.scaler.fit_transform(dataset)
        
        train_size = int(len(scaled_data) * 0.8)
        train, test = scaled_data[:train_size], scaled_data[train_size:]
        
        X_train, y_train = self.prepare_sequences(train)
        X_test, y_test = self.prepare_sequences(test)
        X_train = X_train.reshape(-1, self.look_back, 1)
        X_test = X_test.reshape(-1, self.look_back, 1)
    
        return X_train, X_test, y_train, y_test
    
    def build_model(self, input_shape):
        input_layer = Input(shape=input_shape)
        lstm_1 = LSTM(units=50, return_sequences=True)(input_layer)
        batch_norm_1 = BatchNormalization()(lstm_1)
        lstm_2 = LSTM(units=50, return_sequences=True)(batch_norm_1)
        batch_norm_2 = BatchNormalization()(lstm_2)
        #Attention Mechanism
        attention = AdditiveAttention(name='attention_weight')
        attention_output = attention([batch_norm_2, batch_norm_2])
        multiply_layer = Multiply()([batch_norm_2, attention_output])
        
        flatten_layer = Flatten()(multiply_layer)
        dropout_layer = Dropout(0.3)(flatten_layer)
        output_layer = Dense(1)(dropout_layer)
        
        self.model = Model(inputs=input_layer, outputs=output_layer)
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        
    def train(self, epochs=20, batch_size=128):
        X_train, X_test, y_train, y_test = self.prepare_data()
        if self.model is None:
            self.build_model((self.look_back, 1))
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(self.model_path, save_best_only=True, monitor='val_loss'),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
        ]
        history = self.model.fit(X_train, y_train,epochs=epochs,batch_size=batch_size,
                                 validation_split=0.1, callbacks=callbacks)
        
        with open(self.scaler_path, 'wb') as f:  # Using self.scaler_path
            pickle.dump(self.scaler, f)
            print(f"Scaler saved to {self.scaler_path}")
        
        print("\nEvaluating model performance...")
        metrics = self.evaluate(X_test, y_test)
        return history, metrics
    
    def evaluate(self, X_test, y_test):
        if self.model is None:
            raise ValueError("Model hasn't been trained yet")
            
        y_pred = self.model.predict(X_test)
        return {
            'mae': mean_absolute_error(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }
    def save_model(self): 
        if self.model is not None:
            self.model.save(self.model_path)
            print(f"Model saved to {self.model_path}")

if __name__ == "__main__":
    predictor = EnergyPredictor('C:/Harshil/Data Science/End to end  Project/Energy_Demand/data/processed/featured_power_consumption.csv')
    history, metrics = predictor.train()
    
    print("Model Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    predictor.save_model()