import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
import seaborn as sns
from src.model_training import EnergyPredictor
from src.utils import plot_to_image

def load_model_and_scaler(model_path, scaler_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
    scaler = joblib.load(scaler_path)
    
    return model, scaler

def predict_next_hours(data_path, model_path, scaler_path, hours, look_back=168):
    return predict_and_plot(data_path, model_path, scaler_path, hours=hours, look_back=look_back)

def predict_next_days(data_path, model_path, scaler_path, days, look_back=168):
    return predict_and_plot(data_path, model_path, scaler_path, days=days, look_back=look_back)

def predict_and_plot(data_path, model_path, scaler_path, hours=0, days=0, look_back=168):
    model, scaler = load_model_and_scaler(model_path, scaler_path)
    df = pd.read_csv(data_path)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    last_sequence = df.Global_active_power.values[-look_back:].reshape(-1, 1)
        
    last_sequence_scaled = scaler.transform(last_sequence)
        
    X = last_sequence_scaled.reshape(1, look_back, 1)
        
    predictions = []
    current_sequence = X.copy()
        
    total_hours = max(hours, days * 24)
        
    if total_hours <= 0:
        raise ValueError("Total hours for prediction must be greater than zero.")
        
    for _ in range(total_hours):
        next_pred = model.predict(current_sequence, verbose=0)
        predictions.append(next_pred[0])
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[0, -1, 0] = next_pred
        
    predictions = np.array(predictions).reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions)
        
    last_date = df['Datetime'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1), 
                                     periods=total_hours, freq='H')
        
    predictions_df = pd.DataFrame({
        'Timestamp': future_dates,
        'Predicted_Power': predictions.flatten()
    })
        
    peak_demand = predictions.max()
    low_demand = predictions.min()
        
    return predictions_df, peak_demand, low_demand

def plot_predictions(predictions_df, peak_demand, low_demand):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(predictions_df['Timestamp'], predictions_df['Predicted_Power'], color='blue', marker='o')
    ax.set_title('Predicted Power Consumption', fontsize=14, fontweight='bold')
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Predicted Power (kW)')
    ax.grid(True)
    fig.autofmt_xdate()
    
    # Annotate peak and low demand
    ax.annotate(f'Peak: {peak_demand:.2f} kW', xy=(predictions_df['Timestamp'].iloc[-1], peak_demand),
                xytext=(0, 10), textcoords='offset points', arrowprops=dict(arrowstyle='->', color='red'))
    ax.annotate(f'Low: {low_demand:.2f} kW', xy=(predictions_df['Timestamp'].iloc[0], low_demand),
                xytext=(0, -15), textcoords='offset points', arrowprops=dict(arrowstyle='->', color='green'))
    
    return plot_to_image(fig)

def plot_actual_vs_predicted(model_path, steps=1000):
    model = tf.keras.models.load_model(model_path)
    predictor = EnergyPredictor(data_path='C:/Harshil/Data Science/End to end  Project/Energy_Demand/data/processed/featured_power_consumption.csv')
    _, X_test, _, y_test = predictor.prepare_data()
    y_pred = model.predict(X_test)
    
    steps = min(steps, len(y_test), len(y_pred))
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.plot(range(steps), y_test[:steps], marker='.', label="Actual", color='purple')
    ax.plot(range(steps), y_pred[:steps], '-', label="Prediction", color='red')
    sns.despine(top=True)
    plt.subplots_adjust(left=0.07)
    ax.set_ylabel('Global_active_power', size=14)
    ax.set_xlabel('Time step', size=14)
    ax.legend(fontsize=16)
    ax.grid(True)
    return plot_to_image(fig)

if __name__ == "__main__":
    predictions_df, peak_demand, low_demand = predict_next_hours(
        data_path='C:/Harshil/Data Science/End to end  Project/Energy_Demand/data/processed/featured_power_consumption.csv',
        model_path='models/best_model.keras',
        scaler_path='models/scaler.pkl',
        hours=10
    )