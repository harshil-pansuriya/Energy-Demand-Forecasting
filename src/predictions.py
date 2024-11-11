import tensorflow as tf
import pandas as pd
import numpy as np
import json
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

class PowerPredictor:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.model, self.scaler, self.feature_cols = self.load_model_artifacts()

    def load_model_artifacts(self):
        """Loads the saved model and related artifacts"""
        try:
            model = tf.keras.models.load_model(f'{self.model_dir}/power_prediction_model.keras')
            
            with open(f'{self.model_dir}/scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
            
            with open(f'{self.model_dir}/feature_cols.json', 'r') as f:
                feature_cols = json.load(f)
                
            return model, scaler, feature_cols
        except Exception as e:
            print(f"Error loading model artifacts: {str(e)}")
            raise

    def prepare_test_data(self, df):
        """Prepare data for prediction using only the features used in training"""
        # Ensure all required features are present
        required_features = [
            'Global_active_power',
            'Global_reactive_power',
            'total_submetering',
            'hour_sin',
            'hour_cos',
            'is_weekend',
            'Global_active_power_24h_avg',
            'submetering_ma'
        ]
        
        missing_features = [col for col in required_features if col not in df.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
            
        return df[self.feature_cols].fillna(0)

    def validate_model_performance(self, X_test, y_test, threshold_rmse=1.0):
        """Validates model performance with detailed metrics"""
        predictions = self.model.predict(X_test)
        
        # Reshape for inverse transform
        predictions_reshaped = np.zeros((predictions.shape[0], len(self.feature_cols)))
        predictions_reshaped[:, 0] = predictions.flatten()
        
        y_test_reshaped = np.zeros((y_test.shape[0], len(self.feature_cols)))
        y_test_reshaped[:, 0] = y_test
        
        # Inverse transform
        predictions = self.scaler.inverse_transform(predictions_reshaped)[:, 0]
        y_test_actual = self.scaler.inverse_transform(y_test_reshaped)[:, 0]
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
        mae = mean_absolute_error(y_test_actual, predictions)
        
        print("\nModel Performance Metrics:")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        
        # Calculate percentage within range
        within_range = np.mean(np.abs((predictions - y_test_actual) / y_test_actual) <= 0.10) * 100
        print(f"Predictions within 10% of actual values: {within_range:.2f}%")
        
        # Detailed error analysis
        percentage_errors = ((predictions - y_test_actual) / y_test_actual) * 100
        
        print("\nDetailed Error Analysis:")
        print(f"Mean Percentage Error: {np.mean(np.abs(percentage_errors)):.2f}%")
        print(f"Median Percentage Error: {np.median(np.abs(percentage_errors)):.2f}%")
        print("\nError Distribution:")
        print(f"Within ±5%: {np.mean(np.abs(percentage_errors) <= 5):.1%}")
        print(f"Within ±10%: {np.mean(np.abs(percentage_errors) <= 10):.1%}")
        print(f"Within ±15%: {np.mean(np.abs(percentage_errors) <= 15):.1%}")
        print(f"Within ±20%: {np.mean(np.abs(percentage_errors) <= 20):.1%}")
        
        # Visualize predictions
        self.plot_predictions(y_test_actual, predictions)
        
        return rmse < threshold_rmse, predictions, y_test_actual

    def plot_predictions(self, actual, predictions, samples=100):
        """Plot actual vs predicted values"""
        plt.figure(figsize=(12, 6))
        plt.plot(actual[:samples], label='Actual', alpha=0.7)
        plt.plot(predictions[:samples], label='Predicted', alpha=0.7)
        plt.title('Actual vs Predicted Power Consumption')
        plt.xlabel('Time Steps')
        plt.ylabel('Global Active Power')
        plt.legend()
        plt.savefig('prediction_analysis.png')
        plt.close()

def main():
    try:
        predictor = PowerPredictor()
        
        print("Loading test data...")
        df = pd.read_csv('data/processed/featuredd_power_consumption.csv')
        
        # Prepare test data
        test_data = predictor.prepare_test_data(df)
        
        # Create sequences
        scaled_data = predictor.scaler.transform(test_data)
        X, y = [], []
        sequence_length = 24
        
        for i in range(len(scaled_data) - sequence_length):
            X.append(scaled_data[i:(i + sequence_length), :])
            y.append(scaled_data[i + sequence_length, 0])
            
        X_test, y_test = np.array(X), np.array(y)
        
        print("\nValidating model performance...")
        is_valid, predictions, actuals = predictor.validate_model_performance(X_test, y_test)
        
        return is_valid
        
    except Exception as e:
        print(f"An error occurred during validation: {str(e)}")
        raise

if __name__ == "__main__":
    is_valid = main()