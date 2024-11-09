from tensorflow.keras.models import model_from_json
import os
import pickle
import pandas as pd

def load_model_artifacts(model_dir='models'):
    """
    Load saved model and artifacts with version compatibility
    """
    try:
        # Load model architecture
        with open(os.path.join(model_dir, 'model_architecture.json'), 'r') as f:
            model_json = f.read()
        model = model_from_json(model_json)
        
        # Load model weights
        model.load_weights(os.path.join(model_dir, 'model_weights.h5'))
        
        # Compile the model
        model.compile(optimizer='adam', loss='mse')
        
        # Load scaler
        with open(os.path.join(model_dir, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
            
        return model, scaler
        
    except Exception as e:
        print(f"Error loading model artifacts: {str(e)}")
        raise

def predict_next_hour(df, feature_cols):
    """
    Main prediction function with error handling
    """
    try:
        # Load model and scaler
        model, scaler = load_model_artifacts()
        
        # Verify input data
        if len(df) < 24:
            raise ValueError(f"Need at least 24 hours of data, got {len(df)} rows")
            
        # Prepare input data
        X = prepare_input_data(df, feature_cols, scaler)
        
        # Make prediction
        prediction = make_prediction(model, X, scaler)
        
        # Create results dictionary
        results = {
            feature: float(value)  # Convert to native Python float
            for feature, value in zip(feature_cols, prediction)
        }
        
        return results
        
    except Exception as e:
        print(f"Error making prediction: {str(e)}")
        raise

# Example usage
if __name__ == "__main__":
    try:
        # Load your data
        df = pd.read_csv('data/processed/featuredd_power_consumption.csv')
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        
        # Make prediction
        results = predict_next_hour(df.tail(24), feature_cols)
        
        # Print results
        print("\nPredictions for next hour:")
        for feature, value in results.items():
            print(f"{feature}: {value:.4f}")
            
    except Exception as e:
        print(f"Exception encountered: {str(e)}") 