# 🔋 Energy Demand Forecasting System

## 📊 Overview
An energy demand forecasting system that leverages Time Series Forecasting to predict future energy consumption patterns. The system provides both visualization of historical data and predictions for future energy demand.

## ⭐ Key Features
- 🔮 Predict energy demand for next hours/days
- 📈 Interactive visualizations of energy consumption patterns
- 📊 Statistical analysis of historical data
- 🕒 Time series decomposition and analysis
- 📱 User-friendly Gradio interface

## 🏗️ Architecture
The project is structured into several key components:

### Data Processing Pipeline
- Data preprocessing and cleaning
- Feature engineering
- Time series decomposition
- Statistical analysis

### Time Series Model
- LSTM-based neural network
- Attention mechanism for improved predictions
- Batch normalization and dropout for regularization

### Visualization Components
- Power consumption trends
- Weekly power heatmaps
- Average power by hour/day/month
- Time series components visualization

### Installation
    ```
    git clone https://github.com/harshil-pansuriya/Energy-Demand-Forecasting.git
    ```
### Setup Environment
    ```
    python -m venv env_name
    env_name/Scripts/activate
    ```
pip install -r requirements.txt

### Usage

1. Model Training 
```
python src/model_training.py
```
2. Running Application
```
python app.py
```
## 📊 Features in Detail

### 1. Historical Data Analysis
- Comprehensive statistical analysis
- Pattern recognition in energy consumption
- Seasonal trend decomposition
- Anomaly detection

### 2. Prediction Capabilities
- Short-term predictions (hourly)
- Medium-term predictions (daily)
- Peak & Low demand forecasting

## 🔧 Technical Implementation

### Model Architecture
- LSTM layers with attention mechanism
- Batch normalization for stable training
- Dropout layers for preventing overfitting

### Data Processing
- Data cleaning
- Feature engineering pipeline
- Time-based feature extraction
- Outlier detection and handling

## 📈 Performance Metrics
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- R² Score
- Peak Demand Accuracy