import pandas as pd
import numpy as np
import logging
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.signal import savgol_filter

class DataPreprocessor:
    
    def __init__(self, logging_level = logging.INFO):
        self.numerical_columns = [
            'Global_active_power', 'Global_reactive_power',
            'Voltage', 'Global_intensity', 'Sub_metering_1',
            'Sub_metering_2', 'Sub_metering_3'
        ]
        logging.basicConfig(level=logging_level)
        self.logger = logging.getLogger(__name__)
        
    def load_data(self, filepath):
        try:
            df = pd.read_csv(filepath)
            self.logger.info(f"Successfully loaded data from {filepath}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def handle_datetime(self, df):
        try:
            # Split dataset 
            split_idx = 186352  # Based on EDA findings
            df_part1 = df.iloc[:split_idx].copy()
            df_part2 = df.iloc[split_idx:].copy()
            
            df_part1['Datetime'] = pd.to_datetime(df_part1['Date'] + ' ' + df_part1['Time'],dayfirst=True, errors='coerce')
            df_part2['Datetime'] = pd.to_datetime(df_part2['Date'] + ' ' + df_part2['Time'], dayfirst=True, errors='coerce')
            
            df = pd.concat([df_part1, df_part2], ignore_index=True)
            
            df = df.drop(columns=['Date', 'Time', 'index'], errors='ignore')
            df = df.dropna(subset=['Datetime'])
            df.sort_values('Datetime', inplace=True)
            self.logger.info("Successfully processed datetime")
            return df
        
        except Exception as e:
            self.logger.error(f"Error processing datetime: {str(e)}")
            raise
        
    def convert_numeric_columns(self, df):
       
        df[self.numerical_columns] = df[self.numerical_columns].apply(pd.to_numeric, errors='coerce')
        return df
    
    def handle_outliers(self, df, method = 'iqr', threshold= 1.5):
        
        for col in self.numerical_columns:
            Q1, Q3 = df[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            lower_bound, upper_bound = Q1 - threshold * IQR, Q3 + threshold * IQR
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        return df
    
    def decompose_time_series(self, df, column='Global_active_power', period=24):
        try:
            df.set_index('Datetime', inplace=True)
            decomposition = seasonal_decompose(df[column], period=period, extrapolate_trend='freq')
            df[f'{column}_trend'] = decomposition.trend
            df[f'{column}_seasonal'] = decomposition.seasonal
            df[f'{column}_residual'] = decomposition.resid
            self.logger.info(f"Time series decomposition successful for {column}")
            return df.reset_index()
        except Exception as e:
            self.logger.error(f"Error in time series decomposition: {str(e)}")
            raise

    def smooth_time_series(self, df, column='Global_active_power', window_size=24):
            
        try:
            df[f'{column}_smoothed_ma'] = df[column].rolling(window=window_size, center=True).mean()
            df[f'{column}_smoothed_sg'] = savgol_filter(df[column].fillna(method='ffill'), window_length=25, polyorder=3)
            self.logger.info(f"Successfully smoothed {column}")
            return df
        except Exception as e:
            self.logger.error(f"Error in smoothing: {str(e)}")
            raise

    def preprocess_pipeline(self, filepath):
        
        try:
            df = self.load_data(filepath) 
            df = df.dropna(subset=self.numerical_columns)
            df = self.handle_datetime(df)
            df = self.convert_numeric_columns(df)
            df = self.handle_outliers(df)
            df = self.decompose_time_series(df)
            df = self.smooth_time_series(df)
            
            self.logger.info("Preprocessing completed successfully")
            return df
            
        except Exception as e:
            self.logger.error(f"Error in preprocessing pipeline: {str(e)}")
            raise
        
    
    