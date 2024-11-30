import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

class FeatureEngineer:
    def __init__(self):
        self.power_columns = ['Global_active_power', 'Global_reactive_power']
        self.submetering_columns = ['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
        self.decomposition_columns = ['Global_active_power_trend','Global_active_power_seasonal','Global_active_power_residual','Global_active_power_smooth',]
        
    def create_time_features(self, df):
        df['hour'] = df['Datetime'].dt.hour
        df['day'] = df['Datetime'].dt.day
        df['day_of_week'] = df['Datetime'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
        
        # Add peak hours
        df['peak_hours'] = ((df['hour'] >= 7) & (df['hour'] <= 9) | (df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
        
        return df
    
    def create_power_features(self, df):
        # Calculate total power from submeters
        df['total_submetering'] = df[self.submetering_columns].sum(axis=1)
        df['unknown_power'] = (df['Global_active_power'] * 1000/60) - df['total_submetering']
        
        # Calculate power factor (relationship between active and reactive power)
        df['power_factor'] = np.where(
            df['Global_active_power'] == 0, 0,
            df['Global_active_power'] / np.sqrt(
                df['Global_active_power']**2 + df['Global_reactive_power']**2
            )
        )
        # Trend and seasonality based features
        if 'Global_active_power_trend' in df.columns:
            df['trend_ratio'] = df['Global_active_power'] / df['Global_active_power_trend']
            df['seasonal_strength'] = abs(df['Global_active_power_seasonal']) / df['Global_active_power']
            df['residual_ratio'] = abs(df['Global_active_power_residual']) / df['Global_active_power']
        
        return df
    
    def create_statistical_features(self, df,window=[24]):
        for window in window:
            for col in self.power_columns + self.decomposition_columns:
                if col in df.columns:
                    df[f"{col}_{window}h_mean"] = df[col].rolling(window).mean()
                    df[f"{col}_{window}h_std"] = df[col].rolling(window).std()
                    
            for col in self.decomposition_columns:
                if col in df.columns:
                    df[f"{col}_{window}h_mean"] = df[col].rolling(window).mean()
        return df
    
    def create_all_features(self, df):
        try:
            print("Creating time features...")
            df = self.create_time_features(df)
            df = self.create_power_features(df)
            df = self.create_statistical_features(df)
            df = df.dropna()
            print(f"\nTotal features: {len(df.columns)}")
            return df
            
        except Exception as e:
            print(f"Error during feature creation: {str(e)}")
            raise

