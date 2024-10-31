import pandas as pd
import numpy as np
import logging
from typing import List
from data_preprocessing import DataPreprocessor
    
class FeatureEngineer:
    #A class to create features for household power consumption prediction.
    def __init__(self, logging_level: int = logging.INFO):
        # Initialize FeatureEngineer with configuration
        self.power_columns = ['Global_active_power', 'Global_reactive_power','Global_intensity']
        self.submetering_columns = ['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
        # Setup logging
        logging.basicConfig(level=logging_level)
        self.logger = logging.getLogger(__name__)
        self.preprocessor = DataPreprocessor() 
        
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Create comprehensive time-based features
        df = df.copy()
        # Basic time features
        df['hour'] = df['Datetime'].dt.hour
        df['day'] = df['Datetime'].dt.day
        df['month'] = df['Datetime'].dt.month
        df['year'] = df['Datetime'].dt.year
        df['day_of_week'] = df['Datetime'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Time of day categories
        df['time_of_day'] = pd.cut(df['hour'], 
                                 bins=[0, 6, 12, 18, 24], 
                                 labels=['night', 'morning', 'afternoon', 'evening'])
        
        return df
        
    def create_power_features(self, df: pd.DataFrame) -> pd.DataFrame:
        
        #Create power consumption related features
        df = df.copy()
        # Total submetering
        df['total_submetering'] = df[self.submetering_columns].sum(axis=1)
        # Unknown power consumption
        df['unknown_power'] = df['Global_active_power'] * 1000/60 - df['total_submetering']
        # Power factor
        df['power_factor'] = np.where(
            df['Global_active_power'] == 0,
            0,
            df['Global_active_power'] / np.sqrt(
                df['Global_active_power']**2 + df['Global_reactive_power']**2
            )
        )
        # Submetering ratios
        for col in self.submetering_columns:
            df[f'{col}_ratio'] = np.where(
                df['total_submetering'] == 0,
                0,
                df[col] / df['total_submetering']
            )
        # Power variations
        for col in self.power_columns:
            df[f'{col}_diff'] = df[col].diff()
            df[f'{col}_pct_change'] = df[col].pct_change()
        
        return df
    
    
    def create_statistical_features(self, df: pd.DataFrame,  windows: List[int] = [24]) -> pd.DataFrame:  # Reduced windows
        # Create statistical features using different time windows
        df = df.copy()
        # Specify important columns for lagged features
        important_columns = ['Global_active_power', 'Global_reactive_power']  # Limited to important columns
        for col in important_columns + self.submetering_columns:
            for window in windows:
                # Rolling statistics
                df[f'{col}_rolling_mean_{window}h'] = df[col].rolling(window).mean()
                df[f'{col}_rolling_std_{window}h'] = df[col].rolling(window).std()
            
                # Expanding statistics
                df[f'{col}_expanding_mean'] = df[col].expanding().mean()
                df[f'{col}_expanding_std'] = df[col].expanding().std()

                # Lagged features only for important columns
                if col in important_columns:
                    df[f'{col}_lag_{window}h'] = df[col].shift(window)
                
        return df
            
    def preprocess_and_create_features(self, filepath: str, handle_outliers: bool = True) -> pd.DataFrame:
        # Preprocess the data
        df = self.preprocessor.preprocess_pipeline(filepath, handle_outliers)
        
        # Apply all feature engineering steps
        try:
            # Create time features
            df = self.create_time_features(df)
            self.logger.info("Created time features")
            # Create power features
            df = self.create_power_features(df)
            self.logger.info("Created power features")
            # Create statistical features if requested
            
            df = self.create_statistical_features(df)
            self.logger.info("Created statistical features")
            # Drop any rows with NaN values
            df = df.dropna()
            self.logger.info("Feature engineering completed successfully")
            return df
            
        except Exception as e:
            self.logger.error(f"Error in feature engineering: {str(e)}")
            raise

if __name__ == "__main__":
    # Create features
    feature_engineer = FeatureEngineer()
    
    featured_df = feature_engineer.preprocess_and_create_features('data/raw/household_power_consumption.csv')
    
    featured_df.to_csv('data/processed/featured_power_consumption.csv', index=False)