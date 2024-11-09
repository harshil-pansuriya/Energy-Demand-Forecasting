import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self):
        # Columns we'll work with
        self.power_columns = ['Global_active_power', 'Global_reactive_power']
        self.submetering_columns = ['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    
    def create_time_features(self, df):
        """
        Creates new features based on time (hour, day, month, etc.)
        """
        df = df.copy()
        
        # Extract time information
        df['hour'] = df['Datetime'].dt.hour
        df['day'] = df['Datetime'].dt.day
        df['month'] = df['Datetime'].dt.month
        df['year'] = df['Datetime'].dt.year
        
        # Add day of week (0=Monday, 6=Sunday)
        df['day_of_week'] = df['Datetime'].dt.dayofweek
        
        # Mark weekends (1 for weekend, 0 for weekday)
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Add time of day category
        df['time_of_day'] = pd.cut(
            df['hour'], 
            bins=[0, 6, 12, 18, 24], 
            labels=['night', 'morning', 'afternoon', 'evening']
        )
        
        return df
    
    def create_power_features(self, df):
        """
        Creates new features related to power consumption
        """
        df = df.copy()
        
        # Calculate total power from submeters
        df['total_submetering'] = df[self.submetering_columns].sum(axis=1)
        
        # Calculate unknown power consumption
        df['unknown_power'] = (df['Global_active_power'] * 1000/60) - df['total_submetering']
        
        # Calculate power factor (relationship between active and reactive power)
        df['power_factor'] = np.where(
            df['Global_active_power'] == 0,
            0,
            df['Global_active_power'] / np.sqrt(
                df['Global_active_power']**2 + df['Global_reactive_power']**2
            )
        )
        
        # Calculate changes in active power (most important for prediction)
        df['active_power_change'] = df['Global_active_power'].diff()
        return df
    
    def create_statistical_features(self, df):
        """
        Creates statistical features using different time windows
        windows: list of hour values to look back
        """
        df = df.copy()
        window=24
        # Main columns to analyze
        imp_col ='Global_active_power'
        
        # Daily average and standard deviation
        df[f'{imp_col}_24h_avg'] = df[imp_col].rolling(window).mean()
        df[f'{imp_col}_24h_std'] = df[imp_col].rolling(window).std()
        
        # Previous day's value (important for prediction)
        df[f'{imp_col}_previous_24h'] = df[imp_col].shift(window)
        return df
    
    def create_all_features(self, df):
        """
        Main function to create all features
        """
        try:
            # First, preprocess the data
            print("Preprocessing data...")
            
            # Create all features
            print("Creating time features...")
            df = self.create_time_features(df)
            
            print("Creating power features...")
            df = self.create_power_features(df)
            
            print("Creating statistical features...")
            df = self.create_statistical_features(df)
            
            # Remove any rows with missing values
            df = df.dropna()
            
            
            # List final features for visibility
            new_features = set(df.columns) - set(['Datetime'] + self.power_columns + self.submetering_columns)
            print("\nNew features created:")
            for feature in sorted(new_features):
                print(f"- {feature}")
                
            print("Feature creation completed!")
            return df
            
        except Exception as e:
            print(f"Error during feature creation: {str(e)}")
            raise

