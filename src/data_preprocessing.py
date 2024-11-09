import pandas as pd
import numpy as np
import logging

class DataPreprocessor:
    
    def __init__(self, logging_level = logging.INFO):
        # Initialize DataPreprocessor with configuration
        self.numerical_columns = [
            'Global_active_power', 'Global_reactive_power',
            'Voltage', 'Global_intensity', 'Sub_metering_1',
            'Sub_metering_2', 'Sub_metering_3'
        ]
        
        # Setup logging
        logging.basicConfig(level=logging_level)
        self.logger = logging.getLogger(__name__)
        
    def load_data(self, filepath):
       
        # Load and perform initial parsing of the dataset
        try:
            df = pd.read_csv(filepath)
            self.logger.info(f"Successfully loaded data from {filepath}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def handle_datetime(self, df):

        #Handle datetime parsing with mixed formats
        df = df.copy()
        try:
            # Split dataset based on date format
            split_idx = 186352  # Based on your EDA findings
            df_part1 = df.iloc[:split_idx].copy()
            df_part2 = df.iloc[split_idx:].copy()
            
            # Parse datetime for both parts
            df_part1['Datetime'] = pd.to_datetime(df_part1['Date'] + ' ' + df_part1['Time'],dayfirst=True, errors='coerce')
            df_part2['Datetime'] = pd.to_datetime(df_part2['Date'] + ' ' + df_part2['Time'], dayfirst=True, errors='coerce')
            
            # Combine parts
            df = pd.concat([df_part1, df_part2], ignore_index=True)
            # Drop original columns and any rows with invalid dates
            df = df.drop(columns=['Date', 'Time', 'index'], errors='ignore')
            df = df.dropna(subset=['Datetime'])
            self.logger.info("Successfully processed datetime")
            return df
        
        except Exception as e:
            self.logger.error(f"Error processing datetime: {str(e)}")
            raise
        
    def convert_numeric_columns(self, df):
       
        # Convert columns to appropriate numeric types
        df = df.copy()
        df[self.numerical_columns] = df[self.numerical_columns].apply(pd.to_numeric, errors='coerce')
        return df
    
    def handle_outliers(self, df, method = 'iqr', threshold= 1.5):
        
        # Handle outliers using specified method
        df = df.copy()
        for col in self.numerical_columns:
            if method == 'iqr':
                Q1, Q3 = df[col].quantile([0.25, 0.75])
                IQR = Q3 - Q1
                lower_bound, upper_bound = Q1 - threshold * IQR, Q3 + threshold * IQR
            elif method == 'zscore':
                mean, std = df[col].mean(), df[col].std()
                lower_bound, upper_bound = mean - threshold * std, mean + threshold * std
            else:
                raise ValueError(f"Unknown outlier handling method: {method}")
            
            # Cap outliers at boundaries
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            
        return df
    
    def preprocess_pipeline(self, filepath, handle_outliers= True):
        
        # Complete preprocessing pipeline
        try:
            # Load data
            df = self.load_data(filepath) 
            df = self.handle_datetime(df)
            df = self.convert_numeric_columns(df)
            if handle_outliers:
                df = self.handle_outliers(df)
            # Sort by datetime
            df = df.sort_values('Datetime').reset_index(drop=True)
            self.logger.info("Preprocessing completed successfully")
            return df
            
        except Exception as e:
            self.logger.error(f"Error in preprocessing pipeline: {str(e)}")
            raise
        