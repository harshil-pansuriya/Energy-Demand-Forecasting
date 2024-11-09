import pandas as pd
import numpy as np
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def main():
    
    raw_data='data/raw/household_power_consumption.csv'
    processed_data='data/processed/featuredd_power_consumption.csv'
    
    try:
        print('Starting data processing')
        
        preprocessor=DataPreprocessor()
        feature_engg=FeatureEngineer()
        
        print('\nCleaning data')
        cleaned_data=preprocessor.preprocess_pipeline(filepath=raw_data, handle_outliers=True)
        
        print('\n Data Cleaning done')
        
        print('\nFeature Engg. Started')
        featured_data=feature_engg.create_all_features(cleaned_data)
        print('\n Feature engg done')
        featured_data.to_csv(processed_data, index=False)
        
        print('\n Data Saved to : {processed_data}')
        
        print("\nProcessing Summary:")
        print(f"Raw data shape: {cleaned_data.shape}")
        print(f"Processed data shape: {featured_data.shape}")
        print(f"Number of features created: {featured_data.shape[1]}")
        
        return featured_data
        
    except Exception as e:
        print(f"\n Error Occured in Data Processing: {str(e)}")
        raise
    
if __name__ == "__main__":
    processed_data= main()
    print("\Processing Completed successfully!")