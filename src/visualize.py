import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import plot_to_image

def display_features(df):
    if df is not None:
        print("Features in the dataset:")
        print(df.columns.tolist())

def calculate_statistics(df):
    if df is not None:
        min_power = df['Global_active_power'].min()
        max_power = df['Global_active_power'].max()
        avg_power = df['Global_active_power'].mean()
        return f"Minimum Power Consumption: {min_power:.4f} kW\n" \
               f"Maximum Power Consumption: {max_power:.4f} kW\n" \
               f"Average Power Consumption: {avg_power:.4f} kW"
    return "Error calculating statistics: Data not available"

def plot_power_consumption(df):
    if df is not None:
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(df['Datetime'], df['Global_active_power'], color='purple')
        ax.set_ylabel('Global Active Power (kW)', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_title('Active Power Consumption for a Particular Time Frame', fontsize=14)
        ax.grid(True)
        sns.despine(bottom=True, left=True)
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        fig.autofmt_xdate()
        return plot_to_image(fig)
    print("Error plotting power consumption: Data not available")
    return None

def plot_average_power_by_month_and_day(df):
    if df is not None:
        df['month'] = df['Datetime'].dt.month
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Average Global Active Power by Month
        df.groupby('month').Global_active_power.mean().plot(ax=axes[0], color='purple')
        axes[0].set_title('Average Global Active Power by Month', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Month')
        axes[0].set_ylabel('Global Active Power')

        # Average Global Active Power by Day
        df.groupby('day').Global_active_power.mean().plot(ax=axes[1], color='green')
        axes[1].set_title('Average Global Active Power by Day', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Day')
        axes[1].set_ylabel('Global Active Power')

        plt.tight_layout()
        return plot_to_image(fig)
    print("Error plotting average power by month and day: Data not available")
    return None

def plot_average_power_by_hour(df):
    if df is not None:
        fig, ax = plt.subplots(figsize=(10, 5))
        df.groupby('hour').Global_active_power.mean().plot(ax=ax, color='blue')
        ax.set_title('Average Global Active Power by Hour of Day', fontsize=14, fontweight='bold')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Global Active Power (kW)')
        ax.grid(True)

        return plot_to_image(fig)
    print("Error plotting average power by hour: Data not available")
    return None

def plot_weekly_power_heatmap(df):
    if df is not None:
        df['hour'] = df['Datetime'].dt.hour
        df['day_of_week'] = df['Datetime'].dt.day_name()
        
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot_table = df.pivot_table(values='Global_active_power', index='hour', columns='day_of_week', aggfunc='mean').reindex(columns=days_order)

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(pivot_table, ax=ax, cmap='YlGnBu', cbar_kws={'label': 'Average Power (kW)'})
        ax.set_title('Weekly Power Consumption Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Day of Week')
        ax.set_ylabel('Hour of Day')

        return plot_to_image(fig)
    print("Error plotting weekly power heatmap: Data not available")
    return None

def plot_time_series_components(df, sample_days=7):
    if df is not None:
        sample_data = df.set_index('Datetime').iloc[:sample_days*24]

        fig, axes = plt.subplots(5, 1, figsize=(10, 20))
        # Original Data
        axes[0].plot(sample_data.index, sample_data['Global_active_power'], label='Original')
        axes[0].set_title('Original Time Series')
        axes[0].set_ylabel('Global Active Power')
        axes[0].legend()
        axes[0].grid(True)
        # Trend
        axes[1].plot(sample_data.index, sample_data['Global_active_power_trend'], label='Trend', color='green')
        axes[1].set_title('Trend Component')
        axes[1].set_ylabel('Trend Value')
        axes[1].legend()
        axes[1].grid(True)
        # Seasonal
        axes[2].plot(sample_data.index, sample_data['Global_active_power_seasonal'], label='Seasonal', color='red')
        axes[2].set_title('Seasonal Component')
        axes[2].set_ylabel('Seasonal Value')
        axes[2].legend()
        axes[2].grid(True)
        # Residual
        axes[3].plot(sample_data.index, sample_data['Global_active_power_residual'], label='Residual', color='purple')
        axes[3].set_title('Residual Component')
        axes[3].set_ylabel('Residual Value')
        axes[3].legend()
        axes[3].grid(True)
        # Moving Average and Smoothing
        axes[4].plot(sample_data.index, sample_data['Global_active_power'], label='Original', alpha=0.5)
        axes[4].plot(sample_data.index, sample_data['Global_active_power_smoothed_ma'], label='Moving Average', color='orange')
        axes[4].plot(sample_data.index, sample_data['Global_active_power_smoothed_sg'], label='Savitzky-Golay', color='red')
        axes[4].set_title('Smoothing Comparison')
        axes[4].set_ylabel('Power Value')
        axes[4].legend()
        axes[4].grid(True)

        plt.tight_layout()
        return plot_to_image(fig)
    print("Error plotting time series components: Data not available")
    return None