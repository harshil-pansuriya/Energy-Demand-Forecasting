import gradio as gr
import pandas as pd
import numpy as np
import tensorflow as tf

from src.prediction import (
    predict_next_hours, plot_predictions, predict_next_days, plot_actual_vs_predicted
)
from src.visualize import (
    calculate_statistics, plot_power_consumption, plot_average_power_by_month_and_day,
    plot_average_power_by_hour,plot_weekly_power_heatmap,plot_time_series_components
)

import warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

featured_data = 'data/processed/featured_power_consumption.csv'
model_path = 'models/best_model.keras'
scaler_path = 'models/scaler.pkl'

def display_statistics_and_visualizations():
    df = pd.read_csv(featured_data, parse_dates=['Datetime'])
    stats = calculate_statistics(df)
    
    power_consumption_img = plot_power_consumption(df) or "Error in Power Consumption Plot"
    avg_power_img = plot_average_power_by_month_and_day(df) or "Error in Average Power Plot"
    avg_hour_img = plot_average_power_by_hour(df) or "Error in Average Power by Hour Plot"
    weekly_heatmap_img = plot_weekly_power_heatmap(df) or "Error in Weekly Heatmap Plot"
    time_series_img = plot_time_series_components(df) or "Error in Time Series Components"
    
    return stats, power_consumption_img, avg_power_img, avg_hour_img, weekly_heatmap_img, time_series_img

def predict_energy_consumption(hours, days):
    if hours > 0:
        predictions_df, peak_demand, low_demand = predict_next_hours(
            data_path=featured_data,model_path=model_path,
            scaler_path=scaler_path,hours=hours,look_back=168
        )
    else:
        predictions_df, peak_demand, low_demand = predict_next_days(
            data_path=featured_data,model_path=model_path,
            scaler_path=scaler_path, days=days,look_back=168
        )
        
    prediction_img = plot_predictions(predictions_df, peak_demand, low_demand) or "Error in Prediction Plot"
    
    total_energy_demand = predictions_df['Predicted_Power'].sum()
    demand_info = (
        f"Peak Demand: {peak_demand:.2f} kW\n"
        f"Low Demand: {low_demand:.2f} kW\n"
        f"Total Energy Demand: {total_energy_demand:.2f} kWh"
    )
        
    return "Prediction Results", prediction_img, demand_info

# Gradio Interface
def gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Energy Demand Forecasting")

        gr.Markdown("""
        Welcome to the Energy Demand Forecasting. It provides insights into energy consumption patterns and predicts future energy demand using time series models. Explore the visualizations to understand past trends and use the prediction tool to forecast future energy needs.
        """)
        
        with gr.Tab("Visualization"):
            gr.Markdown("## Statistics and Visualizations")
            gr.Markdown("Explore the historical data through various visualizations and statistics.")
            stats, power_img, avg_img, avg_hour_img, weekly_heatmap_img, time_series_img = display_statistics_and_visualizations()
            gr.Textbox(value=stats, label="Statistics")
            gr.Image(value=power_img, label="Power Consumption")
            gr.Image(value=avg_img, label="Average Power by Month and Day")
            gr.Image(value=avg_hour_img, label="Average Power by Hour")
            gr.Image(value=weekly_heatmap_img, label="Weekly Power Heatmap")
            gr.Image(value=time_series_img, label="Time Series Components")

        with gr.Tab("Prediction"):
            gr.Markdown("## Actual vs Predicted Power Consumption")
            gr.Markdown("Compare the model's predictions with actual data to evaluate performance.")
            actual_vs_predicted_img = plot_actual_vs_predicted(model_path='models/best_model.keras', steps=1000)
            gr.Image(value=actual_vs_predicted_img, label="Actual vs Predicted")

            gr.Markdown("## Predict Future Energy Consumption")
            gr.Markdown("Use the sliders to set the number of hours or days you want to predict.")
            hours_input = gr.Slider(minimum=0, maximum=24, step=1, label="Hours to Predict")
            days_input = gr.Slider(minimum=0, maximum=5, step=1, label="Days to Predict")
            predict_button = gr.Button("Predict")
            title_output = gr.Markdown(label="Prediction Title")
            predict_output = gr.Image(label="Predicted Power Consumption")
            demand_output = gr.Textbox(label="Demand Information")

            predict_button.click(
                predict_energy_consumption, 
                inputs=[hours_input, days_input], 
                outputs=[title_output, predict_output, demand_output]
            )
    demo.launch()

if __name__ == "__main__":
    gradio_interface()