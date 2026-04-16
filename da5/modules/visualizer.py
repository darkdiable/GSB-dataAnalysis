import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class Visualizer:
    def __init__(self, output_dir='output'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_temperature_trend(self, data, save_path=None):
        fig, ax = plt.subplots(figsize=(14, 6))

        ax.plot(data['date'], data['temperature'], color='steelblue', linewidth=1, alpha=0.7, label='Daily Temperature')

        monthly_avg = data.groupby(data['date'].dt.to_period('M'))['temperature'].mean()
        monthly_avg.index = monthly_avg.index.to_timestamp()
        ax.plot(monthly_avg.index, monthly_avg.values, color='red', linewidth=2, label='Monthly Average')

        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Temperature (°C)', fontsize=12)
        ax.set_title('Temperature Trend Analysis', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.output_dir, 'temperature_trend.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return save_path

    def plot_correlation_heatmap(self, correlation_matrix, save_path=None):
        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='RdYlBu_r',
                    center=0, square=True, linewidths=0.5, ax=ax,
                    cbar_kws={'shrink': 0.8, 'label': 'Correlation'})

        ax.set_title('Meteorological Elements Correlation Heatmap', fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.output_dir, 'correlation_heatmap.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return save_path

    def plot_prediction_comparison(self, actual, predicted, dates=None, save_path=None):
        fig, ax = plt.subplots(figsize=(12, 6))

        if dates is None:
            dates = range(len(actual))

        ax.plot(dates, actual, color='blue', linewidth=2, label='Actual Temperature', marker='o', markersize=4)
        ax.plot(dates, predicted, color='red', linewidth=2, label='Predicted Temperature', marker='s', markersize=4, linestyle='--')

        ax.fill_between(dates, actual, predicted, alpha=0.3, color='gray')

        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Temperature (°C)', fontsize=12)
        ax.set_title('ARIMA Model Prediction Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.output_dir, 'prediction_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return save_path

    def plot_seasonal_analysis(self, data, save_path=None):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        monthly_avg = data.groupby('month')['temperature'].mean()
        axes[0, 0].bar(monthly_avg.index, monthly_avg.values, color='steelblue', alpha=0.7)
        axes[0, 0].set_xlabel('Month')
        axes[0, 0].set_ylabel('Average Temperature (°C)')
        axes[0, 0].set_title('Monthly Average Temperature')
        axes[0, 0].grid(True, alpha=0.3)

        seasons = {'Spring': [3, 4, 5], 'Summer': [6, 7, 8], 'Autumn': [9, 10, 11], 'Winter': [12, 1, 2]}
        seasonal_avg = {}
        for season, months in seasons.items():
            seasonal_avg[season] = data[data['month'].isin(months)]['temperature'].mean()

        axes[0, 1].bar(seasonal_avg.keys(), seasonal_avg.values(), color=['green', 'red', 'orange', 'blue'], alpha=0.7)
        axes[0, 1].set_xlabel('Season')
        axes[0, 1].set_ylabel('Average Temperature (°C)')
        axes[0, 1].set_title('Seasonal Average Temperature')
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].scatter(data['humidity'], data['precipitation'], alpha=0.5, color='steelblue')
        axes[1, 0].set_xlabel('Humidity (%)')
        axes[1, 0].set_ylabel('Precipitation (mm)')
        axes[1, 0].set_title('Humidity vs Precipitation')
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].hist(data['temperature'], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Temperature (°C)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Temperature Distribution')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.output_dir, 'seasonal_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return save_path

    def plot_forecast(self, historical_data, forecast_df, save_path=None):
        fig, ax = plt.subplots(figsize=(14, 6))

        recent_data = historical_data.tail(60)
        ax.plot(recent_data['date'], recent_data['temperature'],
                color='steelblue', linewidth=2, label='Historical Temperature')

        ax.plot(forecast_df['date'], forecast_df['predicted_temperature'],
                color='red', linewidth=2, linestyle='--', marker='o',
                markersize=6, label='Forecast Temperature')

        ax.axvline(x=historical_data['date'].max(), color='gray', linestyle=':', linewidth=1.5, label='Forecast Start')

        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Temperature (°C)', fontsize=12)
        ax.set_title('Temperature Forecast (Next 7 Days)', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.output_dir, 'temperature_forecast.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return save_path
