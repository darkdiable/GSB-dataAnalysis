import pandas as pd
import numpy as np
from scipy import stats


class FeatureAnalyzer:
    def __init__(self, data):
        self.data = data
        self.analysis_results = {}

    def analyze_temperature_seasonality(self):
        monthly_temp = self.data.groupby('month')['temperature'].agg(['mean', 'std', 'min', 'max'])

        seasonal_stats = {
            'spring': self.data[self.data['month'].isin([3, 4, 5])]['temperature'].mean(),
            'summer': self.data[self.data['month'].isin([6, 7, 8])]['temperature'].mean(),
            'autumn': self.data[self.data['month'].isin([9, 10, 11])]['temperature'].mean(),
            'winter': self.data[self.data['month'].isin([12, 1, 2])]['temperature'].mean()
        }

        self.analysis_results['temperature_seasonality'] = {
            'monthly_stats': monthly_temp.to_dict(),
            'seasonal_stats': seasonal_stats
        }

        return monthly_temp, seasonal_stats

    def analyze_humidity_precipitation_correlation(self):
        humidity = self.data['humidity'].values
        precipitation = self.data['precipitation'].values

        correlation, p_value = stats.pearsonr(humidity, precipitation)

        high_humidity_days = self.data[self.data['humidity'] > 70]
        low_humidity_days = self.data[self.data['humidity'] <= 70]

        precip_high_humidity = high_humidity_days['precipitation'].mean()
        precip_low_humidity = low_humidity_days['precipitation'].mean()

        self.analysis_results['humidity_precipitation'] = {
            'correlation': correlation,
            'p_value': p_value,
            'precip_high_humidity': precip_high_humidity,
            'precip_low_humidity': precip_low_humidity
        }

        return correlation, p_value

    def calculate_correlation_matrix(self):
        numeric_cols = ['temperature', 'humidity', 'precipitation', 'wind_speed', 'pressure']
        available_cols = [col for col in numeric_cols if col in self.data.columns]

        correlation_matrix = self.data[available_cols].corr()

        self.analysis_results['correlation_matrix'] = correlation_matrix.to_dict()

        return correlation_matrix

    def analyze_temperature_distribution(self):
        temp_stats = {
            'mean': self.data['temperature'].mean(),
            'median': self.data['temperature'].median(),
            'std': self.data['temperature'].std(),
            'skewness': stats.skew(self.data['temperature'].dropna()),
            'kurtosis': stats.kurtosis(self.data['temperature'].dropna())
        }

        self.analysis_results['temperature_distribution'] = temp_stats

        return temp_stats

    def analyze_extreme_weather(self):
        temp_q10 = self.data['temperature'].quantile(0.1)
        temp_q90 = self.data['temperature'].quantile(0.9)

        cold_days = self.data[self.data['temperature'] < temp_q10]
        hot_days = self.data[self.data['temperature'] > temp_q90]

        extreme_stats = {
            'cold_days_count': len(cold_days),
            'hot_days_count': len(hot_days),
            'cold_threshold': temp_q10,
            'hot_threshold': temp_q90,
            'coldest_day': self.data.loc[self.data['temperature'].idxmin(), 'date'].strftime('%Y-%m-%d'),
            'hottest_day': self.data.loc[self.data['temperature'].idxmax(), 'date'].strftime('%Y-%m-%d')
        }

        self.analysis_results['extreme_weather'] = extreme_stats

        return extreme_stats

    def get_analysis_results(self):
        return self.analysis_results
