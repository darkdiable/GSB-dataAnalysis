import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class DataLoader:
    def __init__(self):
        self.data = None

    def generate_sample_data(self, start_date='2023-01-01', periods=365):
        np.random.seed(42)
        dates = pd.date_range(start=start_date, periods=periods, freq='D')

        base_temp = 15
        seasonal = 10 * np.sin(2 * np.pi * np.arange(periods) / 365)
        trend = np.linspace(0, 2, periods)
        noise = np.random.normal(0, 3, periods)
        temperature = base_temp + seasonal + trend + noise

        humidity = 60 + 20 * np.sin(2 * np.pi * np.arange(periods) / 365 + np.pi/4) + np.random.normal(0, 5, periods)
        humidity = np.clip(humidity, 20, 100)

        precipitation = np.zeros(periods)
        for i in range(periods):
            if humidity[i] > 70 and np.random.random() > 0.6:
                precipitation[i] = np.random.exponential(5)

        wind_speed = 10 + np.random.normal(0, 3, periods)
        wind_speed = np.clip(wind_speed, 0, 50)

        pressure = 1013 + np.random.normal(0, 10, periods)

        missing_indices = np.random.choice(periods, size=int(periods * 0.05), replace=False)
        temperature_with_missing = temperature.copy()
        temperature_with_missing[missing_indices[:len(missing_indices)//2]] = np.nan
        humidity_with_missing = humidity.copy()
        humidity_with_missing[missing_indices[len(missing_indices)//2:]] = np.nan

        self.data = pd.DataFrame({
            'date': dates,
            'temperature': temperature_with_missing,
            'humidity': humidity_with_missing,
            'precipitation': precipitation,
            'wind_speed': wind_speed,
            'pressure': pressure
        })

        outlier_indices = np.random.choice(periods, size=5, replace=False)
        self.data.loc[outlier_indices, 'temperature'] = self.data.loc[outlier_indices, 'temperature'] + np.random.choice([-30, 30], size=5)

        return self.data

    def load_from_csv(self, filepath):
        self.data = pd.read_csv(filepath)
        if 'date' in self.data.columns:
            self.data['date'] = pd.to_datetime(self.data['date'])
        return self.data

    def get_data(self):
        return self.data
