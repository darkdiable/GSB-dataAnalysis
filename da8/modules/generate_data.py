import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class DataGenerator:
    
    def __init__(self, start_date: str = '2023-01-01', 
                 end_date: str = '2023-12-31',
                 num_stations: int = 10,
                 seed: int = 42):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.num_stations = num_stations
        self.seed = seed
        np.random.seed(seed)
        
        self.weather_types = ['晴天', '多云', '小雨', '大雨', '阴天']
        self.weather_weights = [0.35, 0.25, 0.15, 0.10, 0.15]
    
    def _generate_timestamps(self) -> pd.DatetimeIndex:
        return pd.date_range(start=self.start_date, end=self.end_date, freq='H')
    
    def _generate_weather(self, n: int) -> np.ndarray:
        return np.random.choice(self.weather_types, size=n, p=self.weather_weights)
    
    def _generate_temperature(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        hour = timestamps.hour
        month = timestamps.month
        
        base_temp = 15 + 10 * np.sin((month - 1) * np.pi / 6)
        hourly_variation = 5 * np.sin((hour - 6) * np.pi / 12)
        noise = np.random.normal(0, 3, len(timestamps))
        
        temperature = base_temp + hourly_variation + noise
        return np.clip(temperature, -5, 40)
    
    def _generate_humidity(self, n: int) -> np.ndarray:
        humidity = np.random.normal(60, 15, n)
        return np.clip(humidity, 20, 100)
    
    def _generate_wind_speed(self, n: int) -> np.ndarray:
        wind_speed = np.random.exponential(3, n)
        return np.clip(wind_speed, 0, 30)
    
    def _generate_ridership(self, timestamps: pd.DatetimeIndex, 
                           temperature: np.ndarray,
                           humidity: np.ndarray,
                           weather: np.ndarray) -> np.ndarray:
        hour = timestamps.hour
        weekday = timestamps.weekday
        
        base_ridership = 100
        
        hour_pattern = np.where(
            (hour >= 7) & (hour <= 9), 2.5,
            np.where((hour >= 17) & (hour <= 19), 2.8,
            np.where((hour >= 10) & (hour <= 16), 1.2,
            np.where((hour >= 20) & (hour <= 22), 0.8, 0.3))))
        
        weekday_pattern = np.where(weekday < 5, 1.0, 0.6)
        
        temp_effect = np.where(
            (temperature >= 15) & (temperature <= 28), 1.3,
            np.where(temperature < 5, 0.3,
            np.where(temperature > 35, 0.4, 1.0)))
        
        humidity_effect = np.where(humidity > 80, 0.8, 1.0)
        
        weather_effect = np.where(weather == '晴天', 1.3,
                        np.where(weather == '多云', 1.1,
                        np.where(weather == '阴天', 0.9,
                        np.where(weather == '小雨', 0.5, 0.2))))
        
        ridership = base_ridership * hour_pattern * weekday_pattern * \
                   temp_effect * humidity_effect * weather_effect
        noise = np.random.normal(0, 15, len(timestamps))
        ridership = ridership + noise
        
        return np.clip(ridership, 0, 500).astype(int)
    
    def generate(self) -> pd.DataFrame:
        timestamps = self._generate_timestamps()
        n = len(timestamps)
        
        station_ids = np.random.randint(1, self.num_stations + 1, n)
        weather = self._generate_weather(n)
        temperature = self._generate_temperature(timestamps)
        humidity = self._generate_humidity(n)
        wind_speed = self._generate_wind_speed(n)
        ridership = self._generate_ridership(timestamps, temperature, humidity, weather)
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'station_id': station_ids,
            'weather': weather,
            'temperature': np.round(temperature, 1),
            'humidity': np.round(humidity, 1),
            'wind_speed': np.round(wind_speed, 1),
            'ridership': ridership
        })
        
        missing_indices = np.random.choice(n, size=int(n * 0.02), replace=False)
        for idx in missing_indices:
            col = np.random.choice(['temperature', 'humidity', 'wind_speed'])
            df.loc[idx, col] = np.nan
        
        return df
    
    def save_data(self, df: pd.DataFrame, filepath: str) -> None:
        df.to_csv(filepath, index=False, encoding='utf-8')
        print(f"数据已保存至: {filepath}")
