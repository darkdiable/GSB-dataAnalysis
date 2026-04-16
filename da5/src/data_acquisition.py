import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_weather_data(start_date='2023-01-01', days=365):
    start = datetime.strptime(start_date, '%Y-%m-%d')
    dates = [start + timedelta(days=i) for i in range(days)]
    
    np.random.seed(42)
    day_of_year = np.arange(days)
    
    temperature = 15 + 15 * np.sin(2 * np.pi * day_of_year / 365) + np.random.normal(0, 3, days)
    humidity = 60 + 20 * np.sin(2 * np.pi * day_of_year / 365 + np.pi) + np.random.normal(0, 8, days)
    humidity = np.clip(humidity, 20, 100)
    precipitation = np.maximum(0, 5 * np.sin(2 * np.pi * day_of_year / 365 + np.pi/2) + np.random.normal(0, 3, days))
    wind_speed = 5 + 3 * np.random.normal(0, 2, days)
    wind_speed = np.clip(wind_speed, 0, 20)
    pressure = 1013 + 5 * np.sin(2 * np.pi * day_of_year / 365) + np.random.normal(0, 2, days)
    
    data = pd.DataFrame({
        'date': dates,
        'temperature': temperature,
        'humidity': humidity,
        'precipitation': precipitation,
        'wind_speed': wind_speed,
        'pressure': pressure
    })
    
    return data


def load_or_generate_data(data_path=None, start_date='2023-01-01', days=365):
    if data_path:
        try:
            data = pd.read_csv(data_path, parse_dates=['date'])
            return data
        except FileNotFoundError:
            print(f"文件 {data_path} 不存在，正在生成模拟数据...")
    
    return generate_weather_data(start_date, days)
