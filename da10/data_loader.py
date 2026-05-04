import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


def generate_sample_data(start_date='2023-01-01', end_date='2025-12-31'):
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    dates = pd.date_range(start=start, end=end, freq='D')
    
    n_days = len(dates)
    day_of_year = dates.dayofyear
    
    temperature_base = 15 + 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    temperature = temperature_base + np.random.normal(0, 3, n_days)
    
    humidity_base = 60 + 20 * np.cos(2 * np.pi * (day_of_year - 30) / 365)
    humidity = np.clip(humidity_base + np.random.normal(0, 10, n_days), 20, 100)
    
    precipitation_base = np.where(
        (day_of_year > 120) & (day_of_year < 270),
        8,
        2
    )
    precipitation = np.maximum(
        0,
        precipitation_base + np.random.exponential(3, n_days) - 2
    )
    for i in range(20):
        idx = np.random.randint(0, n_days)
        precipitation[idx] += np.random.uniform(20, 100)
    
    wind_speed_base = 5 + 3 * np.sin(2 * np.pi * (day_of_year - 100) / 365)
    wind_speed = np.maximum(0, wind_speed_base + np.random.normal(0, 2, n_days))
    
    df = pd.DataFrame({
        'date': dates,
        'temperature': temperature,
        'humidity': humidity,
        'precipitation': precipitation,
        'wind_speed': wind_speed
    })
    
    return df


def load_data(file_path=None):
    if file_path and os.path.exists(file_path):
        df = pd.read_csv(file_path, parse_dates=['date'])
        df['date'] = pd.to_datetime(df['date'])
        return df
    else:
        print("未找到外部数据文件，正在生成示例数据...")
        return generate_sample_data()


def save_raw_data(df, output_path='output/raw_data.csv'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"原始数据已保存到: {output_path}")


if __name__ == '__main__':
    df = generate_sample_data()
    print(f"生成数据形状: {df.shape}")
    print(f"数据列: {df.columns.tolist()}")
    print(df.head())
