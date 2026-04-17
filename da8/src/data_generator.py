import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class BikeShareDataGenerator:
    def __init__(self, start_date='2024-01-01', days=365, stations=20, random_seed=42):
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.days = days
        self.stations = stations
        np.random.seed(random_seed)
        
    def generate_timestamps(self):
        timestamps = []
        for day in range(self.days):
            current_date = self.start_date + timedelta(days=day)
            for hour in range(24):
                for station in range(1, self.stations + 1):
                    ts = current_date + timedelta(hours=hour)
                    timestamps.append((ts, station))
        return timestamps
    
    def generate_weather(self, hour, month):
        weather_probs = {
            '晴天': 0.6, '多云': 0.25, '小雨': 0.1, '大雨': 0.05
        }
        if month in [6, 7, 8]:
            weather_probs['小雨'] = 0.15
            weather_probs['大雨'] = 0.08
        
        probs = np.array(list(weather_probs.values()))
        probs = probs / probs.sum()
        weather = np.random.choice(list(weather_probs.keys()), p=probs)
        
        base_temp = 10 + 15 * np.sin((month - 4) * np.pi / 6)
        temp = base_temp + 5 * np.sin((hour - 14) * np.pi / 12) + np.random.normal(0, 3)
        
        humidity = 60 + 20 * np.random.random()
        if weather in ['小雨', '大雨']:
            humidity = 80 + 15 * np.random.random()
        
        wind_speed = 5 + 10 * np.random.random()
        if weather == '大雨':
            wind_speed = 15 + 10 * np.random.random()
            
        return weather, round(temp, 1), round(humidity, 1), round(wind_speed, 1)
    
    def calculate_rider_count(self, hour, weekday, weather, temp, station_id):
        base_riders = 50
        
        hour_effect = np.zeros(24)
        hour_effect[7:9] = 3  
        hour_effect[17:19] = 2.5  
        hour_effect[10:16] = 1.5  
        hour_effect[22:24] = 0.3
        hour_effect[0:6] = 0.1
        
        weekday_effect = 1.2 if weekday < 5 else 0.8
        
        weather_effect = {
            '晴天': 1.2,
            '多云': 1.0,
            '小雨': 0.5,
            '大雨': 0.15
        }.get(weather, 0.8)
        
        temp_effect = 1 / (1 + np.exp(-(temp - 20) / 5))
        
        station_effect = 0.5 + (station_id % 5) * 0.3
        
        noise = np.random.normal(0, 0.2)
        
        riders = base_riders * hour_effect[hour] * weekday_effect * weather_effect * temp_effect * station_effect * (1 + noise)
        return max(0, int(round(riders)))
    
    def generate_dataset(self):
        timestamps = self.generate_timestamps()
        data = []
        
        for ts, station_id in timestamps:
            hour = ts.hour
            month = ts.month
            weekday = ts.weekday()
            
            weather, temp, humidity, wind_speed = self.generate_weather(hour, month)
            riders = self.calculate_rider_count(hour, weekday, weather, temp, station_id)
            
            data.append({
                'timestamp': ts,
                'station_id': station_id,
                'weather': weather,
                'temperature': temp,
                'humidity': humidity,
                'wind_speed': wind_speed,
                'riders': riders
            })
        
        df = pd.DataFrame(data)
        
        missing_mask = np.random.choice([True, False], size=len(df), p=[0.02, 0.98])
        df.loc[missing_mask, 'temperature'] = np.nan
        df.loc[missing_mask, 'humidity'] = np.nan
        
        return df
    
    def save_dataset(self, df, filepath='data/bike_share_data.csv'):
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"数据集已保存至: {filepath}")


if __name__ == '__main__':
    generator = BikeShareDataGenerator()
    df = generator.generate_dataset()
    print(f"生成数据集形状: {df.shape}")
    print(df.head())
    generator.save_dataset(df)
