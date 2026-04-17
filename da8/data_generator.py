"""
数据生成模块：模拟生成共享单车运营数据集
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class BikeDataGenerator:
    """共享单车数据生成器"""
    
    def __init__(self, start_date: str = '2023-01-01', 
                 end_date: str = '2023-12-31',
                 num_stations: int = 10,
                 random_seed: int = 42):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.num_stations = num_stations
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        self.weather_types = ['晴天', '多云', '小雨', '大雨', '雪']
        self.station_ids = [f'S{str(i).zfill(3)}' for i in range(1, num_stations + 1)]
    
    def _generate_weather(self, month: int) -> str:
        """根据月份生成天气状况"""
        if month in [12, 1, 2]:
            weights = [0.3, 0.3, 0.1, 0.1, 0.2]
        elif month in [3, 4, 5]:
            weights = [0.4, 0.3, 0.2, 0.1, 0.0]
        elif month in [6, 7, 8]:
            weights = [0.3, 0.3, 0.25, 0.15, 0.0]
        else:
            weights = [0.35, 0.35, 0.2, 0.1, 0.0]
        return np.random.choice(self.weather_types, p=weights)
    
    def _generate_temperature(self, month: int, hour: int) -> float:
        """根据月份和小时生成温度"""
        base_temp = {
            1: 2, 2: 5, 3: 12, 4: 18, 5: 24, 6: 28,
            7: 32, 8: 30, 9: 25, 10: 18, 11: 10, 12: 4
        }
        hour_effect = 5 * np.sin((hour - 6) * np.pi / 12)
        temp = base_temp[month] + hour_effect + np.random.normal(0, 3)
        return round(max(-10, min(40, temp)), 1)
    
    def _generate_humidity(self, weather: str) -> float:
        """根据天气生成湿度"""
        base_humidity = {
            '晴天': 40, '多云': 55, '小雨': 75, '大雨': 90, '雪': 70
        }
        humidity = base_humidity[weather] + np.random.normal(0, 10)
        return round(max(20, min(100, humidity)), 1)
    
    def _generate_wind_speed(self, weather: str) -> float:
        """根据天气生成风速"""
        base_wind = {
            '晴天': 3, '多云': 4, '小雨': 5, '大雨': 8, '雪': 6
        }
        wind = base_wind[weather] + np.random.exponential(2)
        return round(max(0, min(30, wind)), 1)
    
    def _generate_ridership(self, hour: int, is_weekend: bool, 
                           weather: str, temperature: float) -> int:
        """生成骑行人数"""
        if is_weekend:
            hour_pattern = {
                (0, 6): 5, (6, 9): 30, (9, 12): 80,
                (12, 14): 100, (14, 18): 120, (18, 21): 90, (21, 24): 40
            }
        else:
            hour_pattern = {
                (0, 6): 3, (6, 9): 150, (9, 12): 60,
                (12, 14): 80, (14, 18): 70, (18, 21): 140, (21, 24): 25
            }
        
        base_count = 50
        for (start, end), multiplier in hour_pattern.items():
            if start <= hour < end:
                base_count = multiplier
                break
        
        weather_factor = {
            '晴天': 1.0, '多云': 0.85, '小雨': 0.5, '大雨': 0.2, '雪': 0.3
        }
        
        if temperature < 5 or temperature > 35:
            temp_factor = 0.5
        elif 15 <= temperature <= 25:
            temp_factor = 1.2
        else:
            temp_factor = 0.9
        
        count = base_count * weather_factor[weather] * temp_factor
        count = count * np.random.uniform(0.8, 1.2)
        
        return max(0, int(count))
    
    def generate(self) -> pd.DataFrame:
        """生成完整数据集"""
        data = []
        current_date = self.start_date
        
        while current_date <= self.end_date:
            for hour in range(24):
                timestamp = current_date + timedelta(hours=hour)
                month = timestamp.month
                is_weekend = timestamp.weekday() >= 5
                
                weather = self._generate_weather(month)
                temperature = self._generate_temperature(month, hour)
                humidity = self._generate_humidity(weather)
                wind_speed = self._generate_wind_speed(weather)
                
                for station_id in self.station_ids:
                    ridership = self._generate_ridership(
                        hour, is_weekend, weather, temperature
                    )
                    
                    if np.random.random() < 0.02:
                        ridership = np.nan
                    
                    data.append({
                        'timestamp': timestamp,
                        'station_id': station_id,
                        'weather': weather,
                        'temperature': temperature,
                        'humidity': humidity,
                        'wind_speed': wind_speed,
                        'ridership': ridership
                    })
            
            current_date += timedelta(days=1)
        
        df = pd.DataFrame(data)
        return df
    
    def save_data(self, df: pd.DataFrame, filepath: str):
        """保存数据到CSV文件"""
        df.to_csv(filepath, index=False, encoding='utf-8')
        print(f"数据已保存至: {filepath}")


if __name__ == '__main__':
    generator = BikeDataGenerator()
    df = generator.generate()
    generator.save_data(df, 'data/bike_data.csv')
    print(f"数据集形状: {df.shape}")
    print(df.head())
