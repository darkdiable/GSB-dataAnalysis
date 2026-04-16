import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class DataFetcher:
    def __init__(self):
        self.data = None

    def generate_sample_data(self, start_date='2020-01-01', end_date='2024-12-31', city='北京'):
        """
        生成模拟城市气象数据
        """
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        np.random.seed(42)
        
        data = []
        for date in date_range:
            month = date.month
            
            # 模拟季节性温度变化
            base_temp = 15 + 15 * np.sin(2 * np.pi * (month - 3) / 12)
            temperature = base_temp + np.random.normal(0, 5)
            
            # 湿度与温度呈负相关
            base_humidity = 60 - 10 * np.sin(2 * np.pi * (month - 3) / 12)
            humidity = base_humidity + np.random.normal(0, 10)
            humidity = np.clip(humidity, 20, 95)
            
            # 降水量（夏季较多）
            if month in [6, 7, 8]:
                precipitation = max(0, np.random.exponential(8))
            else:
                precipitation = max(0, np.random.exponential(2))
            
            # 风速
            wind_speed = np.random.gamma(2, 2)
            
            # 气压
            pressure = 1013 + np.random.normal(0, 15)
            
            data.append({
                'date': date,
                'city': city,
                'temperature': round(temperature, 2),
                'humidity': round(humidity, 2),
                'precipitation': round(precipitation, 2),
                'wind_speed': round(wind_speed, 2),
                'pressure': round(pressure, 2)
            })
        
        self.data = pd.DataFrame(data)
        return self.data

    def load_from_csv(self, file_path):
        """
        从CSV文件加载数据
        """
        self.data = pd.read_csv(file_path, parse_dates=['date'])
        return self.data

    def save_to_csv(self, file_path):
        """
        保存数据到CSV文件
        """
        if self.data is not None:
            self.data.to_csv(file_path, index=False)
            print(f"数据已保存到: {file_path}")
        else:
            print("没有数据可保存")

    def get_data(self):
        """
        获取当前数据
        """
        return self.data
