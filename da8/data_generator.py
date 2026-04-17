"""
数据生成模块：模拟共享单车运营数据集
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class BikeSharingDataGenerator:
    """共享单车数据生成器"""
    
    def __init__(self, n_samples=10000, random_state=42):
        self.n_samples = n_samples
        self.random_state = random_state
        np.random.seed(random_state)
        
    def generate(self):
        """
        生成模拟数据集
        
        Returns:
            pd.DataFrame: 包含以下列的数据集
                - timestamp: 时间戳
                - station_id: 站点ID
                - weather: 天气状况
                - temperature: 温度(摄氏度)
                - humidity: 湿度(%)
                - windspeed: 风速(km/h)
                - ridership: 骑行人数
        """
        # 生成时间戳（模拟一个月的数据，每小时记录）
        start_date = datetime(2024, 6, 1)
        timestamps = [start_date + timedelta(hours=i) for i in range(self.n_samples)]
        
        # 站点ID（10个站点）
        station_ids = np.random.randint(1, 11, self.n_samples)
        
        # 天气状况
        weather_conditions = ['晴天', '多云', '小雨', '大雨', '阴天']
        weather_probs = [0.4, 0.25, 0.15, 0.05, 0.15]
        weather = np.random.choice(weather_conditions, self.n_samples, p=weather_probs)
        
        # 温度（根据月份和天气调整）
        base_temp = 25
        temperature = base_temp + np.random.normal(0, 5, self.n_samples)
        # 根据天气调整温度
        for i, w in enumerate(weather):
            if w == '大雨':
                temperature[i] -= 3
            elif w == '晴天':
                temperature[i] += 2
        temperature = np.clip(temperature, 5, 40)
        
        # 湿度
        humidity = np.random.normal(65, 15, self.n_samples)
        humidity = np.clip(humidity, 20, 100)
        
        # 风速
        windspeed = np.random.gamma(2, 3, self.n_samples)
        windspeed = np.clip(windspeed, 0, 30)
        
        # 骑行人数（基于多因素的复杂关系）
        ridership = self._generate_ridership(timestamps, weather, temperature, humidity, windspeed)
        
        # 创建DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'station_id': station_ids,
            'weather': weather,
            'temperature': np.round(temperature, 1),
            'humidity': np.round(humidity, 1),
            'windspeed': np.round(windspeed, 1),
            'ridership': ridership
        })
        
        # 随机引入一些缺失值（约2%）
        df = self._introduce_missing_values(df)
        
        return df
    
    def _generate_ridership(self, timestamps, weather, temperature, humidity, windspeed):
        """基于多因素生成骑行人数"""
        n = len(timestamps)
        ridership = np.zeros(n)
        
        for i in range(n):
            base = 50  # 基础骑行量
            
            # 时间因素（小时）
            hour = timestamps[i].hour
            if 7 <= hour <= 9 or 17 <= hour <= 19:  # 早晚高峰
                base += 80
            elif 10 <= hour <= 16:  # 白天
                base += 40
            elif 0 <= hour <= 5:  # 深夜
                base -= 30
            
            # 工作日因素
            if timestamps[i].weekday() < 5:  # 工作日
                base += 20
            else:  # 周末
                base += 10
            
            # 天气因素
            weather_factor = {
                '晴天': 30,
                '多云': 15,
                '阴天': 0,
                '小雨': -25,
                '大雨': -50
            }
            base += weather_factor.get(weather[i], 0)
            
            # 温度因素（最适宜温度20-28度）
            temp = temperature[i]
            if 20 <= temp <= 28:
                base += 20
            elif 15 <= temp < 20 or 28 < temp <= 32:
                base += 5
            elif temp < 10 or temp > 35:
                base -= 20
            
            # 湿度因素
            hum = humidity[i]
            if hum > 85:
                base -= 15
            elif hum < 40:
                base += 5
            
            # 风速因素
            wind = windspeed[i]
            if wind > 20:
                base -= 20
            elif wind > 15:
                base -= 10
            
            # 添加随机噪声
            base += np.random.normal(0, 15)
            
            ridership[i] = max(0, int(base))
        
        return ridership.astype(int)
    
    def _introduce_missing_values(self, df, missing_rate=0.02):
        """随机引入缺失值"""
        df_missing = df.copy()
        n_missing = int(len(df) * missing_rate)
        
        # 随机选择行和列
        cols = ['temperature', 'humidity', 'windspeed']
        for _ in range(n_missing):
            row_idx = np.random.randint(0, len(df))
            col_idx = np.random.choice(cols)
            df_missing.loc[row_idx, col_idx] = np.nan
        
        return df_missing
    
    def save_to_csv(self, df, filepath='bike_sharing_data.csv'):
        """保存数据到CSV文件"""
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"数据已保存至: {filepath}")
        return filepath


if __name__ == '__main__':
    # 测试数据生成
    generator = BikeSharingDataGenerator(n_samples=5000)
    data = generator.generate()
    print(f"生成数据形状: {data.shape}")
    print("\n数据预览:")
    print(data.head(10))
    print("\n数据统计信息:")
    print(data.describe())
    print("\n缺失值统计:")
    print(data.isnull().sum())
