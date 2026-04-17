"""
数据预处理模块：处理缺失值、提取时间特征、独热编码
"""
import pandas as pd
import numpy as np


class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.preprocessing_log = []
    
    def handle_missing_values(self, strategy: str = 'median') -> pd.DataFrame:
        """处理缺失值"""
        missing_count = self.df['ridership'].isna().sum()
        
        if strategy == 'median':
            fill_value = self.df['ridership'].median()
            self.df['ridership'].fillna(fill_value, inplace=True)
            self.preprocessing_log.append(
                f"使用中位数({fill_value:.1f})填充{missing_count}个缺失值"
            )
        elif strategy == 'mean':
            fill_value = self.df['ridership'].mean()
            self.df['ridership'].fillna(fill_value, inplace=True)
            self.preprocessing_log.append(
                f"使用均值({fill_value:.1f})填充{missing_count}个缺失值"
            )
        elif strategy == 'interpolation':
            self.df['ridership'] = self.df['ridership'].interpolate(method='linear')
            self.preprocessing_log.append(
                f"使用线性插值填充{missing_count}个缺失值"
            )
        
        return self.df
    
    def extract_time_features(self) -> pd.DataFrame:
        """提取时间特征"""
        self.df['hour'] = self.df['timestamp'].dt.hour
        self.df['day'] = self.df['timestamp'].dt.day
        self.df['month'] = self.df['timestamp'].dt.month
        self.df['weekday'] = self.df['timestamp'].dt.weekday
        self.df['is_weekend'] = (self.df['weekday'] >= 5).astype(int)
        self.df['is_workday'] = (self.df['weekday'] < 5).astype(int)
        
        def get_time_period(hour):
            if 6 <= hour < 9:
                return 'morning_peak'
            elif 9 <= hour < 17:
                return 'daytime'
            elif 17 <= hour < 20:
                return 'evening_peak'
            else:
                return 'night'
        
        self.df['time_period'] = self.df['hour'].apply(get_time_period)
        
        self.preprocessing_log.append(
            "提取时间特征: hour, day, month, weekday, is_weekend, is_workday, time_period"
        )
        
        return self.df
    
    def one_hot_encode_weather(self) -> pd.DataFrame:
        """对天气进行独热编码"""
        weather_dummies = pd.get_dummies(self.df['weather'], prefix='weather')
        self.df = pd.concat([self.df, weather_dummies], axis=1)
        
        self.preprocessing_log.append(
            f"天气独热编码完成，新增{len(weather_dummies.columns)}个特征"
        )
        
        return self.df
    
    def preprocess(self, missing_strategy: str = 'median') -> pd.DataFrame:
        """执行完整预处理流程"""
        self.handle_missing_values(missing_strategy)
        self.extract_time_features()
        self.one_hot_encode_weather()
        
        return self.df
    
    def get_preprocessing_log(self) -> str:
        """获取预处理日志"""
        return '\n'.join(self.preprocessing_log)
    
    def get_feature_columns(self) -> list:
        """获取用于建模的特征列"""
        base_features = ['temperature', 'humidity', 'wind_speed', 
                        'hour', 'month', 'weekday', 'is_weekend']
        weather_features = [col for col in self.df.columns if col.startswith('weather_')]
        return base_features + weather_features


if __name__ == '__main__':
    df = pd.read_csv('data/bike_data.csv', parse_dates=['timestamp'])
    preprocessor = DataPreprocessor(df)
    processed_df = preprocessor.preprocess()
    print(preprocessor.get_preprocessing_log())
    print(f"\n处理后数据形状: {processed_df.shape}")
    print(f"特征列: {preprocessor.get_feature_columns()}")
