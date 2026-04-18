import pandas as pd
import numpy as np
from typing import Tuple


class DataPreprocessor:
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.preprocessing_log = []
    
    def handle_missing_values(self, strategy: str = 'mean') -> pd.DataFrame:
        missing_before = self.df.isnull().sum().sum()
        self.preprocessing_log.append(f"缺失值处理前总数: {missing_before}")
        
        numeric_cols = ['temperature', 'humidity', 'wind_speed']
        
        for col in numeric_cols:
            if self.df[col].isnull().any():
                if strategy == 'mean':
                    fill_value = self.df[col].mean()
                elif strategy == 'median':
                    fill_value = self.df[col].median()
                elif strategy == 'interpolate':
                    self.df[col] = self.df[col].interpolate(method='linear')
                    continue
                else:
                    fill_value = self.df[col].mean()
                
                missing_count = self.df[col].isnull().sum()
                self.df[col].fillna(fill_value, inplace=True)
                self.preprocessing_log.append(
                    f"列 '{col}' 使用 {strategy} 填充 {missing_count} 个缺失值")
        
        missing_after = self.df.isnull().sum().sum()
        self.preprocessing_log.append(f"缺失值处理后总数: {missing_after}")
        
        return self.df
    
    def extract_time_features(self) -> pd.DataFrame:
        self.df['hour'] = self.df['timestamp'].dt.hour
        self.df['day'] = self.df['timestamp'].dt.day
        self.df['month'] = self.df['timestamp'].dt.month
        self.df['weekday'] = self.df['timestamp'].dt.weekday
        self.df['is_workday'] = (self.df['weekday'] < 5).astype(int)
        self.df['is_weekend'] = (self.df['weekday'] >= 5).astype(int)
        
        self.preprocessing_log.append("已提取时间特征: hour, day, month, weekday, is_workday, is_weekend")
        
        return self.df
    
    def one_hot_encode_weather(self) -> pd.DataFrame:
        weather_dummies = pd.get_dummies(self.df['weather'], prefix='weather')
        self.df = pd.concat([self.df, weather_dummies], axis=1)
        
        weather_cols = [col for col in self.df.columns if col.startswith('weather_')]
        self.preprocessing_log.append(f"天气独热编码完成，新增列: {weather_cols}")
        
        return self.df
    
    def preprocess(self) -> Tuple[pd.DataFrame, list]:
        self.handle_missing_values(strategy='mean')
        self.extract_time_features()
        self.one_hot_encode_weather()
        
        return self.df, self.preprocessing_log
    
    def get_feature_columns(self) -> list:
        feature_cols = ['hour', 'day', 'month', 'weekday', 'is_workday', 'is_weekend',
                       'temperature', 'humidity', 'wind_speed', 'station_id']
        
        weather_cols = [col for col in self.df.columns if col.startswith('weather_')]
        feature_cols.extend(weather_cols)
        
        return feature_cols
    
    def prepare_model_data(self, target_col: str = 'ridership') -> Tuple[pd.DataFrame, pd.Series]:
        feature_cols = self.get_feature_columns()
        
        X = self.df[feature_cols]
        y = self.df[target_col]
        
        self.preprocessing_log.append(f"模型特征列: {feature_cols}")
        self.preprocessing_log.append(f"目标列: {target_col}")
        
        return X, y
