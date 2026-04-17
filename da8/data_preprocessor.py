"""
数据预处理模块：处理缺失值、提取时间特征、天气独热编码
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self):
        self.encoder = None
        self.feature_columns = None
        
    def preprocess(self, df):
        """
        完整的数据预处理流程
        
        Args:
            df: 原始数据DataFrame
            
        Returns:
            pd.DataFrame: 预处理后的数据
        """
        df_processed = df.copy()
        
        # 1. 处理缺失值
        df_processed = self._handle_missing_values(df_processed)
        
        # 2. 提取时间特征
        df_processed = self._extract_time_features(df_processed)
        
        # 3. 天气独热编码
        df_processed = self._encode_weather(df_processed)
        
        # 保存特征列名
        self.feature_columns = [col for col in df_processed.columns 
                               if col not in ['timestamp', 'weather', 'ridership']]
        
        return df_processed
    
    def _handle_missing_values(self, df):
        """
        处理缺失值
        - 数值型特征：使用中位数填充
        """
        df_clean = df.copy()
        
        numeric_cols = ['temperature', 'humidity', 'windspeed']
        
        for col in numeric_cols:
            if col in df_clean.columns:
                missing_count = df_clean[col].isnull().sum()
                if missing_count > 0:
                    median_val = df_clean[col].median()
                    df_clean[col].fillna(median_val, inplace=True)
                    print(f"  列 '{col}': 使用 median={median_val:.2f} 填充 {missing_count} 个缺失值")
        
        return df_clean
    
    def _extract_time_features(self, df):
        """
        提取时间特征
        - hour: 小时 (0-23)
        - day_of_week: 星期几 (0-6)
        - is_weekday: 是否为工作日
        - month: 月份
        """
        df_time = df.copy()
        
        # 确保timestamp是datetime类型
        df_time['timestamp'] = pd.to_datetime(df_time['timestamp'])
        
        # 提取小时
        df_time['hour'] = df_time['timestamp'].dt.hour
        
        # 提取星期几
        df_time['day_of_week'] = df_time['timestamp'].dt.dayofweek
        
        # 是否为工作日（周一到周五）
        df_time['is_weekday'] = (df_time['day_of_week'] < 5).astype(int)
        
        # 提取月份
        df_time['month'] = df_time['timestamp'].dt.month
        
        # 提取是否高峰时段（早高峰7-9点，晚高峰17-19点）
        df_time['is_rush_hour'] = ((df_time['hour'].between(7, 9)) | 
                                   (df_time['hour'].between(17, 19))).astype(int)
        
        print("  时间特征提取完成: hour, day_of_week, is_weekday, month, is_rush_hour")
        
        return df_time
    
    def _encode_weather(self, df):
        """
        对天气状况进行独热编码
        """
        df_encoded = df.copy()
        
        if 'weather' in df_encoded.columns:
            # 使用pandas的get_dummies进行独热编码
            weather_dummies = pd.get_dummies(df_encoded['weather'], prefix='weather')
            
            # 合并到原数据
            df_encoded = pd.concat([df_encoded, weather_dummies], axis=1)
            
            print(f"  天气独热编码完成: {list(weather_dummies.columns)}")
        
        return df_encoded
    
    def get_feature_columns(self):
        """获取特征列名"""
        return self.feature_columns
    
    def get_training_data(self, df_processed):
        """
        获取训练用的特征和目标变量
        
        Returns:
            X: 特征矩阵
            y: 目标变量（骑行人数）
        """
        # 特征列（排除原始列和timestamp）
        exclude_cols = ['timestamp', 'weather', 'ridership']
        feature_cols = [col for col in df_processed.columns if col not in exclude_cols]
        
        X = df_processed[feature_cols]
        y = df_processed['ridership']
        
        return X, y, feature_cols


if __name__ == '__main__':
    # 测试预处理
    from data_generator import BikeSharingDataGenerator
    
    # 生成测试数据
    generator = BikeSharingDataGenerator(n_samples=1000)
    raw_data = generator.generate()
    
    print("原始数据缺失值:")
    print(raw_data.isnull().sum())
    
    # 预处理
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.preprocess(raw_data)
    
    print("\n预处理后的数据列:")
    print(processed_data.columns.tolist())
    
    print("\n预处理后缺失值:")
    print(processed_data.isnull().sum())
    
    print("\n数据预览:")
    print(processed_data.head())
