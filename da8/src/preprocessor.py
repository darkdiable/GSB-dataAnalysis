import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


class DataPreprocessor:
    def __init__(self):
        self.encoder = None
        self.weather_categories = None
        
    def handle_missing_values(self, df, strategy='interpolate'):
        df_processed = df.copy()
        
        missing_count = df_processed.isnull().sum()
        print(f"处理前缺失值统计:\n{missing_count[missing_count > 0]}")
        
        if strategy == 'interpolate':
            if not isinstance(df_processed.index, pd.DatetimeIndex):
                df_processed = df_processed.set_index('timestamp')
            df_processed['temperature'] = df_processed['temperature'].interpolate(method='time')
            df_processed['humidity'] = df_processed['humidity'].interpolate(method='time')
            df_processed = df_processed.reset_index()
        elif strategy == 'mean':
            df_processed['temperature'] = df_processed['temperature'].fillna(df_processed['temperature'].mean())
            df_processed['humidity'] = df_processed['humidity'].fillna(df_processed['humidity'].mean())
        elif strategy == 'forward':
            df_processed['temperature'] = df_processed['temperature'].ffill()
            df_processed['humidity'] = df_processed['humidity'].ffill()
        
        remaining_missing = df_processed.isnull().sum().sum()
        print(f"处理后剩余缺失值: {remaining_missing}")
        
        return df_processed
    
    def extract_time_features(self, df):
        df_processed = df.copy()
        
        if not pd.api.types.is_datetime64_any_dtype(df_processed['timestamp']):
            df_processed['timestamp'] = pd.to_datetime(df_processed['timestamp'])
        
        df_processed['hour'] = df_processed['timestamp'].dt.hour
        df_processed['day'] = df_processed['timestamp'].dt.day
        df_processed['month'] = df_processed['timestamp'].dt.month
        df_processed['weekday'] = df_processed['timestamp'].dt.weekday
        df_processed['is_weekend'] = (df_processed['weekday'] >= 5).astype(int)
        df_processed['is_weekday'] = (df_processed['weekday'] < 5).astype(int)
        df_processed['date'] = df_processed['timestamp'].dt.date
        
        df_processed['is_rush_hour_morning'] = ((df_processed['hour'] >= 7) & (df_processed['hour'] <= 9) & (df_processed['is_weekday'])).astype(int)
        df_processed['is_rush_hour_evening'] = ((df_processed['hour'] >= 17) & (df_processed['hour'] <= 19) & (df_processed['is_weekday'])).astype(int)
        
        print(f"已提取时间特征: hour, weekday, is_weekend, is_weekday, rush_hour标志")
        return df_processed
    
    def one_hot_encode_weather(self, df, fit_encoder=True):
        df_processed = df.copy()
        
        if fit_encoder:
            self.encoder = OneHotEncoder(sparse_output=False, drop='first')
            weather_encoded = self.encoder.fit_transform(df_processed[['weather']])
            self.weather_categories = self.encoder.categories_[0][1:]
        else:
            weather_encoded = self.encoder.transform(df_processed[['weather']])
        
        weather_columns = [f'weather_{cat}' for cat in self.weather_categories]
        weather_df = pd.DataFrame(weather_encoded, columns=weather_columns, index=df_processed.index)
        df_processed = pd.concat([df_processed, weather_df], axis=1)
        
        print(f"天气独热编码完成，新增特征: {weather_columns}")
        return df_processed
    
    def add_weather_severity(self, df):
        df_processed = df.copy()
        
        weather_score = {
            '晴天': 3,
            '多云': 2,
            '小雨': 1,
            '大雨': 0
        }
        df_processed['weather_severity'] = df_processed['weather'].map(weather_score)
        
        return df_processed
    
    def preprocess_pipeline(self, df, encode_weather=True):
        print("=" * 50)
        print("开始数据预处理流程...")
        print("=" * 50)
        
        df_processed = self.handle_missing_values(df)
        df_processed = self.extract_time_features(df_processed)
        
        if encode_weather:
            df_processed = self.one_hot_encode_weather(df_processed, fit_encoder=True)
        
        df_processed = self.add_weather_severity(df_processed)
        
        print("=" * 50)
        print("数据预处理完成!")
        print(f"处理后数据集形状: {df_processed.shape}")
        print("=" * 50)
        
        return df_processed
    
    def get_processing_report(self, df_raw, df_processed):
        report = []
        report.append("=" * 60)
        report.append("数据预处理报告")
        report.append("=" * 60)
        report.append(f"原始数据集形状: {df_raw.shape}")
        report.append(f"处理后数据集形状: {df_processed.shape}")
        report.append(f"原始数据缺失值总数: {df_raw.isnull().sum().sum()}")
        report.append(f"处理后缺失值总数: {df_processed.isnull().sum().sum()}")
        report.append("")
        report.append("数据列信息:")
        for col in df_processed.columns:
            dtype = df_processed[col].dtype
            report.append(f"  - {col}: {dtype}")
        report.append("")
        report.append("时间特征统计:")
        report.append(f"  - 时间范围: {df_processed['timestamp'].min()} ~ {df_processed['timestamp'].max()}")
        report.append(f"  - 站点数量: {df_processed['station_id'].nunique()}")
        report.append(f"  - 天气类型: {df_processed['weather'].unique().tolist()}")
        report.append("=" * 60)
        
        return "\n".join(report)


if __name__ == '__main__':
    from data_generator import BikeShareDataGenerator
    
    generator = BikeShareDataGenerator(days=30)
    df_raw = generator.generate_dataset()
    
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.preprocess_pipeline(df_raw)
    print(preprocessor.get_processing_report(df_raw, df_processed))
