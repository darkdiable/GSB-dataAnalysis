import pandas as pd
import numpy as np


def extract_time_features(df):
    df_features = df.copy()
    
    df_features['year'] = df_features['date'].dt.year
    df_features['month'] = df_features['date'].dt.month
    df_features['day'] = df_features['date'].dt.day
    df_features['dayofweek'] = df_features['date'].dt.dayofweek
    df_features['dayofyear'] = df_features['date'].dt.dayofyear
    df_features['weekofyear'] = df_features['date'].dt.isocalendar().week.astype(int)
    df_features['quarter'] = df_features['date'].dt.quarter
    df_features['is_weekend'] = (df_features['dayofweek'] >= 5).astype(int)
    df_features['is_month_start'] = df_features['date'].dt.is_month_start.astype(int)
    df_features['is_month_end'] = df_features['date'].dt.is_month_end.astype(int)
    
    print("已提取基础时间特征")
    return df_features


def extract_seasonality_features(df):
    df_features = df.copy()
    
    df_features['sin_dayofweek'] = np.sin(2 * np.pi * df_features['dayofweek'] / 7)
    df_features['cos_dayofweek'] = np.cos(2 * np.pi * df_features['dayofweek'] / 7)
    df_features['sin_dayofyear'] = np.sin(2 * np.pi * df_features['dayofyear'] / 365)
    df_features['cos_dayofyear'] = np.cos(2 * np.pi * df_features['dayofyear'] / 365)
    df_features['sin_month'] = np.sin(2 * np.pi * df_features['month'] / 12)
    df_features['cos_month'] = np.cos(2 * np.pi * df_features['month'] / 12)
    
    print("已提取季节性特征（三角函数编码）")
    return df_features


def extract_trend_features(df):
    df_features = df.copy()
    
    df_features['trend'] = np.arange(len(df_features))
    
    df_features['sales_lag_1'] = df_features['sales'].shift(1)
    df_features['sales_lag_7'] = df_features['sales'].shift(7)
    df_features['sales_lag_14'] = df_features['sales'].shift(14)
    df_features['sales_lag_30'] = df_features['sales'].shift(30)
    
    df_features['sales_rolling_7'] = df_features['sales'].rolling(window=7).mean()
    df_features['sales_rolling_14'] = df_features['sales'].rolling(window=14).mean()
    df_features['sales_rolling_30'] = df_features['sales'].rolling(window=30).mean()
    
    df_features['sales_rolling_std_7'] = df_features['sales'].rolling(window=7).std()
    df_features['sales_rolling_std_30'] = df_features['sales'].rolling(window=30).std()
    
    df_features['sales_diff_1'] = df_features['sales'].diff(1)
    df_features['sales_diff_7'] = df_features['sales'].diff(7)
    
    print("已提取趋势项和滞后特征")
    return df_features


def create_future_features(last_date, days=30, last_trend_value=365):
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days, freq='D')
    
    future_df = pd.DataFrame({'date': future_dates})
    future_df = extract_time_features(future_df)
    future_df = extract_seasonality_features(future_df)
    
    future_df['is_holiday'] = 0
    future_df['trend'] = np.arange(len(future_df)) + last_trend_value + 1
    
    return future_df


def engineer_features(df, include_target=True):
    print("=" * 50)
    print("开始特征工程...")
    
    df = extract_time_features(df)
    df = extract_seasonality_features(df)
    
    if include_target and 'sales' in df.columns:
        df = extract_trend_features(df)
    
    df = df.dropna()
    
    exclude_columns = ['date', 'sales', 'weekday']
    feature_columns = [col for col in df.columns if col not in exclude_columns]
    print(f"特征工程完成! 生成 {len(feature_columns)} 个特征")
    print(f"特征列表: {feature_columns}")
    print("=" * 50)
    
    return df, feature_columns
