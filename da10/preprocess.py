import pandas as pd
import numpy as np
from scipy import stats


def handle_missing_values(df, method='interpolate'):
    df_clean = df.copy()
    
    missing_stats = df_clean.isnull().sum()
    print(f"缺失值统计:\n{missing_stats}")
    
    if method == 'interpolate':
        if 'date' in df_clean.columns:
            df_clean = df_clean.set_index('date')
            df_clean = df_clean.interpolate(method='time')
            df_clean = df_clean.reset_index()
        else:
            df_clean = df_clean.interpolate(method='linear')
    elif method == 'ffill':
        df_clean = df_clean.fillna(method='ffill')
    elif method == 'bfill':
        df_clean = df_clean.fillna(method='bfill')
    elif method == 'mean':
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
    
    df_clean = df_clean.dropna()
    
    return df_clean


def detect_outliers_zscore(df, threshold=3.0):
    outliers = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if col == 'precipitation':
            continue
        z_scores = np.abs(stats.zscore(df[col].dropna()))
        outlier_indices = df.index[np.abs(stats.zscore(df[col].fillna(df[col].mean()))) > threshold]
        outliers[col] = list(outlier_indices)
    
    return outliers


def detect_outliers_iqr(df):
    outliers = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if col == 'precipitation':
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_indices = df.index[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outliers[col] = list(outlier_indices)
    
    return outliers


def handle_outliers(df, method='clip', outliers=None):
    df_clean = df.copy()
    
    if outliers is None:
        outliers = detect_outliers_iqr(df_clean)
    
    for col, indices in outliers.items():
        if len(indices) > 0:
            if method == 'clip':
                Q1 = df_clean[col].quantile(0.05)
                Q3 = df_clean[col].quantile(0.95)
                df_clean.loc[indices, col] = np.clip(df_clean.loc[indices, col], Q1, Q3)
            elif method == 'remove':
                df_clean = df_clean.drop(indices)
    
    return df_clean


def add_time_features(df):
    df_enhanced = df.copy()
    
    if 'date' in df_enhanced.columns:
        df_enhanced['year'] = df_enhanced['date'].dt.year
        df_enhanced['month'] = df_enhanced['date'].dt.month
        df_enhanced['quarter'] = df_enhanced['date'].dt.quarter
        df_enhanced['day_of_year'] = df_enhanced['date'].dt.dayofyear
        df_enhanced['weekday'] = df_enhanced['date'].dt.weekday
    
    return df_enhanced


def validate_data_ranges(df):
    validation_report = {}
    
    if 'temperature' in df.columns:
        temp_valid = (df['temperature'] >= -50) & (df['temperature'] <= 50)
        validation_report['temperature'] = {
            'valid': temp_valid.all(),
            'invalid_count': (~temp_valid).sum()
        }
    
    if 'humidity' in df.columns:
        humidity_valid = (df['humidity'] >= 0) & (df['humidity'] <= 100)
        validation_report['humidity'] = {
            'valid': humidity_valid.all(),
            'invalid_count': (~humidity_valid).sum()
        }
    
    if 'precipitation' in df.columns:
        precip_valid = df['precipitation'] >= 0
        validation_report['precipitation'] = {
            'valid': precip_valid.all(),
            'invalid_count': (~precip_valid).sum()
        }
    
    if 'wind_speed' in df.columns:
        wind_valid = (df['wind_speed'] >= 0) & (df['wind_speed'] <= 100)
        validation_report['wind_speed'] = {
            'valid': wind_valid.all(),
            'invalid_count': (~wind_valid).sum()
        }
    
    return validation_report


def preprocess_data(df, missing_method='interpolate', outlier_method='clip'):
    print("开始数据预处理...")
    
    print("\n1. 验证数据范围...")
    validation = validate_data_ranges(df)
    for col, result in validation.items():
        if result['invalid_count'] > 0:
            print(f"  {col}: 发现 {result['invalid_count']} 个超出范围的值")
        else:
            print(f"  {col}: 所有值均在有效范围内")
    
    print("\n2. 处理缺失值...")
    df_clean = handle_missing_values(df, method=missing_method)
    
    print("\n3. 检测并处理异常值...")
    outliers = detect_outliers_iqr(df_clean)
    total_outliers = sum(len(indices) for indices in outliers.values())
    print(f"  检测到 {total_outliers} 个潜在异常值")
    
    df_clean = handle_outliers(df_clean, method=outlier_method, outliers=outliers)
    
    print("\n4. 添加时间特征...")
    df_enhanced = add_time_features(df_clean)
    
    print("\n数据预处理完成！")
    print(f"原始数据形状: {df.shape}")
    print(f"处理后数据形状: {df_enhanced.shape}")
    
    return df_enhanced


if __name__ == '__main__':
    from data_loader import generate_sample_data
    
    df = generate_sample_data()
    df_processed = preprocess_data(df)
    print("\n处理后数据前5行:")
    print(df_processed.head())
