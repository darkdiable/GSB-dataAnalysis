import pandas as pd
import numpy as np


def handle_missing_values(data, method='interpolate'):
    data_cleaned = data.copy()
    
    if method == 'interpolate':
        data_cleaned = data_cleaned.set_index('date')
        numeric_cols = data_cleaned.select_dtypes(include=[np.number]).columns
        data_cleaned[numeric_cols] = data_cleaned[numeric_cols].interpolate(method='time')
        data_cleaned = data_cleaned.reset_index()
    elif method == 'mean':
        numeric_cols = data_cleaned.select_dtypes(include=[np.number]).columns
        data_cleaned[numeric_cols] = data_cleaned[numeric_cols].fillna(data_cleaned[numeric_cols].mean())
    elif method == 'forward':
        data_cleaned = data_cleaned.fillna(method='ffill')
    
    return data_cleaned


def remove_outliers(data, columns=None, z_threshold=3):
    data_cleaned = data.copy()
    
    if columns is None:
        columns = data_cleaned.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        z_scores = np.abs((data_cleaned[col] - data_cleaned[col].mean()) / data_cleaned[col].std())
        outliers = z_scores > z_threshold
        data_cleaned.loc[outliers, col] = np.nan
        print(f"列 {col} 中检测到 {outliers.sum()} 个异常值，已标记为缺失值")
    
    data_cleaned = handle_missing_values(data_cleaned, method='interpolate')
    
    return data_cleaned


def preprocess_pipeline(data):
    print("开始数据预处理...")
    print(f"原始数据形状: {data.shape}")
    
    missing_count = data.isnull().sum().sum()
    if missing_count > 0:
        print(f"检测到 {missing_count} 个缺失值")
        data = handle_missing_values(data)
    
    data = remove_outliers(data)
    
    data['month'] = data['date'].dt.month
    data['season'] = data['month'].map({1: '冬季', 2: '冬季', 3: '春季', 4: '春季', 5: '春季', 
                                        6: '夏季', 7: '夏季', 8: '夏季', 9: '秋季', 10: '秋季', 
                                        11: '秋季', 12: '冬季'})
    
    print("数据预处理完成")
    print(f"处理后数据形状: {data.shape}")
    
    return data
