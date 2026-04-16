import pandas as pd
import numpy as np
from scipy import stats


def handle_missing_values(df, method='interpolate'):
    df_cleaned = df.copy()
    
    missing_count = df_cleaned['sales'].isna().sum()
    print(f"检测到 {missing_count} 个缺失值")
    
    if missing_count > 0:
        if method == 'interpolate':
            df_temp = df_cleaned.set_index('date')
            df_temp['sales'] = df_temp['sales'].interpolate(method='time')
            df_cleaned['sales'] = df_temp['sales'].values
        elif method == 'mean':
            df_cleaned['sales'] = df_cleaned['sales'].fillna(df_cleaned['sales'].mean())
        elif method == 'ffill':
            df_cleaned['sales'] = df_cleaned['sales'].ffill()
        
        print(f"已使用 {method} 方法处理缺失值")
    
    return df_cleaned


def detect_outliers(df, method='zscore', threshold=3):
    df_cleaned = df.copy()
    
    if method == 'zscore':
        z_scores = np.abs(stats.zscore(df_cleaned['sales']))
        outlier_mask = z_scores > threshold
    elif method == 'iqr':
        Q1 = df_cleaned['sales'].quantile(0.25)
        Q3 = df_cleaned['sales'].quantile(0.75)
        IQR = Q3 - Q1
        outlier_mask = (df_cleaned['sales'] < (Q1 - 1.5 * IQR)) | (df_cleaned['sales'] > (Q3 + 1.5 * IQR))
    
    outlier_count = outlier_mask.sum()
    print(f"检测到 {outlier_count} 个异常值")
    
    if outlier_count > 0:
        median_sales = df_cleaned['sales'].median()
        df_cleaned.loc[outlier_mask, 'sales'] = median_sales
        print(f"已使用中位数替换 {outlier_count} 个异常值")
    
    return df_cleaned, outlier_count


def clean_data(df):
    print("=" * 50)
    print("开始数据清洗...")
    
    df = df.sort_values('date').reset_index(drop=True)
    
    df = handle_missing_values(df, method='interpolate')
    
    df, outlier_count = detect_outliers(df, method='zscore', threshold=3)
    
    print("数据清洗完成!")
    print(f"清洗后数据形状: {df.shape}")
    print(f"日期范围: {df['date'].min()} 至 {df['date'].max()}")
    print("=" * 50)
    
    return df
