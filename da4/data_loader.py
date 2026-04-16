import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_sample_data(start_date='2023-01-01', days=365):
    date_range = pd.date_range(start=start_date, periods=days, freq='D')
    np.random.seed(42)
    
    trend = np.linspace(1000, 3000, days)
    weekly_seasonality = 150 * np.sin(2 * np.pi * date_range.dayofweek / 7)
    monthly_seasonality = 200 * np.sin(2 * np.pi * date_range.day / 30)
    noise = np.random.normal(0, 100, days)
    
    sales = trend + weekly_seasonality + monthly_seasonality + noise
    sales = np.maximum(sales, 500)
    
    df = pd.DataFrame({
        'date': date_range,
        'sales': sales,
        'weekday': date_range.dayofweek,
        'month': date_range.month,
        'is_holiday': np.random.choice([0, 1], size=days, p=[0.95, 0.05])
    })
    
    missing_indices = np.random.choice(days, size=10, replace=False)
    df.loc[missing_indices, 'sales'] = np.nan
    
    outlier_indices = np.random.choice(days, size=5, replace=False)
    df.loc[outlier_indices, 'sales'] *= 3
    
    return df


def load_data(file_path=None):
    if file_path is None:
        print("未提供数据文件，生成示例销售数据...")
        return generate_sample_data()
    
    try:
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        print(f"加载数据失败: {e}，使用示例数据")
        return generate_sample_data()


def save_predictions(predictions_df, output_path='output/predictions.csv'):
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    predictions_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"预测结果已保存至: {output_path}")
