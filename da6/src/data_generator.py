import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta


class ECommerceDataGenerator:
    def __init__(self, n_users=10000, n_records=100000, random_seed=42):
        self.n_users = n_users
        self.n_records = n_records
        self.random_seed = random_seed
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        self.behavior_types = ['pv', 'fav', 'cart', 'buy']
        self.behavior_weights = [0.7, 0.1, 0.15, 0.05]
        self.categories = ['电子产品', '服装鞋帽', '食品饮料', '家居用品', '美妆护肤', '运动户外']
    
    def generate_timestamps(self):
        start_date = datetime(2025, 1, 1)
        end_date = datetime(2025, 3, 31)
        time_between_dates = end_date - start_date
        days_between_dates = time_between_dates.days
        random_days = np.random.randint(0, days_between_dates, size=self.n_records)
        random_seconds = np.random.randint(0, 86400, size=self.n_records)
        
        timestamps = []
        for d, s in zip(random_days, random_seconds):
            timestamp = start_date + timedelta(days=int(d), seconds=int(s))
            timestamps.append(timestamp)
        
        return timestamps
    
    def introduce_missing_values(self, df, missing_rate=0.01):
        for col in df.columns:
            mask = np.random.random(len(df)) < missing_rate
            df.loc[mask, col] = np.nan
        return df
    
    def introduce_outliers(self, df, outlier_rate=0.005):
        outlier_idx = np.random.choice(len(df), size=int(len(df) * outlier_rate), replace=False)
        df.loc[outlier_idx, 'user_id'] = 999999999
        return df
    
    def generate_data(self):
        user_ids = np.random.randint(10000, 10000 + self.n_users, size=self.n_records)
        behaviors = np.random.choice(self.behavior_types, size=self.n_records, p=self.behavior_weights)
        categories = np.random.choice(self.categories, size=self.n_records)
        timestamps = self.generate_timestamps()
        
        df = pd.DataFrame({
            'user_id': user_ids,
            'behavior_type': behaviors,
            'category': categories,
            'timestamp': timestamps
        })
        
        df = df.sort_values('timestamp').reset_index(drop=True)
        df = self.introduce_missing_values(df)
        df = self.introduce_outliers(df)
        
        return df
    
    def save_data(self, df, filepath='../data/ecommerce_behavior.csv'):
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"数据已保存至: {filepath}")
        print(f"数据集大小: {df.shape[0]} 条记录, {df.shape[1]} 个字段")
        print(f"用户数量: {df['user_id'].nunique()}")
        print(f"行为类型分布:")
        print(df['behavior_type'].value_counts())


if __name__ == '__main__':
    generator = ECommerceDataGenerator(n_users=5000, n_records=50000)
    df = generator.generate_data()
    generator.save_data(df)
