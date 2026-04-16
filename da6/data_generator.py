import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random


class EcommerceDataGenerator:
    
    def __init__(self, n_users=1000, n_records=50000, seed=42):
        self.n_users = n_users
        self.n_records = n_records
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        
        self.behavior_types = ['pv', 'fav', 'cart', 'buy']
        self.behavior_weights = [0.6, 0.15, 0.15, 0.1]
        
        self.categories = [
            'Electronics', 'Clothing', 'Home', 'Beauty', 'Sports',
            'Books', 'Food', 'Toys', 'Jewelry', 'Automotive'
        ]
        
    def generate_timestamps(self, start_date='2023-01-01', end_date='2023-12-31'):
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        delta = end - start
        
        timestamps = []
        for _ in range(self.n_records):
            random_days = random.randint(0, delta.days)
            random_seconds = random.randint(0, 86399)
            timestamp = start + timedelta(days=random_days, seconds=random_seconds)
            timestamps.append(timestamp)
        
        return timestamps
    
    def generate_user_ids(self):
        user_ids = [f'user_{i:05d}' for i in range(1, self.n_users + 1)]
        return np.random.choice(user_ids, size=self.n_records)
    
    def generate_behaviors(self):
        return np.random.choice(
            self.behavior_types,
            size=self.n_records,
            p=self.behavior_weights
        )
    
    def generate_categories(self):
        return np.random.choice(self.categories, size=self.n_records)
    
    def inject_missing_values(self, df, missing_rate=0.02):
        df_with_missing = df.copy()
        n_missing = int(len(df) * missing_rate)
        
        for col in ['user_id', 'behavior_type', 'category']:
            missing_indices = np.random.choice(df.index, size=n_missing, replace=False)
            df_with_missing.loc[missing_indices, col] = np.nan
        
        return df_with_missing
    
    def inject_outliers(self, df, outlier_rate=0.01):
        df_with_outliers = df.copy()
        n_outliers = int(len(df) * outlier_rate)
        
        outlier_indices = np.random.choice(df.index, size=n_outliers, replace=False)
        invalid_behaviors = ['invalid', 'unknown', 'error', 'test']
        df_with_outliers.loc[outlier_indices, 'behavior_type'] = np.random.choice(
            invalid_behaviors, size=n_outliers
        )
        
        return df_with_outliers
    
    def generate(self, inject_issues=True):
        timestamps = self.generate_timestamps()
        
        data = {
            'user_id': self.generate_user_ids(),
            'behavior_type': self.generate_behaviors(),
            'timestamp': timestamps,
            'category': self.generate_categories()
        }
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        if inject_issues:
            df = self.inject_missing_values(df)
            df = self.inject_outliers(df)
        
        return df
    
    def save_data(self, df, filepath='data/ecommerce_data.csv'):
        df.to_csv(filepath, index=False)
        print(f"数据已保存至: {filepath}")
        return filepath


if __name__ == '__main__':
    generator = EcommerceDataGenerator(n_users=1000, n_records=50000)
    df = generator.generate(inject_issues=True)
    generator.save_data(df)
    print(f"生成数据集形状: {df.shape}")
    print(f"\n数据预览:\n{df.head()}")
    print(f"\n缺失值统计:\n{df.isnull().sum()}")
