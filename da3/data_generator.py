"""
数据生成模块
生成模拟电商销售数据，包含用户ID、时间、金额、类别、评分等字段
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os


class DataGenerator:
    """电商销售数据生成器"""
    
    def __init__(self, seed: int = 42):
        """
        初始化数据生成器
        
        Args:
            seed: 随机种子，确保数据可复现
        """
        np.random.seed(seed)
        random.seed(seed)
        
        self.categories = [
            '电子产品', '服装', '食品', '家居', '美妆',
            '运动户外', '图书', '母婴', '数码配件', '家电'
        ]
        
        self.category_price_ranges = {
            '电子产品': (500, 8000),
            '服装': (50, 1000),
            '食品': (10, 300),
            '家居': (100, 3000),
            '美妆': (30, 800),
            '运动户外': (80, 2000),
            '图书': (20, 150),
            '母婴': (50, 500),
            '数码配件': (20, 500),
            '家电': (200, 5000)
        }
        
        self.user_pool_size = 150
    
    def generate_data(self, n_records: int = 600, months: int = 4) -> pd.DataFrame:
        """
        生成模拟电商销售数据
        
        Args:
            n_records: 记录数量
            months: 时间跨度（月）
            
        Returns:
            包含销售数据的DataFrame
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=months * 30)
            
            user_ids = [f'U{str(i).zfill(4)}' for i in range(1, self.user_pool_size + 1)]
            
            data = []
            
            for i in range(n_records):
                order_id = f'O{str(i + 1).zfill(6)}'
                user_id = random.choice(user_ids)
                category = random.choice(self.categories)
                
                price_min, price_max = self.category_price_ranges[category]
                amount = round(np.random.uniform(price_min, price_max), 2)
                
                random_days = random.randint(0, months * 30)
                random_seconds = random.randint(0, 86400)
                order_time = start_date + timedelta(days=random_days, seconds=random_seconds)
                
                base_rating = np.random.choice([3, 4, 5], p=[0.1, 0.3, 0.6])
                rating = min(5, max(1, base_rating + np.random.choice([-1, 0, 1], p=[0.1, 0.7, 0.2])))
                
                quantity = np.random.choice([1, 1, 1, 2, 2, 3, 4, 5], p=[0.3, 0.2, 0.1, 0.15, 0.1, 0.08, 0.04, 0.03])
                
                payment_method = np.random.choice(
                    ['支付宝', '微信支付', '银行卡', '信用卡'],
                    p=[0.4, 0.35, 0.15, 0.1]
                )
                
                region = np.random.choice(
                    ['华东', '华南', '华北', '华中', '西南', '东北', '西北'],
                    p=[0.25, 0.2, 0.18, 0.15, 0.1, 0.07, 0.05]
                )
                
                data.append({
                    'order_id': order_id,
                    'user_id': user_id,
                    'order_time': order_time,
                    'category': category,
                    'amount': amount,
                    'quantity': quantity,
                    'total_amount': round(amount * quantity, 2),
                    'rating': rating,
                    'payment_method': payment_method,
                    'region': region
                })
            
            df = pd.DataFrame(data)
            df = df.sort_values('order_time').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            raise RuntimeError(f"数据生成失败: {str(e)}")
    
    def introduce_issues(self, df: pd.DataFrame, 
                         missing_rate: float = 0.02,
                         duplicate_rate: float = 0.01,
                         outlier_rate: float = 0.01) -> pd.DataFrame:
        """
        引入数据质量问题（缺失值、重复值、异常值）
        用于演示数据清洗功能
        
        Args:
            df: 原始数据
            missing_rate: 缺失值比例
            duplicate_rate: 重复值比例
            outlier_rate: 异常值比例
            
        Returns:
            包含数据质量问题的DataFrame
        """
        try:
            df_dirty = df.copy()
            n_records = len(df_dirty)
            
            n_missing = int(n_records * missing_rate)
            missing_indices = np.random.choice(n_records, n_missing, replace=False)
            
            for idx in missing_indices:
                col = np.random.choice(['rating', 'amount', 'region'])
                df_dirty.loc[idx, col] = np.nan
            
            n_duplicates = int(n_records * duplicate_rate)
            duplicate_indices = np.random.choice(n_records, n_duplicates, replace=False)
            duplicates = df_dirty.iloc[duplicate_indices].copy()
            df_dirty = pd.concat([df_dirty, duplicates], ignore_index=True)
            
            n_outliers = int(n_records * outlier_rate)
            outlier_indices = np.random.choice(n_records, n_outliers, replace=False)
            df_dirty.loc[outlier_indices, 'amount'] = df_dirty.loc[outlier_indices, 'amount'] * 10
            
            return df_dirty
            
        except Exception as e:
            raise RuntimeError(f"引入数据问题失败: {str(e)}")
    
    def save_to_csv(self, df: pd.DataFrame, filepath: str) -> None:
        """
        保存数据到CSV文件
        
        Args:
            df: 要保存的数据
            filepath: 文件路径
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            print(f"数据已保存至: {filepath}")
        except Exception as e:
            raise RuntimeError(f"保存数据失败: {str(e)}")
    
    def load_from_csv(self, filepath: str) -> pd.DataFrame:
        """
        从CSV文件加载数据
        
        Args:
            filepath: 文件路径
            
        Returns:
            加载的DataFrame
        """
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"文件不存在: {filepath}")
            df = pd.read_csv(filepath, encoding='utf-8-sig')
            df['order_time'] = pd.to_datetime(df['order_time'])
            return df
        except Exception as e:
            raise RuntimeError(f"加载数据失败: {str(e)}")


def generate_and_save_data(output_dir: str = 'da3/data', 
                           n_records: int = 600,
                           include_issues: bool = True) -> tuple:
    """
    生成并保存模拟数据
    
    Args:
        output_dir: 输出目录
        n_records: 记录数量
        include_issues: 是否包含数据质量问题
        
    Returns:
        (干净数据, 脏数据) 元组
    """
    generator = DataGenerator()
    
    clean_data = generator.generate_data(n_records=n_records)
    
    if include_issues:
        dirty_data = generator.introduce_issues(clean_data)
    else:
        dirty_data = clean_data.copy()
    
    clean_path = os.path.join(output_dir, 'sales_data_clean.csv')
    dirty_path = os.path.join(output_dir, 'sales_data_dirty.csv')
    
    generator.save_to_csv(clean_data, clean_path)
    generator.save_to_csv(dirty_data, dirty_path)
    
    return clean_data, dirty_data


if __name__ == '__main__':
    clean_df, dirty_df = generate_and_save_data()
    print(f"\n生成数据统计:")
    print(f"记录数: {len(clean_df)}")
    print(f"用户数: {clean_df['user_id'].nunique()}")
    print(f"时间范围: {clean_df['order_time'].min()} 至 {clean_df['order_time'].max()}")
    print(f"类别数: {clean_df['category'].nunique()}")
