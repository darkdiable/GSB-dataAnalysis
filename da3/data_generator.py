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
            '电子产品', '服装鞋帽', '食品饮料', '家居用品', 
            '美妆护肤', '运动户外', '图书文具', '母婴用品'
        ]
        
        self.category_price_ranges = {
            '电子产品': (100, 8000),
            '服装鞋帽': (30, 2000),
            '食品饮料': (5, 500),
            '家居用品': (20, 3000),
            '美妆护肤': (50, 1500),
            '运动户外': (50, 2500),
            '图书文具': (10, 300),
            '母婴用品': (30, 1000)
        }
        
    def generate_user_ids(self, n_records: int, n_users: int = 100) -> list:
        """
        生成用户ID列表
        
        Args:
            n_records: 记录数量
            n_users: 用户数量
            
        Returns:
            用户ID列表
        """
        user_ids = [f'U{str(i).zfill(4)}' for i in range(1, n_users + 1)]
        return np.random.choice(user_ids, n_records).tolist()
    
    def generate_dates(self, n_records: int, start_date: str, end_date: str) -> list:
        """
        生成日期列表
        
        Args:
            n_records: 记录数量
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            日期列表
        """
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        date_range = (end - start).days
        
        dates = [start + timedelta(days=random.randint(0, date_range)) 
                 for _ in range(n_records)]
        return sorted(dates)
    
    def generate_amounts(self, categories: list) -> list:
        """
        根据类别生成金额
        
        Args:
            categories: 类别列表
            
        Returns:
            金额列表
        """
        amounts = []
        for cat in categories:
            min_price, max_price = self.category_price_ranges[cat]
            amount = round(np.random.uniform(min_price, max_price), 2)
            amounts.append(amount)
        return amounts
    
    def generate_ratings(self, n_records: int) -> list:
        """
        生成评分数据
        
        Args:
            n_records: 记录数量
            
        Returns:
            评分列表
        """
        ratings = np.random.choice([1, 2, 3, 4, 5], n_records, 
                                   p=[0.05, 0.1, 0.2, 0.35, 0.3])
        return ratings.tolist()
    
    def generate_quantities(self, n_records: int) -> list:
        """
        生成购买数量
        
        Args:
            n_records: 记录数量
            
        Returns:
            数量列表
        """
        quantities = np.random.choice([1, 2, 3, 4, 5], n_records,
                                      p=[0.5, 0.25, 0.15, 0.07, 0.03])
        return quantities.tolist()
    
    def generate_payment_methods(self, n_records: int) -> list:
        """
        生成支付方式
        
        Args:
            n_records: 记录数量
            
        Returns:
            支付方式列表
        """
        methods = ['支付宝', '微信支付', '银行卡', '信用卡', '货到付款']
        return np.random.choice(methods, n_records, 
                               p=[0.35, 0.35, 0.1, 0.15, 0.05]).tolist()
    
    def generate_regions(self, n_records: int) -> list:
        """
        生成地区
        
        Args:
            n_records: 记录数量
            
        Returns:
            地区列表
        """
        regions = ['华东', '华南', '华北', '华中', '西南', '东北', '西北']
        return np.random.choice(regions, n_records).tolist()
    
    def generate_sales_data(self, n_records: int = 600, 
                           start_date: str = '2024-01-01',
                           end_date: str = '2024-04-30') -> pd.DataFrame:
        """
        生成完整的销售数据
        
        Args:
            n_records: 记录数量，默认600条
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            包含销售数据的DataFrame
        """
        try:
            print(f"开始生成{n_records}条销售数据...")
            
            categories = np.random.choice(self.categories, n_records).tolist()
            
            data = {
                'order_id': [f'ORD{str(i).zfill(6)}' for i in range(1, n_records + 1)],
                'user_id': self.generate_user_ids(n_records, n_users=150),
                'order_date': self.generate_dates(n_records, start_date, end_date),
                'category': categories,
                'amount': self.generate_amounts(categories),
                'quantity': self.generate_quantities(n_records),
                'rating': self.generate_ratings(n_records),
                'payment_method': self.generate_payment_methods(n_records),
                'region': self.generate_regions(n_records)
            }
            
            df = pd.DataFrame(data)
            
            df['total_amount'] = df['amount'] * df['quantity']
            
            df['order_date'] = pd.to_datetime(df['order_date'])
            
            print(f"数据生成完成，共{len(df)}条记录")
            print(f"时间范围: {df['order_date'].min()} 至 {df['order_date'].max()}")
            
            return df
            
        except Exception as e:
            print(f"数据生成错误: {str(e)}")
            raise
    
    def save_to_csv(self, df: pd.DataFrame, filepath: str) -> None:
        """
        保存数据到CSV文件
        
        Args:
            df: 数据DataFrame
            filepath: 保存路径
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            print(f"数据已保存至: {filepath}")
        except Exception as e:
            print(f"保存数据错误: {str(e)}")
            raise


def generate_and_save_data(output_dir: str = 'da3/data', 
                          n_records: int = 600) -> pd.DataFrame:
    """
    生成并保存销售数据的主函数
    
    Args:
        output_dir: 输出目录
        n_records: 记录数量
        
    Returns:
        生成的DataFrame
    """
    generator = DataGenerator(seed=42)
    df = generator.generate_sales_data(n_records=n_records)
    
    filepath = os.path.join(output_dir, 'sales_data.csv')
    generator.save_to_csv(df, filepath)
    
    return df


if __name__ == '__main__':
    df = generate_and_save_data()
    print("\n数据预览:")
    print(df.head(10))
    print("\n数据统计:")
    print(df.describe())
