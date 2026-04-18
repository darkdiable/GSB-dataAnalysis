# -*- coding: utf-8 -*-
"""
数据生成模块
模拟真实电商销售数据，包含用户ID、时间、金额、类别、评分等字段
生成不少于500条数据，时间跨度至少3个月
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os


class DataGenerator:
    """电商销售数据生成器"""
    
    def __init__(self, n_records=800, start_date=None, end_date=None, random_seed=42):
        """
        初始化数据生成器
        
        参数:
            n_records: 生成记录数，默认800条
            start_date: 开始日期，默认3个月前
            end_date: 结束日期，默认今天
            random_seed: 随机种子，保证可复现
        """
        self.n_records = n_records
        self.random_seed = random_seed
        
        # 设置默认日期范围（3个月）
        if end_date is None:
            self.end_date = datetime.now()
        else:
            self.end_date = end_date
            
        if start_date is None:
            self.start_date = self.end_date - timedelta(days=90)
        else:
            self.start_date = start_date
        
        # 设置随机种子
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # 定义商品类别
        self.categories = ['电子产品', '服装', '食品', '家居', '美妆', '运动', '图书', '母婴']
        
        # 定义类别价格区间
        self.price_ranges = {
            '电子产品': (500, 8000),
            '服装': (50, 800),
            '食品': (10, 300),
            '家居': (30, 2000),
            '美妆': (50, 1000),
            '运动': (80, 1500),
            '图书': (20, 200),
            '母婴': (30, 1500)
        }
        
    def _generate_user_ids(self):
        """生成用户ID列表"""
        # 模拟200个不同用户
        n_users = min(200, self.n_records // 3)
        user_ids = [f"USER_{str(i).zfill(4)}" for i in range(1, n_users + 1)]
        return user_ids
    
    def _generate_product_ids(self):
        """生成商品ID列表"""
        # 模拟300个不同商品
        n_products = min(300, self.n_records // 2)
        product_ids = [f"PROD_{str(i).zfill(5)}" for i in range(1, n_products + 1)]
        return product_ids
    
    def _generate_timestamps(self):
        """生成交易时间戳"""
        # 生成在日期范围内的随机时间
        time_range = (self.end_date - self.start_date).total_seconds()
        random_seconds = np.random.uniform(0, time_range, self.n_records)
        timestamps = [self.start_date + timedelta(seconds=int(s)) for s in random_seconds]
        return sorted(timestamps)
    
    def _generate_amounts(self, categories):
        """根据类别生成交易金额"""
        amounts = []
        for cat in categories:
            min_price, max_price = self.price_ranges[cat]
            # 使用对数正态分布模拟真实价格分布
            mean = np.log((min_price + max_price) / 2)
            sigma = 0.5
            amount = np.random.lognormal(mean, sigma)
            # 限制在合理范围内
            amount = np.clip(amount, min_price, max_price)
            amounts.append(round(amount, 2))
        return amounts
    
    def _generate_ratings(self, n):
        """生成用户评分（1-5星）"""
        # 大部分评分在3-5之间，使用偏态分布
        ratings = np.random.choice([1, 2, 3, 4, 5], n, p=[0.03, 0.07, 0.15, 0.35, 0.40])
        return ratings
    
    def _generate_quantities(self, n):
        """生成购买数量"""
        # 大部分购买1-3件
        quantities = np.random.choice([1, 2, 3, 4, 5], n, p=[0.50, 0.25, 0.15, 0.07, 0.03])
        return quantities
    
    def generate_data(self):
        """
        生成完整的销售数据
        
        返回:
            DataFrame: 包含所有字段的销售数据
        """
        try:
            print(f"开始生成 {self.n_records} 条销售数据...")
            
            # 生成基础数据
            user_ids = np.random.choice(self._generate_user_ids(), self.n_records)
            product_ids = np.random.choice(self._generate_product_ids(), self.n_records)
            timestamps = self._generate_timestamps()
            categories = np.random.choice(self.categories, self.n_records, 
                                         p=[0.20, 0.18, 0.15, 0.12, 0.12, 0.10, 0.08, 0.05])
            
            # 生成金额、数量、评分
            amounts = self._generate_amounts(categories)
            quantities = self._generate_quantities(self.n_records)
            ratings = self._generate_ratings(self.n_records)
            
            # 计算实际交易金额
            total_amounts = [round(a * q, 2) for a, q in zip(amounts, quantities)]
            
            # 生成订单状态
            statuses = np.random.choice(['已完成', '已完成', '已完成', '已完成', '已退款'], 
                                       self.n_records, p=[0.75, 0.10, 0.08, 0.05, 0.02])
            
            # 生成支付方式
            payment_methods = np.random.choice(['支付宝', '微信支付', '银行卡', '信用卡', '货到付款'],
                                              self.n_records, p=[0.40, 0.35, 0.15, 0.08, 0.02])
            
            # 生成用户性别和年龄段
            genders = np.random.choice(['男', '女'], self.n_records, p=[0.45, 0.55])
            age_groups = np.random.choice(['18-25', '26-35', '36-45', '46-55', '55+'], 
                                         self.n_records, p=[0.25, 0.35, 0.25, 0.12, 0.03])
            
            # 生成地区
            regions = np.random.choice(['华东', '华南', '华北', '华中', '西南', '西北', '东北'],
                                      self.n_records, p=[0.30, 0.25, 0.18, 0.12, 0.08, 0.04, 0.03])
            
            # 构建DataFrame
            df = pd.DataFrame({
                '订单ID': [f"ORD_{str(i).zfill(6)}" for i in range(1, self.n_records + 1)],
                '用户ID': user_ids,
                '商品ID': product_ids,
                '交易时间': timestamps,
                '商品类别': categories,
                '单价': amounts,
                '数量': quantities,
                '交易金额': total_amounts,
                '用户评分': ratings,
                '订单状态': statuses,
                '支付方式': payment_methods,
                '用户性别': genders,
                '年龄段': age_groups,
                '地区': regions
            })
            
            # 添加一些特殊字段用于分析
            df['交易日期'] = pd.to_datetime(df['交易时间']).dt.date
            df['交易月份'] = pd.to_datetime(df['交易时间']).dt.to_period('M')
            df['交易星期'] = pd.to_datetime(df['交易时间']).dt.day_name()
            df['交易小时'] = pd.to_datetime(df['交易时间']).dt.hour
            
            print(f"数据生成完成！共 {len(df)} 条记录")
            print(f"时间范围: {df['交易时间'].min()} 至 {df['交易时间'].max()}")
            print(f"涉及用户数: {df['用户ID'].nunique()}")
            print(f"涉及商品数: {df['商品ID'].nunique()}")
            
            return df
            
        except Exception as e:
            print(f"数据生成过程中出现错误: {str(e)}")
            raise
    
    def save_data(self, df, filepath='../data/sales_data.csv'):
        """
        保存数据到CSV文件
        
        参数:
            df: DataFrame数据
            filepath: 保存路径
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # 保存数据
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            print(f"数据已保存至: {filepath}")
            return True
            
        except Exception as e:
            print(f"保存数据时出错: {str(e)}")
            return False
    
    def add_missing_values(self, df, missing_rate=0.05):
        """
        故意添加缺失值（用于测试清洗功能）
        
        参数:
            df: 原始数据
            missing_rate: 缺失率
            
        返回:
            DataFrame: 添加了缺失值的数据
        """
        df_with_missing = df.copy()
        n_missing = int(len(df) * missing_rate)
        
        # 随机选择行和列添加缺失值
        columns_to_missing = ['用户评分', '支付方式', '年龄段']
        for col in columns_to_missing:
            missing_indices = np.random.choice(df.index, size=n_missing // len(columns_to_missing), replace=False)
            df_with_missing.loc[missing_indices, col] = np.nan
        
        print(f"已添加 {n_missing} 个缺失值用于测试")
        return df_with_missing
    
    def add_outliers(self, df, outlier_rate=0.02):
        """
        添加异常值（用于测试清洗功能）
        
        参数:
            df: 原始数据
            outlier_rate: 异常值比例
            
        返回:
            DataFrame: 添加了异常值的数据
        """
        df_with_outliers = df.copy()
        n_outliers = int(len(df) * outlier_rate)
        
        # 添加极端金额异常值
        outlier_indices = np.random.choice(df.index, size=n_outliers, replace=False)
        df_with_outliers.loc[outlier_indices, '交易金额'] = np.random.uniform(50000, 100000, n_outliers)
        
        # 添加极端评分异常值
        rating_outliers = np.random.choice(df.index, size=n_outliers // 2, replace=False)
        df_with_outliers.loc[rating_outliers, '用户评分'] = np.random.choice([0, 6, 7, 10], len(rating_outliers))
        
        print(f"已添加 {n_outliers + len(rating_outliers)} 个异常值用于测试")
        return df_with_outliers


if __name__ == '__main__':
    # 测试数据生成
    generator = DataGenerator(n_records=800)
    data = generator.generate_data()
    
    # 添加一些缺失值和异常值用于测试
    data = generator.add_missing_values(data, missing_rate=0.03)
    data = generator.add_outliers(data, outlier_rate=0.01)
    
    # 保存数据
    generator.save_data(data, '../data/sales_data_raw.csv')
    
    print("\n数据预览:")
    print(data.head())
    print(f"\n数据形状: {data.shape}")
