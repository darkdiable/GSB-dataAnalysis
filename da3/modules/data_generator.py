"""
数据生成模块
模拟真实电商销售数据，包含用户ID、时间、金额、类别、评分等字段
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os


class DataGenerator:
    """电商销售数据生成器"""
    
    def __init__(self, n_records=1000, start_date=None, end_date=None, random_seed=42):
        """
        初始化数据生成器
        
        Args:
            n_records: 生成记录数，默认1000条
            start_date: 开始日期，默认3个月前
            end_date: 结束日期，默认今天
            random_seed: 随机种子，保证可重复性
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
        
        # 定义商品类别及其权重
        self.categories = {
            '电子产品': 0.25,
            '服装鞋帽': 0.30,
            '食品饮料': 0.20,
            '家居用品': 0.15,
            '美妆护肤': 0.10
        }
        
        # 定义各类别的价格区间
        self.price_ranges = {
            '电子产品': (500, 5000),
            '服装鞋帽': (50, 800),
            '食品饮料': (10, 300),
            '家居用品': (30, 1000),
            '美妆护肤': (100, 1500)
        }
        
        # 定义用户数量
        self.n_users = 200
        
    def _generate_user_ids(self):
        """生成用户ID列表"""
        return [f"USER_{str(i).zfill(5)}" for i in range(1, self.n_users + 1)]
    
    def _generate_order_dates(self):
        """生成订单日期，模拟真实的时间分布（周末和促销期销量更高）"""
        dates = []
        date_range = (self.end_date - self.start_date).days
        
        # 先生成基础日期，再根据周末/工作日加权
        base_dates = []
        while len(base_dates) < self.n_records:
            days_offset = np.random.randint(0, date_range)
            base_date = self.start_date + timedelta(days=int(days_offset))
            
            # 周末权重更高，增加周末样本
            if base_date.weekday() >= 5:  # 周末
                # 周末记录增加权重
                n_samples = np.random.choice([1, 2], p=[0.3, 0.7])
                for _ in range(n_samples):
                    if len(base_dates) < self.n_records:
                        base_dates.append(base_date)
            else:
                base_dates.append(base_date)
        
        # 为每个日期添加具体时间
        for base_date in base_dates[:self.n_records]:
            if base_date.weekday() >= 5:  # 周末
                hour = np.random.choice(
                    range(24), 
                    p=self._get_hour_weights_weekend()
                )
            else:  # 工作日
                hour = np.random.choice(
                    range(24), 
                    p=self._get_hour_weights_weekday()
                )
            
            minute = np.random.randint(0, 60)
            second = np.random.randint(0, 60)
            
            order_date = base_date.replace(hour=hour, minute=minute, second=second)
            dates.append(order_date)
        
        return dates
    
    def _get_hour_weights_weekday(self):
        """获取工作日小时权重"""
        weights = np.ones(24) * 0.02
        weights[9:12] = 0.06  # 上午高峰
        weights[14:18] = 0.05  # 下午高峰
        weights[20:23] = 0.08  # 晚间高峰
        return weights / weights.sum()
    
    def _get_hour_weights_weekend(self):
        """获取周末小时权重"""
        weights = np.ones(24) * 0.03
        weights[10:14] = 0.07  # 上午高峰
        weights[15:18] = 0.06  # 下午高峰
        weights[19:23] = 0.09  # 晚间高峰
        return weights / weights.sum()
    
    def _generate_categories(self):
        """生成商品类别"""
        categories = list(self.categories.keys())
        weights = list(self.categories.values())
        return np.random.choice(categories, size=self.n_records, p=weights)
    
    def _generate_amounts(self, categories):
        """根据类别生成订单金额"""
        amounts = []
        for cat in categories:
            min_price, max_price = self.price_ranges[cat]
            # 使用对数正态分布模拟真实价格分布
            mean = np.log((min_price + max_price) / 2)
            sigma = 0.5
            amount = np.random.lognormal(mean, sigma)
            # 限制在合理范围内
            amount = max(min_price, min(max_price, amount))
            amounts.append(round(amount, 2))
        return amounts
    
    def _generate_ratings(self, amounts):
        """生成用户评分，与金额有一定相关性"""
        ratings = []
        for amount in amounts:
            # 基础评分
            base_rating = np.random.normal(4.0, 0.8)
            # 高金额订单倾向于更高评分（可能更谨慎选择）
            if amount > 1000:
                base_rating += 0.2
            # 限制在1-5范围内
            rating = max(1, min(5, base_rating))
            ratings.append(round(rating, 1))
        return ratings
    
    def _generate_quantities(self):
        """生成购买数量"""
        # 大部分订单购买1-3件
        return np.random.choice([1, 2, 3, 4, 5], 
                               size=self.n_records, 
                               p=[0.5, 0.25, 0.15, 0.07, 0.03])
    
    def _generate_payment_methods(self):
        """生成支付方式"""
        methods = ['支付宝', '微信支付', '银行卡', '信用卡', '货到付款']
        weights = [0.35, 0.40, 0.10, 0.10, 0.05]
        return np.random.choice(methods, size=self.n_records, p=weights)
    
    def _generate_regions(self):
        """生成地区"""
        regions = ['华东', '华南', '华北', '华中', '西南', '西北', '东北']
        weights = [0.30, 0.25, 0.18, 0.12, 0.08, 0.04, 0.03]
        return np.random.choice(regions, size=self.n_records, p=weights)
    
    def generate_data(self):
        """
        生成完整的电商销售数据
        
        Returns:
            DataFrame: 包含所有字段的销售数据
        """
        try:
            print(f"开始生成 {self.n_records} 条销售记录...")
            
            # 生成用户ID
            user_ids = np.random.choice(
                self._generate_user_ids(), 
                size=self.n_records
            )
            
            # 生成订单日期
            order_dates = self._generate_order_dates()
            
            # 生成商品类别
            categories = self._generate_categories()
            
            # 生成订单金额
            amounts = self._generate_amounts(categories)
            
            # 生成购买数量
            quantities = self._generate_quantities()
            
            # 计算总金额
            total_amounts = [round(a * q, 2) for a, q in zip(amounts, quantities)]
            
            # 生成用户评分
            ratings = self._generate_ratings(total_amounts)
            
            # 生成支付方式
            payment_methods = self._generate_payment_methods()
            
            # 生成地区
            regions = self._generate_regions()
            
            # 生成订单ID
            order_ids = [f"ORD{datetime.now().strftime('%Y%m%d')}{str(i).zfill(6)}" 
                        for i in range(1, len(order_dates) + 1)]
            
            # 创建DataFrame
            df = pd.DataFrame({
                '订单ID': order_ids,
                '用户ID': user_ids,
                '订单时间': order_dates,
                '商品类别': categories,
                '单价': amounts,
                '数量': quantities,
                '总金额': total_amounts,
                '用户评分': ratings,
                '支付方式': payment_methods,
                '地区': regions
            })
            
            # 按时间排序
            df = df.sort_values('订单时间').reset_index(drop=True)
            
            print(f"成功生成 {len(df)} 条销售记录")
            print(f"时间跨度: {df['订单时间'].min()} 至 {df['订单时间'].max()}")
            
            return df
            
        except Exception as e:
            print(f"数据生成失败: {str(e)}")
            raise
    
    def add_missing_values(self, df, missing_rate=0.05):
        """
        故意添加缺失值以模拟真实数据情况
        
        Args:
            df: 原始数据框
            missing_rate: 缺失率，默认5%
            
        Returns:
            DataFrame: 包含缺失值的数据
        """
        df_with_missing = df.copy()
        n_missing = int(len(df) * missing_rate)
        
        # 随机选择部分记录的评分字段设为缺失
        missing_indices = np.random.choice(df.index, size=n_missing//2, replace=False)
        df_with_missing.loc[missing_indices, '用户评分'] = np.nan
        
        # 随机选择少量记录的支付方式设为缺失
        missing_indices = np.random.choice(df.index, size=n_missing//4, replace=False)
        df_with_missing.loc[missing_indices, '支付方式'] = np.nan
        
        print(f"已添加缺失值: {df_with_missing.isnull().sum().sum()} 个")
        return df_with_missing
    
    def add_outliers(self, df, outlier_rate=0.02):
        """
        添加异常值以模拟真实数据情况
        
        Args:
            df: 原始数据框
            outlier_rate: 异常值比例，默认2%
            
        Returns:
            DataFrame: 包含异常值的数据
        """
        df_with_outliers = df.copy()
        n_outliers = int(len(df) * outlier_rate)
        
        # 随机选择部分记录，将金额设为异常高或异常低
        outlier_indices = np.random.choice(df.index, size=n_outliers, replace=False)
        for idx in outlier_indices:
            if np.random.random() < 0.5:
                # 异常高金额
                df_with_outliers.loc[idx, '总金额'] = df_with_outliers.loc[idx, '总金额'] * 10
            else:
                # 异常低金额（可能是退款或错误）
                df_with_outliers.loc[idx, '总金额'] = 0.01
        
        print(f"已添加 {n_outliers} 个异常值")
        return df_with_outliers
    
    def add_duplicates(self, df, duplicate_rate=0.03):
        """
        添加重复记录以模拟真实数据情况
        
        Args:
            df: 原始数据框
            duplicate_rate: 重复率，默认3%
            
        Returns:
            DataFrame: 包含重复记录的数据
        """
        n_duplicates = int(len(df) * duplicate_rate)
        duplicate_indices = np.random.choice(df.index, size=n_duplicates, replace=False)
        duplicates = df.loc[duplicate_indices].copy()
        
        df_with_duplicates = pd.concat([df, duplicates], ignore_index=True)
        print(f"已添加 {n_duplicates} 条重复记录")
        return df_with_duplicates
    
    def save_to_csv(self, df, filepath):
        """
        保存数据到CSV文件
        
        Args:
            df: 数据框
            filepath: 文件路径
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            print(f"数据已保存至: {filepath}")
        except Exception as e:
            print(f"保存数据失败: {str(e)}")
            raise


def generate_raw_data(output_path='data/raw_sales_data.csv', n_records=1000):
    """
    生成原始数据（包含缺失值、异常值和重复值）
    
    Args:
        output_path: 输出文件路径
        n_records: 记录数
        
    Returns:
        DataFrame: 生成的原始数据
    """
    generator = DataGenerator(n_records=n_records)
    
    # 生成基础数据
    df = generator.generate_data()
    
    # 添加数据质量问题
    df = generator.add_missing_values(df, missing_rate=0.05)
    df = generator.add_outliers(df, outlier_rate=0.02)
    df = generator.add_duplicates(df, duplicate_rate=0.03)
    
    # 保存数据
    generator.save_to_csv(df, output_path)
    
    return df


if __name__ == '__main__':
    # 测试数据生成
    df = generate_raw_data(n_records=1000)
    print("\n数据预览:")
    print(df.head())
    print("\n数据统计:")
    print(df.describe())
