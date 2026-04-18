# -*- coding: utf-8 -*-
"""
数据清洗模块
处理缺失值、异常值、重复数据
"""

import pandas as pd
import numpy as np
from datetime import datetime


class DataCleaner:
    """电商销售数据清洗器"""
    
    def __init__(self, df):
        """
        初始化数据清洗器
        
        参数:
            df: 原始DataFrame数据
        """
        self.original_df = df.copy()
        self.cleaned_df = None
        self.cleaning_report = {}
        
    def detect_missing_values(self):
        """检测缺失值"""
        missing_counts = self.original_df.isnull().sum()
        missing_percent = (missing_counts / len(self.original_df)) * 100
        
        missing_info = pd.DataFrame({
            '缺失数量': missing_counts,
            '缺失比例(%)': missing_percent.round(2)
        })
        
        missing_info = missing_info[missing_info['缺失数量'] > 0]
        
        self.cleaning_report['missing_values'] = missing_info
        
        if len(missing_info) > 0:
            print("检测到缺失值:")
            print(missing_info)
        else:
            print("未检测到缺失值")
            
        return missing_info
    
    def detect_duplicates(self):
        """检测重复数据"""
        duplicate_count = self.original_df.duplicated().sum()
        
        # 基于订单ID的重复
        order_dup_count = self.original_df['订单ID'].duplicated().sum()
        
        self.cleaning_report['duplicates'] = {
            '完全重复行数': duplicate_count,
            '订单ID重复数': order_dup_count
        }
        
        print(f"检测到完全重复行: {duplicate_count}")
        print(f"检测到订单ID重复: {order_dup_count}")
        
        return duplicate_count, order_dup_count
    
    def detect_outliers(self, columns=None):
        """
        使用IQR方法检测异常值
        
        参数:
            columns: 需要检测的数值列列表
            
        返回:
            dict: 各列的异常值信息
        """
        if columns is None:
            columns = ['交易金额', '单价', '数量', '用户评分']
        
        outlier_info = {}
        
        for col in columns:
            if col in self.original_df.columns and pd.api.types.is_numeric_dtype(self.original_df[col]):
                Q1 = self.original_df[col].quantile(0.25)
                Q3 = self.original_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = self.original_df[(self.original_df[col] < lower_bound) | 
                                          (self.original_df[col] > upper_bound)]
                
                outlier_info[col] = {
                    '异常值数量': len(outliers),
                    '异常值比例(%)': round(len(outliers) / len(self.original_df) * 100, 2),
                    '下限': lower_bound,
                    '上限': upper_bound,
                    '最小值': self.original_df[col].min(),
                    '最大值': self.original_df[col].max()
                }
        
        self.cleaning_report['outliers'] = outlier_info
        
        print("\n异常值检测结果:")
        for col, info in outlier_info.items():
            print(f"{col}: {info['异常值数量']} 个异常值 ({info['异常值比例(%)']}%)")
        
        return outlier_info
    
    def handle_missing_values(self, df):
        """
        处理缺失值
        
        策略:
        - 用户评分: 用中位数填充
        - 支付方式: 用众数填充
        - 年龄段: 用众数填充
        """
        df_cleaned = df.copy()
        
        # 处理用户评分缺失
        if '用户评分' in df_cleaned.columns and df_cleaned['用户评分'].isnull().sum() > 0:
            median_rating = df_cleaned['用户评分'].median()
            df_cleaned['用户评分'].fillna(median_rating, inplace=True)
            print(f"用户评分缺失值已用中位数 {median_rating} 填充")
        
        # 处理支付方式缺失
        if '支付方式' in df_cleaned.columns and df_cleaned['支付方式'].isnull().sum() > 0:
            mode_payment = df_cleaned['支付方式'].mode()[0]
            df_cleaned['支付方式'].fillna(mode_payment, inplace=True)
            print(f"支付方式缺失值已用众数 '{mode_payment}' 填充")
        
        # 处理年龄段缺失
        if '年龄段' in df_cleaned.columns and df_cleaned['年龄段'].isnull().sum() > 0:
            mode_age = df_cleaned['年龄段'].mode()[0]
            df_cleaned['年龄段'].fillna(mode_age, inplace=True)
            print(f"年龄段缺失值已用众数 '{mode_age}' 填充")
        
        return df_cleaned
    
    def handle_duplicates(self, df):
        """
        处理重复数据
        
        策略:
        - 删除完全重复的行
        - 保留第一个订单ID重复的记录
        """
        df_cleaned = df.copy()
        
        # 删除完全重复的行
        before_count = len(df_cleaned)
        df_cleaned = df_cleaned.drop_duplicates()
        after_count = len(df_cleaned)
        
        if before_count != after_count:
            print(f"删除了 {before_count - after_count} 行完全重复数据")
        
        # 处理订单ID重复（保留第一条）
        before_count = len(df_cleaned)
        df_cleaned = df_cleaned.drop_duplicates(subset=['订单ID'], keep='first')
        after_count = len(df_cleaned)
        
        if before_count != after_count:
            print(f"删除了 {before_count - after_count} 行订单ID重复数据")
        
        return df_cleaned
    
    def handle_outliers(self, df):
        """
        处理异常值
        
        策略:
        - 交易金额: 使用IQR方法，超出范围的用上下限替换
        - 用户评分: 限制在1-5范围内
        """
        df_cleaned = df.copy()
        
        # 处理交易金额异常值
        if '交易金额' in df_cleaned.columns:
            Q1 = df_cleaned['交易金额'].quantile(0.25)
            Q3 = df_cleaned['交易金额'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # 替换异常值
            outliers_count = ((df_cleaned['交易金额'] < lower_bound) | 
                            (df_cleaned['交易金额'] > upper_bound)).sum()
            
            df_cleaned['交易金额'] = df_cleaned['交易金额'].clip(lower_bound, upper_bound)
            
            if outliers_count > 0:
                print(f"交易金额异常值已处理: {outliers_count} 个值被限制在 [{lower_bound:.2f}, {upper_bound:.2f}]")
        
        # 处理用户评分异常值（限制在1-5）
        if '用户评分' in df_cleaned.columns:
            invalid_ratings = ((df_cleaned['用户评分'] < 1) | (df_cleaned['用户评分'] > 5)).sum()
            df_cleaned['用户评分'] = df_cleaned['用户评分'].clip(1, 5)
            
            if invalid_ratings > 0:
                print(f"用户评分异常值已处理: {invalid_ratings} 个值被限制在 [1, 5]")
        
        # 处理单价异常值
        if '单价' in df_cleaned.columns:
            Q1 = df_cleaned['单价'].quantile(0.25)
            Q3 = df_cleaned['单价'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = max(0, Q1 - 1.5 * IQR)  # 单价不能为负
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_count = ((df_cleaned['单价'] < lower_bound) | 
                            (df_cleaned['单价'] > upper_bound)).sum()
            
            df_cleaned['单价'] = df_cleaned['单价'].clip(lower_bound, upper_bound)
            
            if outliers_count > 0:
                print(f"单价异常值已处理: {outliers_count} 个值被限制")
        
        return df_cleaned
    
    def clean_data(self):
        """
        执行完整的数据清洗流程
        
        返回:
            DataFrame: 清洗后的数据
        """
        print("=" * 50)
        print("开始数据清洗...")
        print("=" * 50)
        
        # 步骤1: 检测数据问题
        print("\n【步骤1】检测缺失值...")
        self.detect_missing_values()
        
        print("\n【步骤2】检测重复数据...")
        self.detect_duplicates()
        
        print("\n【步骤3】检测异常值...")
        self.detect_outliers()
        
        # 步骤2: 处理数据问题
        df_cleaned = self.original_df.copy()
        
        print("\n【步骤4】处理缺失值...")
        df_cleaned = self.handle_missing_values(df_cleaned)
        
        print("\n【步骤5】处理重复数据...")
        df_cleaned = self.handle_duplicates(df_cleaned)
        
        print("\n【步骤6】处理异常值...")
        df_cleaned = self.handle_outliers(df_cleaned)
        
        # 记录清洗结果
        self.cleaned_df = df_cleaned
        self.cleaning_report['original_count'] = len(self.original_df)
        self.cleaning_report['cleaned_count'] = len(df_cleaned)
        self.cleaning_report['removed_count'] = len(self.original_df) - len(df_cleaned)
        
        print("\n" + "=" * 50)
        print("数据清洗完成!")
        print(f"原始数据: {len(self.original_df)} 条")
        print(f"清洗后数据: {len(df_cleaned)} 条")
        print(f"删除/修正: {len(self.original_df) - len(df_cleaned)} 条")
        print("=" * 50)
        
        return df_cleaned
    
    def get_cleaning_report(self):
        """获取清洗报告"""
        return self.cleaning_report
    
    def save_cleaned_data(self, filepath='../data/sales_data_cleaned.csv'):
        """保存清洗后的数据"""
        if self.cleaned_df is not None:
            self.cleaned_df.to_csv(filepath, index=False, encoding='utf-8-sig')
            print(f"清洗后的数据已保存至: {filepath}")
            return True
        else:
            print("错误: 请先执行clean_data()方法")
            return False


if __name__ == '__main__':
    # 测试数据清洗
    # 先生成测试数据
    from data_generator import DataGenerator
    
    generator = DataGenerator(n_records=500)
    raw_data = generator.generate_data()
    raw_data = generator.add_missing_values(raw_data, missing_rate=0.05)
    raw_data = generator.add_outliers(raw_data, outlier_rate=0.03)
    
    # 执行清洗
    cleaner = DataCleaner(raw_data)
    cleaned_data = cleaner.clean_data()
    
    print("\n清洗后数据预览:")
    print(cleaned_data.head())
