"""
数据清洗模块 - 处理缺失值和异常值
"""
import pandas as pd
import numpy as np


class DataCleaner:
    """数据清洗器类，处理缺失值和异常值"""
    
    def __init__(self, outlier_threshold=3):
        """
        初始化数据清洗器
        
        Parameters:
            outlier_threshold: 异常值检测阈值（标准差倍数）
        """
        self.outlier_threshold = outlier_threshold
        self.cleaning_report = {}
    
    def clean(self, df):
        """
        执行完整的数据清洗流程
        
        Parameters:
            df: 原始数据DataFrame
            
        Returns:
            DataFrame: 清洗后的数据
        """
        original_count = len(df)
        df_cleaned = df.copy()
        
        # 1. 处理缺失值
        df_cleaned = self._handle_missing_values(df_cleaned)
        
        # 2. 检测和处理异常值
        df_cleaned = self._handle_outliers(df_cleaned)
        
        # 3. 数据类型转换
        df_cleaned = self._convert_data_types(df_cleaned)
        
        # 生成清洗报告
        self.cleaning_report = {
            '原始数据量': original_count,
            '清洗后数据量': len(df_cleaned),
            '删除记录数': original_count - len(df_cleaned)
        }
        
        print(f"数据清洗完成: 原始 {original_count} 条 -> 清洗后 {len(df_cleaned)} 条")
        return df_cleaned
    
    def _handle_missing_values(self, df):
        """
        处理缺失值
        
        策略：
        - 日期列：删除缺失
        - 销售额列：使用线性插值填充
        """
        df_cleaned = df.copy()
        missing_before = df_cleaned.isnull().sum().sum()
        
        # 删除日期缺失的行
        if 'date' in df_cleaned.columns:
            df_cleaned = df_cleaned.dropna(subset=['date'])
        
        # 销售额缺失值使用线性插值
        if 'sales' in df_cleaned.columns:
            missing_sales = df_cleaned['sales'].isnull().sum()
            if missing_sales > 0:
                df_cleaned['sales'] = df_cleaned['sales'].interpolate(method='linear')
                # 如果首尾有缺失，使用前向/后向填充
                df_cleaned['sales'] = df_cleaned['sales'].fillna(method='ffill').fillna(method='bfill')
                print(f"插值填充销售额缺失值: {missing_sales} 条")
        
        missing_after = df_cleaned.isnull().sum().sum()
        self.cleaning_report['缺失值处理'] = f"处理前 {missing_before} 个，处理后 {missing_after} 个"
        
        return df_cleaned
    
    def _handle_outliers(self, df):
        """
        检测和处理异常值
        
        使用Z-score方法检测异常值，并用上下边界值替换
        """
        df_cleaned = df.copy()
        
        if 'sales' not in df_cleaned.columns:
            return df_cleaned
        
        # 计算Z-score
        mean = df_cleaned['sales'].mean()
        std = df_cleaned['sales'].std()
        z_scores = np.abs((df_cleaned['sales'] - mean) / std)
        
        # 识别异常值
        outlier_mask = z_scores > self.outlier_threshold
        outlier_count = outlier_mask.sum()
        
        if outlier_count > 0:
            # 使用上下边界值替换异常值（Winsorization）
            lower_bound = mean - self.outlier_threshold * std
            upper_bound = mean + self.outlier_threshold * std
            
            df_cleaned.loc[df_cleaned['sales'] < lower_bound, 'sales'] = lower_bound
            df_cleaned.loc[df_cleaned['sales'] > upper_bound, 'sales'] = upper_bound
            
            print(f"检测到并处理 {outlier_count} 个异常值 (|Z-score| > {self.outlier_threshold})")
        else:
            print(f"未检测到异常值 (阈值: {self.outlier_threshold}倍标准差)")
        
        self.cleaning_report['异常值检测'] = f"检测到 {outlier_count} 个异常值，已用边界值替换"
        
        return df_cleaned
    
    def _convert_data_types(self, df):
        """转换数据类型"""
        df_cleaned = df.copy()
        
        # 确保日期列为datetime类型
        if 'date' in df_cleaned.columns:
            df_cleaned['date'] = pd.to_datetime(df_cleaned['date'])
        
        # 确保销售额为数值类型
        if 'sales' in df_cleaned.columns:
            df_cleaned['sales'] = pd.to_numeric(df_cleaned['sales'], errors='coerce')
        
        return df_cleaned
    
    def get_cleaning_report(self):
        """获取清洗报告"""
        return self.cleaning_report
