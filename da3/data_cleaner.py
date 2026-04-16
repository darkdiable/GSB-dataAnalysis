"""
数据清洗模块
处理缺失值、异常值、重复值等数据质量问题
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any


class DataCleaner:
    """数据清洗类"""
    
    def __init__(self, df: pd.DataFrame):
        """
        初始化数据清洗器
        
        Args:
            df: 待清洗的DataFrame
        """
        self.df = df.copy()
        self.cleaning_report = {}
        
    def check_missing_values(self) -> pd.DataFrame:
        """
        检查缺失值情况
        
        Returns:
            缺失值统计DataFrame
        """
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df) * 100).round(2)
        
        missing_df = pd.DataFrame({
            '缺失数量': missing,
            '缺失比例(%)': missing_pct
        })
        
        self.cleaning_report['missing_values'] = missing_df[missing_df['缺失数量'] > 0]
        
        print("缺失值检查:")
        if len(missing_df[missing_df['缺失数量'] > 0]) > 0:
            print(missing_df[missing_df['缺失数量'] > 0])
        else:
            print("未发现缺失值")
            
        return missing_df
    
    def handle_missing_values(self, strategy: str = 'drop') -> pd.DataFrame:
        """
        处理缺失值
        
        Args:
            strategy: 处理策略，'drop'删除, 'fill_mean'均值填充, 'fill_median'中位数填充
            
        Returns:
            处理后的DataFrame
        """
        try:
            original_len = len(self.df)
            
            if strategy == 'drop':
                self.df = self.df.dropna()
                print(f"删除缺失值: {original_len - len(self.df)}条记录")
                
            elif strategy == 'fill_mean':
                numeric_cols = self.df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if self.df[col].isnull().any():
                        self.df[col].fillna(self.df[col].mean(), inplace=True)
                print("使用均值填充数值型缺失值")
                
            elif strategy == 'fill_median':
                numeric_cols = self.df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if self.df[col].isnull().any():
                        self.df[col].fillna(self.df[col].median(), inplace=True)
                print("使用中位数填充数值型缺失值")
            
            self.cleaning_report['missing_strategy'] = strategy
            return self.df
            
        except Exception as e:
            print(f"处理缺失值错误: {str(e)}")
            raise
    
    def check_duplicates(self) -> int:
        """
        检查重复值
        
        Returns:
            重复记录数量
        """
        duplicates = self.df.duplicated().sum()
        self.cleaning_report['duplicates'] = duplicates
        
        print(f"重复记录数量: {duplicates}")
        return duplicates
    
    def remove_duplicates(self, subset: list = None) -> pd.DataFrame:
        """
        删除重复值
        
        Args:
            subset: 用于判断重复的列名列表
            
        Returns:
            处理后的DataFrame
        """
        try:
            original_len = len(self.df)
            self.df = self.df.drop_duplicates(subset=subset, keep='first')
            removed = original_len - len(self.df)
            
            print(f"删除重复记录: {removed}条")
            self.cleaning_report['duplicates_removed'] = removed
            
            return self.df
            
        except Exception as e:
            print(f"删除重复值错误: {str(e)}")
            raise
    
    def detect_outliers_iqr(self, column: str) -> Tuple[pd.DataFrame, int]:
        """
        使用IQR方法检测异常值
        
        Args:
            column: 要检测的列名
            
        Returns:
            (异常值DataFrame, 异常值数量)
        """
        try:
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.df[(self.df[column] < lower_bound) | 
                              (self.df[column] > upper_bound)]
            
            print(f"列 '{column}' 异常值检测:")
            print(f"  正常范围: [{lower_bound:.2f}, {upper_bound:.2f}]")
            print(f"  异常值数量: {len(outliers)}")
            
            return outliers, len(outliers)
            
        except Exception as e:
            print(f"检测异常值错误: {str(e)}")
            raise
    
    def detect_outliers_zscore(self, column: str, threshold: float = 3.0) -> Tuple[pd.DataFrame, int]:
        """
        使用Z-score方法检测异常值
        
        Args:
            column: 要检测的列名
            threshold: Z-score阈值
            
        Returns:
            (异常值DataFrame, 异常值数量)
        """
        try:
            z_scores = np.abs((self.df[column] - self.df[column].mean()) / 
                             self.df[column].std())
            outliers = self.df[z_scores > threshold]
            
            print(f"列 '{column}' Z-score异常值检测:")
            print(f"  阈值: {threshold}")
            print(f"  异常值数量: {len(outliers)}")
            
            return outliers, len(outliers)
            
        except Exception as e:
            print(f"检测异常值错误: {str(e)}")
            raise
    
    def handle_outliers(self, column: str, method: str = 'clip') -> pd.DataFrame:
        """
        处理异常值
        
        Args:
            column: 要处理的列名
            method: 处理方法，'clip'截断, 'remove'删除, 'log'对数变换
            
        Returns:
            处理后的DataFrame
        """
        try:
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            if method == 'clip':
                self.df[column] = self.df[column].clip(lower_bound, upper_bound)
                print(f"异常值已截断至 [{lower_bound:.2f}, {upper_bound:.2f}]")
                
            elif method == 'remove':
                original_len = len(self.df)
                self.df = self.df[(self.df[column] >= lower_bound) & 
                                  (self.df[column] <= upper_bound)]
                print(f"删除异常值: {original_len - len(self.df)}条记录")
                
            elif method == 'log':
                self.df[column] = np.log1p(self.df[column])
                print(f"对列 '{column}' 进行对数变换")
            
            return self.df
            
        except Exception as e:
            print(f"处理异常值错误: {str(e)}")
            raise
    
    def validate_data_types(self) -> Dict[str, str]:
        """
        验证数据类型
        
        Returns:
            列名和数据类型的字典
        """
        print("数据类型检查:")
        for col in self.df.columns:
            print(f"  {col}: {self.df[col].dtype}")
        
        return self.df.dtypes.to_dict()
    
    def convert_date_column(self, column: str = 'order_date') -> pd.DataFrame:
        """
        转换日期列
        
        Args:
            column: 日期列名
            
        Returns:
            转换后的DataFrame
        """
        try:
            if column in self.df.columns:
                self.df[column] = pd.to_datetime(self.df[column])
                print(f"列 '{column}' 已转换为日期类型")
            return self.df
        except Exception as e:
            print(f"日期转换错误: {str(e)}")
            raise
    
    def get_cleaning_report(self) -> Dict[str, Any]:
        """
        获取清洗报告
        
        Returns:
            清洗报告字典
        """
        return self.cleaning_report
    
    def clean_all(self) -> pd.DataFrame:
        """
        执行完整的数据清洗流程
        
        Returns:
            清洗后的DataFrame
        """
        print("=" * 50)
        print("开始数据清洗流程")
        print("=" * 50)
        
        print("\n1. 检查缺失值...")
        self.check_missing_values()
        
        print("\n2. 处理缺失值...")
        self.handle_missing_values(strategy='drop')
        
        print("\n3. 检查重复值...")
        self.check_duplicates()
        
        print("\n4. 删除重复值...")
        self.remove_duplicates()
        
        print("\n5. 检测异常值...")
        if 'amount' in self.df.columns:
            self.detect_outliers_iqr('amount')
        if 'total_amount' in self.df.columns:
            self.detect_outliers_iqr('total_amount')
        
        print("\n6. 验证数据类型...")
        self.validate_data_types()
        
        print("\n7. 转换日期列...")
        self.convert_date_column()
        
        print("\n" + "=" * 50)
        print("数据清洗完成")
        print(f"原始记录数: {self.cleaning_report.get('original_count', 'N/A')}")
        print(f"清洗后记录数: {len(self.df)}")
        print("=" * 50)
        
        return self.df


def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    数据清洗主函数
    
    Args:
        df: 待清洗的DataFrame
        
    Returns:
        (清洗后的DataFrame, 清洗报告)
    """
    cleaner = DataCleaner(df)
    cleaner.cleaning_report['original_count'] = len(df)
    
    cleaned_df = cleaner.clean_all()
    report = cleaner.get_cleaning_report()
    
    return cleaned_df, report


if __name__ == '__main__':
    from data_generator import generate_and_save_data
    
    df = generate_and_save_data(n_records=100)
    cleaned_df, report = clean_data(df)
    
    print("\n清洗后数据预览:")
    print(cleaned_df.head())
