"""
数据清洗模块
处理缺失值、异常值、重复值等数据质量问题
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any


class DataCleaner:
    """数据清洗器"""
    
    def __init__(self, df: pd.DataFrame):
        """
        初始化数据清洗器
        
        Args:
            df: 待清洗的DataFrame
        """
        self.df = df.copy()
        self.clean_log = []
    
    def get_data_quality_report(self) -> Dict[str, Any]:
        """
        生成数据质量报告
        
        Returns:
            包含数据质量指标的字典
        """
        try:
            report = {
                'total_records': len(self.df),
                'total_columns': len(self.df.columns),
                'missing_values': self.df.isnull().sum().to_dict(),
                'missing_percentage': (self.df.isnull().sum() / len(self.df) * 100).to_dict(),
                'duplicate_records': self.df.duplicated().sum(),
                'duplicate_percentage': self.df.duplicated().sum() / len(self.df) * 100,
                'memory_usage': self.df.memory_usage(deep=True).sum() / 1024 / 1024
            }
            
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                q1 = self.df[col].quantile(0.25)
                q3 = self.df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
                report[f'{col}_outliers'] = outliers
            
            return report
            
        except Exception as e:
            raise RuntimeError(f"生成数据质量报告失败: {str(e)}")
    
    def handle_missing_values(self, strategy: str = 'auto') -> pd.DataFrame:
        """
        处理缺失值
        
        Args:
            strategy: 处理策略 ('auto', 'drop', 'mean', 'median', 'mode', 'fill')
            
        Returns:
            处理后的DataFrame
        """
        try:
            missing_before = self.df.isnull().sum().sum()
            
            for col in self.df.columns:
                if self.df[col].isnull().sum() > 0:
                    if strategy == 'auto':
                        if self.df[col].dtype in ['int64', 'float64']:
                            self.df[col].fillna(self.df[col].median(), inplace=True)
                        else:
                            mode_val = self.df[col].mode()
                            if len(mode_val) > 0:
                                self.df[col].fillna(mode_val[0], inplace=True)
                    elif strategy == 'drop':
                        self.df.dropna(subset=[col], inplace=True)
                    elif strategy == 'mean' and self.df[col].dtype in ['int64', 'float64']:
                        self.df[col].fillna(self.df[col].mean(), inplace=True)
                    elif strategy == 'median' and self.df[col].dtype in ['int64', 'float64']:
                        self.df[col].fillna(self.df[col].median(), inplace=True)
                    elif strategy == 'mode':
                        mode_val = self.df[col].mode()
                        if len(mode_val) > 0:
                            self.df[col].fillna(mode_val[0], inplace=True)
            
            missing_after = self.df.isnull().sum().sum()
            self.clean_log.append(f"缺失值处理: {missing_before} -> {missing_after} (策略: {strategy})")
            
            return self.df
            
        except Exception as e:
            raise RuntimeError(f"处理缺失值失败: {str(e)}")
    
    def handle_duplicates(self, keep: str = 'first') -> pd.DataFrame:
        """
        处理重复值
        
        Args:
            keep: 保留策略 ('first', 'last', False)
            
        Returns:
            处理后的DataFrame
        """
        try:
            duplicates_before = self.df.duplicated().sum()
            
            self.df.drop_duplicates(keep=keep, inplace=True)
            self.df.reset_index(drop=True, inplace=True)
            
            duplicates_after = self.df.duplicated().sum()
            self.clean_log.append(f"重复值处理: {duplicates_before} -> {duplicates_after}")
            
            return self.df
            
        except Exception as e:
            raise RuntimeError(f"处理重复值失败: {str(e)}")
    
    def handle_outliers(self, columns: list = None, 
                        method: str = 'iqr',
                        action: str = 'cap') -> pd.DataFrame:
        """
        处理异常值
        
        Args:
            columns: 要处理的列，默认为所有数值列
            method: 检测方法 ('iqr', 'zscore')
            action: 处理方式 ('cap', 'remove', 'mark')
            
        Returns:
            处理后的DataFrame
        """
        try:
            if columns is None:
                columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
            
            outliers_info = {}
            
            for col in columns:
                if col not in self.df.columns:
                    continue
                    
                if method == 'iqr':
                    q1 = self.df[col].quantile(0.25)
                    q3 = self.df[col].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                elif method == 'zscore':
                    mean = self.df[col].mean()
                    std = self.df[col].std()
                    lower_bound = mean - 3 * std
                    upper_bound = mean + 3 * std
                
                outliers_mask = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
                outliers_count = outliers_mask.sum()
                
                if outliers_count > 0:
                    outliers_info[col] = outliers_count
                    
                    if action == 'cap':
                        self.df.loc[self.df[col] < lower_bound, col] = lower_bound
                        self.df.loc[self.df[col] > upper_bound, col] = upper_bound
                    elif action == 'remove':
                        self.df = self.df[~outliers_mask]
                    elif action == 'mark':
                        self.df[f'{col}_is_outlier'] = outliers_mask
            
            self.clean_log.append(f"异常值处理: {outliers_info} (方法: {method}, 动作: {action})")
            self.df.reset_index(drop=True, inplace=True)
            
            return self.df
            
        except Exception as e:
            raise RuntimeError(f"处理异常值失败: {str(e)}")
    
    def validate_data(self) -> Tuple[bool, list]:
        """
        验证数据有效性
        
        Returns:
            (是否通过验证, 问题列表)
        """
        try:
            issues = []
            is_valid = True
            
            if self.df.isnull().sum().sum() > 0:
                issues.append(f"存在缺失值: {self.df.isnull().sum().to_dict()}")
                is_valid = False
            
            if self.df.duplicated().sum() > 0:
                issues.append(f"存在重复值: {self.df.duplicated().sum()}条")
                is_valid = False
            
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if (self.df[col] < 0).any():
                    issues.append(f"{col}列存在负值")
                    is_valid = False
            
            if 'amount' in self.df.columns and self.df['amount'].max() > 100000:
                issues.append("金额列存在异常大值")
                is_valid = False
            
            if 'rating' in self.df.columns:
                invalid_ratings = self.df[(self.df['rating'] < 1) | (self.df['rating'] > 5)]
                if len(invalid_ratings) > 0:
                    issues.append(f"评分列存在无效值: {len(invalid_ratings)}条")
                    is_valid = False
            
            return is_valid, issues
            
        except Exception as e:
            raise RuntimeError(f"验证数据失败: {str(e)}")
    
    def clean_all(self) -> pd.DataFrame:
        """
        执行完整的数据清洗流程
        
        Returns:
            清洗后的DataFrame
        """
        try:
            print("=" * 50)
            print("开始数据清洗...")
            print("=" * 50)
            
            quality_report = self.get_data_quality_report()
            print(f"\n数据质量报告:")
            print(f"  总记录数: {quality_report['total_records']}")
            print(f"  缺失值: {sum(quality_report['missing_values'].values())}")
            print(f"  重复值: {quality_report['duplicate_records']}")
            
            self.handle_missing_values(strategy='auto')
            print("  ✓ 缺失值处理完成")
            
            self.handle_duplicates(keep='first')
            print("  ✓ 重复值处理完成")
            
            self.handle_outliers(method='iqr', action='cap')
            print("  ✓ 异常值处理完成")
            
            is_valid, issues = self.validate_data()
            if is_valid:
                print("\n✓ 数据清洗完成，数据验证通过")
            else:
                print(f"\n⚠ 数据清洗完成，但仍存在问题: {issues}")
            
            print("=" * 50)
            
            return self.df
            
        except Exception as e:
            raise RuntimeError(f"数据清洗流程失败: {str(e)}")
    
    def get_clean_log(self) -> list:
        """
        获取清洗日志
        
        Returns:
            清洗操作日志列表
        """
        return self.clean_log


def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    便捷函数：清洗数据
    
    Args:
        df: 待清洗的DataFrame
        
    Returns:
        (清洗后的DataFrame, 数据质量报告)
    """
    cleaner = DataCleaner(df)
    quality_report = cleaner.get_data_quality_report()
    cleaned_df = cleaner.clean_all()
    
    return cleaned_df, quality_report


if __name__ == '__main__':
    from data_generator import DataGenerator
    
    generator = DataGenerator()
    df = generator.generate_data(n_records=100)
    df_dirty = generator.introduce_issues(df)
    
    cleaner = DataCleaner(df_dirty)
    cleaned_df = cleaner.clean_all()
    
    print("\n清洗日志:")
    for log in cleaner.get_clean_log():
        print(f"  - {log}")
