import pandas as pd
import numpy as np


class DataCleaner:
    def __init__(self, data):
        self.data = data.copy()
        self.cleaning_report = {}

    def handle_missing_values(self, strategy='mean', columns=None):
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        missing_before = self.data[columns].isnull().sum().to_dict()
        
        for col in columns:
            if col not in self.data.columns:
                continue
            
            missing_count = self.data[col].isnull().sum()
            if missing_count == 0:
                continue
            
            if strategy == 'mean':
                fill_value = self.data[col].mean()
            elif strategy == 'median':
                fill_value = self.data[col].median()
            elif strategy == 'mode':
                fill_value = self.data[col].mode()[0]
            elif strategy == 'forward_fill':
                self.data[col] = self.data[col].fillna(method='ffill')
                continue
            elif strategy == 'interpolate':
                self.data[col] = self.data[col].interpolate(method='linear')
                continue
            else:
                fill_value = 0
            
            self.data[col] = self.data[col].fillna(fill_value)
            print(f"列 '{col}' 使用 {strategy} 策略填充了 {missing_count} 个缺失值")
        
        missing_after = self.data[columns].isnull().sum().to_dict()
        self.cleaning_report['missing_values'] = {
            'before': missing_before,
            'after': missing_after,
            'strategy': strategy
        }
        
        return self.data

    def detect_outliers(self, column, method='iqr', threshold=1.5):
        if column not in self.data.columns:
            raise ValueError(f"列 '{column}' 不存在")
        
        data_series = self.data[column].dropna()
        
        if method == 'iqr':
            Q1 = data_series.quantile(0.25)
            Q3 = data_series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers_mask = (self.data[column] < lower_bound) | (self.data[column] > upper_bound)
        elif method == 'zscore':
            from scipy import stats
            z_scores = np.abs(stats.zscore(data_series))
            outliers_mask = self.data[column].isin(data_series[z_scores > threshold])
        else:
            raise ValueError(f"不支持的异常值检测方法: {method}")
        
        outlier_count = outliers_mask.sum()
        outlier_indices = self.data[outliers_mask].index.tolist()
        
        self.cleaning_report['outliers'] = {
            'column': column,
            'method': method,
            'count': outlier_count,
            'indices': outlier_indices,
            'lower_bound': lower_bound if method == 'iqr' else None,
            'upper_bound': upper_bound if method == 'iqr' else None
        }
        
        print(f"检测到 {outlier_count} 个异常值 (方法: {method})")
        return outliers_mask

    def handle_outliers(self, column, method='iqr', action='clip', threshold=1.5):
        outliers_mask = self.detect_outliers(column, method, threshold)
        
        if action == 'clip':
            data_series = self.data[column].dropna()
            if method == 'iqr':
                Q1 = data_series.quantile(0.25)
                Q3 = data_series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                self.data[column] = self.data[column].clip(lower_bound, upper_bound)
                print(f"异常值已被裁剪到 [{lower_bound:.2f}, {upper_bound:.2f}] 范围")
        elif action == 'remove':
            self.data = self.data[~outliers_mask]
            print(f"已移除 {outliers_mask.sum()} 条异常记录")
        elif action == 'replace':
            median_value = self.data[column].median()
            self.data.loc[outliers_mask, column] = median_value
            print(f"异常值已被替换为中位数 {median_value:.2f}")
        
        return self.data

    def remove_duplicates(self, subset=None):
        duplicates_before = self.data.duplicated(subset=subset).sum()
        self.data = self.data.drop_duplicates(subset=subset)
        duplicates_after = self.data.duplicated(subset=subset).sum()
        
        self.cleaning_report['duplicates'] = {
            'removed': duplicates_before,
            'remaining': duplicates_after
        }
        
        print(f"移除了 {duplicates_before} 条重复记录")
        return self.data

    def get_cleaning_report(self):
        return self.cleaning_report
