import pandas as pd
import numpy as np
from scipy import stats


class DataPreprocessor:
    def __init__(self, data):
        self.data = data.copy()
        self.original_stats = {}

    def handle_missing_values(self, method='interpolate'):
        """
        处理缺失值
        method: 'interpolate' - 线性插值, 'forward' - 前向填充, 'mean' - 均值填充
        """
        numeric_columns = ['temperature', 'humidity', 'precipitation', 'wind_speed', 'pressure']
        
        missing_before = self.data[numeric_columns].isnull().sum().sum()
        
        for col in numeric_columns:
            if col in self.data.columns:
                if method == 'interpolate':
                    self.data[col] = self.data[col].interpolate(method='linear')
                    self.data[col] = self.data[col].fillna(method='ffill').fillna(method='bfill')
                elif method == 'forward':
                    self.data[col] = self.data[col].fillna(method='ffill').fillna(method='bfill')
                elif method == 'mean':
                    self.data[col] = self.data[col].fillna(self.data[col].mean())
        
        missing_after = self.data[numeric_columns].isnull().sum().sum()
        print(f"缺失值处理: 从 {missing_before} 减少到 {missing_after}")
        return self.data

    def remove_outliers(self, method='iqr', threshold=3):
        """
        异常值剔除
        method: 'iqr' - IQR方法, 'zscore' - Z-score方法
        """
        numeric_columns = ['temperature', 'humidity', 'precipitation', 'wind_speed', 'pressure']
        outlier_info = {}
        
        for col in numeric_columns:
            if col in self.data.columns:
                original_count = len(self.data)
                
                if method == 'iqr':
                    Q1 = self.data[col].quantile(0.25)
                    Q3 = self.data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = (self.data[col] < lower_bound) | (self.data[col] > upper_bound)
                    
                elif method == 'zscore':
                    z_scores = np.abs(stats.zscore(self.data[col].dropna()))
                    outliers = pd.Series(False, index=self.data.index)
                    outliers[self.data[col].dropna().index] = z_scores > threshold
                
                outlier_count = outliers.sum()
                outlier_info[col] = {
                    'count': outlier_count,
                    'percentage': round(outlier_count / original_count * 100, 2)
                }
                
                # 使用中位数替换异常值
                median_value = self.data[col].median()
                self.data.loc[outliers, col] = median_value
        
        print("异常值剔除信息:")
        for col, info in outlier_info.items():
            print(f"  {col}: {info['count']} 个 ({info['percentage']}%)")
        
        return self.data

    def add_time_features(self):
        """
        添加时间特征
        """
        self.data['year'] = self.data['date'].dt.year
        self.data['month'] = self.data['date'].dt.month
        self.data['day'] = self.data['date'].dt.day
        self.data['dayofyear'] = self.data['date'].dt.dayofyear
        self.data['weekofyear'] = self.data['date'].dt.isocalendar().week
        self.data['quarter'] = self.data['date'].dt.quarter
        
        # 季节
        self.data['season'] = self.data['month'].map({
            12: '冬季', 1: '冬季', 2: '冬季',
            3: '春季', 4: '春季', 5: '春季',
            6: '夏季', 7: '夏季', 8: '夏季',
            9: '秋季', 10: '秋季', 11: '秋季'
        })
        
        return self.data

    def normalize_features(self, columns=None):
        """
        特征标准化 (Z-score标准化)
        """
        if columns is None:
            columns = ['temperature', 'humidity', 'precipitation', 'wind_speed', 'pressure']
        
        for col in columns:
            if col in self.data.columns:
                mean = self.data[col].mean()
                std = self.data[col].std()
                self.data[f'{col}_normalized'] = (self.data[col] - mean) / std
        
        return self.data

    def get_preprocessed_data(self):
        """
        获取预处理后的数据
        """
        return self.data

    def get_statistics(self):
        """
        获取数据统计信息
        """
        numeric_columns = ['temperature', 'humidity', 'precipitation', 'wind_speed', 'pressure']
        stats_dict = {}
        
        for col in numeric_columns:
            if col in self.data.columns:
                stats_dict[col] = {
                    'mean': round(self.data[col].mean(), 2),
                    'std': round(self.data[col].std(), 2),
                    'min': round(self.data[col].min(), 2),
                    'max': round(self.data[col].max(), 2),
                    'median': round(self.data[col].median(), 2)
                }
        
        return stats_dict
