import pandas as pd
import numpy as np


class Preprocessor:
    def __init__(self, data):
        self.data = data.copy()
        self.preprocessing_log = []

    def fill_missing_values(self, method='interpolate'):
        missing_before = self.data.isnull().sum()
        self.preprocessing_log.append(f"缺失值填充前: {missing_before.to_dict()}")

        numeric_cols = self.data.select_dtypes(include=[np.number]).columns

        if method == 'interpolate':
            for col in numeric_cols:
                self.data[col] = self.data[col].interpolate(method='linear')
        elif method == 'mean':
            for col in numeric_cols:
                self.data[col].fillna(self.data[col].mean(), inplace=True)
        elif method == 'forward':
            for col in numeric_cols:
                self.data[col].fillna(method='ffill', inplace=True)
                self.data[col].fillna(method='bfill', inplace=True)

        missing_after = self.data.isnull().sum()
        self.preprocessing_log.append(f"缺失值填充后: {missing_after.to_dict()}")

        return self.data

    def remove_outliers(self, columns=None, method='iqr', threshold=1.5):
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns

        outliers_info = {}

        for col in columns:
            if col not in self.data.columns:
                continue

            original_count = len(self.data)

            if method == 'iqr':
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR

                outlier_mask = (self.data[col] < lower_bound) | (self.data[col] > upper_bound)
                outliers_count = outlier_mask.sum()

                self.data.loc[outlier_mask, col] = np.nan
                self.data[col] = self.data[col].interpolate(method='linear')

                outliers_info[col] = {
                    'count': outliers_count,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }

            elif method == 'zscore':
                mean = self.data[col].mean()
                std = self.data[col].std()
                z_scores = np.abs((self.data[col] - mean) / std)

                outlier_mask = z_scores > threshold
                outliers_count = outlier_mask.sum()

                self.data.loc[outlier_mask, col] = np.nan
                self.data[col] = self.data[col].interpolate(method='linear')

                outliers_info[col] = {
                    'count': outliers_count,
                    'threshold': threshold
                }

        self.preprocessing_log.append(f"异常值剔除信息: {outliers_info}")
        return self.data

    def normalize_data(self, columns=None, method='minmax'):
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns

        normalized_data = self.data.copy()

        if method == 'minmax':
            for col in columns:
                min_val = self.data[col].min()
                max_val = self.data[col].max()
                if max_val - min_val != 0:
                    normalized_data[col] = (self.data[col] - min_val) / (max_val - min_val)
        elif method == 'standard':
            for col in columns:
                mean_val = self.data[col].mean()
                std_val = self.data[col].std()
                if std_val != 0:
                    normalized_data[col] = (self.data[col] - mean_val) / std_val

        return normalized_data

    def add_time_features(self):
        self.data['year'] = self.data['date'].dt.year
        self.data['month'] = self.data['date'].dt.month
        self.data['day'] = self.data['date'].dt.day
        self.data['dayofweek'] = self.data['date'].dt.dayofweek
        self.data['dayofyear'] = self.data['date'].dt.dayofyear
        self.data['week'] = self.data['date'].dt.isocalendar().week

        return self.data

    def get_preprocessing_log(self):
        return self.preprocessing_log

    def get_processed_data(self):
        return self.data
