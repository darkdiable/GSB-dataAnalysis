import pandas as pd
import numpy as np
from scipy import signal


class FeatureEngineer:
    def __init__(self, data, date_column, sales_column):
        self.data = data.copy()
        self.date_column = date_column
        self.sales_column = sales_column
        self.features = pd.DataFrame()

    def extract_time_features(self):
        self.data[self.date_column] = pd.to_datetime(self.data[self.date_column])
        
        self.features['year'] = self.data[self.date_column].dt.year
        self.features['month'] = self.data[self.date_column].dt.month
        self.features['day'] = self.data[self.date_column].dt.day
        self.features['dayofweek'] = self.data[self.date_column].dt.dayofweek
        self.features['dayofyear'] = self.data[self.date_column].dt.dayofyear
        self.features['weekofyear'] = self.data[self.date_column].dt.isocalendar().week.astype(int)
        self.features['quarter'] = self.data[self.date_column].dt.quarter
        self.features['is_weekend'] = (self.data[self.date_column].dt.dayofweek >= 5).astype(int)
        self.features['is_month_start'] = self.data[self.date_column].dt.is_month_start.astype(int)
        self.features['is_month_end'] = self.data[self.date_column].dt.is_month_end.astype(int)
        
        print("时间特征提取完成")
        return self.features

    def extract_seasonal_features(self):
        self.features['season'] = self.features['month'].apply(self._get_season)
        
        self.features['is_holiday_season'] = self.features['month'].apply(
            lambda x: 1 if x in [11, 12, 1] else 0
        )
        
        print("季节性特征提取完成")
        return self.features

    def _get_season(self, month):
        if month in [3, 4, 5]:
            return 1
        elif month in [6, 7, 8]:
            return 2
        elif month in [9, 10, 11]:
            return 3
        else:
            return 4

    def extract_lag_features(self, lags=[1, 7, 14, 30]):
        for lag in lags:
            self.features[f'sales_lag_{lag}'] = self.data[self.sales_column].shift(lag)
        
        print(f"滞后特征提取完成 (滞后期数: {lags})")
        return self.features

    def extract_rolling_features(self, windows=[7, 14, 30]):
        for window in windows:
            self.features[f'sales_rolling_mean_{window}'] = self.data[self.sales_column].rolling(window=window).mean()
            self.features[f'sales_rolling_std_{window}'] = self.data[self.sales_column].rolling(window=window).std()
            self.features[f'sales_rolling_min_{window}'] = self.data[self.sales_column].rolling(window=window).min()
            self.features[f'sales_rolling_max_{window}'] = self.data[self.sales_column].rolling(window=window).max()
        
        print(f"滚动统计特征提取完成 (窗口大小: {windows})")
        return self.features

    def decompose_time_series(self, period=365):
        sales = self.data[self.sales_column].values
        
        if len(sales) < period * 2:
            period = len(sales) // 2
            print(f"数据长度不足，调整分解周期为 {period}")
        
        try:
            from scipy.signal import savgol_filter
            
            trend = savgol_filter(sales, min(51, len(sales) // 2 * 2 + 1), 3)
            
            detrended = sales - trend
            seasonal = np.zeros_like(sales)
            
            for i in range(period):
                seasonal[i::period] = np.mean(detrended[i::period]) if len(detrended[i::period]) > 0 else 0
            
            residual = sales - trend - seasonal
            
            self.features['trend'] = trend
            self.features['seasonal'] = seasonal
            self.features['residual'] = residual
            
            print("时间序列分解完成（趋势项、季节项、残差项）")
            
        except Exception as e:
            print(f"时间序列分解失败: {e}")
            self.features['trend'] = np.nan
            self.features['seasonal'] = np.nan
            self.features['residual'] = np.nan
        
        return self.features

    def create_features(self):
        self.extract_time_features()
        self.extract_seasonal_features()
        self.extract_lag_features()
        self.extract_rolling_features()
        self.decompose_time_series()
        
        self.features[self.sales_column] = self.data[self.sales_column].values
        self.features[self.date_column] = self.data[self.date_column].values
        
        return self.features

    def get_feature_importance_names(self):
        feature_names = [col for col in self.features.columns 
                        if col not in [self.date_column, self.sales_column]]
        return feature_names
