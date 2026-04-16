"""
特征工程模块 - 提取时间序列特征（季节性、趋势项等）
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class FeatureEngineer:
    """特征工程类，提取时间序列相关特征"""
    
    def __init__(self):
        """初始化特征工程器"""
        self.feature_names = []
    
    def extract_features(self, df):
        """
        提取所有特征
        
        Parameters:
            df: 清洗后的数据DataFrame
            
        Returns:
            DataFrame: 包含特征的数据框
        """
        df_features = df.copy()
        
        # 1. 时间基础特征
        df_features = self._extract_datetime_features(df_features)
        
        # 2. 滞后特征（历史销售额）
        df_features = self._extract_lag_features(df_features)
        
        # 3. 滚动统计特征
        df_features = self._extract_rolling_features(df_features)
        
        # 4. 季节性特征
        df_features = self._extract_seasonal_features(df_features)
        
        # 5. 趋势特征
        df_features = self._extract_trend_features(df_features)
        
        # 删除因滞后特征产生的缺失值
        df_features = df_features.dropna()
        
        # 记录特征名
        self.feature_names = [col for col in df_features.columns 
                              if col not in ['date', 'sales']]
        
        print(f"特征工程完成: 提取了 {len(self.feature_names)} 个特征")
        return df_features
    
    def _extract_datetime_features(self, df):
        """提取日期时间基础特征"""
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['dayofweek'] = df['date'].dt.dayofweek
        df['dayofyear'] = df['date'].dt.dayofyear
        df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
        df['quarter'] = df['date'].dt.quarter
        
        # 是否周末
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        
        # 是否月初/月末
        df['is_month_start'] = (df['day'] <= 5).astype(int)
        df['is_month_end'] = (df['day'] >= 25).astype(int)
        
        return df
    
    def _extract_lag_features(self, df, lags=[1, 7, 14, 30]):
        """
        提取滞后特征（过去N天的销售额）
        
        Parameters:
            lags: 滞后天数列表
        """
        for lag in lags:
            df[f'sales_lag_{lag}'] = df['sales'].shift(lag)
        return df
    
    def _extract_rolling_features(self, df, windows=[7, 14, 30]):
        """
        提取滚动统计特征
        
        Parameters:
            windows: 滚动窗口大小列表
        """
        for window in windows:
            # 滚动平均
            df[f'sales_roll_mean_{window}'] = df['sales'].shift(1).rolling(window=window).mean()
            # 滚动标准差
            df[f'sales_roll_std_{window}'] = df['sales'].shift(1).rolling(window=window).std()
            # 滚动最大值
            df[f'sales_roll_max_{window}'] = df['sales'].shift(1).rolling(window=window).max()
            # 滚动最小值
            df[f'sales_roll_min_{window}'] = df['sales'].shift(1).rolling(window=window).min()
        return df
    
    def _extract_seasonal_features(self, df):
        """提取季节性特征（正弦/余弦变换）"""
        # 周季节性
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        
        # 月季节性
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # 年季节性（日周期）
        df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
        df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)
        
        return df
    
    def _extract_trend_features(self, df):
        """提取趋势特征"""
        # 距离起始日期的天数（作为趋势项）
        min_date = df['date'].min()
        df['days_since_start'] = (df['date'] - min_date).dt.days
        
        # 时间序列序号
        df['time_idx'] = range(len(df))
        
        return df
    
    def prepare_future_features(self, last_date, days=30, historical_df=None):
        """
        为未来日期准备特征
        
        Parameters:
            last_date: 最后一条数据的日期
            days: 预测天数
            historical_df: 历史数据，用于计算滞后特征
            
        Returns:
            DataFrame: 未来日期的特征数据框
        """
        # 生成未来日期
        future_dates = [last_date + timedelta(days=i+1) for i in range(days)]
        df_future = pd.DataFrame({'date': future_dates})
        
        # 提取基础时间特征
        df_future = self._extract_datetime_features(df_future)
        
        # 提取季节性特征
        df_future = self._extract_seasonal_features(df_future)
        
        # 提取趋势特征
        if historical_df is not None:
            min_date = historical_df['date'].min()
            df_future['days_since_start'] = (df_future['date'] - min_date).dt.days
            last_idx = len(historical_df)
            df_future['time_idx'] = range(last_idx, last_idx + days)
        
        return df_future
    
    def get_feature_names(self):
        """获取特征名称列表"""
        return self.feature_names
