import pandas as pd
import numpy as np
from datetime import datetime


class DataPreprocessor:
    def __init__(self):
        self.df = None
        self.rfm_df = None
    
    def load_data(self, filepath):
        self.df = pd.read_csv(filepath, parse_dates=['timestamp'])
        print(f"数据加载成功: {self.df.shape}")
        return self.df
    
    def handle_missing_values(self):
        print("\n=== 处理缺失值 ===")
        print(f"处理前缺失值统计:\n{self.df.isnull().sum()}")
        
        self.df = self.df.dropna(subset=['user_id', 'behavior_type', 'timestamp'])
        self.df['category'] = self.df['category'].fillna('未知类别')
        
        print(f"处理后缺失值统计:\n{self.df.isnull().sum()}")
        return self.df
    
    def handle_outliers(self):
        print("\n=== 处理异常值 ===")
        user_counts = self.df['user_id'].value_counts()
        valid_users = user_counts[user_counts < user_counts.quantile(0.999)].index
        self.df = self.df[self.df['user_id'].isin(valid_users)]
        
        self.df = self.df[self.df['user_id'] != 999999999]
        
        print(f"异常值处理后数据量: {self.df.shape}")
        return self.df
    
    def calculate_rfm(self):
        print("\n=== 构建RFM特征 ===")
        snapshot_date = self.df['timestamp'].max() + pd.Timedelta(days=1)
        
        user_behavior = self.df.copy()
        user_behavior['date'] = user_behavior['timestamp'].dt.date
        
        recency = (snapshot_date - user_behavior.groupby('user_id')['timestamp'].max()).dt.days
        frequency = user_behavior[user_behavior['behavior_type'] == 'buy'].groupby('user_id').size()
        monetary = user_behavior[user_behavior['behavior_type'] == 'buy'].groupby('user_id').size() * 50
        
        self.rfm_df = pd.DataFrame({
            'Recency': recency,
            'Frequency': frequency,
            'Monetary': monetary
        }).fillna(0)
        
        self.rfm_df['R_Score'] = pd.qcut(self.rfm_df['Recency'], 5, labels=[5, 4, 3, 2, 1]).astype(int)
        self.rfm_df['F_Score'] = pd.qcut(self.rfm_df['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
        self.rfm_df['M_Score'] = pd.qcut(self.rfm_df['Monetary'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
        
        self.rfm_df['RFM_Score'] = self.rfm_df['R_Score'] + self.rfm_df['F_Score'] + self.rfm_df['M_Score']
        
        def segment_user(row):
            if row['R_Score'] >= 4 and row['F_Score'] >= 4:
                return '重要价值用户'
            elif row['R_Score'] >= 4 and row['F_Score'] <= 2:
                return '新用户'
            elif row['R_Score'] <= 2 and row['F_Score'] >= 4:
                return '重要召回用户'
            elif row['R_Score'] <= 2 and row['F_Score'] <= 2:
                return '流失用户'
            else:
                return '一般用户'
        
        self.rfm_df['User_Segment'] = self.rfm_df.apply(segment_user, axis=1)
        
        print("RFM特征统计:")
        print(self.rfm_df['User_Segment'].value_counts())
        return self.rfm_df
    
    def create_repurchase_label(self):
        print("\n=== 创建复购标签 ===")
        purchase_counts = self.df[self.df['behavior_type'] == 'buy'].groupby('user_id').size()
        self.rfm_df['Repurchase'] = (purchase_counts >= 2).astype(int)
        self.rfm_df['Repurchase'] = self.rfm_df['Repurchase'].fillna(0)
        
        print(f"复购用户占比: {self.rfm_df['Repurchase'].mean():.2%}")
        return self.rfm_df
    
    def add_behavior_features(self):
        behavior_pivot = self.df.groupby(['user_id', 'behavior_type']).size().unstack(fill_value=0)
        behavior_pivot.columns = [f'count_{col}' for col in behavior_pivot.columns]
        
        self.rfm_df = self.rfm_df.join(behavior_pivot, how='left').fillna(0)
        
        self.rfm_df['cart_conversion'] = np.where(
            self.rfm_df['count_cart'] > 0,
            self.rfm_df['count_buy'] / (self.rfm_df['count_cart'] + 1),
            0
        )
        self.rfm_df['pv_cart_rate'] = np.where(
            self.rfm_df['count_pv'] > 0,
            self.rfm_df['count_cart'] / self.rfm_df['count_pv'],
            0
        )
        
        return self.rfm_df
    
    def preprocess(self, filepath):
        self.load_data(filepath)
        self.handle_missing_values()
        self.handle_outliers()
        self.calculate_rfm()
        self.create_repurchase_label()
        self.add_behavior_features()
        
        print(f"\n预处理完成，最终特征数: {self.rfm_df.shape}")
        return self.df, self.rfm_df


if __name__ == '__main__':
    preprocessor = DataPreprocessor()
    df, rfm_df = preprocessor.preprocess('../data/ecommerce_behavior.csv')
    print(rfm_df.head())
