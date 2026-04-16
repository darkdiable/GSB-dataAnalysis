import pandas as pd
import numpy as np
from datetime import datetime


class DataPreprocessor:
    
    def __init__(self, df):
        self.df = df.copy()
        self.preprocessing_log = []
        
    def handle_missing_values(self):
        initial_rows = len(self.df)
        missing_count = self.df.isnull().sum()
        
        self.df = self.df.dropna(subset=['user_id', 'behavior_type'])
        
        self.df['category'] = self.df['category'].fillna('Unknown')
        
        final_rows = len(self.df)
        self.preprocessing_log.append({
            'step': '缺失值处理',
            'removed_rows': initial_rows - final_rows,
            'missing_before': missing_count.to_dict()
        })
        
        return self
    
    def handle_outliers(self):
        valid_behaviors = ['pv', 'fav', 'cart', 'buy']
        initial_rows = len(self.df)
        
        invalid_mask = ~self.df['behavior_type'].isin(valid_behaviors)
        invalid_count = invalid_mask.sum()
        
        self.df = self.df[~invalid_mask]
        
        valid_categories = [
            'Electronics', 'Clothing', 'Home', 'Beauty', 'Sports',
            'Books', 'Food', 'Toys', 'Jewelry', 'Automotive', 'Unknown'
        ]
        self.df = self.df[self.df['category'].isin(valid_categories)]
        
        final_rows = len(self.df)
        self.preprocessing_log.append({
            'step': '异常值处理',
            'removed_rows': initial_rows - final_rows,
            'invalid_behaviors': invalid_count
        })
        
        return self
    
    def get_processed_data(self):
        return self.df.reset_index(drop=True)


class RFMFeatureBuilder:
    
    def __init__(self, df, reference_date=None):
        self.df = df.copy()
        self.reference_date = reference_date or (df['timestamp'].max() + pd.Timedelta(days=1))
        self.rfm_df = None
        
    def calculate_rfm(self):
        purchase_df = self.df[self.df['behavior_type'] == 'buy'].copy()
        
        if len(purchase_df) == 0:
            raise ValueError("数据集中没有购买记录，无法计算RFM")
        
        rfm = purchase_df.groupby('user_id').agg({
            'timestamp': lambda x: (self.reference_date - x.max()).days,
            'user_id': 'count',
        }).rename(columns={
            'timestamp': 'recency',
            'user_id': 'frequency'
        })
        
        monetary = purchase_df.groupby('user_id').size() * np.random.uniform(50, 500, len(rfm))
        rfm['monetary'] = monetary
        
        rfm = rfm.reset_index()
        self.rfm_df = rfm
        
        return rfm
    
    def calculate_rfm_scores(self):
        if self.rfm_df is None:
            self.calculate_rfm()
        
        rfm = self.rfm_df.copy()
        
        rfm['r_score'] = pd.qcut(rfm['recency'], q=5, labels=[5, 4, 3, 2, 1], duplicates='drop')
        rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
        rfm['m_score'] = pd.qcut(rfm['monetary'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
        
        for col in ['r_score', 'f_score', 'm_score']:
            rfm[col] = rfm[col].astype(int)
        
        rfm['rfm_score'] = rfm['r_score'] + rfm['f_score'] + rfm['m_score']
        
        def segment_customer(row):
            if row['rfm_score'] >= 12:
                return '高价值客户'
            elif row['rfm_score'] >= 9:
                return '中高价值客户'
            elif row['rfm_score'] >= 6:
                return '中等价值客户'
            else:
                return '低价值客户'
        
        rfm['customer_segment'] = rfm.apply(segment_customer, axis=1)
        
        self.rfm_df = rfm
        return rfm
    
    def get_rfm_data(self):
        return self.rfm_df


class UserBehaviorFeatures:
    
    def __init__(self, df):
        self.df = df.copy()
        self.features_df = None
        
    def build_features(self):
        behavior_pivot = self.df.pivot_table(
            index='user_id',
            columns='behavior_type',
            aggfunc='size',
            fill_value=0
        ).reset_index()
        
        behavior_pivot.columns.name = None
        for col in ['pv', 'fav', 'cart', 'buy']:
            if col not in behavior_pivot.columns:
                behavior_pivot[col] = 0
        
        behavior_pivot['total_actions'] = (
            behavior_pivot['pv'] + behavior_pivot['fav'] + 
            behavior_pivot['cart'] + behavior_pivot['buy']
        )
        
        behavior_pivot['conversion_rate'] = np.where(
            behavior_pivot['pv'] > 0,
            behavior_pivot['buy'] / behavior_pivot['pv'],
            0
        )
        
        behavior_pivot['cart_to_buy_rate'] = np.where(
            behavior_pivot['cart'] > 0,
            behavior_pivot['buy'] / behavior_pivot['cart'],
            0
        )
        
        behavior_pivot['fav_to_buy_rate'] = np.where(
            behavior_pivot['fav'] > 0,
            behavior_pivot['buy'] / behavior_pivot['fav'],
            0
        )
        
        category_diversity = self.df.groupby('user_id')['category'].nunique().reset_index()
        category_diversity.columns = ['user_id', 'category_diversity']
        
        self.features_df = behavior_pivot.merge(category_diversity, on='user_id', how='left')
        
        return self.features_df
    
    def get_features(self):
        return self.features_df


if __name__ == '__main__':
    from data_generator import EcommerceDataGenerator
    
    generator = EcommerceDataGenerator(n_users=1000, n_records=50000)
    df = generator.generate(inject_issues=True)
    
    print("=== 数据预处理 ===")
    preprocessor = DataPreprocessor(df)
    clean_df = preprocessor.handle_missing_values().handle_outliers().get_processed_data()
    print(f"清洗后数据量: {len(clean_df)}")
    
    print("\n=== RFM特征构建 ===")
    rfm_builder = RFMFeatureBuilder(clean_df)
    rfm_df = rfm_builder.calculate_rfm_scores()
    print(rfm_df.head())
    
    print("\n=== 用户行为特征 ===")
    feature_builder = UserBehaviorFeatures(clean_df)
    features_df = feature_builder.build_features()
    print(features_df.head())
