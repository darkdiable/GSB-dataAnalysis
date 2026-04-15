import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


class CustomerValueAnalyzer:
    def __init__(self, orders_df: pd.DataFrame, customers_df: pd.DataFrame):
        self.orders_df = orders_df.copy()
        self.customers_df = customers_df.copy()
        self.orders_df['order_date'] = pd.to_datetime(self.orders_df['order_date'])
        self.completed_orders = self.orders_df[self.orders_df['order_status'] == '已完成']
        
    def calculate_rfm(self) -> pd.DataFrame:
        analysis_date = self.completed_orders['order_date'].max() + pd.Timedelta(days=1)
        
        rfm = self.completed_orders.groupby('customer_id').agg({
            'order_date': lambda x: (analysis_date - x.max()).days,
            'order_id': 'count',
            'total_amount': 'sum'
        })
        
        rfm.columns = ['recency', 'frequency', 'monetary']
        
        rfm['r_score'] = pd.qcut(rfm['recency'], q=5, labels=[5, 4, 3, 2, 1], duplicates='drop')
        rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
        rfm['m_score'] = pd.qcut(rfm['monetary'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
        
        rfm['r_score'] = rfm['r_score'].astype(int)
        rfm['f_score'] = rfm['f_score'].astype(int)
        rfm['m_score'] = rfm['m_score'].astype(int)
        
        rfm['rfm_score'] = rfm['r_score'] * 100 + rfm['f_score'] * 10 + rfm['m_score']
        
        return rfm.reset_index()
    
    def segment_customers_rfm(self, rfm_df: pd.DataFrame) -> pd.DataFrame:
        def segment_customer(row):
            r, f, m = row['r_score'], row['f_score'], row['m_score']
            
            if r >= 4 and f >= 4 and m >= 4:
                return '重要价值客户'
            elif r >= 4 and f >= 3:
                return '重要发展客户'
            elif f >= 4 and m >= 3:
                return '重要保持客户'
            elif r <= 2 and f >= 3:
                return '重要挽留客户'
            elif r >= 4 and f <= 2:
                return '新客户'
            elif r <= 2 and f <= 2 and m <= 2:
                return '低价值客户'
            else:
                return '一般客户'
        
        rfm_df['segment'] = rfm_df.apply(segment_customer, axis=1)
        return rfm_df
    
    def cluster_customers(self, n_clusters: int = 5) -> pd.DataFrame:
        customer_features = self.completed_orders.groupby('customer_id').agg({
            'total_amount': ['sum', 'mean', 'std'],
            'order_id': 'count',
            'quantity': 'sum',
            'product_id': 'nunique'
        })
        
        customer_features.columns = [
            'total_spent', 'avg_order_value', 'std_order_value',
            'order_count', 'total_quantity', 'unique_products'
        ]
        customer_features['std_order_value'] = customer_features['std_order_value'].fillna(0)
        
        features_for_clustering = customer_features[['total_spent', 'order_count', 'avg_order_value', 'unique_products']]
        
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features_for_clustering)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        customer_features['cluster'] = kmeans.fit_predict(scaled_features)
        
        return customer_features.reset_index()
    
    def analyze_customer_lifetime_value(self) -> pd.DataFrame:
        customer_orders = self.completed_orders.groupby('customer_id').agg({
            'order_date': ['min', 'max', 'count'],
            'total_amount': ['sum', 'mean']
        })
        
        customer_orders.columns = [
            'first_order', 'last_order', 'order_count',
            'total_revenue', 'avg_order_value'
        ]
        
        customer_orders['customer_lifetime'] = (
            customer_orders['last_order'] - customer_orders['first_order']
        ).dt.days
        
        customer_orders['purchase_frequency'] = customer_orders['order_count'] / (
            customer_orders['customer_lifetime'] + 1
        )
        
        customer_orders['clv'] = (
            customer_orders['avg_order_value'] * 
            customer_orders['purchase_frequency'] * 
            365
        )
        
        return customer_orders.reset_index()
    
    def analyze_customer_behavior(self) -> pd.DataFrame:
        behavior = self.completed_orders.groupby('customer_id').agg({
            'category': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else '未知',
            'payment_method': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else '未知',
            'total_amount': ['sum', 'mean', 'max', 'min'],
            'quantity': ['sum', 'mean'],
            'order_id': 'count',
            'product_id': 'nunique'
        })
        
        behavior.columns = [
            'favorite_category', 'preferred_payment',
            'total_spent', 'avg_order_value', 'max_order_value', 'min_order_value',
            'total_quantity', 'avg_quantity',
            'order_count', 'unique_products'
        ]
        
        return behavior.reset_index()
    
    def get_segment_summary(self, rfm_df: pd.DataFrame) -> pd.DataFrame:
        segment_summary = rfm_df.groupby('segment').agg({
            'customer_id': 'count',
            'recency': 'mean',
            'frequency': 'mean',
            'monetary': ['sum', 'mean']
        })
        
        segment_summary.columns = [
            'customer_count', 'avg_recency', 'avg_frequency',
            'total_monetary', 'avg_monetary'
        ]
        
        segment_summary['revenue_share'] = (
            segment_summary['total_monetary'] / segment_summary['total_monetary'].sum() * 100
        )
        
        segment_summary['customer_share'] = (
            segment_summary['customer_count'] / segment_summary['customer_count'].sum() * 100
        )
        
        return segment_summary.reset_index()
    
    def get_top_customers(self, n: int = 20) -> pd.DataFrame:
        top_customers = self.completed_orders.groupby('customer_id').agg({
            'total_amount': 'sum',
            'order_id': 'count',
            'product_id': 'nunique'
        })
        
        top_customers.columns = ['total_spent', 'order_count', 'unique_products']
        top_customers = top_customers.sort_values('total_spent', ascending=False).head(n)
        
        top_customers = top_customers.merge(
            self.customers_df[['customer_id', 'customer_name', 'city', 'membership_level']],
            on='customer_id',
            how='left'
        )
        
        return top_customers.reset_index(drop=True)
    
    def get_customer_distribution(self) -> Dict:
        rfm = self.calculate_rfm()
        rfm_segmented = self.segment_customers_rfm(rfm)
        
        segment_dist = rfm_segmented['segment'].value_counts().to_dict()
        
        city_dist = self.customers_df['city'].value_counts().to_dict()
        
        membership_dist = self.customers_df['membership_level'].value_counts().to_dict()
        
        return {
            'segment_distribution': segment_dist,
            'city_distribution': city_dist,
            'membership_distribution': membership_dist
        }
    
    def get_all_analysis(self) -> Dict:
        rfm = self.calculate_rfm()
        rfm_segmented = self.segment_customers_rfm(rfm)
        
        return {
            'rfm_analysis': rfm_segmented,
            'segment_summary': self.get_segment_summary(rfm_segmented),
            'customer_clusters': self.cluster_customers(),
            'clv_analysis': self.analyze_customer_lifetime_value(),
            'behavior_analysis': self.analyze_customer_behavior(),
            'top_customers': self.get_top_customers(),
            'distribution': self.get_customer_distribution()
        }
