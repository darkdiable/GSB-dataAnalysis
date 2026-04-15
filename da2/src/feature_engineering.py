import pandas as pd
import numpy as np
from typing import Optional, Dict


class FeatureEngineer:
    def __init__(self, orders_df: pd.DataFrame, customers_df: pd.DataFrame, products_df: pd.DataFrame):
        self.orders_df = orders_df.copy()
        self.customers_df = customers_df.copy()
        self.products_df = products_df.copy()
        
    def create_time_features(self, df: pd.DataFrame, date_column: str = 'order_date') -> pd.DataFrame:
        df = df.copy()
        df['year'] = df[date_column].dt.year
        df['month'] = df[date_column].dt.month
        df['day'] = df[date_column].dt.day
        df['day_of_week'] = df[date_column].dt.dayofweek
        df['day_name'] = df[date_column].dt.day_name()
        df['week_of_year'] = df[date_column].dt.isocalendar().week
        df['quarter'] = df[date_column].dt.quarter
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['year_month'] = df[date_column].dt.to_period('M')
        return df
    
    def create_customer_features(self) -> pd.DataFrame:
        completed_orders = self.orders_df[self.orders_df['order_status'] == '已完成']
        
        customer_features = completed_orders.groupby('customer_id').agg({
            'order_id': 'count',
            'total_amount': ['sum', 'mean', 'std'],
            'quantity': ['sum', 'mean'],
            'order_date': ['min', 'max'],
            'product_id': 'nunique',
            'category': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else '未知'
        })
        
        customer_features.columns = [
            'order_count', 'total_spent', 'avg_order_value', 'std_order_value',
            'total_quantity', 'avg_quantity', 'first_order_date', 'last_order_date',
            'unique_products', 'favorite_category'
        ]
        
        customer_features['days_as_customer'] = (
            customer_features['last_order_date'] - customer_features['first_order_date']
        ).dt.days
        
        customer_features['avg_days_between_orders'] = (
            customer_features['days_as_customer'] / customer_features['order_count']
        ).fillna(0)
        
        customer_features['std_order_value'] = customer_features['std_order_value'].fillna(0)
        
        customer_features = customer_features.reset_index()
        
        return customer_features
    
    def create_product_features(self) -> pd.DataFrame:
        completed_orders = self.orders_df[self.orders_df['order_status'] == '已完成']
        
        product_features = completed_orders.groupby('product_id').agg({
            'order_id': 'count',
            'quantity': ['sum', 'mean'],
            'total_amount': ['sum', 'mean'],
            'customer_id': 'nunique',
            'order_date': ['min', 'max']
        })
        
        product_features.columns = [
            'order_count', 'total_quantity_sold', 'avg_quantity_per_order',
            'total_revenue', 'avg_revenue_per_order', 'unique_customers',
            'first_sale_date', 'last_sale_date'
        ]
        
        product_features['days_on_sale'] = (
            product_features['last_sale_date'] - product_features['first_sale_date']
        ).dt.days
        
        product_features['daily_sales_rate'] = (
            product_features['total_quantity_sold'] / (product_features['days_on_sale'] + 1)
        )
        
        product_features = product_features.reset_index()
        
        return product_features
    
    def create_rfm_features(self) -> pd.DataFrame:
        completed_orders = self.orders_df[self.orders_df['order_status'] == '已完成']
        
        analysis_date = completed_orders['order_date'].max() + pd.Timedelta(days=1)
        
        rfm = completed_orders.groupby('customer_id').agg({
            'order_date': lambda x: (analysis_date - x.max()).days,
            'order_id': 'count',
            'total_amount': 'sum'
        })
        
        rfm.columns = ['recency', 'frequency', 'monetary']
        
        rfm['r_score'] = pd.qcut(rfm['recency'], q=5, labels=[5, 4, 3, 2, 1], duplicates='drop')
        rfm['f_score'] = pd.qcut(rfm['frequency'], q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
        rfm['m_score'] = pd.qcut(rfm['monetary'], q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
        
        rfm['r_score'] = rfm['r_score'].astype(int)
        rfm['f_score'] = rfm['f_score'].astype(int)
        rfm['m_score'] = rfm['m_score'].astype(int)
        
        rfm['rfm_score'] = rfm['r_score'] * 100 + rfm['f_score'] * 10 + rfm['m_score']
        
        def segment_customer(row):
            if row['rfm_score'] >= 444:
                return '重要价值客户'
            elif row['r_score'] >= 4 and row['f_score'] >= 3:
                return '重要发展客户'
            elif row['f_score'] >= 4 and row['m_score'] >= 3:
                return '重要保持客户'
            elif row['r_score'] <= 2 and row['f_score'] >= 3:
                return '重要挽留客户'
            elif row['rfm_score'] <= 222:
                return '低价值客户'
            else:
                return '一般客户'
        
        rfm['customer_segment'] = rfm.apply(segment_customer, axis=1)
        
        rfm = rfm.reset_index()
        
        return rfm
    
    def create_sales_features(self) -> pd.DataFrame:
        orders = self.create_time_features(self.orders_df, 'order_date')
        
        daily_sales = orders.groupby(['year', 'month', 'day']).agg({
            'total_amount': 'sum',
            'order_id': 'count',
            'quantity': 'sum',
            'customer_id': 'nunique'
        }).reset_index()
        
        daily_sales.columns = ['year', 'month', 'day', 'daily_revenue', 'daily_orders', 'daily_quantity', 'daily_customers']
        
        daily_sales['date'] = pd.to_datetime(daily_sales[['year', 'month', 'day']])
        daily_sales = daily_sales.sort_values('date')
        
        daily_sales['revenue_7d_ma'] = daily_sales['daily_revenue'].rolling(window=7, min_periods=1).mean()
        daily_sales['revenue_30d_ma'] = daily_sales['daily_revenue'].rolling(window=30, min_periods=1).mean()
        
        daily_sales['revenue_lag_1'] = daily_sales['daily_revenue'].shift(1)
        daily_sales['revenue_lag_7'] = daily_sales['daily_revenue'].shift(7)
        
        return daily_sales
    
    def create_category_features(self) -> pd.DataFrame:
        completed_orders = self.orders_df[self.orders_df['order_status'] == '已完成']
        
        category_features = completed_orders.groupby('category').agg({
            'order_id': 'count',
            'quantity': 'sum',
            'total_amount': ['sum', 'mean'],
            'customer_id': 'nunique'
        })
        
        category_features.columns = [
            'order_count', 'total_quantity', 'total_revenue', 
            'avg_order_value', 'unique_customers'
        ]
        
        category_features['revenue_share'] = (
            category_features['total_revenue'] / category_features['total_revenue'].sum() * 100
        )
        
        category_features = category_features.reset_index()
        
        return category_features
    
    def get_all_features(self) -> Dict[str, pd.DataFrame]:
        return {
            'customer_features': self.create_customer_features(),
            'product_features': self.create_product_features(),
            'rfm_features': self.create_rfm_features(),
            'sales_features': self.create_sales_features(),
            'category_features': self.create_category_features()
        }
