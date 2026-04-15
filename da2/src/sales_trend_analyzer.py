import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional


class SalesTrendAnalyzer:
    def __init__(self, orders_df: pd.DataFrame):
        self.orders_df = orders_df.copy()
        self.orders_df['order_date'] = pd.to_datetime(self.orders_df['order_date'])
        self.completed_orders = self.orders_df[self.orders_df['order_status'] == '已完成']
        
    def analyze_daily_trend(self) -> pd.DataFrame:
        daily_trend = self.completed_orders.groupby(
            self.completed_orders['order_date'].dt.date
        ).agg({
            'total_amount': 'sum',
            'order_id': 'count',
            'customer_id': 'nunique',
            'quantity': 'sum'
        }).reset_index()
        
        daily_trend.columns = ['date', 'revenue', 'orders', 'customers', 'quantity']
        daily_trend['date'] = pd.to_datetime(daily_trend['date'])
        daily_trend = daily_trend.sort_values('date')
        
        daily_trend['revenue_ma7'] = daily_trend['revenue'].rolling(window=7, min_periods=1).mean()
        daily_trend['revenue_ma30'] = daily_trend['revenue'].rolling(window=30, min_periods=1).mean()
        
        return daily_trend
    
    def analyze_monthly_trend(self) -> pd.DataFrame:
        completed = self.completed_orders.copy()
        completed['year'] = completed['order_date'].dt.year
        completed['month'] = completed['order_date'].dt.month
        
        monthly_trend = completed.groupby(['year', 'month']).agg({
            'total_amount': 'sum',
            'order_id': 'count',
            'customer_id': 'nunique',
            'quantity': 'sum'
        }).reset_index()
        
        monthly_trend.columns = ['year', 'month', 'revenue', 'orders', 'customers', 'quantity']
        monthly_trend['year_month'] = monthly_trend['year'].astype(str) + '-' + monthly_trend['month'].astype(str).str.zfill(2)
        
        monthly_trend['mom_growth'] = monthly_trend['revenue'].pct_change() * 100
        
        return monthly_trend
    
    def analyze_weekly_pattern(self) -> pd.DataFrame:
        completed = self.completed_orders.copy()
        completed['day_of_week'] = completed['order_date'].dt.dayofweek
        
        weekly_pattern = completed.groupby('day_of_week').agg({
            'total_amount': ['sum', 'mean'],
            'order_id': 'count'
        }).reset_index()
        
        weekly_pattern.columns = ['day_of_week', 'total_revenue', 'avg_revenue', 'order_count']
        
        day_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
        weekly_pattern['day_name'] = weekly_pattern['day_of_week'].map(lambda x: day_names[x])
        
        return weekly_pattern
    
    def analyze_hourly_pattern(self) -> pd.DataFrame:
        completed = self.completed_orders.copy()
        completed['hour'] = completed['order_date'].dt.hour
        
        hourly_pattern = completed.groupby('hour').agg({
            'total_amount': ['sum', 'mean'],
            'order_id': 'count'
        }).reset_index()
        
        hourly_pattern.columns = ['hour', 'total_revenue', 'avg_revenue', 'order_count']
        
        return hourly_pattern
    
    def analyze_category_trend(self) -> pd.DataFrame:
        completed = self.completed_orders.copy()
        completed['year_month'] = completed['order_date'].dt.to_period('M')
        
        category_monthly = completed.groupby([
            'category',
            'year_month'
        ]).agg({
            'total_amount': 'sum',
            'order_id': 'count'
        }).reset_index()
        
        category_monthly.columns = ['category', 'year_month', 'revenue', 'orders']
        category_monthly['year_month'] = category_monthly['year_month'].astype(str)
        
        return category_monthly
    
    def calculate_growth_metrics(self) -> Dict:
        monthly_trend = self.analyze_monthly_trend()
        
        if len(monthly_trend) >= 2:
            last_month = monthly_trend.iloc[-1]
            prev_month = monthly_trend.iloc[-2]
            
            revenue_growth = ((last_month['revenue'] - prev_month['revenue']) / prev_month['revenue']) * 100
            order_growth = ((last_month['orders'] - prev_month['orders']) / prev_month['orders']) * 100
            customer_growth = ((last_month['customers'] - prev_month['customers']) / prev_month['customers']) * 100
        else:
            revenue_growth = order_growth = customer_growth = 0
        
        daily_trend = self.analyze_daily_trend()
        
        if len(daily_trend) >= 7:
            last_7d_revenue = daily_trend.tail(7)['revenue'].sum()
            prev_7d_revenue = daily_trend.tail(14).head(7)['revenue'].sum()
            weekly_growth = ((last_7d_revenue - prev_7d_revenue) / prev_7d_revenue) * 100 if prev_7d_revenue > 0 else 0
        else:
            weekly_growth = 0
        
        return {
            'monthly_revenue_growth': round(revenue_growth, 2),
            'monthly_order_growth': round(order_growth, 2),
            'monthly_customer_growth': round(customer_growth, 2),
            'weekly_revenue_growth': round(weekly_growth, 2)
        }
    
    def identify_peak_periods(self) -> Dict:
        daily_trend = self.analyze_daily_trend()
        
        peak_day = daily_trend.loc[daily_trend['revenue'].idxmax()]
        
        monthly_trend = self.analyze_monthly_trend()
        peak_month = monthly_trend.loc[monthly_trend['revenue'].idxmax()]
        
        weekly_pattern = self.analyze_weekly_pattern()
        peak_weekday = weekly_pattern.loc[weekly_pattern['total_revenue'].idxmax()]
        
        return {
            'peak_day': {
                'date': peak_day['date'].strftime('%Y-%m-%d'),
                'revenue': round(peak_day['revenue'], 2)
            },
            'peak_month': {
                'month': peak_month['year_month'],
                'revenue': round(peak_month['revenue'], 2)
            },
            'peak_weekday': {
                'day': peak_weekday['day_name'],
                'avg_revenue': round(peak_weekday['avg_revenue'], 2)
            }
        }
    
    def get_sales_summary(self) -> Dict:
        total_revenue = self.completed_orders['total_amount'].sum()
        total_orders = len(self.completed_orders)
        total_customers = self.completed_orders['customer_id'].nunique()
        avg_order_value = total_revenue / total_orders if total_orders > 0 else 0
        
        return {
            'total_revenue': round(total_revenue, 2),
            'total_orders': total_orders,
            'total_customers': total_customers,
            'avg_order_value': round(avg_order_value, 2),
            'growth_metrics': self.calculate_growth_metrics(),
            'peak_periods': self.identify_peak_periods()
        }
    
    def get_all_analysis(self) -> Dict:
        return {
            'daily_trend': self.analyze_daily_trend(),
            'monthly_trend': self.analyze_monthly_trend(),
            'weekly_pattern': self.analyze_weekly_pattern(),
            'hourly_pattern': self.analyze_hourly_pattern(),
            'category_trend': self.analyze_category_trend(),
            'summary': self.get_sales_summary()
        }
