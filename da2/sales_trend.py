"""
销售趋势分析模块
提供多维度销售趋势分析功能
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SalesTrendAnalyzer:
    """销售趋势分析器"""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.results = {}

    def daily_sales_trend(self) -> pd.DataFrame:
        """日销售趋势"""
        try:
            daily = self.df.groupby('date').agg({
                'total_amount': 'sum',
                'quantity': 'sum',
                'transaction_id': 'count'
            }).reset_index()
            daily.columns = ['date', 'revenue', 'quantity', 'transactions']
            daily = daily.sort_values('date')

            # 计算7日移动平均
            daily['revenue_ma7'] = daily['revenue'].rolling(window=7, min_periods=1).mean()
            daily['quantity_ma7'] = daily['quantity'].rolling(window=7, min_periods=1).mean()

            self.results['daily_trend'] = daily
            logger.info("日销售趋势分析完成")
            return daily
        except Exception as e:
            logger.error(f"日销售趋势分析出错: {e}")
            return pd.DataFrame()

    def monthly_sales_trend(self) -> pd.DataFrame:
        """月销售趋势"""
        try:
            monthly = self.df.groupby('year_month').agg({
                'total_amount': 'sum',
                'quantity': 'sum',
                'transaction_id': 'count',
                'customer_id': 'nunique'
            }).reset_index()
            monthly.columns = ['year_month', 'revenue', 'quantity', 'transactions', 'unique_customers']

            # 计算环比增长
            monthly['revenue_growth'] = monthly['revenue'].pct_change() * 100
            monthly['quantity_growth'] = monthly['quantity'].pct_change() * 100

            self.results['monthly_trend'] = monthly
            logger.info("月销售趋势分析完成")
            return monthly
        except Exception as e:
            logger.error(f"月销售趋势分析出错: {e}")
            return pd.DataFrame()

    def quarterly_sales_trend(self) -> pd.DataFrame:
        """季度销售趋势"""
        try:
            self.df['year_quarter'] = self.df['year'].astype(str) + '-Q' + self.df['quarter'].astype(str)

            quarterly = self.df.groupby('year_quarter').agg({
                'total_amount': 'sum',
                'quantity': 'sum',
                'transaction_id': 'count',
                'customer_id': 'nunique'
            }).reset_index()
            quarterly.columns = ['year_quarter', 'revenue', 'quantity', 'transactions', 'unique_customers']

            self.results['quarterly_trend'] = quarterly
            logger.info("季度销售趋势分析完成")
            return quarterly
        except Exception as e:
            logger.error(f"季度销售趋势分析出错: {e}")
            return pd.DataFrame()

    def weekday_sales_analysis(self) -> pd.DataFrame:
        """星期销售分析"""
        try:
            weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']

            weekday = self.df.groupby('weekday').agg({
                'total_amount': ['sum', 'mean'],
                'quantity': ['sum', 'mean'],
                'transaction_id': 'count'
            }).reset_index()

            weekday.columns = ['weekday', 'total_revenue', 'avg_revenue',
                              'total_quantity', 'avg_quantity', 'transactions']
            weekday['weekday_name'] = weekday['weekday'].map(lambda x: weekday_names[x])

            self.results['weekday_analysis'] = weekday
            logger.info("星期销售分析完成")
            return weekday
        except Exception as e:
            logger.error(f"星期销售分析出错: {e}")
            return pd.DataFrame()

    def category_sales_trend(self) -> pd.DataFrame:
        """品类销售趋势"""
        try:
            if 'category' not in self.df.columns:
                logger.warning("缺少category列")
                return pd.DataFrame()

            category_trend = self.df.groupby(['year_month', 'category']).agg({
                'total_amount': 'sum',
                'quantity': 'sum'
            }).reset_index()

            # 透视表
            category_pivot = category_trend.pivot(index='year_month', columns='category', values='total_amount').fillna(0)

            self.results['category_trend'] = category_trend
            self.results['category_pivot'] = category_pivot
            logger.info("品类销售趋势分析完成")
            return category_trend
        except Exception as e:
            logger.error(f"品类销售趋势分析出错: {e}")
            return pd.DataFrame()

    def channel_sales_analysis(self) -> pd.DataFrame:
        """渠道销售分析"""
        try:
            if 'channel' not in self.df.columns:
                logger.warning("缺少channel列")
                return pd.DataFrame()

            channel = self.df.groupby('channel').agg({
                'total_amount': ['sum', 'mean'],
                'quantity': 'sum',
                'transaction_id': 'count',
                'customer_id': 'nunique'
            }).reset_index()

            channel.columns = ['channel', 'total_revenue', 'avg_order_value',
                              'total_quantity', 'transactions', 'unique_customers']

            # 计算占比
            channel['revenue_share'] = channel['total_revenue'] / channel['total_revenue'].sum() * 100

            self.results['channel_analysis'] = channel
            logger.info("渠道销售分析完成")
            return channel
        except Exception as e:
            logger.error(f"渠道销售分析出错: {e}")
            return pd.DataFrame()

    def payment_method_analysis(self) -> pd.DataFrame:
        """支付方式分析"""
        try:
            if 'payment_method' not in self.df.columns:
                logger.warning("缺少payment_method列")
                return pd.DataFrame()

            payment = self.df.groupby('payment_method').agg({
                'total_amount': ['sum', 'mean'],
                'transaction_id': 'count'
            }).reset_index()

            payment.columns = ['payment_method', 'total_revenue', 'avg_order_value', 'transactions']
            payment['share'] = payment['total_revenue'] / payment['total_revenue'].sum() * 100

            self.results['payment_analysis'] = payment
            logger.info("支付方式分析完成")
            return payment
        except Exception as e:
            logger.error(f"支付方式分析出错: {e}")
            return pd.DataFrame()

    def growth_analysis(self, period: str = 'monthly') -> pd.DataFrame:
        """
        增长率分析

        Args:
            period: 周期 ('daily', 'monthly', 'quarterly')
        """
        try:
            if period == 'daily':
                trend = self.daily_sales_trend()
                col = 'revenue'
            elif period == 'monthly':
                trend = self.monthly_sales_trend()
                col = 'revenue'
            elif period == 'quarterly':
                trend = self.quarterly_sales_trend()
                col = 'revenue'
            else:
                return pd.DataFrame()

            if trend.empty:
                return pd.DataFrame()

            # 计算多种增长率指标
            trend['yoy_growth'] = trend[col].pct_change(periods=12 if period == 'monthly' else 4) * 100
            trend['mom_growth'] = trend[col].pct_change() * 100
            trend['cumulative'] = trend[col].cumsum()

            self.results[f'{period}_growth'] = trend
            logger.info(f"{period}增长率分析完成")
            return trend
        except Exception as e:
            logger.error(f"增长率分析出错: {e}")
            return pd.DataFrame()

    def seasonal_analysis(self) -> pd.DataFrame:
        """季节性分析"""
        try:
            seasonal = self.df.groupby('month').agg({
                'total_amount': ['sum', 'mean'],
                'quantity': 'sum',
                'transaction_id': 'count'
            }).reset_index()

            seasonal.columns = ['month', 'total_revenue', 'avg_daily_revenue',
                               'total_quantity', 'transactions']

            # 计算季节性指数
            avg_revenue = seasonal['total_revenue'].mean()
            seasonal['seasonal_index'] = seasonal['total_revenue'] / avg_revenue * 100

            self.results['seasonal_analysis'] = seasonal
            logger.info("季节性分析完成")
            return seasonal
        except Exception as e:
            logger.error(f"季节性分析出错: {e}")
            return pd.DataFrame()

    def top_products_analysis(self, top_n: int = 10) -> pd.DataFrame:
        """热销商品分析"""
        try:
            if 'product_name' not in self.df.columns:
                group_col = 'product_id'
            else:
                group_col = 'product_name'

            products = self.df.groupby(group_col).agg({
                'total_amount': 'sum',
                'quantity': 'sum',
                'transaction_id': 'count'
            }).reset_index()

            products.columns = [group_col, 'revenue', 'quantity', 'transactions']
            products = products.sort_values('revenue', ascending=False).head(top_n)

            self.results['top_products'] = products
            logger.info(f"热销商品分析完成，Top {top_n}")
            return products
        except Exception as e:
            logger.error(f"热销商品分析出错: {e}")
            return pd.DataFrame()

    def calculate_kpis(self) -> Dict:
        """计算关键绩效指标"""
        try:
            kpis = {
                'total_revenue': self.df['total_amount'].sum(),
                'total_transactions': len(self.df),
                'total_quantity': self.df['quantity'].sum(),
                'avg_order_value': self.df['total_amount'].mean(),
                'unique_customers': self.df['customer_id'].nunique(),
                'unique_products': self.df['product_id'].nunique(),
            }

            if 'date' in self.df.columns:
                date_range = (self.df['date'].max() - self.df['date'].min()).days
                kpis['date_range_days'] = date_range
                kpis['daily_avg_revenue'] = kpis['total_revenue'] / max(date_range, 1)

            # 计算客单价
            customer_orders = self.df.groupby('customer_id')['total_amount'].sum()
            kpis['avg_customer_value'] = customer_orders.mean()

            self.results['kpis'] = kpis
            logger.info("KPI计算完成")
            return kpis
        except Exception as e:
            logger.error(f"KPI计算出错: {e}")
            return {}

    def run_all_analysis(self) -> Dict:
        """运行所有分析"""
        logger.info("开始销售趋势分析...")

        self.daily_sales_trend()
        self.monthly_sales_trend()
        self.quarterly_sales_trend()
        self.weekday_sales_analysis()
        self.category_sales_trend()
        self.channel_sales_analysis()
        self.payment_method_analysis()
        self.growth_analysis('monthly')
        self.seasonal_analysis()
        self.top_products_analysis()
        self.calculate_kpis()

        logger.info("销售趋势分析全部完成")
        return self.results

    def get_results(self) -> Dict:
        """获取分析结果"""
        return self.results
