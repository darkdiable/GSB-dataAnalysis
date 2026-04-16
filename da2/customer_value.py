"""
客户价值分析模块
提供客户分群、RFM分析、客户生命周期价值分析等功能
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CustomerValueAnalyzer:
    """客户价值分析器"""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.results = {}

    def rfm_analysis(self, reference_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        RFM分析（Recency, Frequency, Monetary）

        Args:
            reference_date: 参考日期，默认为数据中的最大日期
        """
        try:
            if reference_date is None:
                reference_date = self.df['date'].max()

            # 计算RFM指标
            rfm = self.df.groupby('customer_id').agg({
                'date': lambda x: (reference_date - x.max()).days,  # Recency: 最近一次购买距今天数
                'transaction_id': 'count',  # Frequency: 购买频次
                'total_amount': 'sum'  # Monetary: 总消费金额
            }).reset_index()

            rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']

            # 使用五分位数进行评分（1-5分）
            rfm['r_score'] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
            rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
            rfm['m_score'] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

            # 转换为数值
            rfm['r_score'] = rfm['r_score'].astype(int)
            rfm['f_score'] = rfm['f_score'].astype(int)
            rfm['m_score'] = rfm['m_score'].astype(int)

            # RFM总分
            rfm['rfm_score'] = rfm['r_score'] + rfm['f_score'] + rfm['m_score']

            # RFM组合标签
            rfm['rfm_segment'] = (rfm['r_score'].astype(str) +
                                  rfm['f_score'].astype(str) +
                                  rfm['m_score'].astype(str))

            # 客户价值分层
            def segment_customer(row):
                if row['r_score'] >= 4 and row['f_score'] >= 4 and row['m_score'] >= 4:
                    return '重要价值客户'
                elif row['r_score'] >= 4 and row['f_score'] >= 3:
                    return '重要保持客户'
                elif row['r_score'] >= 4 and row['m_score'] >= 3:
                    return '重要发展客户'
                elif row['r_score'] >= 4:
                    return '新客户'
                elif row['f_score'] >= 4 and row['m_score'] >= 4:
                    return '重要挽留客户'
                elif row['f_score'] >= 4:
                    return '一般保持客户'
                elif row['m_score'] >= 4:
                    return '潜力客户'
                elif row['r_score'] >= 3 and row['f_score'] >= 3:
                    return '一般价值客户'
                else:
                    return '低价值客户'

            rfm['customer_segment'] = rfm.apply(segment_customer, axis=1)

            self.results['rfm'] = rfm
            logger.info("RFM分析完成")
            return rfm

        except Exception as e:
            logger.error(f"RFM分析出错: {e}")
            return pd.DataFrame()

    def customer_lifetime_value(self) -> pd.DataFrame:
        """客户生命周期价值（CLV）分析"""
        try:
            # 计算每个客户的购买行为
            clv = self.df.groupby('customer_id').agg({
                'date': ['min', 'max', 'count'],
                'total_amount': ['sum', 'mean'],
                'transaction_id': 'count'
            }).reset_index()

            clv.columns = ['customer_id', 'first_purchase', 'last_purchase',
                          'purchase_days', 'total_revenue', 'avg_order_value', 'transaction_count']

            # 计算客户生命周期（天）
            clv['lifetime_days'] = (clv['last_purchase'] - clv['first_purchase']).dt.days

            # 计算购买频率（每年购买次数）
            clv['purchase_frequency'] = clv['transaction_count'] / (clv['lifetime_days'] / 365 + 1)

            # 预测CLV（简化模型：平均订单价值 * 购买频率 * 2年）
            clv['predicted_clv'] = clv['avg_order_value'] * clv['purchase_frequency'] * 2

            # 计算历史CLV
            clv['historical_clv'] = clv['total_revenue']

            self.results['clv'] = clv
            logger.info("客户生命周期价值分析完成")
            return clv

        except Exception as e:
            logger.error(f"CLV分析出错: {e}")
            return pd.DataFrame()

    def customer_cohort_analysis(self) -> pd.DataFrame:
        """客户同期群分析"""
        try:
            # 获取每个客户的首次购买月份（同期群）
            customer_first = self.df.groupby('customer_id')['date'].min().reset_index()
            customer_first.columns = ['customer_id', 'first_purchase_date']
            customer_first['cohort_month'] = customer_first['first_purchase_date'].dt.to_period('M')

            # 合并到原始数据
            df_cohort = self.df.merge(customer_first, on='customer_id')

            # 计算每个订单的购买周期（距首次购买的月数）
            df_cohort['order_month'] = df_cohort['date'].dt.to_period('M')
            df_cohort['period_number'] = (df_cohort['order_month'] - df_cohort['cohort_month']).apply(attrgetter('n'))

            # 构建同期群表
            cohort_data = df_cohort.groupby(['cohort_month', 'period_number'])['customer_id'].nunique().reset_index()
            cohort_counts = cohort_data.pivot(index='cohort_month', columns='period_number', values='customer_id')

            # 计算留存率
            cohort_sizes = customer_first.groupby('cohort_month')['customer_id'].nunique()
            retention = cohort_counts.divide(cohort_sizes, axis=0)

            self.results['cohort_counts'] = cohort_counts
            self.results['retention'] = retention
            logger.info("客户同期群分析完成")
            return retention

        except Exception as e:
            logger.error(f"同期群分析出错: {e}")
            return pd.DataFrame()

    def customer_segment_analysis(self) -> pd.DataFrame:
        """客户分群统计分析"""
        try:
            if 'rfm' not in self.results:
                self.rfm_analysis()

            rfm = self.results['rfm']

            # 按客户分层统计
            segment_stats = rfm.groupby('customer_segment').agg({
                'customer_id': 'count',
                'recency': 'mean',
                'frequency': 'mean',
                'monetary': ['mean', 'sum'],
                'rfm_score': 'mean'
            }).reset_index()

            segment_stats.columns = ['customer_segment', 'customer_count', 'avg_recency',
                                    'avg_frequency', 'avg_monetary', 'total_monetary', 'avg_rfm_score']

            # 计算占比
            segment_stats['customer_share'] = segment_stats['customer_count'] / segment_stats['customer_count'].sum() * 100
            segment_stats['revenue_share'] = segment_stats['total_monetary'] / segment_stats['total_monetary'].sum() * 100

            self.results['segment_stats'] = segment_stats
            logger.info("客户分群统计分析完成")
            return segment_stats

        except Exception as e:
            logger.error(f"客户分群分析出错: {e}")
            return pd.DataFrame()

    def customer_behavior_analysis(self) -> pd.DataFrame:
        """客户行为分析"""
        try:
            behavior = self.df.groupby('customer_id').agg({
                'date': ['min', 'max', 'count'],
                'total_amount': ['sum', 'mean', 'std'],
                'quantity': 'sum',
                'product_id': 'nunique',
                'category': lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown'
            }).reset_index()

            behavior.columns = ['customer_id', 'first_purchase', 'last_purchase', 'purchase_count',
                               'total_spent', 'avg_order_value', 'order_value_std',
                               'total_quantity', 'unique_products', 'preferred_category']

            # 计算购买间隔（平均天数）
            behavior['avg_purchase_interval'] = (
                (behavior['last_purchase'] - behavior['first_purchase']).dt.days /
                (behavior['purchase_count'] - 1).replace(0, 1)
            )

            # 计算客户活跃度
            max_date = self.df['date'].max()
            behavior['days_since_last_purchase'] = (max_date - behavior['last_purchase']).dt.days

            def get_activity_status(days):
                if days <= 30:
                    return '活跃'
                elif days <= 90:
                    return '近期活跃'
                elif days <= 180:
                    return '沉睡'
                else:
                    return '流失'

            behavior['activity_status'] = behavior['days_since_last_purchase'].apply(get_activity_status)

            self.results['behavior'] = behavior
            logger.info("客户行为分析完成")
            return behavior

        except Exception as e:
            logger.error(f"客户行为分析出错: {e}")
            return pd.DataFrame()

    def customer_geographic_analysis(self) -> pd.DataFrame:
        """客户地理分析"""
        try:
            if 'city' not in self.df.columns:
                logger.warning("缺少city列，跳过地理分析")
                return pd.DataFrame()

            geo = self.df.groupby('city').agg({
                'customer_id': 'nunique',
                'total_amount': ['sum', 'mean'],
                'transaction_id': 'count'
            }).reset_index()

            geo.columns = ['city', 'unique_customers', 'total_revenue', 'avg_order_value', 'transactions']
            geo['revenue_per_customer'] = geo['total_revenue'] / geo['unique_customers']

            geo = geo.sort_values('total_revenue', ascending=False)

            self.results['geographic'] = geo
            logger.info("客户地理分析完成")
            return geo

        except Exception as e:
            logger.error(f"客户地理分析出错: {e}")
            return pd.DataFrame()

    def customer_demographic_analysis(self) -> pd.DataFrame:
        """客户人口统计分析"""
        try:
            demo_stats = {}

            # 性别分析
            if 'gender' in self.df.columns:
                gender = self.df.groupby('gender').agg({
                    'customer_id': 'nunique',
                    'total_amount': ['sum', 'mean']
                }).reset_index()
                gender.columns = ['gender', 'customer_count', 'total_revenue', 'avg_order_value']
                demo_stats['gender'] = gender

            # 年龄段分析
            if 'age_group' in self.df.columns:
                age = self.df.groupby('age_group').agg({
                    'customer_id': 'nunique',
                    'total_amount': ['sum', 'mean']
                }).reset_index()
                age.columns = ['age_group', 'customer_count', 'total_revenue', 'avg_order_value']
                demo_stats['age'] = age

            # 会员等级分析
            if 'membership_level' in self.df.columns:
                membership = self.df.groupby('membership_level').agg({
                    'customer_id': 'nunique',
                    'total_amount': ['sum', 'mean']
                }).reset_index()
                membership.columns = ['membership_level', 'customer_count', 'total_revenue', 'avg_order_value']
                demo_stats['membership'] = membership

            self.results['demographic'] = demo_stats
            logger.info("客户人口统计分析完成")
            return demo_stats

        except Exception as e:
            logger.error(f"客户人口统计分析出错: {e}")
            return {}

    def churn_analysis(self, churn_days: int = 90) -> pd.DataFrame:
        """
        客户流失分析

        Args:
            churn_days: 流失定义天数（超过此天数未购买视为流失）
        """
        try:
            max_date = self.df['date'].max()

            churn = self.df.groupby('customer_id').agg({
                'date': 'max',
                'total_amount': 'sum',
                'transaction_id': 'count'
            }).reset_index()

            churn.columns = ['customer_id', 'last_purchase', 'total_spent', 'transaction_count']
            churn['days_since_last_purchase'] = (max_date - churn['last_purchase']).dt.days
            churn['is_churned'] = churn['days_since_last_purchase'] > churn_days

            # 流失客户特征
            churn_stats = {
                'total_customers': len(churn),
                'churned_customers': churn['is_churned'].sum(),
                'churn_rate': churn['is_churned'].mean() * 100,
                'avg_days_since_purchase': churn['days_since_last_purchase'].mean()
            }

            self.results['churn'] = churn
            self.results['churn_stats'] = churn_stats
            logger.info(f"客户流失分析完成，流失率: {churn_stats['churn_rate']:.2f}%")
            return churn

        except Exception as e:
            logger.error(f"客户流失分析出错: {e}")
            return pd.DataFrame()

    def run_all_analysis(self) -> Dict:
        """运行所有客户价值分析"""
        logger.info("开始客户价值分析...")

        self.rfm_analysis()
        self.customer_lifetime_value()
        # self.customer_cohort_analysis()  # 暂时跳过，需要导入attrgetter
        self.customer_segment_analysis()
        self.customer_behavior_analysis()
        self.customer_geographic_analysis()
        self.customer_demographic_analysis()
        self.churn_analysis()

        logger.info("客户价值分析全部完成")
        return self.results

    def get_results(self) -> Dict:
        """获取分析结果"""
        return self.results


# 导入attrgetter用于同期群分析
from operator import attrgetter
