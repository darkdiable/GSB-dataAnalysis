import pandas as pd
import numpy as np
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SalesAnalyzer:
    def __init__(self):
        pass
        
    def analyze_sales_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        try:
            logger.info("开始销售趋势分析...")
            
            monthly_sales = df.groupby('year_month')['total_amount'].sum().reset_index()
            monthly_sales['year_month_str'] = monthly_sales['year_month'].astype(str)
            
            monthly_growth = monthly_sales['total_amount'].pct_change() * 100
            
            sales_by_year = df.groupby('year')['total_amount'].agg(['sum', 'mean', 'count']).round(2)
            sales_by_year.columns = ['总销售额', '平均客单价', '订单数']
            
            sales_by_month = df.groupby('month')['total_amount'].sum()
            peak_month = sales_by_month.idxmax()
            peak_month_sales = sales_by_month.max()
            
            sales_by_weekday = df.groupby('weekday')['total_amount'].agg(['sum', 'mean'])
            sales_by_weekday.index = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
            
            trend_summary = {
                'total_period_growth': ((monthly_sales['total_amount'].iloc[-1] / monthly_sales['total_amount'].iloc[0] - 1) * 100) if len(monthly_sales) > 1 else 0,
                'avg_monthly_growth': monthly_growth.mean(),
                'peak_month': peak_month,
                'peak_month_sales': peak_month_sales,
                'best_weekday': sales_by_weekday['sum'].idxmax()
            }
            
            logger.info(f"销售趋势分析完成: 整体增长 {trend_summary['total_period_growth']:.1f}%")
            
            return {
                'monthly_sales': monthly_sales,
                'monthly_growth': monthly_growth,
                'sales_by_year': sales_by_year,
                'sales_by_weekday': sales_by_weekday,
                'summary': trend_summary
            }
            
        except Exception as e:
            logger.error(f"销售趋势分析失败: {str(e)}")
            raise
            
    def analyze_region_performance(self, df: pd.DataFrame) -> Dict[str, Any]:
        try:
            logger.info("开始区域销售分析...")
            
            region_sales = df.groupby('region')['total_amount'].agg(['sum', 'mean', 'count', 'nunique']).round(2)
            region_sales.columns = ['总销售额', '平均订单金额', '订单数', '客户数']
            region_sales = region_sales.sort_values('总销售额', ascending=False)
            
            region_category = df.groupby(['region', 'category'])['total_amount'].sum().unstack(fill_value=0)
            
            top_region = region_sales.index[0]
            top_region_sales = region_sales.iloc[0]['总销售额']
            
            logger.info(f"区域销售TOP1: {top_region} - {top_region_sales}")
            
            return {
                'region_sales': region_sales,
                'region_category_heatmap': region_category,
                'top_region': top_region
            }
            
        except Exception as e:
            logger.error(f"区域销售分析失败: {str(e)}")
            raise
            
    def analyze_customer_value(self, customer_features: pd.DataFrame) -> Dict[str, Any]:
        try:
            logger.info("开始客户价值分析...")
            
            segment_distribution = customer_features['Customer_Segment'].value_counts()
            segment_value = customer_features.groupby('Customer_Segment').agg({
                'Monetary': ['sum', 'mean'],
                'Frequency': 'mean',
                'Recency': 'mean'
            }).round(2)
            
            value_by_membership = customer_features.groupby('membership_level').agg({
                'Monetary': ['sum', 'count']
            }).round(2)
            value_by_membership.columns = ['总消费金额', '客户数']
            
            value_by_age = customer_features.groupby('age_group').agg({
                'Monetary': 'sum',
                'Frequency': 'mean'
            }).sort_values('Monetary', ascending=False)
            
            top_customers = customer_features.nlargest(10, 'Monetary')[['Monetary', 'Frequency', 'Recency', 'membership_level', 'region']]
            
            logger.info(f"客户分群: {segment_distribution.to_dict()}")
            
            return {
                'segment_distribution': segment_distribution,
                'segment_value': segment_value,
                'value_by_membership': value_by_membership,
                'value_by_age': value_by_age,
                'top_customers': top_customers
            }
            
        except Exception as e:
            logger.error(f"客户价值分析失败: {str(e)}")
            raise
            
    def analyze_product_performance(self, df: pd.DataFrame, product_features: pd.DataFrame) -> Dict[str, Any]:
        try:
            logger.info("开始商品表现分析...")
            
            category_sales = df.groupby('category')['total_amount'].agg(['sum', 'count', 'mean']).round(2)
            category_sales.columns = ['总销售额', '销量', '平均单价']
            category_sales = category_sales.sort_values('总销售额', ascending=False)
            
            subcategory_sales = df.groupby(['category', 'subcategory'])['total_amount'].sum().unstack(fill_value=0)
            
            top_products = product_features.nlargest(10, 'total_revenue')[['category', 'total_revenue', 'total_quantity', 'order_count']]
            
            brand_sales = df.groupby('brand')['total_amount'].sum().sort_values(ascending=False).head(10)
            
            category_contribution = (category_sales['总销售额'] / category_sales['总销售额'].sum() * 100).round(2)
            
            logger.info(f"品类贡献度: {category_contribution.to_dict()}")
            
            return {
                'category_sales': category_sales,
                'subcategory_sales': subcategory_sales,
                'top_products': top_products,
                'brand_sales': brand_sales,
                'category_contribution': category_contribution
            }
            
        except Exception as e:
            logger.error(f"商品表现分析失败: {str(e)}")
            raise
            
    def analyze_correlation(self, df: pd.DataFrame) -> Dict[str, Any]:
        try:
            logger.info("开始关联分析...")
            
            numeric_cols = ['quantity', 'unit_price', 'total_amount', 'day_of_week', 'day_of_month']
            correlation_matrix = df[numeric_cols].corr()
            
            payment_performance = df.groupby('payment_method').agg({
                'total_amount': ['sum', 'mean'],
                'order_id': 'count'
            }).round(2)
            
            gender_preference = df.groupby(['gender', 'category'])['total_amount'].sum().unstack(fill_value=0)
            
            return {
                'correlation_matrix': correlation_matrix,
                'payment_performance': payment_performance,
                'gender_preference': gender_preference
            }
            
        except Exception as e:
            logger.error(f"关联分析失败: {str(e)}")
            raise
            
    def generate_insights(self, analysis_results: Dict[str, Any]) -> Dict[str, str]:
        try:
            logger.info("生成业务洞察...")
            
            insights = {}
            
            trend = analysis_results['sales_trend']['summary']
            insights['销售趋势'] = (f"分析期内整体销售增长{trend['total_period_growth']:.1f}%，"
                                  f"{trend['peak_month']}月为销售峰值，最佳销售日为{trend['best_weekday']}")
            
            region = analysis_results['region_perf']['top_region']
            insights['区域表现'] = f"{region}地区贡献最大销售额，是核心市场"
            
            cv = analysis_results['customer_value']
            total_high_value = cv['segment_value'].loc[['重要价值客户', '高价值客户'], ('Monetary', 'sum')].sum()
            total_revenue = cv['segment_value'][('Monetary', 'sum')].sum()
            insights['客户价值'] = (f"重要价值客户和高价值客户贡献{total_high_value/total_revenue*100:.1f}%的收入，"
                                  f"TOP10客户贡献显著，建议重点维护")
            
            product = analysis_results['product_perf']
            top_category = product['category_sales'].index[0]
            insights['商品表现'] = (f"{top_category}是最畅销品类，"
                                  f"贡献{product['category_contribution'][top_category]}%的销售额，头部商品效应明显")
            
            logger.info("业务洞察生成完成")
            return insights
            
        except Exception as e:
            logger.error(f"生成业务洞察失败: {str(e)}")
            raise
            
    def run_full_analysis(self, df: pd.DataFrame, customer_features: pd.DataFrame, product_features: pd.DataFrame) -> Dict[str, Any]:
        try:
            logger.info("开始全维度分析...")
            
            results = {
                'sales_trend': self.analyze_sales_trend(df),
                'region_perf': self.analyze_region_performance(df),
                'customer_value': self.analyze_customer_value(customer_features),
                'product_perf': self.analyze_product_performance(df, product_features),
                'correlation': self.analyze_correlation(df)
            }
            
            results['insights'] = self.generate_insights(results)
            
            logger.info("全维度分析完成")
            return results
            
        except Exception as e:
            logger.error(f"全维度分析失败: {str(e)}")
            raise

if __name__ == '__main__':
    from data_loader import DataLoader
    from feature_engineering import FeatureEngineer
    
    loader = DataLoader()
    loader.load_all_data()
    merged = loader.merge_data()
    
    fe = FeatureEngineer()
    features = fe.transform_all(merged, loader.customers_df)
    
    analyzer = SalesAnalyzer()
    results = analyzer.run_full_analysis(features['enhanced_data'], features['customer_features'], features['product_features'])
    
    print("分析洞察:", results['insights'])
