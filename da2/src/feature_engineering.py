import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self):
        self.rfm_df: Optional[pd.DataFrame] = None
        self.customer_features: Optional[pd.DataFrame] = None
        
    def calculate_rfm(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info("开始计算RFM特征...")
            
            if df.empty:
                raise ValueError("输入数据为空")
                
            max_date = df['order_date'].max()
            
            rfm = df.groupby('customer_id').agg({
                'order_date': lambda x: (max_date - x.max()).days,
                'order_id': 'nunique',
                'total_amount': 'sum'
            }).rename(columns={
                'order_date': 'Recency',
                'order_id': 'Frequency',
                'total_amount': 'Monetary'
            })
            
            rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1], duplicates='drop')
            rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
            rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
            
            rfm['R_Score'] = rfm['R_Score'].astype(int)
            rfm['F_Score'] = rfm['F_Score'].astype(int)
            rfm['M_Score'] = rfm['M_Score'].astype(int)
            
            rfm['RFM_Score'] = rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']
            
            def segment_customer(row):
                if row['R_Score'] >= 4 and row['F_Score'] >= 4:
                    return '重要价值客户'
                elif row['R_Score'] >= 4 and row['F_Score'] <= 2:
                    return '新客户'
                elif row['R_Score'] <= 2 and row['F_Score'] >= 4:
                    return '重要流失客户'
                elif row['R_Score'] <= 2 and row['F_Score'] <= 2:
                    return '流失客户'
                elif row['M_Score'] >= 4:
                    return '高价值客户'
                else:
                    return '一般客户'
            
            rfm['Customer_Segment'] = rfm.apply(segment_customer, axis=1)
            
            logger.info(f"RFM计算完成，共 {len(rfm)} 个客户")
            logger.info(f"客户分群分布:\n{rfm['Customer_Segment'].value_counts()}")
            
            self.rfm_df = rfm
            return rfm
            
        except Exception as e:
            logger.error(f"RFM计算失败: {str(e)}")
            raise
            
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info("创建时间特征...")
            
            df = df.copy()
            df['day_of_week'] = df['order_date'].dt.dayofweek
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            df['day_of_month'] = df['order_date'].dt.day
            df['week_of_year'] = df['order_date'].dt.isocalendar().week
            df['season'] = df['month'].map({1: '冬季', 2: '冬季', 3: '春季', 4: '春季', 5: '春季',
                                            6: '夏季', 7: '夏季', 8: '夏季', 9: '秋季', 10: '秋季',
                                            11: '秋季', 12: '冬季'})
            
            df['hour'] = df['order_date'].dt.hour
            df['time_of_day'] = pd.cut(df['hour'], 
                                      bins=[-1, 6, 12, 18, 24],
                                      labels=['凌晨', '上午', '下午', '晚上'])
            
            logger.info("时间特征创建完成")
            return df
            
        except Exception as e:
            logger.error(f"创建时间特征失败: {str(e)}")
            raise
            
    def create_product_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        try:
            logger.info("创建商品特征...")
            
            product_sales = df.groupby(['category', 'product_id']).agg({
                'quantity': 'sum',
                'total_amount': 'sum',
                'order_id': 'nunique'
            }).rename(columns={
                'quantity': 'total_quantity',
                'total_amount': 'total_revenue',
                'order_id': 'order_count'
            }).reset_index()
            
            category_ranking = product_sales.groupby('category')['total_revenue'].sum().sort_values(ascending=False)
            logger.info(f"品类销售额排名:\n{category_ranking}")
            
            product_sales['category_rank'] = product_sales.groupby('category')['total_revenue'].rank(ascending=False)
            product_sales['is_top_product'] = (product_sales['category_rank'] <= 10).astype(int)
            
            logger.info("商品特征创建完成")
            return df, product_sales
            
        except Exception as e:
            logger.error(f"创建商品特征失败: {str(e)}")
            raise
            
    def create_customer_features(self, df: pd.DataFrame, customers_df: pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info("创建客户特征...")
            
            rfm = self.calculate_rfm(df)
            
            customer_order_stats = df.groupby('customer_id').agg({
                'quantity': ['mean', 'sum'],
                'unit_price': ['mean', 'max'],
                'category': 'nunique',
                'payment_method': lambda x: x.mode()[0] if not x.mode().empty else '未知'
            }).round(2)
            
            customer_order_stats.columns = ['avg_quantity', 'total_quantity', 'avg_price', 
                                           'max_price', 'category_count', 'preferred_payment']
            
            customer_features = customers_df.set_index('customer_id').join(rfm).join(customer_order_stats)
            
            customer_features['customer_lifetime'] = (
                datetime.now() - customer_features['registration_date']
            ).dt.days
            
            logger.info(f"客户特征创建完成，共 {len(customer_features)} 条记录")
            self.customer_features = customer_features
            return customer_features
            
        except Exception as e:
            logger.error(f"创建客户特征失败: {str(e)}")
            raise
            
    def calculate_association_rules(self, df: pd.DataFrame, min_support: float = 0.001) -> pd.DataFrame:
        try:
            logger.info("计算商品关联规则...")
            
            order_products = df.groupby('order_id')['product_id'].apply(list)
            
            from collections import defaultdict
            product_pairs = defaultdict(int)
            product_counts = defaultdict(int)
            
            for products in order_products:
                unique_products = list(set(products))
                for p in unique_products:
                    product_counts[p] += 1
                for i in range(len(unique_products)):
                    for j in range(i + 1, len(unique_products)):
                        pair = tuple(sorted([unique_products[i], unique_products[j]]))
                        product_pairs[pair] += 1
            
            n_orders = len(order_products)
            rules = []
            
            for (p1, p2), count in product_pairs.items():
                support = count / n_orders
                if support >= min_support and p1 in product_counts and p2 in product_counts:
                    if product_counts[p1] > 0 and product_counts[p2] > 0:
                        confidence_p1p2 = count / product_counts[p1]
                        confidence_p2p1 = count / product_counts[p2]
                        lift = (count * n_orders) / (product_counts[p1] * product_counts[p2])
                        
                        rules.append({
                            'product_a': p1,
                            'product_b': p2,
                            'support': round(support, 4),
                            'confidence_a_to_b': round(confidence_p1p2, 4),
                            'confidence_b_to_a': round(confidence_p2p1, 4),
                            'lift': round(lift, 4),
                            'co_occurrence_count': count
                        })
            
            if rules:
                rules_df = pd.DataFrame(rules).sort_values('lift', ascending=False)
                logger.info(f"关联规则计算完成，共 {len(rules_df)} 条规则")
                return rules_df.head(20)
            else:
                logger.info("未发现满足条件的关联规则")
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"计算关联规则失败: {str(e)}")
            logger.warning("跳过关联规则计算")
            return pd.DataFrame()
            
    def transform_all(self, merged_df: pd.DataFrame, customers_df: pd.DataFrame) -> dict:
        try:
            logger.info("开始特征工程全流程...")
            
            df_with_time = self.create_time_features(merged_df)
            df_with_features, product_features = self.create_product_features(df_with_time)
            customer_features = self.create_customer_features(df_with_features, customers_df)
            association_rules = self.calculate_association_rules(df_with_features)
            
            result = {
                'enhanced_data': df_with_features,
                'rfm_data': self.rfm_df,
                'customer_features': customer_features,
                'product_features': product_features,
                'association_rules': association_rules
            }
            
            logger.info("特征工程完成")
            return result
            
        except Exception as e:
            logger.error(f"特征工程全流程失败: {str(e)}")
            raise

if __name__ == '__main__':
    from data_loader import DataLoader
    loader = DataLoader()
    loader.load_all_data()
    merged = loader.merge_data()
    
    fe = FeatureEngineer()
    features = fe.transform_all(merged, loader.customers_df)
    print("特征工程结果:", features.keys())
