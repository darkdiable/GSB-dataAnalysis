import pandas as pd
import os
import logging
from typing import Tuple, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, data_dir: str = "../data"):
        self.data_dir = data_dir
        self.orders_df: Optional[pd.DataFrame] = None
        self.customers_df: Optional[pd.DataFrame] = None
        self.products_df: Optional[pd.DataFrame] = None
        self.merged_df: Optional[pd.DataFrame] = None
        
    def load_orders(self, filename: str = "orders.csv") -> pd.DataFrame:
        file_path = os.path.join(self.data_dir, filename)
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"订单文件不存在: {file_path}")
            
            df = pd.read_csv(file_path, encoding='utf-8-sig', parse_dates=['order_date'])
            logger.info(f"成功加载订单数据: {len(df)} 条记录")
            
            required_columns = ['order_id', 'customer_id', 'product_id', 'order_date', 'quantity', 'total_amount']
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"订单数据缺少必要列: {missing_cols}")
            
            df['order_date'] = pd.to_datetime(df['order_date'])
            df = df[df['order_status'] == '已完成'].copy()
            logger.info(f"筛选已完成订单: {len(df)} 条")
            
            self.orders_df = df
            return df
            
        except Exception as e:
            logger.error(f"加载订单数据失败: {str(e)}")
            raise
            
    def load_customers(self, filename: str = "customers.csv") -> pd.DataFrame:
        file_path = os.path.join(self.data_dir, filename)
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"客户文件不存在: {file_path}")
                
            df = pd.read_csv(file_path, encoding='utf-8-sig', parse_dates=['registration_date'])
            logger.info(f"成功加载客户数据: {len(df)} 条记录")
            
            required_columns = ['customer_id', 'region', 'age_group', 'gender', 'membership_level']
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"客户数据缺少必要列: {missing_cols}")
            
            self.customers_df = df
            return df
            
        except Exception as e:
            logger.error(f"加载客户数据失败: {str(e)}")
            raise
            
    def load_products(self, filename: str = "products.csv") -> pd.DataFrame:
        file_path = os.path.join(self.data_dir, filename)
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"商品文件不存在: {file_path}")
                
            df = pd.read_csv(file_path, encoding='utf-8-sig')
            logger.info(f"成功加载商品数据: {len(df)} 条记录")
            
            required_columns = ['product_id', 'category', 'subcategory', 'price', 'brand']
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"商品数据缺少必要列: {missing_cols}")
            
            self.products_df = df
            return df
            
        except Exception as e:
            logger.error(f"加载商品数据失败: {str(e)}")
            raise
            
    def load_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        try:
            orders = self.load_orders()
            customers = self.load_customers()
            products = self.load_products()
            return orders, customers, products
        except Exception as e:
            logger.error(f"加载所有数据失败: {str(e)}")
            raise
            
    def merge_data(self) -> pd.DataFrame:
        try:
            if self.orders_df is None or self.customers_df is None or self.products_df is None:
                self.load_all_data()
            
            merged = pd.merge(
                self.orders_df,
                self.customers_df,
                on='customer_id',
                how='left'
            )
            
            merged = pd.merge(
                merged,
                self.products_df,
                on='product_id',
                how='left'
            )
            
            if merged.empty:
                raise ValueError("合并后的数据为空")
            
            merged['year'] = merged['order_date'].dt.year
            merged['month'] = merged['order_date'].dt.month
            merged['quarter'] = merged['order_date'].dt.to_period('Q')
            merged['weekday'] = merged['order_date'].dt.weekday
            merged['year_month'] = merged['order_date'].dt.to_period('M')
            
            logger.info(f"数据合并完成: {len(merged)} 条记录, {len(merged.columns)} 个字段")
            self.merged_df = merged
            return merged
            
        except Exception as e:
            logger.error(f"数据合并失败: {str(e)}")
            raise
            
    def data_summary(self) -> dict:
        try:
            if self.merged_df is None:
                self.merge_data()
            
            summary = {
                'total_orders': self.merged_df['order_id'].nunique(),
                'total_customers': self.merged_df['customer_id'].nunique(),
                'total_products': self.merged_df['product_id'].nunique(),
                'total_revenue': round(self.merged_df['total_amount'].sum(), 2),
                'date_range': (self.merged_df['order_date'].min(), self.merged_df['order_date'].max()),
                'avg_order_value': round(self.merged_df.groupby('order_id')['total_amount'].sum().mean(), 2)
            }
            
            logger.info(f"""数据概览:
                订单数: {summary['total_orders']}
                客户数: {summary['total_customers']}
                商品数: {summary['total_products']}
                总销售额: {summary['total_revenue']}
                客单价: {summary['avg_order_value']}
            """)
            
            return summary
            
        except Exception as e:
            logger.error(f"生成数据摘要失败: {str(e)}")
            raise

if __name__ == '__main__':
    loader = DataLoader()
    loader.load_all_data()
    loader.merge_data()
    loader.data_summary()
