import pandas as pd
import os
from typing import Optional, Dict, List


class DataLoader:
    def __init__(self, data_dir: str = 'data'):
        self.data_dir = data_dir
        self.customers_df: Optional[pd.DataFrame] = None
        self.products_df: Optional[pd.DataFrame] = None
        self.orders_df: Optional[pd.DataFrame] = None
        
    def load_customers(self, filename: str = 'customers.csv') -> pd.DataFrame:
        filepath = os.path.join(self.data_dir, filename)
        try:
            self.customers_df = pd.read_csv(filepath, encoding='utf-8-sig')
            self.customers_df['register_date'] = pd.to_datetime(self.customers_df['register_date'])
            print(f"成功加载客户数据: {len(self.customers_df)} 条记录")
            return self.customers_df
        except FileNotFoundError:
            raise FileNotFoundError(f"客户数据文件不存在: {filepath}")
        except Exception as e:
            raise Exception(f"加载客户数据时发生错误: {str(e)}")
    
    def load_products(self, filename: str = 'products.csv') -> pd.DataFrame:
        filepath = os.path.join(self.data_dir, filename)
        try:
            self.products_df = pd.read_csv(filepath, encoding='utf-8-sig')
            print(f"成功加载商品数据: {len(self.products_df)} 条记录")
            return self.products_df
        except FileNotFoundError:
            raise FileNotFoundError(f"商品数据文件不存在: {filepath}")
        except Exception as e:
            raise Exception(f"加载商品数据时发生错误: {str(e)}")
    
    def load_orders(self, filename: str = 'orders.csv') -> pd.DataFrame:
        filepath = os.path.join(self.data_dir, filename)
        try:
            self.orders_df = pd.read_csv(filepath, encoding='utf-8-sig')
            self.orders_df['order_date'] = pd.to_datetime(self.orders_df['order_date'])
            print(f"成功加载订单数据: {len(self.orders_df)} 条记录")
            return self.orders_df
        except FileNotFoundError:
            raise FileNotFoundError(f"订单数据文件不存在: {filepath}")
        except Exception as e:
            raise Exception(f"加载订单数据时发生错误: {str(e)}")
    
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        self.load_customers()
        self.load_products()
        self.load_orders()
        return {
            'customers': self.customers_df,
            'products': self.products_df,
            'orders': self.orders_df
        }
    
    def validate_data(self) -> Dict[str, List[str]]:
        issues = {
            'customers': [],
            'products': [],
            'orders': []
        }
        
        if self.customers_df is not None:
            if self.customers_df['customer_id'].duplicated().any():
                issues['customers'].append("存在重复的客户ID")
            if self.customers_df.isnull().any().any():
                null_cols = self.customers_df.columns[self.customers_df.isnull().any()].tolist()
                issues['customers'].append(f"存在空值列: {null_cols}")
        
        if self.products_df is not None:
            if self.products_df['product_id'].duplicated().any():
                issues['products'].append("存在重复的商品ID")
            if (self.products_df['price'] <= 0).any():
                issues['products'].append("存在价格小于等于0的商品")
        
        if self.orders_df is not None:
            if self.orders_df['order_id'].duplicated().any():
                issues['orders'].append("存在重复的订单ID")
            if (self.orders_df['total_amount'] < 0).any():
                issues['orders'].append("存在金额为负的订单")
        
        return issues
    
    def get_merged_data(self) -> pd.DataFrame:
        if self.orders_df is None or self.customers_df is None or self.products_df is None:
            raise ValueError("请先加载所有数据")
        
        merged = self.orders_df.merge(
            self.customers_df, 
            on='customer_id', 
            how='left',
            suffixes=('', '_customer')
        )
        merged = merged.merge(
            self.products_df[['product_id', 'product_name', 'category', 'cost']],
            on='product_id',
            how='left',
            suffixes=('', '_product')
        )
        return merged
    
    def get_data_summary(self) -> Dict:
        summary = {}
        
        if self.customers_df is not None:
            summary['customers'] = {
                'total': len(self.customers_df),
                'cities': self.customers_df['city'].nunique(),
                'membership_levels': self.customers_df['membership_level'].value_counts().to_dict()
            }
        
        if self.products_df is not None:
            summary['products'] = {
                'total': len(self.products_df),
                'categories': self.products_df['category'].value_counts().to_dict(),
                'avg_price': self.products_df['price'].mean()
            }
        
        if self.orders_df is not None:
            summary['orders'] = {
                'total': len(self.orders_df),
                'total_revenue': self.orders_df['total_amount'].sum(),
                'avg_order_value': self.orders_df['total_amount'].mean(),
                'date_range': {
                    'start': self.orders_df['order_date'].min().strftime('%Y-%m-%d'),
                    'end': self.orders_df['order_date'].max().strftime('%Y-%m-%d')
                }
            }
        
        return summary
