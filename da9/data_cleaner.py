import pandas as pd
import numpy as np


class DataCleaner:
    """
    数据清洗类，处理缺失值、异常值等
    """
    
    def __init__(self, users, products, orders, order_details):
        self.users = users.copy()
        self.products = products.copy()
        self.orders = orders.copy()
        self.order_details = order_details.copy()
        self.cleaning_report = {}
    
    def handle_missing_values(self):
        """
        处理缺失值
        """
        print("\n开始处理缺失值...")
        
        # 记录各表的缺失值情况
        self.cleaning_report['missing_values'] = {
            'users': self.users.isnull().sum().to_dict(),
            'products': self.products.isnull().sum().to_dict(),
            'orders': self.orders.isnull().sum().to_dict(),
            'order_details': self.order_details.isnull().sum().to_dict()
        }
        
        # 实际数据可能没有缺失值，但为了演示，我们添加一些模拟缺失值的处理逻辑
        # 用户表：地区缺失用"未知"填充，年龄缺失用中位数填充
        if 'region' in self.users.columns and self.users['region'].isnull().any():
            self.users['region'] = self.users['region'].fillna('未知')
            print("  - 用户表：地区缺失值已用'未知'填充")
        
        if 'age' in self.users.columns and self.users['age'].isnull().any():
            median_age = self.users['age'].median()
            self.users['age'] = self.users['age'].fillna(median_age)
            print(f"  - 用户表：年龄缺失值已用中位数{median_age:.0f}填充")
        
        # 产品表：单价缺失用类别均值填充
        if 'unit_price' in self.products.columns and self.products['unit_price'].isnull().any():
            for category in self.products['category'].unique():
                mask = self.products['category'] == category
                mean_price = self.products.loc[mask, 'unit_price'].mean()
                self.products.loc[mask & self.products['unit_price'].isnull(), 'unit_price'] = mean_price
            print("  - 产品表：单价缺失值已用类别均值填充")
        
        # 订单表：支付方式缺失用"其他"填充
        if 'payment_method' in self.orders.columns and self.orders['payment_method'].isnull().any():
            self.orders['payment_method'] = self.orders['payment_method'].fillna('其他')
            print("  - 订单表：支付方式缺失值已用'其他'填充")
        
        # 订单详情：金额缺失重新计算
        if 'line_amount' in self.order_details.columns and self.order_details['line_amount'].isnull().any():
            mask = self.order_details['line_amount'].isnull()
            self.order_details.loc[mask, 'line_amount'] = (
                self.order_details.loc[mask, 'unit_price'] * 
                self.order_details.loc[mask, 'quantity'] * 
                self.order_details.loc[mask, 'discount']
            ).round(2)
            print("  - 订单详情表：金额缺失值已重新计算填充")
        
        # 重新计算订单总金额
        order_totals = self.order_details.groupby('order_id')['line_amount'].sum().reset_index()
        order_totals.columns = ['order_id', 'total_amount']
        
        # 更新订单表中的总金额
        self.orders = self.orders.drop(columns=['total_amount'], errors='ignore')
        self.orders = self.orders.merge(order_totals, on='order_id', how='left')
        
        # 对于没有订单详情的订单，总金额设为0
        self.orders['total_amount'] = self.orders['total_amount'].fillna(0)
        
        print("缺失值处理完成。")
    
    def handle_outliers(self):
        """
        处理异常值
        """
        print("\n开始处理异常值...")
        
        self.cleaning_report['outliers'] = {}
        
        # 订单金额异常值检测（使用IQR方法）
        if 'total_amount' in self.orders.columns:
            Q1 = self.orders['total_amount'].quantile(0.25)
            Q3 = self.orders['total_amount'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = max(0, Q1 - 1.5 * IQR)
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_mask = (self.orders['total_amount'] < lower_bound) | (self.orders['total_amount'] > upper_bound)
            n_outliers = outlier_mask.sum()
            
            self.cleaning_report['outliers']['total_amount'] = {
                'count': int(n_outliers),
                'Q1': float(Q1),
                'Q3': float(Q3),
                'IQR': float(IQR),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound)
            }
            
            if n_outliers > 0:
                print(f"  - 订单金额：检测到 {n_outliers} 个异常值")
                print(f"    IQR范围: [{lower_bound:.2f}, {upper_bound:.2f}]")
                
                # 对于异常值，我们将其限制在合理范围内（截断处理）
                self.orders.loc[self.orders['total_amount'] > upper_bound, 'total_amount'] = upper_bound
                self.orders.loc[self.orders['total_amount'] < lower_bound, 'total_amount'] = lower_bound
                
                print("    已将异常值截断到IQR范围内")
            else:
                print("  - 订单金额：未检测到异常值")
        
        # 订单详情数量异常值
        if 'quantity' in self.order_details.columns:
            max_quantity = 10  # 假设最大合理购买数量为10
            outlier_mask = self.order_details['quantity'] > max_quantity
            n_outliers = outlier_mask.sum()
            
            self.cleaning_report['outliers']['quantity'] = {
                'count': int(n_outliers),
                'max_allowed': max_quantity
            }
            
            if n_outliers > 0:
                print(f"  - 购买数量：检测到 {n_outliers} 个异常值（超过{max_quantity}件）")
                self.order_details.loc[outlier_mask, 'quantity'] = max_quantity
                print("    已将异常数量截断到最大值")
            else:
                print("  - 购买数量：未检测到异常值")
        
        # 产品单价异常值
        if 'unit_price' in self.products.columns:
            # 按类别检查异常值
            for category in self.products['category'].unique():
                mask = self.products['category'] == category
                category_prices = self.products.loc[mask, 'unit_price']
                
                if len(category_prices) > 0:
                    Q1 = category_prices.quantile(0.25)
                    Q3 = category_prices.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = max(0, Q1 - 1.5 * IQR)
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outlier_mask = mask & (
                        (self.products['unit_price'] < lower_bound) | 
                        (self.products['unit_price'] > upper_bound)
                    )
                    n_outliers = outlier_mask.sum()
                    
                    if n_outliers > 0 and category not in self.cleaning_report['outliers']:
                        self.cleaning_report['outliers'][f'product_price_{category}'] = {
                            'count': int(n_outliers)
                        }
                        
                        # 截断异常值
                        self.products.loc[outlier_mask & (self.products['unit_price'] > upper_bound), 'unit_price'] = upper_bound
                        self.products.loc[outlier_mask & (self.products['unit_price'] < lower_bound), 'unit_price'] = lower_bound
        
        # 重新计算订单详情中的金额
        if 'unit_price' in self.order_details.columns and 'quantity' in self.order_details.columns:
            self.order_details['line_amount'] = (
                self.order_details['unit_price'] * 
                self.order_details['quantity'] * 
                self.order_details['discount']
            ).round(2)
        
        print("异常值处理完成。")
    
    def filter_valid_data(self):
        """
        过滤有效数据
        """
        print("\n开始过滤有效数据...")
        
        self.cleaning_report['filtering'] = {}
        
        # 只保留已完成的订单
        original_order_count = len(self.orders)
        self.orders = self.orders[self.orders['order_status'] == '已完成'].copy()
        
        filtered_order_count = len(self.orders)
        removed_orders = original_order_count - filtered_order_count
        
        self.cleaning_report['filtering']['orders'] = {
            'original': original_order_count,
            'filtered': filtered_order_count,
            'removed': removed_orders
        }
        
        if removed_orders > 0:
            print(f"  - 订单：已过滤掉 {removed_orders} 个非'已完成'状态的订单")
        
        # 过滤订单详情，只保留已完成订单的详情
        valid_order_ids = set(self.orders['order_id'])
        original_detail_count = len(self.order_details)
        self.order_details = self.order_details[
            self.order_details['order_id'].isin(valid_order_ids)
        ].copy()
        
        filtered_detail_count = len(self.order_details)
        removed_details = original_detail_count - filtered_detail_count
        
        self.cleaning_report['filtering']['order_details'] = {
            'original': original_detail_count,
            'filtered': filtered_detail_count,
            'removed': removed_details
        }
        
        if removed_details > 0:
            print(f"  - 订单详情：已过滤掉 {removed_details} 条无效订单详情")
        
        # 重新计算订单总金额
        order_totals = self.order_details.groupby('order_id')['line_amount'].sum().reset_index()
        order_totals.columns = ['order_id', 'total_amount']
        
        self.orders = self.orders.drop(columns=['total_amount'], errors='ignore')
        self.orders = self.orders.merge(order_totals, on='order_id', how='left')
        
        print("数据过滤完成。")
    
    def clean_all(self):
        """
        执行所有清洗步骤
        """
        print("\n" + "="*50)
        print("开始数据清洗")
        print("="*50)
        
        self.handle_missing_values()
        self.handle_outliers()
        self.filter_valid_data()
        
        print("\n" + "="*50)
        print("数据清洗完成")
        print("="*50)
        
        return self.users, self.products, self.orders, self.order_details
    
    def get_cleaning_report(self):
        """
        获取清洗报告
        """
        return self.cleaning_report


if __name__ == '__main__':
    from data_generator import load_datasets
    
    # 加载数据
    users, products, orders, order_details = load_datasets()
    
    # 创建清洗器并执行清洗
    cleaner = DataCleaner(users, products, orders, order_details)
    cleaned_users, cleaned_products, cleaned_orders, cleaned_order_details = cleaner.clean_all()
    
    # 打印清洗后的数据统计
    print("\n清洗后数据统计:")
    print(f"  - 用户数: {len(cleaned_users)}")
    print(f"  - 产品数: {len(cleaned_products)}")
    print(f"  - 订单数: {len(cleaned_orders)}")
    print(f"  - 订单详情数: {len(cleaned_order_details)}")
