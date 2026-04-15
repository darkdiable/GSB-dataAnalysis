import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os


class DataGenerator:
    def __init__(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        
    def generate_customers(self, n_customers=500):
        cities = ['北京', '上海', '广州', '深圳', '杭州', '成都', '武汉', '西安', '南京', '重庆']
        genders = ['男', '女']
        age_groups = ['18-25', '26-35', '36-45', '46-55', '55+']
        membership_levels = ['普通会员', '银卡会员', '金卡会员', '钻石会员']
        
        customers = []
        for i in range(n_customers):
            customer = {
                'customer_id': f'C{str(i+1).zfill(6)}',
                'customer_name': f'客户{i+1}',
                'gender': random.choice(genders),
                'age_group': random.choice(age_groups),
                'city': random.choice(cities),
                'membership_level': random.choices(membership_levels, weights=[50, 30, 15, 5])[0],
                'register_date': datetime(2022, 1, 1) + timedelta(days=random.randint(0, 730))
            }
            customers.append(customer)
        return pd.DataFrame(customers)
    
    def generate_products(self, n_products=100):
        categories = {
            '电子产品': ['手机', '平板', '笔记本', '耳机', '充电器', '数据线', '移动电源', '智能手表'],
            '服装': ['T恤', '牛仔裤', '连衣裙', '外套', '运动鞋', '休闲鞋', '羽绒服', '毛衣'],
            '家居': ['沙发', '床', '书桌', '台灯', '窗帘', '床上用品', '收纳盒', '装饰画'],
            '食品': ['零食', '饮料', '水果', '坚果', '茶叶', '咖啡', '保健品', '调味品'],
            '美妆': ['口红', '粉底', '面膜', '护肤水', '精华液', '眼霜', '防晒霜', '卸妆水']
        }
        
        products = []
        product_id = 1
        for category, items in categories.items():
            for item in items:
                base_price = {
                    '电子产品': random.uniform(100, 5000),
                    '服装': random.uniform(50, 1000),
                    '家居': random.uniform(30, 3000),
                    '食品': random.uniform(10, 200),
                    '美妆': random.uniform(30, 800)
                }[category]
                
                product = {
                    'product_id': f'P{str(product_id).zfill(6)}',
                    'product_name': item,
                    'category': category,
                    'price': round(base_price, 2),
                    'cost': round(base_price * random.uniform(0.4, 0.7), 2),
                    'stock': random.randint(50, 500)
                }
                products.append(product)
                product_id += 1
        return pd.DataFrame(products)
    
    def generate_orders(self, customers_df, products_df, n_orders=5000):
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2024, 12, 31)
        date_range = (end_date - start_date).days
        
        payment_methods = ['支付宝', '微信支付', '银行卡', '信用卡', '花呗']
        order_status = ['已完成', '已发货', '待发货', '已取消', '退货中']
        
        orders = []
        for i in range(n_orders):
            customer = customers_df.sample(1).iloc[0]
            product = products_df.sample(1).iloc[0]
            quantity = random.choices([1, 2, 3, 4, 5], weights=[50, 25, 15, 7, 3])[0]
            
            order_date = start_date + timedelta(days=random.randint(0, date_range))
            status = random.choices(order_status, weights=[70, 15, 5, 5, 5])[0]
            
            discount = random.choices([0, 0.9, 0.85, 0.8, 0.7], weights=[60, 15, 10, 10, 5])[0]
            total_amount = product['price'] * quantity * discount
            
            order = {
                'order_id': f'O{str(i+1).zfill(8)}',
                'customer_id': customer['customer_id'],
                'product_id': product['product_id'],
                'order_date': order_date,
                'quantity': quantity,
                'unit_price': product['price'],
                'discount_rate': discount,
                'total_amount': round(total_amount, 2),
                'payment_method': random.choice(payment_methods),
                'order_status': status,
                'category': product['category']
            }
            orders.append(order)
        
        return pd.DataFrame(orders)
    
    def generate_all_data(self, output_dir='data', n_customers=500, n_products=100, n_orders=5000):
        os.makedirs(output_dir, exist_ok=True)
        
        print("正在生成客户数据...")
        customers_df = self.generate_customers(n_customers)
        customers_df.to_csv(os.path.join(output_dir, 'customers.csv'), index=False, encoding='utf-8-sig')
        
        print("正在生成商品数据...")
        products_df = self.generate_products(n_products)
        products_df.to_csv(os.path.join(output_dir, 'products.csv'), index=False, encoding='utf-8-sig')
        
        print("正在生成订单数据...")
        orders_df = self.generate_orders(customers_df, products_df, n_orders)
        orders_df.to_csv(os.path.join(output_dir, 'orders.csv'), index=False, encoding='utf-8-sig')
        
        print(f"数据生成完成！")
        print(f"- 客户数据: {len(customers_df)} 条")
        print(f"- 商品数据: {len(products_df)} 条")
        print(f"- 订单数据: {len(orders_df)} 条")
        
        return customers_df, products_df, orders_df


if __name__ == '__main__':
    generator = DataGenerator()
    generator.generate_all_data()
