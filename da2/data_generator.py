"""
电商销售数据生成器
生成示例CSV数据用于分析
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os


class DataGenerator:
    """生成电商示例数据"""

    def __init__(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)

    def generate_products(self, n_products=50, output_path='data/products.csv'):
        """生成商品数据"""
        categories = ['电子产品', '服装', '食品', '家居', '美妆', '运动', '图书', '玩具']
        brands = ['品牌A', '品牌B', '品牌C', '品牌D', '品牌E']

        products = []
        for i in range(n_products):
            category = random.choice(categories)
            base_price = random.uniform(10, 1000)
            products.append({
                'product_id': f'P{i+1:04d}',
                'product_name': f'{category}商品{i+1}',
                'category': category,
                'brand': random.choice(brands),
                'price': round(base_price, 2),
                'cost': round(base_price * random.uniform(0.4, 0.7), 2),
                'stock': random.randint(0, 500)
            })

        df = pd.DataFrame(products)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"生成商品数据: {output_path}, 共{len(df)}条记录")
        return df

    def generate_customers(self, n_customers=1000, output_path='data/customers.csv'):
        """生成客户数据"""
        cities = ['北京', '上海', '广州', '深圳', '杭州', '南京', '成都', '武汉', '西安', '重庆']
        genders = ['男', '女']
        age_groups = ['18-25', '26-35', '36-45', '46-55', '55+']

        customers = []
        for i in range(n_customers):
            register_date = datetime(2020, 1, 1) + timedelta(days=random.randint(0, 1460))
            customers.append({
                'customer_id': f'C{i+1:06d}',
                'gender': random.choice(genders),
                'age_group': random.choice(age_groups),
                'city': random.choice(cities),
                'register_date': register_date.strftime('%Y-%m-%d'),
                'membership_level': random.choice(['普通', '银卡', '金卡', '钻石']),
                'is_active': random.choice([0, 1])
            })

        df = pd.DataFrame(customers)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"生成客户数据: {output_path}, 共{len(df)}条记录")
        return df

    def generate_sales(self, n_transactions=5000, start_date='2023-01-01',
                       end_date='2023-12-31', output_path='data/sales.csv'):
        """生成销售数据"""
        products_df = self.generate_products()
        customers_df = self.generate_customers()

        product_ids = products_df['product_id'].tolist()
        product_prices = dict(zip(products_df['product_id'], products_df['price']))
        customer_ids = customers_df['customer_id'].tolist()

        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        date_range = (end - start).days

        sales = []
        for i in range(n_transactions):
            transaction_date = start + timedelta(days=random.randint(0, date_range))
            product_id = random.choice(product_ids)
            customer_id = random.choice(customer_ids)
            quantity = random.randint(1, 5)
            unit_price = product_prices[product_id]

            # 添加季节性波动
            month = transaction_date.month
            if month in [11, 12]:  # 双十一、双十二
                quantity = int(quantity * random.uniform(1.5, 3))
            elif month in [6, 7, 8]:  # 夏季促销
                quantity = int(quantity * random.uniform(1.2, 2))

            discount = random.choice([0, 0, 0, 0.1, 0.15, 0.2, 0.3])
            total_amount = round(unit_price * quantity * (1 - discount), 2)

            sales.append({
                'transaction_id': f'T{i+1:08d}',
                'date': transaction_date.strftime('%Y-%m-%d'),
                'customer_id': customer_id,
                'product_id': product_id,
                'quantity': quantity,
                'unit_price': unit_price,
                'discount': discount,
                'total_amount': total_amount,
                'payment_method': random.choice(['支付宝', '微信支付', '银行卡', '信用卡']),
                'channel': random.choice(['APP', '网页', '小程序'])
            })

        df = pd.DataFrame(sales)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"生成销售数据: {output_path}, 共{len(df)}条记录")
        return df

    def generate_all_data(self, data_dir='data'):
        """生成所有示例数据"""
        os.makedirs(data_dir, exist_ok=True)

        products = self.generate_products(output_path=f'{data_dir}/products.csv')
        customers = self.generate_customers(output_path=f'{data_dir}/customers.csv')
        sales = self.generate_sales(output_path=f'{data_dir}/sales.csv')

        return {
            'products': products,
            'customers': customers,
            'sales': sales
        }


if __name__ == '__main__':
    generator = DataGenerator(seed=42)
    generator.generate_all_data()
