import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

def generate_customers(n=500):
    customer_ids = [f'C{i:04d}' for i in range(1, n+1)]
    regions = ['华东', '华南', '华北', '华中', '西南', '西北', '东北']
    age_groups = ['18-25', '26-35', '36-45', '46-55', '55+']
    genders = ['男', '女']
    membership_levels = ['普通', '银卡', '金卡', '钻石']
    
    customers = pd.DataFrame({
        'customer_id': customer_ids,
        'region': np.random.choice(regions, n, p=[0.25, 0.2, 0.18, 0.12, 0.1, 0.08, 0.07]),
        'age_group': np.random.choice(age_groups, n, p=[0.15, 0.35, 0.25, 0.15, 0.1]),
        'gender': np.random.choice(genders, n, p=[0.45, 0.55]),
        'membership_level': np.random.choice(membership_levels, n, p=[0.5, 0.25, 0.15, 0.1]),
        'registration_date': [datetime(2022, 1, 1) + timedelta(days=random.randint(0, 730)) for _ in range(n)]
    })
    return customers

def generate_products(n=100):
    product_ids = [f'P{i:03d}' for i in range(1, n+1)]
    categories = ['电子产品', '服装鞋帽', '家居用品', '食品饮料', '美妆护肤', '母婴用品', '运动户外']
    subcategories = {
        '电子产品': ['手机', '电脑', '耳机', '配件'],
        '服装鞋帽': ['上衣', '裤子', '鞋子', '配饰'],
        '家居用品': ['家具', '厨具', '床品', '收纳'],
        '食品饮料': ['零食', '饮料', '生鲜', '粮油'],
        '美妆护肤': ['护肤', '彩妆', '香水', '工具'],
        '母婴用品': ['奶粉', '尿裤', '玩具', '童装'],
        '运动户外': ['健身', '户外', '球类', '装备']
    }
    
    products = []
    for pid in product_ids:
        cat = random.choice(categories)
        subcat = random.choice(subcategories[cat])
        products.append({
            'product_id': pid,
            'category': cat,
            'subcategory': subcat,
            'price': round(np.random.uniform(10, 5000), 2),
            'cost': round(np.random.uniform(5, 3000), 2),
            'brand': f'品牌{random.randint(1, 20)}'
        })
    return pd.DataFrame(products)

def generate_orders(customers, products, n_orders=3000):
    order_ids = [f'O{i:05d}' for i in range(1, n_orders+1)]
    customer_ids = customers['customer_id'].tolist()
    product_ids = products['product_id'].tolist()
    payment_methods = ['支付宝', '微信', '银行卡', '花呗']
    order_statuses = ['已完成', '已取消', '退款中']
    
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    orders = []
    for oid in order_ids:
        cid = random.choice(customer_ids)
        n_items = random.randint(1, 5)
        order_date = start_date + timedelta(
            seconds=random.randint(0, int((end_date - start_date).total_seconds()))
        )
        
        for _ in range(n_items):
            pid = random.choice(product_ids)
            quantity = random.randint(1, 3)
            product_price = products.loc[products['product_id'] == pid, 'price'].iloc[0]
            
            orders.append({
                'order_id': oid,
                'customer_id': cid,
                'product_id': pid,
                'order_date': order_date,
                'quantity': quantity,
                'unit_price': product_price,
                'total_amount': round(product_price * quantity, 2),
                'payment_method': np.random.choice(payment_methods, p=[0.4, 0.35, 0.15, 0.1]),
                'order_status': np.random.choice(order_statuses, p=[0.85, 0.1, 0.05])
            })
    
    return pd.DataFrame(orders)

if __name__ == '__main__':
    print('正在生成客户数据...')
    customers = generate_customers(500)
    customers.to_csv('/Users/bilei/work/LargeModelAnnotation/GBS/260410/GSB-dataAnalysis/da2/data/customers.csv', index=False, encoding='utf-8-sig')
    print(f'生成客户数据: {len(customers)} 条')
    
    print('正在生成商品数据...')
    products = generate_products(100)
    products.to_csv('/Users/bilei/work/LargeModelAnnotation/GBS/260410/GSB-dataAnalysis/da2/data/products.csv', index=False, encoding='utf-8-sig')
    print(f'生成商品数据: {len(products)} 条')
    
    print('正在生成订单数据...')
    orders = generate_orders(customers, products, 3000)
    orders.to_csv('/Users/bilei/work/LargeModelAnnotation/GBS/260410/GSB-dataAnalysis/da2/data/orders.csv', index=False, encoding='utf-8-sig')
    print(f'生成订单数据: {len(orders)} 条')
    
    print('数据生成完成！')
