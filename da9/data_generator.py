import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# 获取项目根目录 (da9)
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'output')

def generate_datasets(n_orders=1000, n_users=200, n_products=100, random_seed=42):
    """
    生成模拟电商数据集
    """
    np.random.seed(random_seed)
    
    # 确保目录存在
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 生成用户数据
    regions = ['华东', '华北', '华南', '华中', '西南', '西北', '东北']
    user_ids = [f'user_{i:04d}' for i in range(1, n_users + 1)]
    
    users = pd.DataFrame({
        'user_id': user_ids,
        'region': np.random.choice(regions, n_users, p=[0.25, 0.20, 0.18, 0.15, 0.10, 0.07, 0.05]),
        'register_date': pd.to_datetime([
            (datetime.now() - timedelta(days=np.random.randint(30, 730))).strftime('%Y-%m-%d')
            for _ in range(n_users)
        ]),
        'age': np.random.randint(18, 65, n_users),
        'gender': np.random.choice(['男', '女'], n_users, p=[0.55, 0.45])
    })
    
    # 生成产品数据
    categories = ['电子产品', '服装服饰', '家居用品', '图书文具', '运动户外', '食品饮料', '美妆个护']
    product_ids = [f'product_{i:04d}' for i in range(1, n_products + 1)]
    
    # 为每个类别设置价格范围
    category_price_ranges = {
        '电子产品': (100, 5000),
        '服装服饰': (50, 1000),
        '家居用品': (20, 2000),
        '图书文具': (10, 200),
        '运动户外': (50, 3000),
        '食品饮料': (5, 500),
        '美妆个护': (20, 1000)
    }
    
    products = pd.DataFrame({
        'product_id': product_ids,
        'category': np.random.choice(categories, n_products, p=[0.18, 0.22, 0.15, 0.10, 0.12, 0.13, 0.10]),
        'product_name': [f'产品{i}' for i in range(1, n_products + 1)],
        'brand': np.random.choice(['品牌A', '品牌B', '品牌C', '品牌D', '品牌E'], n_products)
    })
    
    # 为每个产品设置价格
    products['unit_price'] = products.apply(
        lambda row: np.random.uniform(*category_price_ranges[row['category']]),
        axis=1
    ).round(2)
    
    # 生成订单数据
    order_ids = [f'order_{i:06d}' for i in range(1, n_orders + 1)]
    
    # 生成日期范围（过去12个月）
    start_date = datetime.now() - timedelta(days=365)
    dates = [
        (start_date + timedelta(days=np.random.randint(0, 365))).strftime('%Y-%m-%d')
        for _ in range(n_orders)
    ]
    
    # 生成订单主表
    orders = pd.DataFrame({
        'order_id': order_ids,
        'user_id': np.random.choice(user_ids, n_orders),
        'order_date': pd.to_datetime(dates),
        'order_status': np.random.choice(['已完成', '已取消', '退款中'], n_orders, p=[0.85, 0.10, 0.05]),
        'payment_method': np.random.choice(['支付宝', '微信支付', '信用卡', '货到付款'], n_orders)
    })
    
    # 生成订单详情（每个订单1-5个产品）
    order_details = []
    for order_id in order_ids:
        n_products_per_order = np.random.randint(1, 6)
        for _ in range(n_products_per_order):
            product_id = np.random.choice(product_ids)
            quantity = np.random.randint(1, 4)  # 每个产品购买1-3件
            # 获取产品价格
            unit_price = products.loc[products['product_id'] == product_id, 'unit_price'].iloc[0]
            # 随机折扣（0.7-1.0倍）
            discount = np.random.uniform(0.7, 1.0)
            line_amount = round(unit_price * quantity * discount, 2)
            
            order_details.append({
                'order_id': order_id,
                'product_id': product_id,
                'quantity': quantity,
                'unit_price': unit_price,
                'discount': round(discount, 2),
                'line_amount': line_amount
            })
    
    order_details = pd.DataFrame(order_details)
    
    # 为订单主表添加总金额
    order_totals = order_details.groupby('order_id')['line_amount'].sum().reset_index()
    order_totals.columns = ['order_id', 'total_amount']
    orders = orders.merge(order_totals, on='order_id', how='left')
    
    # 保存数据到CSV
    users.to_csv(os.path.join(DATA_DIR, 'users.csv'), index=False)
    products.to_csv(os.path.join(DATA_DIR, 'products.csv'), index=False)
    orders.to_csv(os.path.join(DATA_DIR, 'orders.csv'), index=False)
    order_details.to_csv(os.path.join(DATA_DIR, 'order_details.csv'), index=False)
    
    print(f"数据生成完成:")
    print(f"  - 用户数: {len(users)}")
    print(f"  - 产品数: {len(products)}")
    print(f"  - 订单数: {len(orders)}")
    print(f"  - 订单详情数: {len(order_details)}")
    
    return users, products, orders, order_details


def load_datasets():
    """
    加载已生成的数据集
    """
    # 确保数据目录存在
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 检查是否已有数据文件
    data_files = ['users.csv', 'products.csv', 'orders.csv', 'order_details.csv']
    all_exist = all(os.path.exists(os.path.join(DATA_DIR, f)) for f in data_files)
    
    if not all_exist:
        print("未找到数据文件，开始生成...")
        return generate_datasets()
    
    # 加载数据
    users = pd.read_csv(os.path.join(DATA_DIR, 'users.csv'))
    products = pd.read_csv(os.path.join(DATA_DIR, 'products.csv'))
    orders = pd.read_csv(os.path.join(DATA_DIR, 'orders.csv'))
    order_details = pd.read_csv(os.path.join(DATA_DIR, 'order_details.csv'))
    
    # 转换日期列
    users['register_date'] = pd.to_datetime(users['register_date'])
    orders['order_date'] = pd.to_datetime(orders['order_date'])
    
    print(f"数据加载完成:")
    print(f"  - 用户数: {len(users)}")
    print(f"  - 产品数: {len(products)}")
    print(f"  - 订单数: {len(orders)}")
    print(f"  - 订单详情数: {len(order_details)}")
    
    return users, products, orders, order_details


if __name__ == '__main__':
    # 确保目录存在
    import os
    os.makedirs('da9/data', exist_ok=True)
    os.makedirs('da9/output', exist_ok=True)
    
    generate_datasets(n_orders=1000)
