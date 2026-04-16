import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_ecommerce_data(n_records=2000, start_date='2023-01-01', end_date='2023-04-30'):
    """
    生成模拟电商销售数据
    
    参数:
        n_records: 记录数
        start_date: 开始日期
        end_date: 结束日期
    
    返回:
        DataFrame: 包含销售数据的DataFrame
    """
    np.random.seed(42)
    
    n_users = int(n_records * 0.3)
    user_ids = [f'USER_{str(i).zfill(4)}' for i in range(1, n_users + 1)]
    
    categories = ['电子产品', '服装鞋帽', '食品饮料', '家居用品', '美妆护肤', '运动户外', '图书音像']
    category_weights = [0.25, 0.20, 0.15, 0.12, 0.12, 0.10, 0.06]
    
    category_price_range = {
        '电子产品': (100, 5000),
        '服装鞋帽': (50, 800),
        '食品饮料': (10, 200),
        '家居用品': (30, 1000),
        '美妆护肤': (20, 500),
        '运动户外': (50, 2000),
        '图书音像': (20, 200)
    }
    
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    date_range = (end - start).days
    
    data = []
    for _ in range(n_records):
        user_id = np.random.choice(user_ids)
        
        random_days = np.random.randint(0, date_range + 1)
        order_date = start + timedelta(days=random_days)
        order_date = order_date.replace(
            hour=np.random.randint(9, 22),
            minute=np.random.randint(0, 60),
            second=np.random.randint(0, 60)
        )
        
        category = np.random.choice(categories, p=category_weights)
        price_min, price_max = category_price_range[category]
        amount = round(np.random.uniform(price_min, price_max), 2)
        
        rating = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.08, 0.22, 0.35, 0.30])
        
        quantity = np.random.randint(1, 6)
        total_amount = round(amount * quantity, 2)
        
        payment_method = np.random.choice(['支付宝', '微信支付', '银行卡'], p=[0.5, 0.4, 0.1])
        
        data.append({
            '用户ID': user_id,
            '订单时间': order_date,
            '商品类别': category,
            '单价': amount,
            '数量': quantity,
            '订单金额': total_amount,
            '评分': rating,
            '支付方式': payment_method
        })
    
    df = pd.DataFrame(data)
    
    df.loc[np.random.choice(df.index, size=20), '订单金额'] = np.nan
    df.loc[np.random.choice(df.index, size=15), '评分'] = np.nan
    df.loc[np.random.choice(df.index, size=10), '订单金额'] = df.loc[np.random.choice(df.index, size=10), '订单金额'] * 10
    
    df = pd.concat([df, df.sample(n=15)], ignore_index=True)
    
    return df

def save_data(df, output_path='../data/raw_ecommerce_data.csv'):
    """
    保存数据到CSV文件
    
    参数:
        df: DataFrame
        output_path: 输出路径
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"数据已保存至: {output_path}")
    print(f"数据形状: {df.shape}")

def load_data(file_path):
    """
    从CSV文件加载数据
    
    参数:
        file_path: 文件路径
    
    返回:
        DataFrame
    """
    if os.path.exists(file_path):
        return pd.read_csv(file_path, parse_dates=['订单时间'])
    else:
        raise FileNotFoundError(f"文件不存在: {file_path}")

if __name__ == '__main__':
    df = generate_ecommerce_data(n_records=2000)
    save_data(df, '../data/raw_ecommerce_data.csv')
    print("\n数据预览:")
    print(df.head())
    print("\n数据统计:")
    print(df.describe(include='all'))
