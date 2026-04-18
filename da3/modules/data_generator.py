import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import random
import warnings
warnings.filterwarnings('ignore')


def generate_sales_data(n_records=1000, start_date='2024-01-01', days=120, output_path='data/raw_sales.csv'):
    """
    生成模拟电商销售数据
    
    参数:
        n_records: 生成数据条数
        start_date: 开始日期
        days: 时间跨度天数
        output_path: 输出CSV文件路径
    
    返回:
        DataFrame: 包含模拟销售数据的DataFrame
    """
    try:
        print("正在生成模拟销售数据...")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        n_users = int(n_records * 0.3)
        user_ids = [f'USER_{str(i).zfill(4)}' for i in range(1, n_users + 1)]
        
        categories = ['电子产品', '服装鞋帽', '食品饮料', '家居用品', '美妆护肤', '运动户外', '图书文具']
        category_price_range = {
            '电子产品': (200, 5000),
            '服装鞋帽': (50, 800),
            '食品饮料': (10, 200),
            '家居用品': (30, 500),
            '美妆护肤': (50, 600),
            '运动户外': (100, 2000),
            '图书文具': (10, 150)
        }
        
        category_weights = [0.2, 0.18, 0.15, 0.15, 0.12, 0.12, 0.08]
        
        start = datetime.strptime(start_date, '%Y-%m-%d')
        dates = []
        for _ in range(n_records):
            random_days = np.random.randint(0, days)
            random_hours = np.random.randint(9, 23)
            random_minutes = np.random.randint(0, 60)
            order_date = start + timedelta(days=random_days, hours=random_hours, minutes=random_minutes)
            dates.append(order_date)
        
        data = []
        for i in range(n_records):
            user_id = random.choice(user_ids)
            category = random.choices(categories, weights=category_weights)[0]
            price_min, price_max = category_price_range[category]
            
            base_price = np.random.uniform(price_min, price_max)
            quantity = np.random.randint(1, 5)
            amount = round(base_price * quantity, 2)
            
            rating = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.08, 0.15, 0.35, 0.37])
            
            if random.random() < 0.02:
                amount = np.nan
            if random.random() < 0.015:
                rating = np.nan
            if random.random() < 0.01:
                category = np.nan
            
            data.append({
                '订单ID': f'ORD_{str(i + 1).zfill(6)}',
                '用户ID': user_id,
                '下单时间': dates[i],
                '商品类别': category,
                '商品数量': quantity,
                '订单金额': amount,
                '用户评分': rating,
                '是否会员': random.choices(['是', '否'], weights=[0.4, 0.6])[0]
            })
        
        df = pd.DataFrame(data)
        
        for _ in range(int(n_records * 0.01)):
            dup_idx = random.randint(0, n_records - 1)
            df.loc[len(df)] = df.loc[dup_idx]
        
        outlier_idx = random.sample(range(n_records), 5)
        for idx in outlier_idx:
            df.loc[idx, '订单金额'] = df.loc[idx, '订单金额'] * 10
        
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"数据生成完成！共 {len(df)} 条记录，已保存至 {output_path}")
        print(f"时间跨度: {df['下单时间'].min()} 至 {df['下单时间'].max()}")
        
        return df
    
    except Exception as e:
        print(f"数据生成过程中发生错误: {str(e)}")
        raise


def load_data(file_path):
    """
    从CSV文件加载数据
    
    参数:
        file_path: CSV文件路径
    
    返回:
        DataFrame: 加载的数据
    """
    try:
        if not os.path.exists(file_path):
            print(f"文件 {file_path} 不存在，正在生成新数据...")
            return generate_sales_data(output_path=file_path)
        
        df = pd.read_csv(file_path, parse_dates=['下单时间'])
        print(f"数据加载成功！共 {len(df)} 条记录")
        return df
    
    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        raise


if __name__ == '__main__':
    generate_sales_data()
