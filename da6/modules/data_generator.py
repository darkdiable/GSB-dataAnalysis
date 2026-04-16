import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random


class ECommerceDataGenerator:
    """电商用户行为数据生成器"""

    def __init__(self, n_users=5000, n_records=50000, random_seed=42):
        self.n_users = n_users
        self.n_records = n_records
        self.random_seed = random_seed
        np.random.seed(random_seed)
        random.seed(random_seed)

        # 商品类别
        self.categories = ['电子产品', '服装', '食品', '家居', '美妆', '运动', '图书', '母婴']

        # 行为类型权重
        self.behavior_weights = {
            '浏览': 0.60,
            '收藏': 0.15,
            '加购': 0.15,
            '购买': 0.10
        }

    def _generate_user_ids(self):
        """生成用户ID列表"""
        return [f'U{str(i).zfill(6)}' for i in range(1, self.n_users + 1)]

    def _generate_timestamps(self, n):
        """生成时间戳（最近6个月）"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)

        timestamps = []
        for _ in range(n):
            random_days = random.randint(0, 180)
            random_hours = random.randint(0, 23)
            random_minutes = random.randint(0, 59)
            random_seconds = random.randint(0, 59)

            ts = start_date + timedelta(
                days=random_days,
                hours=random_hours,
                minutes=random_minutes,
                seconds=random_seconds
            )
            timestamps.append(ts)

        return timestamps

    def _generate_behavior_data(self):
        """生成行为数据"""
        user_ids = self._generate_user_ids()

        data = {
            'user_id': [],
            'timestamp': [],
            'behavior_type': [],
            'category': [],
            'product_id': [],
            'amount': []
        }

        timestamps = self._generate_timestamps(self.n_records)
        timestamps.sort()

        behaviors = list(self.behavior_weights.keys())
        behavior_probs = list(self.behavior_weights.values())

        for i in range(self.n_records):
            user_id = random.choice(user_ids)
            behavior = np.random.choice(behaviors, p=behavior_probs)
            category = random.choice(self.categories)
            product_id = f'P{category[:2]}{random.randint(1000, 9999)}'

            # 购买金额（仅购买行为有）
            if behavior == '购买':
                amount = round(random.uniform(20, 2000), 2)
            elif behavior == '加购':
                amount = round(random.uniform(20, 2000), 2)
            else:
                amount = 0.0

            data['user_id'].append(user_id)
            data['timestamp'].append(timestamps[i])
            data['behavior_type'].append(behavior)
            data['category'].append(category)
            data['product_id'].append(product_id)
            data['amount'].append(amount)

        df = pd.DataFrame(data)

        # 引入一些缺失值（模拟真实数据）
        missing_indices = np.random.choice(df.index, size=int(len(df) * 0.02), replace=False)
        df.loc[missing_indices[:len(missing_indices)//2], 'category'] = np.nan
        df.loc[missing_indices[len(missing_indices)//2:], 'amount'] = np.nan

        # 引入一些异常值
        outlier_indices = np.random.choice(df.index, size=int(len(df) * 0.01), replace=False)
        df.loc[outlier_indices, 'amount'] = df.loc[outlier_indices, 'amount'] * 100

        return df

    def generate(self, save_path=None):
        """生成数据集并可选保存"""
        df = self._generate_behavior_data()

        if save_path:
            df.to_csv(save_path, index=False, encoding='utf-8-sig')
            print(f"数据集已保存至: {save_path}")

        return df


if __name__ == '__main__':
    generator = ECommerceDataGenerator(n_users=5000, n_records=50000)
    df = generator.generate(save_path='../data/ecommerce_behavior.csv')
    print(f"数据集形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")
    print(f"前5行:\n{df.head()}")
