import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    """数据预处理类：处理缺失值、异常值，构建RFM特征"""

    def __init__(self, df):
        self.raw_df = df.copy()
        self.processed_df = None
        self.rfm_df = None
        self.scaler = StandardScaler()

    def handle_missing_values(self):
        """处理缺失值"""
        df = self.raw_df.copy()

        # 统计缺失值
        missing_stats = df.isnull().sum()
        print("缺失值统计:")
        print(missing_stats[missing_stats > 0])

        # 类别缺失值填充为'未知'
        if df['category'].isnull().sum() > 0:
            df['category'].fillna('未知', inplace=True)

        # 金额缺失值填充为0（非购买行为）
        if df['amount'].isnull().sum() > 0:
            df['amount'].fillna(0, inplace=True)

        self.processed_df = df
        return df

    def handle_outliers(self, column='amount', method='iqr'):
        """处理异常值"""
        df = self.processed_df.copy()

        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
            print(f"\n异常值检测（{column}）:")
            print(f"  下界: {lower_bound:.2f}, 上界: {upper_bound:.2f}")
            print(f"  异常值数量: {len(outliers)}")

            # 使用上下界截断
            df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)

        self.processed_df = df
        return df

    def build_rfm_features(self):
        """构建RFM特征（Recency, Frequency, Monetary）"""
        df = self.processed_df.copy()

        # 计算参考日期（数据集最大日期）
        reference_date = df['timestamp'].max()

        # 按用户聚合
        rfm_data = []

        for user_id in df['user_id'].unique():
            user_df = df[df['user_id'] == user_id]

            # Recency: 最近一次购买距今天数
            purchase_df = user_df[user_df['behavior_type'] == '购买']
            if len(purchase_df) > 0:
                last_purchase = purchase_df['timestamp'].max()
                recency = (reference_date - last_purchase).days
            else:
                recency = 999  # 未购买用户

            # Frequency: 购买频次
            frequency = len(purchase_df)

            # Monetary: 购买总金额
            monetary = purchase_df['amount'].sum()

            # 额外特征
            total_behaviors = len(user_df)
            browse_count = len(user_df[user_df['behavior_type'] == '浏览'])
            cart_count = len(user_df[user_df['behavior_type'] == '加购'])
            favorite_count = len(user_df[user_df['behavior_type'] == '收藏'])

            # 转化率特征
            conversion_rate = frequency / browse_count if browse_count > 0 else 0
            cart_conversion = frequency / cart_count if cart_count > 0 else 0

            rfm_data.append({
                'user_id': user_id,
                'recency': recency,
                'frequency': frequency,
                'monetary': monetary,
                'total_behaviors': total_behaviors,
                'browse_count': browse_count,
                'cart_count': cart_count,
                'favorite_count': favorite_count,
                'conversion_rate': conversion_rate,
                'cart_conversion': cart_conversion
            })

        self.rfm_df = pd.DataFrame(rfm_data)

        # RFM评分（1-5分）
        self.rfm_df['R_score'] = pd.qcut(self.rfm_df['recency'].rank(method='first'), 
                                          5, labels=[5,4,3,2,1])
        self.rfm_df['F_score'] = pd.qcut(self.rfm_df['frequency'].rank(method='first'), 
                                          5, labels=[1,2,3,4,5])
        self.rfm_df['M_score'] = pd.qcut(self.rfm_df['monetary'].rank(method='first'), 
                                          5, labels=[1,2,3,4,5])

        # RFM综合得分
        self.rfm_df['RFM_score'] = (self.rfm_df['R_score'].astype(int) + 
                                     self.rfm_df['F_score'].astype(int) + 
                                     self.rfm_df['M_score'].astype(int))

        return self.rfm_df

    def get_user_segments(self):
        """基于RFM进行用户分群"""
        if self.rfm_df is None:
            self.build_rfm_features()

        def segment_user(row):
            r, f, m = int(row['R_score']), int(row['F_score']), int(row['M_score'])

            if r >= 4 and f >= 4 and m >= 4:
                return '重要价值客户'
            elif r >= 4 and f >= 4 and m < 4:
                return '重要发展客户'
            elif r >= 4 and f < 4 and m >= 4:
                return '重要保持客户'
            elif r >= 4 and f < 4 and m < 4:
                return '新客户'
            elif r < 4 and f >= 4 and m >= 4:
                return '重要挽留客户'
            elif r < 4 and f >= 4 and m < 4:
                return '一般客户'
            elif r < 4 and f < 4 and m >= 4:
                return '潜力客户'
            else:
                return '流失客户'

        self.rfm_df['segment'] = self.rfm_df.apply(segment_user, axis=1)

        segment_counts = self.rfm_df['segment'].value_counts()
        print("\n用户分群统计:")
        print(segment_counts)

        return self.rfm_df

    def prepare_modeling_data(self):
        """准备建模数据（复购预测）"""
        if self.rfm_df is None:
            self.build_rfm_features()

        # 定义复购用户（购买次数>=2）
        self.rfm_df['is_repurchase'] = (self.rfm_df['frequency'] >= 2).astype(int)

        # 特征列
        feature_cols = ['recency', 'monetary', 'total_behaviors', 
                       'browse_count', 'cart_count', 'favorite_count',
                       'conversion_rate', 'cart_conversion']

        X = self.rfm_df[feature_cols]
        y = self.rfm_df['is_repurchase']

        # 标准化
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

        return X_scaled, y, feature_cols

    def run_preprocessing(self):
        """运行完整预处理流程"""
        print("=" * 50)
        print("开始数据预处理...")
        print("=" * 50)

        # 1. 处理缺失值
        self.handle_missing_values()

        # 2. 处理异常值
        self.handle_outliers()

        # 3. 构建RFM特征
        self.build_rfm_features()

        # 4. 用户分群
        self.get_user_segments()

        print("\n预处理完成!")
        print(f"RFM数据集形状: {self.rfm_df.shape}")

        return self.processed_df, self.rfm_df


if __name__ == '__main__':
    # 测试代码
    from data_generator import ECommerceDataGenerator

    generator = ECommerceDataGenerator(n_users=1000, n_records=5000)
    df = generator.generate()

    preprocessor = DataPreprocessor(df)
    processed_df, rfm_df = preprocessor.run_preprocessing()

    print("\nRFM特征预览:")
    print(rfm_df.head())
