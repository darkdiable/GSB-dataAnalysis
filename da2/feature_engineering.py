"""
特征工程模块
提供数据预处理和特征构建功能
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """特征工程类"""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.features = {}

    def create_time_features(self) -> pd.DataFrame:
        """创建时间相关特征"""
        try:
            if 'date' in self.df.columns:
                self.df['date'] = pd.to_datetime(self.df['date'])
                self.df['year'] = self.df['date'].dt.year
                self.df['month'] = self.df['date'].dt.month
                self.df['day'] = self.df['date'].dt.day
                self.df['quarter'] = self.df['date'].dt.quarter
                self.df['weekday'] = self.df['date'].dt.weekday
                self.df['is_weekend'] = self.df['weekday'].isin([5, 6]).astype(int)
                self.df['day_of_year'] = self.df['date'].dt.dayofyear
                self.df['week_of_year'] = self.df['date'].dt.isocalendar().week
                self.df['year_month'] = self.df['date'].dt.to_period('M').astype(str)

                # 节假日标记（简化版）
                self.df['is_month_start'] = self.df['date'].dt.is_month_start.astype(int)
                self.df['is_month_end'] = self.df['date'].dt.is_month_end.astype(int)

                logger.info("时间特征创建完成")
        except Exception as e:
            logger.error(f"创建时间特征时出错: {e}")

        return self.df

    def create_customer_features(self) -> pd.DataFrame:
        """创建客户相关特征"""
        try:
            if 'customer_id' in self.df.columns:
                # 客户交易频次
                customer_freq = self.df.groupby('customer_id').size().reset_index(name='customer_transaction_count')
                self.df = self.df.merge(customer_freq, on='customer_id', how='left')

                # 客户总消费金额
                if 'total_amount' in self.df.columns:
                    customer_amount = self.df.groupby('customer_id')['total_amount'].sum().reset_index(name='customer_total_amount')
                    self.df = self.df.merge(customer_amount, on='customer_id', how='left')

                    # 客户平均订单金额
                    self.df['customer_avg_order'] = self.df['customer_total_amount'] / self.df['customer_transaction_count']

                # 客户首次和最近购买日期
                if 'date' in self.df.columns:
                    customer_dates = self.df.groupby('customer_id')['date'].agg(['min', 'max']).reset_index()
                    customer_dates.columns = ['customer_id', 'first_purchase', 'last_purchase']
                    self.df = self.df.merge(customer_dates, on='customer_id', how='left')

                    # 客户生命周期（天）
                    self.df['customer_lifetime_days'] = (self.df['last_purchase'] - self.df['first_purchase']).dt.days

                logger.info("客户特征创建完成")
        except Exception as e:
            logger.error(f"创建客户特征时出错: {e}")

        return self.df

    def create_product_features(self) -> pd.DataFrame:
        """创建商品相关特征"""
        try:
            if 'product_id' in self.df.columns:
                # 商品销量
                product_sales = self.df.groupby('product_id')['quantity'].sum().reset_index(name='product_total_sales')
                self.df = self.df.merge(product_sales, on='product_id', how='left')

                # 商品销售额
                if 'total_amount' in self.df.columns:
                    product_revenue = self.df.groupby('product_id')['total_amount'].sum().reset_index(name='product_total_revenue')
                    self.df = self.df.merge(product_revenue, on='product_id', how='left')

                # 商品交易次数
                product_transactions = self.df.groupby('product_id').size().reset_index(name='product_transaction_count')
                self.df = self.df.merge(product_transactions, on='product_id', how='left')

                # 商品平均单价
                if 'price' in self.df.columns:
                    product_avg_price = self.df.groupby('product_id')['price'].mean().reset_index(name='product_avg_price')
                    self.df = self.df.merge(product_avg_price, on='product_id', how='left')

                logger.info("商品特征创建完成")
        except Exception as e:
            logger.error(f"创建商品特征时出错: {e}")

        return self.df

    def create_rfm_features(self, reference_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """
        创建RFM特征（Recency, Frequency, Monetary）

        Args:
            reference_date: 参考日期，默认为数据中的最大日期
        """
        try:
            if 'customer_id' not in self.df.columns or 'date' not in self.df.columns or 'total_amount' not in self.df.columns:
                logger.warning("缺少必要的列来创建RFM特征")
                return self.df

            if reference_date is None:
                reference_date = self.df['date'].max()

            rfm = self.df.groupby('customer_id').agg({
                'date': lambda x: (reference_date - x.max()).days,  # Recency
                'transaction_id': 'count',  # Frequency
                'total_amount': 'sum'  # Monetary
            }).reset_index()

            rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']

            # RFM评分（1-5分）
            rfm['r_score'] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
            rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
            rfm['m_score'] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

            # RFM总分
            rfm['rfm_score'] = rfm['r_score'].astype(str) + rfm['f_score'].astype(str) + rfm['m_score'].astype(str)

            # 客户价值分层
            def segment_customer(row):
                if row['r_score'] >= 4 and row['f_score'] >= 4 and row['m_score'] >= 4:
                    return '重要价值客户'
                elif row['r_score'] >= 3 and row['f_score'] >= 3:
                    return '潜力客户'
                elif row['r_score'] >= 4:
                    return '新客户'
                elif row['f_score'] >= 4:
                    return '忠诚客户'
                else:
                    return '一般客户'

            rfm['customer_segment'] = rfm.apply(segment_customer, axis=1)

            self.df = self.df.merge(rfm, on='customer_id', how='left')
            self.features['rfm'] = rfm

            logger.info("RFM特征创建完成")
        except Exception as e:
            logger.error(f"创建RFM特征时出错: {e}")

        return self.df

    def create_category_features(self) -> pd.DataFrame:
        """创建品类相关特征"""
        try:
            if 'category' in self.df.columns:
                # 品类销售额
                if 'total_amount' in self.df.columns:
                    category_revenue = self.df.groupby('category')['total_amount'].sum().reset_index(name='category_total_revenue')
                    self.df = self.df.merge(category_revenue, on='category', how='left')

                # 品类销量
                if 'quantity' in self.df.columns:
                    category_sales = self.df.groupby('category')['quantity'].sum().reset_index(name='category_total_sales')
                    self.df = self.df.merge(category_sales, on='category', how='left')

                # 品类商品数
                category_products = self.df.groupby('category')['product_id'].nunique().reset_index(name='category_product_count')
                self.df = self.df.merge(category_products, on='category', how='left')

                logger.info("品类特征创建完成")
        except Exception as e:
            logger.error(f"创建品类特征时出错: {e}")

        return self.df

    def create_lag_features(self, group_col: str = 'product_id', target_col: str = 'quantity',
                           lags: List[int] = [1, 7, 30]) -> pd.DataFrame:
        """
        创建滞后特征

        Args:
            group_col: 分组列
            target_col: 目标列
            lags: 滞后天数列表
        """
        try:
            if 'date' not in self.df.columns:
                logger.warning("缺少date列，无法创建滞后特征")
                return self.df

            self.df = self.df.sort_values(['date', group_col])

            for lag in lags:
                lag_col = f'{target_col}_lag_{lag}'
                self.df[lag_col] = self.df.groupby(group_col)[target_col].shift(lag)

            logger.info(f"滞后特征创建完成: lags={lags}")
        except Exception as e:
            logger.error(f"创建滞后特征时出错: {e}")

        return self.df

    def create_rolling_features(self, group_col: str = 'product_id', target_col: str = 'quantity',
                                windows: List[int] = [7, 30, 90]) -> pd.DataFrame:
        """
        创建滚动统计特征

        Args:
            group_col: 分组列
            target_col: 目标列
            windows: 窗口大小列表
        """
        try:
            if 'date' not in self.df.columns:
                logger.warning("缺少date列，无法创建滚动特征")
                return self.df

            self.df = self.df.sort_values(['date', group_col])

            for window in windows:
                # 滚动平均
                self.df[f'{target_col}_rolling_mean_{window}'] = (
                    self.df.groupby(group_col)[target_col]
                    .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
                )

                # 滚动标准差
                self.df[f'{target_col}_rolling_std_{window}'] = (
                    self.df.groupby(group_col)[target_col]
                    .transform(lambda x: x.rolling(window=window, min_periods=1).std())
                )

                # 滚动最大值
                self.df[f'{target_col}_rolling_max_{window}'] = (
                    self.df.groupby(group_col)[target_col]
                    .transform(lambda x: x.rolling(window=window, min_periods=1).max())
                )

            logger.info(f"滚动特征创建完成: windows={windows}")
        except Exception as e:
            logger.error(f"创建滚动特征时出错: {e}")

        return self.df

    def handle_missing_values(self, strategy: str = 'fill', fill_value: float = 0) -> pd.DataFrame:
        """
        处理缺失值

        Args:
            strategy: 处理策略 ('fill', 'drop', 'interpolate')
            fill_value: 填充值
        """
        try:
            missing_before = self.df.isnull().sum().sum()

            if strategy == 'fill':
                # 数值列用fill_value填充，类别列用众数填充
                numeric_cols = self.df.select_dtypes(include=[np.number]).columns
                categorical_cols = self.df.select_dtypes(include=['object']).columns

                self.df[numeric_cols] = self.df[numeric_cols].fillna(fill_value)
                for col in categorical_cols:
                    if self.df[col].isnull().any():
                        mode_val = self.df[col].mode()
                        if len(mode_val) > 0:
                            self.df[col] = self.df[col].fillna(mode_val[0])

            elif strategy == 'drop':
                self.df = self.df.dropna()

            elif strategy == 'interpolate':
                numeric_cols = self.df.select_dtypes(include=[np.number]).columns
                self.df[numeric_cols] = self.df[numeric_cols].interpolate()
                self.df = self.df.fillna(fill_value)

            missing_after = self.df.isnull().sum().sum()
            logger.info(f"缺失值处理完成: 处理前{missing_before}个, 处理后{missing_after}个")

        except Exception as e:
            logger.error(f"处理缺失值时出错: {e}")

        return self.df

    def encode_categorical(self, columns: List[str], method: str = 'onehot') -> pd.DataFrame:
        """
        编码类别特征

        Args:
            columns: 要编码的列
            method: 编码方法 ('onehot', 'label')
        """
        try:
            for col in columns:
                if col not in self.df.columns:
                    continue

                if method == 'onehot':
                    dummies = pd.get_dummies(self.df[col], prefix=col, drop_first=True)
                    self.df = pd.concat([self.df, dummies], axis=1)

                elif method == 'label':
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    self.df[f'{col}_encoded'] = le.fit_transform(self.df[col].astype(str))

            logger.info(f"类别编码完成: columns={columns}, method={method}")
        except Exception as e:
            logger.error(f"编码类别特征时出错: {e}")

        return self.df

    def scale_features(self, columns: List[str], method: str = 'standard') -> Tuple[pd.DataFrame, Dict]:
        """
        特征缩放

        Args:
            columns: 要缩放的列
            method: 缩放方法 ('standard', 'minmax')

        Returns:
            (缩放后的DataFrame, 缩放器字典)
        """
        scalers = {}
        try:
            from sklearn.preprocessing import StandardScaler, MinMaxScaler

            for col in columns:
                if col not in self.df.columns:
                    continue

                if method == 'standard':
                    scaler = StandardScaler()
                elif method == 'minmax':
                    scaler = MinMaxScaler()
                else:
                    continue

                self.df[f'{col}_scaled'] = scaler.fit_transform(self.df[[col]])
                scalers[col] = scaler

            logger.info(f"特征缩放完成: columns={columns}, method={method}")
        except Exception as e:
            logger.error(f"特征缩放时出错: {e}")

        return self.df, scalers

    def get_feature_importance(self, target_col: str, feature_cols: List[str]) -> pd.DataFrame:
        """
        计算特征重要性

        Args:
            target_col: 目标列
            feature_cols: 特征列

        Returns:
            特征重要性DataFrame
        """
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.feature_selection import mutual_info_regression

            # 准备数据
            X = self.df[feature_cols].fillna(0)
            y = self.df[target_col]

            # 随机森林特征重要性
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)
            rf_importance = rf.feature_importances_

            # 互信息
            mi_importance = mutual_info_regression(X, y, random_state=42)

            importance_df = pd.DataFrame({
                'feature': feature_cols,
                'rf_importance': rf_importance,
                'mutual_info': mi_importance
            })
            importance_df = importance_df.sort_values('rf_importance', ascending=False)

            self.features['importance'] = importance_df
            logger.info("特征重要性计算完成")

            return importance_df
        except Exception as e:
            logger.error(f"计算特征重要性时出错: {e}")
            return pd.DataFrame()

    def build_all_features(self) -> pd.DataFrame:
        """构建所有特征"""
        logger.info("开始构建所有特征...")

        self.create_time_features()
        self.create_customer_features()
        self.create_product_features()
        self.create_rfm_features()
        self.create_category_features()
        self.handle_missing_values()

        logger.info(f"特征构建完成，当前共有{self.df.shape[1]}列")
        return self.df

    def get_features(self) -> Dict:
        """获取构建的特征"""
        return self.features
