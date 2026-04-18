"""
数据分析模块
包含探索性分析、时间序列分析、RFM分群、聚类分析、销售预测
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class ExploratoryAnalyzer:
    """探索性数据分析器"""
    
    def __init__(self, df: pd.DataFrame):
        """
        初始化探索性分析器
        
        Args:
            df: 待分析的DataFrame
        """
        self.df = df.copy()
    
    def descriptive_statistics(self) -> Dict[str, Any]:
        """
        计算描述性统计量
        
        Returns:
            包含描述统计的字典
        """
        try:
            stats = {}
            
            numeric_cols = ['amount', 'total_amount', 'quantity', 'rating']
            for col in numeric_cols:
                if col in self.df.columns:
                    stats[col] = {
                        'count': int(self.df[col].count()),
                        'mean': float(self.df[col].mean()),
                        'std': float(self.df[col].std()),
                        'min': float(self.df[col].min()),
                        '25%': float(self.df[col].quantile(0.25)),
                        '50%': float(self.df[col].quantile(0.5)),
                        '75%': float(self.df[col].quantile(0.75)),
                        'max': float(self.df[col].max())
                    }
            
            if 'category' in self.df.columns:
                stats['category_distribution'] = self.df['category'].value_counts().to_dict()
            
            if 'region' in self.df.columns:
                stats['region_distribution'] = self.df['region'].value_counts().to_dict()
            
            if 'payment_method' in self.df.columns:
                stats['payment_distribution'] = self.df['payment_method'].value_counts().to_dict()
            
            return stats
            
        except Exception as e:
            raise RuntimeError(f"计算描述统计失败: {str(e)}")
    
    def correlation_analysis(self) -> pd.DataFrame:
        """
        计算相关性矩阵
        
        Returns:
            相关性矩阵DataFrame
        """
        try:
            numeric_cols = ['amount', 'total_amount', 'quantity', 'rating']
            available_cols = [col for col in numeric_cols if col in self.df.columns]
            
            if len(available_cols) < 2:
                raise ValueError("数值列不足，无法计算相关性")
            
            corr_matrix = self.df[available_cols].corr()
            return corr_matrix
            
        except Exception as e:
            raise RuntimeError(f"计算相关性失败: {str(e)}")
    
    def get_analysis_summary(self) -> str:
        """
        生成分析摘要文本
        
        Returns:
            分析摘要字符串
        """
        try:
            stats = self.descriptive_statistics()
            
            summary = []
            summary.append("=" * 60)
            summary.append("探索性数据分析报告")
            summary.append("=" * 60)
            
            summary.append("\n【基础统计】")
            summary.append(f"总订单数: {len(self.df)}")
            summary.append(f"唯一用户数: {self.df['user_id'].nunique()}")
            summary.append(f"商品类别数: {self.df['category'].nunique()}")
            
            if 'order_time' in self.df.columns:
                summary.append(f"时间范围: {self.df['order_time'].min().strftime('%Y-%m-%d')} 至 {self.df['order_time'].max().strftime('%Y-%m-%d')}")
            
            summary.append("\n【金额统计】")
            if 'total_amount' in stats:
                summary.append(f"订单总金额: ¥{stats['total_amount']['sum'] if 'sum' in stats['total_amount'] else stats['total_amount']['mean'] * stats['total_amount']['count']:.2f}")
                summary.append(f"平均订单金额: ¥{stats['total_amount']['mean']:.2f}")
                summary.append(f"金额标准差: ¥{stats['total_amount']['std']:.2f}")
                summary.append(f"最小订单金额: ¥{stats['total_amount']['min']:.2f}")
                summary.append(f"最大订单金额: ¥{stats['total_amount']['max']:.2f}")
            
            summary.append("\n【评分统计】")
            if 'rating' in stats:
                summary.append(f"平均评分: {stats['rating']['mean']:.2f}")
                summary.append(f"评分中位数: {stats['rating']['50%']:.2f}")
            
            summary.append("\n【类别分布TOP5】")
            if 'category_distribution' in stats:
                sorted_cats = sorted(stats['category_distribution'].items(), key=lambda x: x[1], reverse=True)[:5]
                for cat, count in sorted_cats:
                    summary.append(f"  {cat}: {count}单")
            
            summary.append("\n【地区分布TOP5】")
            if 'region_distribution' in stats:
                sorted_regions = sorted(stats['region_distribution'].items(), key=lambda x: x[1], reverse=True)[:5]
                for region, count in sorted_regions:
                    summary.append(f"  {region}: {count}单")
            
            return "\n".join(summary)
            
        except Exception as e:
            raise RuntimeError(f"生成分析摘要失败: {str(e)}")


class TimeSeriesAnalyzer:
    """时间序列分析器"""
    
    def __init__(self, df: pd.DataFrame):
        """
        初始化时间序列分析器
        
        Args:
            df: 包含时间列的DataFrame
        """
        self.df = df.copy()
        if 'order_time' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['order_time']).dt.date
    
    def daily_sales_trend(self) -> pd.DataFrame:
        """
        计算每日销售趋势
        
        Returns:
            每日销售数据DataFrame
        """
        try:
            daily_sales = self.df.groupby('date').agg({
                'total_amount': 'sum',
                'order_id': 'count',
                'quantity': 'sum'
            }).reset_index()
            daily_sales.columns = ['date', 'daily_sales', 'order_count', 'total_quantity']
            daily_sales['date'] = pd.to_datetime(daily_sales['date'])
            daily_sales = daily_sales.sort_values('date')
            
            return daily_sales
            
        except Exception as e:
            raise RuntimeError(f"计算每日销售趋势失败: {str(e)}")
    
    def moving_average(self, window: int = 7) -> pd.DataFrame:
        """
        计算移动平均
        
        Args:
            window: 移动窗口大小
            
        Returns:
            包含移动平均的DataFrame
        """
        try:
            daily_sales = self.daily_sales_trend()
            daily_sales['ma_7'] = daily_sales['daily_sales'].rolling(window=window).mean()
            daily_sales['ma_14'] = daily_sales['daily_sales'].rolling(window=window*2).mean()
            daily_sales['ma_30'] = daily_sales['daily_sales'].rolling(window=30).mean()
            
            return daily_sales
            
        except Exception as e:
            raise RuntimeError(f"计算移动平均失败: {str(e)}")
    
    def monthly_sales(self) -> pd.DataFrame:
        """
        计算月度销售统计
        
        Returns:
            月度销售数据DataFrame
        """
        try:
            self.df['month'] = pd.to_datetime(self.df['order_time']).dt.to_period('M')
            monthly = self.df.groupby('month').agg({
                'total_amount': ['sum', 'mean', 'count'],
                'quantity': 'sum'
            }).reset_index()
            monthly.columns = ['month', 'total_sales', 'avg_sales', 'order_count', 'total_quantity']
            monthly['month'] = monthly['month'].astype(str)
            
            return monthly
            
        except Exception as e:
            raise RuntimeError(f"计算月度销售失败: {str(e)}")
    
    def weekly_pattern(self) -> pd.DataFrame:
        """
        分析周内销售模式
        
        Returns:
            周内销售模式DataFrame
        """
        try:
            self.df['weekday'] = pd.to_datetime(self.df['order_time']).dt.dayofweek
            weekday_names = {0: '周一', 1: '周二', 2: '周三', 3: '周四', 4: '周五', 5: '周六', 6: '周日'}
            
            weekly = self.df.groupby('weekday').agg({
                'total_amount': ['sum', 'mean', 'count']
            }).reset_index()
            weekly.columns = ['weekday', 'total_sales', 'avg_sales', 'order_count']
            weekly['weekday_name'] = weekly['weekday'].map(weekday_names)
            
            return weekly
            
        except Exception as e:
            raise RuntimeError(f"分析周内模式失败: {str(e)}")


class RFMAnalyzer:
    """RFM分析器"""
    
    def __init__(self, df: pd.DataFrame):
        """
        初始化RFM分析器
        
        Args:
            df: 包含用户交易数据的DataFrame
        """
        self.df = df.copy()
        self.rfm_df = None
    
    def calculate_rfm(self, reference_date: datetime = None) -> pd.DataFrame:
        """
        计算RFM指标
        
        Args:
            reference_date: 参考日期，默认为数据中最后日期+1天
            
        Returns:
            RFM指标DataFrame
        """
        try:
            if reference_date is None:
                reference_date = pd.to_datetime(self.df['order_time']).max() + timedelta(days=1)
            
            rfm = self.df.groupby('user_id').agg({
                'order_time': lambda x: (reference_date - pd.to_datetime(x).max()).days,
                'order_id': 'count',
                'total_amount': 'sum'
            }).reset_index()
            
            rfm.columns = ['user_id', 'recency', 'frequency', 'monetary']
            
            rfm['r_score'] = pd.qcut(rfm['recency'], q=5, labels=[5, 4, 3, 2, 1], duplicates='drop')
            rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
            rfm['m_score'] = pd.qcut(rfm['monetary'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
            
            rfm['rfm_score'] = rfm['r_score'].astype(str) + rfm['f_score'].astype(str) + rfm['m_score'].astype(str)
            rfm['rfm_total'] = rfm['r_score'].astype(int) + rfm['f_score'].astype(int) + rfm['m_score'].astype(int)
            
            self.rfm_df = rfm
            return rfm
            
        except Exception as e:
            raise RuntimeError(f"计算RFM失败: {str(e)}")
    
    def segment_customers(self) -> pd.DataFrame:
        """
        根据RFM分数对客户分层
        
        Returns:
            包含客户分层的DataFrame
        """
        try:
            if self.rfm_df is None:
                self.calculate_rfm()
            
            def segment_user(row):
                r, f, m = int(row['r_score']), int(row['f_score']), int(row['m_score'])
                
                if r >= 4 and f >= 4 and m >= 4:
                    return '重要价值客户'
                elif r >= 4 and f >= 3:
                    return '重要发展客户'
                elif r >= 3 and m >= 4:
                    return '重要保持客户'
                elif r >= 4 and f <= 2:
                    return '新客户'
                elif r <= 2 and f >= 4 and m >= 4:
                    return '重要挽留客户'
                elif r <= 2 and f <= 2 and m <= 2:
                    return '流失客户'
                else:
                    return '一般客户'
            
            self.rfm_df['segment'] = self.rfm_df.apply(segment_user, axis=1)
            
            return self.rfm_df
            
        except Exception as e:
            raise RuntimeError(f"客户分层失败: {str(e)}")
    
    def get_segment_summary(self) -> Dict[str, Any]:
        """
        获取客户分层摘要
        
        Returns:
            分层摘要字典
        """
        try:
            if 'segment' not in self.rfm_df.columns:
                self.segment_customers()
            
            summary = self.rfm_df.groupby('segment').agg({
                'user_id': 'count',
                'recency': 'mean',
                'frequency': 'mean',
                'monetary': ['mean', 'sum']
            }).round(2)
            
            summary.columns = ['用户数', '平均R(天)', '平均F(次)', '平均M(元)', '总M(元)']
            summary['占比(%)'] = (summary['用户数'] / summary['用户数'].sum() * 100).round(2)
            
            return summary.to_dict()
            
        except Exception as e:
            raise RuntimeError(f"获取分层摘要失败: {str(e)}")


class CustomerClusterAnalyzer:
    """客户聚类分析器"""
    
    def __init__(self, df: pd.DataFrame):
        """
        初始化聚类分析器
        
        Args:
            df: 包含用户交易数据的DataFrame
        """
        self.df = df.copy()
        self.cluster_df = None
        self.model = None
        self.scaler = StandardScaler()
    
    def prepare_features(self) -> pd.DataFrame:
        """
        准备聚类特征
        
        Returns:
            特征DataFrame
        """
        try:
            features = self.df.groupby('user_id').agg({
                'total_amount': ['sum', 'mean', 'std'],
                'order_id': 'count',
                'quantity': ['sum', 'mean'],
                'rating': 'mean',
                'category': 'nunique'
            }).reset_index()
            
            features.columns = [
                'user_id', 'total_spent', 'avg_order_value', 'std_order_value',
                'order_count', 'total_quantity', 'avg_quantity',
                'avg_rating', 'category_diversity'
            ]
            
            features['std_order_value'] = features['std_order_value'].fillna(0)
            
            self.cluster_df = features
            return features
            
        except Exception as e:
            raise RuntimeError(f"准备特征失败: {str(e)}")
    
    def perform_clustering(self, n_clusters: int = 4) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        执行KMeans聚类
        
        Args:
            n_clusters: 聚类数量
            
        Returns:
            (聚类结果DataFrame, 聚类指标字典)
        """
        try:
            if self.cluster_df is None:
                self.prepare_features()
            
            feature_cols = ['total_spent', 'avg_order_value', 'order_count', 
                           'total_quantity', 'avg_rating', 'category_diversity']
            
            X = self.cluster_df[feature_cols].values
            X_scaled = self.scaler.fit_transform(X)
            
            self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            self.cluster_df['cluster'] = self.model.fit_predict(X_scaled)
            
            metrics = {
                'n_clusters': n_clusters,
                'inertia': float(self.model.inertia_),
                'cluster_sizes': self.cluster_df['cluster'].value_counts().to_dict()
            }
            
            return self.cluster_df, metrics
            
        except Exception as e:
            raise RuntimeError(f"执行聚类失败: {str(e)}")
    
    def get_cluster_profile(self) -> pd.DataFrame:
        """
        获取聚类画像
        
        Returns:
            聚类画像DataFrame
        """
        try:
            if 'cluster' not in self.cluster_df.columns:
                raise ValueError("请先执行聚类分析")
            
            profile = self.cluster_df.groupby('cluster').agg({
                'total_spent': 'mean',
                'avg_order_value': 'mean',
                'order_count': 'mean',
                'total_quantity': 'mean',
                'avg_rating': 'mean',
                'category_diversity': 'mean',
                'user_id': 'count'
            }).round(2)
            
            profile.columns = ['平均消费总额', '平均订单金额', '平均订单数', 
                              '平均购买数量', '平均评分', '类别多样性', '用户数']
            
            def label_cluster(row):
                if row['平均消费总额'] > profile['平均消费总额'].quantile(0.75):
                    return '高价值用户'
                elif row['平均订单数'] > profile['平均订单数'].quantile(0.75):
                    return '高频用户'
                elif row['平均评分'] > profile['平均评分'].quantile(0.75):
                    return '高满意度用户'
                else:
                    return '普通用户'
            
            profile['用户标签'] = profile.apply(label_cluster, axis=1)
            
            return profile
            
        except Exception as e:
            raise RuntimeError(f"获取聚类画像失败: {str(e)}")


class SalesPredictor:
    """销售预测器"""
    
    def __init__(self, df: pd.DataFrame):
        """
        初始化销售预测器
        
        Args:
            df: 包含销售数据的DataFrame
        """
        self.df = df.copy()
        self.model = None
        self.scaler = StandardScaler()
    
    def prepare_prediction_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备预测数据
        
        Returns:
            (特征矩阵, 目标向量)
        """
        try:
            daily = self.df.groupby(pd.to_datetime(self.df['order_time']).dt.date).agg({
                'total_amount': 'sum',
                'order_id': 'count',
                'quantity': 'sum',
                'rating': 'mean'
            }).reset_index()
            
            daily.columns = ['date', 'sales', 'orders', 'quantity', 'avg_rating']
            daily['date'] = pd.to_datetime(daily['date'])
            
            daily['day_of_week'] = daily['date'].dt.dayofweek
            daily['day_of_month'] = daily['date'].dt.day
            daily['is_weekend'] = (daily['day_of_week'] >= 5).astype(int)
            
            daily['sales_lag_1'] = daily['sales'].shift(1)
            daily['sales_lag_7'] = daily['sales'].shift(7)
            daily['sales_ma_7'] = daily['sales'].rolling(7).mean()
            
            daily = daily.dropna()
            
            feature_cols = ['orders', 'quantity', 'avg_rating', 'day_of_week', 
                           'day_of_month', 'is_weekend', 'sales_lag_1', 'sales_lag_7', 'sales_ma_7']
            
            X = daily[feature_cols].values
            y = daily['sales'].values
            
            return X, y
            
        except Exception as e:
            raise RuntimeError(f"准备预测数据失败: {str(e)}")
    
    def train_model(self) -> Dict[str, float]:
        """
        训练线性回归模型
        
        Returns:
            模型评估指标字典
        """
        try:
            X, y = self.prepare_prediction_data()
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            self.model = LinearRegression()
            self.model.fit(X_train_scaled, y_train)
            
            y_pred = self.model.predict(X_test_scaled)
            
            metrics = {
                'r2_score': float(r2_score(y_test, y_pred)),
                'mse': float(mean_squared_error(y_test, y_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
                'mae': float(mean_absolute_error(y_test, y_pred))
            }
            
            return metrics
            
        except Exception as e:
            raise RuntimeError(f"训练模型失败: {str(e)}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        获取特征重要性
        
        Returns:
            特征重要性字典
        """
        try:
            if self.model is None:
                raise ValueError("请先训练模型")
            
            feature_names = ['orders', 'quantity', 'avg_rating', 'day_of_week', 
                           'day_of_month', 'is_weekend', 'sales_lag_1', 'sales_lag_7', 'sales_ma_7']
            
            importance = dict(zip(feature_names, np.abs(self.model.coef_)))
            
            return importance
            
        except Exception as e:
            raise RuntimeError(f"获取特征重要性失败: {str(e)}")


if __name__ == '__main__':
    from data_generator import DataGenerator
    
    generator = DataGenerator()
    df = generator.generate_data(n_records=600)
    
    print("探索性分析:")
    explorer = ExploratoryAnalyzer(df)
    print(explorer.get_analysis_summary())
    
    print("\n时间序列分析:")
    ts_analyzer = TimeSeriesAnalyzer(df)
    print(ts_analyzer.moving_average().head())
    
    print("\nRFM分析:")
    rfm = RFMAnalyzer(df)
    rfm_df = rfm.calculate_rfm()
    rfm_segmented = rfm.segment_customers()
    print(rfm_segmented[['user_id', 'recency', 'frequency', 'monetary', 'segment']].head())
    
    print("\n聚类分析:")
    cluster_analyzer = CustomerClusterAnalyzer(df)
    cluster_df, metrics = cluster_analyzer.perform_clustering()
    print(cluster_analyzer.get_cluster_profile())
    
    print("\n销售预测:")
    predictor = SalesPredictor(df)
    metrics = predictor.train_model()
    print(f"模型指标: {metrics}")
