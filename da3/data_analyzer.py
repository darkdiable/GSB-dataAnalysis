"""
数据分析模块
包含探索性分析、RFM分群、时间序列分析、聚类分析、销售预测等功能
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class DataAnalyzer:
    """数据分析类"""
    
    def __init__(self, df: pd.DataFrame):
        """
        初始化数据分析器
        
        Args:
            df: 待分析的DataFrame
        """
        self.df = df.copy()
        self.analysis_results = {}
        
    def descriptive_statistics(self) -> Dict[str, Any]:
        """
        描述性统计分析
        
        Returns:
            统计结果字典
        """
        print("\n" + "=" * 50)
        print("描述性统计分析")
        print("=" * 50)
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        stats = {
            '记录总数': len(self.df),
            '用户数量': self.df['user_id'].nunique() if 'user_id' in self.df.columns else 'N/A',
            '订单数量': self.df['order_id'].nunique() if 'order_id' in self.df.columns else 'N/A',
        }
        
        if 'amount' in self.df.columns:
            stats['金额统计'] = {
                '总销售额': f"{self.df['total_amount'].sum():,.2f}" if 'total_amount' in self.df.columns else 'N/A',
                '平均订单金额': f"{self.df['amount'].mean():.2f}",
                '金额标准差': f"{self.df['amount'].std():.2f}",
                '最小金额': f"{self.df['amount'].min():.2f}",
                '最大金额': f"{self.df['amount'].max():.2f}"
            }
        
        if 'rating' in self.df.columns:
            stats['评分统计'] = {
                '平均评分': f"{self.df['rating'].mean():.2f}",
                '评分分布': self.df['rating'].value_counts().sort_index().to_dict()
            }
        
        if 'category' in self.df.columns:
            stats['类别统计'] = {
                '类别数量': self.df['category'].nunique(),
                '各类别订单数': self.df['category'].value_counts().to_dict()
            }
        
        self.analysis_results['descriptive_stats'] = stats
        
        for key, value in stats.items():
            print(f"\n{key}:")
            if isinstance(value, dict):
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"  {value}")
        
        return stats
    
    def correlation_analysis(self) -> pd.DataFrame:
        """
        相关性分析
        
        Returns:
            相关系数矩阵
        """
        print("\n" + "=" * 50)
        print("相关性分析")
        print("=" * 50)
        
        numeric_cols = ['amount', 'quantity', 'rating', 'total_amount']
        available_cols = [col for col in numeric_cols if col in self.df.columns]
        
        if len(available_cols) < 2:
            print("数值列不足，无法进行相关性分析")
            return None
        
        corr_matrix = self.df[available_cols].corr()
        
        print("\n相关系数矩阵:")
        print(corr_matrix.round(3))
        
        self.analysis_results['correlation_matrix'] = corr_matrix
        
        return corr_matrix
    
    def rfm_analysis(self, reference_date: Optional[str] = None) -> pd.DataFrame:
        """
        RFM分析（Recency, Frequency, Monetary）
        
        Args:
            reference_date: 参考日期，默认为数据中最后日期+1天
            
        Returns:
            RFM分析结果DataFrame
        """
        print("\n" + "=" * 50)
        print("RFM用户价值分析")
        print("=" * 50)
        
        if 'order_date' not in self.df.columns or 'user_id' not in self.df.columns:
            print("缺少必要字段，无法进行RFM分析")
            return None
        
        if reference_date is None:
            reference_date = self.df['order_date'].max() + timedelta(days=1)
        else:
            reference_date = pd.to_datetime(reference_date)
        
        rfm = self.df.groupby('user_id').agg({
            'order_date': lambda x: (reference_date - x.max()).days,
            'order_id': 'count',
            'total_amount': 'sum'
        }).reset_index()
        
        rfm.columns = ['user_id', 'Recency', 'Frequency', 'Monetary']
        
        rfm['R_Score'] = pd.qcut(rfm['Recency'], q=5, labels=[5, 4, 3, 2, 1], duplicates='drop')
        rfm['F_Score'] = pd.qcut(rfm['Frequency'], q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
        rfm['M_Score'] = pd.qcut(rfm['Monetary'], q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
        
        rfm['RFM_Score'] = (rfm['R_Score'].astype(str) + 
                           rfm['F_Score'].astype(str) + 
                           rfm['M_Score'].astype(str))
        
        def classify_customer(row):
            r, f, m = int(row['R_Score']), int(row['F_Score']), int(row['M_Score'])
            if r >= 4 and f >= 4 and m >= 4:
                return '重要价值客户'
            elif r >= 4 and f >= 4:
                return '重要发展客户'
            elif r >= 4 and m >= 4:
                return '重要保持客户'
            elif r >= 4:
                return '一般发展客户'
            elif f >= 4 and m >= 4:
                return '重要挽留客户'
            elif f >= 4:
                return '一般保持客户'
            elif m >= 4:
                return '一般挽留客户'
            else:
                return '低价值客户'
        
        rfm['Customer_Segment'] = rfm.apply(classify_customer, axis=1)
        
        segment_summary = rfm['Customer_Segment'].value_counts()
        print("\n客户分群统计:")
        for segment, count in segment_summary.items():
            print(f"  {segment}: {count}人 ({count/len(rfm)*100:.1f}%)")
        
        self.analysis_results['rfm_analysis'] = rfm
        self.analysis_results['segment_summary'] = segment_summary.to_dict()
        
        return rfm
    
    def time_series_analysis(self, freq: str = 'D') -> pd.DataFrame:
        """
        时间序列分析
        
        Args:
            freq: 时间频率，'D'日, 'W'周, 'M'月
            
        Returns:
            时间序列聚合数据
        """
        print("\n" + "=" * 50)
        print("时间序列分析")
        print("=" * 50)
        
        if 'order_date' not in self.df.columns:
            print("缺少日期字段，无法进行时间序列分析")
            return None
        
        self.df['order_date'] = pd.to_datetime(self.df['order_date'])
        
        daily_sales = self.df.groupby(self.df['order_date'].dt.date).agg({
            'total_amount': 'sum',
            'order_id': 'count',
            'user_id': 'nunique'
        }).reset_index()
        
        daily_sales.columns = ['date', 'total_sales', 'order_count', 'unique_users']
        daily_sales['date'] = pd.to_datetime(daily_sales['date'])
        daily_sales = daily_sales.sort_values('date')
        
        daily_sales['MA_7'] = daily_sales['total_sales'].rolling(window=7, min_periods=1).mean()
        daily_sales['MA_30'] = daily_sales['total_sales'].rolling(window=30, min_periods=1).mean()
        
        print(f"\n时间范围: {daily_sales['date'].min()} 至 {daily_sales['date'].max()}")
        print(f"总天数: {len(daily_sales)}")
        print(f"日均销售额: {daily_sales['total_sales'].mean():,.2f}")
        print(f"日均订单数: {daily_sales['order_count'].mean():.1f}")
        
        self.analysis_results['daily_sales'] = daily_sales
        
        return daily_sales
    
    def kmeans_clustering(self, n_clusters: int = 4) -> Tuple[pd.DataFrame, Any]:
        """
        KMeans聚类分析
        
        Args:
            n_clusters: 聚类数量
            
        Returns:
            (带聚类标签的DataFrame, 聚类模型)
        """
        print("\n" + "=" * 50)
        print("KMeans用户聚类分析")
        print("=" * 50)
        
        if 'rfm_analysis' not in self.analysis_results:
            print("请先执行RFM分析")
            return None, None
        
        rfm = self.analysis_results['rfm_analysis'].copy()
        
        features = ['Recency', 'Frequency', 'Monetary']
        X = rfm[features].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        rfm['Cluster'] = kmeans.fit_predict(X_scaled)
        
        cluster_stats = rfm.groupby('Cluster').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean',
            'user_id': 'count'
        }).round(2)
        
        cluster_stats.columns = ['平均Recency', '平均Frequency', '平均Monetary', '用户数']
        
        print("\n聚类统计:")
        print(cluster_stats)
        
        def label_cluster(row):
            r, f, m = row['平均Recency'], row['平均Frequency'], row['平均Monetary']
            if r < rfm['Recency'].median() and f > rfm['Frequency'].median() and m > rfm['Monetary'].median():
                return '高价值活跃客户'
            elif r < rfm['Recency'].median() and f <= rfm['Frequency'].median():
                return '新客户'
            elif r >= rfm['Recency'].median() and m > rfm['Monetary'].median():
                return '流失风险高价值客户'
            else:
                return '普通客户'
        
        cluster_labels = cluster_stats.apply(label_cluster, axis=1)
        cluster_stats['客户类型'] = cluster_labels
        
        print("\n客户类型标签:")
        for idx, row in cluster_stats.iterrows():
            print(f"  群组{idx}: {row['客户类型']} ({int(row['用户数'])}人)")
        
        self.analysis_results['cluster_stats'] = cluster_stats
        self.analysis_results['kmeans_model'] = kmeans
        self.analysis_results['rfm_analysis'] = rfm
        
        return rfm, kmeans
    
    def sales_prediction(self) -> Dict[str, Any]:
        """
        销售预测（线性回归）
        
        Returns:
            预测结果字典
        """
        print("\n" + "=" * 50)
        print("销售预测分析（线性回归）")
        print("=" * 50)
        
        if 'daily_sales' not in self.analysis_results:
            self.time_series_analysis()
        
        daily_sales = self.analysis_results['daily_sales'].copy()
        
        daily_sales['day_of_week'] = daily_sales['date'].dt.dayofweek
        daily_sales['day_of_month'] = daily_sales['date'].dt.day
        daily_sales['month'] = daily_sales['date'].dt.month
        daily_sales['day_number'] = (daily_sales['date'] - daily_sales['date'].min()).dt.days
        
        features = ['day_number', 'day_of_week', 'day_of_month', 'month']
        X = daily_sales[features].values
        y = daily_sales['total_sales'].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        print(f"\n训练集评估:")
        print(f"  RMSE: {train_rmse:,.2f}")
        print(f"  MAE: {train_mae:,.2f}")
        print(f"  R²: {train_r2:.4f}")
        
        print(f"\n测试集评估:")
        print(f"  RMSE: {test_rmse:,.2f}")
        print(f"  MAE: {test_mae:,.2f}")
        print(f"  R²: {test_r2:.4f}")
        
        daily_sales['predicted_sales'] = model.predict(X)
        
        prediction_results = {
            'model': model,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'predictions': daily_sales[['date', 'total_sales', 'predicted_sales']]
        }
        
        self.analysis_results['prediction_results'] = prediction_results
        
        return prediction_results
    
    def category_analysis(self) -> pd.DataFrame:
        """
        类别销售分析
        
        Returns:
            类别销售统计DataFrame
        """
        print("\n" + "=" * 50)
        print("类别销售分析")
        print("=" * 50)
        
        if 'category' not in self.df.columns:
            print("缺少类别字段")
            return None
        
        category_stats = self.df.groupby('category').agg({
            'total_amount': ['sum', 'mean', 'count'],
            'rating': 'mean',
            'quantity': 'sum'
        }).round(2)
        
        category_stats.columns = ['总销售额', '平均订单金额', '订单数', '平均评分', '总销量']
        category_stats = category_stats.sort_values('总销售额', ascending=False)
        
        category_stats['销售占比(%)'] = (
            category_stats['总销售额'] / category_stats['总销售额'].sum() * 100
        ).round(2)
        
        print("\n类别销售统计:")
        print(category_stats)
        
        self.analysis_results['category_stats'] = category_stats
        
        return category_stats
    
    def get_all_results(self) -> Dict[str, Any]:
        """
        获取所有分析结果
        
        Returns:
            分析结果字典
        """
        return self.analysis_results


def analyze_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    执行完整的数据分析流程
    
    Args:
        df: 待分析的DataFrame
        
    Returns:
        分析结果字典
    """
    analyzer = DataAnalyzer(df)
    
    analyzer.descriptive_statistics()
    analyzer.correlation_analysis()
    analyzer.rfm_analysis()
    analyzer.time_series_analysis()
    analyzer.kmeans_clustering(n_clusters=4)
    analyzer.sales_prediction()
    analyzer.category_analysis()
    
    return analyzer.get_all_results()


if __name__ == '__main__':
    from data_generator import generate_and_save_data
    from data_cleaner import clean_data
    
    df = generate_and_save_data(n_records=100)
    cleaned_df, _ = clean_data(df)
    results = analyze_data(cleaned_df)
    
    print("\n分析完成，结果键:", list(results.keys()))
