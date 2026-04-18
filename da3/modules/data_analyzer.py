# -*- coding: utf-8 -*-
"""
数据分析模块
包含探索性分析、时间序列分析、用户聚类、销售预测等功能
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


class DataAnalyzer:
    """电商销售数据分析器"""
    
    def __init__(self, df):
        """
        初始化数据分析器
        
        参数:
            df: 清洗后的DataFrame数据
        """
        self.df = df.copy()
        self.analysis_results = {}
        
    def descriptive_statistics(self):
        """
        描述性统计分析
        
        返回:
            dict: 各类统计指标
        """
        print("\n" + "=" * 50)
        print("执行描述性统计分析...")
        print("=" * 50)
        
        stats = {}
        
        # 1. 基本统计量
        numeric_cols = ['交易金额', '单价', '数量', '用户评分']
        available_numeric_cols = [col for col in numeric_cols if col in self.df.columns]
        
        basic_stats = self.df[available_numeric_cols].describe()
        stats['basic_statistics'] = basic_stats
        print("\n基本统计量:")
        print(basic_stats)
        
        # 2. 类别统计
        if '商品类别' in self.df.columns:
            category_stats = self.df.groupby('商品类别').agg({
                '交易金额': ['count', 'sum', 'mean'],
                '用户评分': 'mean'
            }).round(2)
            category_stats.columns = ['订单数', '总销售额', '平均订单金额', '平均评分']
            stats['category_statistics'] = category_stats
            print("\n类别统计:")
            print(category_stats)
        
        # 3. 时间统计
        if '交易日期' in self.df.columns:
            self.df['交易日期'] = pd.to_datetime(self.df['交易日期'])
            daily_sales = self.df.groupby('交易日期')['交易金额'].sum()
            stats['daily_sales_stats'] = {
                '日均销售额': round(daily_sales.mean(), 2),
                '日销售额标准差': round(daily_sales.std(), 2),
                '最高日销售额': round(daily_sales.max(), 2),
                '最低日销售额': round(daily_sales.min(), 2)
            }
            print("\n时间统计:")
            for key, value in stats['daily_sales_stats'].items():
                print(f"  {key}: {value}")
        
        # 4. 用户统计
        if '用户ID' in self.df.columns:
            user_stats = self.df.groupby('用户ID').agg({
                '交易金额': ['count', 'sum', 'mean']
            })
            user_stats.columns = ['购买次数', '总消费', '平均消费']
            stats['user_statistics'] = {
                '总用户数': self.df['用户ID'].nunique(),
                '人均购买次数': round(user_stats['购买次数'].mean(), 2),
                '人均消费金额': round(user_stats['总消费'].mean(), 2),
                '最高消费用户金额': round(user_stats['总消费'].max(), 2)
            }
            print("\n用户统计:")
            for key, value in stats['user_statistics'].items():
                print(f"  {key}: {value}")
        
        self.analysis_results['descriptive'] = stats
        return stats
    
    def correlation_analysis(self):
        """
        相关性分析
        
        返回:
            DataFrame: 相关性矩阵
        """
        print("\n" + "=" * 50)
        print("执行相关性分析...")
        print("=" * 50)
        
        numeric_cols = ['交易金额', '单价', '数量', '用户评分']
        available_cols = [col for col in numeric_cols if col in self.df.columns]
        
        corr_matrix = self.df[available_cols].corr()
        
        print("\n相关性矩阵:")
        print(corr_matrix.round(3))
        
        # 找出强相关关系
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    strong_correlations.append({
                        '变量1': corr_matrix.columns[i],
                        '变量2': corr_matrix.columns[j],
                        '相关系数': round(corr_val, 3)
                    })
        
        if strong_correlations:
            print("\n强相关关系 (|r| > 0.5):")
            for corr in strong_correlations:
                print(f"  {corr['变量1']} - {corr['变量2']}: {corr['相关系数']}")
        
        self.analysis_results['correlation'] = {
            'correlation_matrix': corr_matrix,
            'strong_correlations': strong_correlations
        }
        
        return corr_matrix
    
    def time_series_analysis(self):
        """
        时间序列分析
        包括移动平均、趋势分析
        
        返回:
            DataFrame: 时间序列数据
        """
        print("\n" + "=" * 50)
        print("执行时间序列分析...")
        print("=" * 50)
        
        if '交易日期' not in self.df.columns:
            print("错误: 数据中缺少交易日期字段")
            return None
        
        # 确保日期格式正确
        self.df['交易日期'] = pd.to_datetime(self.df['交易日期'])
        
        # 按日期聚合销售额
        daily_sales = self.df.groupby('交易日期').agg({
            '交易金额': 'sum',
            '订单ID': 'count',
            '用户ID': 'nunique'
        }).reset_index()
        daily_sales.columns = ['日期', '销售额', '订单数', '活跃用户数']
        
        # 计算移动平均
        daily_sales['MA_7'] = daily_sales['销售额'].rolling(window=7, min_periods=1).mean()
        daily_sales['MA_14'] = daily_sales['销售额'].rolling(window=14, min_periods=1).mean()
        daily_sales['MA_30'] = daily_sales['销售额'].rolling(window=30, min_periods=1).mean()
        
        # 计算日环比增长率
        daily_sales['日环比增长率'] = daily_sales['销售额'].pct_change() * 100
        
        # 计算趋势（简单线性回归）
        x = np.arange(len(daily_sales)).reshape(-1, 1)
        y = daily_sales['销售额'].values
        
        model = LinearRegression()
        model.fit(x, y)
        daily_sales['趋势值'] = model.predict(x)
        
        trend_direction = "上升" if model.coef_[0] > 0 else "下降"
        trend_slope = model.coef_[0]
        
        print(f"\n时间序列分析结果:")
        print(f"  分析天数: {len(daily_sales)} 天")
        print(f"  总销售额: {daily_sales['销售额'].sum():.2f}")
        print(f"  日均销售额: {daily_sales['销售额'].mean():.2f}")
        print(f"  销售额标准差: {daily_sales['销售额'].std():.2f}")
        print(f"  趋势方向: {trend_direction}")
        print(f"  趋势斜率: {trend_slope:.4f} (每天变化)")
        print(f"  7日移动平均最新值: {daily_sales['MA_7'].iloc[-1]:.2f}")
        
        self.analysis_results['time_series'] = {
            'daily_sales': daily_sales,
            'trend_slope': trend_slope,
            'trend_direction': trend_direction,
            'r_squared': model.score(x, y)
        }
        
        return daily_sales
    
    def user_clustering(self, n_clusters=4):
        """
        用户价值聚类分析（基于RFM模型）
        
        参数:
            n_clusters: 聚类数量
            
        返回:
            DataFrame: 带聚类标签的用户数据
        """
        print("\n" + "=" * 50)
        print("执行用户价值聚类分析 (K-Means)...")
        print("=" * 50)
        
        if '用户ID' not in self.df.columns or '交易日期' not in self.df.columns:
            print("错误: 数据中缺少必要的字段")
            return None
        
        # 确保日期格式正确
        self.df['交易日期'] = pd.to_datetime(self.df['交易日期'])
        
        # 计算RFM指标
        current_date = self.df['交易日期'].max() + timedelta(days=1)
        
        rfm = self.df.groupby('用户ID').agg({
            '交易日期': lambda x: (current_date - x.max()).days,  # Recency
            '订单ID': 'count',  # Frequency
            '交易金额': 'sum'   # Monetary
        }).reset_index()
        rfm.columns = ['用户ID', 'R_最近购买天数', 'F_购买频次', 'M_消费金额']
        
        # 准备聚类数据
        features = ['R_最近购买天数', 'F_购买频次', 'M_消费金额']
        X = rfm[features].values
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # K-Means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        rfm['聚类标签'] = kmeans.fit_predict(X_scaled)
        
        # 分析每个聚类
        cluster_analysis = rfm.groupby('聚类标签').agg({
            'R_最近购买天数': 'mean',
            'F_购买频次': 'mean',
            'M_消费金额': 'mean',
            '用户ID': 'count'
        }).round(2)
        cluster_analysis.columns = ['平均最近购买天数', '平均购买频次', '平均消费金额', '用户数']
        cluster_analysis['用户占比'] = (cluster_analysis['用户数'] / len(rfm) * 100).round(2)
        
        # 为聚类命名
        cluster_names = {}
        for idx, row in cluster_analysis.iterrows():
            r, f, m = row['平均最近购买天数'], row['平均购买频次'], row['平均消费金额']
            if m > cluster_analysis['平均消费金额'].median() and f > cluster_analysis['平均购买频次'].median():
                name = "高价值客户"
            elif m > cluster_analysis['平均消费金额'].median():
                name = "高消费客户"
            elif f > cluster_analysis['平均购买频次'].median():
                name = "频繁购买客户"
            elif r < cluster_analysis['平均最近购买天数'].median():
                name = "新客户"
            else:
                name = "低价值客户"
            cluster_names[idx] = name
        
        rfm['客户类型'] = rfm['聚类标签'].map(cluster_names)
        
        print(f"\n聚类分析结果 (K={n_clusters}):")
        print(cluster_analysis)
        
        print(f"\n客户类型分布:")
        for label, name in cluster_names.items():
            count = (rfm['聚类标签'] == label).sum()
            percentage = count / len(rfm) * 100
            print(f"  {name} (聚类{label}): {count}人 ({percentage:.1f}%)")
        
        # 使用PCA进行可视化准备
        pca = PCA(n_components=2)
        rfm_pca = pca.fit_transform(X_scaled)
        rfm['PCA1'] = rfm_pca[:, 0]
        rfm['PCA2'] = rfm_pca[:, 1]
        
        print(f"\nPCA解释方差比: {pca.explained_variance_ratio_}")
        
        self.analysis_results['clustering'] = {
            'rfm_data': rfm,
            'cluster_analysis': cluster_analysis,
            'cluster_names': cluster_names,
            'kmeans_model': kmeans,
            'scaler': scaler,
            'pca': pca
        }
        
        return rfm
    
    def sales_prediction(self, forecast_days=30):
        """
        销售预测（基于时间序列的线性回归）
        
        参数:
            forecast_days: 预测未来天数
            
        返回:
            dict: 预测结果
        """
        print("\n" + "=" * 50)
        print("执行销售预测...")
        print("=" * 50)
        
        if '交易日期' not in self.df.columns:
            print("错误: 数据中缺少交易日期字段")
            return None
        
        # 准备数据
        self.df['交易日期'] = pd.to_datetime(self.df['交易日期'])
        daily_sales = self.df.groupby('交易日期')['交易金额'].sum().reset_index()
        daily_sales['天数'] = (daily_sales['交易日期'] - daily_sales['交易日期'].min()).dt.days
        
        # 添加特征
        daily_sales['星期'] = daily_sales['交易日期'].dt.dayofweek
        daily_sales['是否周末'] = (daily_sales['星期'] >= 5).astype(int)
        daily_sales['月份'] = daily_sales['交易日期'].dt.month
        
        # 分割训练集和测试集 (80%训练, 20%测试)
        train_size = int(len(daily_sales) * 0.8)
        train_data = daily_sales[:train_size]
        test_data = daily_sales[train_size:]
        
        # 特征工程
        feature_cols = ['天数', '是否周末']
        
        # 训练模型
        X_train = train_data[feature_cols]
        y_train = train_data['交易金额']
        X_test = test_data[feature_cols]
        y_test = test_data['交易金额']
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # 预测
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # 评估
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        print(f"\n模型评估结果:")
        print(f"  训练集 MAE: {train_mae:.2f}")
        print(f"  测试集 MAE: {test_mae:.2f}")
        print(f"  训练集 RMSE: {train_rmse:.2f}")
        print(f"  测试集 RMSE: {test_rmse:.2f}")
        print(f"  训练集 R²: {train_r2:.4f}")
        print(f"  测试集 R²: {test_r2:.4f}")
        
        # 未来预测
        last_date = daily_sales['交易日期'].max()
        future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]
        future_days = [(date - daily_sales['交易日期'].min()).days for date in future_dates]
        future_weekends = [1 if date.weekday() >= 5 else 0 for date in future_dates]
        
        future_df = pd.DataFrame({
            '天数': future_days,
            '是否周末': future_weekends
        })
        
        future_predictions = model.predict(future_df)
        
        future_results = pd.DataFrame({
            '日期': future_dates,
            '预测销售额': future_predictions.round(2)
        })
        
        print(f"\n未来{forecast_days}天预测:")
        print(f"  预测总销售额: {future_predictions.sum():.2f}")
        print(f"  预测日均销售额: {future_predictions.mean():.2f}")
        print(f"  预测最高日销售额: {future_predictions.max():.2f}")
        print(f"  预测最低日销售额: {future_predictions.min():.2f}")
        
        self.analysis_results['prediction'] = {
            'model': model,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'future_predictions': future_results,
            'daily_sales': daily_sales,
            'y_pred_test': y_pred_test
        }
        
        return self.analysis_results['prediction']
    
    def get_all_results(self):
        """获取所有分析结果"""
        return self.analysis_results


if __name__ == '__main__':
    # 测试分析功能
    import sys
    sys.path.append('..')
    from modules.data_generator import DataGenerator
    from modules.data_cleaner import DataCleaner
    
    # 生成并清洗数据
    generator = DataGenerator(n_records=600)
    raw_data = generator.generate_data()
    cleaner = DataCleaner(raw_data)
    clean_data = cleaner.clean_data()
    
    # 执行分析
    analyzer = DataAnalyzer(clean_data)
    analyzer.descriptive_statistics()
    analyzer.correlation_analysis()
    analyzer.time_series_analysis()
    analyzer.user_clustering(n_clusters=4)
    analyzer.sales_prediction(forecast_days=30)
