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
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import STL
import warnings
warnings.filterwarnings('ignore')


class DataAnalyzer:
    """数据分析器"""
    
    def __init__(self, df):
        """
        初始化数据分析器
        
        Args:
            df: 清洗后的DataFrame
        """
        self.df = df.copy()
        self.df['订单时间'] = pd.to_datetime(self.df['订单时间'])
        self.analysis_results = {}
        
    def descriptive_statistics(self):
        """
        描述性统计分析
        
        Returns:
            dict: 描述性统计结果
        """
        print("\n执行描述性统计分析...")
        
        stats = {}
        
        # 基础统计量
        numeric_cols = ['单价', '数量', '总金额', '用户评分']
        stats['基础统计'] = self.df[numeric_cols].describe().round(2)
        
        # 类别统计
        stats['类别销售统计'] = self.df.groupby('商品类别').agg({
            '总金额': ['sum', 'mean', 'count'],
            '用户评分': 'mean'
        }).round(2)
        stats['类别销售统计'].columns = ['销售额总计', '平均订单金额', '订单数', '平均评分']
        
        # 支付方式统计
        stats['支付方式分布'] = self.df['支付方式'].value_counts()
        
        # 地区统计
        stats['地区销售统计'] = self.df.groupby('地区')['总金额'].sum().sort_values(ascending=False)
        
        # 时间范围
        stats['时间范围'] = {
            '开始日期': self.df['订单时间'].min(),
            '结束日期': self.df['订单时间'].max(),
            '时间跨度天数': (self.df['订单时间'].max() - self.df['订单时间'].min()).days
        }
        
        self.analysis_results['描述性统计'] = stats
        print("描述性统计分析完成")
        
        return stats
    
    def correlation_analysis(self):
        """
        相关性分析
        
        Returns:
            DataFrame: 相关系数矩阵
        """
        print("\n执行相关性分析...")
        
        # 选择数值型字段
        numeric_cols = ['单价', '数量', '总金额', '用户评分']
        corr_matrix = self.df[numeric_cols].corr().round(3)
        
        self.analysis_results['相关性分析'] = corr_matrix
        print("相关性分析完成")
        
        return corr_matrix
    
    def time_series_analysis(self, freq='D'):
        """
        时间序列分析，包含趋势分析和STL分解
        
        Args:
            freq: 时间频率，'D'为天，'W'为周
            
        Returns:
            dict: 时间序列分析结果
        """
        print(f"\n执行时间序列分析 (频率: {freq})...")
        
        # 按时间聚合销售额
        if freq == 'D':
            ts_data = self.df.set_index('订单时间').resample('D')['总金额'].sum().fillna(0)
            window = 7  # 7天移动平均
        else:
            ts_data = self.df.set_index('订单时间').resample('W')['总金额'].sum().fillna(0)
            window = 4  # 4周移动平均
        
        results = {
            '原始数据': ts_data,
            '移动平均': ts_data.rolling(window=window, center=True).mean(),
            '时间频率': freq
        }
        
        # STL分解（需要至少两个周期的数据）
        if len(ts_data) >= 14:
            try:
                stl = STL(ts_data, period=7 if freq == 'D' else 4, robust=True)
                res = stl.fit()
                results['STL分解'] = {
                    '趋势': res.trend,
                    '季节性': res.seasonal,
                    '残差': res.resid
                }
                print("STL分解完成")
            except Exception as e:
                print(f"STL分解失败: {str(e)}")
                results['STL分解'] = None
        
        # 计算增长率
        if len(ts_data) > 1:
            results['日增长率'] = ts_data.pct_change().mean() * 100
        
        self.analysis_results['时间序列分析'] = results
        print("时间序列分析完成")
        
        return results
    
    def rfm_analysis(self):
        """
        RFM分析（Recency, Frequency, Monetary）
        
        Returns:
            DataFrame: RFM分析结果
        """
        print("\n执行RFM分析...")
        
        # 计算分析日期（数据最后一天）
        analysis_date = self.df['订单时间'].max()
        
        # 计算每个用户的RFM指标
        rfm = self.df.groupby('用户ID').agg({
            '订单时间': lambda x: (analysis_date - x.max()).days,  # Recency
            '订单ID': 'count',  # Frequency
            '总金额': 'sum'  # Monetary
        }).reset_index()
        
        rfm.columns = ['用户ID', '最近购买天数', '购买频率', '消费金额']
        
        # RFM评分（1-5分）
        rfm['R评分'] = pd.qcut(rfm['最近购买天数'], 5, labels=[5,4,3,2,1])
        rfm['F评分'] = pd.qcut(rfm['购买频率'].rank(method='first'), 5, labels=[1,2,3,4,5])
        rfm['M评分'] = pd.qcut(rfm['消费金额'], 5, labels=[1,2,3,4,5])
        
        # 转换为数值
        rfm['R评分'] = rfm['R评分'].astype(int)
        rfm['F评分'] = rfm['F评分'].astype(int)
        rfm['M评分'] = rfm['M评分'].astype(int)
        
        # RFM综合评分
        rfm['RFM评分'] = rfm['R评分'] + rfm['F评分'] + rfm['M评分']
        
        # 用户分层
        def segment_rfm(row):
            if row['RFM评分'] >= 13:
                return '重要价值客户'
            elif row['RFM评分'] >= 10:
                return '潜力客户'
            elif row['RFM评分'] >= 7:
                return '一般客户'
            else:
                return '低价值客户'
        
        rfm['客户分层'] = rfm.apply(segment_rfm, axis=1)
        
        self.analysis_results['RFM分析'] = rfm
        print(f"RFM分析完成，共分析 {len(rfm)} 个用户")
        
        return rfm
    
    def customer_clustering(self, n_clusters=4):
        """
        用户聚类分析（基于KMeans）
        
        Args:
            n_clusters: 聚类数量
            
        Returns:
            dict: 聚类分析结果
        """
        print(f"\n执行用户聚类分析 (K={n_clusters})...")
        
        # 获取RFM数据
        if 'RFM分析' not in self.analysis_results:
            self.rfm_analysis()
        
        rfm = self.analysis_results['RFM分析'].copy()
        
        # 准备聚类特征
        features = ['最近购买天数', '购买频率', '消费金额']
        X = rfm[features].copy()
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # KMeans聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        rfm['聚类标签'] = kmeans.fit_predict(X_scaled)
        
        # 计算轮廓系数
        silhouette_avg = silhouette_score(X_scaled, rfm['聚类标签'])
        
        # 分析每个聚类的特征
        cluster_profiles = rfm.groupby('聚类标签')[features].mean().round(2)
        cluster_sizes = rfm['聚类标签'].value_counts().sort_index()
        
        results = {
            '聚类数据': rfm,
            '聚类中心': cluster_profiles,
            '聚类大小': cluster_sizes,
            '轮廓系数': silhouette_avg,
            '标准化器': scaler,
            '聚类模型': kmeans
        }
        
        self.analysis_results['用户聚类'] = results
        print(f"聚类分析完成，轮廓系数: {silhouette_avg:.3f}")
        
        return results
    
    def sales_prediction(self, forecast_days=30):
        """
        销售预测（基于线性回归）
        
        Args:
            forecast_days: 预测天数
            
        Returns:
            dict: 预测结果
        """
        print(f"\n执行销售预测 (预测未来 {forecast_days} 天)...")
        
        # 按天聚合销售额
        daily_sales = self.df.set_index('订单时间').resample('D')['总金额'].sum().fillna(0)
        daily_sales = daily_sales.reset_index()
        daily_sales['日期序号'] = range(len(daily_sales))
        
        # 添加时间特征
        daily_sales['月份'] = daily_sales['订单时间'].dt.month
        daily_sales['星期'] = daily_sales['订单时间'].dt.dayofweek
        daily_sales['是否周末'] = (daily_sales['星期'] >= 5).astype(int)
        
        # 特征和目标
        feature_cols = ['日期序号', '月份', '星期', '是否周末']
        X = daily_sales[feature_cols]
        y = daily_sales['总金额']
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # 训练模型
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # 预测测试集
        y_pred = model.predict(X_test)
        
        # 计算评估指标
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        # 预测未来
        last_date = daily_sales['订单时间'].max()
        future_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]
        
        future_data = pd.DataFrame({
            '订单时间': future_dates,
            '日期序号': range(len(daily_sales), len(daily_sales) + forecast_days),
            '月份': [d.month for d in future_dates],
            '星期': [d.dayofweek for d in future_dates],
            '是否周末': [(1 if d.dayofweek >= 5 else 0) for d in future_dates]
        })
        
        future_predictions = model.predict(future_data[feature_cols])
        future_data['预测销售额'] = future_predictions
        
        results = {
            '历史数据': daily_sales,
            '测试集预测': pd.DataFrame({
                '实际值': y_test,
                '预测值': y_pred
            }),
            '未来预测': future_data[['订单时间', '预测销售额']],
            '模型评估': {
                'MSE': round(mse, 2),
                'RMSE': round(rmse, 2),
                'R²': round(r2, 4),
                'MAPE(%)': round(mape, 2)
            },
            '模型系数': dict(zip(feature_cols, model.coef_)),
            '截距': model.intercept_
        }
        
        self.analysis_results['销售预测'] = results
        print(f"预测完成，R²={r2:.4f}, RMSE={rmse:.2f}")
        
        return results
    
    def run_full_analysis(self):
        """
        执行完整的分析流程
        
        Returns:
            dict: 所有分析结果
        """
        print("\n" + "=" * 60)
        print("开始执行完整数据分析...")
        print("=" * 60)
        
        # 1. 描述性统计
        self.descriptive_statistics()
        
        # 2. 相关性分析
        self.correlation_analysis()
        
        # 3. 时间序列分析
        self.time_series_analysis(freq='D')
        
        # 4. RFM分析
        self.rfm_analysis()
        
        # 5. 用户聚类
        self.customer_clustering(n_clusters=4)
        
        # 6. 销售预测
        self.sales_prediction(forecast_days=30)
        
        print("\n" + "=" * 60)
        print("数据分析完成!")
        print("=" * 60)
        
        return self.analysis_results
    
    def generate_summary_report(self):
        """
        生成分析摘要报告
        
        Returns:
            str: 报告文本
        """
        if not self.analysis_results:
            return "请先执行分析"
        
        report = []
        report.append("=" * 70)
        report.append("电商销售数据分析报告")
        report.append("=" * 70)
        report.append(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 1. 数据概览
        report.append("【一、数据概览】")
        stats = self.analysis_results.get('描述性统计', {})
        time_range = stats.get('时间范围', {})
        report.append(f"  分析时间跨度: {time_range.get('开始日期', 'N/A')} 至 {time_range.get('结束日期', 'N/A')}")
        report.append(f"  总订单数: {len(self.df)}")
        report.append(f"  总用户数: {self.df['用户ID'].nunique()}")
        report.append(f"  总销售额: ¥{self.df['总金额'].sum():,.2f}")
        report.append(f"  平均订单金额: ¥{self.df['总金额'].mean():.2f}")
        report.append("")
        
        # 2. 关键统计量
        report.append("【二、关键统计量】")
        basic_stats = stats.get('基础统计', pd.DataFrame())
        if not basic_stats.empty:
            report.append(f"  订单金额统计:")
            report.append(f"    - 最小值: ¥{basic_stats.loc['min', '总金额']:.2f}")
            report.append(f"    - 最大值: ¥{basic_stats.loc['max', '总金额']:.2f}")
            report.append(f"    - 中位数: ¥{basic_stats.loc['50%', '总金额']:.2f}")
            report.append(f"    - 标准差: ¥{basic_stats.loc['std', '总金额']:.2f}")
        report.append("")
        
        # 3. 类别分析
        report.append("【三、商品类别销售分析】")
        category_stats = stats.get('类别销售统计', pd.DataFrame())
        if not category_stats.empty:
            for cat in category_stats.index:
                row = category_stats.loc[cat]
                report.append(f"  {cat}:")
                report.append(f"    - 销售额: ¥{row['销售额总计']:,.2f}")
                report.append(f"    - 订单数: {row['订单数']:.0f}")
                report.append(f"    - 平均评分: {row['平均评分']:.2f}")
        report.append("")
        
        # 4. 时间序列分析
        report.append("【四、时间序列分析】")
        ts_results = self.analysis_results.get('时间序列分析', {})
        growth_rate = ts_results.get('日增长率')
        if growth_rate is not None:
            report.append(f"  日均销售增长率: {growth_rate:.2f}%")
        report.append("")
        
        # 5. 聚类结论
        report.append("【五、用户聚类分析】")
        cluster_results = self.analysis_results.get('用户聚类', {})
        silhouette = cluster_results.get('轮廓系数', 0)
        report.append(f"  聚类质量（轮廓系数）: {silhouette:.3f}")
        report.append(f"  聚类数量: {len(cluster_results.get('聚类大小', []))}")
        
        cluster_profiles = cluster_results.get('聚类中心', pd.DataFrame())
        cluster_sizes = cluster_results.get('聚类大小', pd.Series())
        if not cluster_profiles.empty:
            report.append("  各聚类特征:")
            for i in cluster_profiles.index:
                row = cluster_profiles.loc[i]
                size = cluster_sizes.get(i, 0)
                report.append(f"    聚类 {i} (用户数: {size}):")
                report.append(f"      - 平均最近购买天数: {row['最近购买天数']:.1f} 天")
                report.append(f"      - 平均购买频率: {row['购买频率']:.1f} 次")
                report.append(f"      - 平均消费金额: ¥{row['消费金额']:.2f}")
        report.append("")
        
        # 6. 预测结果
        report.append("【六、销售预测结果】")
        pred_results = self.analysis_results.get('销售预测', {})
        model_eval = pred_results.get('模型评估', {})
        if model_eval:
            report.append(f"  模型评估指标:")
            report.append(f"    - R² 决定系数: {model_eval.get('R²', 'N/A')}")
            report.append(f"    - RMSE 均方根误差: {model_eval.get('RMSE', 'N/A')}")
            report.append(f"    - MAPE 平均绝对百分比误差: {model_eval.get('MAPE(%)', 'N/A')}%")
        
        future_pred = pred_results.get('未来预测', pd.DataFrame())
        if not future_pred.empty:
            total_forecast = future_pred['预测销售额'].sum()
            report.append(f"  未来30天预测总销售额: ¥{total_forecast:,.2f}")
        report.append("")
        
        # 7. 业务建议
        report.append("【七、业务建议】")
        
        # 根据分析结果生成建议
        if not category_stats.empty:
            top_category = category_stats['销售额总计'].idxmax()
            report.append(f"  1. {top_category}是销售额最高的类别，建议加大投入和库存")
        
        if not cluster_profiles.empty:
            # 找出高价值聚类（高消费金额）
            high_value_cluster = cluster_profiles['消费金额'].idxmax()
            report.append(f"  2. 聚类 {high_value_cluster} 为高价值用户群，建议重点维护")
        
        report.append("  3. 建议根据RFM模型对用户进行精准营销")
        report.append("  4. 关注销售趋势变化，及时调整库存策略")
        report.append("")
        
        report.append("=" * 70)
        report.append("报告结束")
        report.append("=" * 70)
        
        return "\n".join(report)


def analyze_data(df):
    """
    分析数据的便捷函数
    
    Args:
        df: 清洗后的DataFrame
        
    Returns:
        DataAnalyzer: 分析器实例
    """
    analyzer = DataAnalyzer(df)
    analyzer.run_full_analysis()
    return analyzer


if __name__ == '__main__':
    # 测试分析功能
    import sys
    sys.path.append('..')
    from modules.data_generator import generate_raw_data
    from modules.data_cleaner import clean_data
    
    # 生成并清洗数据
    raw_df = generate_raw_data('data/test_raw.csv', n_records=500)
    cleaned_df, _ = clean_data('data/test_raw.csv', 'data/test_cleaned.csv')
    
    # 执行分析
    analyzer = analyze_data(cleaned_df)
    
    # 打印报告
    print(analyzer.generate_summary_report())
