import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.seasonal import STL
import warnings
warnings.filterwarnings('ignore')


def exploratory_analysis(df):
    """
    探索性数据分析：描述统计 + 相关性分析
    
    参数:
        df: 清洗后的DataFrame
    
    返回:
        dict: 分析结果
    """
    print("\n开始探索性数据分析...")
    results = {}
    
    descriptive_stats = df[['订单金额', '商品数量', '用户评分']].describe()
    results['描述统计'] = descriptive_stats
    
    numeric_cols = ['订单金额', '商品数量', '用户评分', '月份', '星期', '小时']
    correlation_matrix = df[numeric_cols].corr()
    results['相关性矩阵'] = correlation_matrix
    
    category_stats = df.groupby('商品类别').agg({
        '订单金额': ['sum', 'mean', 'count'],
        '用户评分': 'mean'
    }).round(2)
    results['类别统计'] = category_stats
    
    print("探索性分析完成")
    return results


def rfm_analysis(df):
    """
    RFM用户价值分析
    
    参数:
        df: 清洗后的DataFrame
    
    返回:
        DataFrame: 包含RFM指标和聚类结果的用户数据
    """
    print("\n开始RFM用户价值分析...")
    
    max_date = df['下单时间'].max()
    
    rfm = df.groupby('用户ID').agg({
        '下单时间': lambda x: (max_date - x.max()).days,
        '订单ID': 'count',
        '订单金额': 'sum'
    }).rename(columns={
        '下单时间': 'Recency',
        '订单ID': 'Frequency',
        '订单金额': 'Monetary'
    })
    
    rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5])
    
    rfm['R_Score'] = rfm['R_Score'].astype(int)
    rfm['F_Score'] = rfm['F_Score'].astype(int)
    rfm['M_Score'] = rfm['M_Score'].astype(int)
    rfm['RFM_Score'] = rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']
    
    print("RFM分析完成")
    return rfm


def customer_clustering(rfm_df, n_clusters=4):
    """
    KMeans用户聚类
    
    参数:
        rfm_df: RFM分析结果
        n_clusters: 聚类数量
    
    返回:
        DataFrame: 带聚类标签的RFM数据
        dict: 聚类分析结果
    """
    print("\n开始KMeans用户聚类...")
    cluster_results = {}
    
    features = ['Recency', 'Frequency', 'Monetary']
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_df[features])
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    cluster_summary = rfm_df.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': ['mean', 'count']
    }).round(2)
    
    cluster_names = {
        0: '高价值忠诚客户',
        1: '流失高价值客户',
        2: '普通客户',
        3: '新客户'
    }
    
    if n_clusters == 4:
        cluster_means = rfm_df.groupby('Cluster')['RFM_Score'].mean().sort_values(ascending=False)
        cluster_names = {}
        for i, (cluster, _) in enumerate(cluster_means.items()):
            if i == 0:
                cluster_names[cluster] = '高价值忠诚客户'
            elif i == 1:
                cluster_names[cluster] = '潜力客户'
            elif i == 2:
                cluster_names[cluster] = '流失风险客户'
            else:
                cluster_names[cluster] = '低价值客户'
    
    rfm_df['客户类型'] = rfm_df['Cluster'].map(cluster_names)
    cluster_results['聚类摘要'] = cluster_summary
    cluster_results['聚类名称'] = cluster_names
    cluster_results['聚类分布'] = rfm_df['客户类型'].value_counts()
    
    print("KMeans聚类完成")
    return rfm_df, cluster_results


def time_series_analysis(df):
    """
    时间序列分析：趋势、季节分解 + 移动平均
    
    参数:
        df: 清洗后的DataFrame
    
    返回:
        DataFrame: 日销售数据
        dict: 时间序列分析结果
    """
    print("\n开始时间序列分析...")
    ts_results = {}
    
    daily_sales = df.groupby('日期')['订单金额'].sum().reset_index()
    daily_sales['日期'] = pd.to_datetime(daily_sales['日期'])
    daily_sales = daily_sales.sort_values('日期')
    
    daily_sales['MA7'] = daily_sales['订单金额'].rolling(window=7).mean()
    daily_sales['MA30'] = daily_sales['订单金额'].rolling(window=30).mean()
    
    if len(daily_sales) >= 45:
        try:
            daily_sales = daily_sales.set_index('日期')
            stl = STL(daily_sales['订单金额'], period=7, robust=True)
            result = stl.fit()
            daily_sales['趋势'] = result.trend
            daily_sales['季节'] = result.seasonal
            daily_sales['残差'] = result.resid
            daily_sales = daily_sales.reset_index()
            ts_results['STL分解'] = True
        except:
            ts_results['STL分解'] = False
            daily_sales = daily_sales.reset_index()
    else:
        ts_results['STL分解'] = False
    
    ts_results['日销售数据'] = daily_sales
    
    monthly_trend = df.groupby('月份')['订单金额'].agg(['sum', 'mean', 'count'])
    ts_results['月度趋势'] = monthly_trend
    
    weekday_sales = df.groupby('星期')['订单金额'].agg(['sum', 'mean'])
    ts_results['周销售分布'] = weekday_sales
    
    print("时间序列分析完成")
    return daily_sales, ts_results


def sales_prediction(df):
    """
    销售预测：线性回归
    
    参数:
        df: 清洗后的DataFrame
    
    返回:
        dict: 预测结果和评估指标
    """
    print("\n开始销售预测...")
    pred_results = {}
    
    daily_sales = df.groupby('日期')['订单金额'].sum().reset_index()
    daily_sales['日期'] = pd.to_datetime(daily_sales['日期'])
    daily_sales = daily_sales.sort_values('日期')
    daily_sales['DayIndex'] = range(len(daily_sales))
    
    X = daily_sales[['DayIndex']]
    y = daily_sales['订单金额']
    
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    pred_results['模型系数'] = model.coef_[0]
    pred_results['截距'] = model.intercept_
    pred_results['均方误差(MSE)'] = mean_squared_error(y_test, y_pred)
    pred_results['均方根误差(RMSE)'] = np.sqrt(mean_squared_error(y_test, y_pred))
    pred_results['R方得分'] = r2_score(y_test, y_pred)
    pred_results['预测值'] = y_pred
    pred_results['实际值'] = y_test.values
    pred_results['日销售数据'] = daily_sales
    
    print(f"销售预测完成，R方得分: {pred_results['R方得分']:.4f}")
    return pred_results


def generate_summary(df, eda_results, cluster_results, ts_results, pred_results, output_path='output/summary.txt'):
    """
    生成分析总结报告
    """
    print("\n生成分析报告...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("电商销售数据分析总结报告\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("一、数据概览\n")
        f.write("-" * 40 + "\n")
        f.write(f"总订单数: {len(df)}\n")
        f.write(f"总用户数: {df['用户ID'].nunique()}\n")
        f.write(f"时间跨度: {df['下单时间'].min().strftime('%Y-%m-%d')} 至 {df['下单时间'].max().strftime('%Y-%m-%d')}\n")
        f.write(f"总销售额: {df['订单金额'].sum():.2f} 元\n")
        f.write(f"平均订单金额: {df['订单金额'].mean():.2f} 元\n")
        f.write(f"平均用户评分: {df['用户评分'].mean():.2f} 分\n\n")
        
        f.write("二、关键统计量\n")
        f.write("-" * 40 + "\n")
        stats = eda_results['描述统计']
        f.write(f"订单金额 - 中位数: {stats.loc['50%', '订单金额']:.2f} 元, 标准差: {stats.loc['std', '订单金额']:.2f}\n")
        f.write(f"商品数量 - 平均: {stats.loc['mean', '商品数量']:.2f} 件\n")
        f.write(f"用户评分 - 平均: {stats.loc['mean', '用户评分']:.2f} 分\n\n")
        
        f.write("三、商品类别销售分析\n")
        f.write("-" * 40 + "\n")
        category_stats = eda_results['类别统计']
        for category in category_stats.index:
            total_sales = category_stats.loc[category, ('订单金额', 'sum')]
            f.write(f"{category}: 销售额 {total_sales:.2f} 元, 共 {category_stats.loc[category, ('订单金额', 'count')]} 单\n")
        f.write("\n")
        
        f.write("四、用户聚类分析结论\n")
        f.write("-" * 40 + "\n")
        cluster_dist = cluster_results['聚类分布']
        total_users = cluster_dist.sum()
        for customer_type, count in cluster_dist.items():
            f.write(f"{customer_type}: {count} 人 ({count/total_users*100:.1f}%)\n")
        f.write("\n")
        f.write("建议营销策略:\n")
        f.write("  - 高价值客户: VIP专属优惠、定制服务\n")
        f.write("  - 流失风险客户: 召回优惠、满意度调研\n")
        f.write("  - 潜力客户: 交叉销售、提升消费频次\n")
        f.write("  - 低价值客户: 优惠券激励、推荐爆款商品\n\n")
        
        f.write("五、销售预测模型评估\n")
        f.write("-" * 40 + "\n")
        f.write(f"线性回归模型 R方得分: {pred_results['R方得分']:.4f}\n")
        f.write(f"均方根误差(RMSE): {pred_results['均方根误差(RMSE)']:.2f} 元\n")
        f.write(f"销售趋势系数: {pred_results['模型系数']:.2f} 元/天\n")
        if pred_results['模型系数'] > 0:
            f.write("趋势判断: 销售额整体呈上升趋势\n")
        else:
            f.write("趋势判断: 销售额整体呈下降趋势\n")
        
    print(f"分析报告已保存至: {output_path}")


if __name__ == '__main__':
    print("分析模块测试")
