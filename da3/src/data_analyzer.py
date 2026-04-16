import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.seasonal import STL
import warnings
warnings.filterwarnings('ignore')

class DataAnalyzer:
    def __init__(self, df):
        self.df = df.copy()
        self.analysis_results = {}
        self.rfm_df = None
        self.cluster_results = None
    
    def descriptive_statistics(self):
        """描述性统计分析"""
        print("=" * 50)
        print("执行描述性统计分析...")
        print("=" * 50)
        
        desc_stats = self.df.describe().round(2)
        
        category_sales = self.df.groupby('商品类别')['订单金额'].agg(
            ['count', 'sum', 'mean', 'std']).round(2).sort_values('sum', ascending=False)
        
        daily_stats = self.df.groupby('日期')['订单金额'].agg(['sum', 'count']).describe().round(2)
        
        self.analysis_results['描述统计'] = {
            '总订单数': len(self.df),
            '总销售额': round(self.df['订单金额'].sum(), 2),
            '平均订单金额': round(self.df['订单金额'].mean(), 2),
            '订单金额中位数': round(self.df['订单金额'].median(), 2),
            '客单价': round(self.df.groupby('用户ID')['订单金额'].sum().mean(), 2),
            '用户总数': self.df['用户ID'].nunique(),
            '平均评分': round(self.df['评分'].mean(), 2),
            '类别销售详情': category_sales.to_dict(),
            '每日统计详情': daily_stats.to_dict()
        }
        
        print(f"总订单数: {self.analysis_results['描述统计']['总订单数']}")
        print(f"总销售额: {self.analysis_results['描述统计']['总销售额']}")
        print(f"用户总数: {self.analysis_results['描述统计']['用户总数']}")
        
        return self.analysis_results['描述统计']
    
    def correlation_analysis(self):
        """相关性分析"""
        print("=" * 50)
        print("执行相关性分析...")
        print("=" * 50)
        
        numeric_cols = ['单价', '数量', '订单金额', '评分', '小时']
        correlation_matrix = self.df[numeric_cols].corr().round(3)
        
        self.analysis_results['相关性分析'] = correlation_matrix.to_dict()
        
        print("相关性矩阵:")
        print(correlation_matrix)
        
        return correlation_matrix
    
    def time_series_analysis(self):
        """时间序列分析 - STL分解和移动平均"""
        print("=" * 50)
        print("执行时间序列分析...")
        print("=" * 50)
        
        daily_sales = self.df.groupby('日期')['订单金额'].sum().reset_index()
        daily_sales['日期'] = pd.to_datetime(daily_sales['日期'])
        daily_sales = daily_sales.set_index('日期').sort_index()
        
        daily_sales['7日移动平均'] = daily_sales['订单金额'].rolling(window=7, min_periods=1).mean()
        
        daily_sales_freq = daily_sales['订单金额'].asfreq('D').fillna(0)
        try:
            stl = STL(daily_sales_freq, period=7)
            result = stl.fit()
            
            self.analysis_results['时间序列分析'] = {
                '日销售数据': daily_sales.reset_index().to_dict(),
                '趋势': result.trend.tolist(),
                '季节': result.seasonal.tolist(),
                '残差': result.resid.tolist(),
                '移动平均': daily_sales['7日移动平均'].tolist()
            }
        except Exception as e:
            print(f"STL分解警告: {e}")
            self.analysis_results['时间序列分析'] = {
                '日销售数据': daily_sales.reset_index().to_dict(),
                '移动平均': daily_sales['7日移动平均'].tolist()
            }
        
        print("时间序列分析完成")
        
        return daily_sales
    
    def calculate_rfm(self):
        """计算RFM指标"""
        print("=" * 50)
        print("计算RFM用户价值指标...")
        print("=" * 50)
        
        max_date = self.df['订单时间'].max()
        
        rfm = self.df.groupby('用户ID').agg(
            Recency=('订单时间', lambda x: (max_date - x.max()).days),
            Frequency=('用户ID', 'count'),
            Monetary=('订单金额', 'sum')
        ).reset_index()
        
        rfm.columns = ['用户ID', 'Recency', 'Frequency', 'Monetary']
        
        self.rfm_df = rfm
        
        print(f"RFM计算完成，共 {len(rfm)} 位用户")
        print(f"R均值: {round(rfm['Recency'].mean(), 2)}, F均值: {round(rfm['Frequency'].mean(), 2)}, M均值: {round(rfm['Monetary'].mean(), 2)}")
        
        return rfm
    
    def customer_clustering(self, n_clusters=4):
        """用户分层聚类 - KMeans"""
        print("=" * 50)
        print(f"执行用户分层聚类 (K={n_clusters})...")
        print("=" * 50)
        
        if self.rfm_df is None:
            self.calculate_rfm()
        
        features = ['Recency', 'Frequency', 'Monetary']
        X = self.rfm_df[features]
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        self.rfm_df['聚类'] = clusters
        self.cluster_results = self.rfm_df.copy()
        
        cluster_analysis = self.rfm_df.groupby('聚类').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean',
            '用户ID': 'count'
        }).round(2)
        cluster_analysis.columns = ['平均R', '平均F', '平均M', '用户数']
        
        cluster_names = {
            cluster_analysis['平均M'].idxmax(): '高价值用户',
            cluster_analysis['平均F'].idxmax(): '忠诚用户',
            (n_clusters - 1) if (n_clusters - 1) not in [cluster_analysis['平均M'].idxmax(), cluster_analysis['平均F'].idxmax()] else 0: '流失预警用户',
        }
        
        for i in range(n_clusters):
            if i not in cluster_names:
                cluster_names[i] = '普通用户'
        
        cluster_analysis['用户类型'] = cluster_analysis.index.map(cluster_names)
        
        self.analysis_results['用户聚类'] = {
            '聚类中心': kmeans.cluster_centers_.tolist(),
            '聚类详情': cluster_analysis.to_dict('index'),
            '用户聚类结果': self.rfm_df.to_dict('records')
        }
        
        print("聚类结果:")
        print(cluster_analysis)
        
        return self.rfm_df, cluster_analysis
    
    def sales_prediction(self):
        """销售预测 - 线性回归"""
        print("=" * 50)
        print("执行销售预测...")
        print("=" * 50)
        
        daily_sales = self.df.groupby('日期')['订单金额'].sum().reset_index()
        daily_sales['日期'] = pd.to_datetime(daily_sales['日期'])
        daily_sales['日序号'] = (daily_sales['日期'] - daily_sales['日期'].min()).dt.days
        daily_sales['星期'] = daily_sales['日期'].dt.dayofweek
        daily_sales['月份'] = daily_sales['日期'].dt.month
        
        X = daily_sales[['日序号', '星期', '月份']]
        y = daily_sales['订单金额']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        daily_sales['预测销售额'] = model.predict(X)
        
        self.analysis_results['销售预测'] = {
            'MSE': round(mse, 2),
            'RMSE': round(rmse, 2),
            'R2': round(r2, 3),
            '系数': dict(zip(X.columns, model.coef_.round(2))),
            '预测数据': daily_sales[['日期', '订单金额', '预测销售额']].to_dict('records')
        }
        
        print(f"预测模型评估: RMSE={round(rmse, 2)}, R2={round(r2, 3)}")
        
        return self.analysis_results['销售预测']
    
    def run_full_analysis(self):
        """执行完整分析流程"""
        self.descriptive_statistics()
        self.correlation_analysis()
        self.time_series_analysis()
        self.customer_clustering(n_clusters=4)
        self.sales_prediction()
        
        print("=" * 50)
        print("所有分析完成!")
        print("=" * 50)
        
        return self.analysis_results

if __name__ == '__main__':
    from data_generator import generate_ecommerce_data
    from data_cleaner import DataCleaner
    
    df_raw = generate_ecommerce_data(n_records=2000)
    cleaner = DataCleaner(df_raw)
    df_cleaned = cleaner.clean()
    
    analyzer = DataAnalyzer(df_cleaned)
    results = analyzer.run_full_analysis()
