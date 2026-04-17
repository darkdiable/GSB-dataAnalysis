"""
建模分析模块：时间序列分析和需求预测模型
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesAnalyzer:
    """时间序列分析器"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.decomposition = None
        self.analysis_results = {}
    
    def prepare_daily_data(self) -> pd.DataFrame:
        """准备每日聚合数据"""
        daily_df = self.df.groupby(
            self.df['timestamp'].dt.date
        ).agg({
            'ridership': 'sum',
            'temperature': 'mean',
            'humidity': 'mean',
            'wind_speed': 'mean'
        }).reset_index()
        daily_df.columns = ['date', 'ridership', 'temperature', 'humidity', 'wind_speed']
        daily_df['date'] = pd.to_datetime(daily_df['date'])
        daily_df.set_index('date', inplace=True)
        return daily_df
    
    def decompose_time_series(self, period: int = 7) -> dict:
        """时间序列分解"""
        daily_df = self.prepare_daily_data()
        
        self.decomposition = seasonal_decompose(
            daily_df['ridership'], 
            model='additive', 
            period=period
        )
        
        trend_strength = 1 - (np.var(self.decomposition.resid.dropna()) / 
                             np.var(self.decomposition.trend.dropna() + self.decomposition.resid.dropna()))
        seasonal_strength = 1 - (np.var(self.decomposition.resid.dropna()) / 
                                np.var(self.decomposition.seasonal.dropna() + self.decomposition.resid.dropna()))
        
        self.analysis_results['decomposition'] = {
            'trend_strength': round(trend_strength, 4),
            'seasonal_strength': round(seasonal_strength, 4),
            'period': period
        }
        
        return self.analysis_results['decomposition']
    
    def adf_test(self) -> dict:
        """ADF平稳性检验"""
        daily_df = self.prepare_daily_data()
        result = adfuller(daily_df['ridership'].dropna())
        
        self.analysis_results['adf_test'] = {
            'adf_statistic': round(result[0], 4),
            'p_value': round(result[1], 4),
            'is_stationary': result[1] < 0.05,
            'critical_values': {k: round(v, 4) for k, v in result[4].items()}
        }
        
        return self.analysis_results['adf_test']
    
    def analyze_hourly_pattern(self) -> dict:
        """分析小时骑行模式"""
        hourly_pattern = self.df.groupby('hour')['ridership'].agg(['mean', 'std', 'max', 'min'])
        
        peak_hour = hourly_pattern['mean'].idxmax()
        valley_hour = hourly_pattern['mean'].idxmin()
        
        self.analysis_results['hourly_pattern'] = {
            'peak_hour': int(peak_hour),
            'peak_avg_ridership': round(hourly_pattern.loc[peak_hour, 'mean'], 2),
            'valley_hour': int(valley_hour),
            'valley_avg_ridership': round(hourly_pattern.loc[valley_hour, 'mean'], 2),
            'hourly_stats': hourly_pattern.to_dict()
        }
        
        return self.analysis_results['hourly_pattern']
    
    def analyze_weekly_pattern(self) -> dict:
        """分析周骑行模式"""
        weekly_pattern = self.df.groupby('weekday')['ridership'].mean()
        
        weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
        
        self.analysis_results['weekly_pattern'] = {
            'daily_avg': {weekday_names[i]: round(weekly_pattern[i], 2) for i in range(7)},
            'weekend_avg': round(weekly_pattern[5:7].mean(), 2),
            'workday_avg': round(weekly_pattern[0:5].mean(), 2)
        }
        
        return self.analysis_results['weekly_pattern']
    
    def get_analysis_summary(self) -> str:
        """获取分析摘要"""
        summary = []
        summary.append("=" * 50)
        summary.append("时间序列分析结果")
        summary.append("=" * 50)
        
        if 'decomposition' in self.analysis_results:
            dec = self.analysis_results['decomposition']
            summary.append(f"\n趋势强度: {dec['trend_strength']}")
            summary.append(f"季节性强度: {dec['seasonal_strength']}")
            summary.append(f"周期: {dec['period']}天")
        
        if 'adf_test' in self.analysis_results:
            adf = self.analysis_results['adf_test']
            summary.append(f"\nADF检验统计量: {adf['adf_statistic']}")
            summary.append(f"p值: {adf['p_value']}")
            summary.append(f"序列是否平稳: {'是' if adf['is_stationary'] else '否'}")
        
        if 'hourly_pattern' in self.analysis_results:
            hp = self.analysis_results['hourly_pattern']
            summary.append(f"\n高峰时段: {hp['peak_hour']}:00 (平均{hp['peak_avg_ridership']}人次)")
            summary.append(f"低谷时段: {hp['valley_hour']}:00 (平均{hp['valley_avg_ridership']}人次)")
        
        if 'weekly_pattern' in self.analysis_results:
            wp = self.analysis_results['weekly_pattern']
            summary.append(f"\n工作日平均: {wp['weekday_avg']}人次")
            summary.append(f"周末平均: {wp['weekend_avg']}人次")
        
        return '\n'.join(summary)


class DemandPredictor:
    """骑行需求预测器"""
    
    def __init__(self, df: pd.DataFrame, feature_columns: list):
        self.df = df.copy()
        self.feature_columns = feature_columns
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.predictions = None
        self.metrics = {}
        self.feature_importance = None
    
    def prepare_data(self, test_size: float = 0.2, random_state: int = 42):
        """准备训练和测试数据"""
        X = self.df[self.feature_columns]
        y = self.df['ridership']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_model(self, n_estimators: int = 100, max_depth: int = 10, 
                   random_state: int = 42):
        """训练随机森林模型"""
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        
        self.model.fit(self.X_train, self.y_train)
        
        self.predictions = self.model.predict(self.X_test)
        
        self.metrics = {
            'mse': round(mean_squared_error(self.y_test, self.predictions), 2),
            'rmse': round(np.sqrt(mean_squared_error(self.y_test, self.predictions)), 2),
            'mae': round(mean_absolute_error(self.y_test, self.predictions), 2),
            'r2': round(r2_score(self.y_test, self.predictions), 4)
        }
        
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.feature_importance = importance_df
        
        return self.metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """获取特征重要性"""
        return self.feature_importance
    
    def get_model_summary(self) -> str:
        """获取模型摘要"""
        summary = []
        summary.append("=" * 50)
        summary.append("随机森林回归模型结果")
        summary.append("=" * 50)
        summary.append(f"\n模型参数:")
        summary.append(f"  - 树的数量: {self.model.n_estimators}")
        summary.append(f"  - 最大深度: {self.model.max_depth}")
        summary.append(f"\n模型性能:")
        summary.append(f"  - MSE: {self.metrics['mse']}")
        summary.append(f"  - RMSE: {self.metrics['rmse']}")
        summary.append(f"  - MAE: {self.metrics['mae']}")
        summary.append(f"  - R²: {self.metrics['r2']}")
        summary.append(f"\n特征重要性排序:")
        
        for idx, row in self.feature_importance.head(10).iterrows():
            summary.append(f"  - {row['feature']}: {row['importance']:.4f}")
        
        return '\n'.join(summary)


if __name__ == '__main__':
    df = pd.read_csv('data/processed_data.csv')
    
    ts_analyzer = TimeSeriesAnalyzer(df)
    ts_analyzer.decompose_time_series()
    ts_analyzer.adf_test()
    ts_analyzer.analyze_hourly_pattern()
    print(ts_analyzer.get_analysis_summary())
    
    feature_cols = [col for col in df.columns if col.startswith('weather_') or 
                   col in ['temperature', 'humidity', 'wind_speed', 'hour', 'month', 'weekday', 'is_weekend']]
    
    predictor = DemandPredictor(df, feature_cols)
    predictor.prepare_data()
    predictor.train_model()
    print(predictor.get_model_summary())
