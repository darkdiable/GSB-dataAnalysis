import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesAnalyzer:
    def __init__(self, freq='H'):
        self.freq = freq
        self.decomposition = None
        self.adf_result = None
        
    def prepare_daily_data(self, df):
        daily_data = df.groupby('date')['riders'].sum().reset_index()
        daily_data['date'] = pd.to_datetime(daily_data['date'])
        daily_data = daily_data.set_index('date').asfreq('D')
        return daily_data['riders']
    
    def prepare_hourly_data(self, df):
        hourly_data = df.groupby('timestamp')['riders'].sum().reset_index()
        hourly_data = hourly_data.set_index('timestamp')
        return hourly_data['riders']
    
    def decompose_series(self, series, model='additive', period=7):
        print(f"开始时间序列分解，周期={period}...")
        self.decomposition = seasonal_decompose(series, model=model, period=period)
        print("时间序列分解完成")
        return self.decomposition
    
    def test_stationarity(self, series):
        print("开始ADF平稳性检验...")
        result = adfuller(series.dropna())
        self.adf_result = {
            'ADF统计量': round(result[0], 4),
            'p值': round(result[1], 4),
            '临界值': {k: round(v, 4) for k, v in result[4].items()}
        }
        
        print(f"ADF统计量: {self.adf_result['ADF统计量']}")
        print(f"p值: {self.adf_result['p值']}")
        
        if self.adf_result['p值'] < 0.05:
            print("结论: 序列是平稳的 (p < 0.05)")
        else:
            print("结论: 序列是非平稳的")
        
        return self.adf_result
    
    def analyze_periodicity(self, df):
        print("=" * 60)
        print("时间序列周期性与趋势分析")
        print("=" * 60)
        
        daily_series = self.prepare_daily_data(df)
        hourly_series = self.prepare_hourly_data(df)
        
        daily_decomp = self.decompose_series(daily_series, period=7)
        trend_strength = np.var(daily_decomp.trend.dropna()) / np.var(daily_series)
        seasonal_strength = np.var(daily_decomp.seasonal) / np.var(daily_series)
        
        print(f"\n日度序列分析结果:")
        print(f"  - 趋势强度: {round(trend_strength, 4)}")
        print(f"  - 周周期强度: {round(seasonal_strength, 4)}")
        
        hourly_decomp = self.decompose_series(hourly_series, period=24)
        hourly_seasonal_strength = np.var(hourly_decomp.seasonal) / np.var(hourly_series)
        print(f"\n小时序列分析结果:")
        print(f"  - 日周期强度: {round(hourly_seasonal_strength, 4)}")
        
        adf_result = self.test_stationarity(daily_series)
        
        return {
            'daily_trend_strength': trend_strength,
            'weekly_seasonal_strength': seasonal_strength,
            'daily_seasonal_strength': hourly_seasonal_strength,
            'adf_test': adf_result,
            'decomposition': daily_decomp
        }


class DemandPredictor:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.feature_importance = None
        self.feature_names = None
        
    def prepare_features(self, df):
        feature_cols = [
            'hour', 'weekday', 'is_weekend', 'temperature', 'humidity', 
            'wind_speed', 'weather_severity', 'is_rush_hour_morning', 
            'is_rush_hour_evening', 'station_id'
        ]
        
        weather_cols = [col for col in df.columns if col.startswith('weather_')]
        feature_cols.extend(weather_cols)
        
        X = df[feature_cols].copy()
        y = df['riders']
        
        self.feature_names = feature_cols
        return X, y, feature_cols
    
    def train_model(self, X, y, n_estimators=100, max_depth=None, test_size=0.2):
        print("=" * 60)
        print("训练随机森林回归模型")
        print("=" * 60)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")
        
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        metrics = self._calculate_metrics(y_train, y_pred_train, y_test, y_pred_test)
        self._print_metrics(metrics)
        
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).reset_index(drop=True)
        
        print("\n特征重要性排名:")
        for idx, row in self.feature_importance.head(10).iterrows():
            print(f"  {idx+1}. {row['feature']}: {round(row['importance'], 4)}")
        
        return metrics, X_train, X_test, y_train, y_test, y_pred_train, y_pred_test
    
    def _calculate_metrics(self, y_train, y_pred_train, y_test, y_pred_test):
        return {
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test)
        }
    
    def _print_metrics(self, metrics):
        print("\n模型评估指标:")
        print(f"  训练集 R²: {round(metrics['train_r2'], 4)}")
        print(f"  测试集 R²: {round(metrics['test_r2'], 4)}")
        print(f"  训练集 RMSE: {round(metrics['train_rmse'], 4)}")
        print(f"  测试集 RMSE: {round(metrics['test_rmse'], 4)}")
        print(f"  训练集 MAE: {round(metrics['train_mae'], 4)}")
        print(f"  测试集 MAE: {round(metrics['test_mae'], 4)}")
    
    def cross_validation(self, X, y, cv=5):
        print(f"\n开始 {cv} 折交叉验证...")
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='r2', n_jobs=-1)
        print(f"交叉验证 R² 分数: {[round(s, 4) for s in scores]}")
        print(f"平均 R²: {round(scores.mean(), 4)} (±{round(scores.std(), 4)})")
        return scores
    
    def get_modeling_report(self, metrics, cv_scores=None):
        report = []
        report.append("=" * 60)
        report.append("建模分析报告")
        report.append("=" * 60)
        
        report.append("\n1. 模型性能指标:")
        report.append(f"   训练集 R²: {round(metrics['train_r2'], 4)}")
        report.append(f"   测试集 R²: {round(metrics['test_r2'], 4)}")
        report.append(f"   训练集 RMSE: {round(metrics['train_rmse'], 2)}")
        report.append(f"   测试集 RMSE: {round(metrics['test_rmse'], 2)}")
        
        if cv_scores is not None:
            report.append(f"\n2. {len(cv_scores)} 折交叉验证结果:")
            report.append(f"   平均 R²: {round(cv_scores.mean(), 4)} (±{round(cv_scores.std(), 4)})")
        
        report.append("\n3. 特征重要性排名 TOP 10:")
        for idx, row in self.feature_importance.head(10).iterrows():
            report.append(f"   {idx+1}. {row['feature']}: {round(row['importance'], 4)}")
        
        report.append("=" * 60)
        return "\n".join(report)


if __name__ == '__main__':
    from data_generator import BikeShareDataGenerator
    from preprocessor import DataPreprocessor
    
    generator = BikeShareDataGenerator(days=90)
    df_raw = generator.generate_dataset()
    
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.preprocess_pipeline(df_raw)
    
    ts_analyzer = TimeSeriesAnalyzer()
    ts_results = ts_analyzer.analyze_periodicity(df_processed)
    
    predictor = DemandPredictor()
    X, y, feature_cols = predictor.prepare_features(df_processed)
    metrics, *_ = predictor.train_model(X, y)
    cv_scores = predictor.cross_validation(X, y)
    
    print(predictor.get_modeling_report(metrics, cv_scores))
