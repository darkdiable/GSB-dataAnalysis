import pandas as pd
import numpy as np
from typing import Dict, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
import warnings
warnings.filterwarnings('ignore')


class BikeDemandModel:
    
    def __init__(self):
        self.rf_model = None
        self.ts_result = None
        self.feature_importance = None
        self.model_metrics = {}
        self.ts_analysis = {}
    
    def time_series_analysis(self, df: pd.DataFrame, 
                            date_col: str = 'timestamp',
                            value_col: str = 'ridership') -> Dict:
        ts_data = df.set_index(date_col)[value_col].resample('D').mean()
        ts_data = ts_data.dropna()
        
        decomposition = seasonal_decompose(ts_data, model='additive', period=7)
        
        self.ts_analysis['trend'] = decomposition.trend
        self.ts_analysis['seasonal'] = decomposition.seasonal
        self.ts_analysis['residual'] = decomposition.resid
        self.ts_analysis['observed'] = decomposition.observed
        
        adf_result = adfuller(ts_data.dropna())
        self.ts_analysis['adf_test'] = {
            'statistic': adf_result[0],
            'p_value': adf_result[1],
            'is_stationary': adf_result[1] < 0.05
        }
        
        acf_values = acf(ts_data.dropna(), nlags=24)
        pacf_values = pacf(ts_data.dropna(), nlags=24)
        self.ts_analysis['acf'] = acf_values
        self.ts_analysis['pacf'] = pacf_values
        
        self.ts_analysis['summary'] = {
            'trend_direction': '上升' if decomposition.trend.dropna().iloc[-1] > 
                              decomposition.trend.dropna().iloc[0] else '下降',
            'seasonal_period': '周周期性明显' if decomposition.seasonal.std() > 10 
                              else '周周期性较弱',
            'is_stationary': self.ts_analysis['adf_test']['is_stationary']
        }
        
        return self.ts_analysis
    
    def train_random_forest(self, X: pd.DataFrame, y: pd.Series,
                           test_size: float = 0.2,
                           random_state: int = 42) -> Dict:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        )
        
        self.rf_model.fit(X_train, y_train)
        
        y_pred_train = self.rf_model.predict(X_train)
        y_pred_test = self.rf_model.predict(X_test)
        
        self.model_metrics = {
            'train': {
                'mse': mean_squared_error(y_train, y_pred_train),
                'mae': mean_absolute_error(y_train, y_pred_train),
                'r2': r2_score(y_train, y_pred_train),
                'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train))
            },
            'test': {
                'mse': mean_squared_error(y_test, y_pred_test),
                'mae': mean_absolute_error(y_test, y_pred_test),
                'r2': r2_score(y_test, y_pred_test),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test))
            }
        }
        
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return self.model_metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        return self.feature_importance
    
    def get_time_series_summary(self) -> str:
        summary = self.ts_analysis.get('summary', {})
        lines = [
            "时间序列分析结果:",
            f"  - 趋势方向: {summary.get('trend_direction', 'N/A')}",
            f"  - 周期性: {summary.get('seasonal_period', 'N/A')}",
            f"  - 序列平稳性: {'平稳' if summary.get('is_stationary') else '非平稳'}"
        ]
        return '\n'.join(lines)
    
    def get_model_summary(self) -> str:
        lines = [
            "随机森林模型评估指标:",
            "  训练集:",
            f"    - RMSE: {self.model_metrics['train']['rmse']:.2f}",
            f"    - MAE: {self.model_metrics['train']['mae']:.2f}",
            f"    - R²: {self.model_metrics['train']['r2']:.4f}",
            "  测试集:",
            f"    - RMSE: {self.model_metrics['test']['rmse']:.2f}",
            f"    - MAE: {self.model_metrics['test']['mae']:.2f}",
            f"    - R²: {self.model_metrics['test']['r2']:.4f}"
        ]
        return '\n'.join(lines)
    
    def get_top_features(self, top_n: int = 10) -> str:
        top_features = self.feature_importance.head(top_n)
        lines = [f"特征重要性排名 (Top {top_n}):"]
        for idx, row in top_features.iterrows():
            lines.append(f"  {row['feature']}: {row['importance']:.4f}")
        return '\n'.join(lines)
