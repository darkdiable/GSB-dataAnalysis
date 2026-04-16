import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    def __init__(self, features, target_column, date_column):
        self.features = features.copy()
        self.target_column = target_column
        self.date_column = date_column
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.model_metrics = {}

    def prepare_data(self, test_size=0.2):
        feature_cols = [col for col in self.features.columns 
                       if col not in [self.date_column, self.target_column]]
        
        self.features = self.features.dropna()
        
        X = self.features[feature_cols]
        y = self.features[self.target_column]
        
        self.feature_names = feature_cols
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")
        print(f"特征数量: {len(self.feature_names)}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test

    def train_model(self, model_type='random_forest', **kwargs):
        X_train, X_test, y_train, y_test = self.prepare_data()
        
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 10),
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 5),
                learning_rate=kwargs.get('learning_rate', 0.1),
                random_state=42
            )
        elif model_type == 'ridge':
            self.model = Ridge(alpha=kwargs.get('alpha', 1.0))
        else:
            self.model = LinearRegression()
        
        self.model.fit(X_train, y_train)
        
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        self.model_metrics = {
            'train': {
                'mse': mean_squared_error(y_train, y_pred_train),
                'mae': mean_absolute_error(y_train, y_pred_train),
                'r2': r2_score(y_train, y_pred_train)
            },
            'test': {
                'mse': mean_squared_error(y_test, y_pred_test),
                'mae': mean_absolute_error(y_test, y_pred_test),
                'r2': r2_score(y_test, y_pred_test)
            }
        }
        
        print(f"\n模型训练完成 ({model_type})")
        print(f"训练集 - R²: {self.model_metrics['train']['r2']:.4f}, MAE: {self.model_metrics['train']['mae']:.2f}")
        print(f"测试集 - R²: {self.model_metrics['test']['r2']:.4f}, MAE: {self.model_metrics['test']['mae']:.2f}")
        
        return self.model

    def get_feature_importance(self):
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_)
        else:
            return None
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance

    def predict_future(self, days=30):
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        last_date = pd.to_datetime(self.features[self.date_column].max())
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days)
        
        future_df = pd.DataFrame({self.date_column: future_dates})
        
        future_df['year'] = future_df[self.date_column].dt.year
        future_df['month'] = future_df[self.date_column].dt.month
        future_df['day'] = future_df[self.date_column].dt.day
        future_df['dayofweek'] = future_df[self.date_column].dt.dayofweek
        future_df['dayofyear'] = future_df[self.date_column].dt.dayofyear
        future_df['weekofyear'] = future_df[self.date_column].dt.isocalendar().week.astype(int)
        future_df['quarter'] = future_df[self.date_column].dt.quarter
        future_df['is_weekend'] = (future_df[self.date_column].dt.dayofweek >= 5).astype(int)
        future_df['is_month_start'] = future_df[self.date_column].dt.is_month_start.astype(int)
        future_df['is_month_end'] = future_df[self.date_column].dt.is_month_end.astype(int)
        
        future_df['season'] = future_df['month'].apply(self._get_season)
        future_df['is_holiday_season'] = future_df['month'].apply(lambda x: 1 if x in [11, 12, 1] else 0)
        
        last_sales = self.features[self.target_column].iloc[-30:].values
        for lag in [1, 7, 14, 30]:
            if lag <= len(last_sales):
                future_df[f'sales_lag_{lag}'] = last_sales[-lag]
            else:
                future_df[f'sales_lag_{lag}'] = last_sales[-1]
        
        for window in [7, 14, 30]:
            future_df[f'sales_rolling_mean_{window}'] = np.mean(last_sales[-window:]) if window <= len(last_sales) else np.mean(last_sales)
            future_df[f'sales_rolling_std_{window}'] = np.std(last_sales[-window:]) if window <= len(last_sales) else np.std(last_sales)
            future_df[f'sales_rolling_min_{window}'] = np.min(last_sales[-window:]) if window <= len(last_sales) else np.min(last_sales)
            future_df[f'sales_rolling_max_{window}'] = np.max(last_sales[-window:]) if window <= len(last_sales) else np.max(last_sales)
        
        trend_value = self.features['trend'].iloc[-1] if 'trend' in self.features.columns else 0
        seasonal_value = self.features['seasonal'].iloc[-1] if 'seasonal' in self.features.columns else 0
        residual_value = self.features['residual'].iloc[-1] if 'residual' in self.features.columns else 0
        
        future_df['trend'] = trend_value
        future_df['seasonal'] = seasonal_value
        future_df['residual'] = residual_value
        
        X_future = future_df[self.feature_names]
        X_future_scaled = self.scaler.transform(X_future)
        
        predictions = self.model.predict(X_future_scaled)
        
        future_df['predicted_sales'] = predictions
        
        print(f"\n未来 {days} 天销售预测完成")
        print(f"预测销售额范围: {predictions.min():.2f} - {predictions.max():.2f}")
        print(f"平均预测销售额: {predictions.mean():.2f}")
        
        return future_df[[self.date_column, 'predicted_sales']]

    def _get_season(self, month):
        if month in [3, 4, 5]:
            return 1
        elif month in [6, 7, 8]:
            return 2
        elif month in [9, 10, 11]:
            return 3
        else:
            return 4

    def get_model_metrics(self):
        return self.model_metrics
