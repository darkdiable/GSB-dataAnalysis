import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class SalesPredictor:
    def __init__(self, daily_sales: pd.DataFrame):
        self.daily_sales = daily_sales.copy()
        self.daily_sales['date'] = pd.to_datetime(self.daily_sales['date'])
        self.daily_sales = self.daily_sales.sort_values('date').reset_index(drop=True)
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.metrics = {}
        
    def create_features(self) -> pd.DataFrame:
        df = self.daily_sales.copy()
        
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_year'] = df['date'].dt.dayofyear
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['quarter'] = df['date'].dt.quarter
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
        
        for lag in [1, 2, 3, 7, 14, 30]:
            df[f'revenue_lag_{lag}'] = df['daily_revenue'].shift(lag)
        
        for window in [7, 14, 30]:
            df[f'revenue_rolling_mean_{window}'] = df['daily_revenue'].rolling(window=window).mean()
            df[f'revenue_rolling_std_{window}'] = df['daily_revenue'].rolling(window=window).std()
        
        df['revenue_diff_1'] = df['daily_revenue'].diff(1)
        df['revenue_diff_7'] = df['daily_revenue'].diff(7)
        
        df = df.dropna()
        
        return df
    
    def prepare_data(self, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        df = self.create_features()
        
        self.feature_columns = [col for col in df.columns 
                               if col not in ['date', 'daily_revenue', 'year_month']]
        
        X = df[self.feature_columns]
        y = df['daily_revenue']
        
        split_idx = int(len(df) * (1 - test_size))
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=self.feature_columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=self.feature_columns,
            index=X_test.index
        )
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_model(self, model_type: str = 'random_forest') -> Dict:
        X_train, X_test, y_train, y_test = self.prepare_data()
        
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                min_samples_split=5,
                random_state=42
            )
        elif model_type == 'ridge':
            self.model = Ridge(alpha=1.0, random_state=42)
        else:
            self.model = LinearRegression()
        
        self.model.fit(X_train, y_train)
        
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        self.metrics = {
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
        
        return self.metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("请先训练模型")
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_)
        else:
            return pd.DataFrame()
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance
        })
        
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        return feature_importance
    
    def predict_future(self, days: int = 30) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("请先训练模型")
        
        last_date = self.daily_sales['date'].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days)
        
        predictions = []
        current_data = self.daily_sales.copy()
        
        for future_date in future_dates:
            temp_df = current_data.copy()
            temp_df['date'] = pd.to_datetime(temp_df['date'])
            
            features = self._create_single_day_features(temp_df, future_date)
            
            features_scaled = self.scaler.transform([features])
            
            pred = self.model.predict(features_scaled)[0]
            
            predictions.append({
                'date': future_date,
                'predicted': pred
            })
            
            new_row = pd.DataFrame({
                'date': [future_date],
                'daily_revenue': [pred]
            })
            current_data = pd.concat([current_data, new_row], ignore_index=True)
        
        predictions_df = pd.DataFrame(predictions)
        
        std_error = self.metrics['test']['rmse']
        predictions_df['predicted_lower'] = predictions_df['predicted'] - 1.96 * std_error
        predictions_df['predicted_upper'] = predictions_df['predicted'] + 1.96 * std_error
        
        predictions_df['predicted_lower'] = predictions_df['predicted_lower'].clip(lower=0)
        
        return predictions_df
    
    def _create_single_day_features(self, historical_data: pd.DataFrame, target_date: pd.Timestamp) -> list:
        features = {}
        
        features['year'] = target_date.year
        features['month'] = target_date.month
        features['day'] = target_date.day
        features['day_of_week'] = target_date.dayofweek
        features['day_of_year'] = target_date.dayofyear
        features['week_of_year'] = target_date.isocalendar().week
        features['quarter'] = target_date.quarter
        features['is_weekend'] = int(target_date.dayofweek in [5, 6])
        features['is_month_start'] = int(target_date.is_month_start)
        features['is_month_end'] = int(target_date.is_month_end)
        
        revenue = historical_data['daily_revenue'].values
        
        for lag in [1, 2, 3, 7, 14, 30]:
            if len(revenue) >= lag:
                features[f'revenue_lag_{lag}'] = revenue[-lag]
            else:
                features[f'revenue_lag_{lag}'] = revenue[-1]
        
        for window in [7, 14, 30]:
            if len(revenue) >= window:
                features[f'revenue_rolling_mean_{window}'] = np.mean(revenue[-window:])
                features[f'revenue_rolling_std_{window}'] = np.std(revenue[-window:])
            else:
                features[f'revenue_rolling_mean_{window}'] = np.mean(revenue)
                features[f'revenue_rolling_std_{window}'] = np.std(revenue)
        
        if len(revenue) >= 2:
            features['revenue_diff_1'] = revenue[-1] - revenue[-2]
        else:
            features['revenue_diff_1'] = 0
        
        if len(revenue) >= 8:
            features['revenue_diff_7'] = revenue[-1] - revenue[-8]
        else:
            features['revenue_diff_7'] = 0
        
        return [features[col] for col in self.feature_columns]
    
    def evaluate_model(self) -> Dict:
        if not self.metrics:
            raise ValueError("请先训练模型")
        
        return {
            'model_performance': self.metrics,
            'feature_importance': self.get_feature_importance().head(10).to_dict('records')
        }
    
    def get_prediction_summary(self, predictions_df: pd.DataFrame) -> Dict:
        return {
            'total_predicted_revenue': predictions_df['predicted'].sum(),
            'avg_predicted_revenue': predictions_df['predicted'].mean(),
            'max_predicted_revenue': predictions_df['predicted'].max(),
            'min_predicted_revenue': predictions_df['predicted'].min(),
            'prediction_period': f"{predictions_df['date'].min().strftime('%Y-%m-%d')} 至 {predictions_df['date'].max().strftime('%Y-%m-%d')}"
        }
    
    def compare_models(self) -> pd.DataFrame:
        models = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
        }
        
        X_train, X_test, y_train, y_test = self.prepare_data()
        
        results = []
        
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                results.append({
                    'model': name,
                    'mse': mean_squared_error(y_test, y_pred),
                    'mae': mean_absolute_error(y_test, y_pred),
                    'r2': r2_score(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
                })
            except Exception as e:
                print(f"模型 {name} 训练失败: {str(e)}")
        
        return pd.DataFrame(results).sort_values('rmse')
