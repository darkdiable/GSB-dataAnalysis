import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SalesPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        
    def prepare_time_series_data(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info("准备时间序列预测数据...")
            
            monthly_sales = df.groupby('year_month').agg({
                'total_amount': 'sum',
                'order_id': 'nunique',
                'quantity': 'sum',
                'customer_id': 'nunique'
            }).reset_index()
            
            monthly_sales['year'] = monthly_sales['year_month'].dt.year
            monthly_sales['month'] = monthly_sales['year_month'].dt.month
            monthly_sales['quarter'] = monthly_sales['year_month'].dt.quarter
            
            for i in range(1, 4):
                monthly_sales[f'sales_lag_{i}'] = monthly_sales['total_amount'].shift(i)
                monthly_sales[f'orders_lag_{i}'] = monthly_sales['order_id'].shift(i)
            
            monthly_sales['sales_ma3'] = monthly_sales['total_amount'].rolling(window=3).mean()
            monthly_sales['sales_ma6'] = monthly_sales['total_amount'].rolling(window=6).mean()
            
            monthly_sales = monthly_sales.dropna()
            
            logger.info(f"时间序列数据准备完成: {len(monthly_sales)} 条记录")
            return monthly_sales
            
        except Exception as e:
            logger.error(f"准备时间序列数据失败: {str(e)}")
            raise
            
    def train_model(self, monthly_sales: pd.DataFrame, model_type: str = 'rf') -> Dict[str, Any]:
        try:
            logger.info(f"训练{model_type}预测模型...")
            
            feature_cols = ['year', 'month', 'quarter', 'sales_lag_1', 'sales_lag_2', 
                           'sales_lag_3', 'orders_lag_1', 'sales_ma3', 'sales_ma6']
            
            X = monthly_sales[feature_cols]
            y = monthly_sales['total_amount']
            
            X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
                X, y, monthly_sales['year_month'], test_size=0.3, random_state=42, shuffle=False
            )
            
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            if model_type == 'rf':
                self.model = RandomForestRegressor(random_state=42)
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10]
                }
                grid_search = GridSearchCV(self.model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
                grid_search.fit(X_train_scaled, y_train)
                self.model = grid_search.best_estimator_
                logger.info(f"最佳参数: {grid_search.best_params_}")
                
                self.feature_importance = pd.DataFrame({
                    'feature': feature_cols,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)
                
            else:
                self.model = LinearRegression()
                self.model.fit(X_train_scaled, y_train)
            
            y_pred_train = self.model.predict(X_train_scaled)
            y_pred_test = self.model.predict(X_test_scaled)
            
            metrics = {
                'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'train_mae': mean_absolute_error(y_train, y_pred_train),
                'test_mae': mean_absolute_error(y_test, y_pred_test),
                'train_r2': r2_score(y_train, y_pred_train),
                'test_r2': r2_score(y_test, y_pred_test)
            }
            
            logger.info(f"""模型训练完成:
                训练集RMSE: {metrics['train_rmse']:.2f}
                测试集RMSE: {metrics['test_rmse']:.2f}
                训练集R2: {metrics['train_r2']:.4f}
                测试集R2: {metrics['test_r2']:.4f}
            """)
            
            return {
                'metrics': metrics,
                'y_train': y_train.values,
                'y_train_pred': y_pred_train,
                'y_test': y_test.values,
                'y_test_pred': y_pred_test,
                'dates_test': dates_test,
                'feature_importance': self.feature_importance
            }
            
        except Exception as e:
            logger.error(f"训练模型失败: {str(e)}")
            raise
            
    def predict_next_months(self, monthly_sales: pd.DataFrame, months: int = 3) -> pd.DataFrame:
        try:
            logger.info(f"预测未来 {months} 个月的销售...")
            
            if self.model is None:
                raise ValueError("模型尚未训练，请先调用train_model()")
            
            feature_cols = ['year', 'month', 'quarter', 'sales_lag_1', 'sales_lag_2', 
                           'sales_lag_3', 'orders_lag_1', 'sales_ma3', 'sales_ma6']
            
            last_row = monthly_sales.iloc[-1].copy()
            predictions = []
            current_date = last_row['year_month']
            
            for i in range(months):
                current_date = current_date + 1
                next_row = pd.Series(dtype='float64')
                next_row['year'] = current_date.year
                next_row['month'] = current_date.month
                next_row['quarter'] = (current_date.month - 1) // 3 + 1
                
                if i == 0:
                    next_row['sales_lag_1'] = last_row['total_amount']
                    next_row['sales_lag_2'] = last_row['sales_lag_1']
                    next_row['sales_lag_3'] = last_row['sales_lag_2']
                    next_row['orders_lag_1'] = last_row['order_id']
                    sales_history = [last_row['sales_lag_2'], last_row['sales_lag_1'], last_row['total_amount']]
                elif i == 1:
                    next_row['sales_lag_1'] = predictions[-1]['predicted_sales']
                    next_row['sales_lag_2'] = last_row['total_amount']
                    next_row['sales_lag_3'] = last_row['sales_lag_1']
                    next_row['orders_lag_1'] = last_row['order_id'] * (0.95 + np.random.rand() * 0.1)
                    sales_history = [last_row['total_amount'], predictions[-1]['predicted_sales']]
                else:
                    next_row['sales_lag_1'] = predictions[-1]['predicted_sales']
                    next_row['sales_lag_2'] = predictions[-2]['predicted_sales']
                    next_row['sales_lag_3'] = predictions[-3]['predicted_sales'] if len(predictions) >= 3 else last_row['total_amount']
                    next_row['orders_lag_1'] = last_row['order_id'] * (0.95 + np.random.rand() * 0.1)
                    sales_history = [predictions[-2]['predicted_sales'], predictions[-1]['predicted_sales']]
                
                next_row['sales_ma3'] = np.mean(sales_history[-3:])
                next_row['sales_ma6'] = next_row['sales_ma3'] * (0.95 + np.random.rand() * 0.1)
                
                X_next = pd.DataFrame([next_row[feature_cols]])
                X_next_scaled = self.scaler.transform(X_next)
                predicted = self.model.predict(X_next_scaled)[0]
                
                predictions.append({
                    'year_month': str(current_date),
                    'predicted_sales': round(predicted, 2),
                    'predicted_growth': round((predicted / last_row['total_amount'] - 1) * 100, 2)
                })
            
            predictions_df = pd.DataFrame(predictions)
            logger.info(f"未来销售预测:\n{predictions_df}")
            return predictions_df
            
        except Exception as e:
            logger.error(f"预测未来销售失败: {str(e)}")
            raise
            
    def run_prediction_pipeline(self, df: pd.DataFrame) -> Dict[str, Any]:
        try:
            logger.info("开始销售预测流程...")
            
            monthly_sales = self.prepare_time_series_data(df)
            train_results = self.train_model(monthly_sales, model_type='rf')
            forecast = self.predict_next_months(monthly_sales, months=3)
            
            actual_vs_predicted = {
                'actual': train_results['y_test'],
                'predicted': train_results['y_test_pred'],
                'dates': train_results['dates_test']
            }
            
            result = {
                'monthly_sales': monthly_sales,
                'metrics': train_results['metrics'],
                'feature_importance': train_results['feature_importance'],
                'actual_vs_predicted': actual_vs_predicted,
                'forecast': forecast
            }
            
            logger.info("销售预测流程完成")
            return result
            
        except Exception as e:
            logger.error(f"销售预测流程失败: {str(e)}")
            raise

if __name__ == '__main__':
    from data_loader import DataLoader
    
    loader = DataLoader()
    loader.load_all_data()
    merged = loader.merge_data()
    
    predictor = SalesPredictor()
    results = predictor.run_prediction_pipeline(merged)
    
    print("预测指标:", results['metrics'])
    print("未来预测:\n", results['forecast'])
    print("特征重要性:\n", results['feature_importance'])
