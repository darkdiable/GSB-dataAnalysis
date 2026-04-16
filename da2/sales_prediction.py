"""
销售预测模块
使用scikit-learn构建销售预测模型
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta

from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SalesPredictor:
    """销售预测器"""

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_importance = None
        self.metrics = {}

    def prepare_time_series_data(self, df: pd.DataFrame, target_col: str = 'total_amount',
                                  freq: str = 'D') -> pd.DataFrame:
        """
        准备时间序列数据

        Args:
            df: 原始数据
            target_col: 目标列
            freq: 时间频率 ('D'日, 'W'周, 'M'月)
        """
        try:
            # 按日期聚合
            if freq == 'D':
                ts_data = df.groupby('date')[target_col].sum().reset_index()
            elif freq == 'W':
                ts_data = df.copy()
                ts_data['week'] = ts_data['date'].dt.to_period('W')
                ts_data = ts_data.groupby('week')[target_col].sum().reset_index()
                ts_data['date'] = ts_data['week'].dt.start_time
            elif freq == 'M':
                ts_data = df.copy()
                ts_data['month'] = ts_data['date'].dt.to_period('M')
                ts_data = ts_data.groupby('month')[target_col].sum().reset_index()
                ts_data['date'] = ts_data['month'].dt.start_time
            else:
                ts_data = df.groupby('date')[target_col].sum().reset_index()

            ts_data = ts_data.sort_values('date').reset_index(drop=True)
            ts_data.columns = ['date', 'sales']

            # 创建时间特征
            ts_data['year'] = ts_data['date'].dt.year
            ts_data['month'] = ts_data['date'].dt.month
            ts_data['day'] = ts_data['date'].dt.day
            ts_data['dayofweek'] = ts_data['date'].dt.dayofweek
            ts_data['dayofyear'] = ts_data['date'].dt.dayofyear
            ts_data['quarter'] = ts_data['date'].dt.quarter
            ts_data['is_month_start'] = ts_data['date'].dt.is_month_start.astype(int)
            ts_data['is_month_end'] = ts_data['date'].dt.is_month_end.astype(int)
            ts_data['is_weekend'] = (ts_data['dayofweek'] >= 5).astype(int)

            # 创建滞后特征
            for lag in [1, 7, 14, 30]:
                if len(ts_data) > lag:
                    ts_data[f'sales_lag_{lag}'] = ts_data['sales'].shift(lag)

            # 创建滚动统计特征
            for window in [7, 14, 30]:
                if len(ts_data) >= window:
                    ts_data[f'sales_rolling_mean_{window}'] = ts_data['sales'].rolling(window=window, min_periods=1).mean()
                    ts_data[f'sales_rolling_std_{window}'] = ts_data['sales'].rolling(window=window, min_periods=1).std()

            # 创建趋势特征
            ts_data['trend'] = np.arange(len(ts_data))

            # 填充缺失值
            ts_data = ts_data.fillna(method='bfill').fillna(method='ffill').fillna(0)

            logger.info(f"时间序列数据准备完成，共{len(ts_data)}条记录")
            return ts_data

        except Exception as e:
            logger.error(f"准备时间序列数据出错: {e}")
            return pd.DataFrame()

    def train_test_split_time_series(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple:
        """时间序列数据分割"""
        split_idx = int(len(df) * (1 - test_size))
        train = df.iloc[:split_idx]
        test = df.iloc[split_idx:]
        return train, test

    def build_features_target(self, df: pd.DataFrame, target_col: str = 'sales') -> Tuple:
        """构建特征和目标变量"""
        exclude_cols = ['date', target_col, 'week', 'month']
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        X = df[feature_cols]
        y = df[target_col]

        return X, y, feature_cols

    def train_linear_regression(self, X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
        """训练线性回归模型"""
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model

    def train_ridge(self, X_train: pd.DataFrame, y_train: pd.Series, alpha: float = 1.0) -> Ridge:
        """训练Ridge回归模型"""
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        return model

    def train_lasso(self, X_train: pd.DataFrame, y_train: pd.Series, alpha: float = 1.0) -> Lasso:
        """训练Lasso回归模型"""
        model = Lasso(alpha=alpha)
        model.fit(X_train, y_train)
        return model

    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series,
                            n_estimators: int = 100) -> RandomForestRegressor:
        """训练随机森林模型"""
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        return model

    def train_gradient_boosting(self, X_train: pd.DataFrame, y_train: pd.Series,
                                 n_estimators: int = 100) -> GradientBoostingRegressor:
        """训练梯度提升模型"""
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X_train, y_train)
        return model

    def evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series,
                       model_name: str) -> Dict:
        """评估模型性能"""
        try:
            y_pred = model.predict(X_test)

            metrics = {
                'mae': mean_absolute_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred),
                'mape': np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100
            }

            self.metrics[model_name] = metrics
            logger.info(f"{model_name} - MAE: {metrics['mae']:.2f}, RMSE: {metrics['rmse']:.2f}, R2: {metrics['r2']:.4f}")
            return metrics

        except Exception as e:
            logger.error(f"评估模型出错: {e}")
            return {}

    def cross_validate(self, model: Any, X: pd.DataFrame, y: pd.Series,
                       cv: int = 5) -> Dict:
        """交叉验证"""
        try:
            tscv = TimeSeriesSplit(n_splits=cv)
            scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_absolute_error')

            return {
                'cv_mae_mean': -scores.mean(),
                'cv_mae_std': scores.std()
            }
        except Exception as e:
            logger.error(f"交叉验证出错: {e}")
            return {}

    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                         X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """训练并评估所有模型"""
        logger.info("开始训练模型...")

        # 线性回归
        lr_model = self.train_linear_regression(X_train, y_train)
        self.models['linear_regression'] = lr_model
        self.evaluate_model(lr_model, X_test, y_test, 'linear_regression')

        # Ridge回归
        ridge_model = self.train_ridge(X_train, y_train)
        self.models['ridge'] = ridge_model
        self.evaluate_model(ridge_model, X_test, y_test, 'ridge')

        # Lasso回归
        lasso_model = self.train_lasso(X_train, y_train)
        self.models['lasso'] = lasso_model
        self.evaluate_model(lasso_model, X_test, y_test, 'lasso')

        # 随机森林
        rf_model = self.train_random_forest(X_train, y_train)
        self.models['random_forest'] = rf_model
        self.evaluate_model(rf_model, X_test, y_test, 'random_forest')

        # 梯度提升
        gb_model = self.train_gradient_boosting(X_train, y_train)
        self.models['gradient_boosting'] = gb_model
        self.evaluate_model(gb_model, X_test, y_test, 'gradient_boosting')

        # 获取特征重要性（随机森林）
        if hasattr(rf_model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)

        logger.info("所有模型训练完成")
        return self.models

    def predict(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """使用指定模型进行预测"""
        if model_name not in self.models:
            logger.error(f"模型 {model_name} 不存在")
            return np.array([])

        return self.models[model_name].predict(X)

    def forecast_future(self, df: pd.DataFrame, model_name: str = 'random_forest',
                        days: int = 30, freq: str = 'D') -> pd.DataFrame:
        """
        预测未来销售

        Args:
            df: 历史数据
            model_name: 使用的模型
            days: 预测天数
            freq: 频率
        """
        try:
            if model_name not in self.models:
                logger.error(f"模型 {model_name} 未训练")
                return pd.DataFrame()

            # 获取最后一条数据作为起点
            last_row = df.iloc[-1:].copy()
            last_date = last_row['date'].iloc[0]

            predictions = []
            current_df = df.copy()

            for i in range(1, days + 1):
                # 生成未来日期
                future_date = last_date + timedelta(days=i)

                # 构建特征
                future_row = pd.DataFrame({
                    'date': [future_date],
                    'sales': [current_df['sales'].iloc[-1]]  # 使用上一天的值作为初始值
                })

                # 添加时间特征
                future_row['year'] = future_date.year
                future_row['month'] = future_date.month
                future_row['day'] = future_date.day
                future_row['dayofweek'] = future_date.weekday()
                future_row['dayofyear'] = future_date.timetuple().tm_yday
                future_row['quarter'] = (future_date.month - 1) // 3 + 1
                future_row['is_month_start'] = int(future_date.day == 1)
                future_row['is_month_end'] = int(future_date.day == pd.Period(future_date, freq='M').days_in_month)
                future_row['is_weekend'] = int(future_date.weekday() >= 5)

                # 添加滞后特征
                for lag in [1, 7, 14, 30]:
                    if len(current_df) >= lag:
                        future_row[f'sales_lag_{lag}'] = current_df['sales'].iloc[-lag]
                    else:
                        future_row[f'sales_lag_{lag}'] = current_df['sales'].mean()

                # 添加滚动特征
                for window in [7, 14, 30]:
                    future_row[f'sales_rolling_mean_{window}'] = current_df['sales'].tail(window).mean()
                    future_row[f'sales_rolling_std_{window}'] = current_df['sales'].tail(window).std()

                future_row['trend'] = len(current_df) + i

                # 预测
                X_future, _, _ = self.build_features_target(future_row)
                pred = self.models[model_name].predict(X_future)[0]

                future_row['sales'] = pred
                future_row['predicted'] = True
                predictions.append(future_row)

                # 更新current_df用于下一个预测
                current_df = pd.concat([current_df, future_row], ignore_index=True)

            # 合并预测结果
            forecast_df = pd.concat(predictions, ignore_index=True)
            forecast_df = forecast_df[['date', 'sales', 'predicted']]
            forecast_df.columns = ['date', 'predicted_sales', 'is_prediction']

            logger.info(f"未来{days}天预测完成")
            return forecast_df

        except Exception as e:
            logger.error(f"预测未来销售出错: {e}")
            return pd.DataFrame()

    def get_best_model(self) -> Tuple[str, Any]:
        """获取最佳模型"""
        if not self.metrics:
            return None, None

        # 按R2排序
        best_model_name = max(self.metrics, key=lambda x: self.metrics[x].get('r2', -np.inf))
        return best_model_name, self.models.get(best_model_name)

    def get_metrics_summary(self) -> pd.DataFrame:
        """获取模型评估指标汇总"""
        if not self.metrics:
            return pd.DataFrame()

        summary = []
        for model_name, metrics in self.metrics.items():
            summary.append({
                'model': model_name,
                'mae': metrics.get('mae', 0),
                'rmse': metrics.get('rmse', 0),
                'r2': metrics.get('r2', 0),
                'mape': metrics.get('mape', 0)
            })

        return pd.DataFrame(summary)

    def run_prediction_pipeline(self, df: pd.DataFrame, target_col: str = 'total_amount',
                                 freq: str = 'D', test_size: float = 0.2) -> Dict:
        """运行完整的预测流程"""
        logger.info("开始销售预测流程...")

        # 准备数据
        ts_data = self.prepare_time_series_data(df, target_col, freq)
        if ts_data.empty:
            return {}

        # 分割数据
        train, test = self.train_test_split_time_series(ts_data, test_size)

        # 构建特征
        X_train, y_train, feature_cols = self.build_features_target(train)
        X_test, y_test, _ = self.build_features_target(test)

        # 训练模型
        self.train_all_models(X_train, y_train, X_test, y_test)

        # 预测未来
        best_model_name, _ = self.get_best_model()
        if best_model_name:
            forecast = self.forecast_future(ts_data, best_model_name, days=30, freq=freq)
            self.results = {
                'ts_data': ts_data,
                'train': train,
                'test': test,
                'forecast': forecast,
                'metrics': self.get_metrics_summary(),
                'feature_importance': self.feature_importance,
                'best_model': best_model_name
            }

        logger.info("销售预测流程完成")
        return self.results

    def get_results(self) -> Dict:
        """获取预测结果"""
        return getattr(self, 'results', {})
