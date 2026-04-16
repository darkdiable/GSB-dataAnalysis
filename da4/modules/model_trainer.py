"""
模型训练模块 - 基于scikit-learn的回归模型训练和预测
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime, timedelta


class ModelTrainer:
    """模型训练器类，支持多种回归模型"""
    
    def __init__(self, model_type='random_forest'):
        """
        初始化模型训练器
        
        Parameters:
            model_type: 模型类型 ('random_forest', 'gradient_boosting', 'linear')
        """
        self.model_type = model_type
        self.model = None
        self.feature_names = []
        self.metrics = {}
        self.feature_importance = None
        
    def _create_model(self):
        """创建模型实例"""
        if self.model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        elif self.model_type == 'linear':
            return LinearRegression()
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
    
    def train(self, df_features, target_col='sales', test_size=0.2):
        """
        训练模型
        
        Parameters:
            df_features: 包含特征的数据框
            target_col: 目标列名
            test_size: 测试集比例
            
        Returns:
            dict: 训练评估指标
        """
        # 准备特征和目标
        feature_cols = [col for col in df_features.columns 
                       if col not in ['date', target_col]]
        self.feature_names = feature_cols
        
        X = df_features[feature_cols]
        y = df_features[target_col]
        
        # 划分训练集和测试集（按时间顺序）
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # 创建并训练模型
        self.model = self._create_model()
        self.model.fit(X_train, y_train)
        
        # 预测
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # 计算评估指标
        self.metrics = {
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'train_r2': r2_score(y_train, y_train_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'test_r2': r2_score(y_test, y_test_pred)
        }
        
        # 计算特征重要性
        self._calculate_feature_importance()
        
        print(f"模型训练完成 ({self.model_type})")
        print(f"  训练集 R²: {self.metrics['train_r2']:.4f}")
        print(f"  测试集 R²: {self.metrics['test_r2']:.4f}")
        print(f"  测试集 MAE: {self.metrics['test_mae']:.2f}")
        
        return self.metrics
    
    def _calculate_feature_importance(self):
        """计算特征重要性"""
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_)
        else:
            importance = np.zeros(len(self.feature_names))
        
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def predict(self, X):
        """
        使用训练好的模型进行预测
        
        Parameters:
            X: 特征数据框或数组
            
        Returns:
            array: 预测结果
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用train方法")
        
        # 确保特征顺序一致
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_names]
        
        return self.model.predict(X)
    
    def predict_future(self, df_future_features, historical_df=None):
        """
        预测未来销售额
        
        Parameters:
            df_future_features: 未来日期的特征数据框
            historical_df: 历史数据，用于填充滞后特征
            
        Returns:
            DataFrame: 包含预测结果的数据框
        """
        df_pred = df_future_features.copy()
        
        # 迭代预测，使用预测值填充滞后特征
        predictions = []
        
        for i in range(len(df_pred)):
            # 获取当前行的特征
            row = df_pred.iloc[i:i+1].copy()
            
            # 填充滞后特征（使用历史数据或已预测值）
            if historical_df is not None:
                combined = pd.concat([historical_df, df_pred.iloc[:i]], ignore_index=True)
                
                # 填充滞后特征
                for lag in [1, 7, 14, 30]:
                    if f'sales_lag_{lag}' in self.feature_names:
                        if len(combined) >= lag:
                            row[f'sales_lag_{lag}'] = combined['sales'].iloc[-lag]
                        else:
                            row[f'sales_lag_{lag}'] = combined['sales'].mean()
                
                # 填充滚动特征
                for window in [7, 14, 30]:
                    if f'sales_roll_mean_{window}' in self.feature_names:
                        if len(combined) >= window:
                            row[f'sales_roll_mean_{window}'] = combined['sales'].tail(window).mean()
                            row[f'sales_roll_std_{window}'] = combined['sales'].tail(window).std()
                            row[f'sales_roll_max_{window}'] = combined['sales'].tail(window).max()
                            row[f'sales_roll_min_{window}'] = combined['sales'].tail(window).min()
                        else:
                            row[f'sales_roll_mean_{window}'] = combined['sales'].mean()
                            row[f'sales_roll_std_{window}'] = combined['sales'].std()
                            row[f'sales_roll_max_{window}'] = combined['sales'].max()
                            row[f'sales_roll_min_{window}'] = combined['sales'].min()
            
            # 确保所有特征都存在
            for col in self.feature_names:
                if col not in row.columns:
                    row[col] = 0
            
            # 预测
            X = row[self.feature_names]
            pred = self.model.predict(X)[0]
            predictions.append(pred)
            
            # 将预测值添加到数据框中，用于后续滞后特征计算
            df_pred.loc[df_pred.index[i], 'sales'] = pred
        
        df_pred['predicted_sales'] = predictions
        return df_pred[['date', 'predicted_sales']]
    
    def get_feature_importance(self, top_n=10):
        """
        获取特征重要性
        
        Parameters:
            top_n: 返回前N个重要特征
            
        Returns:
            DataFrame: 特征重要性数据框
        """
        if self.feature_importance is None:
            return None
        return self.feature_importance.head(top_n)
    
    def get_metrics(self):
        """获取模型评估指标"""
        return self.metrics
