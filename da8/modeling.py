"""
建模分析模块：时间序列分析、随机森林回归、特征重要性
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class TimeSeriesAnalyzer:
    """时间序列分析器"""
    
    def __init__(self):
        self.decomposition = None
        self.analysis_results = {}
        
    def analyze(self, df, value_col='ridership', time_col='timestamp'):
        """
        执行时间序列分析
        
        Args:
            df: 数据DataFrame
            value_col: 数值列名
            time_col: 时间列名
            
        Returns:
            dict: 分析结果
        """
        # 按时间排序
        df_ts = df.copy()
        df_ts[time_col] = pd.to_datetime(df_ts[time_col])
        df_ts = df_ts.sort_values(time_col)
        
        # 按小时聚合骑行量
        hourly_data = df_ts.set_index(time_col)[value_col].resample('H').sum().fillna(0)
        
        print(f"  时间序列数据点: {len(hourly_data)}")
        
        # 1. 季节性分解
        try:
            self.decomposition = seasonal_decompose(
                hourly_data, 
                model='additive', 
                period=24  # 日周期
            )
            self.analysis_results['decomposition'] = {
                'trend_mean': float(np.nanmean(self.decomposition.trend)),
                'seasonal_std': float(np.nanstd(self.decomposition.seasonal)),
                'residual_std': float(np.nanstd(self.decomposition.resid))
            }
            print("  季节性分解完成")
        except Exception as e:
            print(f"  季节性分解失败: {e}")
            self.analysis_results['decomposition'] = None
        
        # 2. 周期性分析
        # 计算日周期强度
        hourly_pattern = df_ts.groupby(df_ts[time_col].dt.hour)[value_col].mean()
        peak_hours = hourly_pattern.nlargest(3).index.tolist()
        low_hours = hourly_pattern.nsmallest(3).index.tolist()
        
        self.analysis_results['periodicity'] = {
            'peak_hours': peak_hours,
            'low_hours': low_hours,
            'hourly_pattern': hourly_pattern.to_dict()
        }
        print(f"  高峰时段: {peak_hours}, 低谷时段: {low_hours}")
        
        # 3. 趋势分析
        # 简单线性趋势
        x = np.arange(len(hourly_data))
        y = hourly_data.values
        slope = np.polyfit(x, y, 1)[0]
        
        self.analysis_results['trend'] = {
            'slope': float(slope),
            'direction': '上升' if slope > 0 else '下降' if slope < 0 else '平稳'
        }
        print(f"  整体趋势: {self.analysis_results['trend']['direction']} (斜率={slope:.4f})")
        
        # 4. 工作日vs周末分析
        df_ts['is_weekday'] = (df_ts[time_col].dt.dayofweek < 5).astype(int)
        weekday_avg = df_ts[df_ts['is_weekday'] == 1][value_col].mean()
        weekend_avg = df_ts[df_ts['is_weekday'] == 0][value_col].mean()
        
        self.analysis_results['weekday_weekend'] = {
            'weekday_avg': float(weekday_avg),
            'weekend_avg': float(weekend_avg),
            'difference': float(weekday_avg - weekend_avg)
        }
        print(f"  工作日平均骑行量: {weekday_avg:.2f}, 周末: {weekend_avg:.2f}")
        
        return self.analysis_results
    
    def get_decomposition_components(self):
        """获取分解后的组件"""
        if self.decomposition is None:
            return None
        return {
            'trend': self.decomposition.trend,
            'seasonal': self.decomposition.seasonal,
            'resid': self.decomposition.resid,
            'observed': self.decomposition.observed
        }


class DemandPredictor:
    """骑行需求预测器（随机森林）"""
    
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1
        )
        self.feature_importance = None
        self.metrics = {}
        self.is_trained = False
        
    def train(self, X, y, test_size=0.2):
        """
        训练预测模型
        
        Args:
            X: 特征矩阵
            y: 目标变量
            test_size: 测试集比例
            
        Returns:
            dict: 训练评估指标
        """
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        print(f"  训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
        
        # 训练模型
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # 预测
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # 计算指标
        self.metrics = {
            'train': {
                'mse': mean_squared_error(y_train, y_pred_train),
                'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'mae': mean_absolute_error(y_train, y_pred_train),
                'r2': r2_score(y_train, y_pred_train)
            },
            'test': {
                'mse': mean_squared_error(y_test, y_pred_test),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'mae': mean_absolute_error(y_test, y_pred_test),
                'r2': r2_score(y_test, y_pred_test)
            }
        }
        
        print(f"  训练集 R²: {self.metrics['train']['r2']:.4f}")
        print(f"  测试集 R²: {self.metrics['test']['r2']:.4f}")
        print(f"  测试集 RMSE: {self.metrics['test']['rmse']:.4f}")
        
        # 计算特征重要性
        self._calculate_feature_importance(X.columns.tolist())
        
        return self.metrics
    
    def _calculate_feature_importance(self, feature_names):
        """计算并排序特征重要性"""
        importance = self.model.feature_importances_
        
        self.feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print("\n  特征重要性排名:")
        for idx, row in self.feature_importance.head(10).iterrows():
            print(f"    {row['feature']}: {row['importance']:.4f}")
    
    def predict(self, X):
        """预测骑行需求"""
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train()")
        return self.model.predict(X)
    
    def get_feature_importance(self):
        """获取特征重要性"""
        return self.feature_importance
    
    def get_metrics(self):
        """获取评估指标"""
        return self.metrics


class ModelingPipeline:
    """建模流程整合"""
    
    def __init__(self):
        self.ts_analyzer = TimeSeriesAnalyzer()
        self.predictor = DemandPredictor()
        self.results = {}
        
    def run_full_analysis(self, df, X, y, feature_cols):
        """
        执行完整分析流程
        
        Args:
            df: 原始数据
            X: 特征矩阵
            y: 目标变量
            feature_cols: 特征列名
        """
        print("\n" + "="*50)
        print("时间序列分析")
        print("="*50)
        ts_results = self.ts_analyzer.analyze(df)
        self.results['time_series'] = ts_results
        
        print("\n" + "="*50)
        print("随机森林回归建模")
        print("="*50)
        metrics = self.predictor.train(X, y)
        self.results['modeling'] = {
            'metrics': metrics,
            'feature_importance': self.predictor.get_feature_importance().to_dict('records')
        }
        
        return self.results
    
    def get_results(self):
        """获取所有分析结果"""
        return self.results


if __name__ == '__main__':
    # 测试建模
    from data_generator import BikeSharingDataGenerator
    from data_preprocessor import DataPreprocessor
    
    # 生成并预处理数据
    generator = BikeSharingDataGenerator(n_samples=2000)
    raw_data = generator.generate()
    
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.preprocess(raw_data)
    
    X, y, feature_cols = preprocessor.get_training_data(processed_data)
    
    # 运行分析
    pipeline = ModelingPipeline()
    results = pipeline.run_full_analysis(processed_data, X, y, feature_cols)
    
    print("\n分析完成!")
