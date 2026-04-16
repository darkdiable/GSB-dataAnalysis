import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


class TemperaturePredictor:
    def __init__(self, data):
        self.data = data.copy()
        self.model = None
        self.model_fit = None
        self.forecast_results = None
        self.model_metrics = {}

    def check_stationarity(self, series):
        """
        检验时间序列的平稳性 (ADF检验)
        """
        result = adfuller(series.dropna())
        return {
            'adf_statistic': round(result[0], 4),
            'p_value': round(result[1], 4),
            'is_stationary': result[1] < 0.05,
            'critical_values': {k: round(v, 4) for k, v in result[4].items()}
        }

    def make_stationary(self, series, max_diff=2):
        """
        通过差分使序列平稳
        """
        diff_count = 0
        current_series = series.copy()
        
        while diff_count < max_diff:
            adf_result = self.check_stationarity(current_series)
            if adf_result['is_stationary']:
                break
            current_series = current_series.diff().dropna()
            diff_count += 1
        
        return current_series, diff_count

    def find_best_arima_params(self, series, max_p=3, max_d=2, max_q=3):
        """
        使用AIC准则寻找最佳ARIMA参数
        """
        best_aic = float('inf')
        best_params = None
        best_model = None
        
        results_list = []
        
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(series, order=(p, d, q))
                        fitted = model.fit()
                        
                        results_list.append({
                            'order': (p, d, q),
                            'aic': fitted.aic,
                            'bic': fitted.bic
                        })
                        
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_params = (p, d, q)
                            best_model = fitted
                    except:
                        continue
        
        results_df = pd.DataFrame(results_list)
        results_df = results_df.sort_values('aic').head(10)
        
        return best_params, best_model, results_df

    def train_model(self, test_size=30, auto_select=True, order=None):
        """
        训练ARIMA模型
        """
        # 准备数据
        temp_series = self.data.set_index('date')['temperature']
        
        # 划分训练集和测试集
        train_data = temp_series[:-test_size]
        test_data = temp_series[-test_size:]
        
        # 检查平稳性
        stationarity = self.check_stationarity(train_data)
        print(f"原始序列平稳性检验: p-value = {stationarity['p_value']}")
        
        if auto_select:
            # 自动选择最佳参数
            best_params, best_model, comparison = self.find_best_arima_params(train_data)
            self.model_fit = best_model
            self.order = best_params
            print(f"最佳ARIMA参数: {best_params}")
            print(f"AIC: {best_model.aic:.2f}, BIC: {best_model.bic:.2f}")
        else:
            # 使用指定参数
            self.order = order if order else (2, 1, 2)
            self.model = ARIMA(train_data, order=self.order)
            self.model_fit = self.model.fit()
        
        # 在测试集上评估
        forecast = self.model_fit.forecast(steps=test_size)
        
        # 计算评估指标
        mae = mean_absolute_error(test_data, forecast)
        rmse = np.sqrt(mean_squared_error(test_data, forecast))
        mape = np.mean(np.abs((test_data - forecast) / test_data)) * 100
        
        self.model_metrics = {
            'MAE': round(mae, 4),
            'RMSE': round(rmse, 4),
            'MAPE': round(mape, 4),
            'AIC': round(self.model_fit.aic, 2),
            'BIC': round(self.model_fit.bic, 2)
        }
        
        self.test_data = test_data
        self.test_forecast = forecast
        
        return self.model_fit

    def forecast(self, steps=30):
        """
        预测未来气温
        """
        if self.model_fit is None:
            raise ValueError("请先训练模型")
        
        # 进行预测
        forecast_result = self.model_fit.get_forecast(steps=steps)
        forecast_mean = forecast_result.predicted_mean
        forecast_ci = forecast_result.conf_int()
        
        # 生成未来日期
        last_date = self.data['date'].iloc[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps, freq='D')
        
        self.forecast_results = pd.DataFrame({
            'date': future_dates,
            'forecast': forecast_mean.values,
            'lower_ci': forecast_ci.iloc[:, 0].values,
            'upper_ci': forecast_ci.iloc[:, 1].values
        })
        
        return self.forecast_results

    def get_model_summary(self):
        """
        获取模型摘要
        """
        if self.model_fit is None:
            return "模型尚未训练"
        
        return self.model_fit.summary()

    def get_residual_analysis(self):
        """
        残差分析
        """
        if self.model_fit is None:
            return None
        
        residuals = self.model_fit.resid
        
        # Ljung-Box检验
        lb_test = acorr_ljungbox(residuals, lags=10, return_df=True)
        
        return {
            'residuals': residuals,
            'ljung_box_pvalues': lb_test['lb_pvalue'].values,
            'residual_mean': round(residuals.mean(), 4),
            'residual_std': round(residuals.std(), 4)
        }

    def generate_prediction_summary(self):
        """
        生成预测结果摘要
        """
        summary = []
        summary.append("=" * 60)
        summary.append("气温预测结果摘要")
        summary.append("=" * 60)
        
        # 模型信息
        summary.append("\n【模型信息】")
        summary.append(f"  模型类型: ARIMA{self.order if hasattr(self, 'order') else '(自动选择)'}")
        summary.append(f"  AIC: {self.model_metrics.get('AIC', 'N/A')}")
        summary.append(f"  BIC: {self.model_metrics.get('BIC', 'N/A')}")
        
        # 模型评估
        summary.append("\n【模型评估指标】")
        summary.append(f"  MAE (平均绝对误差): {self.model_metrics.get('MAE', 'N/A')}°C")
        summary.append(f"  RMSE (均方根误差): {self.model_metrics.get('RMSE', 'N/A')}°C")
        summary.append(f"  MAPE (平均绝对百分比误差): {self.model_metrics.get('MAPE', 'N/A')}%")
        
        # 预测结果
        if self.forecast_results is not None:
            summary.append("\n【未来30天预测结果】")
            summary.append(f"  预测平均温度: {self.forecast_results['forecast'].mean():.2f}°C")
            summary.append(f"  预测最高温度: {self.forecast_results['forecast'].max():.2f}°C")
            summary.append(f"  预测最低温度: {self.forecast_results['forecast'].min():.2f}°C")
            summary.append(f"  温度变化趋势: {'上升' if self.forecast_results['forecast'].iloc[-1] > self.forecast_results['forecast'].iloc[0] else '下降'}")
            
            # 前7天详细预测
            summary.append("\n【未来7天详细预测】")
            for i in range(min(7, len(self.forecast_results))):
                row = self.forecast_results.iloc[i]
                summary.append(f"  {row['date'].strftime('%Y-%m-%d')}: {row['forecast']:.2f}°C "
                             f"(置信区间: {row['lower_ci']:.2f} ~ {row['upper_ci']:.2f}°C)")
        
        summary.append("\n" + "=" * 60)
        
        return "\n".join(summary)
