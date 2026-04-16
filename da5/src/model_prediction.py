import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')


def test_stationarity(timeseries):
    print("正在检验时间序列平稳性...")
    result = adfuller(timeseries, autolag='AIC')
    is_stationary = result[1] <= 0.05
    
    return {
        'adf_statistic': round(result[0], 4),
        'p_value': round(result[1], 4),
        'is_stationary': is_stationary
    }


def find_best_arima_order(timeseries, max_p=5, max_d=2, max_q=5):
    print("正在搜索最优ARIMA参数...")
    best_aic = np.inf
    best_order = None
    
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                try:
                    model = ARIMA(timeseries, order=(p, d, q))
                    results = model.fit()
                    if results.aic < best_aic:
                        best_aic = results.aic
                        best_order = (p, d, q)
                except:
                    continue
    
    print(f"最优ARIMA阶数: {best_order}, AIC: {round(best_aic, 2)}")
    return best_order


def train_arima_model(data, forecast_days=30):
    print(f"开始训练ARIMA模型，预测未来 {forecast_days} 天气温...")
    
    temperature_data = data['temperature'].values
    
    stationarity = test_stationarity(temperature_data)
    
    best_order = find_best_arima_order(temperature_data)
    
    model = ARIMA(temperature_data, order=best_order)
    model_fit = model.fit()
    
    forecast = model_fit.get_forecast(steps=forecast_days)
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int(alpha=0.05)
    
    last_date = data['date'].iloc[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
    
    forecast_result = pd.DataFrame({
        'date': forecast_dates,
        'predicted_temperature': forecast_mean.round(2),
        'lower_bound': forecast_ci[:, 0].round(2),
        'upper_bound': forecast_ci[:, 1].round(2)
    })
    
    in_sample_pred = model_fit.predict(start=1, end=len(temperature_data)-1)
    
    mae = np.mean(np.abs(temperature_data[1:] - in_sample_pred))
    rmse = np.sqrt(np.mean((temperature_data[1:] - in_sample_pred) ** 2))
    
    print(f"模型训练完成 - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    
    return {
        'model': model_fit,
        'best_order': best_order,
        'stationarity': stationarity,
        'forecast': forecast_result,
        'in_sample_predictions': in_sample_pred,
        'mae': round(mae, 2),
        'rmse': round(rmse, 2)
    }


def generate_prediction_summary(prediction_result):
    summary = []
    summary.append("=" * 50)
    summary.append("气温预测结果摘要")
    summary.append("=" * 50)
    summary.append("")
    
    summary.append("一、模型信息:")
    summary.append("-" * 30)
    summary.append(f"ARIMA阶数: {prediction_result['best_order']}")
    summary.append(f"序列平稳性检验p值: {prediction_result['stationarity']['p_value']}")
    summary.append(f"序列平稳: {'是' if prediction_result['stationarity']['is_stationary'] else '否'}")
    summary.append("")
    
    summary.append("二、模型评估:")
    summary.append("-" * 30)
    summary.append(f"平均绝对误差(MAE): {prediction_result['mae']} °C")
    summary.append(f"均方根误差(RMSE): {prediction_result['rmse']} °C")
    summary.append("")
    
    summary.append("三、未来7天气温预测:")
    summary.append("-" * 30)
    forecast_7days = prediction_result['forecast'].head(7)
    for _, row in forecast_7days.iterrows():
        summary.append(f"  {row['date'].strftime('%Y-%m-%d')}: {row['predicted_temperature']:>6.2f} °C "
                       f"(95%置信区间: [{row['lower_bound']:.2f}, {row['upper_bound']:.2f}])")
    summary.append("")
    
    return "\n".join(summary)
