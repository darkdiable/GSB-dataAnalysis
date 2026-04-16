import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


class ARIMAPredictor:
    def __init__(self, data):
        self.data = data
        self.model = None
        self.results = None
        self.predictions = None
        self.forecast = None

    def check_stationarity(self, series):
        result = adfuller(series.dropna())
        is_stationary = result[1] < 0.05

        stationarity_info = {
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'is_stationary': is_stationary
        }

        return stationarity_info

    def find_best_arima_params(self, series, p_range=(0, 3), d_range=(0, 2), q_range=(0, 3)):
        best_aic = float('inf')
        best_params = None

        for p in range(p_range[0], p_range[1] + 1):
            for d in range(d_range[0], d_range[1] + 1):
                for q in range(q_range[0], q_range[1] + 1):
                    try:
                        model = ARIMA(series, order=(p, d, q))
                        results = model.fit()
                        if results.aic < best_aic:
                            best_aic = results.aic
                            best_params = (p, d, q)
                    except:
                        continue

        return best_params, best_aic

    def fit_model(self, order=None, auto_select=True):
        series = self.data.set_index('date')['temperature']

        if auto_select and order is None:
            order, _ = self.find_best_arima_params(series)
        elif order is None:
            order = (1, 1, 1)

        self.model = ARIMA(series, order=order)
        self.results = self.model.fit()

        return self.results

    def predict(self, steps=7):
        if self.results is None:
            raise ValueError("模型尚未训练，请先调用fit_model方法")

        self.forecast = self.results.forecast(steps=steps)

        last_date = self.data['date'].max()
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps, freq='D')

        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'predicted_temperature': self.forecast.values
        })

        return forecast_df

    def evaluate_model(self, test_size=30):
        series = self.data.set_index('date')['temperature']

        train_size = len(series) - test_size
        train, test = series[:train_size], series[train_size:]

        model = ARIMA(train, order=(1, 1, 1))
        results = model.fit()

        predictions = results.forecast(steps=test_size)

        mse = mean_squared_error(test, predictions)
        mae = mean_absolute_error(test, predictions)
        rmse = np.sqrt(mse)

        evaluation = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'test_size': test_size
        }

        return evaluation, predictions, test

    def get_model_summary(self):
        if self.results is None:
            raise ValueError("模型尚未训练")

        return {
            'order': self.results.model_orders,
            'aic': self.results.aic,
            'bic': self.results.bic,
            'params': self.results.params.to_dict()
        }
