import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import plotly.graph_objects as go
from .config import OUTPUT_DIR, DPI, FIGSIZE

def calculate_cagr(values, years):
    start_val, end_val = values.iloc[0], values.iloc[-1]
    n_years = years.iloc[-1] - years.iloc[0]
    return (end_val / start_val) ** (1 / n_years) - 1

def plot_production_sales_trend(df, interactive=False):
    yearly = df.groupby('year')[['production', 'sales']].sum().reset_index()
    yearly = yearly[(yearly['year'] >= 2015) & (yearly['year'] <= 2025)]
    
    yearly['prod_ma'] = yearly['production'].rolling(window=3, center=True).mean()
    yearly['sales_ma'] = yearly['sales'].rolling(window=3, center=True).mean()
    
    cagr_prod = calculate_cagr(yearly['production'], yearly['year'])
    cagr_sales = calculate_cagr(yearly['sales'], yearly['year'])
    
    if interactive:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=yearly['year'], y=yearly['production'],
                                mode='lines+markers', name='产量', line=dict(color='#1f77b4', width=2)))
        fig.add_trace(go.Scatter(x=yearly['year'], y=yearly['prod_ma'],
                                mode='lines', name='产量MA(3)', line=dict(color='#1f77b4', dash='dash')))
        fig.add_trace(go.Scatter(x=yearly['year'], y=yearly['sales'],
                                mode='lines+markers', name='销量', line=dict(color='#ff7f0e', width=2)))
        fig.add_trace(go.Scatter(x=yearly['year'], y=yearly['sales_ma'],
                                mode='lines', name='销量MA(3)', line=dict(color='#ff7f0e', dash='dash')))
        fig.update_layout(title='2015-2025年全球燃油性能车产销趋势',
                         xaxis_title='年份', yaxis_title='数量',
                         hovermode='x unified')
        fig.write_html(f'{OUTPUT_DIR}/timeseries_trend_interactive.html')
    else:
        plt.figure(figsize=FIGSIZE)
        plt.plot(yearly['year'], yearly['production'], 'o-', label='产量', color='#1f77b4', linewidth=2)
        plt.plot(yearly['year'], yearly['prod_ma'], '--', label='产量MA(3)', color='#1f77b4', alpha=0.7)
        plt.plot(yearly['year'], yearly['sales'], 's-', label='销量', color='#ff7f0e', linewidth=2)
        plt.plot(yearly['year'], yearly['sales_ma'], '--', label='销量MA(3)', color='#ff7f0e', alpha=0.7)
        plt.title(f'2015-2025年全球燃油性能车产销趋势\n产量CAGR: {cagr_prod:.2%}, 销量CAGR: {cagr_sales:.2%}', fontsize=14, pad=20)
        plt.xlabel('年份', fontsize=12)
        plt.ylabel('数量', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/timeseries_trend.png', dpi=DPI)
        plt.close()
    
    return yearly, cagr_prod, cagr_sales

def arima_forecast(df, steps=2, interactive=False):
    yearly = df.groupby('year')['sales'].sum().reset_index()
    yearly = yearly[(yearly['year'] >= 2015) & (yearly['year'] <= 2025)]
    
    ts = yearly.set_index('year')['sales']
    model = ARIMA(ts, order=(1, 1, 1))
    results = model.fit()
    
    forecast = results.get_forecast(steps=steps)
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int()
    
    future_years = [2026, 2027]
    forecast_df = pd.DataFrame({
        'year': future_years,
        'sales': forecast_mean.values,
        'lower': forecast_ci.iloc[:, 0].values,
        'upper': forecast_ci.iloc[:, 1].values
    })
    
    if interactive:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=yearly['year'], y=yearly['sales'],
                                mode='lines+markers', name='历史销量'))
        fig.add_trace(go.Scatter(x=forecast_df['year'], y=forecast_df['sales'],
                                mode='lines+markers', name='预测销量', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=forecast_df['year'].tolist() + forecast_df['year'].tolist()[::-1],
                                y=forecast_df['upper'].tolist() + forecast_df['lower'].tolist()[::-1],
                                fill='toself', fillcolor='rgba(255,0,0,0.2)',
                                line=dict(color='rgba(255,255,255,0)'),
                                name='95%置信区间'))
        fig.update_layout(title='ARIMA销量预测 2026-2027',
                         xaxis_title='年份', yaxis_title='销量')
        fig.write_html(f'{OUTPUT_DIR}/arima_forecast_interactive.html')
    else:
        plt.figure(figsize=FIGSIZE)
        plt.plot(yearly['year'], yearly['sales'], 'o-', label='历史销量', color='#1f77b4', linewidth=2)
        plt.plot(forecast_df['year'], forecast_df['sales'], 'ro-', label='预测销量', linewidth=2)
        plt.fill_between(forecast_df['year'], forecast_df['lower'], forecast_df['upper'],
                        color='red', alpha=0.2, label='95%置信区间')
        plt.title('ARIMA模型销量预测 (2026-2027)', fontsize=14, pad=20)
        plt.xlabel('年份', fontsize=12)
        plt.ylabel('销量', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)
        plt.xticks(list(range(2015, 2028)))
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/arima_forecast.png', dpi=DPI)
        plt.close()
    
    return forecast_df

def run_time_series_analysis(df):
    print("=== 开始时间序列分析 ===")
    
    yearly, cagr_prod, cagr_sales = plot_production_sales_trend(df)
    print(f"产量年均复合增长率 (CAGR): {cagr_prod:.2%}")
    print(f"销量年均复合增长率 (CAGR): {cagr_sales:.2%}")
    
    plot_production_sales_trend(df, interactive=True)
    
    forecast = arima_forecast(df)
    arima_forecast(df, interactive=True)
    print("2026-2027年销量预测:")
    print(forecast[['year', 'sales']])
    
    print("=== 时间序列分析完成 ===")
    return {
        'yearly_data': yearly,
        'cagr_production': cagr_prod,
        'cagr_sales': cagr_sales,
        'forecast': forecast
    }
