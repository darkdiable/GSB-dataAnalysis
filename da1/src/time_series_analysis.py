import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class TimeSeriesAnalyzer:
    def __init__(self, df):
        self.df = df.copy()
        self.yearly_data = None
        self.forecast_result = None
        
    def prepare_yearly_data(self):
        self.yearly_data = self.df.groupby('Year').agg({
            'Sales_Units': 'sum',
            'Production_Units': 'sum',
            'Price_USD': 'mean',
            'Horsepower': 'mean'
        }).reset_index()
        return self
    
    def calculate_cagr(self, start_year=2015, end_year=2025):
        if self.yearly_data is None:
            self.prepare_yearly_data()
            
        cagr_results = {}
        metrics = ['Sales_Units', 'Production_Units']
        
        for metric in metrics:
            start_value = self.yearly_data[self.yearly_data['Year'] == start_year][metric].values
            end_value = self.yearly_data[self.yearly_data['Year'] == end_year][metric].values
            
            if len(start_value) > 0 and len(end_value) > 0:
                n_years = end_year - start_year
                cagr = ((end_value[0] / start_value[0]) ** (1/n_years) - 1) * 100
                cagr_results[metric] = cagr
                
        return cagr_results
    
    def plot_trend_with_ma(self, save_path, window=3):
        if self.yearly_data is None:
            self.prepare_yearly_data()
            
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        ax1 = axes[0, 0]
        years = self.yearly_data['Year'].values
        sales = self.yearly_data['Sales_Units'].values
        ax1.plot(years, sales, 'b-o', linewidth=2, markersize=8, label='实际销量')
        ma_sales = pd.Series(sales).rolling(window=window, min_periods=1).mean()
        ax1.plot(years, ma_sales, 'r--', linewidth=2, label=f'{window}年移动平均')
        ax1.fill_between(years, sales, alpha=0.3)
        ax1.set_xlabel('年份')
        ax1.set_ylabel('销量（辆）')
        ax1.set_title('2015-2025年全球销量趋势')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[0, 1]
        production = self.yearly_data['Production_Units'].values
        ax2.plot(years, production, 'g-s', linewidth=2, markersize=8, label='实际产量')
        ma_prod = pd.Series(production).rolling(window=window, min_periods=1).mean()
        ax2.plot(years, ma_prod, 'r--', linewidth=2, label=f'{window}年移动平均')
        ax2.fill_between(years, production, alpha=0.3, color='green')
        ax2.set_xlabel('年份')
        ax2.set_ylabel('产量（辆）')
        ax2.set_title('2015-2025年全球产量趋势')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3 = axes[1, 0]
        width = 0.35
        x = np.arange(len(years))
        ax3.bar(x - width/2, sales/1000, width, label='销量', color='steelblue')
        ax3.bar(x + width/2, production/1000, width, label='产量', color='coral')
        ax3.set_xlabel('年份')
        ax3.set_ylabel('数量（千辆）')
        ax3.set_title('销量与产量对比')
        ax3.set_xticks(x)
        ax3.set_xticklabels(years)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        ax4 = axes[1, 1]
        cagr = self.calculate_cagr()
        if cagr:
            metrics = list(cagr.keys())
            values = list(cagr.values())
            colors = ['green' if v > 0 else 'red' for v in values]
            bars = ax4.bar(metrics, values, color=colors)
            ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax4.set_ylabel('CAGR (%)')
            ax4.set_title('复合年增长率 (CAGR)')
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.2f}%', ha='center', va='bottom' if val > 0 else 'top')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"趋势分析图已保存: {save_path}")
        
    def arima_forecast(self, forecast_years=2, order=(1, 1, 1)):
        if self.yearly_data is None:
            self.prepare_yearly_data()
            
        sales_series = self.yearly_data.set_index('Year')['Sales_Units']
        production_series = self.yearly_data.set_index('Year')['Production_Units']
        
        forecasts = {}
        forecast_years = int(forecast_years)
        
        for name, series in [('Sales', sales_series), ('Production', production_series)]:
            try:
                model = ARIMA(series, order=order)
                fitted_model = model.fit()
                
                last_year = int(series.index[-1])
                forecast_years_list = list(range(last_year + 1, last_year + forecast_years + 1))
                forecast = fitted_model.forecast(steps=forecast_years)
                
                forecasts[name] = {
                    'years': forecast_years_list,
                    'values': forecast.values,
                    'model': fitted_model
                }
            except Exception as e:
                print(f"ARIMA预测 {name} 失败: {e}")
                
        self.forecast_result = forecasts
        return forecasts
    
    def plot_forecast(self, save_path):
        if self.forecast_result is None:
            self.arima_forecast()
            
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        for idx, (name, data) in enumerate(self.forecast_result.items()):
            ax = axes[idx]
            
            if name == 'Sales':
                historical = self.yearly_data.set_index('Year')['Sales_Units']
                ylabel = '销量（辆）'
                title = '销量ARIMA预测'
            else:
                historical = self.yearly_data.set_index('Year')['Production_Units']
                ylabel = '产量（辆）'
                title = '产量ARIMA预测'
            
            ax.plot(historical.index, historical.values, 'b-o', 
                   linewidth=2, markersize=8, label='历史数据')
            
            forecast_years = data['years']
            forecast_values = data['values']
            ax.plot(forecast_years, forecast_values, 'r--s', 
                   linewidth=2, markersize=8, label='预测值')
            
            ax.axvline(x=historical.index[-1], color='gray', 
                      linestyle=':', linewidth=1, label='预测起点')
            
            ax.set_xlabel('年份')
            ax.set_ylabel(ylabel)
            ax.set_title(f'{title} (2026-2027)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            for year, val in zip(forecast_years, forecast_values):
                ax.annotate(f'{val:,.0f}', xy=(year, val), 
                          xytext=(5, 5), textcoords='offset points')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"预测图已保存: {save_path}")
        
    def create_interactive_plot(self, save_path):
        if self.yearly_data is None:
            self.prepare_yearly_data()
            
        fig = make_subplots(rows=2, cols=2, 
                           subplot_titles=('销量趋势', '产量趋势', 
                                          '平均价格趋势', '平均马力趋势'))
        
        years = self.yearly_data['Year']
        
        fig.add_trace(go.Scatter(x=years, y=self.yearly_data['Sales_Units'],
                                mode='lines+markers', name='销量',
                                line=dict(color='blue', width=2)),
                     row=1, col=1)
        
        fig.add_trace(go.Scatter(x=years, y=self.yearly_data['Production_Units'],
                                mode='lines+markers', name='产量',
                                line=dict(color='green', width=2)),
                     row=1, col=2)
        
        fig.add_trace(go.Scatter(x=years, y=self.yearly_data['Price_USD'],
                                mode='lines+markers', name='平均价格',
                                line=dict(color='orange', width=2)),
                     row=2, col=1)
        
        fig.add_trace(go.Scatter(x=years, y=self.yearly_data['Horsepower'],
                                mode='lines+markers', name='平均马力',
                                line=dict(color='red', width=2)),
                     row=2, col=2)
        
        fig.update_layout(height=800, width=1200, 
                         title_text="全球燃油性能车时间序列分析",
                         showlegend=True)
        fig.update_xaxes(title_text="年份")
        fig.update_yaxes(title_text="数值")
        
        fig.write_html(save_path)
        print(f"交互式图表已保存: {save_path}")
        
    def generate_report(self):
        report = "# 时间序列分析报告\n\n"
        
        cagr = self.calculate_cagr()
        report += "## 1. 复合年增长率 (CAGR)\n\n"
        for metric, value in cagr.items():
            report += f"- **{metric}**: {value:.2f}%\n"
        report += "\n"
        
        if self.forecast_result:
            report += "## 2. ARIMA预测结果 (2026-2027)\n\n"
            for name, data in self.forecast_result.items():
                report += f"### {name}预测\n"
                for year, val in zip(data['years'], data['values']):
                    report += f"- {year}年: {val:,.0f}\n"
                report += "\n"
        
        report += "## 3. 趋势分析\n\n"
        report += "从2015年到2025年，全球燃油性能车市场呈现稳步增长态势。\n"
        
        return report


if __name__ == "__main__":
    df = pd.read_csv("../data/fuel_performance_cars.csv")
    analyzer = TimeSeriesAnalyzer(df)
    analyzer.prepare_yearly_data()
    print(analyzer.calculate_cagr())
    analyzer.plot_trend_with_ma("../output/trend_analysis.png")
    analyzer.arima_forecast()
    analyzer.plot_forecast("../output/forecast.png")
