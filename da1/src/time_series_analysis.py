"""
时间序列分析模块
功能：产销趋势分析、移动平均线、CAGR计算、ARIMA预测
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class TimeSeriesAnalyzer:
    """时间序列分析类"""
    
    def __init__(self, df, output_dir='../output'):
        """
        初始化
        Args:
            df: pandas DataFrame
            output_dir: 输出目录
        """
        self.df = df.copy()
        self.output_dir = output_dir
        self.yearly_stats = None
        
    def calculate_yearly_stats(self):
        """计算年度统计数据"""
        self.yearly_stats = self.df.groupby('Year').agg({
            'Units_Sold': ['sum', 'mean', 'count'],
            'Price_USD': 'mean',
            'Market_Share_Percent': 'sum'
        }).reset_index()
        
        self.yearly_stats.columns = ['Year', 'Total_Sales', 'Avg_Sales', 'Model_Count', 
                                      'Avg_Price', 'Total_Market_Share']
        return self.yearly_stats
    
    def calculate_cagr(self, start_year=None, end_year=None, column='Total_Sales'):
        """
        计算复合年增长率(CAGR)
        Args:
            start_year: 起始年份
            end_year: 结束年份
            column: 计算CAGR的列
        Returns:
            CAGR值
        """
        if self.yearly_stats is None:
            self.calculate_yearly_stats()
        
        stats = self.yearly_stats
        
        if start_year is None:
            start_year = stats['Year'].min()
        if end_year is None:
            end_year = stats['Year'].max()
        
        start_value = stats[stats['Year'] == start_year][column].values[0]
        end_value = stats[stats['Year'] == end_year][column].values[0]
        n_years = end_year - start_year
        
        cagr = (end_value / start_value) ** (1 / n_years) - 1
        
        print(f"CAGR计算 ({start_year}-{end_year}):")
        print(f"  起始值: {start_value:,.0f}")
        print(f"  结束值: {end_value:,.0f}")
        print(f"  年数: {n_years}")
        print(f"  CAGR: {cagr*100:.2f}%")
        
        return cagr
    
    def plot_sales_trend(self, save_path=None, dpi=300):
        """
        绘制产销趋势图（含移动平均线）
        Args:
            save_path: 保存路径
            dpi: 图像分辨率
        """
        if self.yearly_stats is None:
            self.calculate_yearly_stats()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        stats = self.yearly_stats
        
        ax1 = axes[0, 0]
        ax1.plot(stats['Year'], stats['Total_Sales'], marker='o', linewidth=2, 
                markersize=8, label='年度销量', color='#2E86AB')
        
        if len(stats) >= 3:
            stats['MA_3'] = stats['Total_Sales'].rolling(window=3, center=True).mean()
            ax1.plot(stats['Year'], stats['MA_3'], '--', linewidth=2, 
                    label='3年移动平均线', color='#A23B72')
        
        ax1.set_xlabel('年份', fontsize=12)
        ax1.set_ylabel('销量（辆）', fontsize=12)
        ax1.set_title('2015-2025年全球燃油性能车销量趋势', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(stats['Year'])
        
        ax2 = axes[0, 1]
        ax2.bar(stats['Year'], stats['Model_Count'], color='#F18F01', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('年份', fontsize=12)
        ax2.set_ylabel('车型数量', fontsize=12)
        ax2.set_title('年度车型发布数量', fontsize=14, fontweight='bold')
        ax2.set_xticks(stats['Year'])
        ax2.grid(True, alpha=0.3, axis='y')
        
        ax3 = axes[1, 0]
        ax3.plot(stats['Year'], stats['Avg_Price'], marker='s', linewidth=2, 
                markersize=8, color='#C73E1D', label='平均价格')
        ax3.fill_between(stats['Year'], stats['Avg_Price'], alpha=0.3, color='#C73E1D')
        ax3.set_xlabel('年份', fontsize=12)
        ax3.set_ylabel('平均价格（USD）', fontsize=12)
        ax3.set_title('年度平均价格趋势', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xticks(stats['Year'])
        
        ax4 = axes[1, 1]
        growth_rate = stats['Total_Sales'].pct_change() * 100
        colors = ['green' if x > 0 else 'red' for x in growth_rate.dropna()]
        ax4.bar(stats['Year'][1:], growth_rate.dropna(), color=colors, alpha=0.7, edgecolor='black')
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax4.set_xlabel('年份', fontsize=12)
        ax4.set_ylabel('增长率（%）', fontsize=12)
        ax4.set_title('年度销量增长率', fontsize=14, fontweight='bold')
        ax4.set_xticks(stats['Year'][1:])
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
            print(f"趋势图已保存至: {save_path}")
        
        return fig
    
    def arima_forecast(self, forecast_years=2, save_path=None, dpi=300):
        """
        使用ARIMA模型进行预测
        Args:
            forecast_years: 预测年数
            save_path: 保存路径
            dpi: 图像分辨率
        Returns:
            预测结果DataFrame
        """
        if self.yearly_stats is None:
            self.calculate_yearly_stats()
        
        ts_data = self.yearly_stats.set_index('Year')['Total_Sales']
        
        result = adfuller(ts_data)
        print(f"\nADF检验结果:")
        print(f"  ADF统计量: {result[0]:.4f}")
        print(f"  p-value: {result[1]:.4f}")
        print(f"  是否平稳: {'是' if result[1] < 0.05 else '否'}")
        
        try:
            model = ARIMA(ts_data, order=(2, 1, 2))
            fitted_model = model.fit()
            
            print(f"\nARIMA模型拟合完成")
            print(f"  AIC: {fitted_model.aic:.2f}")
            print(f"  BIC: {fitted_model.bic:.2f}")
            
            forecast = fitted_model.get_forecast(steps=forecast_years)
            forecast_mean = forecast.predicted_mean
            forecast_ci = forecast.conf_int(alpha=0.05)
            
            forecast_years_list = list(range(int(ts_data.index.max()) + 1, 
                                            int(ts_data.index.max()) + 1 + forecast_years))
            
            fig, ax = plt.subplots(figsize=(14, 8))
            
            ax.plot(ts_data.index, ts_data.values, 'o-', linewidth=2, 
                   markersize=8, label='历史销量', color='#2E86AB')
            
            ax.plot(forecast_years_list, forecast_mean.values, 's--', linewidth=2, 
                   markersize=10, label='ARIMA预测', color='#F18F01')
            
            ax.fill_between(forecast_years_list, 
                           forecast_ci.iloc[:, 0].values, 
                           forecast_ci.iloc[:, 1].values,
                           alpha=0.3, color='#F18F01', label='95%置信区间')
            
            ax.set_xlabel('年份', fontsize=12)
            ax.set_ylabel('销量（辆）', fontsize=12)
            ax.set_title('ARIMA销量预测 (2026-2027)', fontsize=14, fontweight='bold')
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            
            all_years = list(ts_data.index) + forecast_years_list
            ax.set_xticks(all_years)
            ax.set_xticklabels(all_years, rotation=45)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
                print(f"预测图已保存至: {save_path}")
            
            forecast_df = pd.DataFrame({
                'Year': forecast_years_list,
                'Forecast_Sales': forecast_mean.values.astype(int),
                'Lower_CI': forecast_ci.iloc[:, 0].values.astype(int),
                'Upper_CI': forecast_ci.iloc[:, 1].values.astype(int)
            })
            
            print(f"\n预测结果:")
            print(forecast_df.to_string(index=False))
            
            return forecast_df
            
        except Exception as e:
            print(f"ARIMA模型拟合失败: {e}")
            
            last_value = ts_data.iloc[-1]
            growth_rate = self.calculate_cagr()
            
            forecast_years_list = list(range(int(ts_data.index.max()) + 1, 
                                            int(ts_data.index.max()) + 1 + forecast_years))
            forecast_values = [int(last_value * (1 + growth_rate) ** (i+1)) 
                              for i in range(forecast_years)]
            
            forecast_df = pd.DataFrame({
                'Year': forecast_years_list,
                'Forecast_Sales': forecast_values,
                'Lower_CI': [int(v * 0.9) for v in forecast_values],
                'Upper_CI': [int(v * 1.1) for v in forecast_values]
            })
            
            fig, ax = plt.subplots(figsize=(14, 8))
            ax.plot(ts_data.index, ts_data.values, 'o-', linewidth=2, 
                   markersize=8, label='历史销量', color='#2E86AB')
            ax.plot(forecast_years_list, forecast_values, 's--', linewidth=2, 
                   markersize=10, label='CAGR趋势预测', color='#F18F01')
            ax.set_xlabel('年份', fontsize=12)
            ax.set_ylabel('销量（辆）', fontsize=12)
            ax.set_title('销量趋势预测 (2026-2027)', fontsize=14, fontweight='bold')
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            
            all_years = list(ts_data.index) + forecast_years_list
            ax.set_xticks(all_years)
            ax.set_xticklabels(all_years, rotation=45)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
                print(f"预测图已保存至: {save_path}")
            
            return forecast_df
    
    def brand_trend_analysis(self, top_n=5, save_path=None, dpi=300):
        """
        品牌趋势分析
        Args:
            top_n: 显示前N个品牌
            save_path: 保存路径
            dpi: 图像分辨率
        """
        brand_yearly = self.df.groupby(['Year', 'Brand'])['Units_Sold'].sum().reset_index()
        
        top_brands = self.df.groupby('Brand')['Units_Sold'].sum().nlargest(top_n).index.tolist()
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, top_n))
        
        for i, brand in enumerate(top_brands):
            brand_data = brand_yearly[brand_yearly['Brand'] == brand]
            ax.plot(brand_data['Year'], brand_data['Units_Sold'], 
                   marker='o', linewidth=2.5, markersize=8, 
                   label=brand, color=colors[i])
        
        ax.set_xlabel('年份', fontsize=12)
        ax.set_ylabel('销量（辆）', fontsize=12)
        ax.set_title(f'主要品牌销量趋势 (Top {top_n})', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
        ax.grid(True, alpha=0.3)
        ax.set_xticks(sorted(self.df['Year'].unique()))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
            print(f"品牌趋势图已保存至: {save_path}")
        
        return fig


def run_time_series_analysis(df, output_dir='../output'):
    """
    运行完整的时间序列分析
    Args:
        df: DataFrame
        output_dir: 输出目录
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    analyzer = TimeSeriesAnalyzer(df, output_dir)
    
    print("=" * 60)
    print("时间序列分析")
    print("=" * 60)
    
    stats = analyzer.calculate_yearly_stats()
    print("\n年度统计:")
    print(stats.to_string(index=False))
    
    analyzer.calculate_cagr()
    
    analyzer.plot_sales_trend(save_path=f'{output_dir}/01_sales_trend.png')
    
    analyzer.arima_forecast(forecast_years=2, save_path=f'{output_dir}/02_arima_forecast.png')
    
    analyzer.brand_trend_analysis(top_n=6, save_path=f'{output_dir}/03_brand_trend.png')
    
    print("\n时间序列分析完成！")
    
    return analyzer


if __name__ == '__main__':
    df = pd.read_csv('../data/fuel_performance_cars.csv')
    run_time_series_analysis(df)
