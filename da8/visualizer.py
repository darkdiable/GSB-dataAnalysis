"""
可视化模块：生成各类分析图表
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300


class BikeVisualizer:
    """共享单车数据可视化器"""
    
    def __init__(self, df: pd.DataFrame, output_dir: str = 'figures'):
        self.df = df
        self.output_dir = output_dir
        self.figures = []
    
    def plot_hourly_distribution(self, save: bool = True) -> plt.Figure:
        """绘制骑行量小时分布图"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        hourly_data = self.df.groupby('hour')['ridership'].agg(['mean', 'std'])
        
        ax1 = axes[0]
        ax1.bar(hourly_data.index, hourly_data['mean'], 
               color='steelblue', alpha=0.7, edgecolor='navy')
        ax1.errorbar(hourly_data.index, hourly_data['mean'], 
                    yerr=hourly_data['std'], fmt='none', 
                    color='darkred', capsize=3, alpha=0.5)
        ax1.set_xlabel('小时', fontsize=12)
        ax1.set_ylabel('平均骑行人数', fontsize=12)
        ax1.set_title('骑行量小时分布', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(24))
        ax1.grid(axis='y', alpha=0.3)
        
        ax2 = axes[1]
        workday_data = self.df[self.df['is_weekend'] == 0].groupby('hour')['ridership'].mean()
        weekend_data = self.df[self.df['is_weekend'] == 1].groupby('hour')['ridership'].mean()
        
        ax2.plot(workday_data.index, workday_data.values, 
                'b-o', label='工作日', linewidth=2, markersize=6)
        ax2.plot(weekend_data.index, weekend_data.values, 
                'r-s', label='周末', linewidth=2, markersize=6)
        ax2.set_xlabel('小时', fontsize=12)
        ax2.set_ylabel('平均骑行人数', fontsize=12)
        ax2.set_title('工作日vs周末骑行模式对比', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(24))
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = f'{self.output_dir}/hourly_distribution.png'
            plt.savefig(filepath, bbox_inches='tight', facecolor='white')
            self.figures.append(filepath)
            print(f"图表已保存: {filepath}")
        
        return fig
    
    def plot_temperature_ridership(self, save: bool = True) -> plt.Figure:
        """绘制温度与骑行量关系散点图"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1 = axes[0]
        sample_df = self.df.sample(min(5000, len(self.df)), random_state=42)
        scatter = ax1.scatter(sample_df['temperature'], sample_df['ridership'],
                            c=sample_df['hour'], cmap='coolwarm', 
                            alpha=0.5, s=20)
        ax1.set_xlabel('温度 (°C)', fontsize=12)
        ax1.set_ylabel('骑行人数', fontsize=12)
        ax1.set_title('温度与骑行量关系', fontsize=14, fontweight='bold')
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('小时')
        ax1.grid(alpha=0.3)
        
        temp_bins = pd.cut(self.df['temperature'], 
                          bins=[-10, 5, 15, 25, 35, 45],
                          labels=['极冷(<5°C)', '冷(5-15°C)', '适宜(15-25°C)', 
                                '热(25-35°C)', '极热(>35°C)'])
        temp_ridership = self.df.groupby(temp_bins)['ridership'].mean()
        
        ax2 = axes[1]
        colors = ['#3498db', '#2ecc71', '#f1c40f', '#e74c3c', '#9b59b6']
        bars = ax2.bar(range(len(temp_ridership)), temp_ridership.values, 
                      color=colors, edgecolor='black', alpha=0.8)
        ax2.set_xlabel('温度区间', fontsize=12)
        ax2.set_ylabel('平均骑行人数', fontsize=12)
        ax2.set_title('不同温度区间骑行量对比', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(temp_ridership)))
        ax2.set_xticklabels(temp_ridership.index, rotation=15)
        ax2.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars, temp_ridership.values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.0f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save:
            filepath = f'{self.output_dir}/temperature_ridership.png'
            plt.savefig(filepath, bbox_inches='tight', facecolor='white')
            self.figures.append(filepath)
            print(f"图表已保存: {filepath}")
        
        return fig
    
    def plot_weather_impact(self, save: bool = True) -> plt.Figure:
        """绘制天气状况对骑行量影响箱线图"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1 = axes[0]
        weather_order = ['晴天', '多云', '小雨', '大雨', '雪']
        weather_colors = ['#f39c12', '#95a5a6', '#3498db', '#2c3e50', '#ecf0f1']
        
        box_data = [self.df[self.df['weather'] == w]['ridership'].values 
                   for w in weather_order]
        
        bp = ax1.boxplot(box_data, labels=weather_order, patch_artist=True)
        for patch, color in zip(bp['boxes'], weather_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_xlabel('天气状况', fontsize=12)
        ax1.set_ylabel('骑行人数', fontsize=12)
        ax1.set_title('天气状况对骑行量的影响', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        ax2 = axes[1]
        weather_stats = self.df.groupby('weather')['ridership'].agg(['mean', 'std', 'count'])
        weather_stats = weather_stats.reindex(weather_order)
        
        x_pos = range(len(weather_stats))
        bars = ax2.bar(x_pos, weather_stats['mean'], 
                      yerr=weather_stats['std'],
                      color=weather_colors, edgecolor='black',
                      alpha=0.8, capsize=5)
        
        ax2.set_xlabel('天气状况', fontsize=12)
        ax2.set_ylabel('平均骑行人数', fontsize=12)
        ax2.set_title('各天气状况平均骑行量', fontsize=14, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(weather_order)
        ax2.grid(axis='y', alpha=0.3)
        
        for bar, mean_val in zip(bars, weather_stats['mean']):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'{mean_val:.0f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save:
            filepath = f'{self.output_dir}/weather_impact.png'
            plt.savefig(filepath, bbox_inches='tight', facecolor='white')
            self.figures.append(filepath)
            print(f"图表已保存: {filepath}")
        
        return fig
    
    def plot_feature_importance(self, feature_importance: pd.DataFrame, 
                               save: bool = True) -> plt.Figure:
        """绘制特征重要性图"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        top_features = feature_importance.head(10)
        
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(top_features)))[::-1]
        
        bars = ax.barh(range(len(top_features)), top_features['importance'], 
                      color=colors, edgecolor='navy', alpha=0.8)
        
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('重要性得分', fontsize=12)
        ax.set_title('特征重要性排序 (Top 10)', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()
        
        for bar, val in zip(bars, top_features['importance']):
            ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                   f'{val:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save:
            filepath = f'{self.output_dir}/feature_importance.png'
            plt.savefig(filepath, bbox_inches='tight', facecolor='white')
            self.figures.append(filepath)
            print(f"图表已保存: {filepath}")
        
        return fig
    
    def plot_time_series_decomposition(self, decomposition, save: bool = True) -> plt.Figure:
        """绘制时间序列分解图"""
        fig, axes = plt.subplots(4, 1, figsize=(14, 10))
        
        axes[0].plot(decomposition.observed, color='steelblue', linewidth=0.8)
        axes[0].set_ylabel('原始数据', fontsize=11)
        axes[0].set_title('时间序列分解', fontsize=14, fontweight='bold')
        axes[0].grid(alpha=0.3)
        
        axes[1].plot(decomposition.trend, color='darkorange', linewidth=1.5)
        axes[1].set_ylabel('趋势', fontsize=11)
        axes[1].grid(alpha=0.3)
        
        axes[2].plot(decomposition.seasonal, color='green', linewidth=0.8)
        axes[2].set_ylabel('季节性', fontsize=11)
        axes[2].grid(alpha=0.3)
        
        axes[3].plot(decomposition.resid, color='red', linewidth=0.5, alpha=0.7)
        axes[3].set_ylabel('残差', fontsize=11)
        axes[3].set_xlabel('日期', fontsize=11)
        axes[3].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = f'{self.output_dir}/time_series_decomposition.png'
            plt.savefig(filepath, bbox_inches='tight', facecolor='white')
            self.figures.append(filepath)
            print(f"图表已保存: {filepath}")
        
        return fig
    
    def generate_all_plots(self, decomposition=None, feature_importance=None):
        """生成所有图表"""
        self.plot_hourly_distribution()
        self.plot_temperature_ridership()
        self.plot_weather_impact()
        
        if decomposition is not None:
            self.plot_time_series_decomposition(decomposition)
        
        if feature_importance is not None:
            self.plot_feature_importance(feature_importance)
        
        plt.close('all')
        
        return self.figures


if __name__ == '__main__':
    df = pd.read_csv('data/processed_data.csv')
    visualizer = BikeVisualizer(df)
    visualizer.plot_hourly_distribution()
    visualizer.plot_temperature_ridership()
    visualizer.plot_weather_impact()
    plt.show()
