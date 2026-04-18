import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
import os

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class DataVisualizer:
    
    def __init__(self, output_dir: str = 'output'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.figures = {}
    
    def plot_hourly_distribution(self, df: pd.DataFrame, 
                                 save_path: Optional[str] = None) -> str:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        hourly_avg = df.groupby('hour')['ridership'].mean()
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, 24))
        bars = ax.bar(hourly_avg.index, hourly_avg.values, color=colors, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('小时', fontsize=12)
        ax.set_ylabel('平均骑行量', fontsize=12)
        ax.set_title('骑行量小时分布图', fontsize=14, fontweight='bold')
        ax.set_xticks(range(24))
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.0f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'hourly_distribution.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.figures['hourly_distribution'] = save_path
        return save_path
    
    def plot_temperature_ridership(self, df: pd.DataFrame,
                                   save_path: Optional[str] = None) -> str:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        scatter = ax.scatter(df['temperature'], df['ridership'],
                           c=df['hour'], cmap='coolwarm', alpha=0.6,
                           s=20, edgecolors='black', linewidth=0.2)
        
        z = np.polyfit(df['temperature'].dropna(), 
                      df.loc[df['temperature'].notna(), 'ridership'], 1)
        p = np.poly1d(z)
        temp_range = np.linspace(df['temperature'].min(), df['temperature'].max(), 100)
        ax.plot(temp_range, p(temp_range), "r--", linewidth=2, label='趋势线')
        
        ax.set_xlabel('温度 (°C)', fontsize=12)
        ax.set_ylabel('骑行量', fontsize=12)
        ax.set_title('温度与骑行量关系散点图', fontsize=14, fontweight='bold')
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('小时', fontsize=10)
        
        ax.legend(loc='upper right')
        ax.grid(alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'temperature_ridership.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.figures['temperature_ridership'] = save_path
        return save_path
    
    def plot_weather_impact(self, df: pd.DataFrame,
                           save_path: Optional[str] = None) -> str:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        weather_order = ['晴天', '多云', '阴天', '小雨', '大雨']
        weather_data = df[df['weather'].isin(weather_order)]
        
        colors = ['#FFD700', '#87CEEB', '#A9A9A9', '#4169E1', '#000080']
        
        box = ax.boxplot([weather_data[weather_data['weather'] == w]['ridership'].values 
                         for w in weather_order],
                        labels=weather_order, patch_artist=True)
        
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_xlabel('天气状况', fontsize=12)
        ax.set_ylabel('骑行量', fontsize=12)
        ax.set_title('天气状况对骑行量影响箱线图', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        weather_stats = weather_data.groupby('weather')['ridership'].agg(['mean', 'median', 'std'])
        for i, w in enumerate(weather_order):
            if w in weather_stats.index:
                stats = weather_stats.loc[w]
                ax.annotate(f'均值:{stats["mean"]:.0f}\n中位数:{stats["median"]:.0f}',
                           xy=(i + 1, weather_data[weather_data['weather'] == w]['ridership'].max()),
                           ha='center', fontsize=8)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'weather_impact.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.figures['weather_impact'] = save_path
        return save_path
    
    def plot_feature_importance(self, feature_importance: pd.DataFrame,
                               save_path: Optional[str] = None) -> str:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        top_features = feature_importance.head(10).copy()
        
        feature_name_map = {
            'hour': '小时',
            'day': '日期',
            'month': '月份',
            'weekday': '星期',
            'is_workday': '是否工作日',
            'is_weekend': '是否周末',
            'temperature': '温度',
            'humidity': '湿度',
            'wind_speed': '风速',
            'station_id': '站点ID',
            'weather_晴天': '天气-晴天',
            'weather_多云': '天气-多云',
            'weather_阴天': '天气-阴天',
            'weather_小雨': '天气-小雨',
            'weather_大雨': '天气-大雨'
        }
        
        top_features['feature_cn'] = top_features['feature'].map(
            lambda x: feature_name_map.get(x, x)
        )
        
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(top_features)))
        
        bars = ax.barh(range(len(top_features)), top_features['importance'].values,
                      color=colors, edgecolor='black', linewidth=0.5)
        
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature_cn'].values)
        ax.set_xlabel('重要性', fontsize=12)
        ax.set_ylabel('特征', fontsize=12)
        ax.set_title('特征重要性排名 (Top 10)', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        for bar, val in zip(bars, top_features['importance'].values):
            ax.annotate(f'{val:.4f}',
                       xy=(val, bar.get_y() + bar.get_height() / 2),
                       xytext=(5, 0),
                       textcoords="offset points",
                       ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'feature_importance.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.figures['feature_importance'] = save_path
        return save_path
    
    def plot_time_series_decomposition(self, ts_analysis: dict,
                                       save_path: Optional[str] = None) -> str:
        fig, axes = plt.subplots(4, 1, figsize=(14, 12))
        
        observed = ts_analysis.get('observed')
        trend = ts_analysis.get('trend')
        seasonal = ts_analysis.get('seasonal')
        residual = ts_analysis.get('residual')
        
        if observed is not None:
            axes[0].plot(observed.index, observed.values, color='steelblue', linewidth=1)
            axes[0].set_ylabel('观测值', fontsize=10)
            axes[0].set_title('时间序列分解 - 观测值', fontsize=12, fontweight='bold')
            axes[0].grid(alpha=0.3, linestyle='--')
        
        if trend is not None:
            axes[1].plot(trend.index, trend.values, color='darkorange', linewidth=1.5)
            axes[1].set_ylabel('趋势', fontsize=10)
            axes[1].set_title('趋势成分', fontsize=12, fontweight='bold')
            axes[1].grid(alpha=0.3, linestyle='--')
        
        if seasonal is not None:
            axes[2].plot(seasonal.index, seasonal.values, color='green', linewidth=1)
            axes[2].set_ylabel('季节性', fontsize=10)
            axes[2].set_title('季节性成分 (周期=7天)', fontsize=12, fontweight='bold')
            axes[2].grid(alpha=0.3, linestyle='--')
        
        if residual is not None:
            axes[3].plot(residual.index, residual.values, color='red', linewidth=0.5, alpha=0.7)
            axes[3].set_ylabel('残差', fontsize=10)
            axes[3].set_title('残差成分', fontsize=12, fontweight='bold')
            axes[3].grid(alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'time_series_decomposition.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.figures['time_series_decomposition'] = save_path
        return save_path
    
    def generate_all_plots(self, df: pd.DataFrame, 
                          feature_importance: pd.DataFrame,
                          ts_analysis: dict) -> dict:
        self.plot_hourly_distribution(df)
        self.plot_temperature_ridership(df)
        self.plot_weather_impact(df)
        self.plot_feature_importance(feature_importance)
        self.plot_time_series_decomposition(ts_analysis)
        
        return self.figures
