import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False
rcParams['figure.dpi'] = 150
rcParams['savefig.dpi'] = 300
rcParams['figure.figsize'] = (10, 6)


class Visualizer:
    def __init__(self, output_dir='output/figures'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.set_style()
        
    def set_style(self):
        sns.set_style("whitegrid")
        plt.rcParams.update({
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9
        })
    
    def plot_hourly_distribution(self, df, save_filename='hourly_distribution.png'):
        print("生成骑行量小时分布图...")
        hourly_data = df.groupby(['hour', 'is_weekday'])['riders'].mean().reset_index()
        hourly_data['day_type'] = hourly_data['is_weekday'].map({1: '工作日', 0: '周末'})
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        sns.barplot(
            data=hourly_data,
            x='hour',
            y='riders',
            hue='day_type',
            palette=['#3498db', '#e74c3c'],
            alpha=0.8,
            ax=ax
        )
        
        ax.set_title('共享单车骑行量小时分布对比（工作日 vs 周末）', fontsize=14, pad=20)
        ax.set_xlabel('小时', fontsize=12)
        ax.set_ylabel('平均骑行人数', fontsize=12)
        ax.legend(title='')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.annotate(f'{int(height)}',
                           (p.get_x() + p.get_width() / 2., height),
                           ha='center', va='bottom', fontsize=7)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, save_filename)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"已保存: {save_path}")
        return save_path
    
    def plot_temperature_relationship(self, df, save_filename='temperature_relationship.png'):
        print("生成温度与骑行量关系散点图...")
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        sample_df = df.sample(min(5000, len(df)), random_state=42)
        
        scatter = ax.scatter(
            sample_df['temperature'],
            sample_df['riders'],
            c=sample_df['hour'],
            cmap='viridis',
            alpha=0.6,
            s=30,
            edgecolors='white',
            linewidths=0.5
        )
        
        z = np.polyfit(sample_df['temperature'], sample_df['riders'], 2)
        p = np.poly1d(z)
        x_range = np.linspace(sample_df['temperature'].min(), sample_df['temperature'].max(), 100)
        ax.plot(x_range, p(x_range), "r--", alpha=0.8, linewidth=2, label='二次拟合曲线')
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('小时', fontsize=10)
        
        ax.set_title('温度与骑行量关系散点图', fontsize=14, pad=20)
        ax.set_xlabel('温度 (°C)', fontsize=12)
        ax.set_ylabel('骑行人数', fontsize=12)
        ax.legend()
        ax.grid(linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, save_filename)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"已保存: {save_path}")
        return save_path
    
    def plot_weather_boxplot(self, df, save_filename='weather_effect_boxplot.png'):
        print("生成天气状况对骑行量影响箱线图...")
        
        weather_order = ['晴天', '多云', '小雨', '大雨']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        box_plot = sns.boxplot(
            data=df,
            x='weather',
            y='riders',
            order=weather_order,
            palette=['#2ecc71', '#f1c40f', '#3498db', '#95a5a6'],
            showfliers=False,
            ax=ax
        )
        
        means = df.groupby('weather')['riders'].mean().reindex(weather_order)
        for i, mean_val in enumerate(means):
            ax.text(i, mean_val + 2, f'均值: {mean_val:.1f}',
                   ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        ax.set_title('天气状况对骑行量影响分析', fontsize=14, pad=20)
        ax.set_xlabel('天气状况', fontsize=12)
        ax.set_ylabel('骑行人数', fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, save_filename)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"已保存: {save_path}")
        return save_path
    
    def plot_feature_importance(self, feature_importance, save_filename='feature_importance.png'):
        print("生成特征重要性图...")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        top_features = feature_importance.head(15)
        
        bars = ax.barh(
            top_features['feature'],
            top_features['importance'],
            color=sns.color_palette("viridis", len(top_features))
        )
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.005, bar.get_y() + bar.get_height()/2,
                   f'{width:.4f}',
                   va='center', fontsize=8)
        
        ax.set_title('随机森林模型特征重要性排名', fontsize=14, pad=20)
        ax.set_xlabel('重要性分数', fontsize=12)
        ax.set_ylabel('特征', fontsize=12)
        ax.invert_yaxis()
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, save_filename)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"已保存: {save_path}")
        return save_path
    
    def plot_timeseries_decomposition(self, decomposition, save_filename='timeseries_decomposition.png'):
        print("生成时间序列分解图...")
        
        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
        
        axes[0].plot(decomposition.observed, label='观测值', color='#34495e')
        axes[0].set_title('时间序列分解', fontsize=14, pad=15)
        axes[0].set_ylabel('观测值')
        axes[0].legend()
        axes[0].grid(linestyle='--', alpha=0.5)
        
        axes[1].plot(decomposition.trend, label='趋势项', color='#e74c3c')
        axes[1].set_ylabel('趋势')
        axes[1].legend()
        axes[1].grid(linestyle='--', alpha=0.5)
        
        axes[2].plot(decomposition.seasonal, label='季节项', color='#27ae60')
        axes[2].set_ylabel('季节周期')
        axes[2].legend()
        axes[2].grid(linestyle='--', alpha=0.5)
        
        axes[3].scatter(decomposition.resid.index, decomposition.resid, 
                       label='残差项', color='#95a5a6', s=10, alpha=0.6)
        axes[3].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[3].set_ylabel('残差')
        axes[3].set_xlabel('日期')
        axes[3].legend()
        axes[3].grid(linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, save_filename)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"已保存: {save_path}")
        return save_path
    
    def generate_all_charts(self, df, feature_importance=None, decomposition=None):
        print("=" * 60)
        print("开始生成所有可视化图表...")
        print("=" * 60)
        
        chart_paths = {}
        
        chart_paths['hourly_dist'] = self.plot_hourly_distribution(df)
        chart_paths['temp_relationship'] = self.plot_temperature_relationship(df)
        chart_paths['weather_effect'] = self.plot_weather_boxplot(df)
        
        if feature_importance is not None:
            chart_paths['feature_importance'] = self.plot_feature_importance(feature_importance)
        
        if decomposition is not None:
            chart_paths['ts_decomposition'] = self.plot_timeseries_decomposition(decomposition)
        
        print("=" * 60)
        print(f"可视化完成，共生成 {len(chart_paths)} 张图表")
        print("=" * 60)
        
        return chart_paths


if __name__ == '__main__':
    from data_generator import BikeShareDataGenerator
    from preprocessor import DataPreprocessor
    
    generator = BikeShareDataGenerator(days=60)
    df_raw = generator.generate_dataset()
    
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.preprocess_pipeline(df_raw)
    
    visualizer = Visualizer()
    chart_paths = visualizer.generate_all_charts(df_processed)
