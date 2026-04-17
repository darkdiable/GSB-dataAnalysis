"""
可视化模块：生成各类分析图表
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 设置图表风格
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300


class BikeSharingVisualizer:
    """共享单车数据可视化器"""
    
    def __init__(self, output_dir='output'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.plots_generated = []
        
    def plot_hourly_distribution(self, df, save=True):
        """
        绘制骑行量小时分布图
        
        Args:
            df: 数据DataFrame
            save: 是否保存图片
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 计算每小时的平均骑行量
        df_copy = df.copy()
        df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
        hourly_avg = df_copy.groupby(df_copy['timestamp'].dt.hour)['ridership'].mean()
        
        # 绘制柱状图
        bars = ax.bar(hourly_avg.index, hourly_avg.values, 
                      color='steelblue', alpha=0.8, edgecolor='navy', linewidth=0.5)
        
        # 高亮高峰时段
        for i, bar in enumerate(bars):
            if i in [7, 8, 9, 17, 18, 19]:
                bar.set_color('coral')
        
        ax.set_xlabel('小时 (Hour)', fontsize=12)
        ax.set_ylabel('平均骑行人数', fontsize=12)
        ax.set_title('共享单车骑行量小时分布图', fontsize=14, fontweight='bold')
        ax.set_xticks(range(0, 24))
        ax.set_xticklabels([f'{h:02d}' for h in range(24)])
        ax.grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        for i, v in enumerate(hourly_avg.values):
            if i % 2 == 0:  # 每隔一个显示，避免拥挤
                ax.text(i, v + 1, f'{v:.0f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, '01_hourly_distribution.png')
            plt.savefig(filepath, bbox_inches='tight', facecolor='white')
            self.plots_generated.append(filepath)
            print(f"  已保存: {filepath}")
        
        return fig
    
    def plot_temperature_ridership_scatter(self, df, save=True):
        """
        绘制温度与骑行量关系散点图
        
        Args:
            df: 数据DataFrame
            save: 是否保存图片
        """
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # 按天气分组绘制散点图
        weather_colors = {
            '晴天': '#FF6B6B',
            '多云': '#4ECDC4',
            '阴天': '#95A5A6',
            '小雨': '#3498DB',
            '大雨': '#2C3E50'
        }
        
        for weather in df['weather'].unique():
            subset = df[df['weather'] == weather]
            ax.scatter(subset['temperature'], subset['ridership'],
                      c=weather_colors.get(weather, 'gray'),
                      label=weather, alpha=0.6, s=30, edgecolors='white', linewidth=0.3)
        
        # 添加趋势线
        z = np.polyfit(df['temperature'], df['ridership'], 2)
        p = np.poly1d(z)
        temp_range = np.linspace(df['temperature'].min(), df['temperature'].max(), 100)
        ax.plot(temp_range, p(temp_range), "r--", linewidth=2, label='趋势线', alpha=0.8)
        
        ax.set_xlabel('温度 (°C)', fontsize=12)
        ax.set_ylabel('骑行人数', fontsize=12)
        ax.set_title('温度与骑行量关系散点图', fontsize=14, fontweight='bold')
        ax.legend(title='天气状况', loc='upper left', framealpha=0.9)
        ax.grid(alpha=0.3)
        
        # 添加相关系数
        corr = df['temperature'].corr(df['ridership'])
        ax.text(0.02, 0.98, f'相关系数: {corr:.3f}', 
                transform=ax.transAxes, fontsize=11,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, '02_temperature_ridership.png')
            plt.savefig(filepath, bbox_inches='tight', facecolor='white')
            self.plots_generated.append(filepath)
            print(f"  已保存: {filepath}")
        
        return fig
    
    def plot_weather_boxplot(self, df, save=True):
        """
        绘制天气状况对骑行量影响箱线图
        
        Args:
            df: 数据DataFrame
            save: 是否保存图片
        """
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # 定义天气顺序（从好到差）
        weather_order = ['晴天', '多云', '阴天', '小雨', '大雨']
        
        # 创建箱线图
        box_plot = sns.boxplot(data=df, x='weather', y='ridership', 
                               order=weather_order,
                               palette=['#FF6B6B', '#4ECDC4', '#95A5A6', '#3498DB', '#2C3E50'],
                               ax=ax)
        
        # 添加均值点
        sns.stripplot(data=df, x='weather', y='ridership',
                      order=weather_order,
                      color='red', alpha=0.3, size=3, jitter=True, ax=ax)
        
        ax.set_xlabel('天气状况', fontsize=12)
        ax.set_ylabel('骑行人数', fontsize=12)
        ax.set_title('天气状况对骑行量影响箱线图', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # 添加统计信息
        for i, weather in enumerate(weather_order):
            if weather in df['weather'].values:
                subset = df[df['weather'] == weather]['ridership']
                mean_val = subset.mean()
                ax.text(i, ax.get_ylim()[1] * 0.95, f'均值: {mean_val:.0f}',
                       ha='center', fontsize=9, color='darkred', fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, '03_weather_boxplot.png')
            plt.savefig(filepath, bbox_inches='tight', facecolor='white')
            self.plots_generated.append(filepath)
            print(f"  已保存: {filepath}")
        
        return fig
    
    def plot_feature_importance(self, feature_importance_df, save=True):
        """
        绘制特征重要性图
        
        Args:
            feature_importance_df: 特征重要性DataFrame
            save: 是否保存图片
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 取前15个重要特征
        top_features = feature_importance_df.head(15)
        
        # 水平条形图
        bars = ax.barh(range(len(top_features)), top_features['importance'].values,
                       color='teal', alpha=0.8)
        
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'].values)
        ax.invert_yaxis()  # 重要性高的在上面
        ax.set_xlabel('重要性', fontsize=12)
        ax.set_title('特征重要性排名 (随机森林)', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # 添加数值标签
        for i, (idx, row) in enumerate(top_features.iterrows()):
            ax.text(row['importance'] + 0.005, i, f'{row["importance"]:.3f}',
                   va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, '04_feature_importance.png')
            plt.savefig(filepath, bbox_inches='tight', facecolor='white')
            self.plots_generated.append(filepath)
            print(f"  已保存: {filepath}")
        
        return fig
    
    def plot_time_series_decomposition(self, decomposition_components, save=True):
        """
        绘制时间序列分解图
        
        Args:
            decomposition_components: 分解组件字典
            save: 是否保存图片
        """
        if decomposition_components is None:
            print("  跳过时间序列分解图（无分解数据）")
            return None
        
        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
        
        observed = decomposition_components['observed']
        trend = decomposition_components['trend']
        seasonal = decomposition_components['seasonal']
        residual = decomposition_components['resid']
        
        # 原始数据
        axes[0].plot(observed.index, observed.values, color='black', linewidth=0.8)
        axes[0].set_ylabel('原始数据', fontsize=11)
        axes[0].set_title('时间序列分解', fontsize=14, fontweight='bold')
        axes[0].grid(alpha=0.3)
        
        # 趋势
        axes[1].plot(trend.index, trend.values, color='red', linewidth=1.5)
        axes[1].set_ylabel('趋势', fontsize=11)
        axes[1].grid(alpha=0.3)
        
        # 季节性
        axes[2].plot(seasonal.index, seasonal.values, color='green', linewidth=1)
        axes[2].set_ylabel('季节性', fontsize=11)
        axes[2].grid(alpha=0.3)
        
        # 残差
        axes[3].plot(residual.index, residual.values, color='blue', linewidth=0.5, alpha=0.7)
        axes[3].set_ylabel('残差', fontsize=11)
        axes[3].set_xlabel('时间', fontsize=11)
        axes[3].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, '05_time_series_decomposition.png')
            plt.savefig(filepath, bbox_inches='tight', facecolor='white')
            self.plots_generated.append(filepath)
            print(f"  已保存: {filepath}")
        
        return fig
    
    def generate_all_plots(self, df, feature_importance_df=None, decomposition_components=None):
        """
        生成所有图表
        
        Args:
            df: 数据DataFrame
            feature_importance_df: 特征重要性DataFrame
            decomposition_components: 时间序列分解组件
        """
        print("\n" + "="*50)
        print("生成可视化图表")
        print("="*50)
        
        # 1. 小时分布图
        print("\n1. 骑行量小时分布图")
        self.plot_hourly_distribution(df)
        plt.close()
        
        # 2. 温度与骑行量散点图
        print("\n2. 温度与骑行量关系散点图")
        self.plot_temperature_ridership_scatter(df)
        plt.close()
        
        # 3. 天气箱线图
        print("\n3. 天气状况对骑行量影响箱线图")
        self.plot_weather_boxplot(df)
        plt.close()
        
        # 4. 特征重要性图
        if feature_importance_df is not None:
            print("\n4. 特征重要性图")
            self.plot_feature_importance(feature_importance_df)
            plt.close()
        
        # 5. 时间序列分解图
        if decomposition_components is not None:
            print("\n5. 时间序列分解图")
            self.plot_time_series_decomposition(decomposition_components)
            plt.close()
        
        print(f"\n所有图表已保存至: {self.output_dir}/")
        return self.plots_generated
    
    def get_generated_plots(self):
        """获取已生成的图表路径列表"""
        return self.plots_generated


if __name__ == '__main__':
    # 测试可视化
    from data_generator import BikeSharingDataGenerator
    
    generator = BikeSharingDataGenerator(n_samples=2000)
    data = generator.generate()
    
    visualizer = BikeSharingVisualizer(output_dir='test_output')
    visualizer.generate_all_plots(data)
    
    print("\n测试完成!")
