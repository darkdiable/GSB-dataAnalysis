import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class Visualizer:
    def __init__(self, output_dir='output'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置样式
        sns.set_style("whitegrid")
        plt.style.use('seaborn-v0_8-whitegrid')

    def plot_temperature_trend(self, data, save_path=None):
        """
        绘制气温趋势折线图
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('气温趋势分析', fontsize=16, fontweight='bold')
        
        # 1. 日气温趋势
        ax1 = axes[0, 0]
        ax1.plot(data['date'], data['temperature'], color='steelblue', alpha=0.7, linewidth=0.8)
        ax1.set_title('日平均气温趋势', fontsize=12)
        ax1.set_xlabel('日期')
        ax1.set_ylabel('温度 (°C)')
        ax1.grid(True, alpha=0.3)
        
        # 2. 月平均气温
        ax2 = axes[0, 1]
        monthly_temp = data.groupby(data['date'].dt.to_period('M'))['temperature'].mean()
        monthly_temp.index = monthly_temp.index.to_timestamp()
        ax2.plot(monthly_temp.index, monthly_temp.values, color='darkorange', marker='o', linewidth=2, markersize=4)
        ax2.set_title('月平均气温趋势', fontsize=12)
        ax2.set_xlabel('日期')
        ax2.set_ylabel('温度 (°C)')
        ax2.grid(True, alpha=0.3)
        
        # 3. 季节箱线图
        ax3 = axes[1, 0]
        season_order = ['春季', '夏季', '秋季', '冬季']
        sns.boxplot(data=data, x='season', y='temperature', order=season_order, ax=ax3, palette='Set2')
        ax3.set_title('季节温度分布', fontsize=12)
        ax3.set_xlabel('季节')
        ax3.set_ylabel('温度 (°C)')
        
        # 4. 年度平均气温对比
        ax4 = axes[1, 1]
        yearly_temp = data.groupby('year')['temperature'].mean()
        bars = ax4.bar(yearly_temp.index, yearly_temp.values, color='mediumseagreen', alpha=0.8)
        ax4.set_title('年度平均气温对比', fontsize=12)
        ax4.set_xlabel('年份')
        ax4.set_ylabel('平均温度 (°C)')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 在柱状图上添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'temperature_trend.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"气温趋势图已保存: {save_path}")
        plt.close()
        
        return save_path

    def plot_correlation_heatmap(self, data, save_path=None):
        """
        绘制气象要素相关性热力图
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('气象要素相关性分析', fontsize=16, fontweight='bold')
        
        # 1. 相关系数热力图
        ax1 = axes[0]
        numeric_columns = ['temperature', 'humidity', 'precipitation', 'wind_speed', 'pressure']
        corr_matrix = data[numeric_columns].corr()
        
        # 中文标签映射
        label_map = {
            'temperature': '温度',
            'humidity': '湿度',
            'precipitation': '降水量',
            'wind_speed': '风速',
            'pressure': '气压'
        }
        corr_matrix.index = [label_map.get(col, col) for col in corr_matrix.index]
        corr_matrix.columns = [label_map.get(col, col) for col in corr_matrix.columns]
        
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                   square=True, ax=ax1, fmt='.3f', cbar_kws={'shrink': 0.8})
        ax1.set_title('气象要素相关系数热力图', fontsize=12)
        
        # 2. 散点图矩阵（关键变量）
        ax2 = axes[1]
        scatter_data = data[['temperature', 'humidity', 'precipitation']].copy()
        scatter_data.columns = ['温度', '湿度', '降水量']
        
        # 绘制温度-湿度散点图
        ax2.scatter(data['temperature'], data['humidity'], alpha=0.5, c='steelblue', s=20)
        ax2.set_xlabel('温度 (°C)')
        ax2.set_ylabel('湿度 (%)')
        ax2.set_title('温度-湿度相关性散点图', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # 添加趋势线
        z = np.polyfit(data['temperature'], data['humidity'], 1)
        p = np.poly1d(z)
        ax2.plot(data['temperature'].sort_values(), p(data['temperature'].sort_values()), 
                "r--", alpha=0.8, linewidth=2, label='趋势线')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'correlation_heatmap.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"相关性热力图已保存: {save_path}")
        plt.close()
        
        return save_path

    def plot_prediction_comparison(self, data, predictor, save_path=None):
        """
        绘制预测结果对比图
        """
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        fig.suptitle('气温预测结果分析', fontsize=16, fontweight='bold')
        
        # 1. 历史数据与预测对比
        ax1 = axes[0]
        
        # 绘制历史数据（最近180天）
        recent_data = data.tail(180)
        ax1.plot(recent_data['date'], recent_data['temperature'], 
                label='历史气温', color='steelblue', linewidth=1.5)
        
        # 绘制测试集预测
        if hasattr(predictor, 'test_data') and hasattr(predictor, 'test_forecast'):
            test_dates = predictor.test_data.index
            ax1.plot(test_dates, predictor.test_forecast, 
                    label='测试集预测', color='orange', linestyle='--', linewidth=2)
        
        # 绘制未来预测
        if predictor.forecast_results is not None:
            forecast = predictor.forecast_results
            ax1.plot(forecast['date'], forecast['forecast'], 
                    label='未来预测', color='red', linewidth=2)
            ax1.fill_between(forecast['date'], forecast['lower_ci'], forecast['upper_ci'], 
                           alpha=0.2, color='red', label='95%置信区间')
        
        ax1.set_title('气温预测趋势对比', fontsize=12)
        ax1.set_xlabel('日期')
        ax1.set_ylabel('温度 (°C)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 预测误差分析
        ax2 = axes[1]
        
        if hasattr(predictor, 'test_data') and hasattr(predictor, 'test_forecast'):
            errors = predictor.test_data.values - predictor.test_forecast.values
            
            # 误差分布直方图
            ax2.hist(errors, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
            ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='零误差线')
            ax2.set_title('预测误差分布', fontsize=12)
            ax2.set_xlabel('预测误差 (°C)')
            ax2.set_ylabel('频数')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'prediction_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"预测对比图已保存: {save_path}")
        plt.close()
        
        return save_path

    def plot_comprehensive_dashboard(self, data, analyzer, predictor, save_path=None):
        """
        绘制综合仪表板
        """
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        fig.suptitle('城市气象特征分析与预测综合报表', fontsize=18, fontweight='bold', y=0.98)
        
        # 1. 温度时间序列
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(data['date'], data['temperature'], color='steelblue', alpha=0.7, linewidth=0.8)
        ax1.set_title('日平均气温变化趋势', fontsize=11)
        ax1.set_ylabel('温度 (°C)')
        ax1.grid(True, alpha=0.3)
        
        # 2. 季节统计
        ax2 = fig.add_subplot(gs[0, 2])
        season_order = ['春季', '夏季', '秋季', '冬季']
        seasonal_data = [data[data['season'] == s]['temperature'].values for s in season_order]
        bp = ax2.boxplot(seasonal_data, labels=season_order, patch_artist=True)
        colors = ['lightgreen', 'gold', 'orange', 'lightblue']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        ax2.set_title('季节温度分布', fontsize=11)
        ax2.set_ylabel('温度 (°C)')
        ax2.grid(True, alpha=0.3)
        
        # 3. 相关性热力图
        ax3 = fig.add_subplot(gs[1, 0])
        numeric_columns = ['temperature', 'humidity', 'precipitation', 'wind_speed', 'pressure']
        corr_matrix = data[numeric_columns].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                   square=True, ax=ax3, fmt='.2f', cbar=False)
        ax3.set_title('气象要素相关性', fontsize=11)
        
        # 4. 月度平均温度
        ax4 = fig.add_subplot(gs[1, 1])
        monthly_avg = data.groupby('month')['temperature'].mean()
        ax4.plot(monthly_avg.index, monthly_avg.values, marker='o', color='darkorange', linewidth=2, markersize=6)
        ax4.set_title('月均温变化', fontsize=11)
        ax4.set_xlabel('月份')
        ax4.set_ylabel('温度 (°C)')
        ax4.set_xticks(range(1, 13))
        ax4.grid(True, alpha=0.3)
        
        # 5. 预测结果
        ax5 = fig.add_subplot(gs[1, 2])
        if predictor.forecast_results is not None:
            forecast = predictor.forecast_results
            ax5.plot(forecast['date'][:30], forecast['forecast'][:30], 
                    color='red', linewidth=2, label='预测值')
            ax5.fill_between(forecast['date'][:30], forecast['lower_ci'][:30], 
                           forecast['upper_ci'][:30], alpha=0.2, color='red')
            ax5.set_title('未来30天气温预测', fontsize=11)
            ax5.set_ylabel('温度 (°C)')
            ax5.tick_params(axis='x', rotation=45)
            ax5.grid(True, alpha=0.3)
        
        # 6. 降水量分布
        ax6 = fig.add_subplot(gs[2, 0])
        monthly_rain = data.groupby('month')['precipitation'].sum()
        ax6.bar(monthly_rain.index, monthly_rain.values, color='skyblue', alpha=0.8)
        ax6.set_title('月累计降水量', fontsize=11)
        ax6.set_xlabel('月份')
        ax6.set_ylabel('降水量 (mm)')
        ax6.set_xticks(range(1, 13))
        ax6.grid(True, alpha=0.3, axis='y')
        
        # 7. 温湿度关系
        ax7 = fig.add_subplot(gs[2, 1])
        ax7.scatter(data['temperature'], data['humidity'], alpha=0.3, s=10, c='purple')
        ax7.set_title('温度-湿度关系', fontsize=11)
        ax7.set_xlabel('温度 (°C)')
        ax7.set_ylabel('湿度 (%)')
        ax7.grid(True, alpha=0.3)
        
        # 8. 模型评估指标
        ax8 = fig.add_subplot(gs[2, 2])
        ax8.axis('off')
        if predictor.model_metrics:
            metrics_text = "模型评估指标\n" + "="*25 + "\n\n"
            for key, value in predictor.model_metrics.items():
                metrics_text += f"{key}: {value}\n"
            ax8.text(0.1, 0.9, metrics_text, transform=ax8.transAxes, 
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'comprehensive_dashboard.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"综合仪表板已保存: {save_path}")
        plt.close()
        
        return save_path

    def get_all_visualizations(self, data, analyzer, predictor):
        """
        生成所有可视化图表
        """
        paths = []
        paths.append(self.plot_temperature_trend(data))
        paths.append(self.plot_correlation_heatmap(data))
        paths.append(self.plot_prediction_comparison(data, predictor))
        paths.append(self.plot_comprehensive_dashboard(data, analyzer, predictor))
        return paths
