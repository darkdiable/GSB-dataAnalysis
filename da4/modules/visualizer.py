import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.font_manager import FontProperties
import os


class Visualizer:
    def __init__(self, output_dir='output'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['figure.dpi'] = 100

    def plot_sales_trend(self, data, date_column, sales_column, title='销售趋势图'):
        fig, ax = plt.subplots(figsize=(14, 6))
        
        ax.plot(data[date_column], data[sales_column], 
                color='#2E86AB', linewidth=1.5, alpha=0.8, label='实际销售额')
        
        if 'trend' in data.columns:
            ax.plot(data[date_column], data['trend'], 
                    color='#E94F37', linewidth=2, linestyle='--', label='趋势线')
        
        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel('销售额', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.xticks(rotation=45)
        
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper left', fontsize=10)
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, 'sales_trend.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"销售趋势图已保存: {output_path}")
        return output_path

    def plot_prediction_comparison(self, historical_data, prediction_data, 
                                   date_column, sales_column, title='销售预测对比图'):
        fig, ax = plt.subplots(figsize=(14, 6))
        
        ax.plot(historical_data[date_column], historical_data[sales_column], 
                color='#2E86AB', linewidth=1.5, label='历史销售额', alpha=0.8)
        
        ax.plot(prediction_data[date_column], prediction_data['predicted_sales'], 
                color='#F39C12', linewidth=2, linestyle='--', label='预测销售额')
        
        last_hist_date = historical_data[date_column].iloc[-1]
        last_hist_value = historical_data[sales_column].iloc[-1]
        first_pred_date = prediction_data[date_column].iloc[0]
        first_pred_value = prediction_data['predicted_sales'].iloc[0]
        
        ax.plot([last_hist_date, first_pred_date], [last_hist_value, first_pred_value], 
                color='#F39C12', linewidth=2, linestyle='--')
        
        ax.axvline(x=last_hist_date, color='#E74C3C', linestyle=':', linewidth=1.5, 
                   label='预测起点', alpha=0.7)
        
        ax.fill_between(prediction_data[date_column], 
                        prediction_data['predicted_sales'] * 0.9,
                        prediction_data['predicted_sales'] * 1.1,
                        color='#F39C12', alpha=0.2, label='预测区间(±10%)')
        
        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel('销售额', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.xticks(rotation=45)
        
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper left', fontsize=10)
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, 'prediction_comparison.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"预测对比图已保存: {output_path}")
        return output_path

    def plot_feature_importance(self, feature_importance, title='特征重要性分析', top_n=15):
        if feature_importance is None or len(feature_importance) == 0:
            print("无特征重要性数据")
            return None
        
        top_features = feature_importance.head(top_n)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
        
        bars = ax.barh(range(len(top_features)), top_features['importance'], 
                       color=colors, alpha=0.8)
        
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'], fontsize=10)
        
        ax.set_xlabel('重要性得分', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        for i, (idx, row) in enumerate(top_features.iterrows()):
            ax.text(row['importance'], i, f' {row["importance"]:.4f}', 
                    va='center', fontsize=9)
        
        ax.grid(True, axis='x', alpha=0.3, linestyle='--')
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, 'feature_importance.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"特征重要性图已保存: {output_path}")
        return output_path

    def plot_seasonal_decomposition(self, data, date_column, sales_column, title='时间序列分解'):
        fig, axes = plt.subplots(4, 1, figsize=(14, 10))
        
        axes[0].plot(data[date_column], data[sales_column], color='#2E86AB', linewidth=1)
        axes[0].set_ylabel('原始数据', fontsize=11)
        axes[0].set_title(title, fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3, linestyle='--')
        
        if 'trend' in data.columns:
            axes[1].plot(data[date_column], data['trend'], color='#E94F37', linewidth=1.5)
            axes[1].set_ylabel('趋势项', fontsize=11)
            axes[1].grid(True, alpha=0.3, linestyle='--')
        
        if 'seasonal' in data.columns:
            axes[2].plot(data[date_column], data['seasonal'], color='#27AE60', linewidth=1)
            axes[2].set_ylabel('季节项', fontsize=11)
            axes[2].grid(True, alpha=0.3, linestyle='--')
        
        if 'residual' in data.columns:
            axes[3].plot(data[date_column], data['residual'], color='#8E44AD', linewidth=1)
            axes[3].set_ylabel('残差项', fontsize=11)
            axes[3].set_xlabel('日期', fontsize=11)
            axes[3].grid(True, alpha=0.3, linestyle='--')
        
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, 'seasonal_decomposition.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"时间序列分解图已保存: {output_path}")
        return output_path

    def generate_report(self, historical_data, prediction_data, model_metrics, 
                       feature_importance, output_file='sales_prediction_report.txt'):
        report_path = os.path.join(self.output_dir, output_file)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("零售销售预测与可视化分析报告\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("一、数据概况\n")
            f.write("-" * 40 + "\n")
            f.write(f"历史数据时间范围: {historical_data.iloc[0, 0]} 至 {historical_data.iloc[-1, 0]}\n")
            f.write(f"历史数据记录数: {len(historical_data)} 条\n")
            f.write(f"预测未来天数: {len(prediction_data)} 天\n\n")
            
            f.write("二、模型性能指标\n")
            f.write("-" * 40 + "\n")
            f.write("训练集:\n")
            f.write(f"  R² 决定系数: {model_metrics['train']['r2']:.4f}\n")
            f.write(f"  均方误差 (MSE): {model_metrics['train']['mse']:.2f}\n")
            f.write(f"  平均绝对误差 (MAE): {model_metrics['train']['mae']:.2f}\n\n")
            f.write("测试集:\n")
            f.write(f"  R² 决定系数: {model_metrics['test']['r2']:.4f}\n")
            f.write(f"  均方误差 (MSE): {model_metrics['test']['mse']:.2f}\n")
            f.write(f"  平均绝对误差 (MAE): {model_metrics['test']['mae']:.2f}\n\n")
            
            f.write("三、预测结果摘要\n")
            f.write("-" * 40 + "\n")
            f.write(f"预测销售额最小值: {prediction_data['predicted_sales'].min():.2f}\n")
            f.write(f"预测销售额最大值: {prediction_data['predicted_sales'].max():.2f}\n")
            f.write(f"预测销售额平均值: {prediction_data['predicted_sales'].mean():.2f}\n")
            f.write(f"预测销售额中位数: {prediction_data['predicted_sales'].median():.2f}\n")
            f.write(f"预测销售额标准差: {prediction_data['predicted_sales'].std():.2f}\n\n")
            
            f.write("四、未来30天销售预测详情\n")
            f.write("-" * 40 + "\n")
            for idx, row in prediction_data.iterrows():
                f.write(f"{row[prediction_data.columns[0]].strftime('%Y-%m-%d')}: {row['predicted_sales']:.2f}\n")
            
            f.write("\n五、特征重要性排名 (Top 10)\n")
            f.write("-" * 40 + "\n")
            if feature_importance is not None:
                for idx, row in feature_importance.head(10).iterrows():
                    f.write(f"{row['feature']}: {row['importance']:.4f}\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("报告生成完成\n")
            f.write("=" * 60 + "\n")
        
        print(f"预测报告已保存: {report_path}")
        return report_path
