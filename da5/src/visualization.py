import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


def plot_temperature_trend(data, in_sample_pred=None, forecast=None, save_path='temperature_trend.png'):
    print("生成气温趋势图...")
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    ax1 = axes[0]
    ax1.plot(data['date'], data['temperature'], label='实际气温', color='#1f77b4', alpha=0.7)
    ax1.set_title('城市气温时间序列趋势', fontsize=14, pad=20)
    ax1.set_xlabel('日期', fontsize=12)
    ax1.set_ylabel('气温 (°C)', fontsize=12)
    ax1.legend(loc='best')
    ax1.tick_params(axis='x', rotation=45)

    ax2 = axes[1]
    monthly_avg = data.groupby('month')['temperature'].mean()
    months = ['1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月', '9月', '10月', '11月', '12月']
    bars = ax2.bar(months, monthly_avg.values, color='#ff7f0e', alpha=0.7)
    ax2.set_title('月平均气温分布', fontsize=14, pad=20)
    ax2.set_xlabel('月份', fontsize=12)
    ax2.set_ylabel('平均气温 (°C)', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)

    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"气温趋势图已保存至: {save_path}")


def plot_correlation_heatmap(corr_matrix, save_path='correlation_heatmap.png'):
    print("生成相关性热力图...")
    plt.figure(figsize=(10, 8))

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix,
                annot=True,
                cmap='coolwarm',
                center=0,
                square=True,
                linewidths=1,
                cbar_kws={"shrink": .8},
                fmt='.3f',
                annot_kws={'size': 12})

    plt.title('气象要素相关性热力图', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"相关性热力图已保存至: {save_path}")


def plot_forecast_comparison(data, forecast, save_path='forecast_comparison.png'):
    print("生成预测结果对比图...")
    plt.figure(figsize=(16, 8))

    last_90_days = data.tail(90)
    plt.plot(last_90_days['date'], last_90_days['temperature'],
             label='历史实际气温 (最近90天)', color='#1f77b4', linewidth=2)

    plt.plot(forecast['date'], forecast['predicted_temperature'],
             label='预测气温', color='#ff7f0e', linewidth=2, linestyle='--')

    plt.fill_between(forecast['date'],
                     forecast['lower_bound'],
                     forecast['upper_bound'],
                     color='#ff7f0e', alpha=0.2,
                     label='95% 置信区间')

    plt.title('ARIMA短期气温预测结果', fontsize=16, pad=20)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('气温 (°C)', fontsize=12)
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"预测结果对比图已保存至: {save_path}")


def generate_all_visualizations(data, corr_matrix, forecast, output_dir='.'):
    import os
    os.makedirs(output_dir, exist_ok=True)

    plot_temperature_trend(
        data,
        save_path=f'{output_dir}/temperature_trend.png'
    )

    plot_correlation_heatmap(
        corr_matrix,
        save_path=f'{output_dir}/correlation_heatmap.png'
    )

    plot_forecast_comparison(
        data,
        forecast,
        save_path=f'{output_dir}/forecast_comparison.png'
    )

    print("所有可视化报表生成完成！")
