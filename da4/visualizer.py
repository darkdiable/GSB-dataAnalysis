import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


def setup_plot_style():
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['figure.figsize'] = (12, 8)


def plot_sales_trend(df, output_path='output/sales_trend.png'):
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    ax.plot(df['date'], df['sales'], color='#2E86AB', linewidth=1, alpha=0.7, label='日销售额')
    
    rolling_30 = df['sales'].rolling(window=30).mean()
    ax.plot(df['date'], rolling_30, color='#A23B72', linewidth=2.5, label='30天移动平均')
    
    ax.set_title('零售销售趋势分析', fontsize=16, pad=20)
    ax.set_xlabel('日期', fontsize=12)
    ax.set_ylabel('销售额', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f"销售趋势图已保存至: {output_path}")


def plot_forecast_comparison(historical_df, forecast_df, output_path='output/forecast_comparison.png'):
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    last_90_days = historical_df.tail(90)
    ax.plot(last_90_days['date'], last_90_days['sales'], 
            color='#2E86AB', linewidth=2, label='历史销售额')
    
    last_date = historical_df['date'].iloc[-1]
    last_sales = historical_df['sales'].iloc[-1]
    
    connect_dates = pd.concat([pd.Series([last_date]), forecast_df['date']])
    connect_values = pd.concat([pd.Series([last_sales]), forecast_df['predicted_sales']])
    
    ax.plot(connect_dates, connect_values, color='#F18F01', linewidth=2.5, 
            linestyle='--', label='预测销售额')
    
    ax.fill_between(forecast_df['date'], 
                    forecast_df['predicted_sales'] * 0.9, 
                    forecast_df['predicted_sales'] * 1.1,
                    color='#F18F01', alpha=0.2, label='预测区间 (±10%)')
    
    ax.axvline(x=last_date, color='red', linestyle=':', alpha=0.5, label='预测起始点')
    
    ax.set_title('销售预测结果对比 (最近90天+未来30天)', fontsize=16, pad=20)
    ax.set_xlabel('日期', fontsize=12)
    ax.set_ylabel('销售额', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f"预测结果对比图已保存至: {output_path}")


def plot_feature_importance(feature_importance, top_n=15, output_path='output/feature_importance.png'):
    setup_plot_style()
    
    top_features = feature_importance.head(top_n)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.8, len(top_features)))
    bars = ax.barh(top_features['feature'], top_features['importance'], color=colors)
    
    ax.set_title(f'特征重要性 Top {top_n}', fontsize=16, pad=20)
    ax.set_xlabel('重要性得分', fontsize=12)
    ax.set_ylabel('特征', fontsize=12)
    
    ax.invert_yaxis()
    
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                f'{width:.4f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f"特征重要性图已保存至: {output_path}")


def generate_all_charts(df, forecast_df, feature_importance):
    print("\n" + "=" * 50)
    print("开始生成可视化报表...")
    
    plot_sales_trend(df)
    plot_forecast_comparison(df, forecast_df)
    plot_feature_importance(feature_importance)
    
    print("所有可视化图表生成完成!")
    print("=" * 50)
