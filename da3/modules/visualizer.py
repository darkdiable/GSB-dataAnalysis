import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from matplotlib import font_manager
import warnings
warnings.filterwarnings('ignore')


def setup_chinese_font():
    """配置中文字体，确保图表中文正常显示"""
    possible_fonts = [
        '/System/Library/Fonts/PingFang.ttc',
        '/System/Library/Fonts/STHeiti Light.ttc',
        '/System/Library/Fonts/STHeiti Medium.ttc',
        '/Library/Fonts/Arial Unicode.ttf',
        '/Library/Fonts/SimHei.ttf',
    ]
    
    font_names = []
    for font_path in possible_fonts:
        if os.path.exists(font_path):
            font_prop = font_manager.FontProperties(fname=font_path)
            font_names.append(font_prop.get_name())
    
    font_names.extend([
        'PingFang SC', 'Heiti SC', 'STHeiti',
        'Arial Unicode MS', 'SimHei',
        'Microsoft YaHei', 'DejaVu Sans'
    ])
    
    plt.rcParams['font.sans-serif'] = font_names
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 10
    
    sns.set_style("whitegrid", {"font.sans-serif": font_names})


setup_chinese_font()


def plot_sales_trend(daily_sales, output_dir='output'):
    """
    绘制销售额趋势折线图（含移动平均）
    """
    print("绘制销售额趋势图...")
    os.makedirs(output_dir, exist_ok=True)
    setup_chinese_font()
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.plot(daily_sales['日期'], daily_sales['订单金额'], 
            label='日销售额', alpha=0.6, color='#3498db', linewidth=1)
    
    if 'MA7' in daily_sales.columns:
        ax.plot(daily_sales['日期'], daily_sales['MA7'], 
                label='7日移动平均', color='#e74c3c', linewidth=2)
    
    if 'MA30' in daily_sales.columns and not daily_sales['MA30'].isnull().all():
        ax.plot(daily_sales['日期'], daily_sales['MA30'], 
                label='30日移动平均', color='#2ecc71', linewidth=2)
    
    ax.set_title('销售额趋势分析', fontsize=16, pad=20)
    ax.set_xlabel('日期', fontsize=12)
    ax.set_ylabel('销售额（元）', fontsize=12)
    ax.legend(fontsize=10)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, '1_sales_trend.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存: {output_path}")


def plot_amount_distribution(df, output_dir='output'):
    """
    绘制金额分布：直方图 + 箱线图
    """
    print("绘制订单金额分布图...")
    os.makedirs(output_dir, exist_ok=True)
    setup_chinese_font()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.histplot(data=df, x='订单金额', kde=True, ax=ax1, 
                 color='#3498db', bins=30, alpha=0.7)
    ax1.set_title('订单金额分布直方图', fontsize=14, pad=15)
    ax1.set_xlabel('订单金额（元）', fontsize=11)
    ax1.set_ylabel('频次', fontsize=11)
    
    sns.boxplot(data=df, y='订单金额', ax=ax2, color='#e74c3c', width=0.5)
    ax2.set_title('订单金额箱线图', fontsize=14, pad=15)
    ax2.set_ylabel('订单金额（元）', fontsize=11)
    
    plt.suptitle('订单金额分布分析', fontsize=16, y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, '2_amount_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存: {output_path}")


def plot_category_sales(df, output_dir='output'):
    """
    绘制类别销售额条形图
    """
    print("绘制商品类别销售额图...")
    os.makedirs(output_dir, exist_ok=True)
    setup_chinese_font()
    
    category_sales = df.groupby('商品类别')['订单金额'].sum().sort_values(ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = sns.color_palette("viridis", len(category_sales))
    bars = ax.barh(category_sales.index, category_sales.values, color=colors)
    
    ax.set_title('各商品类别销售总额', fontsize=16, pad=20)
    ax.set_xlabel('销售额（元）', fontsize=12)
    
    for bar in bars:
        width = bar.get_width()
        ax.text(width + width * 0.01, bar.get_y() + bar.get_height()/2,
                f'{width:.0f} 元', va='center', fontsize=10)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, '3_category_sales.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存: {output_path}")


def plot_customer_clustering(rfm_df, output_dir='output'):
    """
    绘制用户价值聚类散点图
    """
    print("绘制用户聚类散点图...")
    os.makedirs(output_dir, exist_ok=True)
    setup_chinese_font()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    scatter = ax.scatter(rfm_df['Recency'], rfm_df['Monetary'], 
                         c=rfm_df['Cluster'], cmap='viridis', 
                         s=rfm_df['Frequency'] * 20 + 30, alpha=0.7, edgecolors='w')
    
    ax.set_title('用户价值聚类分析 (R vs M)', fontsize=16, pad=20)
    ax.set_xlabel('最近购买天数 (Recency)', fontsize=12)
    ax.set_ylabel('总消费金额 (Monetary)', fontsize=12)
    
    legend1 = ax.legend(*scatter.legend_elements(),
                        loc="upper right", title="聚类")
    ax.add_artist(legend1)
    
    for cluster in rfm_df['Cluster'].unique():
        cluster_data = rfm_df[rfm_df['Cluster'] == cluster]
        center_x = cluster_data['Recency'].mean()
        center_y = cluster_data['Monetary'].mean()
        ax.annotate(cluster_data['客户类型'].iloc[0], 
                    (center_x, center_y),
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, '4_customer_clustering.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存: {output_path}")


def plot_correlation_heatmap(df, output_dir='output'):
    """
    绘制相关性热力图
    """
    print("绘制相关性热力图...")
    os.makedirs(output_dir, exist_ok=True)
    setup_chinese_font()
    
    numeric_cols = ['订单金额', '商品数量', '用户评分', '月份', '星期', '小时']
    corr_matrix = df[numeric_cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
                center=0, square=True, linewidths=0.5, 
                fmt='.2f', ax=ax, cbar_kws={"shrink": 0.8})
    
    ax.set_title('特征相关性热力图', fontsize=16, pad=20)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, '5_correlation_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存: {output_path}")


def plot_weekday_hour_heatmap(df, output_dir='output'):
    """
    绘制星期-小时销售热力图
    """
    print("绘制星期-小时销售热力图...")
    os.makedirs(output_dir, exist_ok=True)
    setup_chinese_font()
    
    pivot_data = df.pivot_table(index='小时', columns='星期', 
                                values='订单金额', aggfunc='sum')
    
    weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
    pivot_data.columns = weekday_names[:len(pivot_data.columns)]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot_data, annot=True, cmap='YlOrRd', 
                fmt='.0f', ax=ax, cbar_kws={"label": "销售额（元）"})
    
    ax.set_title('星期-小时销售额分布热力图', fontsize=16, pad=20)
    ax.set_xlabel('星期', fontsize=12)
    ax.set_ylabel('小时', fontsize=12)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, '6_weekday_hour_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存: {output_path}")


def generate_all_charts(df, rfm_df, daily_sales, output_dir='output'):
    """
    生成所有可视化图表
    """
    print("\n" + "=" * 50)
    print("开始生成可视化图表")
    print("=" * 50)
    
    plot_sales_trend(daily_sales, output_dir)
    plot_amount_distribution(df, output_dir)
    plot_category_sales(df, output_dir)
    plot_customer_clustering(rfm_df, output_dir)
    plot_correlation_heatmap(df, output_dir)
    plot_weekday_hour_heatmap(df, output_dir)
    
    print("\n所有图表生成完成！")
    print(f"共生成 6 张图表，保存在 {output_dir} 目录")


if __name__ == '__main__':
    print("可视化模块测试")
