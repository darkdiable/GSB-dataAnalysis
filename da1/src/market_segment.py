import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly import graph_objects as go
from .config import OUTPUT_DIR, DPI, FIGSIZE, HP_ORDER

def plot_hp_range_pie(df, interactive=False):
    hp_sales = df.groupby('hp_range')['sales'].sum().reindex(HP_ORDER).reset_index()
    
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    
    if interactive:
        fig = px.pie(hp_sales, values='sales', names='hp_range',
                     title='马力区间市场份额分布',
                     color_discrete_sequence=colors)
        fig.write_html(f'{OUTPUT_DIR}/segment_hp_pie_interactive.html')
    else:
        plt.figure(figsize=(10, 10))
        wedges, texts, autotexts = plt.pie(
            hp_sales['sales'], labels=hp_sales['hp_range'],
            autopct='%1.1f%%', colors=colors, startangle=90
        )
        plt.setp(autotexts, size=10, weight='bold')
        plt.title('马力区间市场份额分布', fontsize=14, pad=20)
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/segment_hp_pie.png', dpi=DPI)
        plt.close()
    
    return hp_sales

def plot_engine_distribution(df, interactive=False):
    engine_sales = df.groupby('engine_type')['sales'].sum().reset_index()
    
    if interactive:
        fig = px.bar(engine_sales, x='engine_type', y='sales',
                     title='引擎类型销量分布', color='engine_type',
                     labels={'sales': '总销量', 'engine_type': '引擎类型'})
        fig.write_html(f'{OUTPUT_DIR}/segment_engine_interactive.html')
    else:
        plt.figure(figsize=FIGSIZE)
        bars = plt.bar(engine_sales['engine_type'], engine_sales['sales'],
                      color=['#1f77b4', '#ff7f0e'])
        plt.title('引擎类型销量分布', fontsize=14, pad=20)
        plt.ylabel('总销量', fontsize=12)
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}', ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/segment_engine.png', dpi=DPI)
        plt.close()
    
    return engine_sales

def plot_brand_sankey(df):
    brand_engine = df.groupby(['brand', 'engine_type'])['sales'].sum().reset_index()
    engine_hp = df.groupby(['engine_type', 'hp_range'])['sales'].sum().reset_index()
    
    labels = []
    brands = df['brand'].unique().tolist()
    engines = df['engine_type'].unique().tolist()
    hps = HP_ORDER
    labels = brands + engines + hps
    
    label_to_idx = {label: i for i, label in enumerate(labels)}
    
    source = []
    target = []
    value = []
    
    for _, row in brand_engine.iterrows():
        source.append(label_to_idx[row['brand']])
        target.append(label_to_idx[row['engine_type']])
        value.append(row['sales'])
    
    for _, row in engine_hp.iterrows():
        source.append(label_to_idx[row['engine_type']])
        target.append(label_to_idx[row['hp_range']])
        value.append(row['sales'])
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(pad=15, thickness=20, line=dict(color='black', width=0.5),
                 label=labels),
        link=dict(source=source, target=target, value=value)
    )])
    fig.update_layout(title_text='品牌-引擎-马力桑基图', font_size=10)
    fig.write_html(f'{OUTPUT_DIR}/segment_sankey_interactive.html')

def plot_bcg_matrix(df):
    brand_metrics = df.groupby('brand').agg({
        'sales': ['sum', 'mean'],
        'production': 'sum'
    }).reset_index()
    brand_metrics.columns = ['brand', 'total_sales', 'avg_sales', 'total_prod']
    
    brand_years = df.groupby(['brand', 'year'])['sales'].sum().unstack()
    growth_rate = ((brand_years[2025] - brand_years[2023]) / brand_years[2023]).fillna(0)
    
    bcg_data = pd.DataFrame({
        'brand': growth_rate.index,
        'growth': growth_rate.values,
        'market_share': brand_metrics.set_index('brand')['total_sales'] / brand_metrics['total_sales'].sum()
    })
    
    mid_growth = bcg_data['growth'].median()
    mid_share = bcg_data['market_share'].median()
    
    plt.figure(figsize=FIGSIZE)
    plt.scatter(bcg_data['market_share'], bcg_data['growth'], s=500, alpha=0.6)
    
    for _, row in bcg_data.iterrows():
        plt.annotate(row['brand'], (row['market_share'], row['growth']),
                    ha='center', va='center')
    
    plt.axvline(x=mid_share, color='gray', linestyle='--')
    plt.axhline(y=mid_growth, color='gray', linestyle='--')
    
    plt.text(0.02, 0.95, '明星业务', transform=plt.gca().transAxes, fontsize=12)
    plt.text(0.02, 0.05, '问题业务', transform=plt.gca().transAxes, fontsize=12)
    plt.text(0.80, 0.95, '金牛业务', transform=plt.gca().transAxes, fontsize=12)
    plt.text(0.80, 0.05, '瘦狗业务', transform=plt.gca().transAxes, fontsize=12)
    
    plt.title('品牌波士顿矩阵分析', fontsize=14, pad=20)
    plt.xlabel('相对市场份额', fontsize=12)
    plt.ylabel('增长率 (2023-2025)', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/segment_bcg_matrix.png', dpi=DPI)
    plt.close()
    
    return bcg_data

def run_market_segment_analysis(df):
    print("=== 开始市场细分分析 ===")
    
    hp_sales = plot_hp_range_pie(df)
    plot_hp_range_pie(df, interactive=True)
    print("马力区间饼图已生成")
    print("马力区间分布:")
    print(hp_sales)
    
    engine_sales = plot_engine_distribution(df)
    plot_engine_distribution(df, interactive=True)
    print("引擎类型分布图已生成")
    print("引擎类型分布:")
    print(engine_sales)
    
    plot_brand_sankey(df)
    print("品牌桑基图已生成")
    
    bcg_data = plot_bcg_matrix(df)
    print("波士顿矩阵分析已完成")
    
    print("=== 市场细分分析完成 ===")
    return {
        'hp_distribution': hp_sales,
        'engine_distribution': engine_sales,
        'bcg_matrix': bcg_data
    }
