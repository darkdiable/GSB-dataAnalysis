import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from .config import OUTPUT_DIR, DPI, FIGSIZE, COUNTRY_COORDS

def plot_top_countries_bar(df, interactive=False):
    recent = df[(df['year'] >= 2023) & (df['year'] <= 2025)]
    country_sales = recent.groupby('country')['sales'].sum().sort_values(ascending=True).reset_index()
    top10 = country_sales.tail(10)
    
    if interactive:
        fig = px.bar(top10, x='sales', y='country', orientation='h',
                     title='2023-2025各国销量Top10',
                     labels={'sales': '总销量', 'country': '国家'},
                     color='sales', color_continuous_scale='viridis')
        fig.write_html(f'{OUTPUT_DIR}/geo_top10_interactive.html')
    else:
        plt.figure(figsize=FIGSIZE)
        bars = plt.barh(top10['country'], top10['sales'], color=sns.color_palette('viridis', 10))
        plt.title('2023-2025各国燃油性能车销量Top10', fontsize=14, pad=20)
        plt.xlabel('总销量', fontsize=12)
        for bar in bars:
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2,
                    f'{int(width):,}', ha='left', va='center')
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/geo_top10.png', dpi=DPI)
        plt.close()
    
    return country_sales

def plot_world_heatmap(df):
    recent = df[(df['year'] >= 2023) & (df['year'] <= 2025)]
    country_sales = recent.groupby('country')['sales'].sum().reset_index()
    
    country_sales['iso'] = country_sales['country'].map(lambda x: COUNTRY_COORDS.get(x, {}).get('iso', ''))
    country_sales = country_sales[country_sales['iso'] != '']
    
    fig = px.choropleth(country_sales, locations='iso', locationmode='ISO-3',
                       color='sales', hover_name='country',
                       color_continuous_scale='RdYlGn_r',
                       title='2023-2025全球燃油性能车销量热力分布')
    fig.write_html(f'{OUTPUT_DIR}/geo_heatmap_interactive.html')
    
    return country_sales

def calculate_concentration_ratios(df):
    recent = df[(df['year'] >= 2023) & (df['year'] <= 2025)]
    country_sales = recent.groupby('country')['sales'].sum().sort_values(ascending=False)
    total = country_sales.sum()
    
    cr4 = country_sales.head(4).sum() / total
    cr8 = country_sales.head(8).sum() / total
    
    cr_data = pd.DataFrame({
        '国家': country_sales.index,
        '销量': country_sales.values,
        '市场份额': country_sales.values / total
    })
    
    plt.figure(figsize=FIGSIZE)
    plt.plot(range(1, len(cr_data) + 1), cr_data['市场份额'].cumsum(), 'o-', linewidth=2, color='#1f77b4')
    plt.axhline(y=cr4, color='r', linestyle='--', label=f'CR4 = {cr4:.2%}')
    plt.axhline(y=cr8, color='orange', linestyle='--', label=f'CR8 = {cr8:.2%}')
    plt.title('市场集中度曲线 (CR4/CR8)', fontsize=14, pad=20)
    plt.xlabel('国家数量 (按销量排序)', fontsize=12)
    plt.ylabel('累计市场份额', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/geo_concentration.png', dpi=DPI)
    plt.close()
    
    return {'CR4': cr4, 'CR8': cr8, 'country_shares': cr_data}

def run_geo_analysis(df):
    print("=== 开始地理分析 ===")
    
    country_sales = plot_top_countries_bar(df)
    plot_top_countries_bar(df, interactive=True)
    print("Top10国家销量图已生成")
    
    heatmap_data = plot_world_heatmap(df)
    print("世界地图热力图已生成")
    
    cr = calculate_concentration_ratios(df)
    print(f"市场集中度 CR4: {cr['CR4']:.2%}")
    print(f"市场集中度 CR8: {cr['CR8']:.2%}")
    
    print("=== 地理分析完成 ===")
    return {
        'country_sales': country_sales,
        'heatmap_data': heatmap_data,
        'concentration_ratios': cr
    }
