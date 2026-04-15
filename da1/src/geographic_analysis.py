"""
地理分析模块
功能：各国销量Top10横向柱状图、世界地图热力分布、市场集中度CR4/CR8计算
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class GeographicAnalyzer:
    """地理分析类"""
    
    def __init__(self, df, output_dir='../output'):
        """
        初始化
        Args:
            df: pandas DataFrame
            output_dir: 输出目录
        """
        self.df = df.copy()
        self.output_dir = output_dir
        self.recent_years = [2023, 2024, 2025]
        
        self.country_mapping = {
            'USA': 'United States',
            'Germany': 'Germany',
            'Japan': 'Japan',
            'Italy': 'Italy',
            'UK': 'United Kingdom',
            'China': 'China',
            'South Korea': 'South Korea'
        }
        
        self.country_code_mapping = {
            'USA': 'USA',
            'Germany': 'DEU',
            'Japan': 'JPN',
            'Italy': 'ITA',
            'UK': 'GBR',
            'China': 'CHN',
            'South Korea': 'KOR'
        }
    
    def get_recent_data(self):
        """获取2023-2025年数据"""
        return self.df[self.df['Year'].isin(self.recent_years)]
    
    def calculate_market_concentration(self):
        """
        计算市场集中度CR4和CR8
        Returns:
            集中度指标字典
        """
        recent_data = self.get_recent_data()
        
        country_sales = recent_data.groupby('Country')['Units_Sold'].sum().sort_values(ascending=False)
        total_sales = country_sales.sum()
        
        cr4 = country_sales.head(4).sum() / total_sales * 100
        cr8 = country_sales.head(8).sum() / total_sales * 100
        
        hhi = sum((sales / total_sales * 100) ** 2 for sales in country_sales)
        
        print("\n市场集中度分析 (2023-2025):")
        print(f"  CR4 (前4国集中度): {cr4:.2f}%")
        print(f"  CR8 (前8国集中度): {cr8:.2f}%")
        print(f"  HHI (赫芬达尔指数): {hhi:.2f}")
        
        if cr4 < 40:
            concentration_level = "低集中度 (竞争型)"
        elif cr4 < 60:
            concentration_level = "中集中度 (寡占型)"
        else:
            concentration_level = "高集中度 (高度寡占型)"
        
        print(f"  市场结构: {concentration_level}")
        
        return {
            'CR4': cr4,
            'CR8': cr8,
            'HHI': hhi,
            'concentration_level': concentration_level,
            'country_shares': (country_sales / total_sales * 100).to_dict()
        }
    
    def plot_top10_countries(self, save_path=None, dpi=300):
        """
        绘制2023-2025各国销量Top10横向柱状图
        Args:
            save_path: 保存路径
            dpi: 图像分辨率
        """
        recent_data = self.get_recent_data()
        
        country_sales = recent_data.groupby('Country')['Units_Sold'].sum().sort_values(ascending=True)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(country_sales)))
        
        bars = ax.barh(country_sales.index, country_sales.values, color=colors, edgecolor='black', linewidth=0.5)
        
        for i, (country, value) in enumerate(country_sales.items()):
            ax.text(value + max(country_sales.values) * 0.01, i, f'{value:,.0f}', 
                   va='center', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('销量（辆）', fontsize=12)
        ax.set_ylabel('国家/地区', fontsize=12)
        ax.set_title('2023-2025年各国燃油性能车销量排名', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
            print(f"Top10国家柱状图已保存至: {save_path}")
        
        return fig
    
    def create_world_heatmap(self, save_path=None):
        """
        创建世界地图热力分布（交互式）
        Args:
            save_path: 保存路径
        """
        recent_data = self.get_recent_data()
        
        country_sales = recent_data.groupby('Country')['Units_Sold'].sum().reset_index()
        country_sales['Country_Full'] = country_sales['Country'].map(self.country_mapping)
        country_sales['ISO_Code'] = country_sales['Country'].map(self.country_code_mapping)
        
        fig = px.choropleth(
            country_sales,
            locations='ISO_Code',
            color='Units_Sold',
            hover_name='Country_Full',
            color_continuous_scale='Viridis',
            range_color=(0, country_sales['Units_Sold'].max()),
            title='2023-2025年全球燃油性能车销量热力分布',
            labels={'Units_Sold': '销量（辆）'},
            projection='natural earth'
        )
        
        fig.update_layout(
            geo=dict(showframe=False, showcoastlines=True),
            width=1200,
            height=700
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"世界地图热力图已保存至: {save_path}")
        
        return fig
    
    def plot_yearly_country_comparison(self, save_path=None, dpi=300):
        """
        绘制各国年度销量对比图
        Args:
            save_path: 保存路径
            dpi: 图像分辨率
        """
        recent_data = self.get_recent_data()
        
        pivot_data = recent_data.groupby(['Year', 'Country'])['Units_Sold'].sum().unstack(fill_value=0)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        pivot_data.plot(kind='bar', ax=ax, width=0.8, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('年份', fontsize=12)
        ax.set_ylabel('销量（辆）', fontsize=12)
        ax.set_title('2023-2025年各国销量年度对比', fontsize=14, fontweight='bold')
        ax.legend(title='国家', bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
            print(f"年度对比图已保存至: {save_path}")
        
        return fig
    
    def plot_market_share_pie(self, save_path=None, dpi=300):
        """
        绘制市场份额饼图
        Args:
            save_path: 保存路径
            dpi: 图像分辨率
        """
        recent_data = self.get_recent_data()
        
        country_sales = recent_data.groupby('Country')['Units_Sold'].sum().sort_values(ascending=False)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(country_sales)))
        
        wedges, texts, autotexts = ax1.pie(country_sales.values, labels=country_sales.index, 
                                           autopct='%1.1f%%', startangle=90, colors=colors,
                                           explode=[0.05 if i == 0 else 0 for i in range(len(country_sales))],
                                           textprops={'fontsize': 10})
        ax1.set_title('各国市场份额分布', fontsize=14, fontweight='bold')
        
        top5 = country_sales.head(5)
        others = country_sales.iloc[5:].sum()
        pie_data = pd.concat([top5, pd.Series({'其他': others})])
        
        colors2 = plt.cm.Paired(np.linspace(0, 1, len(pie_data)))
        
        wedges2, texts2, autotexts2 = ax2.pie(pie_data.values, labels=pie_data.index,
                                              autopct='%1.1f%%', startangle=90, colors=colors2,
                                              explode=[0.05 if i == 0 else 0 for i in range(len(pie_data))],
                                              textprops={'fontsize': 10})
        ax2.set_title('Top5国家市场份额（其他合并）', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
            print(f"市场份额饼图已保存至: {save_path}")
        
        return fig
    
    def create_interactive_country_chart(self, save_path=None):
        """
        创建交互式国家销量图表
        Args:
            save_path: 保存路径
        """
        recent_data = self.get_recent_data()
        
        country_year = recent_data.groupby(['Country', 'Year'])['Units_Sold'].sum().reset_index()
        
        fig = px.bar(
            country_year,
            x='Country',
            y='Units_Sold',
            color='Year',
            barmode='group',
            title='2023-2025年各国销量交互式对比',
            labels={'Units_Sold': '销量（辆）', 'Country': '国家'},
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            width=1000,
            height=600,
            xaxis_tickangle=-45
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"交互式图表已保存至: {save_path}")
        
        return fig


def run_geographic_analysis(df, output_dir='../output'):
    """
    运行完整的地理分析
    Args:
        df: DataFrame
        output_dir: 输出目录
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    analyzer = GeographicAnalyzer(df, output_dir)
    
    print("=" * 60)
    print("地理分析")
    print("=" * 60)
    
    concentration = analyzer.calculate_market_concentration()
    
    analyzer.plot_top10_countries(save_path=f'{output_dir}/04_top10_countries.png')
    
    analyzer.create_world_heatmap(save_path=f'{output_dir}/05_world_heatmap.html')
    
    analyzer.plot_yearly_country_comparison(save_path=f'{output_dir}/06_yearly_country_comparison.png')
    
    analyzer.plot_market_share_pie(save_path=f'{output_dir}/07_market_share_pie.png')
    
    analyzer.create_interactive_country_chart(save_path=f'{output_dir}/08_interactive_country.html')
    
    print("\n地理分析完成！")
    
    return analyzer, concentration


if __name__ == '__main__':
    df = pd.read_csv('../data/fuel_performance_cars.csv')
    run_geographic_analysis(df)
