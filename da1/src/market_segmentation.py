import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class MarketSegmentation:
    def __init__(self, df):
        self.df = df.copy()
        self.hp_distribution = None
        self.engine_distribution = None
        self.brand_share = None
        
    def analyze_horsepower_segments(self):
        self.hp_distribution = self.df.groupby('HP_Category').agg({
            'Sales_Units': 'sum',
            'Price_USD': 'mean',
            'Horsepower': 'mean'
        }).reset_index()
        return self
    
    def analyze_engine_types(self):
        self.engine_distribution = self.df.groupby('Engine_Type').agg({
            'Sales_Units': 'sum',
            'Price_USD': 'mean'
        }).reset_index()
        return self
    
    def analyze_brand_share(self):
        self.brand_share = self.df.groupby('Brand').agg({
            'Sales_Units': 'sum',
            'Market_Share_Percent': 'mean',
            'Growth_Rate_Percent': 'mean'
        }).reset_index()
        return self
    
    def plot_horsepower_pie(self, save_path):
        if self.hp_distribution is None:
            self.analyze_horsepower_segments()
            
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        ax1 = axes[0]
        sizes = self.hp_distribution['Sales_Units']
        labels = self.hp_distribution['HP_Category']
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        
        wedges, texts, autotexts = ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
                                           colors=colors, startangle=90,
                                           explode=[0.02]*len(labels))
        ax1.set_title('马力区间销量分布', fontsize=14, fontweight='bold')
        
        ax2 = axes[1]
        categories = self.hp_distribution['HP_Category']
        avg_prices = self.hp_distribution['Price_USD']
        
        bars = ax2.bar(categories, avg_prices, color=colors)
        ax2.set_xlabel('马力区间')
        ax2.set_ylabel('平均价格 (USD)')
        ax2.set_title('各马力区间平均价格', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=15)
        
        for bar, price in zip(bars, avg_prices):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'${price:,.0f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"马力区间饼图已保存: {save_path}")
        
    def plot_engine_distribution(self, save_path):
        if self.engine_distribution is None:
            self.analyze_engine_types()
            
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        ax1 = axes[0]
        engine_sorted = self.engine_distribution.sort_values('Sales_Units', ascending=True)
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(engine_sorted)))
        
        bars = ax1.barh(engine_sorted['Engine_Type'], engine_sorted['Sales_Units'], color=colors)
        ax1.set_xlabel('销量（辆）')
        ax1.set_ylabel('引擎类型')
        ax1.set_title('引擎类型销量分布', fontsize=14, fontweight='bold')
        
        for bar, val in zip(bars, engine_sorted['Sales_Units']):
            ax1.text(bar.get_width() + bar.get_width()*0.01, bar.get_y() + bar.get_height()/2,
                    f'{val:,.0f}', va='center')
        
        ax2 = axes[1]
        sizes = engine_sorted['Sales_Units']
        labels = engine_sorted['Engine_Type']
        
        wedges, texts, autotexts = ax2.pie(sizes, labels=labels, autopct='%1.1f%%',
                                           colors=colors, startangle=90)
        ax2.set_title('引擎类型市场份额', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"引擎类型分布图已保存: {save_path}")
        
    def plot_brand_sankey(self, save_path):
        if self.brand_share is None:
            self.analyze_brand_share()
            
        brand_country = self.df.groupby(['Brand', 'Country'])['Sales_Units'].sum().reset_index()
        top_brands = self.brand_share.nlargest(10, 'Sales_Units')['Brand']
        brand_country_filtered = brand_country[brand_country['Brand'].isin(top_brands)]
        
        top_countries = brand_country_filtered.groupby('Country')['Sales_Units'].sum().nlargest(10).index
        brand_country_filtered = brand_country_filtered[brand_country_filtered['Country'].isin(top_countries)]
        
        brands = list(brand_country_filtered['Brand'].unique())
        countries = list(brand_country_filtered['Country'].unique())
        
        all_nodes = brands + countries
        node_indices = {node: i for i, node in enumerate(all_nodes)}
        
        sources = []
        targets = []
        values = []
        
        for _, row in brand_country_filtered.iterrows():
            sources.append(node_indices[row['Brand']])
            targets.append(node_indices[row['Country']])
            values.append(row['Sales_Units'])
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color='black', width=0.5),
                label=all_nodes,
                color=['steelblue']*len(brands) + ['coral']*len(countries)
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color='rgba(150,150,150,0.4)'
            )
        )])
        
        fig.update_layout(
            title='品牌-国家销量流向桑基图',
            height=700,
            width=1200
        )
        
        fig.write_html(save_path)
        print(f"品牌桑基图已保存: {save_path}")
        
    def plot_boston_matrix(self, save_path):
        if self.brand_share is None:
            self.analyze_brand_share()
            
        brand_data = self.brand_share.copy()
        
        median_share = brand_data['Market_Share_Percent'].median()
        median_growth = brand_data['Growth_Rate_Percent'].median()
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        colors = []
        for _, row in brand_data.iterrows():
            if row['Market_Share_Percent'] >= median_share and row['Growth_Rate_Percent'] >= median_growth:
                colors.append('gold')
            elif row['Market_Share_Percent'] >= median_share and row['Growth_Rate_Percent'] < median_growth:
                colors.append('green')
            elif row['Market_Share_Percent'] < median_share and row['Growth_Rate_Percent'] >= median_growth:
                colors.append('skyblue')
            else:
                colors.append('lightcoral')
        
        scatter = ax.scatter(brand_data['Market_Share_Percent'], 
                           brand_data['Growth_Rate_Percent'],
                           s=brand_data['Sales_Units']/brand_data['Sales_Units'].max()*1000,
                           c=colors, alpha=0.6, edgecolors='black')
        
        for i, row in brand_data.iterrows():
            ax.annotate(row['Brand'], (row['Market_Share_Percent'], row['Growth_Rate_Percent']),
                       xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax.axhline(y=median_growth, color='gray', linestyle='--', linewidth=1)
        ax.axvline(x=median_share, color='gray', linestyle='--', linewidth=1)
        
        ax.set_xlabel('市场份额 (%)', fontsize=12)
        ax.set_ylabel('增长率 (%)', fontsize=12)
        ax.set_title('波士顿矩阵分析 - 品牌定位', fontsize=14, fontweight='bold')
        
        ax.text(0.75, 0.85, '明星', transform=ax.transAxes, fontsize=12, 
               fontweight='bold', color='goldenrod')
        ax.text(0.75, 0.15, '金牛', transform=ax.transAxes, fontsize=12,
               fontweight='bold', color='green')
        ax.text(0.15, 0.85, '问题', transform=ax.transAxes, fontsize=12,
               fontweight='bold', color='steelblue')
        ax.text(0.15, 0.15, '瘦狗', transform=ax.transAxes, fontsize=12,
               fontweight='bold', color='indianred')
        
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"波士顿矩阵已保存: {save_path}")
        
    def create_interactive_segmentation(self, save_path):
        if self.hp_distribution is None:
            self.analyze_horsepower_segments()
        if self.engine_distribution is None:
            self.analyze_engine_types()
            
        fig = make_subplots(rows=2, cols=2, 
                           specs=[[{'type':'domain'}, {'type':'domain'}],
                                  [{'type':'xy'}, {'type':'xy'}]],
                           subplot_titles=('马力区间分布', '引擎类型分布',
                                          '品牌销量Top10', '品牌增长率'))
        
        fig.add_trace(go.Pie(labels=self.hp_distribution['HP_Category'],
                            values=self.hp_distribution['Sales_Units'],
                            name='马力区间'),
                     row=1, col=1)
        
        fig.add_trace(go.Pie(labels=self.engine_distribution['Engine_Type'],
                            values=self.engine_distribution['Sales_Units'],
                            name='引擎类型'),
                     row=1, col=2)
        
        top_brands = self.brand_share.nlargest(10, 'Sales_Units')
        fig.add_trace(go.Bar(x=top_brands['Brand'], y=top_brands['Sales_Units'],
                            name='销量', marker_color='steelblue'),
                     row=2, col=1)
        
        fig.add_trace(go.Bar(x=top_brands['Brand'], y=top_brands['Growth_Rate_Percent'],
                            name='增长率', marker_color='coral'),
                     row=2, col=2)
        
        fig.update_layout(height=800, width=1200, 
                         title_text="市场细分分析",
                         showlegend=True)
        
        fig.write_html(save_path)
        print(f"交互式市场细分图已保存: {save_path}")
        
    def generate_report(self):
        if self.hp_distribution is None:
            self.analyze_horsepower_segments()
        if self.engine_distribution is None:
            self.analyze_engine_types()
        if self.brand_share is None:
            self.analyze_brand_share()
            
        report = "# 市场细分分析报告\n\n"
        
        report += "## 1. 马力区间分析\n\n"
        for _, row in self.hp_distribution.sort_values('Sales_Units', ascending=False).iterrows():
            report += f"- **{row['HP_Category']}**: 销量 {row['Sales_Units']:,.0f} 辆, 平均价格 ${row['Price_USD']:,.0f}\n"
        report += "\n"
        
        report += "## 2. 引擎类型分析\n\n"
        for _, row in self.engine_distribution.sort_values('Sales_Units', ascending=False).head(5).iterrows():
            report += f"- **{row['Engine_Type']}**: 销量 {row['Sales_Units']:,.0f} 辆\n"
        report += "\n"
        
        report += "## 3. 品牌分析\n\n"
        top5_brands = self.brand_share.nlargest(5, 'Sales_Units')
        for _, row in top5_brands.iterrows():
            report += f"- **{row['Brand']}**: 市场份额 {row['Market_Share_Percent']:.1f}%, 增长率 {row['Growth_Rate_Percent']:.1f}%\n"
        
        return report


if __name__ == "__main__":
    df = pd.read_csv("../data/fuel_performance_cars.csv")
    segmentation = MarketSegmentation(df)
    segmentation.analyze_horsepower_segments()
    segmentation.analyze_engine_types()
    segmentation.analyze_brand_share()
    segmentation.plot_horsepower_pie("../output/horsepower_pie.png")
    segmentation.plot_engine_distribution("../output/engine_distribution.png")
    segmentation.plot_brand_sankey("../output/brand_sankey.html")
    segmentation.plot_boston_matrix("../output/boston_matrix.png")
