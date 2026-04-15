import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

COUNTRY_COORDS = {
    'USA': {'lat': 39.8283, 'lon': -98.5795, 'name': 'United States'},
    'China': {'lat': 35.8617, 'lon': 104.1954, 'name': 'China'},
    'Japan': {'lat': 36.2048, 'lon': 138.2529, 'name': 'Japan'},
    'Germany': {'lat': 51.1657, 'lon': 10.4515, 'name': 'Germany'},
    'UK': {'lat': 55.3781, 'lon': -3.4360, 'name': 'United Kingdom'},
    'France': {'lat': 46.2276, 'lon': 2.2137, 'name': 'France'},
    'Italy': {'lat': 41.8719, 'lon': 12.5674, 'name': 'Italy'},
    'South Korea': {'lat': 35.9078, 'lon': 127.7669, 'name': 'South Korea'},
    'Canada': {'lat': 56.1304, 'lon': -106.3468, 'name': 'Canada'},
    'Australia': {'lat': -25.2744, 'lon': 133.7751, 'name': 'Australia'},
    'Brazil': {'lat': -14.2350, 'lon': -51.9253, 'name': 'Brazil'},
    'India': {'lat': 20.5937, 'lon': 78.9629, 'name': 'India'},
    'Mexico': {'lat': 23.6345, 'lon': -102.5528, 'name': 'Mexico'},
    'Spain': {'lat': 40.4637, 'lon': -3.7492, 'name': 'Spain'},
    'Russia': {'lat': 61.5240, 'lon': 105.3188, 'name': 'Russia'},
    'Netherlands': {'lat': 52.1326, 'lon': 5.2913, 'name': 'Netherlands'},
    'Sweden': {'lat': 60.1282, 'lon': 18.6435, 'name': 'Sweden'},
    'Belgium': {'lat': 50.5039, 'lon': 4.4699, 'name': 'Belgium'},
    'Switzerland': {'lat': 46.8182, 'lon': 8.2275, 'name': 'Switzerland'},
    'Austria': {'lat': 47.5162, 'lon': 14.5501, 'name': 'Austria'}
}

class GeographicAnalyzer:
    def __init__(self, df):
        self.df = df.copy()
        self.country_sales = None
        
    def prepare_country_data(self, years=None):
        if years is None:
            years = [2023, 2024, 2025]
            
        filtered_df = self.df[self.df['Year'].isin(years)]
        self.country_sales = filtered_df.groupby('Country').agg({
            'Sales_Units': 'sum',
            'Production_Units': 'sum',
            'Price_USD': 'mean',
            'Horsepower': 'mean'
        }).reset_index()
        
        return self
    
    def calculate_market_concentration(self, top_n=[4, 8]):
        if self.country_sales is None:
            self.prepare_country_data()
            
        total_sales = self.country_sales['Sales_Units'].sum()
        sorted_sales = self.country_sales.sort_values('Sales_Units', ascending=False)
        
        concentration = {}
        for n in top_n:
            top_sales = sorted_sales.head(n)['Sales_Units'].sum()
            concentration[f'CR{n}'] = (top_sales / total_sales) * 100
            
        return concentration
    
    def plot_top10_countries(self, save_path):
        if self.country_sales is None:
            self.prepare_country_data()
            
        top10 = self.country_sales.nlargest(10, 'Sales_Units').sort_values('Sales_Units')
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(top10)))
        bars = ax.barh(top10['Country'], top10['Sales_Units'], color=colors)
        
        for bar, val in zip(bars, top10['Sales_Units']):
            ax.text(bar.get_width() + bar.get_width()*0.01, bar.get_y() + bar.get_height()/2,
                   f'{val:,.0f}', va='center', fontsize=10)
        
        ax.set_xlabel('销量（辆）', fontsize=12)
        ax.set_ylabel('国家', fontsize=12)
        ax.set_title('2023-2025年各国销量Top10', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Top10国家柱状图已保存: {save_path}")
        
    def plot_world_heatmap(self, save_path):
        if self.country_sales is None:
            self.prepare_country_data()
            
        fig = go.Figure()
        
        country_data = []
        for _, row in self.country_sales.iterrows():
            country = row['Country']
            if country in COUNTRY_COORDS:
                country_data.append({
                    'Country': country,
                    'Sales': row['Sales_Units'],
                    'lat': COUNTRY_COORDS[country]['lat'],
                    'lon': COUNTRY_COORDS[country]['lon'],
                    'name': COUNTRY_COORDS[country]['name']
                })
        
        df_map = pd.DataFrame(country_data)
        
        fig.add_trace(go.Scattergeo(
            lon=df_map['lon'],
            lat=df_map['lat'],
            text=df_map.apply(lambda x: f"{x['name']}<br>销量: {x['Sales']:,.0f}", axis=1),
            mode='markers',
            marker=dict(
                size=np.log10(df_map['Sales'] + 1) * 10,
                color=df_map['Sales'],
                colorscale='Viridis',
                colorbar=dict(title='销量'),
                line=dict(width=0.5, color='white'),
                sizemode='diameter'
            ),
            name='销量分布'
        ))
        
        fig.update_layout(
            title='全球燃油性能车销量热力分布',
            geo=dict(
                projection_type='natural earth',
                showland=True,
                landcolor='lightgray',
                showocean=True,
                oceancolor='lightblue'
            ),
            height=600,
            width=1000
        )
        
        fig.write_html(save_path)
        print(f"世界地图热力图已保存: {save_path}")
        
    def plot_market_concentration(self, save_path):
        if self.country_sales is None:
            self.prepare_country_data()
            
        concentration = self.calculate_market_concentration()
        sorted_sales = self.country_sales.sort_values('Sales_Units', ascending=False)
        total_sales = sorted_sales['Sales_Units'].sum()
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        ax1 = axes[0]
        cr_labels = list(concentration.keys())
        cr_values = list(concentration.values())
        bars = ax1.bar(cr_labels, cr_values, color=['steelblue', 'coral'])
        ax1.set_ylabel('市场份额 (%)')
        ax1.set_title('市场集中度指标')
        ax1.set_ylim(0, 100)
        for bar, val in zip(bars, cr_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{val:.1f}%', ha='center', fontsize=12)
        
        ax2 = axes[1]
        sorted_sales['Cumulative_Share'] = sorted_sales['Sales_Units'].cumsum() / total_sales * 100
        countries = sorted_sales['Country'].values
        cumulative = sorted_sales['Cumulative_Share'].values
        
        ax2.bar(range(len(countries)), cumulative, color='teal', alpha=0.7)
        ax2.axhline(y=50, color='red', linestyle='--', label='50%线')
        ax2.axhline(y=80, color='orange', linestyle='--', label='80%线')
        ax2.set_xlabel('国家排名')
        ax2.set_ylabel('累计市场份额 (%)')
        ax2.set_title('累计市场份额分布')
        ax2.legend()
        ax2.set_xticks(range(len(countries)))
        ax2.set_xticklabels(countries, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"市场集中度图已保存: {save_path}")
        
    def plot_yearly_country_comparison(self, save_path):
        years = [2023, 2024, 2025]
        fig, ax = plt.subplots(figsize=(16, 10))
        
        yearly_country = self.df[self.df['Year'].isin(years)].groupby(
            ['Year', 'Country'])['Sales_Units'].sum().unstack(fill_value=0)
        
        top_countries = yearly_country.sum().nlargest(10).index
        yearly_country_top = yearly_country[top_countries]
        
        x = np.arange(len(top_countries))
        width = 0.25
        
        for i, year in enumerate(years):
            if year in yearly_country_top.index:
                values = yearly_country_top.loc[year].values
                ax.bar(x + i*width, values, width, label=str(year))
        
        ax.set_xlabel('国家')
        ax.set_ylabel('销量（辆）')
        ax.set_title('2023-2025年Top10国家销量对比')
        ax.set_xticks(x + width)
        ax.set_xticklabels(top_countries, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"年度国家对比图已保存: {save_path}")
        
    def generate_report(self):
        concentration = self.calculate_market_concentration()
        
        report = "# 地理分析报告\n\n"
        
        report += "## 1. 市场集中度分析\n\n"
        report += f"- **CR4**: {concentration['CR4']:.2f}%\n"
        report += f"- **CR8**: {concentration['CR8']:.2f}%\n\n"
        
        if concentration['CR4'] > 50:
            report += "市场集中度较高，前4大市场占据主导地位。\n\n"
        else:
            report += "市场分布相对分散，竞争较为均衡。\n\n"
        
        report += "## 2. 区域市场特点\n\n"
        top5 = self.country_sales.nlargest(5, 'Sales_Units')
        for _, row in top5.iterrows():
            report += f"- **{row['Country']}**: 销量 {row['Sales_Units']:,.0f} 辆\n"
        
        return report


if __name__ == "__main__":
    df = pd.read_csv("../data/fuel_performance_cars.csv")
    analyzer = GeographicAnalyzer(df)
    analyzer.prepare_country_data()
    print(analyzer.calculate_market_concentration())
    analyzer.plot_top10_countries("../output/top10_countries.png")
    analyzer.plot_world_heatmap("../output/world_heatmap.html")
    analyzer.plot_market_concentration("../output/market_concentration.png")
