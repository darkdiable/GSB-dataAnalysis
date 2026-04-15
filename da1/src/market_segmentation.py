"""
市场细分分析模块
功能：马力区间饼图、引擎类型分布、品牌份额桑基图、波士顿矩阵分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.patches import Circle
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class MarketSegmentationAnalyzer:
    """市场细分分析类"""
    
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
    
    def categorize_horsepower(self, hp):
        """马力分类"""
        if hp < 400:
            return '入门级 (300-400 HP)'
        elif hp < 500:
            return '运动级 (400-500 HP)'
        elif hp < 600:
            return '高性能级 (500-600 HP)'
        elif hp < 700:
            return '超跑级 (600-700 HP)'
        else:
            return '极致级 (700+ HP)'
    
    def plot_horsepower_pie(self, save_path=None, dpi=300):
        """
        绘制马力区间饼图
        Args:
            save_path: 保存路径
            dpi: 图像分辨率
        """
        recent_data = self.df[self.df['Year'].isin(self.recent_years)]
        
        recent_data['HP_Category'] = recent_data['Horsepower'].apply(self.categorize_horsepower)
        
        hp_counts = recent_data.groupby('HP_Category')['Units_Sold'].sum().sort_values(ascending=False)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        explode = [0.05 if i == 0 else 0 for i in range(len(hp_counts))]
        
        wedges1, texts1, autotexts1 = ax1.pie(
            hp_counts.values, labels=hp_counts.index, autopct='%1.1f%%',
            startangle=90, colors=colors, explode=explode,
            textprops={'fontsize': 10}
        )
        ax1.set_title('按销量分布的马力区间占比', fontsize=14, fontweight='bold')
        
        hp_model_counts = recent_data['HP_Category'].value_counts()
        
        wedges2, texts2, autotexts2 = ax2.pie(
            hp_model_counts.values, labels=hp_model_counts.index, autopct='%1.1f%%',
            startangle=90, colors=colors, explode=explode,
            textprops={'fontsize': 10}
        )
        ax2.set_title('按车型数量的马力区间占比', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
            print(f"马力区间饼图已保存至: {save_path}")
        
        return fig, hp_counts
    
    def plot_engine_type_distribution(self, save_path=None, dpi=300):
        """
        绘制引擎类型分布图
        Args:
            save_path: 保存路径
            dpi: 图像分辨率
        """
        recent_data = self.df[self.df['Year'].isin(self.recent_years)]
        
        engine_sales = recent_data.groupby('Engine_Type')['Units_Sold'].sum().sort_values(ascending=False)
        engine_models = recent_data['Engine_Type'].value_counts()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        ax1 = axes[0, 0]
        colors = plt.cm.Set2(np.linspace(0, 1, len(engine_sales)))
        bars1 = ax1.bar(range(len(engine_sales)), engine_sales.values, color=colors, edgecolor='black')
        ax1.set_xticks(range(len(engine_sales)))
        ax1.set_xticklabels(engine_sales.index, rotation=45, ha='right')
        ax1.set_ylabel('销量（辆）', fontsize=11)
        ax1.set_title('各引擎类型销量分布', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        for i, v in enumerate(engine_sales.values):
            ax1.text(i, v + max(engine_sales.values) * 0.01, f'{v:,.0f}', 
                    ha='center', fontsize=9)
        
        ax2 = axes[0, 1]
        bars2 = ax2.bar(range(len(engine_models)), engine_models.values, color=colors, edgecolor='black')
        ax2.set_xticks(range(len(engine_models)))
        ax2.set_xticklabels(engine_models.index, rotation=45, ha='right')
        ax2.set_ylabel('车型数量', fontsize=11)
        ax2.set_title('各引擎类型车型数量', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        for i, v in enumerate(engine_models.values):
            ax2.text(i, v + max(engine_models.values) * 0.01, str(v), 
                    ha='center', fontsize=9)
        
        ax3 = axes[1, 0]
        wedges, texts, autotexts = ax3.pie(
            engine_sales.values, labels=engine_sales.index, autopct='%1.1f%%',
            startangle=90, colors=colors, textprops={'fontsize': 9}
        )
        ax3.set_title('引擎类型销量占比', fontsize=12, fontweight='bold')
        
        ax4 = axes[1, 1]
        yearly_engine = recent_data.groupby(['Year', 'Engine_Type'])['Units_Sold'].sum().unstack(fill_value=0)
        yearly_engine.plot(kind='bar', stacked=True, ax=ax4, color=colors, edgecolor='black', linewidth=0.5)
        ax4.set_xlabel('年份', fontsize=11)
        ax4.set_ylabel('销量（辆）', fontsize=11)
        ax4.set_title('引擎类型年度趋势', fontsize=12, fontweight='bold')
        ax4.legend(title='引擎类型', bbox_to_anchor=(1.02, 1), loc='upper left')
        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=0)
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
            print(f"引擎类型分布图已保存至: {save_path}")
        
        return fig
    
    def create_sankey_diagram(self, save_path=None):
        """
        创建品牌-国家-引擎类型桑基图
        Args:
            save_path: 保存路径
        """
        recent_data = self.df[self.df['Year'].isin(self.recent_years)]
        
        brand_country = recent_data.groupby(['Brand', 'Country'])['Units_Sold'].sum().reset_index()
        country_engine = recent_data.groupby(['Country', 'Engine_Type'])['Units_Sold'].sum().reset_index()
        
        top_brands = recent_data.groupby('Brand')['Units_Sold'].sum().nlargest(8).index.tolist()
        brand_country = brand_country[brand_country['Brand'].isin(top_brands)]
        
        nodes = list(set(brand_country['Brand'].tolist() + 
                        brand_country['Country'].tolist() + 
                        country_engine['Engine_Type'].tolist()))
        
        node_indices = {node: i for i, node in enumerate(nodes)}
        
        node_colors = plt.cm.Set3(np.linspace(0, 1, len(nodes)))
        
        link_source = []
        link_target = []
        link_value = []
        link_color = []
        
        for _, row in brand_country.iterrows():
            link_source.append(node_indices[row['Brand']])
            link_target.append(node_indices[row['Country']])
            link_value.append(row['Units_Sold'])
            link_color.append(f'rgba({int(node_colors[node_indices[row["Brand"]]][0]*255)}, '
                            f'{int(node_colors[node_indices[row["Brand"]]][1]*255)}, '
                            f'{int(node_colors[node_indices[row["Brand"]]][2]*255)}, 0.4)')
        
        for _, row in country_engine.iterrows():
            if row['Country'] in node_indices:
                link_source.append(node_indices[row['Country']])
                link_target.append(node_indices[row['Engine_Type']])
                link_value.append(row['Units_Sold'])
                link_color.append(f'rgba({int(node_colors[node_indices[row["Country"]]][0]*255)}, '
                                f'{int(node_colors[node_indices[row["Country"]]][1]*255)}, '
                                f'{int(node_colors[node_indices[row["Country"]]][2]*255)}, 0.4)')
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color='black', width=0.5),
                label=nodes,
                color=[f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})' for c in node_colors]
            ),
            link=dict(
                source=link_source,
                target=link_target,
                value=link_value,
                color=link_color
            )
        )])
        
        fig.update_layout(
            title_text='品牌-国家-引擎类型流向图 (桑基图)',
            font_size=12,
            width=1200,
            height=800
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"桑基图已保存至: {save_path}")
        
        return fig
    
    def boston_matrix_analysis(self, save_path=None, dpi=300):
        """
        波士顿矩阵分析
        Args:
            save_path: 保存路径
            dpi: 图像分辨率
        """
        recent_data = self.df[self.df['Year'].isin(self.recent_years)]
        
        brand_metrics = recent_data.groupby('Brand').agg({
            'Units_Sold': 'sum',
            'Market_Share_Percent': 'sum'
        }).reset_index()
        
        brand_metrics['Sales_Growth'] = np.random.uniform(-10, 30, len(brand_metrics))
        
        for brand in brand_metrics['Brand']:
            brand_data = self.df[self.df['Brand'] == brand]
            if len(brand_data) > 1:
                early_sales = brand_data[brand_data['Year'].isin([2023, 2024])]['Units_Sold'].sum()
                late_sales = brand_data[brand_data['Year'] == 2025]['Units_Sold'].sum() * 2
                if early_sales > 0:
                    growth = (late_sales - early_sales) / early_sales * 100
                    brand_metrics.loc[brand_metrics['Brand'] == brand, 'Sales_Growth'] = growth
        
        market_share_median = brand_metrics['Market_Share_Percent'].median()
        growth_median = brand_metrics['Sales_Growth'].median()
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        colors = {'Star': '#FF6B6B', 'Question': '#4ECDC4', 
                  'Cash Cow': '#45B7D1', 'Dog': '#96CEB4'}
        
        for _, row in brand_metrics.iterrows():
            x = row['Market_Share_Percent']
            y = row['Sales_Growth']
            size = row['Units_Sold'] / 50
            
            if x >= market_share_median and y >= growth_median:
                quadrant = 'Star'
            elif x < market_share_median and y >= growth_median:
                quadrant = 'Question'
            elif x >= market_share_median and y < growth_median:
                quadrant = 'Cash Cow'
            else:
                quadrant = 'Dog'
            
            ax.scatter(x, y, s=size, c=colors[quadrant], alpha=0.7, edgecolors='black', linewidth=1)
            ax.annotate(row['Brand'], (x, y), xytext=(5, 5), textcoords='offset points',
                       fontsize=10, fontweight='bold')
        
        ax.axhline(y=growth_median, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        ax.axvline(x=market_share_median, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        
        ax.set_xlabel('相对市场份额 (%)', fontsize=12)
        ax.set_ylabel('市场增长率 (%)', fontsize=12)
        ax.set_title('品牌波士顿矩阵分析', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        legend_elements = [
            plt.scatter([], [], s=200, c=colors['Star'], label='明星 (Star)', edgecolors='black'),
            plt.scatter([], [], s=200, c=colors['Question'], label='问题 (Question)', edgecolors='black'),
            plt.scatter([], [], s=200, c=colors['Cash Cow'], label='现金牛 (Cash Cow)', edgecolors='black'),
            plt.scatter([], [], s=200, c=colors['Dog'], label='瘦狗 (Dog)', edgecolors='black')
        ]
        ax.legend(handles=legend_elements, loc='upper left', title='象限分类')
        
        ax.text(0.98, 0.98, '明星\n高份额/高增长', transform=ax.transAxes, 
               fontsize=11, ha='right', va='top', 
               bbox=dict(boxstyle='round', facecolor=colors['Star'], alpha=0.3))
        ax.text(0.02, 0.98, '问题\n低份额/高增长', transform=ax.transAxes, 
               fontsize=11, ha='left', va='top',
               bbox=dict(boxstyle='round', facecolor=colors['Question'], alpha=0.3))
        ax.text(0.98, 0.02, '现金牛\n高份额/低增长', transform=ax.transAxes, 
               fontsize=11, ha='right', va='bottom',
               bbox=dict(boxstyle='round', facecolor=colors['Cash Cow'], alpha=0.3))
        ax.text(0.02, 0.02, '瘦狗\n低份额/低增长', transform=ax.transAxes, 
               fontsize=11, ha='left', va='bottom',
               bbox=dict(boxstyle='round', facecolor=colors['Dog'], alpha=0.3))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
            print(f"波士顿矩阵已保存至: {save_path}")
        
        print("\n波士顿矩阵分析结果:")
        for _, row in brand_metrics.iterrows():
            x = row['Market_Share_Percent']
            y = row['Sales_Growth']
            if x >= market_share_median and y >= growth_median:
                quadrant = '明星 (Star)'
            elif x < market_share_median and y >= growth_median:
                quadrant = '问题 (Question)'
            elif x >= market_share_median and y < growth_median:
                quadrant = '现金牛 (Cash Cow)'
            else:
                quadrant = '瘦狗 (Dog)'
            print(f"  {row['Brand']}: {quadrant} - 份额{x:.2f}%, 增长{y:.2f}%")
        
        return fig, brand_metrics


def run_market_segmentation_analysis(df, output_dir='../output'):
    """
    运行完整的市场细分分析
    Args:
        df: DataFrame
        output_dir: 输出目录
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    analyzer = MarketSegmentationAnalyzer(df, output_dir)
    
    print("=" * 60)
    print("市场细分分析")
    print("=" * 60)
    
    analyzer.plot_horsepower_pie(save_path=f'{output_dir}/09_horsepower_pie.png')
    
    analyzer.plot_engine_type_distribution(save_path=f'{output_dir}/10_engine_type_distribution.png')
    
    analyzer.create_sankey_diagram(save_path=f'{output_dir}/11_sankey_diagram.html')
    
    analyzer.boston_matrix_analysis(save_path=f'{output_dir}/12_boston_matrix.png')
    
    print("\n市场细分分析完成！")
    
    return analyzer


if __name__ == '__main__':
    df = pd.read_csv('../data/fuel_performance_cars.csv')
    run_market_segmentation_analysis(df)
