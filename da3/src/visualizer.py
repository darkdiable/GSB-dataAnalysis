import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

class Visualizer:
    def __init__(self, output_dir='../output'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.charts_generated = []
    
    def plot_sales_trend(self, df, save_name='sales_trend.png'):
        """销售额趋势折线图"""
        print("生成销售额趋势折线图...")
        
        daily_sales = df.groupby('日期')['订单金额'].sum().reset_index()
        daily_sales['日期'] = pd.to_datetime(daily_sales['日期'])
        daily_sales = daily_sales.sort_values('日期')
        daily_sales['7日移动平均'] = daily_sales['订单金额'].rolling(window=7, min_periods=1).mean()
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        ax.plot(daily_sales['日期'], daily_sales['订单金额'], 
                alpha=0.4, color='#1f77b4', label='日销售额')
        ax.plot(daily_sales['日期'], daily_sales['7日移动平均'], 
                color='#ff7f0e', linewidth=2.5, label='7日移动平均')
        
        ax.set_title('电商销售趋势分析', fontsize=16, pad=20)
        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel('销售额 (元)', fontsize=12)
        ax.legend(fontsize=10)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.charts_generated.append(save_name)
        print(f"已保存: {save_path}")
        
        return save_path
    
    def plot_amount_distribution(self, df, save_name='amount_distribution.png'):
        """金额分布直方图+箱线图"""
        print("生成金额分布直方图+箱线图...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        sns.histplot(data=df, x='订单金额', bins=30, kde=True, 
                     color='#2ca02c', ax=ax1, alpha=0.7)
        ax1.set_title('订单金额分布直方图', fontsize=14, pad=15)
        ax1.set_xlabel('订单金额 (元)', fontsize=11)
        ax1.set_ylabel('频次', fontsize=11)
        
        sns.boxplot(data=df, y='订单金额', color='#ff9896', ax=ax2)
        ax2.set_title('订单金额箱线图', fontsize=14, pad=15)
        ax2.set_ylabel('订单金额 (元)', fontsize=11)
        
        plt.suptitle('订单金额分布分析', fontsize=16, y=1.02)
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.charts_generated.append(save_name)
        print(f"已保存: {save_path}")
        
        return save_path
    
    def plot_category_sales(self, df, save_name='category_sales.png'):
        """类别销售额条形图"""
        print("生成类别销售额条形图...")
        
        category_sales = df.groupby('商品类别')['订单金额'].agg(
            ['sum', 'count']).reset_index()
        category_sales = category_sales.sort_values('sum', ascending=False)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        colors = sns.color_palette('viridis', len(category_sales))
        
        bars1 = ax1.barh(category_sales['商品类别'], category_sales['sum'], 
                         color=colors, alpha=0.8)
        ax1.set_title('各类别销售额对比', fontsize=14, pad=15)
        ax1.set_xlabel('销售额 (元)', fontsize=11)
        
        for i, bar in enumerate(bars1):
            width = bar.get_width()
            ax1.text(width + width * 0.01, bar.get_y() + bar.get_height()/2,
                     f'{int(width)}', va='center', fontsize=9)
        
        bars2 = ax2.barh(category_sales['商品类别'], category_sales['count'], 
                         color=colors, alpha=0.8)
        ax2.set_title('各类别订单量对比', fontsize=14, pad=15)
        ax2.set_xlabel('订单数量', fontsize=11)
        
        for i, bar in enumerate(bars2):
            width = bar.get_width()
            ax2.text(width + width * 0.01, bar.get_y() + bar.get_height()/2,
                     f'{int(width)}', va='center', fontsize=9)
        
        plt.suptitle('商品类别销售分析', fontsize=16, y=1.02)
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.charts_generated.append(save_name)
        print(f"已保存: {save_path}")
        
        return save_path
    
    def plot_customer_clusters(self, rfm_df, save_name='customer_clusters.png'):
        """用户价值聚类散点图"""
        print("生成用户价值聚类散点图...")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        cluster_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
        
        scatter1 = axes[0].scatter(rfm_df['Recency'], rfm_df['Monetary'],
                                   c=rfm_df['聚类'], cmap='viridis', 
                                   s=60, alpha=0.7, edgecolors='w')
        axes[0].set_title('Recency vs Monetary', fontsize=13, pad=12)
        axes[0].set_xlabel('最近购买天数 (R)', fontsize=10)
        axes[0].set_ylabel('消费总金额 (M)', fontsize=10)
        
        scatter2 = axes[1].scatter(rfm_df['Frequency'], rfm_df['Monetary'],
                                   c=rfm_df['聚类'], cmap='viridis',
                                   s=60, alpha=0.7, edgecolors='w')
        axes[1].set_title('Frequency vs Monetary', fontsize=13, pad=12)
        axes[1].set_xlabel('购买频次 (F)', fontsize=10)
        axes[1].set_ylabel('消费总金额 (M)', fontsize=10)
        
        scatter3 = axes[2].scatter(rfm_df['Recency'], rfm_df['Frequency'],
                                   c=rfm_df['聚类'], cmap='viridis',
                                   s=60, alpha=0.7, edgecolors='w')
        axes[2].set_title('Recency vs Frequency', fontsize=13, pad=12)
        axes[2].set_xlabel('最近购买天数 (R)', fontsize=10)
        axes[2].set_ylabel('购买频次 (F)', fontsize=10)
        
        plt.colorbar(scatter3, ax=axes[2], label='聚类编号')
        plt.suptitle('用户RFM价值聚类分析', fontsize=16, y=1.02)
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.charts_generated.append(save_name)
        print(f"已保存: {save_path}")
        
        return save_path
    
    def plot_correlation_heatmap(self, df, save_name='correlation_heatmap.png'):
        """相关性热力图"""
        print("生成相关性热力图...")
        
        numeric_cols = ['单价', '数量', '订单金额', '评分', '小时']
        corr_matrix = df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm',
                    center=0, square=True, linewidths=.5, fmt='.3f',
                    cbar_kws={"shrink": .8}, ax=ax)
        
        ax.set_title('特征相关性热力图', fontsize=16, pad=20)
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.charts_generated.append(save_name)
        print(f"已保存: {save_path}")
        
        return save_path
    
    def plot_rating_distribution(self, df, save_name='rating_distribution.png'):
        """评分分布图"""
        print("生成评分分布图...")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        rating_counts = df['评分'].value_counts().sort_index()
        colors = sns.color_palette('RdYlGn', 5)
        
        bars = ax.bar(rating_counts.index, rating_counts.values,
                      color=colors, alpha=0.8, edgecolor='black')
        
        ax.set_title('用户评分分布', fontsize=16, pad=15)
        ax.set_xlabel('评分', fontsize=12)
        ax.set_ylabel('评价数量', fontsize=12)
        ax.set_xticks(range(1, 6))
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height * 0.01,
                    f'{int(height)}', ha='center', va='bottom', fontsize=11)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.charts_generated.append(save_name)
        print(f"已保存: {save_path}")
        
        return save_path
    
    def generate_all_charts(self, df, rfm_df):
        """生成所有图表"""
        print("=" * 50)
        print("开始生成可视化图表...")
        print("=" * 50)
        
        self.plot_sales_trend(df)
        self.plot_amount_distribution(df)
        self.plot_category_sales(df)
        self.plot_customer_clusters(rfm_df)
        self.plot_correlation_heatmap(df)
        self.plot_rating_distribution(df)
        
        print("=" * 50)
        print(f"图表生成完成! 共生成 {len(self.charts_generated)} 张图表:")
        for chart in self.charts_generated:
            print(f"  - {chart}")
        print("=" * 50)
        
        return self.charts_generated

if __name__ == '__main__':
    from data_generator import generate_ecommerce_data
    from data_cleaner import DataCleaner
    from data_analyzer import DataAnalyzer
    
    df_raw = generate_ecommerce_data(n_records=2000)
    cleaner = DataCleaner(df_raw)
    df_cleaned = cleaner.clean()
    
    analyzer = DataAnalyzer(df_cleaned)
    analyzer.calculate_rfm()
    
    visualizer = Visualizer(output_dir='../output')
    visualizer.generate_all_charts(df_cleaned, analyzer.rfm_df)
