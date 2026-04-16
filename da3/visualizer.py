"""
可视化模块
生成各类图表并保存到output目录
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, List
import os
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150


class Visualizer:
    """数据可视化类"""
    
    def __init__(self, output_dir: str = 'da3/output'):
        """
        初始化可视化器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.figures = {}
        
    def plot_sales_trend(self, daily_sales: pd.DataFrame, 
                        save_name: str = 'sales_trend.png') -> str:
        """
        绘制销售额趋势折线图
        
        Args:
            daily_sales: 日销售数据
            save_name: 保存文件名
            
        Returns:
            保存路径
        """
        print("\n生成销售额趋势图...")
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        ax.plot(daily_sales['date'], daily_sales['total_sales'], 
                'b-', alpha=0.5, label='日销售额', linewidth=1)
        
        if 'MA_7' in daily_sales.columns:
            ax.plot(daily_sales['date'], daily_sales['MA_7'], 
                    'r-', label='7日移动平均', linewidth=2)
        
        if 'MA_30' in daily_sales.columns:
            ax.plot(daily_sales['date'], daily_sales['MA_30'], 
                    'g-', label='30日移动平均', linewidth=2)
        
        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel('销售额 (元)', fontsize=12)
        ax.set_title('销售额趋势分析', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, save_name)
        plt.savefig(filepath, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.figures['sales_trend'] = filepath
        print(f"图表已保存: {filepath}")
        
        return filepath
    
    def plot_amount_distribution(self, df: pd.DataFrame,
                                save_name: str = 'amount_distribution.png') -> str:
        """
        绘制金额分布直方图和箱线图
        
        Args:
            df: 数据DataFrame
            save_name: 保存文件名
            
        Returns:
            保存路径
        """
        print("\n生成金额分布图...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1 = axes[0]
        ax1.hist(df['amount'], bins=50, color='steelblue', 
                edgecolor='white', alpha=0.7)
        ax1.set_xlabel('订单金额 (元)', fontsize=12)
        ax1.set_ylabel('频数', fontsize=12)
        ax1.set_title('订单金额分布直方图', fontsize=13, fontweight='bold')
        ax1.axvline(df['amount'].mean(), color='red', linestyle='--', 
                   label=f'均值: {df["amount"].mean():.2f}')
        ax1.axvline(df['amount'].median(), color='green', linestyle='--',
                   label=f'中位数: {df["amount"].median():.2f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[1]
        box_data = [df['amount']]
        bp = ax2.boxplot(box_data, patch_artist=True, 
                        labels=['订单金额'])
        bp['boxes'][0].set_facecolor('lightblue')
        ax2.set_ylabel('金额 (元)', fontsize=12)
        ax2.set_title('订单金额箱线图', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, save_name)
        plt.savefig(filepath, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.figures['amount_distribution'] = filepath
        print(f"图表已保存: {filepath}")
        
        return filepath
    
    def plot_category_sales(self, category_stats: pd.DataFrame,
                           save_name: str = 'category_sales.png') -> str:
        """
        绘制类别销售额条形图
        
        Args:
            category_stats: 类别销售统计
            save_name: 保存文件名
            
        Returns:
            保存路径
        """
        print("\n生成类别销售额图...")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        categories = category_stats.index.tolist()
        sales = category_stats['总销售额'].values
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
        
        bars = ax.barh(categories, sales, color=colors, edgecolor='gray')
        
        for bar, val in zip(bars, sales):
            ax.text(val + max(sales)*0.01, bar.get_y() + bar.get_height()/2,
                   f'{val:,.0f}', va='center', fontsize=9)
        
        ax.set_xlabel('销售额 (元)', fontsize=12)
        ax.set_ylabel('商品类别', fontsize=12)
        ax.set_title('各类别销售额对比', fontsize=14, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, save_name)
        plt.savefig(filepath, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.figures['category_sales'] = filepath
        print(f"图表已保存: {filepath}")
        
        return filepath
    
    def plot_cluster_scatter(self, rfm_df: pd.DataFrame,
                            save_name: str = 'cluster_scatter.png') -> str:
        """
        绘制用户价值聚类散点图
        
        Args:
            rfm_df: RFM分析结果DataFrame
            save_name: 保存文件名
            
        Returns:
            保存路径
        """
        print("\n生成用户聚类散点图...")
        
        if 'Cluster' not in rfm_df.columns:
            print("缺少聚类标签，无法绘制聚类图")
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        ax1 = axes[0]
        clusters = rfm_df['Cluster'].unique()
        colors = plt.cm.Set1(np.linspace(0, 1, len(clusters)))
        
        for cluster, color in zip(sorted(clusters), colors):
            mask = rfm_df['Cluster'] == cluster
            ax1.scatter(rfm_df.loc[mask, 'Recency'], 
                       rfm_df.loc[mask, 'Monetary'],
                       c=[color], label=f'群组 {cluster}', 
                       alpha=0.6, s=50)
        
        ax1.set_xlabel('Recency (最近购买天数)', fontsize=12)
        ax1.set_ylabel('Monetary (消费金额)', fontsize=12)
        ax1.set_title('用户聚类: Recency vs Monetary', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[1]
        for cluster, color in zip(sorted(clusters), colors):
            mask = rfm_df['Cluster'] == cluster
            ax2.scatter(rfm_df.loc[mask, 'Frequency'], 
                       rfm_df.loc[mask, 'Monetary'],
                       c=[color], label=f'群组 {cluster}', 
                       alpha=0.6, s=50)
        
        ax2.set_xlabel('Frequency (购买频率)', fontsize=12)
        ax2.set_ylabel('Monetary (消费金额)', fontsize=12)
        ax2.set_title('用户聚类: Frequency vs Monetary', fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, save_name)
        plt.savefig(filepath, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.figures['cluster_scatter'] = filepath
        print(f"图表已保存: {filepath}")
        
        return filepath
    
    def plot_correlation_heatmap(self, corr_matrix: pd.DataFrame,
                                save_name: str = 'correlation_heatmap.png') -> str:
        """
        绘制相关性热力图
        
        Args:
            corr_matrix: 相关系数矩阵
            save_name: 保存文件名
            
        Returns:
            保存路径
        """
        print("\n生成相关性热力图...")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                   cmap='RdBu_r', center=0, square=True,
                   linewidths=0.5, ax=ax,
                   cbar_kws={'shrink': 0.8, 'label': '相关系数'})
        
        ax.set_title('变量相关性热力图', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, save_name)
        plt.savefig(filepath, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.figures['correlation_heatmap'] = filepath
        print(f"图表已保存: {filepath}")
        
        return filepath
    
    def plot_prediction_comparison(self, prediction_df: pd.DataFrame,
                                   save_name: str = 'prediction_comparison.png') -> str:
        """
        绘制预测对比图
        
        Args:
            prediction_df: 包含实际值和预测值的DataFrame
            save_name: 保存文件名
            
        Returns:
            保存路径
        """
        print("\n生成预测对比图...")
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        ax.plot(prediction_df['date'], prediction_df['total_sales'],
                'b-', alpha=0.7, label='实际销售额', linewidth=1.5)
        ax.plot(prediction_df['date'], prediction_df['predicted_sales'],
                'r--', alpha=0.7, label='预测销售额', linewidth=1.5)
        
        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel('销售额 (元)', fontsize=12)
        ax.set_title('销售预测 vs 实际值', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, save_name)
        plt.savefig(filepath, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.figures['prediction_comparison'] = filepath
        print(f"图表已保存: {filepath}")
        
        return filepath
    
    def plot_rfm_distribution(self, rfm_df: pd.DataFrame,
                             save_name: str = 'rfm_distribution.png') -> str:
        """
        绘制RFM分布图
        
        Args:
            rfm_df: RFM分析结果
            save_name: 保存文件名
            
        Returns:
            保存路径
        """
        print("\n生成RFM分布图...")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for idx, col in enumerate(['Recency', 'Frequency', 'Monetary']):
            ax = axes[idx]
            ax.hist(rfm_df[col], bins=30, color='steelblue', 
                   edgecolor='white', alpha=0.7)
            ax.set_xlabel(col, fontsize=11)
            ax.set_ylabel('频数', fontsize=11)
            ax.set_title(f'{col}分布', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, save_name)
        plt.savefig(filepath, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.figures['rfm_distribution'] = filepath
        print(f"图表已保存: {filepath}")
        
        return filepath
    
    def generate_all_plots(self, df: pd.DataFrame, 
                          analysis_results: Dict[str, Any]) -> Dict[str, str]:
        """
        生成所有图表
        
        Args:
            df: 原始数据
            analysis_results: 分析结果字典
            
        Returns:
            图表路径字典
        """
        print("\n" + "=" * 50)
        print("开始生成可视化图表")
        print("=" * 50)
        
        if 'daily_sales' in analysis_results:
            self.plot_sales_trend(analysis_results['daily_sales'])
        
        self.plot_amount_distribution(df)
        
        if 'category_stats' in analysis_results:
            self.plot_category_sales(analysis_results['category_stats'])
        
        if 'rfm_analysis' in analysis_results:
            self.plot_cluster_scatter(analysis_results['rfm_analysis'])
            self.plot_rfm_distribution(analysis_results['rfm_analysis'])
        
        if 'correlation_matrix' in analysis_results:
            self.plot_correlation_heatmap(analysis_results['correlation_matrix'])
        
        if 'prediction_results' in analysis_results:
            self.plot_prediction_comparison(
                analysis_results['prediction_results']['predictions']
            )
        
        print("\n" + "=" * 50)
        print(f"图表生成完成，共{len(self.figures)}张")
        print(f"保存目录: {self.output_dir}")
        print("=" * 50)
        
        return self.figures


def visualize_data(df: pd.DataFrame, 
                  analysis_results: Dict[str, Any],
                  output_dir: str = 'da3/output') -> Dict[str, str]:
    """
    可视化主函数
    
    Args:
        df: 数据DataFrame
        analysis_results: 分析结果
        output_dir: 输出目录
        
    Returns:
        图表路径字典
    """
    visualizer = Visualizer(output_dir=output_dir)
    return visualizer.generate_all_plots(df, analysis_results)


if __name__ == '__main__':
    from data_generator import generate_and_save_data
    from data_cleaner import clean_data
    from data_analyzer import analyze_data
    
    df = generate_and_save_data(n_records=100)
    cleaned_df, _ = clean_data(df)
    results = analyze_data(cleaned_df)
    figures = visualize_data(cleaned_df, results)
    
    print("\n生成的图表:")
    for name, path in figures.items():
        print(f"  {name}: {path}")
