"""
数据可视化模块
生成销售额趋势图、分布图、条形图、散点图、热力图等
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import os
from typing import Dict, Any, List
import warnings
warnings.filterwarnings('ignore')


class Visualizer:
    """数据可视化器"""
    
    def __init__(self, output_dir: str = 'da3/output'):
        """
        初始化可视化器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.chinese_font = self._setup_chinese_font()
        
        plt.style.use('seaborn-v0_8-whitegrid')
        self.colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B',
                      '#95C623', '#4A5899', '#D35400', '#1ABC9C', '#9B59B6']
    
    def _setup_chinese_font(self):
        """设置中文字体，返回FontProperties对象"""
        font_paths = [
            '/System/Library/Fonts/PingFang.ttc',
            '/System/Library/Fonts/Hiragino Sans GB.ttc',
            '/System/Library/Fonts/STHeiti Medium.ttc',
            '/System/Library/Fonts/STHeiti Light.ttc',
            '/System/Library/Fonts/Supplemental/Songti.ttc',
        ]
        
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    font_prop = fm.FontProperties(fname=font_path)
                    plt.rcParams['axes.unicode_minus'] = False
                    print(f"成功加载中文字体: {font_path}")
                    return font_prop
                except Exception as e:
                    print(f"加载字体失败 {font_path}: {e}")
                    continue
        
        print("警告: 未找到可用的中文字体")
        return None
    
    def _set_chinese_text(self, ax, xlabel=None, ylabel=None, title=None, fontsize=12):
        """设置中文文本"""
        font = self.chinese_font
        if xlabel:
            ax.set_xlabel(xlabel, fontproperties=font, fontsize=fontsize)
        if ylabel:
            ax.set_ylabel(ylabel, fontproperties=font, fontsize=fontsize)
        if title:
            ax.set_title(title, fontproperties=font, fontsize=fontsize+2, fontweight='bold')
    
    def _set_legend_font(self, legend):
        """设置图例字体"""
        if self.chinese_font and legend:
            for text in legend.get_texts():
                text.set_fontproperties(self.chinese_font)
    
    def plot_sales_trend(self, daily_sales: pd.DataFrame, 
                         save_name: str = 'sales_trend.png') -> str:
        """
        绘制销售额趋势折线图
        
        Args:
            daily_sales: 包含日期和销售额的DataFrame
            save_name: 保存文件名
            
        Returns:
            保存的文件路径
        """
        try:
            fig, ax = plt.subplots(figsize=(14, 6))
            
            ax.plot(daily_sales['date'], daily_sales['daily_sales'], 
                   color=self.colors[0], linewidth=1.5, alpha=0.7, label='日销售额')
            
            if 'ma_7' in daily_sales.columns:
                ax.plot(daily_sales['date'], daily_sales['ma_7'], 
                       color=self.colors[1], linewidth=2.5, label='7日移动平均')
            
            if 'ma_30' in daily_sales.columns:
                ax.plot(daily_sales['date'], daily_sales['ma_30'], 
                       color=self.colors[2], linewidth=2.5, label='30日移动平均')
            
            ax.fill_between(daily_sales['date'], daily_sales['daily_sales'], 
                           alpha=0.3, color=self.colors[0])
            
            self._set_chinese_text(ax, xlabel='日期', ylabel='销售额 (元)', title='电商销售额趋势分析')
            legend = ax.legend(loc='upper right', fontsize=10)
            self._set_legend_font(legend)
            
            ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            filepath = os.path.join(self.output_dir, save_name)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ 销售趋势图已保存: {filepath}")
            return filepath
            
        except Exception as e:
            raise RuntimeError(f"绘制销售趋势图失败: {str(e)}")
    
    def plot_amount_distribution(self, df: pd.DataFrame,
                                 save_name: str = 'amount_distribution.png') -> str:
        """
        绘制金额分布直方图和箱线图
        
        Args:
            df: 包含金额数据的DataFrame
            save_name: 保存文件名
            
        Returns:
            保存的文件路径
        """
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            ax1 = axes[0]
            data = df['total_amount'].values
            n, bins, patches = ax1.hist(data, bins=50, color=self.colors[0], 
                                        alpha=0.7, edgecolor='white', linewidth=0.5)
            
            for i, patch in enumerate(patches):
                patch.set_facecolor(plt.cm.Blues(0.3 + 0.5 * i / len(patches)))
            
            ax1.axvline(df['total_amount'].mean(), color='red', linestyle='--', 
                       linewidth=2, label=f'均值: ¥{df["total_amount"].mean():.0f}')
            ax1.axvline(df['total_amount'].median(), color='green', linestyle='--', 
                       linewidth=2, label=f'中位数: ¥{df["total_amount"].median():.0f}')
            
            self._set_chinese_text(ax1, xlabel='订单金额 (元)', ylabel='频次', title='订单金额分布直方图')
            legend = ax1.legend(fontsize=10)
            self._set_legend_font(legend)
            
            ax2 = axes[1]
            bp = ax2.boxplot(df['total_amount'], patch_artist=True, vert=True,
                             widths=0.5, notch=True)
            
            bp['boxes'][0].set_facecolor(self.colors[1])
            bp['boxes'][0].set_alpha(0.7)
            bp['medians'][0].set_color('red')
            bp['medians'][0].set_linewidth(2)
            
            stats_text = f"最小值: ¥{df['total_amount'].min():.0f}\n"
            stats_text += f"Q1: ¥{df['total_amount'].quantile(0.25):.0f}\n"
            stats_text += f"中位数: ¥{df['total_amount'].median():.0f}\n"
            stats_text += f"Q3: ¥{df['total_amount'].quantile(0.75):.0f}\n"
            stats_text += f"最大值: ¥{df['total_amount'].max():.0f}"
            
            ax2.text(1.3, df['total_amount'].median(), stats_text, 
                    fontproperties=self.chinese_font, fontsize=10, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            self._set_chinese_text(ax2, ylabel='订单金额 (元)', title='订单金额箱线图')
            ax2.set_xticklabels(['订单金额'], fontproperties=self.chinese_font)
            
            plt.tight_layout()
            
            filepath = os.path.join(self.output_dir, save_name)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ 金额分布图已保存: {filepath}")
            return filepath
            
        except Exception as e:
            raise RuntimeError(f"绘制金额分布图失败: {str(e)}")
    
    def plot_category_sales(self, df: pd.DataFrame,
                           save_name: str = 'category_sales.png') -> str:
        """
        绘制类别销售额条形图
        
        Args:
            df: 包含类别和销售额的DataFrame
            save_name: 保存文件名
            
        Returns:
            保存的文件路径
        """
        try:
            category_sales = df.groupby('category')['total_amount'].sum().sort_values(ascending=True)
            
            fig, ax = plt.subplots(figsize=(12, 7))
            
            bars = ax.barh(category_sales.index, category_sales.values, 
                          color=self.colors[:len(category_sales)], alpha=0.8)
            
            for bar, value in zip(bars, category_sales.values):
                ax.text(value + max(category_sales.values) * 0.01, bar.get_y() + bar.get_height()/2,
                       f'¥{value:,.0f}', va='center', fontsize=10)
            
            self._set_chinese_text(ax, xlabel='销售额 (元)', ylabel='商品类别', title='各类别销售额分析')
            
            total_sales = category_sales.sum()
            ax.text(0.98, 0.02, f'总销售额: ¥{total_sales:,.0f}', 
                   transform=ax.transAxes, fontproperties=self.chinese_font,
                   fontsize=11, ha='right',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
            
            for label in ax.get_yticklabels():
                label.set_fontproperties(self.chinese_font)
            
            plt.tight_layout()
            
            filepath = os.path.join(self.output_dir, save_name)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ 类别销售图已保存: {filepath}")
            return filepath
            
        except Exception as e:
            raise RuntimeError(f"绘制类别销售图失败: {str(e)}")
    
    def plot_cluster_scatter(self, cluster_df: pd.DataFrame,
                            save_name: str = 'cluster_scatter.png') -> str:
        """
        绘制用户价值聚类散点图
        
        Args:
            cluster_df: 包含聚类结果的DataFrame
            save_name: 保存文件名
            
        Returns:
            保存的文件路径
        """
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            clusters = cluster_df['cluster'].unique()
            
            for i, cluster in enumerate(sorted(clusters)):
                cluster_data = cluster_df[cluster_df['cluster'] == cluster]
                ax.scatter(cluster_data['total_spent'], 
                          cluster_data['order_count'],
                          c=[self.colors[i % len(self.colors)]],
                          label=f'聚类 {cluster} ({len(cluster_data)}人)',
                          alpha=0.6, s=80, edgecolors='white', linewidth=0.5)
            
            self._set_chinese_text(ax, xlabel='消费总额 (元)', ylabel='订单数量', title='用户价值聚类分析')
            legend = ax.legend(loc='upper right', fontsize=10)
            self._set_legend_font(legend)
            
            ax.set_xscale('log')
            
            plt.tight_layout()
            
            filepath = os.path.join(self.output_dir, save_name)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ 聚类散点图已保存: {filepath}")
            return filepath
            
        except Exception as e:
            raise RuntimeError(f"绘制聚类散点图失败: {str(e)}")
    
    def plot_correlation_heatmap(self, corr_matrix: pd.DataFrame,
                                 save_name: str = 'correlation_heatmap.png') -> str:
        """
        绘制相关性热力图
        
        Args:
            corr_matrix: 相关性矩阵
            save_name: 保存文件名
            
        Returns:
            保存的文件路径
        """
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            
            sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                       cmap='RdBu_r', center=0, square=True,
                       linewidths=0.5, cbar_kws={"shrink": 0.8},
                       vmin=-1, vmax=1, ax=ax)
            
            self._set_chinese_text(ax, title='变量相关性热力图')
            
            for label in ax.get_xticklabels():
                label.set_fontproperties(self.chinese_font)
            for label in ax.get_yticklabels():
                label.set_fontproperties(self.chinese_font)
            
            plt.tight_layout()
            
            filepath = os.path.join(self.output_dir, save_name)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ 相关性热力图已保存: {filepath}")
            return filepath
            
        except Exception as e:
            raise RuntimeError(f"绘制相关性热力图失败: {str(e)}")
    
    def plot_rfm_segments(self, rfm_df: pd.DataFrame,
                         save_name: str = 'rfm_segments.png') -> str:
        """
        绘制RFM客户分层图
        
        Args:
            rfm_df: 包含RFM分层结果的DataFrame
            save_name: 保存文件名
            
        Returns:
            保存的文件路径
        """
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            ax1 = axes[0]
            segment_counts = rfm_df['segment'].value_counts()
            colors = plt.cm.Set3(np.linspace(0, 1, len(segment_counts)))
            
            wedges, texts, autotexts = ax1.pie(segment_counts.values, 
                                               labels=segment_counts.index,
                                               autopct='%1.1f%%',
                                               colors=colors,
                                               explode=[0.05] * len(segment_counts),
                                               shadow=True)
            
            for text in texts:
                text.set_fontproperties(self.chinese_font)
            
            self._set_chinese_text(ax1, title='客户分层占比')
            
            ax2 = axes[1]
            segment_monetary = rfm_df.groupby('segment')['monetary'].mean().sort_values(ascending=True)
            
            bars = ax2.barh(segment_monetary.index, segment_monetary.values,
                           color=colors[:len(segment_monetary)], alpha=0.8)
            
            for bar, value in zip(bars, segment_monetary.values):
                ax2.text(value + max(segment_monetary.values) * 0.01, 
                        bar.get_y() + bar.get_height()/2,
                        f'¥{value:,.0f}', va='center', fontsize=10)
            
            self._set_chinese_text(ax2, xlabel='平均消费金额 (元)', title='各分层平均消费金额')
            
            for label in ax2.get_yticklabels():
                label.set_fontproperties(self.chinese_font)
            
            plt.tight_layout()
            
            filepath = os.path.join(self.output_dir, save_name)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ RFM分层图已保存: {filepath}")
            return filepath
            
        except Exception as e:
            raise RuntimeError(f"绘制RFM分层图失败: {str(e)}")
    
    def plot_monthly_sales(self, monthly_sales: pd.DataFrame,
                          save_name: str = 'monthly_sales.png') -> str:
        """
        绘制月度销售分析图
        
        Args:
            monthly_sales: 月度销售数据
            save_name: 保存文件名
            
        Returns:
            保存的文件路径
        """
        try:
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            x = range(len(monthly_sales))
            bars = ax1.bar(x, monthly_sales['total_sales'], 
                          color=self.colors[0], alpha=0.7, label='月销售额')
            
            for bar, value in zip(bars, monthly_sales['total_sales']):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'¥{value/1000:.0f}K', ha='center', va='bottom', fontsize=9)
            
            self._set_chinese_text(ax1, xlabel='月份', ylabel='销售额 (元)', title='月度销售趋势分析')
            ax1.tick_params(axis='y', labelcolor=self.colors[0])
            ax1.set_xticks(x)
            ax1.set_xticklabels(monthly_sales['month'], rotation=45)
            
            ax2 = ax1.twinx()
            ax2.plot(x, monthly_sales['order_count'], 
                    color=self.colors[1], marker='o', linewidth=2, 
                    markersize=8, label='订单数')
            ax2.set_ylabel('订单数', fontproperties=self.chinese_font, fontsize=12, color=self.colors[1])
            ax2.tick_params(axis='y', labelcolor=self.colors[1])
            
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            legend = ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            self._set_legend_font(legend)
            
            plt.tight_layout()
            
            filepath = os.path.join(self.output_dir, save_name)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ 月度销售图已保存: {filepath}")
            return filepath
            
        except Exception as e:
            raise RuntimeError(f"绘制月度销售图失败: {str(e)}")
    
    def generate_all_plots(self, df: pd.DataFrame, 
                          daily_sales: pd.DataFrame,
                          corr_matrix: pd.DataFrame,
                          cluster_df: pd.DataFrame,
                          rfm_df: pd.DataFrame,
                          monthly_sales: pd.DataFrame) -> List[str]:
        """
        生成所有图表
        
        Args:
            df: 清洗后的数据
            daily_sales: 每日销售数据
            corr_matrix: 相关性矩阵
            cluster_df: 聚类结果
            rfm_df: RFM分析结果
            monthly_sales: 月度销售数据
            
        Returns:
            保存的文件路径列表
        """
        print("\n开始生成可视化图表...")
        print("=" * 50)
        
        filepaths = []
        
        filepaths.append(self.plot_sales_trend(daily_sales))
        filepaths.append(self.plot_amount_distribution(df))
        filepaths.append(self.plot_category_sales(df))
        filepaths.append(self.plot_cluster_scatter(cluster_df))
        filepaths.append(self.plot_correlation_heatmap(corr_matrix))
        filepaths.append(self.plot_rfm_segments(rfm_df))
        filepaths.append(self.plot_monthly_sales(monthly_sales))
        
        print("=" * 50)
        print(f"共生成 {len(filepaths)} 张图表")
        
        return filepaths


if __name__ == '__main__':
    from data_generator import DataGenerator
    from analyzer import (ExploratoryAnalyzer, TimeSeriesAnalyzer, 
                         RFMAnalyzer, CustomerClusterAnalyzer)
    
    generator = DataGenerator()
    df = generator.generate_data(n_records=600)
    
    visualizer = Visualizer()
    
    ts_analyzer = TimeSeriesAnalyzer(df)
    daily_sales = ts_analyzer.moving_average()
    monthly_sales = ts_analyzer.monthly_sales()
    
    explorer = ExploratoryAnalyzer(df)
    corr_matrix = explorer.correlation_analysis()
    
    cluster_analyzer = CustomerClusterAnalyzer(df)
    cluster_df, _ = cluster_analyzer.perform_clustering()
    
    rfm = RFMAnalyzer(df)
    rfm_df = rfm.calculate_rfm()
    rfm_df = rfm.segment_customers()
    
    visualizer.generate_all_plots(df, daily_sales, corr_matrix, 
                                  cluster_df, rfm_df, monthly_sales)
