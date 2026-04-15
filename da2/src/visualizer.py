import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from typing import Dict, Optional, List
import os

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class Visualizer:
    def __init__(self, output_dir: str = 'output'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.figures = {}
        
    def plot_sales_heatmap(self, daily_trend: pd.DataFrame, filename: str = 'sales_heatmap.png') -> str:
        try:
            daily_trend = daily_trend.copy()
            daily_trend['date'] = pd.to_datetime(daily_trend['date'])
            daily_trend['year'] = daily_trend['date'].dt.year
            daily_trend['month'] = daily_trend['date'].dt.month
            daily_trend['day'] = daily_trend['date'].dt.day
            
            for year in daily_trend['year'].unique():
                year_data = daily_trend[daily_trend['year'] == year]
                
                pivot_data = year_data.pivot_table(
                    values='revenue',
                    index='month',
                    columns='day',
                    aggfunc='sum'
                )
                
                fig, ax = plt.subplots(figsize=(16, 8))
                
                sns.heatmap(
                    pivot_data,
                    cmap='YlOrRd',
                    annot=False,
                    fmt='.0f',
                    cbar_kws={'label': '销售额'},
                    ax=ax
                )
                
                ax.set_title(f'{year}年销售热力图', fontsize=16, fontweight='bold')
                ax.set_xlabel('日期', fontsize=12)
                ax.set_ylabel('月份', fontsize=12)
                
                month_labels = ['1月', '2月', '3月', '4月', '5月', '6月', 
                               '7月', '8月', '9月', '10月', '11月', '12月']
                ax.set_yticklabels(month_labels[:len(pivot_data)], rotation=0)
                
                plt.tight_layout()
                
                filepath = os.path.join(self.output_dir, f'{filename.replace(".png", f"_{year}.png")}')
                plt.savefig(filepath, dpi=150, bbox_inches='tight')
                plt.close()
                
                self.figures[f'sales_heatmap_{year}'] = filepath
            
            return filepath
        except Exception as e:
            print(f"生成销售热力图时出错: {str(e)}")
            return ""
    
    def plot_customer_scatter(self, rfm_df: pd.DataFrame, filename: str = 'customer_scatter.png') -> str:
        try:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            segment_colors = {
                '重要价值客户': '#FF6B6B',
                '重要发展客户': '#4ECDC4',
                '重要保持客户': '#45B7D1',
                '重要挽留客户': '#96CEB4',
                '新客户': '#FFEAA7',
                '一般客户': '#DDA0DD',
                '低价值客户': '#C0C0C0'
            }
            
            ax1 = axes[0]
            for segment in rfm_df['segment'].unique():
                segment_data = rfm_df[rfm_df['segment'] == segment]
                ax1.scatter(
                    segment_data['recency'],
                    segment_data['frequency'],
                    c=segment_colors.get(segment, '#999999'),
                    label=segment,
                    alpha=0.6,
                    s=50
                )
            
            ax1.set_xlabel('最近购买天数 (Recency)', fontsize=12)
            ax1.set_ylabel('购买频次 (Frequency)', fontsize=12)
            ax1.set_title('客户RFM分群散点图 (R vs F)', fontsize=14, fontweight='bold')
            ax1.legend(loc='upper right', fontsize=8)
            ax1.grid(True, alpha=0.3)
            
            ax2 = axes[1]
            for segment in rfm_df['segment'].unique():
                segment_data = rfm_df[rfm_df['segment'] == segment]
                ax2.scatter(
                    segment_data['frequency'],
                    segment_data['monetary'],
                    c=segment_colors.get(segment, '#999999'),
                    label=segment,
                    alpha=0.6,
                    s=50
                )
            
            ax2.set_xlabel('购买频次 (Frequency)', fontsize=12)
            ax2.set_ylabel('消费金额 (Monetary)', fontsize=12)
            ax2.set_title('客户RFM分群散点图 (F vs M)', fontsize=14, fontweight='bold')
            ax2.legend(loc='upper right', fontsize=8)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.figures['customer_scatter'] = filepath
            return filepath
        except Exception as e:
            print(f"生成客户分群散点图时出错: {str(e)}")
            return ""
    
    def plot_time_series(self, daily_trend: pd.DataFrame, monthly_trend: pd.DataFrame, 
                        filename: str = 'time_series.png') -> str:
        try:
            fig, axes = plt.subplots(2, 1, figsize=(16, 10))
            
            ax1 = axes[0]
            daily_trend = daily_trend.copy()
            daily_trend['date'] = pd.to_datetime(daily_trend['date'])
            
            ax1.plot(daily_trend['date'], daily_trend['revenue'], 
                    label='日销售额', alpha=0.5, linewidth=0.8)
            ax1.plot(daily_trend['date'], daily_trend['revenue_ma7'], 
                    label='7日移动平均', linewidth=2, color='orange')
            ax1.plot(daily_trend['date'], daily_trend['revenue_ma30'], 
                    label='30日移动平均', linewidth=2, color='red')
            
            ax1.set_xlabel('日期', fontsize=12)
            ax1.set_ylabel('销售额', fontsize=12)
            ax1.set_title('销售趋势时间序列图', fontsize=14, fontweight='bold')
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            ax2 = axes[1]
            x_pos = range(len(monthly_trend))
            
            bars = ax2.bar(x_pos, monthly_trend['revenue'], color='steelblue', alpha=0.7)
            
            ax2.set_xlabel('月份', fontsize=12)
            ax2.set_ylabel('销售额', fontsize=12)
            ax2.set_title('月度销售趋势', fontsize=14, fontweight='bold')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(monthly_trend['year_month'], rotation=45, ha='right')
            ax2.grid(True, alpha=0.3, axis='y')
            
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height/1000:.0f}K',
                        ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.figures['time_series'] = filepath
            return filepath
        except Exception as e:
            print(f"生成时间序列图时出错: {str(e)}")
            return ""
    
    def plot_category_distribution(self, category_features: pd.DataFrame, 
                                   filename: str = 'category_distribution.png') -> str:
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            ax1 = axes[0]
            colors = plt.cm.Set3(np.linspace(0, 1, len(category_features)))
            
            wedges, texts, autotexts = ax1.pie(
                category_features['total_revenue'],
                labels=category_features['category'],
                autopct='%1.1f%%',
                colors=colors,
                startangle=90
            )
            
            ax1.set_title('品类销售额占比', fontsize=14, fontweight='bold')
            
            ax2 = axes[1]
            x_pos = range(len(category_features))
            
            bars = ax2.bar(x_pos, category_features['total_revenue'], color=colors)
            
            ax2.set_xlabel('品类', fontsize=12)
            ax2.set_ylabel('销售额', fontsize=12)
            ax2.set_title('品类销售额对比', fontsize=14, fontweight='bold')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(category_features['category'], rotation=45, ha='right')
            ax2.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.figures['category_distribution'] = filepath
            return filepath
        except Exception as e:
            print(f"生成品类分布图时出错: {str(e)}")
            return ""
    
    def plot_weekly_pattern(self, weekly_pattern: pd.DataFrame, 
                           filename: str = 'weekly_pattern.png') -> str:
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            colors = ['#FF6B6B' if i >= 5 else '#4ECDC4' for i in range(7)]
            
            bars = ax.bar(weekly_pattern['day_name'], weekly_pattern['total_revenue'], 
                         color=colors, alpha=0.8)
            
            ax.set_xlabel('星期', fontsize=12)
            ax.set_ylabel('销售额', fontsize=12)
            ax.set_title('周销售模式分析', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height/1000:.0f}K',
                       ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.figures['weekly_pattern'] = filepath
            return filepath
        except Exception as e:
            print(f"生成周销售模式图时出错: {str(e)}")
            return ""
    
    def plot_customer_segment(self, segment_summary: pd.DataFrame, 
                             filename: str = 'customer_segment.png') -> str:
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            ax1 = axes[0]
            colors = plt.cm.Spectral(np.linspace(0, 1, len(segment_summary)))
            
            wedges, texts, autotexts = ax1.pie(
                segment_summary['customer_count'],
                labels=segment_summary['segment'],
                autopct='%1.1f%%',
                colors=colors,
                startangle=90
            )
            
            ax1.set_title('客户分群占比', fontsize=14, fontweight='bold')
            
            ax2 = axes[1]
            x_pos = range(len(segment_summary))
            
            width = 0.35
            bars1 = ax2.bar([p - width/2 for p in x_pos], 
                           segment_summary['customer_share'], 
                           width, label='客户占比%', color='#4ECDC4')
            bars2 = ax2.bar([p + width/2 for p in x_pos], 
                           segment_summary['revenue_share'], 
                           width, label='收入占比%', color='#FF6B6B')
            
            ax2.set_xlabel('客户分群', fontsize=12)
            ax2.set_ylabel('占比 (%)', fontsize=12)
            ax2.set_title('客户分群价值对比', fontsize=14, fontweight='bold')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(segment_summary['segment'], rotation=45, ha='right')
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.figures['customer_segment'] = filepath
            return filepath
        except Exception as e:
            print(f"生成客户分群图时出错: {str(e)}")
            return ""
    
    def plot_top_products(self, top_products: pd.DataFrame, 
                         filename: str = 'top_products.png') -> str:
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            top_10 = top_products.head(10)
            
            colors = plt.cm.viridis(np.linspace(0, 0.8, len(top_10)))
            
            bars = ax.barh(range(len(top_10)), top_10['total_revenue'], color=colors)
            
            ax.set_yticks(range(len(top_10)))
            ax.set_yticklabels(top_10['product_name'])
            ax.set_xlabel('销售额', fontsize=12)
            ax.set_title('Top 10 热销商品', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2.,
                       f' {width/1000:.0f}K',
                       ha='left', va='center', fontsize=9)
            
            plt.tight_layout()
            
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.figures['top_products'] = filepath
            return filepath
        except Exception as e:
            print(f"生成热销商品图时出错: {str(e)}")
            return ""
    
    def plot_product_association_heatmap(self, category_association: pd.DataFrame, 
                                        filename: str = 'product_association.png') -> str:
        try:
            if len(category_association) == 0:
                return ""
            
            categories = list(set(category_association['category_a'].tolist() + 
                                 category_association['category_b'].tolist()))
            
            n_categories = len(categories)
            lift_matrix = pd.DataFrame(0, index=categories, columns=categories)
            
            for _, row in category_association.iterrows():
                lift_matrix.loc[row['category_a'], row['category_b']] = row['lift']
                lift_matrix.loc[row['category_b'], row['category_a']] = row['lift']
            
            for cat in categories:
                lift_matrix.loc[cat, cat] = 1.0
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            sns.heatmap(
                lift_matrix,
                annot=True,
                fmt='.2f',
                cmap='RdYlGn',
                center=1.0,
                square=True,
                ax=ax
            )
            
            ax.set_title('品类关联度热力图 (Lift值)', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.figures['product_association'] = filepath
            return filepath
        except Exception as e:
            print(f"生成商品关联热力图时出错: {str(e)}")
            return ""
    
    def plot_prediction_comparison(self, actual: pd.DataFrame, predicted: pd.DataFrame,
                                   filename: str = 'prediction_comparison.png') -> str:
        try:
            fig, ax = plt.subplots(figsize=(14, 6))
            
            ax.plot(actual['date'], actual['actual'], 
                   label='实际值', linewidth=2, color='blue')
            ax.plot(predicted['date'], predicted['predicted'], 
                   label='预测值', linewidth=2, color='red', linestyle='--')
            
            ax.fill_between(
                predicted['date'],
                predicted['predicted_lower'],
                predicted['predicted_upper'],
                alpha=0.2,
                color='red',
                label='预测区间'
            )
            
            ax.set_xlabel('日期', fontsize=12)
            ax.set_ylabel('销售额', fontsize=12)
            ax.set_title('销售预测对比图', fontsize=14, fontweight='bold')
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.figures['prediction_comparison'] = filepath
            return filepath
        except Exception as e:
            print(f"生成预测对比图时出错: {str(e)}")
            return ""
    
    def get_all_figures(self) -> Dict[str, str]:
        return self.figures
