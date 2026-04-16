import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, Any

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid", {'font.sans-serif': ['Arial Unicode MS', 'SimHei']})

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Visualizer:
    def __init__(self, output_dir: str = "../output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def plot_sales_heatmap(self, heatmap_data: pd.DataFrame, filename: str = "sales_heatmap.png") -> str:
        try:
            logger.info("生成销售热力图...")
            
            plt.figure(figsize=(12, 8))
            heatmap_data_normalized = heatmap_data.div(heatmap_data.sum(axis=1), axis=0) * 100
            
            sns.heatmap(heatmap_data_normalized, 
                       annot=True, 
                       fmt='.1f', 
                       cmap='YlOrRd',
                       cbar_kws={'label': '销售额占比 (%)'})
            
            plt.title('各区域商品品类销售热力图', fontsize=16, pad=20)
            plt.xlabel('商品品类', fontsize=12)
            plt.ylabel('销售区域', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"热力图已保存: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"生成热力图失败: {str(e)}")
            raise
            
    def plot_customer_segmentation(self, rfm_df: pd.DataFrame, filename: str = "customer_segmentation.png") -> str:
        try:
            logger.info("生成客户分群散点图...")
            
            plt.figure(figsize=(14, 10))
            
            segments = rfm_df['Customer_Segment'].unique()
            colors = sns.color_palette('husl', len(segments))
            
            for i, segment in enumerate(segments):
                segment_data = rfm_df[rfm_df['Customer_Segment'] == segment]
                plt.scatter(segment_data['Recency'], 
                           segment_data['Monetary'],
                           s=segment_data['Frequency'] * 20,
                           alpha=0.6,
                           color=colors[i],
                           label=segment,
                           edgecolors='white',
                           linewidth=0.5)
            
            plt.xlabel('最近购买天数 (Recency)', fontsize=12)
            plt.ylabel('累计消费金额 (Monetary)', fontsize=12)
            plt.title('RFM客户分群散点图\n(点大小表示购买频率)', fontsize=16, pad=20)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"客户分群图已保存: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"生成客户分群图失败: {str(e)}")
            raise
            
    def plot_sales_trend(self, monthly_sales: pd.DataFrame, filename: str = "sales_trend.png") -> str:
        try:
            logger.info("生成销售趋势图...")
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
            
            ax1.plot(monthly_sales['year_month_str'], monthly_sales['total_amount'], 
                    marker='o', linewidth=2, color='#2E86AB', markersize=6)
            ax1.set_title('月度销售趋势图', fontsize=16, pad=20)
            ax1.set_ylabel('销售额 (元)', fontsize=12)
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
            
            for i, v in enumerate(monthly_sales['total_amount']):
                if i % 3 == 0:
                    ax1.annotate(f'{v/1000:.0f}k', 
                                (i, v),
                                textcoords="offset points",
                                xytext=(0, 10),
                                ha='center',
                                fontsize=8)
            
            growth = monthly_sales['total_amount'].pct_change() * 100
            colors = ['#A23B72' if x >= 0 else '#F18F01' for x in growth]
            ax2.bar(monthly_sales['year_month_str'], growth, color=colors, alpha=0.7)
            ax2.set_ylabel('增长率 (%)', fontsize=12)
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"销售趋势图已保存: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"生成销售趋势图失败: {str(e)}")
            raise
            
    def plot_category_distribution(self, category_sales: pd.DataFrame, filename: str = "category_distribution.png") -> str:
        try:
            logger.info("生成品类分布图...")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            colors = sns.color_palette('Set3', len(category_sales))
            wedges, texts, autotexts = ax1.pie(category_sales['总销售额'], 
                                               labels=category_sales.index,
                                               autopct='%1.1f%%',
                                               colors=colors,
                                               startangle=90)
            ax1.set_title('各品类销售额占比', fontsize=14, pad=20)
            
            sns.barplot(x=category_sales.index, y='总销售额', data=category_sales, 
                       palette='viridis', ax=ax2)
            ax2.set_title('各品类销售额对比', fontsize=14, pad=20)
            ax2.set_ylabel('销售额 (元)', fontsize=12)
            ax2.set_xlabel('品类', fontsize=12)
            ax2.tick_params(axis='x', rotation=45)
            
            for i, v in enumerate(category_sales['总销售额']):
                ax2.text(i, v, f'{v/1000:.0f}k', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"品类分布图已保存: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"生成品类分布图失败: {str(e)}")
            raise
            
    def plot_rfm_distribution(self, rfm_df: pd.DataFrame, filename: str = "rfm_distribution.png") -> str:
        try:
            logger.info("生成RFM分布图...")
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            axes = axes.flatten()
            
            sns.histplot(rfm_df['Recency'], kde=True, ax=axes[0], color='#2E86AB')
            axes[0].set_title('Recency (最近购买天数)分布', fontsize=12)
            axes[0].set_xlabel('天数')
            
            sns.histplot(rfm_df['Frequency'], kde=True, ax=axes[1], color='#A23B72')
            axes[1].set_title('Frequency (购买频率)分布', fontsize=12)
            axes[1].set_xlabel('次数')
            
            sns.histplot(rfm_df['Monetary'], kde=True, ax=axes[2], color='#F18F01')
            axes[2].set_title('Monetary (消费金额)分布', fontsize=12)
            axes[2].set_xlabel('金额 (元)')
            
            segment_counts = rfm_df['Customer_Segment'].value_counts()
            sns.barplot(x=segment_counts.index, y=segment_counts.values, 
                       palette='husl', ax=axes[3])
            axes[3].set_title('客户分群数量分布', fontsize=12)
            axes[3].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"RFM分布图已保存: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"生成RFM分布图失败: {str(e)}")
            raise
            
    def plot_region_sales(self, region_sales: pd.DataFrame, filename: str = "region_sales.png") -> str:
        try:
            logger.info("生成区域销售图...")
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
            
            sns.barplot(x=region_sales.index, y='总销售额', data=region_sales, 
                       palette='Blues_d', ax=axes[0])
            axes[0].set_title('各区域总销售额', fontsize=12)
            axes[0].tick_params(axis='x', rotation=45)
            axes[0].set_ylabel('销售额 (元)')
            
            sns.barplot(x=region_sales.index, y='平均订单金额', data=region_sales, 
                       palette='Greens_d', ax=axes[1])
            axes[1].set_title('各区域平均订单金额', fontsize=12)
            axes[1].tick_params(axis='x', rotation=45)
            axes[1].set_ylabel('金额 (元)')
            
            sns.barplot(x=region_sales.index, y='订单数', data=region_sales, 
                       palette='Oranges_d', ax=axes[2])
            axes[2].set_title('各区域订单数', fontsize=12)
            axes[2].tick_params(axis='x', rotation=45)
            
            sns.barplot(x=region_sales.index, y='客户数', data=region_sales, 
                       palette='Purples_d', ax=axes[3])
            axes[3].set_title('各区域客户数', fontsize=12)
            axes[3].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"区域销售图已保存: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"生成区域销售图失败: {str(e)}")
            raise
            
    def plot_forecast_comparison(self, actual: np.ndarray, predicted: np.ndarray, 
                                dates: pd.Series, filename: str = "forecast_comparison.png") -> str:
        try:
            logger.info("生成预测对比图...")
            
            plt.figure(figsize=(14, 8))
            
            x = np.arange(len(actual))
            
            plt.plot(x, actual, marker='o', label='实际销售额', color='#2E86AB', linewidth=2)
            plt.plot(x, predicted, marker='s', label='预测销售额', color='#A23B72', linewidth=2, linestyle='--')
            
            plt.fill_between(x, actual, predicted, alpha=0.2, color='gray')
            
            rmse = np.sqrt(np.mean((actual - predicted) ** 2))
            mae = np.mean(np.abs(actual - predicted))
            
            plt.title(f'销售预测对比图\nRMSE: {rmse:.2f}, MAE: {mae:.2f}', fontsize=16, pad=20)
            plt.xlabel('时间', fontsize=12)
            plt.ylabel('销售额 (元)', fontsize=12)
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            
            date_labels = dates.astype(str).tolist()
            plt.xticks(x[::2], date_labels[::2], rotation=45)
            
            plt.tight_layout()
            
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"预测对比图已保存: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"生成预测对比图失败: {str(e)}")
            raise
            
    def generate_all_charts(self, analysis_results: Dict[str, Any], feature_results: Dict[str, Any],
                           forecast_data: Dict[str, Any] = None) -> Dict[str, str]:
        try:
            logger.info("开始生成所有图表...")
            
            charts = {}
            
            charts['sales_heatmap'] = self.plot_sales_heatmap(
                analysis_results['region_perf']['region_category_heatmap']
            )
            
            charts['customer_segmentation'] = self.plot_customer_segmentation(
                feature_results['rfm_data']
            )
            
            charts['sales_trend'] = self.plot_sales_trend(
                analysis_results['sales_trend']['monthly_sales']
            )
            
            charts['category_distribution'] = self.plot_category_distribution(
                analysis_results['product_perf']['category_sales']
            )
            
            charts['rfm_distribution'] = self.plot_rfm_distribution(
                feature_results['rfm_data']
            )
            
            charts['region_sales'] = self.plot_region_sales(
                analysis_results['region_perf']['region_sales']
            )
            
            if forecast_data:
                charts['forecast'] = self.plot_forecast_comparison(
                    forecast_data['actual'],
                    forecast_data['predicted'],
                    forecast_data['dates']
                )
            
            logger.info(f"所有图表生成完成，共 {len(charts)} 个文件")
            return charts
            
        except Exception as e:
            logger.error(f"生成图表失败: {str(e)}")
            raise

if __name__ == '__main__':
    from data_loader import DataLoader
    from feature_engineering import FeatureEngineer
    from analysis import SalesAnalyzer
    
    loader = DataLoader()
    loader.load_all_data()
    merged = loader.merge_data()
    
    fe = FeatureEngineer()
    features = fe.transform_all(merged, loader.customers_df)
    
    analyzer = SalesAnalyzer()
    results = analyzer.run_full_analysis(features['enhanced_data'], features['customer_features'], features['product_features'])
    
    viz = Visualizer()
    charts = viz.generate_all_charts(results, features)
    print("生成的图表:", charts)
