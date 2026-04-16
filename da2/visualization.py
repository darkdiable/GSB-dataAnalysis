"""
可视化模块
使用matplotlib和seaborn生成各种图表
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import logging
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Visualizer:
    """可视化类"""

    def __init__(self, output_dir: str = 'output'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.figures = []

    def save_figure(self, filename: str, fig: Optional[plt.Figure] = None) -> str:
        """保存图表"""
        filepath = os.path.join(self.output_dir, filename)
        if fig is None:
            plt.savefig(filepath, bbox_inches='tight', dpi=150)
        else:
            fig.savefig(filepath, bbox_inches='tight', dpi=150)
        logger.info(f"图表已保存: {filepath}")
        return filepath
    def plot_sales_trend(self, daily_data: pd.DataFrame, monthly_data: pd.DataFrame) -> str:
        """销售趋势图"""
        try:
            fig, axes = plt.subplots(2, 1, figsize=(14, 10))

            # 日销售趋势
            axes[0].plot(daily_data['date'], daily_data['revenue'], alpha=0.3, color='blue', label='日销售额')
            if 'revenue_ma7' in daily_data.columns:
                axes[0].plot(daily_data['date'], daily_data['revenue_ma7'], color='red', linewidth=2, label='7日移动平均')
            axes[0].set_title('日销售趋势', fontsize=14, fontweight='bold')
            axes[0].set_xlabel('日期')
            axes[0].set_ylabel('销售额')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # 月销售趋势
            monthly_data['year_month_str'] = monthly_data['year_month'].astype(str)
            axes[1].bar(monthly_data['year_month_str'], monthly_data['revenue'], color='steelblue', alpha=0.7)
            axes[1].set_title('月度销售趋势', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('月份')
            axes[1].set_ylabel('销售额')
            axes[1].tick_params(axis='x', rotation=45)
            axes[1].grid(True, alpha=0.3, axis='y')

            plt.tight_layout()
            filepath = self.save_figure('sales_trend.png', fig)
            plt.close(fig)
            return filepath
        except Exception as e:
            logger.error(f"绘制销售趋势图出错: {e}")
            return ""

    def plot_sales_heatmap(self, df: pd.DataFrame) -> str:
        """销售热力图（按星期和小时）"""
        try:
            df['weekday_name'] = df['date'].dt.day_name()
            df['month'] = df['date'].dt.month

            # 创建透视表
            pivot = df.pivot_table(
                values='total_amount',
                index='month',
                columns='weekday',
                aggfunc='sum'
            )

            fig, ax = plt.subplots(figsize=(12, 6))
            sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax, cbar_kws={'label': '销售额'})
            ax.set_title('月度-星期销售热力图', fontsize=14, fontweight='bold')
            ax.set_xlabel('星期')
            ax.set_ylabel('月份')

            weekday_labels = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
            ax.set_xticklabels(weekday_labels)

            plt.tight_layout()
            filepath = self.save_figure('sales_heatmap.png', fig)
            plt.close(fig)
            return filepath
        except Exception as e:
            logger.error(f"绘制销售热力图出错: {e}")
            return ""

    def plot_category_analysis(self, category_data: pd.DataFrame) -> str:
        """品类分析图"""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            # 品类销售额占比饼图
            colors = plt.cm.Set3(np.linspace(0, 1, len(category_data)))
            axes[0].pie(category_data['total_revenue'], labels=category_data['category'],
                       autopct='%1.1f%%', colors=colors, startangle=90)
            axes[0].set_title('品类销售额占比', fontsize=14, fontweight='bold')

            # 品类销售额柱状图
            axes[1].barh(category_data['category'], category_data['total_revenue'], color='steelblue', alpha=0.7)
            axes[1].set_title('各品类销售额', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('销售额')
            axes[1].grid(True, alpha=0.3, axis='x')

            plt.tight_layout()
            filepath = self.save_figure('category_analysis.png', fig)
            plt.close(fig)
            return filepath
        except Exception as e:
            logger.error(f"绘制品类分析图出错: {e}")
            return ""

    def plot_customer_segmentation(self, rfm_data: pd.DataFrame) -> str:
        """客户分群散点图"""
        try:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            # RFM三维散点图
            scatter = axes[0].scatter(rfm_data['frequency'], rfm_data['monetary'],
                                     c=rfm_data['recency'], s=50, alpha=0.6, cmap='viridis_r')
            axes[0].set_xlabel('购买频次 (Frequency)')
            axes[0].set_ylabel('消费金额 (Monetary)')
            axes[0].set_title('RFM分析: 频次 vs 金额 (颜色=最近购买)', fontsize=12, fontweight='bold')
            plt.colorbar(scatter, ax=axes[0], label='最近购买天数 (Recency)')

            # 客户分层饼图
            segment_counts = rfm_data['customer_segment'].value_counts()
            colors = plt.cm.Set3(np.linspace(0, 1, len(segment_counts)))
            axes[1].pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%',
                       colors=colors, startangle=90)
            axes[1].set_title('客户价值分层', fontsize=12, fontweight='bold')

            # RFM评分分布
            axes[2].hist(rfm_data['rfm_score'], bins=10, color='steelblue', alpha=0.7, edgecolor='black')
            axes[2].set_xlabel('RFM总分')
            axes[2].set_ylabel('客户数量')
            axes[2].set_title('RFM评分分布', fontsize=12, fontweight='bold')
            axes[2].grid(True, alpha=0.3, axis='y')

            plt.tight_layout()
            filepath = self.save_figure('customer_segmentation.png', fig)
            plt.close(fig)
            return filepath
        except Exception as e:
            logger.error(f"绘制客户分群图出错: {e}")
            return ""

    def plot_customer_rfm_distribution(self, rfm_data: pd.DataFrame) -> str:
        """RFM分布图"""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

            # Recency分布
            axes[0, 0].hist(rfm_data['recency'], bins=30, color='skyblue', alpha=0.7, edgecolor='black')
            axes[0, 0].set_title('Recency (最近购买天数) 分布')
            axes[0, 0].set_xlabel('天数')
            axes[0, 0].set_ylabel('客户数')
            axes[0, 0].grid(True, alpha=0.3)

            # Frequency分布
            axes[0, 1].hist(rfm_data['frequency'], bins=30, color='lightgreen', alpha=0.7, edgecolor='black')
            axes[0, 1].set_title('Frequency (购买频次) 分布')
            axes[0, 1].set_xlabel('频次')
            axes[0, 1].set_ylabel('客户数')
            axes[0, 1].grid(True, alpha=0.3)

            # Monetary分布
            axes[0, 2].hist(rfm_data['monetary'], bins=30, color='salmon', alpha=0.7, edgecolor='black')
            axes[0, 2].set_title('Monetary (消费金额) 分布')
            axes[0, 2].set_xlabel('金额')
            axes[0, 2].set_ylabel('客户数')
            axes[0, 2].grid(True, alpha=0.3)

            # R Score分布
            axes[1, 0].bar(range(1, 6), [sum(rfm_data['r_score'] == i) for i in range(1, 6)],
                          color='steelblue', alpha=0.7)
            axes[1, 0].set_title('R Score 分布')
            axes[1, 0].set_xlabel('评分')
            axes[1, 0].set_ylabel('客户数')
            axes[1, 0].grid(True, alpha=0.3, axis='y')

            # F Score分布
            axes[1, 1].bar(range(1, 6), [sum(rfm_data['f_score'] == i) for i in range(1, 6)],
                          color='steelblue', alpha=0.7)
            axes[1, 1].set_title('F Score 分布')
            axes[1, 1].set_xlabel('评分')
            axes[1, 1].set_ylabel('客户数')
            axes[1, 1].grid(True, alpha=0.3, axis='y')

            # M Score分布
            axes[1, 2].bar(range(1, 6), [sum(rfm_data['m_score'] == i) for i in range(1, 6)],
                          color='steelblue', alpha=0.7)
            axes[1, 2].set_title('M Score 分布')
            axes[1, 2].set_xlabel('评分')
            axes[1, 2].set_ylabel('客户数')
            axes[1, 2].grid(True, alpha=0.3, axis='y')

            plt.tight_layout()
            filepath = self.save_figure('rfm_distribution.png', fig)
            plt.close(fig)
            return filepath
        except Exception as e:
            logger.error(f"绘制RFM分布图出错: {e}")
            return ""

    def plot_prediction_results(self, train_data: pd.DataFrame, test_data: pd.DataFrame,
                                forecast_data: pd.DataFrame, model_name: str) -> str:
        """预测结果图"""
        try:
            fig, axes = plt.subplots(2, 1, figsize=(14, 10))

            # 历史数据与预测
            axes[0].plot(train_data['date'], train_data['sales'], label='训练数据', color='blue', alpha=0.7)
            axes[0].plot(test_data['date'], test_data['sales'], label='测试数据', color='green', alpha=0.7)
            if not forecast_data.empty:
                axes[0].plot(forecast_data['date'], forecast_data['predicted_sales'],
                           label='未来预测', color='red', linestyle='--', linewidth=2)
            axes[0].set_title(f'销售预测 ({model_name})', fontsize=14, fontweight='bold')
            axes[0].set_xlabel('日期')
            axes[0].set_ylabel('销售额')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # 预测误差
            if 'predicted' in test_data.columns:
                error = test_data['sales'] - test_data['predicted']
                axes[1].plot(test_data['date'], error, color='red', alpha=0.7)
                axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                axes[1].fill_between(test_data['date'], error, 0, alpha=0.3, color='red')
                axes[1].set_title('预测误差', fontsize=14, fontweight='bold')
                axes[1].set_xlabel('日期')
                axes[1].set_ylabel('误差')
                axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            filepath = self.save_figure('prediction_results.png', fig)
            plt.close(fig)
            return filepath
        except Exception as e:
            logger.error(f"绘制预测结果图出错: {e}")
            return ""

    def plot_feature_importance(self, importance_data: pd.DataFrame, top_n: int = 15) -> str:
        """特征重要性图"""
        try:
            fig, ax = plt.subplots(figsize=(10, 8))

            top_features = importance_data.head(top_n)
            ax.barh(top_features['feature'], top_features['importance'], color='steelblue', alpha=0.7)
            ax.set_title(f'特征重要性 (Top {top_n})', fontsize=14, fontweight='bold')
            ax.set_xlabel('重要性')
            ax.set_ylabel('特征')
            ax.grid(True, alpha=0.3, axis='x')
            ax.invert_yaxis()

            plt.tight_layout()
            filepath = self.save_figure('feature_importance.png', fig)
            plt.close(fig)
            return filepath
        except Exception as e:
            logger.error(f"绘制特征重要性图出错: {e}")
            return ""

    def plot_model_comparison(self, metrics_data: pd.DataFrame) -> str:
        """模型对比图"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # R2对比
            axes[0, 0].bar(metrics_data['model'], metrics_data['r2'], color='steelblue', alpha=0.7)
            axes[0, 0].set_title('R² Score 对比', fontsize=12, fontweight='bold')
            axes[0, 0].set_ylabel('R²')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].grid(True, alpha=0.3, axis='y')

            # MAE对比
            axes[0, 1].bar(metrics_data['model'], metrics_data['mae'], color='coral', alpha=0.7)
            axes[0, 1].set_title('MAE 对比', fontsize=12, fontweight='bold')
            axes[0, 1].set_ylabel('MAE')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3, axis='y')

            # RMSE对比
            axes[1, 0].bar(metrics_data['model'], metrics_data['rmse'], color='lightgreen', alpha=0.7)
            axes[1, 0].set_title('RMSE 对比', fontsize=12, fontweight='bold')
            axes[1, 0].set_ylabel('RMSE')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3, axis='y')

            # MAPE对比
            axes[1, 1].bar(metrics_data['model'], metrics_data['mape'], color='gold', alpha=0.7)
            axes[1, 1].set_title('MAPE 对比', fontsize=12, fontweight='bold')
            axes[1, 1].set_ylabel('MAPE (%)')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3, axis='y')

            plt.tight_layout()
            filepath = self.save_figure('model_comparison.png', fig)
            plt.close(fig)
            return filepath
        except Exception as e:
            logger.error(f"绘制模型对比图出错: {e}")
            return ""

    def plot_channel_analysis(self, channel_data: pd.DataFrame) -> str:
        """渠道分析图"""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # 渠道销售额
            axes[0].bar(channel_data['channel'], channel_data['total_revenue'], color='steelblue', alpha=0.7)
            axes[0].set_title('各渠道销售额', fontsize=14, fontweight='bold')
            axes[0].set_xlabel('渠道')
            axes[0].set_ylabel('销售额')
            axes[0].grid(True, alpha=0.3, axis='y')

            # 渠道占比饼图
            axes[1].pie(channel_data['total_revenue'], labels=channel_data['channel'],
                       autopct='%1.1f%%', startangle=90)
            axes[1].set_title('渠道销售占比', fontsize=14, fontweight='bold')

            plt.tight_layout()
            filepath = self.save_figure('channel_analysis.png', fig)
            plt.close(fig)
            return filepath
        except Exception as e:
            logger.error(f"绘制渠道分析图出错: {e}")
            return ""

    def plot_weekday_analysis(self, weekday_data: pd.DataFrame) -> str:
        """星期分析图"""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            weekday_labels = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']

            # 星期销售额
            axes[0].bar(range(7), [weekday_data[weekday_data['weekday'] == i]['total_revenue'].sum()
                                  if i in weekday_data['weekday'].values else 0 for i in range(7)],
                       color='steelblue', alpha=0.7)
            axes[0].set_title('星期销售额分布', fontsize=14, fontweight='bold')
            axes[0].set_xlabel('星期')
            axes[0].set_ylabel('销售额')
            axes[0].set_xticks(range(7))
            axes[0].set_xticklabels(weekday_labels)
            axes[0].grid(True, alpha=0.3, axis='y')

            # 星期交易量
            axes[1].bar(range(7), [weekday_data[weekday_data['weekday'] == i]['transactions'].sum()
                                  if i in weekday_data['weekday'].values else 0 for i in range(7)],
                       color='coral', alpha=0.7)
            axes[1].set_title('星期交易量分布', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('星期')
            axes[1].set_ylabel('交易量')
            axes[1].set_xticks(range(7))
            axes[1].set_xticklabels(weekday_labels)
            axes[1].grid(True, alpha=0.3, axis='y')

            plt.tight_layout()
            filepath = self.save_figure('weekday_analysis.png', fig)
            plt.close(fig)
            return filepath
        except Exception as e:
            logger.error(f"绘制星期分析图出错: {e}")
            return ""

    def plot_top_products(self, products_data: pd.DataFrame, top_n: int = 10) -> str:
        """热销商品图"""
        try:
            fig, ax = plt.subplots(figsize=(12, 6))

            top_products = products_data.head(top_n)
            col_name = 'product_name' if 'product_name' in top_products.columns else 'product_id'

            ax.barh(top_products[col_name], top_products['revenue'], color='steelblue', alpha=0.7)
            ax.set_title(f'Top {top_n} 热销商品', fontsize=14, fontweight='bold')
            ax.set_xlabel('销售额')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3, axis='x')

            plt.tight_layout()
            filepath = self.save_figure('top_products.png', fig)
            plt.close(fig)
            return filepath
        except Exception as e:
            logger.error(f"绘制热销商品图出错: {e}")
            return ""

    def plot_geographic_analysis(self, geo_data: pd.DataFrame, top_n: int = 10) -> str:
        """地理分析图"""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            top_cities = geo_data.head(top_n)

            # 城市销售额
            axes[0].barh(top_cities['city'], top_cities['total_revenue'], color='steelblue', alpha=0.7)
            axes[0].set_title(f'Top {top_n} 城市销售额', fontsize=14, fontweight='bold')
            axes[0].set_xlabel('销售额')
            axes[0].invert_yaxis()
            axes[0].grid(True, alpha=0.3, axis='x')

            # 城市客户数
            axes[1].barh(top_cities['city'], top_cities['unique_customers'], color='coral', alpha=0.7)
            axes[1].set_title(f'Top {top_n} 城市客户数', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('客户数')
            axes[1].invert_yaxis()
            axes[1].grid(True, alpha=0.3, axis='x')

            plt.tight_layout()
            filepath = self.save_figure('geographic_analysis.png', fig)
            plt.close(fig)
            return filepath
        except Exception as e:
            logger.error(f"绘制地理分析图出错: {e}")
            return ""

    def create_summary_dashboard(self, kpis: Dict, output_filename: str = 'summary_dashboard.png') -> str:
        """创建汇总仪表板"""
        try:
            fig = plt.figure(figsize=(16, 10))
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

            # 标题
            fig.suptitle('电商销售分析仪表板', fontsize=20, fontweight='bold', y=0.98)

            # KPI指标
            ax_kpi = fig.add_subplot(gs[0, :])
            ax_kpi.axis('off')

            kpi_text = f"""
            总销售额: ¥{kpis.get('total_revenue', 0):,.2f}  |  
            总订单数: {kpis.get('total_transactions', 0):,}  |  
            客单价: ¥{kpis.get('avg_order_value', 0):.2f}  |  
            客户数: {kpis.get('unique_customers', 0):,}  |  
            商品数: {kpis.get('unique_products', 0):,}
            """
            ax_kpi.text(0.5, 0.5, kpi_text, ha='center', va='center', fontsize=14,
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

            plt.tight_layout()
            filepath = self.save_figure(output_filename, fig)
            plt.close(fig)
            return filepath
        except Exception as e:
            logger.error(f"创建汇总仪表板出错: {e}")
            return ""

    def get_all_figures(self) -> List[str]:
        """获取所有生成的图表路径"""
        return [os.path.join(self.output_dir, f) for f in os.listdir(self.output_dir)
                if f.endswith('.png')]
