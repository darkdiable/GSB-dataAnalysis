import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, Any
from datetime import datetime

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReportGenerator:
    def __init__(self, output_dir: str = "../output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def _add_title_page(self, pdf: PdfPages, summary: Dict[str, Any]):
        fig = plt.figure(figsize=(12, 16))
        
        fig.text(0.5, 0.85, '电商销售数据分析报告', fontsize=28, ha='center', weight='bold')
        fig.text(0.5, 0.78, 'E-Commerce Sales Data Analysis Report', fontsize=16, ha='center', color='gray')
        
        fig.text(0.5, 0.70, f'生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', fontsize=12, ha='center')
        
        fig.text(0.15, 0.60, '数据概览', fontsize=18, weight='bold')
        
        overview_text = f"""
        分析周期: {summary['date_range'][0].strftime('%Y-%m-%d')} 至 {summary['date_range'][1].strftime('%Y-%m-%d')}
        
        总订单数: {summary['total_orders']:,}
        总客户数: {summary['total_customers']:,}
        总商品数: {summary['total_products']:,}
        总销售额: ¥{summary['total_revenue']:,.2f}
        平均客单价: ¥{summary['avg_order_value']:,.2f}
        """
        fig.text(0.15, 0.45, overview_text, fontsize=14, linespacing=2)
        
        fig.text(0.15, 0.30, '报告目录', fontsize=18, weight='bold')
        
        toc = """
        1. 销售趋势分析
        2. 区域销售分析
        3. 客户价值分析 (RFM)
        4. 商品表现分析
        5. 销售预测
        6. 业务洞察与建议
        """
        fig.text(0.15, 0.15, toc, fontsize=14, linespacing=2)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
    def _add_sales_trend_page(self, pdf: PdfPages, analysis_results: Dict[str, Any], chart_paths: Dict[str, str]):
        fig = plt.figure(figsize=(14, 18))
        
        fig.text(0.5, 0.97, '1. 销售趋势分析', fontsize=20, ha='center', weight='bold')
        
        if 'sales_trend' in chart_paths and os.path.exists(chart_paths['sales_trend']):
            img = plt.imread(chart_paths['sales_trend'])
            ax = fig.add_axes([0.1, 0.55, 0.8, 0.38])
            ax.imshow(img)
            ax.axis('off')
        
        summary = analysis_results['sales_trend']['summary']
        insight_text = f"""
        关键指标:
        
        • 分析期内整体增长率: {summary['total_period_growth']:.1f}%
        • 月均复合增长率: {summary['avg_monthly_growth']:.1f}%
        • 销售峰值月份: {summary['peak_month']}月
        • 最佳销售日: {summary['best_weekday']}
        
        年度销售数据:
        """
        
        fig.text(0.1, 0.50, insight_text, fontsize=12, linespacing=1.8)
        
        sales_by_year = analysis_results['sales_trend']['sales_by_year']
        table_text = sales_by_year.to_string()
        fig.text(0.1, 0.35, table_text, fontsize=11, family='monospace')
        
        weekday_text = "\n\n周度销售趋势:"
        fig.text(0.1, 0.28, weekday_text, fontsize=12, weight='bold')
        weekday_table = analysis_results['sales_trend']['sales_by_weekday'].to_string()
        fig.text(0.1, 0.20, weekday_table, fontsize=11, family='monospace')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
    def _add_region_analysis_page(self, pdf: PdfPages, analysis_results: Dict[str, Any], chart_paths: Dict[str, str]):
        fig = plt.figure(figsize=(14, 18))
        
        fig.text(0.5, 0.97, '2. 区域销售分析', fontsize=20, ha='center', weight='bold')
        
        if 'sales_heatmap' in chart_paths and os.path.exists(chart_paths['sales_heatmap']):
            img = plt.imread(chart_paths['sales_heatmap'])
            ax = fig.add_axes([0.1, 0.60, 0.8, 0.33])
            ax.imshow(img)
            ax.axis('off')
        
        if 'region_sales' in chart_paths and os.path.exists(chart_paths['region_sales']):
            img = plt.imread(chart_paths['region_sales'])
            ax = fig.add_axes([0.1, 0.12, 0.8, 0.43])
            ax.imshow(img)
            ax.axis('off')
        
        top_region = analysis_results['region_perf']['top_region']
        fig.text(0.1, 0.08, f'• 核心销售区域: {top_region}地区贡献最大市场份额', fontsize=12)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
    def _add_customer_analysis_page(self, pdf: PdfPages, analysis_results: Dict[str, Any], chart_paths: Dict[str, str]):
        fig = plt.figure(figsize=(14, 18))
        
        fig.text(0.5, 0.97, '3. 客户价值分析 (RFM模型)', fontsize=20, ha='center', weight='bold')
        
        if 'customer_segmentation' in chart_paths and os.path.exists(chart_paths['customer_segmentation']):
            img = plt.imread(chart_paths['customer_segmentation'])
            ax = fig.add_axes([0.05, 0.55, 0.9, 0.38])
            ax.imshow(img)
            ax.axis('off')
        
        if 'rfm_distribution' in chart_paths and os.path.exists(chart_paths['rfm_distribution']):
            img = plt.imread(chart_paths['rfm_distribution'])
            ax = fig.add_axes([0.05, 0.12, 0.9, 0.40])
            ax.imshow(img)
            ax.axis('off')
        
        segment_dist = analysis_results['customer_value']['segment_distribution']
        fig.text(0.05, 0.08, f'客户分群分布: {segment_dist.to_dict()}', fontsize=11)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
    def _add_product_analysis_page(self, pdf: PdfPages, analysis_results: Dict[str, Any], chart_paths: Dict[str, str]):
        fig = plt.figure(figsize=(14, 18))
        
        fig.text(0.5, 0.97, '4. 商品表现分析', fontsize=20, ha='center', weight='bold')
        
        if 'category_distribution' in chart_paths and os.path.exists(chart_paths['category_distribution']):
            img = plt.imread(chart_paths['category_distribution'])
            ax = fig.add_axes([0.05, 0.52, 0.9, 0.40])
            ax.imshow(img)
            ax.axis('off')
        
        category_contribution = analysis_results['product_perf']['category_contribution']
        fig.text(0.05, 0.48, '品类销售额贡献度:', fontsize=12, weight='bold')
        
        y_pos = 0.45
        for cat, pct in category_contribution.items():
            fig.text(0.05, y_pos, f'• {cat}: {pct}%', fontsize=11)
            y_pos -= 0.025
        
        fig.text(0.05, y_pos - 0.02, 'TOP 10 畅销商品:', fontsize=12, weight='bold')
        y_pos -= 0.05
        
        top_products = analysis_results['product_perf']['top_products'].head(5)
        for idx, (_, row) in enumerate(top_products.iterrows()):
            fig.text(0.05, y_pos, f'{idx+1}. {row.name} - {row["category"]} - ¥{row["total_revenue"]:,.2f}', fontsize=10)
            y_pos -= 0.02
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
    def _add_prediction_page(self, pdf: PdfPages, prediction_results: Dict[str, Any], chart_paths: Dict[str, str]):
        fig = plt.figure(figsize=(14, 18))
        
        fig.text(0.5, 0.97, '5. 销售预测', fontsize=20, ha='center', weight='bold')
        
        if 'forecast' in chart_paths and os.path.exists(chart_paths['forecast']):
            img = plt.imread(chart_paths['forecast'])
            ax = fig.add_axes([0.05, 0.60, 0.9, 0.33])
            ax.imshow(img)
            ax.axis('off')
        
        metrics = prediction_results['metrics']
        metrics_text = f"""
        模型性能指标 (随机森林回归):
        
        • 训练集 RMSE: ¥{metrics['train_rmse']:,.2f}
        • 测试集 RMSE: ¥{metrics['test_rmse']:,.2f}
        • 训练集 R²: {metrics['train_r2']:.4f}
        • 测试集 R²: {metrics['test_r2']:.4f}
        """
        fig.text(0.05, 0.55, metrics_text, fontsize=12, linespacing=1.8)
        
        fig.text(0.05, 0.45, '未来3个月销售预测:', fontsize=14, weight='bold')
        
        forecast = prediction_results['forecast']
        table_text = forecast.to_string(index=False)
        fig.text(0.05, 0.38, table_text, fontsize=12, family='monospace')
        
        fig.text(0.05, 0.30, '特征重要性排序:', fontsize=12, weight='bold')
        fi = prediction_results['feature_importance'].head(5)
        for idx, (_, row) in enumerate(fi.iterrows()):
            fig.text(0.05, 0.26 - idx * 0.03, 
                    f'{idx+1}. {row["feature"]}: {row["importance"]:.4f}', fontsize=11)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
    def _add_insights_page(self, pdf: PdfPages, insights: Dict[str, str]):
        fig = plt.figure(figsize=(12, 16))
        
        fig.text(0.5, 0.90, '6. 业务洞察与建议', fontsize=22, ha='center', weight='bold')
        
        y_pos = 0.80
        for area, insight in insights.items():
            fig.text(0.1, y_pos, f'{area}:', fontsize=16, weight='bold')
            y_pos -= 0.05
            fig.text(0.1, y_pos, f'   {insight}', fontsize=13, linespacing=1.5)
            y_pos -= 0.08
        
        fig.text(0.1, y_pos, '运营建议:', fontsize=16, weight='bold')
        y_pos -= 0.05
        
        recommendations = """
        1. 客户分层运营:
           - 对重要价值客户提供专属客服和定制优惠
           - 对流失客户开展召回营销
           - 对新客户提供首购激励促进复购
        
        2. 区域优化:
           - 强化核心区域市场渗透
           - 针对落后区域制定专项拓展计划
        
        3. 商品策略:
           - 重点推广高贡献度品类
           - 利用商品关联规则开展捆绑销售
           - 优化爆款商品供应链
        
        4. 时间策略:
           - 在销售峰值月份提前备货
           - 在最佳销售日加大营销投入
        """
        fig.text(0.1, y_pos, recommendations, fontsize=13, linespacing=1.8)
        
        fig.text(0.5, 0.05, '- 报告结束 -', fontsize=12, ha='center', color='gray')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
    def generate_full_report(self, data_summary: Dict[str, Any], analysis_results: Dict[str, Any],
                            prediction_results: Dict[str, Any], chart_paths: Dict[str, str],
                            filename: str = "电商销售分析报告.pdf") -> str:
        try:
            logger.info("开始生成PDF分析报告...")
            
            filepath = os.path.join(self.output_dir, filename)
            
            with PdfPages(filepath) as pdf:
                self._add_title_page(pdf, data_summary)
                self._add_sales_trend_page(pdf, analysis_results, chart_paths)
                self._add_region_analysis_page(pdf, analysis_results, chart_paths)
                self._add_customer_analysis_page(pdf, analysis_results, chart_paths)
                self._add_product_analysis_page(pdf, analysis_results, chart_paths)
                self._add_prediction_page(pdf, prediction_results, chart_paths)
                self._add_insights_page(pdf, analysis_results['insights'])
            
            logger.info(f"PDF报告生成完成: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"生成PDF报告失败: {str(e)}")
            raise
            
    def export_analysis_results(self, analysis_results: Dict[str, Any], 
                               prediction_results: Dict[str, Any]) -> Dict[str, str]:
        try:
            logger.info("导出分析结果为CSV...")
            
            output_files = {}
            
            customer_features_path = os.path.join(self.output_dir, "customer_features.csv")
            analysis_results['customer_value']['segment_value'].to_csv(customer_features_path, encoding='utf-8-sig')
            output_files['customer_features'] = customer_features_path
            
            category_sales_path = os.path.join(self.output_dir, "category_sales.csv")
            analysis_results['product_perf']['category_sales'].to_csv(category_sales_path, encoding='utf-8-sig')
            output_files['category_sales'] = category_sales_path
            
            region_sales_path = os.path.join(self.output_dir, "region_sales.csv")
            analysis_results['region_perf']['region_sales'].to_csv(region_sales_path, encoding='utf-8-sig')
            output_files['region_sales'] = region_sales_path
            
            forecast_path = os.path.join(self.output_dir, "sales_forecast.csv")
            prediction_results['forecast'].to_csv(forecast_path, index=False, encoding='utf-8-sig')
            output_files['forecast'] = forecast_path
            
            insights_path = os.path.join(self.output_dir, "business_insights.txt")
            with open(insights_path, 'w', encoding='utf-8') as f:
                for k, v in analysis_results['insights'].items():
                    f.write(f"{k}: {v}\n")
            output_files['insights'] = insights_path
            
            logger.info(f"分析结果导出完成，共 {len(output_files)} 个文件")
            return output_files
            
        except Exception as e:
            logger.error(f"导出分析结果失败: {str(e)}")
            raise

if __name__ == '__main__':
    print("报告生成模块加载成功")
