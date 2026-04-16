"""
电商销售分析系统主程序
整合数据加载、特征工程、分析、预测和报告生成
"""
import pandas as pd
import numpy as np
import os
import sys
import logging
from datetime import datetime

from data_generator import DataGenerator
from data_loader import DataLoader, load_and_prepare_data
from feature_engineering import FeatureEngineer
from sales_trend import SalesTrendAnalyzer
from customer_value import CustomerValueAnalyzer
from product_association import ProductAssociationAnalyzer
from sales_prediction import SalesPredictor
from visualization import Visualizer
from report_generator import PDFReportGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('output/analysis.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


class SalesAnalysisSystem:
    """电商销售分析系统"""

    def __init__(self, data_dir: str = 'data', output_dir: str = 'output'):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        self.raw_data = None
        self.processed_data = None
        self.analysis_results = {}
        self.visualization_paths = {}

    def generate_sample_data(self):
        """生成示例数据"""
        logger.info("=" * 60)
        logger.info("步骤 1: 生成示例数据")
        logger.info("=" * 60)

        try:
            generator = DataGenerator(seed=42)
            generator.generate_all_data(self.data_dir)
            logger.info("示例数据生成完成\n")
        except Exception as e:
            logger.error(f"生成示例数据时出错: {e}")
            raise

    def load_data(self):
        """加载数据"""
        logger.info("=" * 60)
        logger.info("步骤 2: 加载数据")
        logger.info("=" * 60)

        try:
            self.raw_data, data_dict = load_and_prepare_data(self.data_dir)

            if self.raw_data is None or self.raw_data.empty:
                raise ValueError("数据加载失败，请检查数据文件")

            logger.info(f"数据加载完成，共 {len(self.raw_data)} 条记录\n")
        except Exception as e:
            logger.error(f"加载数据时出错: {e}")
            raise

    def feature_engineering(self):
        """特征工程"""
        logger.info("=" * 60)
        logger.info("步骤 3: 特征工程")
        logger.info("=" * 60)

        try:
            engineer = FeatureEngineer(self.raw_data)
            self.processed_data = engineer.build_all_features()

            logger.info(f"特征工程完成，当前数据维度: {self.processed_data.shape}")
            logger.info(f"特征列表: {list(self.processed_data.columns)}\n")
        except Exception as e:
            logger.error(f"特征工程时出错: {e}")
            raise

    def analyze_sales_trend(self):
        """销售趋势分析"""
        logger.info("=" * 60)
        logger.info("步骤 4: 销售趋势分析")
        logger.info("=" * 60)

        try:
            analyzer = SalesTrendAnalyzer(self.processed_data)
            results = analyzer.run_all_analysis()
            self.analysis_results['sales_trend'] = results

            logger.info("销售趋势分析完成")
            if 'kpis' in results:
                kpis = results['kpis']
                logger.info(f"总销售额: ¥{kpis.get('total_revenue', 0):,.2f}")
                logger.info(f"总订单数: {kpis.get('total_transactions', 0):,}")
                logger.info(f"平均订单金额: ¥{kpis.get('avg_order_value', 0):.2f}")
                logger.info(f"客户数: {kpis.get('unique_customers', 0):,}\n")
        except Exception as e:
            logger.error(f"销售趋势分析时出错: {e}")
            raise

    def analyze_customer_value(self):
        """客户价值分析"""
        logger.info("=" * 60)
        logger.info("步骤 5: 客户价值分析")
        logger.info("=" * 60)

        try:
            analyzer = CustomerValueAnalyzer(self.processed_data)
            results = analyzer.run_all_analysis()
            self.analysis_results['customer_value'] = results

            logger.info("客户价值分析完成")
            if 'rfm' in results:
                rfm = results['rfm']
                logger.info(f"RFM分析完成，共分析 {len(rfm)} 个客户")
            if 'segment_stats' in results:
                segments = results['segment_stats']
                logger.info("客户分层统计:")
                for _, row in segments.iterrows():
                    logger.info(f"  {row['customer_segment']}: {row['customer_count']}人 ({row['customer_share']:.1f}%)")
            logger.info("")
        except Exception as e:
            logger.error(f"客户价值分析时出错: {e}")
            raise

    def analyze_product_association(self):
        """商品关联分析"""
        logger.info("=" * 60)
        logger.info("步骤 6: 商品关联分析")
        logger.info("=" * 60)

        try:
            analyzer = ProductAssociationAnalyzer(self.processed_data)
            results = analyzer.run_all_analysis()
            self.analysis_results['product_association'] = results

            logger.info("商品关联分析完成")
            if 'association_rules' in results:
                rules = results['association_rules']
                logger.info(f"发现 {len(rules)} 条关联规则")
            if 'category_performance' in results:
                cat_perf = results['category_performance']
                logger.info("品类表现:")
                for _, row in cat_perf.head(5).iterrows():
                    logger.info(f"  {row['category']}: ¥{row['total_revenue']:,.2f} ({row['revenue_share']:.1f}%)")
            logger.info("")
        except Exception as e:
            logger.error(f"商品关联分析时出错: {e}")
            raise

    def predict_sales(self):
        """销售预测"""
        logger.info("=" * 60)
        logger.info("步骤 7: 销售预测")
        logger.info("=" * 60)

        try:
            predictor = SalesPredictor()
            results = predictor.run_prediction_pipeline(
                self.processed_data,
                target_col='total_amount',
                freq='D',
                test_size=0.2
            )
            self.analysis_results['prediction'] = results

            if results:
                logger.info("销售预测完成")
                if 'metrics' in results:
                    metrics = results['metrics']
                    logger.info("模型性能:")
                    for _, row in metrics.iterrows():
                        logger.info(f"  {row['model']}: R²={row['r2']:.4f}, MAE={row['mae']:.2f}")
                if 'best_model' in results:
                    logger.info(f"最佳模型: {results['best_model']}")
            logger.info("")
        except Exception as e:
            logger.error(f"销售预测时出错: {e}")
            logger.warning("跳过预测分析，继续生成其他报告内容")
            self.analysis_results['prediction'] = {}

    def create_visualizations(self):
        """创建可视化图表"""
        logger.info("=" * 60)
        logger.info("步骤 8: 创建可视化图表")
        logger.info("=" * 60)

        try:
            viz = Visualizer(self.output_dir)

            # 销售趋势图
            if 'sales_trend' in self.analysis_results:
                trend = self.analysis_results['sales_trend']
                if 'daily_trend' in trend and 'monthly_trend' in trend:
                    path = viz.plot_sales_trend(trend['daily_trend'], trend['monthly_trend'])
                    if path:
                        self.visualization_paths['sales_trend'] = path

                # 销售热力图
                path = viz.plot_sales_heatmap(self.processed_data)
                if path:
                    self.visualization_paths['sales_heatmap'] = path

                # 星期分析
                if 'weekday_analysis' in trend:
                    path = viz.plot_weekday_analysis(trend['weekday_analysis'])
                    if path:
                        self.visualization_paths['weekday_analysis'] = path

            # 客户分群图
            if 'customer_value' in self.analysis_results:
                customer = self.analysis_results['customer_value']
                if 'rfm' in customer:
                    path = viz.plot_customer_segmentation(customer['rfm'])
                    if path:
                        self.visualization_paths['customer_segmentation'] = path

                    path = viz.plot_customer_rfm_distribution(customer['rfm'])
                    if path:
                        self.visualization_paths['rfm_distribution'] = path

                # 地理分析
                if 'geographic' in customer:
                    path = viz.plot_geographic_analysis(customer['geographic'])
                    if path:
                        self.visualization_paths['geographic_analysis'] = path

            # 品类分析
            if 'product_association' in self.analysis_results:
                product = self.analysis_results['product_association']
                if 'category_performance' in product:
                    path = viz.plot_category_analysis(product['category_performance'])
                    if path:
                        self.visualization_paths['category_analysis'] = path

                if 'top_products' in product:
                    path = viz.plot_top_products(product['top_products'])
                    if path:
                        self.visualization_paths['top_products'] = path

            # 渠道分析
            if 'sales_trend' in self.analysis_results:
                trend = self.analysis_results['sales_trend']
                if 'channel_analysis' in trend:
                    path = viz.plot_channel_analysis(trend['channel_analysis'])
                    if path:
                        self.visualization_paths['channel_analysis'] = path

            # 预测结果图
            if 'prediction' in self.analysis_results:
                pred = self.analysis_results['prediction']
                if pred and 'ts_data' in pred and 'test' in pred:
                    path = viz.plot_prediction_results(
                        pred['train'],
                        pred['test'],
                        pred.get('forecast', pd.DataFrame()),
                        pred.get('best_model', 'unknown')
                    )
                    if path:
                        self.visualization_paths['prediction_results'] = path

                if pred and 'metrics' in pred and not pred['metrics'].empty:
                    path = viz.plot_model_comparison(pred['metrics'])
                    if path:
                        self.visualization_paths['model_comparison'] = path

                if pred and 'feature_importance' in pred and not pred['feature_importance'].empty:
                    path = viz.plot_feature_importance(pred['feature_importance'])
                    if path:
                        self.visualization_paths['feature_importance'] = path

            logger.info(f"可视化图表创建完成，共生成 {len(self.visualization_paths)} 个图表\n")
        except Exception as e:
            logger.error(f"创建可视化图表时出错: {e}")
            raise

    def generate_report(self):
        """生成PDF报告"""
        logger.info("=" * 60)
        logger.info("步骤 9: 生成PDF报告")
        logger.info("=" * 60)

        try:
            # 准备报告数据
            report_data = {
                'kpis': self.analysis_results.get('sales_trend', {}).get('kpis', {}),
                'sales_trend': self.analysis_results.get('sales_trend', {}),
                'customer_value': self.analysis_results.get('customer_value', {}),
                'product_association': self.analysis_results.get('product_association', {}),
                'prediction': self.analysis_results.get('prediction', {}),
                'images': self.visualization_paths
            }

            # 生成PDF
            report_path = os.path.join(self.output_dir, 'sales_analysis_report.pdf')
            generator = PDFReportGenerator(report_path)
            generator.generate_report(report_data)

            logger.info(f"PDF报告生成完成: {report_path}\n")
        except Exception as e:
            logger.error(f"生成PDF报告时出错: {e}")
            raise

    def save_results(self):
        """保存分析结果到CSV"""
        logger.info("=" * 60)
        logger.info("步骤 10: 保存分析结果")
        logger.info("=" * 60)

        try:
            # 保存KPI
            if 'sales_trend' in self.analysis_results and 'kpis' in self.analysis_results['sales_trend']:
                kpis_df = pd.DataFrame([self.analysis_results['sales_trend']['kpis']])
                kpis_df.to_csv(os.path.join(self.output_dir, 'kpis.csv'), index=False, encoding='utf-8-sig')

            # 保存销售趋势
            if 'sales_trend' in self.analysis_results:
                trend = self.analysis_results['sales_trend']
                for key in ['daily_trend', 'monthly_trend', 'weekday_analysis']:
                    if key in trend:
                        trend[key].to_csv(os.path.join(self.output_dir, f'{key}.csv'), index=False, encoding='utf-8-sig')

            # 保存RFM分析
            if 'customer_value' in self.analysis_results:
                customer = self.analysis_results['customer_value']
                for key in ['rfm', 'segment_stats', 'clv', 'behavior']:
                    if key in customer and isinstance(customer[key], pd.DataFrame):
                        customer[key].to_csv(os.path.join(self.output_dir, f'{key}.csv'), index=False, encoding='utf-8-sig')

            # 保存商品分析
            if 'product_association' in self.analysis_results:
                product = self.analysis_results['product_association']
                for key in ['category_performance', 'top_products', 'association_rules']:
                    if key in product and isinstance(product[key], pd.DataFrame):
                        product[key].to_csv(os.path.join(self.output_dir, f'{key}.csv'), index=False, encoding='utf-8-sig')

            # 保存预测结果
            if 'prediction' in self.analysis_results:
                pred = self.analysis_results['prediction']
                if pred:
                    for key in ['forecast', 'metrics', 'feature_importance']:
                        if key in pred and isinstance(pred[key], pd.DataFrame):
                            pred[key].to_csv(os.path.join(self.output_dir, f'{key}.csv'), index=False, encoding='utf-8-sig')

            logger.info("分析结果保存完成\n")
        except Exception as e:
            logger.error(f"保存分析结果时出错: {e}")

    def run(self):
        """运行完整分析流程"""
        start_time = datetime.now()
        logger.info("\n" + "=" * 60)
        logger.info("电商销售分析系统启动")
        logger.info("=" * 60 + "\n")

        try:
            # 执行各步骤
            self.generate_sample_data()
            self.load_data()
            self.feature_engineering()
            self.analyze_sales_trend()
            self.analyze_customer_value()
            self.analyze_product_association()
            self.predict_sales()
            self.create_visualizations()
            self.generate_report()
            self.save_results()

            # 计算耗时
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            logger.info("=" * 60)
            logger.info("分析完成!")
            logger.info(f"总耗时: {duration:.2f} 秒")
            logger.info(f"输出目录: {os.path.abspath(self.output_dir)}")
            logger.info("=" * 60 + "\n")

            return True

        except Exception as e:
            logger.error(f"分析过程中出现错误: {e}")
            return False


def main():
    """主函数"""
    system = SalesAnalysisSystem(
        data_dir='data',
        output_dir='output'
    )
    success = system.run()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
