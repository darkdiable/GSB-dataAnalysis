import sys
import os
import logging
import traceback
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from analysis import SalesAnalyzer
from visualization import Visualizer
from prediction import SalesPredictor
from report_generator import ReportGenerator

os.makedirs('../output', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../output/analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("电商销售数据分析系统启动")
    logger.info("=" * 60)
    
    try:
        logger.info("\n【1/7】数据加载模块")
        loader = DataLoader(data_dir="../data")
        loader.load_all_data()
        merged_df = loader.merge_data()
        data_summary = loader.data_summary()
        
        logger.info("\n【2/7】特征工程模块")
        fe = FeatureEngineer()
        feature_results = fe.transform_all(merged_df, loader.customers_df)
        enhanced_df = feature_results['enhanced_data']
        
        logger.info("\n【3/7】多维度分析模块")
        analyzer = SalesAnalyzer()
        analysis_results = analyzer.run_full_analysis(
            enhanced_df,
            feature_results['customer_features'],
            feature_results['product_features']
        )
        
        logger.info("\n【4/7】销售预测模块")
        predictor = SalesPredictor()
        prediction_results = predictor.run_prediction_pipeline(enhanced_df)
        
        logger.info("\n【5/7】可视化模块")
        viz = Visualizer(output_dir="../output")
        chart_paths = viz.generate_all_charts(
            analysis_results,
            feature_results,
            prediction_results['actual_vs_predicted']
        )
        
        logger.info("\n【6/7】生成PDF报告")
        reporter = ReportGenerator(output_dir="../output")
        report_path = reporter.generate_full_report(
            data_summary,
            analysis_results,
            prediction_results,
            chart_paths
        )
        
        logger.info("\n【7/7】导出分析结果")
        export_files = reporter.export_analysis_results(
            analysis_results,
            prediction_results
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info("\n" + "=" * 60)
        logger.info("分析完成！")
        logger.info(f"总耗时: {duration:.2f} 秒")
        logger.info("=" * 60)
        logger.info(f"\n生成的文件:")
        logger.info(f"  PDF报告: {report_path}")
        logger.info(f"\n  图表文件:")
        for name, path in chart_paths.items():
            logger.info(f"    - {name}: {path}")
        logger.info(f"\n  分析结果:")
        for name, path in export_files.items():
            logger.info(f"    - {name}: {path}")
        logger.info(f"\n  日志文件: ../output/analysis.log")
        
        print("\n" + "=" * 60)
        print("✅ 电商销售数据分析完成！")
        print(f"📄 PDF报告: {report_path}")
        print(f"📊 共生成 {len(chart_paths)} 个可视化图表")
        print(f"📈 预测模型R²: {prediction_results['metrics']['test_r2']:.4f}")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"\n❌ 分析过程中发生错误: {str(e)}")
        logger.error(traceback.format_exc())
        print(f"\n❌ 分析失败: {str(e)}")
        print("详情请查看日志文件: ../output/analysis.log")
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
