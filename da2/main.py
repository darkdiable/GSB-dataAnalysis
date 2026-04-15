import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_generator import DataGenerator
from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.sales_trend_analyzer import SalesTrendAnalyzer
from src.customer_value_analyzer import CustomerValueAnalyzer
from src.product_association_analyzer import ProductAssociationAnalyzer
from src.visualizer import Visualizer
from src.sales_predictor import SalesPredictor
from src.pdf_report_generator import PDFReportGenerator


def main():
    print("=" * 60)
    print("电商销售分析系统")
    print("=" * 60)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    output_dir = os.path.join(base_dir, 'output')
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        print("\n[步骤1] 检查/生成数据...")
        if not os.path.exists(os.path.join(data_dir, 'customers.csv')):
            print("数据文件不存在，正在生成示例数据...")
            generator = DataGenerator(seed=42)
            generator.generate_all_data(output_dir=data_dir, n_customers=500, n_products=100, n_orders=5000)
        else:
            print("数据文件已存在，跳过数据生成。")
        
        print("\n[步骤2] 加载数据...")
        loader = DataLoader(data_dir=data_dir)
        data = loader.load_all_data()
        
        issues = loader.validate_data()
        for table, issue_list in issues.items():
            if issue_list:
                print(f"警告 - {table}: {issue_list}")
        
        print("\n[步骤3] 特征工程...")
        engineer = FeatureEngineer(
            orders_df=data['orders'],
            customers_df=data['customers'],
            products_df=data['products']
        )
        features = engineer.get_all_features()
        
        print("\n[步骤4] 销售趋势分析...")
        trend_analyzer = SalesTrendAnalyzer(orders_df=data['orders'])
        trend_results = trend_analyzer.get_all_analysis()
        
        print("\n[步骤5] 客户价值分析...")
        customer_analyzer = CustomerValueAnalyzer(
            orders_df=data['orders'],
            customers_df=data['customers']
        )
        customer_results = customer_analyzer.get_all_analysis()
        
        print("\n[步骤6] 商品关联分析...")
        product_analyzer = ProductAssociationAnalyzer(
            orders_df=data['orders'],
            products_df=data['products']
        )
        product_results = product_analyzer.get_all_analysis()
        
        print("\n[步骤7] 销售预测模型...")
        daily_sales = trend_results['daily_trend'][['date', 'revenue']].copy()
        daily_sales.columns = ['date', 'daily_revenue']
        
        predictor = SalesPredictor(daily_sales=daily_sales)
        
        print("比较不同模型...")
        model_comparison = predictor.compare_models()
        print("\n模型性能对比:")
        print(model_comparison.to_string())
        
        print("\n训练最优模型...")
        best_model = model_comparison.iloc[0]['model']
        model_type_map = {
            'LinearRegression': 'linear',
            'Ridge': 'ridge',
            'RandomForest': 'random_forest',
            'GradientBoosting': 'gradient_boosting'
        }
        metrics = predictor.train_model(model_type=model_type_map.get(best_model, 'random_forest'))
        print(f"\n模型训练完成 - 测试集 R²: {metrics['test']['r2']:.4f}")
        
        print("\n生成未来30天预测...")
        predictions = predictor.predict_future(days=30)
        pred_summary = predictor.get_prediction_summary(predictions)
        print(f"预测总销售额: ¥{pred_summary['total_predicted_revenue']:,.2f}")
        
        print("\n[步骤8] 生成可视化图表...")
        visualizer = Visualizer(output_dir=output_dir)
        
        print("  - 生成销售热力图...")
        visualizer.plot_sales_heatmap(trend_results['daily_trend'])
        
        print("  - 生成客户分群散点图...")
        visualizer.plot_customer_scatter(customer_results['rfm_analysis'])
        
        print("  - 生成时间序列图...")
        visualizer.plot_time_series(
            trend_results['daily_trend'],
            trend_results['monthly_trend']
        )
        
        print("  - 生成品类分布图...")
        visualizer.plot_category_distribution(features['category_features'])
        
        print("  - 生成周销售模式图...")
        visualizer.plot_weekly_pattern(trend_results['weekly_pattern'])
        
        print("  - 生成客户分群图...")
        visualizer.plot_customer_segment(customer_results['segment_summary'])
        
        print("  - 生成热销商品图...")
        visualizer.plot_top_products(product_results['top_products'])
        
        print("  - 生成商品关联热力图...")
        visualizer.plot_product_association_heatmap(
            product_results['category_association']
        )
        
        print("  - 生成预测对比图...")
        actual_data = trend_results['daily_trend'].tail(60)[['date', 'revenue']].copy()
        actual_data.columns = ['date', 'actual']
        visualizer.plot_prediction_comparison(actual_data, predictions)
        
        figures = visualizer.get_all_figures()
        
        print("\n[步骤9] 生成PDF报告...")
        report_generator = PDFReportGenerator(output_dir=output_dir)
        
        analysis_results = {
            'sales_summary': trend_results['summary'],
            'customer_analysis': {
                'segment_summary': customer_results['segment_summary'],
                'rfm_analysis': customer_results['rfm_analysis'],
                'top_customers': customer_results['top_customers']
            },
            'product_analysis': {
                'top_products': product_results['top_products'],
                'cross_sell_opportunities': product_results['cross_sell_opportunities'],
                'category_association': product_results['category_association']
            },
            'prediction_results': {
                'metrics': metrics,
                'summary': pred_summary,
                'predictions': predictions
            }
        }
        
        pdf_path = report_generator.create_report(analysis_results, figures)
        
        print("\n" + "=" * 60)
        print("分析完成！")
        print("=" * 60)
        print(f"\n输出文件位置: {output_dir}")
        print(f"- PDF报告: {pdf_path}")
        print(f"- 可视化图表: {len(figures)} 个")
        
        print("\n关键指标摘要:")
        print(f"- 总销售额: ¥{trend_results['summary']['total_revenue']:,.2f}")
        print(f"- 总订单数: {trend_results['summary']['total_orders']:,}")
        print(f"- 总客户数: {trend_results['summary']['total_customers']:,}")
        print(f"- 平均订单价值: ¥{trend_results['summary']['avg_order_value']:,.2f}")
        print(f"- 月度增长率: {trend_results['summary']['growth_metrics']['monthly_revenue_growth']:.2f}%")
        
        return True
        
    except FileNotFoundError as e:
        print(f"\n错误: 文件未找到 - {str(e)}")
        return False
    except ValueError as e:
        print(f"\n错误: 数据验证失败 - {str(e)}")
        return False
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
