"""
电商销售数据分析与可视化项目
主入口文件

功能:
1. 数据生成: 模拟电商销售数据
2. 数据清洗: 处理缺失值、异常值、重复值
3. 数据分析: RFM分群、时间序列分析、聚类分析、销售预测
4. 数据可视化: 生成多维度图表
5. 报告生成: 输出分析报告文本
"""

import os
import sys
from datetime import datetime

from data_generator import DataGenerator, generate_and_save_data
from data_cleaner import DataCleaner, clean_data
from data_analyzer import DataAnalyzer, analyze_data
from visualizer import Visualizer, visualize_data


def generate_summary_report(df, analysis_results: dict, 
                           cleaning_report: dict,
                           output_path: str = 'da3/output/summary.txt') -> str:
    """
    生成分析报告文本
    
    Args:
        df: 清洗后的数据
        analysis_results: 分析结果字典
        cleaning_report: 清洗报告
        output_path: 输出路径
        
    Returns:
        报告文件路径
    """
    print("\n生成分析报告...")
    
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("电商销售数据分析报告")
    report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 70)
    
    report_lines.append("\n" + "-" * 70)
    report_lines.append("一、数据概览")
    report_lines.append("-" * 70)
    
    report_lines.append(f"\n原始数据记录数: {cleaning_report.get('original_count', 'N/A')}")
    report_lines.append(f"清洗后记录数: {len(df)}")
    report_lines.append(f"用户数量: {df['user_id'].nunique()}")
    report_lines.append(f"订单数量: {df['order_id'].nunique()}")
    
    if 'order_date' in df.columns:
        report_lines.append(f"时间范围: {df['order_date'].min().strftime('%Y-%m-%d')} 至 {df['order_date'].max().strftime('%Y-%m-%d')}")
    
    report_lines.append("\n" + "-" * 70)
    report_lines.append("二、关键统计量")
    report_lines.append("-" * 70)
    
    if 'total_amount' in df.columns:
        report_lines.append(f"\n销售额统计:")
        report_lines.append(f"  总销售额: {df['total_amount'].sum():,.2f} 元")
        report_lines.append(f"  平均订单金额: {df['amount'].mean():.2f} 元")
        report_lines.append(f"  金额标准差: {df['amount'].std():.2f} 元")
        report_lines.append(f"  最小订单金额: {df['amount'].min():.2f} 元")
        report_lines.append(f"  最大订单金额: {df['amount'].max():.2f} 元")
    
    if 'rating' in df.columns:
        report_lines.append(f"\n评分统计:")
        report_lines.append(f"  平均评分: {df['rating'].mean():.2f}")
        report_lines.append(f"  评分中位数: {df['rating'].median():.1f}")
    
    if 'quantity' in df.columns:
        report_lines.append(f"\n销量统计:")
        report_lines.append(f"  总销量: {df['quantity'].sum()} 件")
        report_lines.append(f"  平均每单数量: {df['quantity'].mean():.2f} 件")
    
    report_lines.append("\n" + "-" * 70)
    report_lines.append("三、类别销售分析")
    report_lines.append("-" * 70)
    
    if 'category_stats' in analysis_results:
        cat_stats = analysis_results['category_stats']
        report_lines.append("\n各类别销售情况:")
        for idx, row in cat_stats.iterrows():
            report_lines.append(f"  {idx}:")
            report_lines.append(f"    总销售额: {row['总销售额']:,.2f} 元")
            report_lines.append(f"    订单数: {int(row['订单数'])}")
            report_lines.append(f"    销售占比: {row['销售占比(%)']:.2f}%")
    
    report_lines.append("\n" + "-" * 70)
    report_lines.append("四、用户分群分析 (RFM + KMeans)")
    report_lines.append("-" * 70)
    
    if 'segment_summary' in analysis_results:
        report_lines.append("\n客户分群统计:")
        for segment, count in analysis_results['segment_summary'].items():
            report_lines.append(f"  {segment}: {count}人")
    
    if 'cluster_stats' in analysis_results:
        report_lines.append("\n聚类分析结果:")
        cluster_stats = analysis_results['cluster_stats']
        for idx, row in cluster_stats.iterrows():
            report_lines.append(f"  群组{idx} ({row['客户类型']}):")
            report_lines.append(f"    用户数: {int(row['用户数'])}")
            report_lines.append(f"    平均Recency: {row['平均Recency']:.1f} 天")
            report_lines.append(f"    平均Frequency: {row['平均Frequency']:.1f} 次")
            report_lines.append(f"    平均Monetary: {row['平均Monetary']:,.2f} 元")
    
    report_lines.append("\n" + "-" * 70)
    report_lines.append("五、销售预测模型评估")
    report_lines.append("-" * 70)
    
    if 'prediction_results' in analysis_results:
        pred = analysis_results['prediction_results']
        report_lines.append("\n线性回归模型评估指标:")
        report_lines.append(f"  训练集 RMSE: {pred['train_rmse']:,.2f}")
        report_lines.append(f"  测试集 RMSE: {pred['test_rmse']:,.2f}")
        report_lines.append(f"  训练集 MAE: {pred['train_mae']:,.2f}")
        report_lines.append(f"  测试集 MAE: {pred['test_mae']:,.2f}")
        report_lines.append(f"  训练集 R²: {pred['train_r2']:.4f}")
        report_lines.append(f"  测试集 R²: {pred['test_r2']:.4f}")
    
    report_lines.append("\n" + "-" * 70)
    report_lines.append("六、相关性分析")
    report_lines.append("-" * 70)
    
    if 'correlation_matrix' in analysis_results:
        corr = analysis_results['correlation_matrix']
        report_lines.append("\n主要变量相关系数:")
        for col1 in corr.columns:
            for col2 in corr.columns:
                if col1 < col2:
                    coef = corr.loc[col1, col2]
                    report_lines.append(f"  {col1} vs {col2}: {coef:.3f}")
    
    report_lines.append("\n" + "-" * 70)
    report_lines.append("七、业务建议")
    report_lines.append("-" * 70)
    
    report_lines.append("\n基于数据分析，提出以下建议:")
    report_lines.append("  1. 针对高价值活跃客户，提供VIP专属优惠和个性化服务")
    report_lines.append("  2. 对流失风险客户进行主动召回，发放优惠券刺激复购")
    report_lines.append("  3. 加强电子产品类目的营销投入，该品类销售额最高")
    report_lines.append("  4. 关注评分较低的产品类别，优化产品质量和服务")
    report_lines.append("  5. 根据销售预测模型，提前做好库存和促销规划")
    
    report_lines.append("\n" + "=" * 70)
    report_lines.append("报告结束")
    report_lines.append("=" * 70)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"分析报告已保存: {output_path}")
    
    return output_path


def main():
    """
    主函数 - 执行完整的数据分析流程
    """
    print("\n" + "=" * 70)
    print("电商销售数据分析与可视化项目")
    print("=" * 70)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    output_dir = os.path.join(script_dir, 'output')
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        print("\n【步骤1】生成模拟数据...")
        generator = DataGenerator(seed=42)
        df = generator.generate_sales_data(n_records=600)
        csv_path = os.path.join(data_dir, 'sales_data.csv')
        generator.save_to_csv(df, csv_path)
        
        print("\n【步骤2】数据清洗...")
        cleaner = DataCleaner(df)
        cleaner.cleaning_report['original_count'] = len(df)
        cleaned_df = cleaner.clean_all()
        cleaning_report = cleaner.get_cleaning_report()
        
        print("\n【步骤3】数据分析...")
        analyzer = DataAnalyzer(cleaned_df)
        
        analyzer.descriptive_statistics()
        analyzer.correlation_analysis()
        analyzer.rfm_analysis()
        analyzer.time_series_analysis()
        analyzer.kmeans_clustering(n_clusters=4)
        analyzer.sales_prediction()
        analyzer.category_analysis()
        
        analysis_results = analyzer.get_all_results()
        
        print("\n【步骤4】数据可视化...")
        visualizer = Visualizer(output_dir=output_dir)
        figures = visualizer.generate_all_plots(cleaned_df, analysis_results)
        
        print("\n【步骤5】生成分析报告...")
        report_path = generate_summary_report(
            cleaned_df, analysis_results, cleaning_report,
            os.path.join(output_dir, 'summary.txt')
        )
        
        print("\n" + "=" * 70)
        print("项目执行完成!")
        print("=" * 70)
        
        print(f"\n输出文件:")
        print(f"  数据文件: {csv_path}")
        print(f"  分析报告: {report_path}")
        print(f"\n生成的图表 ({len(figures)}张):")
        for name, path in figures.items():
            print(f"  - {name}: {path}")
        
        print("\n" + "=" * 70)
        
        return cleaned_df, analysis_results, figures
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None


if __name__ == '__main__':
    main()
