"""
电商数据分析项目主入口
整合数据生成、清洗、分析、可视化全流程
"""

import os
import sys
from datetime import datetime

from data_generator import DataGenerator, generate_and_save_data
from data_cleaner import DataCleaner, clean_data
from analyzer import (
    ExploratoryAnalyzer, TimeSeriesAnalyzer, 
    RFMAnalyzer, CustomerClusterAnalyzer, SalesPredictor
)
from visualizer import Visualizer


def print_header(title: str):
    """打印标题头"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def generate_summary_report(df, stats, corr_matrix, cluster_profile, 
                           rfm_summary, prediction_metrics, output_path: str):
    """
    生成分析报告文本
    
    Args:
        df: 清洗后的数据
        stats: 描述统计
        corr_matrix: 相关性矩阵
        cluster_profile: 聚类画像
        rfm_summary: RFM分层摘要
        prediction_metrics: 预测指标
        output_path: 输出路径
    """
    try:
        report = []
        
        report.append("=" * 70)
        report.append("电商销售数据分析报告")
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 70)
        
        report.append("\n" + "-" * 70)
        report.append("一、数据概览")
        report.append("-" * 70)
        report.append(f"总订单数: {len(df)}")
        report.append(f"唯一用户数: {df['user_id'].nunique()}")
        report.append(f"商品类别数: {df['category'].nunique()}")
        report.append(f"时间跨度: {df['order_time'].min().strftime('%Y-%m-%d')} 至 {df['order_time'].max().strftime('%Y-%m-%d')}")
        report.append(f"数据天数: {(df['order_time'].max() - df['order_time'].min()).days} 天")
        
        report.append("\n" + "-" * 70)
        report.append("二、关键统计指标")
        report.append("-" * 70)
        
        if 'total_amount' in stats:
            report.append("\n【订单金额统计】")
            report.append(f"  总销售额: ¥{df['total_amount'].sum():,.2f}")
            report.append(f"  平均订单金额: ¥{stats['total_amount']['mean']:,.2f}")
            report.append(f"  金额标准差: ¥{stats['total_amount']['std']:,.2f}")
            report.append(f"  最小订单金额: ¥{stats['total_amount']['min']:,.2f}")
            report.append(f"  25%分位数: ¥{stats['total_amount']['25%']:,.2f}")
            report.append(f"  中位数: ¥{stats['total_amount']['50%']:,.2f}")
            report.append(f"  75%分位数: ¥{stats['total_amount']['75%']:,.2f}")
            report.append(f"  最大订单金额: ¥{stats['total_amount']['max']:,.2f}")
        
        if 'rating' in stats:
            report.append("\n【评分统计】")
            report.append(f"  平均评分: {stats['rating']['mean']:.2f} 分")
            report.append(f"  评分标准差: {stats['rating']['std']:.2f}")
            report.append(f"  最低评分: {stats['rating']['min']:.1f} 分")
            report.append(f"  最高评分: {stats['rating']['max']:.1f} 分")
        
        if 'quantity' in stats:
            report.append("\n【购买数量统计】")
            report.append(f"  总销售数量: {df['quantity'].sum():,} 件")
            report.append(f"  平均每单数量: {stats['quantity']['mean']:.2f} 件")
        
        report.append("\n" + "-" * 70)
        report.append("三、类别销售分析")
        report.append("-" * 70)
        
        category_sales = df.groupby('category').agg({
            'total_amount': 'sum',
            'order_id': 'count',
            'quantity': 'sum'
        }).sort_values('total_amount', ascending=False)
        
        for i, (cat, row) in enumerate(category_sales.iterrows(), 1):
            pct = row['total_amount'] / df['total_amount'].sum() * 100
            report.append(f"{i}. {cat}: ¥{row['total_amount']:,.0f} ({pct:.1f}%), {int(row['order_id'])}单")
        
        report.append("\n" + "-" * 70)
        report.append("四、相关性分析")
        report.append("-" * 70)
        report.append("\n【相关性矩阵】")
        report.append(corr_matrix.round(3).to_string())
        
        strong_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    strong_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
        
        if strong_corr:
            report.append("\n【强相关变量对】(相关系数 > 0.5)")
            for var1, var2, corr in strong_corr:
                report.append(f"  {var1} <-> {var2}: {corr:.3f}")
        
        report.append("\n" + "-" * 70)
        report.append("五、用户聚类分析结论")
        report.append("-" * 70)
        
        if cluster_profile is not None:
            report.append("\n【聚类画像】")
            for cluster_id, row in cluster_profile.iterrows():
                report.append(f"\n聚类 {cluster_id} ({row['用户标签']}):")
                report.append(f"  用户数: {int(row['用户数'])} 人")
                report.append(f"  平均消费总额: ¥{row['平均消费总额']:,.2f}")
                report.append(f"  平均订单数: {row['平均订单数']:.1f} 次")
                report.append(f"  平均订单金额: ¥{row['平均订单金额']:,.2f}")
                report.append(f"  平均评分: {row['平均评分']:.2f} 分")
        
        report.append("\n" + "-" * 70)
        report.append("六、RFM客户分层结论")
        report.append("-" * 70)
        
        if rfm_summary is not None:
            report.append("\n【客户分层统计】")
            for segment in rfm_summary.get('用户数', {}).keys():
                user_count = rfm_summary['用户数'][segment]
                avg_monetary = rfm_summary['平均M(元)'][segment]
                pct = rfm_summary['占比(%)'][segment]
                report.append(f"  {segment}: {int(user_count)} 人 ({pct}%), 平均消费 ¥{avg_monetary:,.0f}")
        
        report.append("\n" + "-" * 70)
        report.append("七、销售预测模型评估")
        report.append("-" * 70)
        
        if prediction_metrics:
            report.append("\n【模型评估指标】")
            report.append(f"  R² 分数: {prediction_metrics['r2_score']:.4f}")
            report.append(f"  均方误差 (MSE): {prediction_metrics['mse']:,.2f}")
            report.append(f"  均方根误差 (RMSE): {prediction_metrics['rmse']:,.2f}")
            report.append(f"  平均绝对误差 (MAE): {prediction_metrics['mae']:,.2f}")
            
            if prediction_metrics['r2_score'] > 0.7:
                report.append("\n  模型拟合效果: 良好")
            elif prediction_metrics['r2_score'] > 0.5:
                report.append("\n  模型拟合效果: 中等")
            else:
                report.append("\n  模型拟合效果: 需要改进")
        
        report.append("\n" + "-" * 70)
        report.append("八、业务建议")
        report.append("-" * 70)
        
        top_category = category_sales.index[0]
        report.append(f"\n1. 重点发展 {top_category} 品类，该品类销售额最高")
        
        if cluster_profile is not None:
            high_value_cluster = cluster_profile['平均消费总额'].idxmax()
            high_value_count = int(cluster_profile.loc[high_value_cluster, '用户数'])
            report.append(f"2. 关注高价值用户群体，共 {high_value_count} 人，可提供VIP服务")
        
        if rfm_summary is not None:
            churn_count = rfm_summary.get('用户数', {}).get('流失客户', 0)
            if churn_count > 0:
                report.append(f"3. 针对流失客户({int(churn_count)}人)制定召回策略")
        
        report.append("4. 建议优化周末促销活动，提升周末销售额")
        report.append("5. 持续监控用户评分，提升客户满意度")
        
        report.append("\n" + "=" * 70)
        report.append("报告结束")
        report.append("=" * 70)
        
        report_text = "\n".join(report)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"\n分析报告已保存: {output_path}")
        
        return report_text
        
    except Exception as e:
        raise RuntimeError(f"生成报告失败: {str(e)}")


def main():
    """主函数"""
    print_header("电商数据分析项目")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    output_dir = os.path.join(base_dir, 'output')
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        print_header("步骤1: 数据生成")
        generator = DataGenerator(seed=42)
        df = generator.generate_data(n_records=600, months=4)
        
        df_dirty = generator.introduce_issues(df, missing_rate=0.02, 
                                              duplicate_rate=0.01, 
                                              outlier_rate=0.01)
        
        clean_path = os.path.join(data_dir, 'sales_data_clean.csv')
        dirty_path = os.path.join(data_dir, 'sales_data_dirty.csv')
        generator.save_to_csv(df, clean_path)
        generator.save_to_csv(df_dirty, dirty_path)
        
        print(f"生成数据: {len(df)} 条记录")
        print(f"时间跨度: {df['order_time'].min().strftime('%Y-%m-%d')} 至 {df['order_time'].max().strftime('%Y-%m-%d')}")
        
        print_header("步骤2: 数据清洗")
        cleaner = DataCleaner(df_dirty)
        df_cleaned = cleaner.clean_all()
        
        cleaned_path = os.path.join(data_dir, 'sales_data_cleaned.csv')
        cleaner.df.to_csv(cleaned_path, index=False, encoding='utf-8-sig')
        print(f"清洗后数据已保存: {cleaned_path}")
        
        print_header("步骤3: 探索性分析")
        explorer = ExploratoryAnalyzer(df_cleaned)
        stats = explorer.descriptive_statistics()
        corr_matrix = explorer.correlation_analysis()
        
        print(explorer.get_analysis_summary())
        
        print_header("步骤4: 时间序列分析")
        ts_analyzer = TimeSeriesAnalyzer(df_cleaned)
        daily_sales = ts_analyzer.moving_average(window=7)
        monthly_sales = ts_analyzer.monthly_sales()
        weekly_pattern = ts_analyzer.weekly_pattern()
        
        print(f"分析天数: {len(daily_sales)} 天")
        print(f"月度数据: {len(monthly_sales)} 个月")
        
        print_header("步骤5: RFM客户分层")
        rfm = RFMAnalyzer(df_cleaned)
        rfm_df = rfm.calculate_rfm()
        rfm_df = rfm.segment_customers()
        rfm_summary = rfm.get_segment_summary()
        
        print("\n客户分层结果:")
        segment_counts = rfm_df['segment'].value_counts()
        for segment, count in segment_counts.items():
            print(f"  {segment}: {count} 人")
        
        print_header("步骤6: 用户聚类分析")
        cluster_analyzer = CustomerClusterAnalyzer(df_cleaned)
        cluster_df = cluster_analyzer.prepare_features()
        cluster_df, cluster_metrics = cluster_analyzer.perform_clustering(n_clusters=4)
        cluster_profile = cluster_analyzer.get_cluster_profile()
        
        print("\n聚类画像:")
        print(cluster_profile.to_string())
        
        print_header("步骤7: 销售预测")
        predictor = SalesPredictor(df_cleaned)
        prediction_metrics = predictor.train_model()
        
        print("\n预测模型指标:")
        for metric, value in prediction_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        print_header("步骤8: 数据可视化")
        visualizer = Visualizer(output_dir=output_dir)
        
        filepaths = visualizer.generate_all_plots(
            df_cleaned, daily_sales, corr_matrix, 
            cluster_df, rfm_df, monthly_sales
        )
        
        print_header("步骤9: 生成分析报告")
        summary_path = os.path.join(output_dir, 'summary.txt')
        report = generate_summary_report(
            df_cleaned, stats, corr_matrix, cluster_profile,
            rfm_summary, prediction_metrics, summary_path
        )
        
        print_header("项目完成")
        print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("\n【输出文件清单】")
        print(f"  数据文件:")
        print(f"    - {clean_path}")
        print(f"    - {dirty_path}")
        print(f"    - {cleaned_path}")
        print(f"  图表文件:")
        for fp in filepaths:
            print(f"    - {fp}")
        print(f"  分析报告:")
        print(f"    - {summary_path}")
        
        return df_cleaned, stats, cluster_profile, rfm_df, prediction_metrics
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
