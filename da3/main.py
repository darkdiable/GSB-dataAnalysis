#!/usr/bin/env python3
"""
电商销售数据分析与可视化项目
主入口文件
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

from data_generator import generate_sales_data, load_data
from data_cleaner import clean_data, data_quality_report
from analyzer import (
    exploratory_analysis, rfm_analysis, customer_clustering,
    time_series_analysis, sales_prediction, generate_summary
)
from visualizer import generate_all_charts


def main():
    """
    主函数：完整数据分析流程
    """
    print("=" * 60)
    print("电商销售数据分析与可视化系统")
    print("=" * 60)
    
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, 'data')
    output_dir = os.path.join(base_dir, 'output')
    
    raw_data_path = os.path.join(data_dir, 'raw_sales.csv')
    cleaned_data_path = os.path.join(data_dir, 'cleaned_sales.csv')
    summary_path = os.path.join(output_dir, 'summary.txt')
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        print("\n【阶段1】数据生成与加载")
        print("-" * 40)
        df_raw = generate_sales_data(
            n_records=1200,
            days=120,
            output_path=raw_data_path
        )
        
        print("\n【阶段2】数据清洗")
        print("-" * 40)
        df_cleaned, cleaning_report = clean_data(
            df_raw,
            output_path=cleaned_data_path
        )
        data_quality_report(
            df_cleaned, 
            cleaning_report,
            output_path=os.path.join(output_dir, 'cleaning_report.txt')
        )
        
        print("\n【阶段3】数据分析")
        print("-" * 40)
        
        eda_results = exploratory_analysis(df_cleaned)
        
        rfm_df = rfm_analysis(df_cleaned)
        rfm_df, cluster_results = customer_clustering(rfm_df, n_clusters=4)
        
        daily_sales, ts_results = time_series_analysis(df_cleaned)
        
        pred_results = sales_prediction(df_cleaned)
        
        generate_summary(
            df_cleaned,
            eda_results,
            cluster_results,
            ts_results,
            pred_results,
            output_path=summary_path
        )
        
        print("\n【阶段4】可视化")
        print("-" * 40)
        generate_all_charts(
            df_cleaned,
            rfm_df,
            daily_sales,
            output_dir=output_dir
        )
        
        print("\n" + "=" * 60)
        print("数据分析流程完成！")
        print("=" * 60)
        print("\n输出文件清单:")
        print(f"  - 原始数据: {raw_data_path}")
        print(f"  - 清洗后数据: {cleaned_data_path}")
        print(f"  - 数据清洗报告: {os.path.join(output_dir, 'cleaning_report.txt')}")
        print(f"  - 分析总结报告: {summary_path}")
        print("\n图表文件 (output/目录):")
        print("  - 1_sales_trend.png (销售额趋势图)")
        print("  - 2_amount_distribution.png (金额分布图)")
        print("  - 3_category_sales.png (类别销售图)")
        print("  - 4_customer_clustering.png (用户聚类图)")
        print("  - 5_correlation_heatmap.png (相关性热力图)")
        print("  - 6_weekday_hour_heatmap.png (星期-小时热力图)")
        print("\n请查看 output/ 目录下的所有输出文件。")
        
        return 0
        
    except Exception as e:
        print(f"\n错误：程序执行过程中发生异常 - {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
