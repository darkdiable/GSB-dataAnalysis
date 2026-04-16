#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
电商销售数据分析与可视化项目
主入口程序
"""

import sys
import os
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_generator import generate_ecommerce_data, save_data
from data_cleaner import DataCleaner
from data_analyzer import DataAnalyzer
from visualizer import Visualizer

def generate_summary_report(analysis_results, cleaning_report, output_path='output/summary.txt'):
    """生成分析报告文本"""
    print("=" * 60)
    print("正在生成分析报告...")
    print("=" * 60)
    
    summary = []
    summary.append("=" * 60)
    summary.append("电商销售数据分析报告")
    summary.append("=" * 60)
    summary.append(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary.append("")
    
    summary.append("-" * 60)
    summary.append("1. 数据清洗报告")
    summary.append("-" * 60)
    for k, v in cleaning_report.items():
        summary.append(f"  {k}: {v}")
    summary.append("")
    
    desc = analysis_results['描述统计']
    summary.append("-" * 60)
    summary.append("2. 描述性统计关键指标")
    summary.append("-" * 60)
    summary.append(f"  总订单数: {desc['总订单数']}")
    summary.append(f"  总销售额: {desc['总销售额']:,.2f} 元")
    summary.append(f"  平均订单金额: {desc['平均订单金额']:.2f} 元")
    summary.append(f"  订单金额中位数: {desc['订单金额中位数']:.2f} 元")
    summary.append(f"  客单价: {desc['客单价']:.2f} 元")
    summary.append(f"  用户总数: {desc['用户总数']} 人")
    summary.append(f"  平均评分: {desc['平均评分']:.2f} / 5.0")
    summary.append("")
    
    summary.append("-" * 60)
    summary.append("3. 商品类别销售排名 (按销售额)")
    summary.append("-" * 60)
    category_sales = pd.DataFrame(desc['类别销售详情']).sort_values('sum', ascending=False)
    for idx, row in category_sales.iterrows():
        summary.append(f"  {idx}: {row['sum']:,.2f} 元 (共{int(row['count'])}单)")
    summary.append("")
    
    summary.append("-" * 60)
    summary.append("4. 用户分层聚类分析结果 (KMeans)")
    summary.append("-" * 60)
    cluster_details = analysis_results['用户聚类']['聚类详情']
    for cluster_id, info in cluster_details.items():
        summary.append(f"  聚类{cluster_id} [{info['用户类型']}]:")
        summary.append(f"    - 用户数: {info['用户数']} 人")
        summary.append(f"    - 平均最近购买天数: {info['平均R']:.1f} 天")
        summary.append(f"    - 平均购买频次: {info['平均F']:.1f} 次")
        summary.append(f"    - 平均消费金额: {info['平均M']:.2f} 元")
    summary.append("")
    
    pred = analysis_results['销售预测']
    summary.append("-" * 60)
    summary.append("5. 销售预测模型评估")
    summary.append("-" * 60)
    summary.append(f"  模型: 多元线性回归")
    summary.append(f"  RMSE (均方根误差): {pred['RMSE']:.2f} 元")
    summary.append(f"  MSE (均方误差): {pred['MSE']:.2f}")
    summary.append(f"  R² (决定系数): {pred['R2']:.3f}")
    summary.append(f"  特征系数: {pred['系数']}")
    summary.append("")
    
    summary.append("-" * 60)
    summary.append("6. 相关性分析结论")
    summary.append("-" * 60)
    corr = analysis_results['相关性分析']
    summary.append("  注: 相关系数绝对值 > 0.5 表示强相关")
    high_corr = []
    for col1 in corr:
        for col2 in corr:
            if col1 != col2 and abs(corr[col1][col2]) > 0.3:
                high_corr.append(f"{col1} & {col2}: {corr[col1][col2]:.3f}")
    if high_corr:
        summary.append("  显著相关特征对:")
        for item in list(set(high_corr)):
            summary.append(f"    - {item}")
    else:
        summary.append("  无显著强相关特征对")
    summary.append("")
    
    summary.append("=" * 60)
    summary.append("报告结束")
    summary.append("=" * 60)
    
    report_content = "\n".join(summary)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"分析报告已保存至: {output_path}")
    print("\n报告摘要预览:")
    print("\n".join(summary[1:20]))
    
    return report_content

def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("电商销售数据分析与可视化项目启动")
    print("=" * 60 + "\n")
    
    try:
        print("【步骤1/5】生成模拟电商销售数据...")
        df_raw = generate_ecommerce_data(n_records=2000, start_date='2023-01-01', end_date='2023-04-30')
        save_data(df_raw, 'data/raw_ecommerce_data.csv')
        print()
        
        print("【步骤2/5】数据清洗...")
        cleaner = DataCleaner(df_raw)
        df_cleaned = cleaner.clean()
        cleaner.save_cleaned_data('data/cleaned_ecommerce_data.csv')
        cleaning_report = cleaner.get_cleaning_report()
        print()
        
        print("【步骤3/5】执行数据分析...")
        analyzer = DataAnalyzer(df_cleaned)
        analysis_results = analyzer.run_full_analysis()
        print()
        
        print("【步骤4/5】生成可视化图表...")
        visualizer = Visualizer(output_dir='output')
        charts = visualizer.generate_all_charts(df_cleaned, analyzer.rfm_df)
        print()
        
        print("【步骤5/5】生成分析报告...")
        import pandas as pd
        generate_summary_report(analysis_results, cleaning_report, 'output/summary.txt')
        print()
        
        print("=" * 60)
        print("项目执行成功! 输出文件清单:")
        print("=" * 60)
        print("数据文件:")
        print("  - data/raw_ecommerce_data.csv (原始数据)")
        print("  - data/cleaned_ecommerce_data.csv (清洗后数据)")
        print("\n图表文件:")
        for chart in charts:
            print(f"  - output/{chart}")
        print("\n报告文件:")
        print("  - output/summary.txt (分析报告)")
        print("\n" + "=" * 60)
        print("所有输出文件已生成完毕!")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n❌ 项目执行出错: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
