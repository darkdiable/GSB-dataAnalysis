#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
电商销售数据分析与可视化项目 - 主入口

功能模块：
1. 数据生成 - 模拟真实电商销售数据
2. 数据清洗 - 处理缺失值、异常值、重复值
3. 数据分析 - 描述统计、时间序列、RFM、聚类、预测
4. 数据可视化 - 生成各类分析图表

使用方法：
    python main.py

输出：
- da3/data/sales_data_raw.csv: 原始数据
- da3/data/sales_data_cleaned.csv: 清洗后数据
- da3/output/*.png: 分析图表（PNG格式）
- da3/output/*.pdf: 分析图表（PDF格式）
- da3/output/summary.txt: 分析报告
"""

import os
import sys
import argparse
from datetime import datetime

# 添加模块路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.data_generator import DataGenerator
from modules.data_cleaner import DataCleaner
from modules.data_analyzer import DataAnalyzer
from modules.visualizer import Visualizer


def setup_directories():
    """创建必要的目录结构"""
    dirs = ['data', 'output']
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("目录结构检查完成\n")


def generate_data(n_records=1000, output_path='data/sales_data_raw.csv'):
    """
    生成模拟电商数据
    
    Args:
        n_records: 生成记录数
        output_path: 输出文件路径
        
    Returns:
        DataFrame: 生成的原始数据
    """
    print("=" * 60)
    print("步骤 1/4: 数据生成")
    print("=" * 60)
    
    try:
        generator = DataGenerator(n_records=n_records)
        
        # 生成基础数据
        df = generator.generate_data()
        
        # 添加数据质量问题（模拟真实场景）
        print("\n添加数据质量问题（用于测试清洗功能）...")
        df = generator.add_missing_values(df, missing_rate=0.05)
        df = generator.add_outliers(df, outlier_rate=0.02)
        df = generator.add_duplicates(df, duplicate_rate=0.03)
        
        # 保存数据
        generator.save_to_csv(df, output_path)
        
        print(f"\n数据生成完成!")
        print(f"  - 记录数: {len(df)}")
        print(f"  - 时间跨度: {df['订单时间'].min()} 至 {df['订单时间'].max()}")
        print(f"  - 用户数: {df['用户ID'].nunique()}")
        print(f"  - 输出文件: {output_path}")
        
        return df
        
    except Exception as e:
        print(f"\n数据生成失败: {str(e)}")
        raise


def clean_data(input_path='data/sales_data_raw.csv', 
               output_path='data/sales_data_cleaned.csv'):
    """
    清洗数据
    
    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
        
    Returns:
        tuple: (清洗后的DataFrame, DataCleaner实例)
    """
    print("\n" + "=" * 60)
    print("步骤 2/4: 数据清洗")
    print("=" * 60)
    
    try:
        # 读取数据
        import pandas as pd
        df = pd.read_csv(input_path, encoding='utf-8-sig')
        df['订单时间'] = pd.to_datetime(df['订单时间'])
        
        print(f"\n读取原始数据: {len(df)} 条记录")
        
        # 创建清洗器并执行清洗
        cleaner = DataCleaner(df)
        cleaned_df = cleaner.clean()
        
        # 保存清洗后的数据
        cleaner.save_cleaned_data(output_path)
        
        # 打印清洗报告
        print("\n" + cleaner.get_cleaning_report())
        
        return cleaned_df, cleaner
        
    except Exception as e:
        print(f"\n数据清洗失败: {str(e)}")
        raise


def analyze_data(df):
    """
    分析数据
    
    Args:
        df: 清洗后的DataFrame
        
    Returns:
        DataAnalyzer: 分析器实例
    """
    print("\n" + "=" * 60)
    print("步骤 3/4: 数据分析")
    print("=" * 60)
    
    try:
        analyzer = DataAnalyzer(df)
        analyzer.run_full_analysis()
        
        print("\n数据分析摘要:")
        print(f"  - 总销售额: ¥{df['总金额'].sum():,.2f}")
        print(f"  - 平均订单金额: ¥{df['总金额'].mean():.2f}")
        print(f"  - 用户总数: {df['用户ID'].nunique()}")
        
        # 聚类结果
        cluster_results = analyzer.analysis_results.get('用户聚类', {})
        if cluster_results:
            silhouette = cluster_results.get('轮廓系数', 0)
            print(f"  - 聚类轮廓系数: {silhouette:.3f}")
        
        # 预测结果
        pred_results = analyzer.analysis_results.get('销售预测', {})
        if pred_results:
            model_eval = pred_results.get('模型评估', {})
            r2 = model_eval.get('R²', 'N/A')
            print(f"  - 预测模型 R²: {r2}")
        
        return analyzer
        
    except Exception as e:
        print(f"\n数据分析失败: {str(e)}")
        raise


def visualize_data(df, analyzer_results, output_dir='output'):
    """
    生成可视化图表
    
    Args:
        df: 数据框
        analyzer_results: 分析结果字典
        output_dir: 输出目录
        
    Returns:
        list: 生成的文件列表
    """
    print("\n" + "=" * 60)
    print("步骤 4/4: 数据可视化")
    print("=" * 60)
    
    try:
        visualizer = Visualizer(output_dir)
        files = visualizer.generate_all_visualizations(df, analyzer_results)
        
        print(f"\n共生成 {len(files)} 个图表文件")
        
        return files
        
    except Exception as e:
        print(f"\n数据可视化失败: {str(e)}")
        raise


def save_summary_report(analyzer, output_path='output/summary.txt'):
    """
    保存分析报告
    
    Args:
        analyzer: DataAnalyzer实例
        output_path: 输出文件路径
    """
    print("\n" + "=" * 60)
    print("生成分析报告")
    print("=" * 60)
    
    try:
        report = analyzer.generate_summary_report()
        
        # 添加项目信息
        header = []
        header.append("=" * 70)
        header.append("电商销售数据分析与可视化项目")
        header.append("=" * 70)
        header.append(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        header.append("")
        
        full_report = "\n".join(header) + "\n" + report
        
        # 保存报告
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_report)
        
        print(f"\n分析报告已保存: {output_path}")
        
        # 同时打印到控制台
        print("\n" + "=" * 60)
        print("分析报告内容预览:")
        print("=" * 60)
        print(full_report[:2000] + "..." if len(full_report) > 2000 else full_report)
        
        return full_report
        
    except Exception as e:
        print(f"\n生成报告失败: {str(e)}")
        raise


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='电商销售数据分析与可视化项目',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py                    # 使用默认参数运行
  python main.py --records 2000     # 生成2000条记录
  python main.py --skip-generate    # 跳过数据生成步骤
        """
    )
    parser.add_argument(
        '--records', 
        type=int, 
        default=1000,
        help='生成记录数 (默认: 1000)'
    )
    parser.add_argument(
        '--skip-generate',
        action='store_true',
        help='跳过数据生成步骤（使用已有数据）'
    )
    parser.add_argument(
        '--skip-clean',
        action='store_true',
        help='跳过数据清洗步骤（使用已清洗数据）'
    )
    
    args = parser.parse_args()
    
    # 记录开始时间
    start_time = datetime.now()
    
    print("\n" + "=" * 70)
    print("电商销售数据分析与可视化项目")
    print("=" * 70)
    print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("")
    
    try:
        # 1. 设置目录结构
        setup_directories()
        
        # 2. 数据生成
        if args.skip_generate and os.path.exists('data/sales_data_raw.csv'):
            print("跳过数据生成，使用已有数据...")
            import pandas as pd
            raw_df = pd.read_csv('data/sales_data_raw.csv', encoding='utf-8-sig')
            raw_df['订单时间'] = pd.to_datetime(raw_df['订单时间'])
        else:
            raw_df = generate_data(n_records=args.records)
        
        # 3. 数据清洗
        if args.skip_clean and os.path.exists('data/sales_data_cleaned.csv'):
            print("\n跳过数据清洗，使用已清洗数据...")
            import pandas as pd
            cleaned_df = pd.read_csv('data/sales_data_cleaned.csv', encoding='utf-8-sig')
            cleaned_df['订单时间'] = pd.to_datetime(cleaned_df['订单时间'])
        else:
            cleaned_df, cleaner = clean_data()
        
        # 4. 数据分析
        analyzer = analyze_data(cleaned_df)
        
        # 5. 数据可视化
        files = visualize_data(cleaned_df, analyzer.analysis_results)
        
        # 6. 生成分析报告
        save_summary_report(analyzer)
        
        # 计算运行时间
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("\n" + "=" * 70)
        print("项目执行成功!")
        print("=" * 70)
        print(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"运行时长: {duration:.2f} 秒")
        print("")
        print("输出文件:")
        print(f"  - 原始数据: data/sales_data_raw.csv")
        print(f"  - 清洗数据: data/sales_data_cleaned.csv")
        print(f"  - 分析报告: output/summary.txt")
        print(f"  - 图表文件: output/ 目录下共 {len(files)} 个文件")
        print("=" * 70)
        
        return 0
        
    except Exception as e:
        print("\n" + "=" * 70)
        print("项目执行失败!")
        print("=" * 70)
        print(f"错误信息: {str(e)}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
