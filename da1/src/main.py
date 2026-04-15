"""
全球燃油性能车深度分析与可视化项目 - 主程序入口
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')

from data_preprocessing import DataPreprocessor, preprocess_pipeline
from time_series_analysis import TimeSeriesAnalyzer, run_time_series_analysis
from geographic_analysis import GeographicAnalyzer, run_geographic_analysis
from market_segmentation import MarketSegmentationAnalyzer, run_market_segmentation_analysis
from statistical_modeling import StatisticalModelingAnalyzer, run_statistical_modeling
from report_generator import generate_full_report


def setup_directories(base_dir):
    """设置目录结构"""
    dirs = {
        'data': os.path.join(base_dir, 'data'),
        'output': os.path.join(base_dir, 'output'),
        'src': os.path.join(base_dir, 'src')
    }

    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    return dirs


def print_banner():
    """打印项目横幅"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║     全球燃油性能车深度分析与可视化项目                        ║
    ║     Global Fuel Performance Cars Analysis & Visualization    ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def main():
    """主程序"""
    print_banner()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dirs = setup_directories(base_dir)

    data_file = os.path.join(dirs['data'], 'fuel_performance_cars.csv')

    if not os.path.exists(data_file):
        print(f"错误: 数据文件不存在 {data_file}")
        sys.exit(1)

    print(f"\n数据文件: {data_file}")
    print(f"输出目录: {dirs['output']}")
    print("\n" + "="*60)

    print("\n【步骤 1/6】数据加载与预处理...")
    print("-"*60)
    df = pd.read_csv(data_file)
    print(f"原始数据: {df.shape[0]}行 x {df.shape[1]}列")
    print(f"数据范围: {df['Year'].min()} - {df['Year'].max()}")
    print(f"国家数量: {df['Country'].nunique()}")
    print(f"品牌数量: {df['Brand'].nunique()}")
    print(f"车型数量: {df['Model'].nunique()}")

    preprocessor = DataPreprocessor(df)
    preprocessor.add_missing_values(missing_rate=0.02, random_state=42)
    preprocessor.multiple_imputation(max_iter=10, random_state=42)
    outlier_info = preprocessor.detect_outliers_iqr(remove=False)
    df_processed = preprocessor.get_processed_data()
    print(f"\n预处理完成: {df_processed.shape[0]}行 x {df_processed.shape[1]}列")

    print("\n【步骤 2/6】时间序列分析...")
    print("-"*60)
    ts_analyzer = run_time_series_analysis(df_processed, dirs['output'])
    cagr = ts_analyzer.calculate_cagr()
    forecast_df = ts_analyzer.arima_forecast(forecast_years=2)

    print("\n【步骤 3/6】地理分析...")
    print("-"*60)
    geo_analyzer, concentration = run_geographic_analysis(df_processed, dirs['output'])

    print("\n【步骤 4/6】市场细分分析...")
    print("-"*60)
    seg_analyzer = run_market_segmentation_analysis(df_processed, dirs['output'])

    print("\n【步骤 5/6】统计建模分析...")
    print("-"*60)
    stat_analyzer = run_statistical_modeling(df_processed, dirs['output'])
    _, gini = stat_analyzer.plot_lorenz_curve()

    print("\n【步骤 6/6】生成分析报告...")
    print("-"*60)
    report_path = generate_full_report(
        df_processed, cagr, concentration, forecast_df, gini, dirs['output']
    )

    print("\n" + "="*60)
    print("分析完成!")
    print("="*60)

    output_files = os.listdir(dirs['output'])
    png_files = [f for f in output_files if f.endswith('.png')]
    html_files = [f for f in output_files if f.endswith('.html')]
    md_files = [f for f in output_files if f.endswith('.md')]

    print(f"\n输出文件统计:")
    print(f"  - PNG图表: {len(png_files)}个")
    print(f"  - HTML交互式图表: {len(html_files)}个")
    print(f"  - Markdown报告: {len(md_files)}个")

    print(f"\n生成的文件列表:")
    print("\n静态图表 (300 DPI PNG):")
    for i, f in enumerate(sorted(png_files), 1):
        print(f"  {i:2d}. {f}")

    print("\n交互式图表 (HTML):")
    for i, f in enumerate(sorted(html_files), 1):
        print(f"  {i}. {f}")

    print("\n分析报告:")
    for f in md_files:
        print(f"  - {f}")

    print(f"\n所有文件已保存至: {dirs['output']}")
    print("\n分析项目执行完毕!")

    return {
        'data': df_processed,
        'cagr': cagr,
        'concentration': concentration,
        'forecast': forecast_df,
        'gini': gini,
        'output_dir': dirs['output']
    }


if __name__ == '__main__':
    results = main()
