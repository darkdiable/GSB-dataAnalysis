import os
import sys

from data_loader import load_data, save_raw_data
from preprocess import preprocess_data
from analysis import run_analysis, save_summary_report
from visualization import run_visualization


def create_output_directory(output_dir='output'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出目录: {output_dir}")
    return output_dir


def print_banner():
    banner = """
╔══════════════════════════════════════════════════════════════╗
║              气象数据分析与可视化系统                            ║
║         Meteorological Data Analysis & Visualization           ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_summary(results, saved_files, report_files):
    print("\n" + "=" * 70)
    print("                      分析报告总结")
    print("=" * 70)
    
    summary = results['summary']
    
    print("\n【数据范围】")
    print(f"  起始日期: {summary['data_range']['start_date']}")
    print(f"  结束日期: {summary['data_range']['end_date']}")
    print(f"  总天数: {summary['data_range']['total_days']} 天")
    
    print("\n【总体统计】")
    print(f"  平均温度: {summary['overall_stats']['temperature_mean']} °C")
    print(f"  温度范围: {summary['overall_stats']['temperature_min']} ~ {summary['overall_stats']['temperature_max']} °C")
    print(f"  平均湿度: {summary['overall_stats']['humidity_mean']} %")
    print(f"  总降水量: {summary['overall_stats']['precipitation_total']} mm")
    print(f"  平均风速: {summary['overall_stats']['wind_speed_mean']} m/s")
    
    print("\n【极端天气事件】")
    for event, stats in summary['extreme_weather'].items():
        event_names_cn = {
            'high_temp': '高温(≥35°C)',
            'low_temp': '低温(≤0°C)',
            'heavy_wind': '大风(≥15m/s)',
            'heavy_rain': '强降水(≥50mm)'
        }
        print(f"  {event_names_cn[event]}: {stats['total_count']} 次")
        if stats['max_value'] is not None:
            print(f"    最大值: {stats['max_value']:.2f}, 平均值: {stats['mean_value']:.2f}")
    
    print("\n【趋势分析】")
    for var, trend in summary['trends'].items():
        var_names_cn = {
            'temperature': '温度',
            'humidity': '湿度',
            'precipitation': '降水量',
            'wind_speed': '风速'
        }
        sig_text = "显著" if trend['is_significant'] else "不显著"
        print(f"  {var_names_cn[var]}: {trend['trend_direction']}趋势 (p-value: {trend['p_value']:.4f}, {sig_text})")
    
    print("\n【生成的文件】")
    print("\n  图表文件:")
    for f in saved_files:
        print(f"    - {f}")
    
    print("\n  数据报表:")
    for name, f in report_files.items():
        print(f"    - {f}")
    
    print("\n" + "=" * 70)
    print("                      分析完成！")
    print("=" * 70)


def main():
    print_banner()
    
    output_dir = create_output_directory('output')
    
    print("\n" + "-" * 60)
    print("步骤 1: 加载数据")
    print("-" * 60)
    
    df = load_data()
    save_raw_data(df, os.path.join(output_dir, 'raw_data.csv'))
    print(f"数据加载完成，共 {len(df)} 条记录")
    print(df.head())
    
    print("\n" + "-" * 60)
    print("步骤 2: 数据预处理")
    print("-" * 60)
    
    df_processed = preprocess_data(df)
    print(f"预处理后数据形状: {df_processed.shape}")
    
    print("\n" + "-" * 60)
    print("步骤 3: 数据分析")
    print("-" * 60)
    
    analysis_results = run_analysis(df_processed)
    
    print("\n" + "-" * 60)
    print("步骤 4: 生成可视化图表")
    print("-" * 60)
    
    saved_files = run_visualization(
        df_processed, 
        analysis_results, 
        output_dir=output_dir,
        save_png=True,
        save_pdf=False
    )
    
    print("\n" + "-" * 60)
    print("步骤 5: 保存汇总报表")
    print("-" * 60)
    
    report_files = save_summary_report(
        analysis_results['summary'],
        analysis_results['monthly_stats'],
        analysis_results['quarterly_stats'],
        analysis_results['correlation_matrix'],
        output_path=os.path.join(output_dir, 'summary_report.csv')
    )
    
    print_summary(analysis_results, saved_files, report_files)
    
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
