#!/usr/bin/env python3

import os
import sys
import time
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src import (
    BikeShareDataGenerator,
    DataPreprocessor,
    TimeSeriesAnalyzer,
    DemandPredictor,
    Visualizer
)


def create_directories():
    dirs = ['data', 'output/figures', 'output/report']
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def generate_analysis_report(results, output_path='output/report/analysis_report.txt'):
    print(f"\n正在生成分析报告: {output_path}")
    
    report = []
    report.append("=" * 80)
    report.append("                     共享单车运营数据分析报告")
    report.append(f"                    生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 80)
    report.append("")
    
    report.append("一、数据集概览")
    report.append("-" * 60)
    ds = results['dataset_stats']
    report.append(f"  数据时间范围: {ds['time_start']} 至 {ds['time_end']}")
    report.append(f"  总记录数: {ds['total_records']:,}")
    report.append(f"  站点数量: {ds['num_stations']}")
    report.append(f"  总骑行人次: {ds['total_riders']:,}")
    report.append(f"  日均骑行人次: {ds['daily_avg_riders']:,.0f}")
    report.append(f"  每小时平均骑行人次: {ds['hourly_avg_riders']:.1f}")
    report.append("")
    
    report.append("二、时间序列分析结果")
    report.append("-" * 60)
    ts = results['time_series']
    report.append(f"  趋势强度: {ts['trend_strength']:.4f}")
    report.append(f"  周周期强度: {ts['weekly_seasonal']:.4f}")
    report.append(f"  日周期强度: {ts['daily_seasonal']:.4f}")
    report.append("")
    report.append("  ADF平稳性检验结果:")
    report.append(f"    ADF统计量: {ts['adf_test']['ADF统计量']}")
    report.append(f"    p值: {ts['adf_test']['p值']}")
    if ts['adf_test']['p值'] < 0.05:
        report.append("    结论: 序列具有平稳性 (p < 0.05)")
    else:
        report.append("    结论: 序列不具有平稳性")
    report.append("")
    
    report.append("三、需求预测模型分析")
    report.append("-" * 60)
    model = results['model_metrics']
    report.append("  随机森林回归模型性能:")
    report.append(f"    训练集 R²: {model['train_r2']:.4f}")
    report.append(f"    测试集 R²: {model['test_r2']:.4f}")
    report.append(f"    训练集 RMSE: {model['train_rmse']:.2f}")
    report.append(f"    测试集 RMSE: {model['test_rmse']:.2f}")
    report.append("")
    report.append(f"  {results['cv_scores'].shape[0]} 折交叉验证:")
    report.append(f"    平均 R²: {results['cv_scores'].mean():.4f} (±{results['cv_scores'].std():.4f})")
    report.append("")
    
    report.append("  特征重要性排名 TOP 10:")
    for idx, row in results['feature_importance'].head(10).iterrows():
        report.append(f"    {idx+1:2d}. {row['feature']:<25} {row['importance']:.4f}")
    report.append("")
    
    report.append("四、可视化图表")
    report.append("-" * 60)
    report.append("  已生成以下高清图表 (PNG格式, 300DPI):")
    for name, path in results['chart_paths'].items():
        report.append(f"    - {name}: {path}")
    report.append("")
    
    report.append("五、关键发现与建议")
    report.append("-" * 60)
    
    if ts['daily_seasonal'] > 0.5:
        report.append("  1. 骑行需求具有强烈的日周期性，早晚高峰特征明显")
        report.append("     - 建议: 在早7-9点、晚17-19点高峰时段增加车辆调度")
    
    if ts['weekly_seasonal'] > 0.3:
        report.append("  2. 骑行需求具有显著周度季节性，工作日与周末差异大")
        report.append("     - 建议: 工作日重点保障通勤站点，周末重点保障商圈/景点站点")
    
    weather_feats = results['feature_importance'][
        results['feature_importance']['feature'].str.contains('weather')
    ]
    if weather_feats['importance'].sum() > 0.05:
        report.append("  3. 天气状况对骑行需求影响显著")
        report.append("     - 建议: 根据天气预报提前调整车辆投放策略，雨天减少户外车辆")
    
    temp_importance = results['feature_importance'][
        results['feature_importance']['feature'] == 'temperature'
    ]['importance'].values[0]
    if temp_importance > 0.1:
        report.append("  4. 温度是影响骑行需求的重要环境因素")
        report.append("     - 建议: 极端温度天气下适当调整运营策略")
    
    report.append("")
    report.append("=" * 80)
    report.append("                          报告结束")
    report.append("=" * 80)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"分析报告已生成: {output_path}")
    return output_path


def main():
    print("\n" + "=" * 80)
    print("                    共享单车数据分析系统启动")
    print("=" * 80 + "\n")
    
    start_time = time.time()
    
    create_directories()
    
    print("【步骤1/5】生成模拟数据集...")
    generator = BikeShareDataGenerator(start_date='2024-01-01', days=180, stations=20)
    df_raw = generator.generate_dataset()
    generator.save_dataset(df_raw, 'data/bike_share_data.csv')
    print(f"数据集生成完成，共 {len(df_raw):,} 条记录\n")
    
    print("【步骤2/5】数据预处理...")
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.preprocess_pipeline(df_raw)
    print()
    
    print("【步骤3/5】时间序列分析...")
    ts_analyzer = TimeSeriesAnalyzer()
    ts_results = ts_analyzer.analyze_periodicity(df_processed)
    print()
    
    print("【步骤4/5】构建需求预测模型...")
    predictor = DemandPredictor()
    X, y, _ = predictor.prepare_features(df_processed)
    metrics, *_ = predictor.train_model(X, y, n_estimators=100)
    cv_scores = predictor.cross_validation(X, y, cv=5)
    print()
    
    print("【步骤5/5】生成可视化图表...")
    visualizer = Visualizer()
    chart_paths = visualizer.generate_all_charts(
        df_processed,
        feature_importance=predictor.feature_importance,
        decomposition=ts_results['decomposition']
    )
    print()
    
    dataset_stats = {
        'time_start': df_processed['timestamp'].min().strftime('%Y-%m-%d'),
        'time_end': df_processed['timestamp'].max().strftime('%Y-%m-%d'),
        'total_records': len(df_processed),
        'num_stations': df_processed['station_id'].nunique(),
        'total_riders': df_processed['riders'].sum(),
        'daily_avg_riders': df_processed.groupby('date')['riders'].sum().mean(),
        'hourly_avg_riders': df_processed['riders'].mean()
    }
    
    results = {
        'dataset_stats': dataset_stats,
        'time_series': {
            'trend_strength': ts_results['daily_trend_strength'],
            'weekly_seasonal': ts_results['weekly_seasonal_strength'],
            'daily_seasonal': ts_results['daily_seasonal_strength'],
            'adf_test': ts_results['adf_test']
        },
        'model_metrics': metrics,
        'cv_scores': cv_scores,
        'feature_importance': predictor.feature_importance,
        'chart_paths': chart_paths
    }
    
    report_path = generate_analysis_report(results)
    
    elapsed_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("                          分析完成!")
    print("=" * 80)
    print(f"总耗时: {elapsed_time:.2f} 秒")
    print(f"分析报告: {report_path}")
    print(f"图表目录: output/figures/")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
