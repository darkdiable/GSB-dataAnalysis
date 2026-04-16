#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
城市气象特征分析与短期气温预测系统
基于Python + pandas + matplotlib + statsmodels技术栈
"""

import os
import sys
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.data_fetcher import DataFetcher
from modules.data_preprocessor import DataPreprocessor
from modules.feature_analyzer import FeatureAnalyzer
from modules.predictor import TemperaturePredictor
from modules.visualizer import Visualizer


def main():
    print("=" * 70)
    print("城市气象特征分析与短期气温预测系统")
    print("=" * 70)
    
    # 创建输出目录
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    # ==================== 1. 数据获取 ====================
    print("\n【步骤1】数据获取...")
    fetcher = DataFetcher()
    
    # 生成模拟气象数据 (2020-2024年)
    print("  正在生成模拟气象数据...")
    raw_data = fetcher.generate_sample_data(
        start_date='2020-01-01',
        end_date='2024-12-31',
        city='北京'
    )
    print(f"  数据时间范围: {raw_data['date'].min()} 至 {raw_data['date'].max()}")
    print(f"  数据记录数: {len(raw_data)} 条")
    
    # 保存原始数据
    raw_data_path = os.path.join(data_dir, 'raw_weather_data.csv')
    fetcher.save_to_csv(raw_data_path)
    
    # ==================== 2. 数据预处理 ====================
    print("\n【步骤2】数据预处理...")
    preprocessor = DataPreprocessor(raw_data)
    
    # 处理缺失值
    print("  处理缺失值...")
    preprocessor.handle_missing_values(method='interpolate')
    
    # 剔除异常值
    print("  剔除异常值...")
    preprocessor.remove_outliers(method='iqr')
    
    # 添加时间特征
    print("  添加时间特征...")
    preprocessor.add_time_features()
    
    # 获取预处理后的数据
    processed_data = preprocessor.get_preprocessed_data()
    
    # 保存预处理后的数据
    processed_data_path = os.path.join(data_dir, 'processed_weather_data.csv')
    processed_data.to_csv(processed_data_path, index=False)
    print(f"  预处理后的数据已保存: {processed_data_path}")
    
    # 输出数据统计信息
    print("\n  数据统计信息:")
    stats = preprocessor.get_statistics()
    for col, stat in stats.items():
        print(f"    {col}: 均值={stat['mean']}, 标准差={stat['std']}, 范围=[{stat['min']}, {stat['max']}]")
    
    # ==================== 3. 特征分析 ====================
    print("\n【步骤3】特征分析...")
    analyzer = FeatureAnalyzer(processed_data)
    
    # 温度季节性分析
    print("  分析温度季节性特征...")
    seasonality = analyzer.analyze_temperature_seasonality()
    
    # 相关性分析
    print("  分析气象要素相关性...")
    correlations = analyzer.analyze_correlations()
    
    # 极端天气分析
    print("  分析极端天气事件...")
    extremes = analyzer.analyze_extreme_weather()
    
    # 趋势分析
    print("  分析长期趋势...")
    trends = analyzer.analyze_trends()
    
    # 生成分析摘要
    analysis_summary = analyzer.generate_analysis_summary()
    print("\n" + analysis_summary)
    
    # ==================== 4. 模型训练与预测 ====================
    print("\n【步骤4】ARIMA模型训练与预测...")
    predictor = TemperaturePredictor(processed_data)
    
    # 训练模型
    print("  训练ARIMA模型...")
    predictor.train_model(test_size=30, auto_select=True)
    
    # 输出模型摘要
    print("\n  模型摘要:")
    print(predictor.get_model_summary())
    
    # 进行预测
    print("\n  进行未来30天气温预测...")
    forecast_results = predictor.forecast(steps=30)
    
    # 生成预测摘要
    prediction_summary = predictor.generate_prediction_summary()
    print("\n" + prediction_summary)
    
    # ==================== 5. 可视化 ====================
    print("\n【步骤5】生成可视化报表...")
    visualizer = Visualizer(output_dir=output_dir)
    
    # 生成所有可视化图表
    viz_paths = visualizer.get_all_visualizations(processed_data, analyzer, predictor)
    
    print("\n  生成的可视化文件:")
    for path in viz_paths:
        print(f"    - {os.path.basename(path)}")
    
    # ==================== 6. 保存结果摘要 ====================
    print("\n【步骤6】保存结果摘要...")
    
    # 合并所有摘要
    full_summary = []
    full_summary.append("=" * 70)
    full_summary.append("城市气象特征分析与短期气温预测系统 - 结果报告")
    full_summary.append("=" * 70)
    full_summary.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    full_summary.append(f"数据时间范围: {processed_data['date'].min()} 至 {processed_data['date'].max()}")
    full_summary.append(f"数据记录数: {len(processed_data)} 条")
    
    full_summary.append("\n" + "=" * 70)
    full_summary.append(analysis_summary)
    
    full_summary.append("\n" + "=" * 70)
    full_summary.append(prediction_summary)
    
    full_summary.append("\n" + "=" * 70)
    full_summary.append("生成的文件列表")
    full_summary.append("=" * 70)
    full_summary.append("\n数据文件:")
    full_summary.append(f"  - {raw_data_path}")
    full_summary.append(f"  - {processed_data_path}")
    full_summary.append("\n可视化报表:")
    for path in viz_paths:
        full_summary.append(f"  - {path}")
    full_summary.append(f"\n本报告文件:")
    full_summary.append(f"  - {os.path.join(output_dir, 'prediction_summary.txt')}")
    
    # 保存摘要到文本文件
    summary_path = os.path.join(output_dir, 'prediction_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(full_summary))
    
    print(f"  结果摘要已保存: {summary_path}")
    
    # ==================== 完成 ====================
    print("\n" + "=" * 70)
    print("系统运行完成！")
    print("=" * 70)
    print(f"\n输出文件位置:")
    print(f"  数据文件: {data_dir}")
    print(f"  可视化报表: {output_dir}")
    print(f"  结果摘要: {summary_path}")
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
