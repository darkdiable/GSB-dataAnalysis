#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
共享单车运营数据分析系统
========================

本程序执行完整的共享单车数据分析流程：
1. 数据生成与加载
2. 数据预处理
3. 时间序列分析
4. 预测模型构建
5. 可视化报表生成
6. 分析报告输出

作者: AI Assistant
日期: 2026-04-17
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# 导入各模块
from data_generator import BikeSharingDataGenerator
from data_preprocessor import DataPreprocessor
from modeling import ModelingPipeline, TimeSeriesAnalyzer, DemandPredictor
from visualization import BikeSharingVisualizer
from report_generator import ReportGenerator


def print_banner():
    """打印程序横幅"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║           共享单车运营数据分析系统                            ║
    ║           Bike Sharing Data Analysis System                  ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def step1_data_generation(n_samples=10000):
    """
    步骤1: 数据生成
    
    Args:
        n_samples: 样本数量
        
    Returns:
        pd.DataFrame: 生成的原始数据
    """
    print("\n" + "="*60)
    print("步骤 1/6: 数据生成")
    print("="*60)
    
    generator = BikeSharingDataGenerator(n_samples=n_samples, random_state=42)
    raw_data = generator.generate()
    
    # 保存原始数据
    data_path = os.path.join('output', 'raw_data.csv')
    os.makedirs('output', exist_ok=True)
    generator.save_to_csv(raw_data, data_path)
    
    print(f"\n数据生成完成:")
    print(f"  - 记录数: {len(raw_data)}")
    print(f"  - 特征数: {len(raw_data.columns)}")
    print(f"  - 时间范围: {raw_data['timestamp'].min()} 至 {raw_data['timestamp'].max()}")
    print(f"  - 缺失值统计:")
    for col, count in raw_data.isnull().sum().items():
        if count > 0:
            print(f"    {col}: {count} ({count/len(raw_data)*100:.1f}%)")
    
    return raw_data


def step2_data_preprocessing(raw_data):
    """
    步骤2: 数据预处理
    
    Args:
        raw_data: 原始数据
        
    Returns:
        tuple: (预处理后的数据, 特征矩阵X, 目标变量y, 特征列名)
    """
    print("\n" + "="*60)
    print("步骤 2/6: 数据预处理")
    print("="*60)
    
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.preprocess(raw_data)
    
    # 获取训练数据
    X, y, feature_cols = preprocessor.get_training_data(processed_data)
    
    # 记录预处理信息
    missing_stats = {}
    for col in ['temperature', 'humidity', 'windspeed']:
        missing_count = raw_data[col].isnull().sum()
        if missing_count > 0:
            missing_stats[col] = f"使用中位数 {raw_data[col].median():.2f} 填充 {missing_count} 个缺失值"
    
    new_features = ['hour', 'day_of_week', 'is_weekday', 'month', 'is_rush_hour']
    
    # 保存预处理后的数据
    processed_path = os.path.join('output', 'processed_data.csv')
    processed_data.to_csv(processed_path, index=False, encoding='utf-8-sig')
    print(f"\n预处理后的数据已保存: {processed_path}")
    
    preprocessing_info = {
        'missing_stats': missing_stats,
        'new_features': new_features
    }
    
    return processed_data, X, y, feature_cols, preprocessing_info


def step3_time_series_analysis(processed_data):
    """
    步骤3: 时间序列分析
    
    Args:
        processed_data: 预处理后的数据
        
    Returns:
        dict: 时间序列分析结果
    """
    print("\n" + "="*60)
    print("步骤 3/6: 时间序列分析")
    print("="*60)
    
    ts_analyzer = TimeSeriesAnalyzer()
    ts_results = ts_analyzer.analyze(processed_data)
    
    # 获取分解组件用于可视化
    decomposition_components = ts_analyzer.get_decomposition_components()
    
    return ts_results, decomposition_components


def step4_modeling(X, y, feature_cols):
    """
    步骤4: 预测建模
    
    Args:
        X: 特征矩阵
        y: 目标变量
        feature_cols: 特征列名
        
    Returns:
        dict: 建模结果
    """
    print("\n" + "="*60)
    print("步骤 4/6: 预测模型构建")
    print("="*60)
    
    predictor = DemandPredictor(n_estimators=100, random_state=42)
    metrics = predictor.train(X, y, test_size=0.2)
    
    feature_importance = predictor.get_feature_importance()
    
    modeling_results = {
        'metrics': metrics,
        'feature_importance': feature_importance.to_dict('records')
    }
    
    return modeling_results, feature_importance


def step5_visualization(processed_data, feature_importance, decomposition_components):
    """
    步骤5: 可视化
    
    Args:
        processed_data: 预处理后的数据
        feature_importance: 特征重要性DataFrame
        decomposition_components: 时间序列分解组件
        
    Returns:
        list: 生成的图表文件路径列表
    """
    print("\n" + "="*60)
    print("步骤 5/6: 生成可视化报表")
    print("="*60)
    
    visualizer = BikeSharingVisualizer(output_dir='output')
    plot_files = visualizer.generate_all_plots(
        processed_data, 
        feature_importance_df=feature_importance,
        decomposition_components=decomposition_components
    )
    
    return plot_files


def step6_report_generation(raw_data, preprocessing_info, ts_results, modeling_results, plot_files):
    """
    步骤6: 生成分析报告
    
    Args:
        raw_data: 原始数据
        preprocessing_info: 预处理信息
        ts_results: 时间序列分析结果
        modeling_results: 建模结果
        plot_files: 图表文件路径列表
        
    Returns:
        str: 报告文件路径
    """
    print("\n" + "="*60)
    print("步骤 6/6: 生成分析报告")
    print("="*60)
    
    # 生成洞察
    insights = []
    
    # 从时间序列分析中提取洞察
    if 'periodicity' in ts_results:
        peak_hours = ts_results['periodicity']['peak_hours']
        insights.append(f"骑行需求呈现明显的日周期性，高峰时段为 {peak_hours}")
    
    if 'weekday_weekend' in ts_results:
        diff = ts_results['weekday_weekend']['difference']
        if diff > 0:
            insights.append(f"工作日骑行量高于周末，平均差异为 {diff:.1f} 人次")
        else:
            insights.append(f"周末骑行量高于工作日，平均差异为 {abs(diff):.1f} 人次")
    
    if 'trend' in ts_results:
        insights.append(f"整体趋势呈{ts_results['trend']['direction']}态势")
    
    # 从建模结果中提取洞察
    if 'feature_importance' in modeling_results:
        top_feature = modeling_results['feature_importance'][0]
        insights.append(f"影响骑行需求的最重要因素是 '{top_feature['feature']}' (重要性: {top_feature['importance']:.3f})")
    
    if 'metrics' in modeling_results:
        r2 = modeling_results['metrics']['test']['r2']
        insights.append(f"随机森林模型在测试集上 R² = {r2:.4f}，具有较好的预测能力")
    
    # 整合报告数据
    report_data = {
        'data': raw_data,
        'preprocessing': preprocessing_info,
        'time_series': ts_results,
        'modeling': modeling_results,
        'plots': plot_files,
        'insights': insights
    }
    
    report_generator = ReportGenerator(output_dir='output')
    report_path = report_generator.generate_report(report_data, filename='analysis_report.txt')
    
    return report_path


def main():
    """主函数"""
    print_banner()
    
    # 创建输出目录
    os.makedirs('output', exist_ok=True)
    
    try:
        # 步骤1: 数据生成
        raw_data = step1_data_generation(n_samples=10000)
        
        # 步骤2: 数据预处理
        processed_data, X, y, feature_cols, preprocessing_info = step2_data_preprocessing(raw_data)
        
        # 步骤3: 时间序列分析
        ts_results, decomposition_components = step3_time_series_analysis(processed_data)
        
        # 步骤4: 预测建模
        modeling_results, feature_importance = step4_modeling(X, y, feature_cols)
        
        # 步骤5: 可视化
        plot_files = step5_visualization(processed_data, feature_importance, decomposition_components)
        
        # 步骤6: 生成报告
        report_path = step6_report_generation(
            raw_data, preprocessing_info, ts_results, modeling_results, plot_files
        )
        
        # 完成
        print("\n" + "="*60)
        print("分析完成!")
        print("="*60)
        print(f"\n输出文件:")
        print(f"  - 原始数据: output/raw_data.csv")
        print(f"  - 处理后数据: output/processed_data.csv")
        print(f"  - 分析报告: {report_path}")
        print(f"  - 可视化图表 ({len(plot_files)} 张):")
        for plot_file in plot_files:
            print(f"    • {os.path.basename(plot_file)}")
        
        print("\n" + "="*60)
        print("感谢使用共享单车运营数据分析系统!")
        print("="*60)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
