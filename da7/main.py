#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在线课程平台数据分析主程序
整合数据生成、预处理、建模和可视化全流程
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))

from modules.data_generator import DataGenerator
from modules.data_preprocessor import DataPreprocessor
from modules.model_trainer import ModelTrainer
from modules.visualizer import Visualizer


def print_header(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def print_section(title):
    print("\n" + "-"*50)
    print(f"  {title}")
    print("-"*50)


def main():
    print_header("在线课程平台数据分析系统")
    print("开始执行数据分析流程...\n")
    
    # 1. 数据模拟
    print_section("步骤 1: 数据模拟")
    generator = DataGenerator(n_samples=5000, random_state=42)
    raw_data = generator.generate_data()
    raw_data_path = generator.save_data(raw_data, 'data/raw_data.csv')
    print(f"✓ 原始数据已生成并保存")
    
    # 2. 数据预处理
    print_section("步骤 2: 数据预处理")
    preprocessor = DataPreprocessor()
    preprocessor.load_data('data/raw_data.csv')
    processed_data = preprocessor.preprocess()
    processed_data_path = preprocessor.save_processed_data('data/processed_data.csv')
    print(f"✓ 数据预处理完成")
    
    # 3. 建模分析
    print_section("步骤 3: 建模分析")
    trainer = ModelTrainer(random_state=42)
    trainer.load_data('data/processed_data.csv')
    trainer.prepare_features()
    trainer.train_model(n_estimators=100, max_depth=10)
    metrics = trainer.evaluate_model()
    feature_importance = trainer.get_feature_importance()
    trainer.save_results('output')
    print(f"✓ 模型训练和评估完成")
    
    # 4. 可视化
    print_section("步骤 4: 数据可视化")
    visualizer = Visualizer(figsize=(12, 8), dpi=300)
    visualizer.load_data('data/processed_data.csv', 'output/feature_importance.csv')
    visualizer.plot_category_completion_rate()
    visualizer.plot_duration_vs_score_scatter()
    visualizer.plot_feature_importance(feature_importance)
    print(f"✓ 可视化图表已生成")
    
    # 5. 输出最终报告
    print_header("分析结果汇总报告")
    
    print("\n【数据集概况】")
    print(f"  - 总样本数: {len(raw_data):,} 条")
    print(f"  - 特征数量: {len(processed_data.columns)} 个")
    print(f"  - 课程类别数: {raw_data['course_category'].nunique()} 个")
    print(f"  - 课程数量: {raw_data['course_id'].nunique()} 门")
    
    print("\n【数据质量】")
    completion_rate = raw_data['is_completed'].mean()
    print(f"  - 课程平均完成率: {completion_rate:.1%}")
    print(f"  - 平均观看时长: {raw_data['watch_duration_minutes'].mean():.1f} 分钟")
    print(f"  - 平均测验分数: {raw_data['quiz_score'].mean():.1f} 分")
    
    print("\n【模型评估结果】")
    for metric, value in metrics.items():
        print(f"  - {metric}: {value}")
    
    print("\n【特征重要性 TOP 3】")
    for idx, row in feature_importance.head(3).iterrows():
        print(f"  {idx+1}. {row['feature']}: {row['importance']:.4f}")
    
    print("\n【输出文件】")
    print(f"  - 原始数据: da7/data/raw_data.csv")
    print(f"  - 处理后数据: da7/data/processed_data.csv")
    print(f"  - 模型指标: da7/output/model_metrics.json")
    print(f"  - 特征重要性: da7/output/feature_importance.csv")
    print(f"  - 图表1: da7/output/figures/category_completion_rate.png")
    print(f"  - 图表2: da7/output/figures/duration_vs_score_scatter.png")
    print(f"  - 图表3: da7/output/figures/feature_importance.png")
    
    print_header("分析流程执行完毕！")
    
    return {
        'raw_data': raw_data,
        'processed_data': processed_data,
        'metrics': metrics,
        'feature_importance': feature_importance
    }


if __name__ == '__main__':
    results = main()
