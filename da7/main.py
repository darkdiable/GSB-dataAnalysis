#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings('ignore')

from data_generator import DataGenerator
from data_preprocessor import DataPreprocessor
from model_trainer import ModelTrainer
from visualizer import Visualizer

def main():
    print("="*70)
    print(" "*15 + "在线课程平台数据分析系统")
    print("="*70)
    
    print("\n【步骤1】生成模拟数据集...")
    generator = DataGenerator(random_state=42)
    raw_data = generator.generate_data(n_samples=5000)
    generator.save_data(raw_data)
    
    preprocessor = DataPreprocessor()
    preprocessor.get_basic_info(raw_data)
    
    print("\n【步骤2】数据预处理...")
    processed_data = preprocessor.preprocess(raw_data, encode=False)
    encoded_data = preprocessor.preprocess(raw_data, encode=True)
    
    print("\n【步骤3】构建预测模型...")
    trainer = ModelTrainer(random_state=42)
    trainer.prepare_data(encoded_data)
    trainer.train_model(n_estimators=100)
    
    metrics = trainer.evaluate_model()
    feature_importance = trainer.show_feature_importance()
    
    print("\n【步骤4】生成可视化报表...")
    visualizer = Visualizer(output_dir='plots', dpi=300)
    visualizer.generate_all_plots(processed_data, feature_importance)
    
    print("\n" + "="*70)
    print(" "*20 + "分析完成!")
    print("="*70)
    print("\n输出文件汇总:")
    print("- 数据集: data/course_data.csv")
    print("- 可视化图表: plots/")
    print("  1. category_completion_rate.png (各类别课程完成率)")
    print("  2. duration_vs_score_scatter.png (观看时长vs测验分数)")
    print("  3. feature_importance.png (特征重要性)")
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
