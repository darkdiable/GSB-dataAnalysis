import os
import sys

from data_generator import generate_course_data
from data_preprocessor import DataPreprocessor
from model_trainer import ModelTrainer
from visualizer import Visualizer


def main():
    print("\n" + "=" * 60)
    print("在线课程平台数据分析工程")
    print("=" * 60)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    output_dir = os.path.join(base_dir, 'output')
    figures_dir = os.path.join(base_dir, 'figures')
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    print("\n【步骤1】数据模拟生成")
    print("-" * 60)
    data_path = os.path.join(data_dir, 'course_data.csv')
    raw_data = generate_course_data(
        n_samples=5000,
        save_path=data_path,
        random_state=42
    )
    
    print("\n【步骤2】数据预处理")
    print("-" * 60)
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.preprocess(raw_data)
    
    processed_data_path = os.path.join(data_dir, 'processed_data.csv')
    processed_data.to_csv(processed_data_path, index=False)
    print(f"预处理后数据已保存至: {processed_data_path}")
    
    print("\n【步骤3】建模分析")
    print("-" * 60)
    feature_columns = preprocessor.get_feature_columns()
    trainer = ModelTrainer(random_state=42)
    trainer.prepare_data(processed_data, feature_columns, target_column='completed')
    trainer.train(n_estimators=100, max_depth=10)
    
    evaluation_report = trainer.evaluate()
    feature_importance = trainer.get_feature_importance(feature_columns)
    trainer.save_results(output_dir)
    
    print("\n【步骤4】可视化分析")
    print("-" * 60)
    visualizer = Visualizer(output_dir=figures_dir, dpi=300)
    visualizer.plot_all(processed_data, feature_importance)
    
    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)
    print(f"\n输出文件位置:")
    print(f"  - 原始数据: {data_path}")
    print(f"  - 预处理数据: {processed_data_path}")
    print(f"  - 模型报告: {os.path.join(output_dir, 'model_report.txt')}")
    print(f"  - 特征重要性: {os.path.join(output_dir, 'feature_importance.csv')}")
    print(f"  - 可视化图表: {figures_dir}/")
    print("\n生成的可视化图表:")
    print(f"  - completion_rate_by_category.png (各类别课程平均完成率)")
    print(f"  - watch_duration_vs_quiz_score.png (观看时长与测验分数关系)")
    print(f"  - feature_importance.png (特征重要性排序)")
    
    return {
        'raw_data': raw_data,
        'processed_data': processed_data,
        'model': trainer.model,
        'feature_importance': feature_importance,
        'evaluation_report': evaluation_report
    }


if __name__ == "__main__":
    results = main()
