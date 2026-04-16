import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.data_loader import DataLoader
from modules.data_cleaner import DataCleaner
from modules.feature_engineer import FeatureEngineer
from modules.model_trainer import ModelTrainer
from modules.visualizer import Visualizer


def generate_sample_data(output_path, days=730):
    np.random.seed(42)
    
    start_date = datetime.now() - timedelta(days=days)
    dates = [start_date + timedelta(days=i) for i in range(days)]
    
    trend = np.linspace(1000, 2000, days)
    
    seasonal = 200 * np.sin(2 * np.pi * np.arange(days) / 365)
    weekly = 50 * np.sin(2 * np.pi * np.arange(days) / 7)
    
    noise = np.random.normal(0, 100, days)
    
    sales = trend + seasonal + weekly + noise
    sales = np.maximum(sales, 0)
    
    missing_indices = np.random.choice(days, size=int(days * 0.02), replace=False)
    sales_with_missing = sales.copy()
    sales_with_missing[missing_indices] = np.nan
    
    outlier_indices = np.random.choice(days, size=int(days * 0.01), replace=False)
    sales_with_outliers = sales_with_missing.copy()
    sales_with_outliers[outlier_indices] = sales_with_outliers[outlier_indices] * np.random.uniform(2, 3, len(outlier_indices))
    
    df = pd.DataFrame({
        'date': dates,
        'sales': sales_with_outliers,
        'store_id': np.random.choice(['A', 'B', 'C'], days),
        'category': np.random.choice(['Electronics', 'Clothing', 'Food'], days)
    })
    
    df.to_csv(output_path, index=False)
    print(f"示例数据已生成: {output_path}")
    print(f"数据包含 {len(df)} 条记录，其中 {df['sales'].isna().sum()} 个缺失值")
    return df


def main():
    print("=" * 60)
    print("零售销售预测与可视化分析系统")
    print("=" * 60)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data', 'sales_data.csv')
    output_dir = os.path.join(base_dir, 'output')
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n步骤 1: 数据加载")
    print("-" * 40)
    if not os.path.exists(data_path):
        print("未找到数据文件，正在生成示例数据...")
        generate_sample_data(data_path)
    
    loader = DataLoader(data_path)
    data = loader.load_data()
    
    date_column = 'date'
    sales_column = 'sales'
    
    loader.validate_date_column(date_column)
    loader.validate_sales_column(sales_column)
    
    print("\n步骤 2: 数据清洗")
    print("-" * 40)
    cleaner = DataCleaner(data)
    
    cleaner.handle_missing_values(strategy='interpolate', columns=[sales_column])
    cleaner.handle_outliers(column=sales_column, method='iqr', action='clip')
    cleaner.remove_duplicates()
    
    cleaned_data = cleaner.data
    
    print("\n步骤 3: 特征工程")
    print("-" * 40)
    engineer = FeatureEngineer(cleaned_data, date_column, sales_column)
    features = engineer.create_features()
    
    print(f"生成特征数量: {len(features.columns) - 2}")
    
    print("\n步骤 4: 模型训练")
    print("-" * 40)
    trainer = ModelTrainer(features, sales_column, date_column)
    trainer.train_model(model_type='random_forest', n_estimators=100, max_depth=10)
    
    feature_importance = trainer.get_feature_importance()
    
    print("\n步骤 5: 未来30天销售预测")
    print("-" * 40)
    predictions = trainer.predict_future(days=30)
    
    print("\n步骤 6: 可视化分析")
    print("-" * 40)
    visualizer = Visualizer(output_dir)
    
    visualizer.plot_sales_trend(
        features, date_column, sales_column, 
        title='历史销售趋势分析'
    )
    
    visualizer.plot_prediction_comparison(
        features, predictions, date_column, sales_column,
        title='销售预测对比分析'
    )
    
    visualizer.plot_feature_importance(
        feature_importance, title='特征重要性分析 (Top 15)'
    )
    
    visualizer.plot_seasonal_decomposition(
        features, date_column, sales_column,
        title='时间序列分解分析'
    )
    
    model_metrics = trainer.get_model_metrics()
    report_path = visualizer.generate_report(
        cleaned_data, predictions, model_metrics, feature_importance
    )
    
    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)
    print(f"\n输出文件位置: {output_dir}")
    print("生成的文件:")
    print("  - sales_trend.png (销售趋势图)")
    print("  - prediction_comparison.png (预测对比图)")
    print("  - feature_importance.png (特征重要性图)")
    print("  - seasonal_decomposition.png (时间序列分解图)")
    print(f"  - sales_prediction_report.txt (预测报告)")
    
    print("\n预测结果摘要:")
    print("-" * 40)
    print(f"未来30天预测总销售额: {predictions['predicted_sales'].sum():.2f}")
    print(f"日均预测销售额: {predictions['predicted_sales'].mean():.2f}")
    print(f"预测销售额范围: {predictions['predicted_sales'].min():.2f} - {predictions['predicted_sales'].max():.2f}")
    
    return predictions


if __name__ == '__main__':
    main()
