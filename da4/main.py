"""
零售销售预测与可视化分析系统 - 主程序入口

功能：
1. 加载/生成零售销售数据
2. 数据清洗（缺失值处理、异常值检测）
3. 特征工程（时间序列特征提取）
4. 模型训练（基于scikit-learn的回归模型）
5. 预测未来30天销售额
6. 生成可视化报表（PNG格式）
7. 输出销售预测结果摘要
"""
import os
import sys
from datetime import datetime

# 添加模块路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.data_loader import DataLoader
from modules.data_cleaner import DataCleaner
from modules.feature_engineering import FeatureEngineer
from modules.model_trainer import ModelTrainer
from modules.visualization import Visualizer


def print_section(title):
    """打印分隔线标题"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def generate_summary_report(data_summary, cleaning_report, metrics, 
                           prediction_df, output_path):
    """
    生成销售预测结果摘要文本文件
    
    Parameters:
        data_summary: 数据摘要
        cleaning_report: 清洗报告
        metrics: 模型评估指标
        prediction_df: 预测结果
        output_path: 输出文件路径
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("       零售销售预测与可视化分析系统 - 预测结果摘要\n")
        f.write("=" * 60 + "\n\n")
        
        # 报告生成时间
        f.write(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 1. 数据摘要
        f.write("-" * 40 + "\n")
        f.write("【数据摘要】\n")
        f.write("-" * 40 + "\n")
        for key, value in data_summary.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        # 2. 数据清洗报告
        f.write("-" * 40 + "\n")
        f.write("【数据清洗报告】\n")
        f.write("-" * 40 + "\n")
        for key, value in cleaning_report.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        # 3. 模型评估指标
        f.write("-" * 40 + "\n")
        f.write("【模型评估指标】\n")
        f.write("-" * 40 + "\n")
        f.write(f"  模型类型: RandomForestRegressor\n")
        f.write(f"  训练集 R²: {metrics['train_r2']:.4f}\n")
        f.write(f"  测试集 R²: {metrics['test_r2']:.4f}\n")
        f.write(f"  训练集 MAE: {metrics['train_mae']:.2f}\n")
        f.write(f"  测试集 MAE: {metrics['test_mae']:.2f}\n")
        f.write(f"  训练集 RMSE: {metrics['train_rmse']:.2f}\n")
        f.write(f"  测试集 RMSE: {metrics['test_rmse']:.2f}\n")
        f.write("\n")
        
        # 4. 未来30天预测结果
        f.write("-" * 40 + "\n")
        f.write("【未来30天销售预测】\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'日期':<15} {'预测销售额':>15}\n")
        f.write("-" * 35 + "\n")
        
        total_pred = 0
        for _, row in prediction_df.iterrows():
            date_str = row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date'])[:10]
            pred_val = row['predicted_sales']
            total_pred += pred_val
            f.write(f"{date_str:<15} {pred_val:>15.2f}\n")
        
        f.write("-" * 35 + "\n")
        f.write(f"{'预测总额':<15} {total_pred:>15.2f}\n")
        f.write(f"{'预测日均':<15} {total_pred/len(prediction_df):>15.2f}\n")
        f.write("\n")
        
        # 5. 输出文件列表
        f.write("-" * 40 + "\n")
        f.write("【生成的可视化报表文件】\n")
        f.write("-" * 40 + "\n")
        f.write("  1. output/sales_trend.png - 销售趋势折线图\n")
        f.write("  2. output/prediction_comparison.png - 预测结果对比图\n")
        f.write("  3. output/feature_importance.png - 特征重要性柱状图\n")
        f.write("  4. output/seasonal_analysis.png - 季节性分析图\n")
        f.write("\n")
        
        f.write("=" * 60 + "\n")
        f.write("                      报告生成完成\n")
        f.write("=" * 60 + "\n")
    
    print(f"预测结果摘要已保存: {output_path}")


def main():
    """主程序入口"""
    print("\n" + "=" * 60)
    print("    零售销售预测与可视化分析系统")
    print("=" * 60)
    
    # 设置路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    output_dir = os.path.join(base_dir, 'output')
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # ==================== 1. 数据加载 ====================
    print_section("1. 数据加载")
    
    # 检查是否有数据文件，没有则生成模拟数据
    data_file = os.path.join(data_dir, 'sales_data.csv')
    loader = DataLoader(file_path=data_file if os.path.exists(data_file) else None)
    df_raw = loader.load_data()
    
    # 获取数据摘要
    data_summary = loader.get_data_summary(df_raw)
    print("\n数据摘要:")
    for key, value in data_summary.items():
        print(f"  {key}: {value}")
    
    # ==================== 2. 数据清洗 ====================
    print_section("2. 数据清洗")
    
    cleaner = DataCleaner(outlier_threshold=3)
    df_cleaned = cleaner.clean(df_raw)
    cleaning_report = cleaner.get_cleaning_report()
    
    print("\n清洗报告:")
    for key, value in cleaning_report.items():
        print(f"  {key}: {value}")
    
    # ==================== 3. 特征工程 ====================
    print_section("3. 特征工程")
    
    engineer = FeatureEngineer()
    df_features = engineer.extract_features(df_cleaned)
    
    print(f"\n提取的特征: {len(engineer.get_feature_names())} 个")
    print(f"特征示例: {', '.join(engineer.get_feature_names()[:5])}...")
    
    # ==================== 4. 模型训练 ====================
    print_section("4. 模型训练")
    
    trainer = ModelTrainer(model_type='random_forest')
    metrics = trainer.train(df_features, target_col='sales', test_size=0.2)
    
    print("\n模型评估指标:")
    print(f"  训练集 R²: {metrics['train_r2']:.4f}")
    print(f"  测试集 R²: {metrics['test_r2']:.4f}")
    print(f"  测试集 MAE: {metrics['test_mae']:.2f}")
    print(f"  测试集 RMSE: {metrics['test_rmse']:.2f}")
    
    # 获取特征重要性
    feature_importance = trainer.get_feature_importance(top_n=15)
    print("\nTop 5 重要特征:")
    for i, (_, row) in enumerate(feature_importance.head(5).iterrows(), 1):
        print(f"  {i}. {row['feature']}: {row['importance']:.4f}")
    
    # ==================== 5. 未来预测 ====================
    print_section("5. 未来30天销售预测")
    
    # 准备未来日期的特征
    last_date = df_cleaned['date'].max()
    df_future_features = engineer.prepare_future_features(
        last_date, days=30, historical_df=df_cleaned
    )
    
    # 预测未来销售额
    prediction_df = trainer.predict_future(df_future_features, historical_df=df_cleaned)
    
    print(f"\n预测日期范围: {prediction_df['date'].min().strftime('%Y-%m-%d')} 至 {prediction_df['date'].max().strftime('%Y-%m-%d')}")
    print(f"预测总额: {prediction_df['predicted_sales'].sum():.2f}")
    print(f"预测日均: {prediction_df['predicted_sales'].mean():.2f}")
    print(f"预测最高: {prediction_df['predicted_sales'].max():.2f}")
    print(f"预测最低: {prediction_df['predicted_sales'].min():.2f}")
    
    # ==================== 6. 可视化 ====================
    print_section("6. 生成可视化报表")
    
    visualizer = Visualizer(output_dir=output_dir)
    generated_files = visualizer.generate_all_visualizations(
        historical_df=df_cleaned,
        prediction_df=prediction_df,
        feature_importance_df=feature_importance
    )
    
    # ==================== 7. 生成摘要报告 ====================
    print_section("7. 生成预测结果摘要")
    
    summary_path = os.path.join(output_dir, 'prediction_summary.txt')
    generate_summary_report(
        data_summary=data_summary,
        cleaning_report=cleaning_report,
        metrics=metrics,
        prediction_df=prediction_df,
        output_path=summary_path
    )
    
    # ==================== 完成 ====================
    print_section("系统运行完成")
    print(f"\n所有输出文件保存在: {output_dir}/")
    print("\n生成的文件列表:")
    for i, filepath in enumerate(generated_files, 1):
        filename = os.path.basename(filepath)
        print(f"  {i}. {filename}")
    print(f"  {len(generated_files)+1}. prediction_summary.txt")
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
