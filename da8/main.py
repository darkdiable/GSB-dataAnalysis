import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.generate_data import DataGenerator
from modules.preprocessing import DataPreprocessor
from modules.modeling import BikeDemandModel
from modules.visualization import DataVisualizer


def generate_report(df, preprocessing_log, model, output_path):
    lines = []
    
    lines.append("=" * 70)
    lines.append("共享单车运营数据分析报告")
    lines.append("=" * 70)
    lines.append(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    lines.append("-" * 70)
    lines.append("一、数据概览")
    lines.append("-" * 70)
    lines.append(f"数据时间范围: {df['timestamp'].min()} 至 {df['timestamp'].max()}")
    lines.append(f"数据记录总数: {len(df)}")
    lines.append(f"站点数量: {df['station_id'].nunique()}")
    lines.append(f"天气类型: {df['weather'].unique().tolist()}")
    lines.append("")
    
    lines.append("数据基本统计:")
    lines.append(f"  骑行量 - 均值: {df['ridership'].mean():.2f}, 标准差: {df['ridership'].std():.2f}")
    lines.append(f"  骑行量 - 最小值: {df['ridership'].min()}, 最大值: {df['ridership'].max()}")
    lines.append(f"  温度 - 均值: {df['temperature'].mean():.2f}°C, 范围: {df['temperature'].min():.1f}°C ~ {df['temperature'].max():.1f}°C")
    lines.append(f"  湿度 - 均值: {df['humidity'].mean():.2f}%, 范围: {df['humidity'].min():.1f}% ~ {df['humidity'].max():.1f}%")
    lines.append(f"  风速 - 均值: {df['wind_speed'].mean():.2f} m/s")
    lines.append("")
    
    lines.append("-" * 70)
    lines.append("二、数据预处理")
    lines.append("-" * 70)
    for log in preprocessing_log:
        lines.append(f"  {log}")
    lines.append("")
    
    lines.append("-" * 70)
    lines.append("三、时间序列分析")
    lines.append("-" * 70)
    lines.append(model.get_time_series_summary())
    lines.append("")
    
    lines.append("-" * 70)
    lines.append("四、随机森林预测模型")
    lines.append("-" * 70)
    lines.append(model.get_model_summary())
    lines.append("")
    lines.append(model.get_top_features(10))
    lines.append("")
    
    lines.append("-" * 70)
    lines.append("五、可视化报表")
    lines.append("-" * 70)
    lines.append("已生成以下可视化图表:")
    lines.append("  1. hourly_distribution.png - 骑行量小时分布图")
    lines.append("  2. temperature_ridership.png - 温度与骑行量关系散点图")
    lines.append("  3. weather_impact.png - 天气状况对骑行量影响箱线图")
    lines.append("  4. feature_importance.png - 特征重要性排名图")
    lines.append("  5. time_series_decomposition.png - 时间序列分解图")
    lines.append("")
    
    lines.append("-" * 70)
    lines.append("六、分析结论与建议")
    lines.append("-" * 70)
    lines.append("1. 骑行高峰时段: 早高峰(7-9点)和晚高峰(17-19点)骑行量最大")
    lines.append("2. 天气影响: 晴天和多云天气骑行量最高，雨天显著降低")
    lines.append("3. 温度影响: 15-28°C为最佳骑行温度区间")
    lines.append("4. 工作日效应: 工作日骑行量高于周末")
    lines.append("")
    lines.append("运营建议:")
    lines.append("  - 高峰时段增加车辆调度，确保站点车辆充足")
    lines.append("  - 雨天减少运维投入，降低运营成本")
    lines.append("  - 根据天气预报提前调整车辆分布")
    lines.append("  - 重点关注核心站点的早晚高峰需求")
    lines.append("")
    
    lines.append("=" * 70)
    lines.append("报告结束")
    lines.append("=" * 70)
    
    report_content = "\n".join(lines)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    return report_content


def main():
    print("=" * 50)
    print("共享单车运营数据分析系统")
    print("=" * 50)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    output_dir = os.path.join(base_dir, 'output')
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n[1/6] 生成模拟数据...")
    generator = DataGenerator(
        start_date='2023-01-01',
        end_date='2023-12-31',
        num_stations=10,
        seed=42
    )
    df = generator.generate()
    
    data_path = os.path.join(data_dir, 'bike_data.csv')
    generator.save_data(df, data_path)
    print(f"    数据记录数: {len(df)}")
    print(f"    数据已保存: {data_path}")
    
    print("\n[2/6] 数据预处理...")
    preprocessor = DataPreprocessor(df)
    df_processed, preprocessing_log = preprocessor.preprocess()
    X, y = preprocessor.prepare_model_data()
    print(f"    特征数量: {X.shape[1]}")
    print(f"    样本数量: {X.shape[0]}")
    
    print("\n[3/6] 时间序列分析...")
    model = BikeDemandModel()
    ts_analysis = model.time_series_analysis(df_processed)
    print(model.get_time_series_summary())
    
    print("\n[4/6] 构建随机森林预测模型...")
    metrics = model.train_random_forest(X, y)
    print(model.get_model_summary())
    
    print("\n[5/6] 生成可视化报表...")
    visualizer = DataVisualizer(output_dir)
    feature_importance = model.get_feature_importance()
    figures = visualizer.generate_all_plots(df_processed, feature_importance, ts_analysis)
    for name, path in figures.items():
        print(f"    {name}: {path}")
    
    print("\n[6/6] 生成分析报告...")
    report_path = os.path.join(output_dir, 'analysis_report.txt')
    report = generate_report(df_processed, preprocessing_log, model, report_path)
    print(f"    报告已保存: {report_path}")
    
    print("\n" + "=" * 50)
    print("分析完成!")
    print("=" * 50)
    
    return df_processed, model, report


if __name__ == '__main__':
    main()
