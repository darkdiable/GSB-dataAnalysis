import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.data_loader import DataLoader
from modules.preprocessor import Preprocessor
from modules.feature_analyzer import FeatureAnalyzer
from modules.predictor import ARIMAPredictor
from modules.visualizer import Visualizer


def main():
    print("=" * 60)
    print("城市气象特征分析与短期气温预测系统")
    print("=" * 60)
    print()

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)

    print("[1/6] 数据获取...")
    loader = DataLoader()
    data = loader.generate_sample_data(start_date='2023-01-01', periods=365)
    print(f"  - 成功生成 {len(data)} 条气象数据记录")
    print(f"  - 数据列: {list(data.columns)}")
    print()

    print("[2/6] 数据预处理...")
    preprocessor = Preprocessor(data)
    data = preprocessor.fill_missing_values(method='interpolate')
    data = preprocessor.remove_outliers(columns=['temperature', 'humidity'], method='iqr', threshold=1.5)
    data = preprocessor.add_time_features()

    for log in preprocessor.get_preprocessing_log():
        print(f"  - {log}")
    print()

    print("[3/6] 特征分析...")
    analyzer = FeatureAnalyzer(data)

    monthly_temp, seasonal_stats = analyzer.analyze_temperature_seasonality()
    print("  温度季节性分析:")
    for season, temp in seasonal_stats.items():
        print(f"    - {season}: {temp:.2f}°C")

    correlation, p_value = analyzer.analyze_humidity_precipitation_correlation()
    print(f"\n  湿度与降水相关性分析:")
    print(f"    - 相关系数: {correlation:.4f}")
    print(f"    - P值: {p_value:.4f}")

    correlation_matrix = analyzer.calculate_correlation_matrix()
    print(f"\n  相关性矩阵已计算完成")

    temp_dist = analyzer.analyze_temperature_distribution()
    print(f"\n  温度分布统计:")
    print(f"    - 均值: {temp_dist['mean']:.2f}°C")
    print(f"    - 中位数: {temp_dist['median']:.2f}°C")
    print(f"    - 标准差: {temp_dist['std']:.2f}°C")

    extreme_stats = analyzer.analyze_extreme_weather()
    print(f"\n  极端天气统计:")
    print(f"    - 最冷日期: {extreme_stats['coldest_day']}")
    print(f"    - 最热日期: {extreme_stats['hottest_day']}")
    print()

    print("[4/6] ARIMA模型训练与预测...")
    predictor = ARIMAPredictor(data)

    stationarity_info = predictor.check_stationarity(data.set_index('date')['temperature'])
    print(f"  平稳性检验:")
    print(f"    - ADF统计量: {stationarity_info['adf_statistic']:.4f}")
    print(f"    - P值: {stationarity_info['p_value']:.4f}")
    print(f"    - 是否平稳: {'是' if stationarity_info['is_stationary'] else '否'}")

    results = predictor.fit_model(auto_select=True)
    model_summary = predictor.get_model_summary()
    print(f"\n  模型参数:")
    print(f"    - ARIMA阶数: {model_summary['order']}")
    print(f"    - AIC: {model_summary['aic']:.2f}")
    print(f"    - BIC: {model_summary['bic']:.2f}")

    forecast_df = predictor.predict(steps=7)
    print(f"\n  未来7天气温预测:")
    for idx, row in forecast_df.iterrows():
        print(f"    - {row['date'].strftime('%Y-%m-%d')}: {row['predicted_temperature']:.2f}°C")

    evaluation, predictions, actual = predictor.evaluate_model(test_size=30)
    print(f"\n  模型评估 (测试集30天):")
    print(f"    - MSE: {evaluation['mse']:.4f}")
    print(f"    - MAE: {evaluation['mae']:.4f}")
    print(f"    - RMSE: {evaluation['rmse']:.4f}")
    print()

    print("[5/6] 生成可视化报表...")
    visualizer = Visualizer(output_dir)

    trend_path = visualizer.plot_temperature_trend(data)
    print(f"  - 气温趋势图: {trend_path}")

    heatmap_path = visualizer.plot_correlation_heatmap(correlation_matrix)
    print(f"  - 相关性热力图: {heatmap_path}")

    comparison_path = visualizer.plot_prediction_comparison(actual.values, predictions.values)
    print(f"  - 预测对比图: {comparison_path}")

    seasonal_path = visualizer.plot_seasonal_analysis(data)
    print(f"  - 季节性分析图: {seasonal_path}")

    forecast_path = visualizer.plot_forecast(data, forecast_df)
    print(f"  - 气温预测图: {forecast_path}")
    print()

    print("[6/6] 生成预测结果摘要...")
    summary_path = os.path.join(output_dir, 'prediction_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("城市气象特征分析与短期气温预测系统 - 结果摘要\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("-" * 60 + "\n")
        f.write("一、数据概况\n")
        f.write("-" * 60 + "\n")
        f.write(f"数据时间范围: {data['date'].min().strftime('%Y-%m-%d')} 至 {data['date'].max().strftime('%Y-%m-%d')}\n")
        f.write(f"数据记录数: {len(data)}\n")
        f.write(f"气象要素: {', '.join(['temperature', 'humidity', 'precipitation', 'wind_speed', 'pressure'])}\n\n")

        f.write("-" * 60 + "\n")
        f.write("二、温度季节性分析\n")
        f.write("-" * 60 + "\n")
        for season, temp in seasonal_stats.items():
            season_cn = {'spring': '春季', 'summer': '夏季', 'autumn': '秋季', 'winter': '冬季'}
            f.write(f"{season_cn.get(season, season)}: {temp:.2f}°C\n")
        f.write("\n")

        f.write("-" * 60 + "\n")
        f.write("三、湿度与降水相关性分析\n")
        f.write("-" * 60 + "\n")
        f.write(f"相关系数: {correlation:.4f}\n")
        f.write(f"P值: {p_value:.4f}\n")
        f.write(f"统计显著性: {'显著' if p_value < 0.05 else '不显著'}\n\n")

        f.write("-" * 60 + "\n")
        f.write("四、温度分布统计\n")
        f.write("-" * 60 + "\n")
        f.write(f"均值: {temp_dist['mean']:.2f}°C\n")
        f.write(f"中位数: {temp_dist['median']:.2f}°C\n")
        f.write(f"标准差: {temp_dist['std']:.2f}°C\n")
        f.write(f"偏度: {temp_dist['skewness']:.4f}\n")
        f.write(f"峰度: {temp_dist['kurtosis']:.4f}\n\n")

        f.write("-" * 60 + "\n")
        f.write("五、极端天气统计\n")
        f.write("-" * 60 + "\n")
        f.write(f"最冷日期: {extreme_stats['coldest_day']}\n")
        f.write(f"最热日期: {extreme_stats['hottest_day']}\n")
        f.write(f"寒冷天数(低于10%分位): {extreme_stats['cold_days_count']}天\n")
        f.write(f"炎热天数(高于90%分位): {extreme_stats['hot_days_count']}天\n\n")

        f.write("-" * 60 + "\n")
        f.write("六、ARIMA模型信息\n")
        f.write("-" * 60 + "\n")
        f.write(f"模型阶数: {model_summary['order']}\n")
        f.write(f"AIC: {model_summary['aic']:.2f}\n")
        f.write(f"BIC: {model_summary['bic']:.2f}\n")
        f.write(f"平稳性: {'是' if stationarity_info['is_stationary'] else '否'}\n\n")

        f.write("-" * 60 + "\n")
        f.write("七、模型评估指标\n")
        f.write("-" * 60 + "\n")
        f.write(f"均方误差(MSE): {evaluation['mse']:.4f}\n")
        f.write(f"平均绝对误差(MAE): {evaluation['mae']:.4f}\n")
        f.write(f"均方根误差(RMSE): {evaluation['rmse']:.4f}\n\n")

        f.write("-" * 60 + "\n")
        f.write("八、未来7天气温预测\n")
        f.write("-" * 60 + "\n")
        for idx, row in forecast_df.iterrows():
            f.write(f"{row['date'].strftime('%Y-%m-%d')}: {row['predicted_temperature']:.2f}°C\n")
        f.write("\n")

        f.write("-" * 60 + "\n")
        f.write("九、生成的可视化文件\n")
        f.write("-" * 60 + "\n")
        f.write(f"1. 气温趋势图: {os.path.basename(trend_path)}\n")
        f.write(f"2. 相关性热力图: {os.path.basename(heatmap_path)}\n")
        f.write(f"3. 预测对比图: {os.path.basename(comparison_path)}\n")
        f.write(f"4. 季节性分析图: {os.path.basename(seasonal_path)}\n")
        f.write(f"5. 气温预测图: {os.path.basename(forecast_path)}\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write("报告生成完毕\n")

    print(f"  - 预测结果摘要: {summary_path}")
    print()

    print("=" * 60)
    print("系统运行完成!")
    print("=" * 60)
    print(f"\n所有输出文件已保存至: {output_dir}")


if __name__ == '__main__':
    main()
