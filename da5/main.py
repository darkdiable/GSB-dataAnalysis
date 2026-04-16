import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_acquisition import load_or_generate_data
from src.preprocessing import preprocess_pipeline
from src.feature_analysis import analyze_seasonality, analyze_correlations, generate_feature_summary
from src.model_prediction import train_arima_model, generate_prediction_summary
from src.visualization import generate_all_visualizations


def main():
    print("=" * 60)
    print("城市气象特征分析与短期气温预测系统")
    print("=" * 60)
    print()

    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)

    print("【1/5】数据获取...")
    data = load_or_generate_data(days=365)
    print(f"数据时间范围: {data['date'].min()} ~ {data['date'].max()}")
    print()

    print("【2/5】数据预处理...")
    data_cleaned = preprocess_pipeline(data)
    print()

    print("【3/5】特征分析...")
    seasonality_result = analyze_seasonality(data_cleaned)
    correlation_result = analyze_correlations(data_cleaned)
    feature_summary = generate_feature_summary(data_cleaned, seasonality_result, correlation_result)
    print()

    print("【4/5】ARIMA气温预测模型训练...")
    prediction_result = train_arima_model(data_cleaned, forecast_days=30)
    prediction_summary = generate_prediction_summary(prediction_result)
    print()

    print("【5/5】生成可视化报表...")
    generate_all_visualizations(
        data_cleaned,
        correlation_result['correlation_matrix'],
        prediction_result['forecast'],
        output_dir=output_dir
    )
    print()

    full_summary = feature_summary + "\n" + prediction_summary

    summary_path = os.path.join(output_dir, 'prediction_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(full_summary)
    print(f"预测结果摘要已保存至: {summary_path}")
    print()

    print("=" * 60)
    print("分析完成！输出文件:")
    print("=" * 60)
    print(f"1. {output_dir}/temperature_trend.png - 气温趋势折线图")
    print(f"2. {output_dir}/correlation_heatmap.png - 气象要素相关性热力图")
    print(f"3. {output_dir}/forecast_comparison.png - 预测结果对比图")
    print(f"4. {output_dir}/prediction_summary.txt - 气温预测结果摘要")
    print()
    print("=" * 60)
    print("预测摘要预览:")
    print("=" * 60)
    print(prediction_summary)


if __name__ == '__main__':
    main()
