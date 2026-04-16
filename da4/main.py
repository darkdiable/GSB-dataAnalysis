import os
import sys
from datetime import datetime

from data_loader import load_data, save_predictions
from data_cleaner import clean_data
from feature_engineering import engineer_features, create_future_features
from model_trainer import train_model, predict_future
from visualizer import generate_all_charts


def generate_summary_report(df, forecast_df, metrics, feature_importance, output_path='output/prediction_summary.txt'):
    print("\n生成销售预测结果摘要...")
    
    total_historical_sales = df['sales'].sum()
    avg_daily_sales = df['sales'].mean()
    max_daily_sales = df['sales'].max()
    min_daily_sales = df['sales'].min()
    
    total_predicted_sales = forecast_df['predicted_sales'].sum()
    avg_predicted_sales = forecast_df['predicted_sales'].mean()
    max_predicted_sales = forecast_df['predicted_sales'].max()
    min_predicted_sales = forecast_df['predicted_sales'].min()
    
    growth_rate = ((avg_predicted_sales - avg_daily_sales) / avg_daily_sales) * 100
    
    summary = f"""
{'='*70}
            零售销售预测与分析系统 - 结果摘要报告
{'='*70}

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

【1. 历史销售数据统计】
   数据时间范围: {df['date'].min().strftime('%Y-%m-%d')} 至 {df['date'].max().strftime('%Y-%m-%d')}
   总数据天数: {len(df)} 天
   历史总销售额: {total_historical_sales:,.2f}
   日均销售额: {avg_daily_sales:,.2f}
   最高日销售额: {max_daily_sales:,.2f}
   最低日销售额: {min_daily_sales:,.2f}

【2. 模型性能评估】
   训练集 R² 得分: {metrics['train_r2']:.4f}
   测试集 R² 得分: {metrics['test_r2']:.4f}
   测试集 MAE: {metrics['test_mae']:.2f}
   测试集 RMSE: {metrics['test_rmse']:.2f}

【3. 未来30天销售预测】
   预测时间范围: {forecast_df['date'].min().strftime('%Y-%m-%d')} 至 {forecast_df['date'].max().strftime('%Y-%m-%d')}
   预测总销售额: {total_predicted_sales:,.2f}
   预测日均销售额: {avg_predicted_sales:,.2f}
   预测最高日销售额: {max_predicted_sales:,.2f} ({forecast_df.loc[forecast_df['predicted_sales'].idxmax(), 'date'].strftime('%Y-%m-%d')})
   预测最低日销售额: {min_predicted_sales:,.2f} ({forecast_df.loc[forecast_df['predicted_sales'].idxmin(), 'date'].strftime('%Y-%m-%d')})
   预计增长率: {growth_rate:+.2f}%

【4. 重要特征 Top 10】
"""
    
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
        summary += f"   {i}. {row['feature']}: {row['importance']:.4f}\n"
    
    summary += f"""

【5. 输出文件清单】
   - 销售趋势图: output/sales_trend.png
   - 预测结果对比图: output/forecast_comparison.png
   - 特征重要性图: output/feature_importance.png
   - 预测结果数据: output/predictions.csv
   - 预测摘要报告: output/prediction_summary.txt

{'='*70}
                           报告结束
{'='*70}
"""
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(summary)
    print(f"\n摘要报告已保存至: {output_path}")
    
    return summary


def main():
    print("\n" + "=" * 70)
    print("          零售销售预测与可视化分析系统")
    print("=" * 70 + "\n")
    
    df = load_data()
    
    df_cleaned = clean_data(df)
    
    df_features, feature_columns = engineer_features(df_cleaned)
    
    model, metrics, feature_importance = train_model(df_features, feature_columns)
    
    last_date = df_features['date'].iloc[-1]
    last_known_sales = df_features['sales'].iloc[-1]
    last_trend_value = df_features['trend'].iloc[-1]
    
    future_df = create_future_features(last_date, days=30, last_trend_value=last_trend_value)
    
    forecast_df = predict_future(model, future_df, feature_columns, last_known_sales)
    
    generate_all_charts(df_cleaned, forecast_df, feature_importance)
    
    save_predictions(forecast_df[['date', 'predicted_sales']])
    
    generate_summary_report(df_cleaned, forecast_df, metrics, feature_importance)
    
    print("\n" + "=" * 70)
    print("        系统运行完成! 所有输出文件已生成")
    print("=" * 70)


if __name__ == "__main__":
    main()
