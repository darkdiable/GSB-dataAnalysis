import pandas as pd
import numpy as np


def analyze_seasonality(data):
    print("正在分析温度季节性特征...")
    
    monthly_stats = data.groupby('month')['temperature'].agg(['mean', 'min', 'max', 'std']).round(2)
    monthly_stats.columns = ['平均温度', '最低温度', '最高温度', '标准差']
    
    seasonal_stats = data.groupby('season')['temperature'].agg(['mean', 'min', 'max', 'std']).round(2)
    seasonal_stats.columns = ['平均温度', '最低温度', '最高温度', '标准差']
    
    season_order = ['春季', '夏季', '秋季', '冬季']
    seasonal_stats = seasonal_stats.reindex(season_order)
    
    return {
        'monthly_stats': monthly_stats,
        'seasonal_stats': seasonal_stats
    }


def analyze_correlations(data):
    print("正在分析气象要素相关性...")
    
    features = ['temperature', 'humidity', 'precipitation', 'wind_speed', 'pressure']
    corr_matrix = data[features].corr().round(3)
    
    corr_matrix.columns = ['温度', '湿度', '降水量', '风速', '气压']
    corr_matrix.index = ['温度', '湿度', '降水量', '风速', '气压']
    
    high_correlations = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) > 0.3:
                high_correlations.append({
                    '要素1': corr_matrix.columns[i],
                    '要素2': corr_matrix.columns[j],
                    '相关系数': corr_value
                })
    
    humidity_precip_corr = data['humidity'].corr(data['precipitation']).round(3)
    
    return {
        'correlation_matrix': corr_matrix,
        'high_correlations': high_correlations,
        'humidity_precipitation_correlation': humidity_precip_corr
    }


def generate_feature_summary(data, seasonality_result, correlation_result):
    summary = []
    summary.append("=" * 50)
    summary.append("气象特征分析报告")
    summary.append("=" * 50)
    summary.append("")
    
    summary.append("一、季节性统计:")
    summary.append("-" * 30)
    summary.append(seasonality_result['seasonal_stats'].to_string())
    summary.append("")
    
    summary.append("二、相关性分析:")
    summary.append("-" * 30)
    summary.append(f"湿度与降水量相关系数: {correlation_result['humidity_precipitation_correlation']}")
    summary.append("")
    summary.append("高相关性对:")
    for corr in correlation_result['high_correlations']:
        summary.append(f"  {corr['要素1']} - {corr['要素2']}: {corr['相关系数']}")
    summary.append("")
    
    return "\n".join(summary)
