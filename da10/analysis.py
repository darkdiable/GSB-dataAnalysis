import pandas as pd
import numpy as np
from scipy import stats


def calculate_monthly_stats(df):
    monthly = df.groupby(['year', 'month']).agg({
        'temperature': ['mean', 'min', 'max', 'std'],
        'humidity': ['mean', 'min', 'max', 'std'],
        'precipitation': ['mean', 'sum', 'max'],
        'wind_speed': ['mean', 'max', 'std']
    }).round(2)
    
    monthly.columns = ['_'.join(col).strip() for col in monthly.columns.values]
    monthly = monthly.reset_index()
    
    monthly['month_name'] = pd.to_datetime(
        monthly['year'].astype(str) + '-' + monthly['month'].astype(str) + '-01'
    ).dt.month_name()
    
    return monthly


def calculate_quarterly_stats(df):
    quarterly = df.groupby(['year', 'quarter']).agg({
        'temperature': ['mean', 'min', 'max', 'std'],
        'humidity': ['mean', 'min', 'max', 'std'],
        'precipitation': ['mean', 'sum', 'max'],
        'wind_speed': ['mean', 'max', 'std']
    }).round(2)
    
    quarterly.columns = ['_'.join(col).strip() for col in quarterly.columns.values]
    quarterly = quarterly.reset_index()
    
    quarter_map = {1: '春季(Q1)', 2: '夏季(Q2)', 3: '秋季(Q3)', 4: '冬季(Q4)'}
    quarterly['quarter_name'] = quarterly['quarter'].map(quarter_map)
    
    return quarterly


def calculate_correlation_matrix(df):
    numeric_cols = ['temperature', 'humidity', 'precipitation', 'wind_speed']
    correlation_matrix = df[numeric_cols].corr().round(3)
    return correlation_matrix


def detect_extreme_weather(df, temp_threshold=35, wind_threshold=15, precip_threshold=50):
    extreme_events = {
        'high_temp': df[df['temperature'] >= temp_threshold].copy(),
        'low_temp': df[df['temperature'] <= 0].copy(),
        'heavy_wind': df[df['wind_speed'] >= wind_threshold].copy(),
        'heavy_rain': df[df['precipitation'] >= precip_threshold].copy()
    }
    
    extreme_summary = {}
    for event_name, event_df in extreme_events.items():
        if len(event_df) > 0:
            event_df['year'] = event_df['date'].dt.year
            event_df['month'] = event_df['date'].dt.month
            
            yearly_count = event_df.groupby('year').size().to_dict()
            
            extreme_summary[event_name] = {
                'total_count': len(event_df),
                'yearly_count': yearly_count,
                'max_value': event_df[get_event_column(event_name)].max(),
                'mean_value': event_df[get_event_column(event_name)].mean().round(2),
                'most_common_month': event_df['month'].mode().tolist()
            }
        else:
            extreme_summary[event_name] = {
                'total_count': 0,
                'yearly_count': {},
                'max_value': None,
                'mean_value': None,
                'most_common_month': []
            }
    
    return extreme_events, extreme_summary


def get_event_column(event_name):
    column_map = {
        'high_temp': 'temperature',
        'low_temp': 'temperature',
        'heavy_wind': 'wind_speed',
        'heavy_rain': 'precipitation'
    }
    return column_map.get(event_name, 'temperature')


def analyze_trends(df):
    trends = {}
    
    yearly = df.groupby('year').agg({
        'temperature': 'mean',
        'humidity': 'mean',
        'precipitation': 'sum',
        'wind_speed': 'mean'
    }).reset_index()
    
    for col in ['temperature', 'humidity', 'precipitation', 'wind_speed']:
        x = yearly['year'].values
        y = yearly[col].values
        
        if len(x) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            trends[col] = {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value ** 2,
                'p_value': p_value,
                'trend_direction': '上升' if slope > 0 else '下降',
                'is_significant': p_value < 0.05
            }
        else:
            trends[col] = {
                'slope': 0,
                'intercept': 0,
                'r_squared': 0,
                'p_value': 1.0,
                'trend_direction': '无趋势',
                'is_significant': False
            }
    
    return trends, yearly


def generate_summary_statistics(df, monthly_stats, quarterly_stats, extreme_summary, trends):
    summary = {
        'data_range': {
            'start_date': df['date'].min().strftime('%Y-%m-%d'),
            'end_date': df['date'].max().strftime('%Y-%m-%d'),
            'total_days': len(df)
        },
        'overall_stats': {
            'temperature_mean': df['temperature'].mean().round(2),
            'temperature_min': df['temperature'].min().round(2),
            'temperature_max': df['temperature'].max().round(2),
            'humidity_mean': df['humidity'].mean().round(2),
            'precipitation_total': df['precipitation'].sum().round(2),
            'wind_speed_mean': df['wind_speed'].mean().round(2)
        },
        'extreme_weather': extreme_summary,
        'trends': trends
    }
    
    return summary


def save_summary_report(summary, monthly_stats, quarterly_stats, correlation_matrix, output_path='output/summary_report.csv'):
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    summary_df = pd.DataFrame({
        '指标': [
            '数据起始日期', '数据结束日期', '总天数',
            '平均温度(°C)', '最低温度(°C)', '最高温度(°C)',
            '平均湿度(%)', '总降水量(mm)', '平均风速(m/s)'
        ],
        '数值': [
            summary['data_range']['start_date'],
            summary['data_range']['end_date'],
            summary['data_range']['total_days'],
            summary['overall_stats']['temperature_mean'],
            summary['overall_stats']['temperature_min'],
            summary['overall_stats']['temperature_max'],
            summary['overall_stats']['humidity_mean'],
            summary['overall_stats']['precipitation_total'],
            summary['overall_stats']['wind_speed_mean']
        ]
    })
    
    base_path = output_path.rsplit('.', 1)[0]
    
    summary_df.to_csv(f'{base_path}_overview.csv', index=False, encoding='utf-8-sig')
    
    monthly_stats.to_csv(f'{base_path}_monthly.csv', index=False, encoding='utf-8-sig')
    
    quarterly_stats.to_csv(f'{base_path}_quarterly.csv', index=False, encoding='utf-8-sig')
    
    correlation_matrix.to_csv(f'{base_path}_correlation.csv', encoding='utf-8-sig')
    
    print(f"汇总报表已保存到: {base_path}_*.csv")
    
    return {
        'overview': f'{base_path}_overview.csv',
        'monthly': f'{base_path}_monthly.csv',
        'quarterly': f'{base_path}_quarterly.csv',
        'correlation': f'{base_path}_correlation.csv'
    }


def run_analysis(df):
    print("\n开始数据分析...")
    
    print("\n1. 计算月度统计指标...")
    monthly_stats = calculate_monthly_stats(df)
    print(f"  共计算 {len(monthly_stats)} 个月度统计")
    
    print("\n2. 计算季度统计指标...")
    quarterly_stats = calculate_quarterly_stats(df)
    print(f"  共计算 {len(quarterly_stats)} 个季度统计")
    
    print("\n3. 计算相关系数矩阵...")
    correlation_matrix = calculate_correlation_matrix(df)
    print("  相关系数矩阵计算完成")
    print(correlation_matrix)
    
    print("\n4. 检测极端天气事件...")
    extreme_events, extreme_summary = detect_extreme_weather(df)
    for event, stats in extreme_summary.items():
        print(f"  {event}: {stats['total_count']} 次")
    
    print("\n5. 分析趋势...")
    trends, yearly_stats = analyze_trends(df)
    for var, trend in trends.items():
        print(f"  {var}: {trend['trend_direction']}趋势 (p-value: {trend['p_value']:.4f})")
    
    print("\n6. 生成汇总统计...")
    summary = generate_summary_statistics(df, monthly_stats, quarterly_stats, extreme_summary, trends)
    
    print("\n数据分析完成！")
    
    return {
        'monthly_stats': monthly_stats,
        'quarterly_stats': quarterly_stats,
        'correlation_matrix': correlation_matrix,
        'extreme_events': extreme_events,
        'extreme_summary': extreme_summary,
        'trends': trends,
        'yearly_stats': yearly_stats,
        'summary': summary
    }


if __name__ == '__main__':
    from data_loader import generate_sample_data
    from preprocess import preprocess_data
    
    df = generate_sample_data()
    df_processed = preprocess_data(df)
    results = run_analysis(df_processed)
    print("\n月度统计前5行:")
    print(results['monthly_stats'].head())
