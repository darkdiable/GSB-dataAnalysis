import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')

from src.config import DATA_FILE
from src.preprocessing import preprocess_pipeline
from src.time_series import run_time_series_analysis
from src.geo_analysis import run_geo_analysis
from src.market_segment import run_market_segment_analysis
from src.stat_modeling import run_statistical_modeling
from src.report_generator import generate_markdown_report

def main():
    print("=" * 60)
    print("    全球燃油性能车深度分析与可视化项目")
    print("=" * 60)
    print()
    
    df_cleaned, df_scaled, outlier_report, scaler = preprocess_pipeline(DATA_FILE)
    print()
    
    ts_results = run_time_series_analysis(df_cleaned)
    print()
    
    geo_results = run_geo_analysis(df_cleaned)
    print()
    
    segment_results = run_market_segment_analysis(df_cleaned)
    print()
    
    stat_results = run_statistical_modeling(df_cleaned)
    print()
    
    all_results = {
        'preprocessing': {
            'shape': df_cleaned.shape,
            'outliers': outlier_report
        },
        'time_series': ts_results,
        'geo': geo_results,
        'segment': segment_results,
        'stat': stat_results
    }
    
    generate_markdown_report(all_results)
    
    print()
    print("=" * 60)
    print("    所有分析完成！输出文件已保存至 output/ 目录")
    print("=" * 60)

if __name__ == "__main__":
    main()
