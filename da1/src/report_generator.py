import os
from datetime import datetime
from .config import OUTPUT_DIR

def generate_markdown_report(results):
    report_content = f"""# 全球燃油性能车深度分析报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. 数据预处理概述

- 数据集大小: {results['preprocessing']['shape']}
- 缺失值处理: 采用 MICE 多重插补法
- 异常值处理: IQR 方法检测并处理
  - 产量异常值: {results['preprocessing']['outliers']['production']['count']} 个
  - 销量异常值: {results['preprocessing']['outliers']['sales']['count']} 个

## 2. 时间序列分析

### 2.1 产销趋势与 CAGR

| 指标 | 年均复合增长率 (CAGR) |
|------|----------------------|
| 产量 | {results['time_series']['cagr_production']:.2%} |
| 销量 | {results['time_series']['cagr_sales']:.2%} |

### 2.2 ARIMA 销量预测 (2026-2027)

| 年份 | 预测销量 |
|------|----------|
"""
    for _, row in results['time_series']['forecast'].iterrows():
        report_content += f"| {row['year']} | {row['sales']:,.0f} |\n"
    
    report_content += f"""
## 3. 地理分析

### 3.1 市场集中度

- **CR4 (前4国市场份额)**: {results['geo']['concentration_ratios']['CR4']:.2%}
- **CR8 (前8国市场份额)**: {results['geo']['concentration_ratios']['CR8']:.2%}

### 3.2 Top5 国家销量排名

"""
    top5 = results['geo']['country_sales'].sort_values('sales', ascending=False).head()
    for i, (_, row) in enumerate(top5.iterrows(), 1):
        report_content += f"{i}. **{row['country']}**: {row['sales']:,.0f}\n"

    report_content += f"""
## 4. 市场细分分析

### 4.1 马力区间分布

"""
    hp_total = results['segment']['hp_distribution']['sales'].sum()
    for _, row in results['segment']['hp_distribution'].iterrows():
        report_content += f"- **{row['hp_range']}**: {row['sales']:,.0f} ({row['sales']/hp_total:.1%})\n"

    report_content += f"""
### 4.2 引擎类型分布

"""
    for _, row in results['segment']['engine_distribution'].iterrows():
        report_content += f"- **{row['engine_type']}**: {row['sales']:,.0f}\n"

    report_content += f"""
## 5. 统计建模结果

### 5.1 相关性分析

主要变量相关系数:
- 产量 vs 销量: {results['stat']['correlation'].loc['production', 'sales']:.3f}

### 5.2 公平性分析

- **基尼系数**: {results['stat']['gini_coefficient']:.3f}

### 5.3 国家聚类结果

市场聚类分组:

"""
    clusters = results['stat']['country_clusters']
    for c in sorted(clusters['cluster'].unique()):
        countries = clusters[clusters['cluster'] == c]['country'].tolist()
        report_content += f"- **聚类 {c}**: {', '.join(countries)}\n"

    report_content += f"""
## 6. 结论与建议

1. **市场趋势**: 全球燃油性能车市场呈现稳定增长态势，年复合增长率约 {results['time_series']['cagr_sales']:.1%}
2. **市场集中度**: 头部国家占据主导地位，CR4 超过 {results['geo']['concentration_ratios']['CR4']:.0%}
3. **产品结构**: 中高马力区间 (300-500hp) 占据最大市场份额
4. **区域差异**: 国家间销量分布存在差异，基尼系数为 {results['stat']['gini_coefficient']:.3f}

---

*本报告由自动化数据分析系统生成*
"""
    
    report_path = os.path.join(OUTPUT_DIR, 'analysis_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"Markdown报告已生成: {report_path}")
    return report_path
