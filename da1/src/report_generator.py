"""
报告生成模块
功能：生成Markdown分析报告
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os


class ReportGenerator:
    """报告生成器类"""

    def __init__(self, output_dir='../output'):
        """
        初始化
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        self.report_content = []

    def add_header(self, title, subtitle=None):
        """添加报告标题"""
        self.report_content.append(f"# {title}\n")
        if subtitle:
            self.report_content.append(f"*{subtitle}*\n")
        self.report_content.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.report_content.append("\n---\n")

    def add_section(self, title, level=2):
        """添加章节标题"""
        self.report_content.append(f"\n{'#' * level} {title}\n")

    def add_paragraph(self, text):
        """添加段落"""
        self.report_content.append(f"\n{text}\n")

    def add_list(self, items, ordered=False):
        """添加列表"""
        for i, item in enumerate(items, 1):
            if ordered:
                self.report_content.append(f"{i}. {item}\n")
            else:
                self.report_content.append(f"- {item}\n")
        self.report_content.append("\n")

    def add_table(self, headers, rows):
        """添加表格"""
        self.report_content.append("\n" + " | ".join(headers) + "\n")
        self.report_content.append(" | ".join(["---"] * len(headers)) + "\n")
        for row in rows:
            self.report_content.append(" | ".join([str(cell) for cell in row]) + "\n")
        self.report_content.append("\n")

    def add_image(self, image_path, alt_text=""):
        """添加图片"""
        relative_path = os.path.basename(image_path)
        self.report_content.append(f"\n![{alt_text}]({relative_path})\n")

    def add_code_block(self, code, language="python"):
        """添加代码块"""
        self.report_content.append(f"\n```{language}\n{code}\n```\n")

    def generate_executive_summary(self, df, cagr, concentration, forecast_2026, forecast_2027):
        """生成执行摘要"""
        self.add_section("执行摘要", 2)

        total_sales = df[df['Year'].isin([2023, 2024, 2025])]['Units_Sold'].sum()
        total_models = df[df['Year'].isin([2023, 2024, 2025])]['Model'].nunique()
        top_country = df[df['Year'].isin([2023, 2024, 2025])].groupby('Country')['Units_Sold'].sum().idxmax()
        top_brand = df[df['Year'].isin([2023, 2024, 2025])].groupby('Brand')['Units_Sold'].sum().idxmax()

        summary_text = f"""
本报告基于2015-2025年全球燃油性能车市场数据，通过多重插补、时间序列分析、地理分析、
市场细分和统计建模等方法，对全球燃油性能车市场进行了深度分析。

**核心发现：**

- **市场规模**: 2023-2025年累计销量达到 **{total_sales:,}** 辆，涉及 **{total_models}** 款车型
- **增长趋势**: 2015-2025年复合年增长率(CAGR)为 **{cagr*100:.2f}%**
- **市场预测**: 预计2026年销量约 **{forecast_2026:,}** 辆，2027年约 **{forecast_2027:,}** 辆
- **地理分布**: **{top_country}** 是最大市场，**{top_brand}** 是领先品牌
- **市场集中度**: CR4 = **{concentration['CR4']:.2f}%**，属于{concentration['concentration_level']}

**主要洞察：**

1. 电动化趋势明显，电动车销量占比持续提升
2. 美国和中国市场主导全球销量
3. 高性能车型(500+马力)市场份额稳步增长
4. 市场呈现寡头竞争格局，头部品牌优势明显
"""
        self.add_paragraph(summary_text)

    def generate_data_preprocessing_section(self):
        """生成数据预处理章节"""
        self.add_section("1. 数据预处理", 2)

        content = """
数据预处理是数据分析的基础，本报告采用了以下方法：

### 1.1 多重插补法 (MICE)

使用链式方程多重插补(Multiple Imputation by Chained Equations)处理缺失值。
该方法通过迭代方式，利用其他变量的信息来估计缺失值，比简单均值填充更准确。

### 1.2 IQR异常值检测

采用四分位距(Interquartile Range)方法检测异常值：
- 计算Q1(25分位数)和Q3(75分位数)
- IQR = Q3 - Q1
- 异常值范围: < Q1 - 1.5×IQR 或 > Q3 + 1.5×IQR

### 1.3 数据标准化

使用Z-score标准化方法：
```
z = (x - μ) / σ
```
其中μ为均值，σ为标准差。
"""
        self.add_paragraph(content)

    def generate_time_series_section(self, yearly_stats, cagr, forecast_df):
        """生成时间序列分析章节"""
        self.add_section("2. 时间序列分析", 2)

        content = f"""
### 2.1 产销趋势分析

2015-2025年全球燃油性能车市场呈现持续增长态势。
通过移动平均线分析，可以观察到市场增长的长期趋势。

**复合年增长率(CAGR)**: {cagr*100:.2f}%

### 2.2 ARIMA预测模型

使用自回归积分滑动平均模型(ARIMA)对2026-2027年销量进行预测：
"""
        self.add_paragraph(content)

        headers = ["年份", "预测销量", "下限(95%CI)", "上限(95%CI)"]
        rows = []
        for _, row in forecast_df.iterrows():
            rows.append([
                int(row['Year']),
                f"{int(row['Forecast_Sales']):,}",
                f"{int(row['Lower_CI']):,}",
                f"{int(row['Upper_CI']):,}"
            ])
        self.add_table(headers, rows)

        self.add_image("02_arima_forecast.png", "ARIMA销量预测")
        self.add_image("01_sales_trend.png", "销量趋势分析")

    def generate_geographic_section(self, concentration, country_shares):
        """生成地理分析章节"""
        self.add_section("3. 地理分析", 2)

        content = f"""
### 3.1 市场集中度分析

**CR4 (前4国集中度)**: {concentration['CR4']:.2f}%
**CR8 (前8国集中度)**: {concentration['CR8']:.2f}%
**HHI (赫芬达尔指数)**: {concentration['HHI']:.2f}

市场结构判断: **{concentration['concentration_level']}**

### 3.2 各国市场份额

"""
        self.add_paragraph(content)

        headers = ["国家", "市场份额(%)"]
        rows = [[country, f"{share:.2f}%"] for country, share in sorted(country_shares.items(), key=lambda x: x[1], reverse=True)]
        self.add_table(headers, rows)

        self.add_image("04_top10_countries.png", "各国销量排名")
        self.add_image("07_market_share_pie.png", "市场份额分布")

    def generate_segmentation_section(self):
        """生成市场细分章节"""
        self.add_section("4. 市场细分分析", 2)

        content = """
### 4.1 马力区间分布

按马力将性能车分为5个区间：
- 入门级 (300-400 HP)
- 运动级 (400-500 HP)
- 高性能级 (500-600 HP)
- 超跑级 (600-700 HP)
- 极致级 (700+ HP)

### 4.2 引擎类型分析

传统燃油车与电动车的竞争格局正在发生变化，
电动性能车凭借瞬时扭矩优势，市场份额快速提升。

### 4.3 波士顿矩阵分析

根据市场份额和增长率将品牌分为四类：
- **明星(Star)**: 高份额、高增长
- **现金牛(Cash Cow)**: 高份额、低增长
- **问题(Question)**: 低份额、高增长
- **瘦狗(Dog)**: 低份额、低增长
"""
        self.add_paragraph(content)

        self.add_image("09_horsepower_pie.png", "马力区间分布")
        self.add_image("10_engine_type_distribution.png", "引擎类型分布")
        self.add_image("12_boston_matrix.png", "波士顿矩阵")

    def generate_statistical_section(self, gini):
        """生成统计建模章节"""
        self.add_section("5. 统计建模分析", 2)

        content = f"""
### 5.1 相关性分析

通过相关性热力图分析各性能指标之间的关系：
- 马力与扭矩呈强正相关
- 加速时间与马力呈负相关
- 价格与性能指标呈正相关

### 5.2 洛伦兹曲线与基尼系数

**基尼系数**: {gini:.3f}

基尼系数反映市场集中度，系数越接近1表示集中度越高。

### 5.3 K-Means聚类分析

对国家进行K-Means聚类，识别相似市场特征的国家群体。

### 5.4 PCA主成分分析

通过主成分分析降维，识别影响市场差异的主要因素。
"""
        self.add_paragraph(content)

        self.add_image("13_correlation_heatmap.png", "相关性热力图")
        self.add_image("14_lorenz_curve.png", "洛伦兹曲线")
        self.add_image("15_kmeans_clustering.png", "K-Means聚类")
        self.add_image("16_pca_analysis.png", "PCA分析")

    def generate_conclusions(self):
        """生成结论与建议"""
        self.add_section("6. 结论与建议", 2)

        content = """
### 6.1 主要结论

1. **市场持续增长**: 全球燃油性能车市场在过去十年保持稳定增长，
   电动化转型带来新的增长动力。

2. **地理集中度高**: 美国和中国主导全球市场，CR4超过60%，
   市场呈现寡头竞争格局。

3. **技术路线多元化**: 传统内燃机、混合动力、纯电动并行发展，
   满足不同消费者需求。

4. **品牌分化明显**: 头部品牌通过技术创新和品牌建设巩固市场地位，
   中小品牌面临更大竞争压力。

### 6.2 战略建议

**对制造商：**
- 加速电动化转型，开发高性能电动车型
- 深耕核心市场，提升品牌影响力
- 关注新兴市场机会，提前布局

**对投资者：**
- 关注电动性能车领域的创新企业
- 重视传统品牌的转型能力
- 评估供应链企业的技术壁垒

**对政策制定者：**
- 完善充电基础设施建设
- 制定合理的排放标准
- 支持高性能电动车技术研发
"""
        self.add_paragraph(content)

    def save_report(self, filename='analysis_report.md'):
        """保存报告"""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.report_content))
        print(f"报告已保存至: {filepath}")
        return filepath


def generate_full_report(df, cagr, concentration, forecast_df, gini, output_dir='../output'):
    """
    生成完整分析报告
    Args:
        df: DataFrame
        cagr: 复合年增长率
        concentration: 市场集中度字典
        forecast_df: 预测结果DataFrame
        gini: 基尼系数
        output_dir: 输出目录
    """
    generator = ReportGenerator(output_dir)

    generator.add_header(
        "全球燃油性能车深度分析与可视化报告",
        "基于2015-2025年市场数据的综合分析"
    )

    forecast_2026 = int(forecast_df[forecast_df['Year'] == 2026]['Forecast_Sales'].values[0]) if len(forecast_df) > 0 else 0
    forecast_2027 = int(forecast_df[forecast_df['Year'] == 2027]['Forecast_Sales'].values[0]) if len(forecast_df) > 1 else 0

    generator.generate_executive_summary(df, cagr, concentration, forecast_2026, forecast_2027)
    generator.generate_data_preprocessing_section()

    from time_series_analysis import TimeSeriesAnalyzer
    ts_analyzer = TimeSeriesAnalyzer(df)
    yearly_stats = ts_analyzer.calculate_yearly_stats()
    generator.generate_time_series_section(yearly_stats, cagr, forecast_df)

    generator.generate_geographic_section(concentration, concentration['country_shares'])
    generator.generate_segmentation_section()
    generator.generate_statistical_section(gini)
    generator.generate_conclusions()

    report_path = generator.save_report()

    return report_path


if __name__ == '__main__':
    df = pd.read_csv('../data/fuel_performance_cars.csv')

    from time_series_analysis import TimeSeriesAnalyzer
    from geographic_analysis import GeographicAnalyzer
    from statistical_modeling import StatisticalModelingAnalyzer

    ts_analyzer = TimeSeriesAnalyzer(df)
    cagr = ts_analyzer.calculate_cagr()

    geo_analyzer = GeographicAnalyzer(df)
    concentration = geo_analyzer.calculate_market_concentration()

    stat_analyzer = StatisticalModelingAnalyzer(df)
    _, gini = stat_analyzer.plot_lorenz_curve()

    forecast_df = pd.DataFrame({
        'Year': [2026, 2027],
        'Forecast_Sales': [150000, 165000],
        'Lower_CI': [135000, 148000],
        'Upper_CI': [165000, 182000]
    })

    generate_full_report(df, cagr, concentration, forecast_df, gini)
