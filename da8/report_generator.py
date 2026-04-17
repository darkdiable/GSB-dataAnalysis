"""
报告生成模块：生成TXT格式的分析报告
"""
import os
from datetime import datetime


class ReportGenerator:
    """分析报告生成器"""
    
    def __init__(self, output_dir='output'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.report_content = []
        
    def add_header(self, title="共享单车运营数据分析报告"):
        """添加报告标题"""
        self.report_content.append("="*70)
        self.report_content.append(f"{title:^70}")
        self.report_content.append("="*70)
        self.report_content.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.report_content.append("\n")
        
    def add_section(self, title):
        """添加章节标题"""
        self.report_content.append("\n" + "-"*70)
        self.report_content.append(f"  {title}")
        self.report_content.append("-"*70 + "\n")
        
    def add_subsection(self, title):
        """添加子章节标题"""
        self.report_content.append(f"\n【{title}】\n")
        
    def add_text(self, text):
        """添加文本内容"""
        self.report_content.append(text)
        
    def add_data_overview(self, df):
        """添加数据概览"""
        self.add_section("一、数据概览")
        
        self.add_subsection("数据集基本信息")
        self.add_text(f"  数据记录数: {len(df)}")
        self.add_text(f"  特征列数: {len(df.columns)}")
        self.add_text(f"  数据时间范围: {df['timestamp'].min()} 至 {df['timestamp'].max()}")
        
        self.add_subsection("字段说明")
        field_descriptions = {
            'timestamp': '时间戳，记录数据采集时间',
            'station_id': '站点ID，标识不同的共享单车站点',
            'weather': '天气状况，包括晴天、多云、阴天、小雨、大雨',
            'temperature': '温度，单位摄氏度(°C)',
            'humidity': '湿度，单位百分比(%)',
            'windspeed': '风速，单位千米/小时(km/h)',
            'ridership': '骑行人数，目标变量'
        }
        
        for field, desc in field_descriptions.items():
            if field in df.columns:
                self.add_text(f"  • {field}: {desc}")
        
        self.add_subsection("数值型特征统计")
        numeric_stats = df.describe()
        self.add_text(str(numeric_stats))
        
        self.add_subsection("类别型特征分布")
        if 'weather' in df.columns:
            weather_dist = df['weather'].value_counts()
            self.add_text("天气状况分布:")
            for weather, count in weather_dist.items():
                pct = count / len(df) * 100
                self.add_text(f"  {weather}: {count} ({pct:.1f}%)")
        
        if 'station_id' in df.columns:
            self.add_text(f"\n站点数量: {df['station_id'].nunique()}")
    
    def add_preprocessing_summary(self, missing_stats, new_features):
        """添加数据预处理摘要"""
        self.add_section("二、数据预处理")
        
        self.add_subsection("缺失值处理")
        if missing_stats:
            for col, info in missing_stats.items():
                self.add_text(f"  • {col}: {info}")
        else:
            self.add_text("  未发现缺失值")
        
        self.add_subsection("特征工程")
        self.add_text("提取的时间特征:")
        for feature in new_features:
            self.add_text(f"  • {feature}")
        
        self.add_text("\n天气状况独热编码:")
        self.add_text("  对weather字段进行独热编码，生成weather_晴天、weather_多云等特征")
    
    def add_time_series_analysis(self, ts_results):
        """添加时间序列分析结果"""
        self.add_section("三、时间序列分析")
        
        if 'periodicity' in ts_results:
            self.add_subsection("周期性分析")
            periodicity = ts_results['periodicity']
            self.add_text(f"  高峰时段: {periodicity['peak_hours']}")
            self.add_text(f"  低谷时段: {periodicity['low_hours']}")
        
        if 'trend' in ts_results:
            self.add_subsection("趋势分析")
            trend = ts_results['trend']
            self.add_text(f"  整体趋势: {trend['direction']}")
            self.add_text(f"  趋势斜率: {trend['slope']:.6f}")
        
        if 'weekday_weekend' in ts_results:
            self.add_subsection("工作日与周末对比")
            ww = ts_results['weekday_weekend']
            self.add_text(f"  工作日平均骑行量: {ww['weekday_avg']:.2f}")
            self.add_text(f"  周末平均骑行量: {ww['weekend_avg']:.2f}")
            self.add_text(f"  差异: {ww['difference']:.2f}")
        
        if 'decomposition' in ts_results and ts_results['decomposition']:
            self.add_subsection("季节性分解")
            decomp = ts_results['decomposition']
            self.add_text(f"  趋势均值: {decomp['trend_mean']:.2f}")
            self.add_text(f"  季节性标准差: {decomp['seasonal_std']:.2f}")
            self.add_text(f"  残差标准差: {decomp['residual_std']:.2f}")
    
    def add_modeling_results(self, metrics, feature_importance):
        """添加建模分析结果"""
        self.add_section("四、预测模型分析")
        
        self.add_subsection("模型选择")
        self.add_text("  使用随机森林回归模型(Random Forest Regressor)")
        self.add_text("  模型参数:")
        self.add_text("    - n_estimators: 100")
        self.add_text("    - max_depth: 15")
        self.add_text("    - min_samples_split: 5")
        self.add_text("    - min_samples_leaf: 2")
        
        self.add_subsection("模型性能评估")
        if 'train' in metrics:
            train_m = metrics['train']
            self.add_text("训练集指标:")
            self.add_text(f"  • R² 决定系数: {train_m['r2']:.4f}")
            self.add_text(f"  • RMSE 均方根误差: {train_m['rmse']:.4f}")
            self.add_text(f"  • MAE 平均绝对误差: {train_m['mae']:.4f}")
        
        if 'test' in metrics:
            test_m = metrics['test']
            self.add_text("\n测试集指标:")
            self.add_text(f"  • R² 决定系数: {test_m['r2']:.4f}")
            self.add_text(f"  • RMSE 均方根误差: {test_m['rmse']:.4f}")
            self.add_text(f"  • MAE 平均绝对误差: {test_m['mae']:.4f}")
        
        self.add_subsection("特征重要性排名")
        if feature_importance is not None:
            self.add_text("Top 10 重要特征:")
            for i, feat_info in enumerate(feature_importance[:10], 1):
                self.add_text(f"  {i:2d}. {feat_info['feature']}: {feat_info['importance']:.4f}")
    
    def add_visualization_summary(self, plot_files):
        """添加可视化摘要"""
        self.add_section("五、数据可视化")
        
        self.add_text("已生成以下可视化图表:")
        plot_descriptions = {
            '01_hourly_distribution.png': '骑行量小时分布图 - 展示不同时段的骑行需求规律',
            '02_temperature_ridership.png': '温度与骑行量关系散点图 - 分析温度对骑行的影响',
            '03_weather_boxplot.png': '天气状况对骑行量影响箱线图 - 对比不同天气下的骑行量分布',
            '04_feature_importance.png': '特征重要性排名图 - 展示随机森林模型的特征重要性',
            '05_time_series_decomposition.png': '时间序列分解图 - 展示趋势、季节性和残差成分'
        }
        
        for plot_file in plot_files:
            filename = os.path.basename(plot_file)
            desc = plot_descriptions.get(filename, '数据可视化图表')
            self.add_text(f"  • {filename}")
            self.add_text(f"    {desc}")
    
    def add_conclusions(self, insights):
        """添加结论与建议"""
        self.add_section("六、结论与建议")
        
        self.add_subsection("主要发现")
        for i, insight in enumerate(insights, 1):
            self.add_text(f"  {i}. {insight}")
        
        self.add_subsection("运营建议")
        suggestions = [
            "根据高峰时段（早7-9点，晚17-19点）增加车辆投放和调度",
            "关注天气预报，在晴天和多云天气增加车辆储备",
            "温度在20-28°C时骑行需求最高，可在此温度区间加强营销",
            "工作日骑行量高于周末，可针对周末推出促销活动",
            "优化站点布局，重点关注高需求时段的站点车辆配置"
        ]
        
        for i, suggestion in enumerate(suggestions, 1):
            self.add_text(f"  {i}. {suggestion}")
    
    def generate_report(self, report_data, filename='analysis_report.txt'):
        """
        生成完整分析报告
        
        Args:
            report_data: 包含所有分析数据的字典
            filename: 报告文件名
            
        Returns:
            str: 报告文件路径
        """
        print("\n" + "="*50)
        print("生成分析报告")
        print("="*50)
        
        # 清空之前的内容
        self.report_content = []
        
        # 添加报告头
        self.add_header()
        
        # 添加数据概览
        if 'data' in report_data:
            self.add_data_overview(report_data['data'])
        
        # 添加预处理摘要
        if 'preprocessing' in report_data:
            self.add_preprocessing_summary(
                report_data['preprocessing'].get('missing_stats', {}),
                report_data['preprocessing'].get('new_features', [])
            )
        
        # 添加时间序列分析
        if 'time_series' in report_data:
            self.add_time_series_analysis(report_data['time_series'])
        
        # 添加建模结果
        if 'modeling' in report_data:
            self.add_modeling_results(
                report_data['modeling'].get('metrics', {}),
                report_data['modeling'].get('feature_importance', [])
            )
        
        # 添加可视化摘要
        if 'plots' in report_data:
            self.add_visualization_summary(report_data['plots'])
        
        # 添加结论
        if 'insights' in report_data:
            self.add_conclusions(report_data['insights'])
        
        # 添加报告尾
        self.report_content.append("\n" + "="*70)
        self.report_content.append(f"{'报告结束':^70}")
        self.report_content.append("="*70 + "\n")
        
        # 保存报告
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.report_content))
        
        print(f"  报告已保存: {filepath}")
        return filepath
    
    def get_report_content(self):
        """获取报告内容"""
        return '\n'.join(self.report_content)


if __name__ == '__main__':
    # 测试报告生成
    generator = ReportGenerator()
    
    # 模拟报告数据
    import pandas as pd
    import numpy as np
    
    # 创建模拟数据
    np.random.seed(42)
    mock_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-06-01', periods=100, freq='H'),
        'station_id': np.random.randint(1, 11, 100),
        'weather': np.random.choice(['晴天', '多云', '阴天', '小雨', '大雨'], 100),
        'temperature': np.random.normal(25, 5, 100),
        'humidity': np.random.normal(65, 10, 100),
        'windspeed': np.random.gamma(2, 3, 100),
        'ridership': np.random.poisson(50, 100)
    })
    
    report_data = {
        'data': mock_data,
        'preprocessing': {
            'missing_stats': {'temperature': '使用中位数25.3填充5个缺失值'},
            'new_features': ['hour', 'day_of_week', 'is_weekday', 'month', 'is_rush_hour']
        },
        'time_series': {
            'periodicity': {'peak_hours': [8, 18, 19], 'low_hours': [3, 4, 5]},
            'trend': {'direction': '上升', 'slope': 0.05},
            'weekday_weekend': {'weekday_avg': 55.2, 'weekend_avg': 42.8, 'difference': 12.4}
        },
        'modeling': {
            'metrics': {
                'train': {'r2': 0.95, 'rmse': 8.5, 'mae': 6.2},
                'test': {'r2': 0.92, 'rmse': 10.3, 'mae': 7.8}
            },
            'feature_importance': [
                {'feature': 'hour', 'importance': 0.35},
                {'feature': 'temperature', 'importance': 0.25},
                {'feature': 'is_weekday', 'importance': 0.15}
            ]
        },
        'plots': ['output/plot1.png', 'output/plot2.png'],
        'insights': [
            '骑行需求呈现明显的日周期性，早晚高峰显著',
            '温度对骑行量有显著正向影响，最佳骑行温度为20-28°C',
            '工作日骑行量明显高于周末'
        ]
    }
    
    generator.generate_report(report_data)
    print("\n测试完成!")
