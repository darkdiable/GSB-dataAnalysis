# -*- coding: utf-8 -*-
"""
电商销售数据分析与可视化项目 - 主入口

功能:
1. 生成/加载电商销售数据
2. 数据清洗（处理缺失值、异常值、重复数据）
3. 数据分析（描述统计、相关性、时间序列、用户聚类、销售预测）
4. 数据可视化（生成5张以上图表）
5. 输出分析报告

作者: Data Analysis Team
版本: 1.0.0
"""

import os
import sys
import pandas as pd
from datetime import datetime

# 添加模块路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))

from modules.data_generator import DataGenerator
from modules.data_cleaner import DataCleaner
from modules.data_analyzer import DataAnalyzer
from modules.visualizer import Visualizer


class DataAnalysisPipeline:
    """数据分析流程管道"""
    
    def __init__(self, data_dir='./data', output_dir='./output', 
                 n_records=800, use_existing_data=False):
        """
        初始化分析管道
        
        参数:
            data_dir: 数据目录
            output_dir: 输出目录
            n_records: 生成记录数
            use_existing_data: 是否使用已有数据
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.n_records = n_records
        self.use_existing_data = use_existing_data
        
        # 创建目录
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # 数据存储
        self.raw_data = None
        self.cleaned_data = None
        self.analysis_results = None
        
        # 报告内容
        self.report_content = []
        
    def step1_generate_data(self):
        """步骤1: 生成或加载数据"""
        print("\n" + "=" * 60)
        print("步骤 1/5: 数据生成/加载")
        print("=" * 60)
        
        raw_data_path = os.path.join(self.data_dir, 'sales_data_raw.csv')
        
        if self.use_existing_data and os.path.exists(raw_data_path):
            print(f"加载已有数据: {raw_data_path}")
            self.raw_data = pd.read_csv(raw_data_path)
        else:
            print(f"生成新的销售数据 ({self.n_records} 条)...")
            generator = DataGenerator(n_records=self.n_records)
            self.raw_data = generator.generate_data()
            
            # 添加一些缺失值和异常值用于测试清洗功能
            self.raw_data = generator.add_missing_values(self.raw_data, missing_rate=0.03)
            self.raw_data = generator.add_outliers(self.raw_data, outlier_rate=0.02)
            
            # 保存原始数据
            generator.save_data(self.raw_data, raw_data_path)
        
        print(f"原始数据形状: {self.raw_data.shape}")
        print(f"数据时间范围: {self.raw_data['交易时间'].min()} 至 {self.raw_data['交易时间'].max()}")
        
        return self.raw_data
    
    def step2_clean_data(self):
        """步骤2: 数据清洗"""
        print("\n" + "=" * 60)
        print("步骤 2/5: 数据清洗")
        print("=" * 60)
        
        if self.raw_data is None:
            raise ValueError("错误: 请先执行数据生成步骤")
        
        cleaner = DataCleaner(self.raw_data)
        self.cleaned_data = cleaner.clean_data()
        
        # 保存清洗后的数据
        cleaned_data_path = os.path.join(self.data_dir, 'sales_data_cleaned.csv')
        cleaner.save_cleaned_data(cleaned_data_path)
        
        # 记录清洗报告
        cleaning_report = cleaner.get_cleaning_report()
        self.report_content.append("\n" + "=" * 60)
        self.report_content.append("数据清洗报告")
        self.report_content.append("=" * 60)
        self.report_content.append(f"原始数据量: {cleaning_report.get('original_count', 'N/A')} 条")
        self.report_content.append(f"清洗后数据量: {cleaning_report.get('cleaned_count', 'N/A')} 条")
        self.report_content.append(f"删除/修正数据: {cleaning_report.get('removed_count', 'N/A')} 条")
        
        return self.cleaned_data
    
    def step3_analyze_data(self):
        """步骤3: 数据分析"""
        print("\n" + "=" * 60)
        print("步骤 3/5: 数据分析")
        print("=" * 60)
        
        if self.cleaned_data is None:
            raise ValueError("错误: 请先执行数据清洗步骤")
        
        analyzer = DataAnalyzer(self.cleaned_data)
        
        # 执行各项分析
        analyzer.descriptive_statistics()
        analyzer.correlation_analysis()
        analyzer.time_series_analysis()
        analyzer.user_clustering(n_clusters=4)
        analyzer.sales_prediction(forecast_days=30)
        
        self.analysis_results = analyzer.get_all_results()
        
        # 记录分析报告
        self._generate_analysis_report()
        
        return self.analysis_results
    
    def _generate_analysis_report(self):
        """生成分析报告的文本内容"""
        self.report_content.append("\n" + "=" * 60)
        self.report_content.append("数据分析报告")
        self.report_content.append("=" * 60)
        
        # 描述性统计
        if 'descriptive' in self.analysis_results:
            desc = self.analysis_results['descriptive']
            self.report_content.append("\n【描述性统计】")
            
            if 'basic_statistics' in desc:
                basic = desc['basic_statistics']
                self.report_content.append(f"  交易金额 - 均值: {basic.loc['mean', '交易金额']:.2f}, "
                                         f"标准差: {basic.loc['std', '交易金额']:.2f}")
                self.report_content.append(f"  交易金额 - 最小值: {basic.loc['min', '交易金额']:.2f}, "
                                         f"最大值: {basic.loc['max', '交易金额']:.2f}")
            
            if 'user_statistics' in desc:
                user_stats = desc['user_statistics']
                for key, value in user_stats.items():
                    self.report_content.append(f"  {key}: {value}")
        
        # 相关性分析
        if 'correlation' in self.analysis_results:
            corr = self.analysis_results['correlation']
            self.report_content.append("\n【相关性分析】")
            if 'strong_correlations' in corr and corr['strong_correlations']:
                for item in corr['strong_correlations']:
                    self.report_content.append(f"  {item['变量1']} - {item['变量2']}: "
                                             f"r = {item['相关系数']}")
            else:
                self.report_content.append("  未发现强相关关系 (|r| > 0.5)")
        
        # 时间序列分析
        if 'time_series' in self.analysis_results:
            ts = self.analysis_results['time_series']
            self.report_content.append("\n【时间序列分析】")
            self.report_content.append(f"  趋势方向: {ts.get('trend_direction', 'N/A')}")
            self.report_content.append(f"  趋势斜率: {ts.get('trend_slope', 0):.4f} (每天变化)")
            self.report_content.append(f"  R²: {ts.get('r_squared', 0):.4f}")
        
        # 聚类分析
        if 'clustering' in self.analysis_results:
            cluster = self.analysis_results['clustering']
            self.report_content.append("\n【用户聚类分析 (K-Means)】")
            self.report_content.append(f"  聚类数量: 4")
            
            if 'cluster_names' in cluster:
                for label, name in cluster['cluster_names'].items():
                    self.report_content.append(f"  聚类{label}: {name}")
        
        # 预测分析
        if 'prediction' in self.analysis_results:
            pred = self.analysis_results['prediction']
            self.report_content.append("\n【销售预测 (线性回归)】")
            self.report_content.append(f"  测试集 MAE: {pred.get('test_mae', 0):.2f}")
            self.report_content.append(f"  测试集 RMSE: {pred.get('test_rmse', 0):.2f}")
            self.report_content.append(f"  测试集 R²: {pred.get('test_r2', 0):.4f}")
            
            if 'future_predictions' in pred:
                future = pred['future_predictions']
                self.report_content.append(f"  未来30天预测总销售额: {future['预测销售额'].sum():.2f}")
                self.report_content.append(f"  未来30天预测日均销售额: {future['预测销售额'].mean():.2f}")
    
    def step4_visualize(self):
        """步骤4: 数据可视化"""
        print("\n" + "=" * 60)
        print("步骤 4/5: 数据可视化")
        print("=" * 60)
        
        if self.cleaned_data is None or self.analysis_results is None:
            raise ValueError("错误: 请先执行数据分析步骤")
        
        visualizer = Visualizer(output_dir=self.output_dir)
        visualizer.generate_all_visualizations(self.cleaned_data, self.analysis_results)
        
        return True
    
    def step5_generate_report(self):
        """步骤5: 生成分析报告"""
        print("\n" + "=" * 60)
        print("步骤 5/5: 生成分析报告")
        print("=" * 60)
        
        # 构建完整报告
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("电商销售数据分析报告")
        report_lines.append("=" * 60)
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"数据记录数: {len(self.cleaned_data) if self.cleaned_data is not None else 'N/A'}")
        report_lines.append("")
        
        # 添加各步骤报告
        report_lines.extend(self.report_content)
        
        # 添加结论
        report_lines.append("\n" + "=" * 60)
        report_lines.append("分析结论")
        report_lines.append("=" * 60)
        
        if 'time_series' in self.analysis_results:
            trend = self.analysis_results['time_series'].get('trend_direction', 'N/A')
            report_lines.append(f"\n1. 销售趋势: 数据显示销售额整体呈{trend}趋势。")
        
        if 'clustering' in self.analysis_results:
            report_lines.append("\n2. 用户分层: 通过K-Means聚类将用户分为4个群体，")
            report_lines.append("   建议针对不同群体制定差异化营销策略。")
        
        if 'prediction' in self.analysis_results:
            r2 = self.analysis_results['prediction'].get('test_r2', 0)
            report_lines.append(f"\n3. 销售预测: 线性回归模型R²={r2:.4f}，")
            if r2 > 0.7:
                report_lines.append("   模型拟合效果良好，预测结果可信度高。")
            elif r2 > 0.5:
                report_lines.append("   模型拟合效果一般，建议结合其他因素进行预测。")
            else:
                report_lines.append("   模型拟合效果较差，建议考虑更复杂的预测模型。")
        
        report_lines.append("\n" + "=" * 60)
        report_lines.append("报告结束")
        report_lines.append("=" * 60)
        
        # 保存报告
        report_text = "\n".join(report_lines)
        report_path = os.path.join(self.output_dir, 'summary.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"分析报告已保存: {report_path}")
        print("\n报告内容预览:")
        print("-" * 60)
        print(report_text)
        print("-" * 60)
        
        return report_path
    
    def run(self):
        """运行完整的数据分析流程"""
        print("\n" + "=" * 60)
        print("电商销售数据分析系统启动")
        print("=" * 60)
        
        try:
            # 执行各步骤
            self.step1_generate_data()
            self.step2_clean_data()
            self.step3_analyze_data()
            self.step4_visualize()
            self.step5_generate_report()
            
            print("\n" + "=" * 60)
            print("数据分析流程全部完成!")
            print("=" * 60)
            print(f"数据文件位置: {self.data_dir}/")
            print(f"图表输出位置: {self.output_dir}/")
            print(f"分析报告位置: {self.output_dir}/summary.txt")
            print("=" * 60)
            
            return True
            
        except Exception as e:
            print(f"\n错误: 数据分析流程执行失败")
            print(f"错误信息: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """主函数"""
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # 创建并运行分析管道
    pipeline = DataAnalysisPipeline(
        data_dir='./data',
        output_dir='./output',
        n_records=800,  # 生成800条数据
        use_existing_data=False  # 每次运行生成新数据
    )
    
    success = pipeline.run()
    
    if success:
        print("\n✓ 所有任务执行成功!")
        return 0
    else:
        print("\n✗ 任务执行失败!")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
