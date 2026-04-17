"""
主程序：整合所有模块，输出分析报告
"""
import os
import sys
from datetime import datetime

import pandas as pd

from data_generator import BikeDataGenerator
from preprocessor import DataPreprocessor
from model_analyzer import TimeSeriesAnalyzer, DemandPredictor
from visualizer import BikeVisualizer


class BikeAnalysisPipeline:
    """共享单车数据分析流水线"""
    
    def __init__(self, base_dir: str = None):
        self.base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.base_dir, 'data')
        self.output_dir = os.path.join(self.base_dir, 'output')
        self.figures_dir = os.path.join(self.base_dir, 'figures')
        
        for dir_path in [self.data_dir, self.output_dir, self.figures_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        self.raw_df = None
        self.processed_df = None
        self.preprocessor = None
        self.ts_analyzer = None
        self.predictor = None
        self.visualizer = None
        self.report_lines = []
    
    def _add_to_report(self, text: str):
        """添加内容到报告"""
        self.report_lines.append(text)
    
    def step1_generate_data(self):
        """步骤1：生成模拟数据"""
        self._add_to_report("\n" + "=" * 60)
        self._add_to_report("步骤1：数据生成")
        self._add_to_report("=" * 60)
        
        generator = BikeDataGenerator(
            start_date='2023-01-01',
            end_date='2023-12-31',
            num_stations=10,
            random_seed=42
        )
        
        self.raw_df = generator.generate()
        
        data_path = os.path.join(self.data_dir, 'bike_data.csv')
        generator.save_data(self.raw_df, data_path)
        
        self._add_to_report(f"数据集生成完成")
        self._add_to_report(f"数据形状: {self.raw_df.shape}")
        self._add_to_report(f"时间范围: {self.raw_df['timestamp'].min()} 至 {self.raw_df['timestamp'].max()}")
        self._add_to_report(f"站点数量: {self.raw_df['station_id'].nunique()}")
        self._add_to_report(f"天气类型: {self.raw_df['weather'].unique().tolist()}")
        
        missing_count = self.raw_df['ridership'].isna().sum()
        self._add_to_report(f"缺失值数量: {missing_count} ({missing_count/len(self.raw_df)*100:.2f}%)")
        
        self._add_to_report(f"\n数据统计摘要:")
        self._add_to_report(f"骑行人数 - 均值: {self.raw_df['ridership'].mean():.2f}, "
                          f"标准差: {self.raw_df['ridership'].std():.2f}, "
                          f"最小值: {self.raw_df['ridership'].min():.0f}, "
                          f"最大值: {self.raw_df['ridership'].max():.0f}")
        
        print("步骤1完成：数据生成")
    
    def step2_preprocess_data(self):
        """步骤2：数据预处理"""
        self._add_to_report("\n" + "=" * 60)
        self._add_to_report("步骤2：数据预处理")
        self._add_to_report("=" * 60)
        
        self.preprocessor = DataPreprocessor(self.raw_df)
        self.processed_df = self.preprocessor.preprocess(missing_strategy='median')
        
        processed_path = os.path.join(self.data_dir, 'processed_data.csv')
        self.processed_df.to_csv(processed_path, index=False, encoding='utf-8')
        
        self._add_to_report(self.preprocessor.get_preprocessing_log())
        self._add_to_report(f"\n处理后数据形状: {self.processed_df.shape}")
        self._add_to_report(f"新增特征列: {self.preprocessor.get_feature_columns()}")
        
        print("步骤2完成：数据预处理")
    
    def step3_time_series_analysis(self):
        """步骤3：时间序列分析"""
        self._add_to_report("\n" + "=" * 60)
        self._add_to_report("步骤3：时间序列分析")
        self._add_to_report("=" * 60)
        
        self.ts_analyzer = TimeSeriesAnalyzer(self.processed_df)
        
        decomposition_result = self.ts_analyzer.decompose_time_series(period=7)
        self._add_to_report(f"\n时间序列分解 (周期=7天):")
        self._add_to_report(f"  趋势强度: {decomposition_result['trend_strength']}")
        self._add_to_report(f"  季节性强度: {decomposition_result['seasonal_strength']}")
        
        adf_result = self.ts_analyzer.adf_test()
        self._add_to_report(f"\nADF平稳性检验:")
        self._add_to_report(f"  ADF统计量: {adf_result['adf_statistic']}")
        self._add_to_report(f"  p值: {adf_result['p_value']}")
        self._add_to_report(f"  序列平稳性: {'平稳' if adf_result['is_stationary'] else '非平稳'}")
        
        hourly_pattern = self.ts_analyzer.analyze_hourly_pattern()
        self._add_to_report(f"\n小时骑行模式:")
        self._add_to_report(f"  高峰时段: {hourly_pattern['peak_hour']}:00 "
                          f"(平均{hourly_pattern['peak_avg_ridership']}人次)")
        self._add_to_report(f"  低谷时段: {hourly_pattern['valley_hour']}:00 "
                          f"(平均{hourly_pattern['valley_avg_ridership']}人次)")
        
        weekly_pattern = self.ts_analyzer.analyze_weekly_pattern()
        self._add_to_report(f"\n周骑行模式:")
        for day, avg in weekly_pattern['daily_avg'].items():
            self._add_to_report(f"  {day}: {avg}人次")
        self._add_to_report(f"  工作日平均: {weekly_pattern['workday_avg']}人次")
        self._add_to_report(f"  周末平均: {weekly_pattern['weekend_avg']}人次")
        
        print("步骤3完成：时间序列分析")
    
    def step4_demand_prediction(self):
        """步骤4：需求预测建模"""
        self._add_to_report("\n" + "=" * 60)
        self._add_to_report("步骤4：骑行需求预测建模")
        self._add_to_report("=" * 60)
        
        feature_columns = self.preprocessor.get_feature_columns()
        
        self.predictor = DemandPredictor(self.processed_df, feature_columns)
        self.predictor.prepare_data(test_size=0.2, random_state=42)
        
        self._add_to_report(f"\n训练集大小: {len(self.predictor.X_train)}")
        self._add_to_report(f"测试集大小: {len(self.predictor.X_test)}")
        self._add_to_report(f"特征数量: {len(feature_columns)}")
        
        metrics = self.predictor.train_model(n_estimators=100, max_depth=10, random_state=42)
        
        self._add_to_report(f"\n随机森林模型参数:")
        self._add_to_report(f"  树的数量: 100")
        self._add_to_report(f"  最大深度: 10")
        
        self._add_to_report(f"\n模型性能指标:")
        self._add_to_report(f"  均方误差 (MSE): {metrics['mse']}")
        self._add_to_report(f"  均方根误差 (RMSE): {metrics['rmse']}")
        self._add_to_report(f"  平均绝对误差 (MAE): {metrics['mae']}")
        self._add_to_report(f"  决定系数 (R²): {metrics['r2']}")
        
        feature_importance = self.predictor.get_feature_importance()
        self._add_to_report(f"\n特征重要性排序 (Top 10):")
        for idx, row in feature_importance.head(10).iterrows():
            self._add_to_report(f"  {row['feature']}: {row['importance']:.4f}")
        
        print("步骤4完成：需求预测建模")
    
    def step5_visualization(self):
        """步骤5：可视化分析"""
        self._add_to_report("\n" + "=" * 60)
        self._add_to_report("步骤5：可视化分析")
        self._add_to_report("=" * 60)
        
        self.visualizer = BikeVisualizer(self.processed_df, self.figures_dir)
        
        self.visualizer.plot_hourly_distribution()
        self._add_to_report("\n生成图表: 骑行量小时分布图")
        
        self.visualizer.plot_temperature_ridership()
        self._add_to_report("生成图表: 温度与骑行量关系散点图")
        
        self.visualizer.plot_weather_impact()
        self._add_to_report("生成图表: 天气状况对骑行量影响箱线图")
        
        if self.ts_analyzer.decomposition is not None:
            self.visualizer.plot_time_series_decomposition(self.ts_analyzer.decomposition)
            self._add_to_report("生成图表: 时间序列分解图")
        
        if self.predictor.feature_importance is not None:
            self.visualizer.plot_feature_importance(self.predictor.feature_importance)
            self._add_to_report("生成图表: 特征重要性图")
        
        print("步骤5完成：可视化分析")
    
    def generate_report(self):
        """生成分析报告"""
        report_path = os.path.join(self.output_dir, 'analysis_report.txt')
        
        header = []
        header.append("=" * 60)
        header.append("共享单车运营数据分析报告")
        header.append("=" * 60)
        header.append(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        header.append(f"分析工具: Python (pandas, statsmodels, sklearn, matplotlib)")
        
        footer = []
        footer.append("\n" + "=" * 60)
        footer.append("分析结论与建议")
        footer.append("=" * 60)
        
        if self.ts_analyzer and 'hourly_pattern' in self.ts_analyzer.analysis_results:
            hp = self.ts_analyzer.analysis_results['hourly_pattern']
            footer.append(f"\n1. 高峰时段调度建议:")
            footer.append(f"   - 早晚高峰({hp['peak_hour']}:00左右)需增加车辆投放")
            footer.append(f"   - 低谷时段({hp['valley_hour']}:00左右)可进行车辆维护")
        
        if self.predictor and self.predictor.feature_importance is not None:
            top_feature = self.predictor.feature_importance.iloc[0]['feature']
            footer.append(f"\n2. 关键影响因素:")
            footer.append(f"   - 最重要的预测特征: {top_feature}")
            footer.append(f"   - 建议重点关注该因素对运营的影响")
        
        if self.ts_analyzer and 'weekly_pattern' in self.ts_analyzer.analysis_results:
            wp = self.ts_analyzer.analysis_results['weekly_pattern']
            footer.append(f"\n3. 工作日与周末差异:")
            footer.append(f"   - 工作日平均骑行量: {wp['workday_avg']}人次")
            footer.append(f"   - 周末平均骑行量: {wp['weekend_avg']}人次")
            footer.append(f"   - 建议针对不同时段制定差异化运营策略")
        
        footer.append("\n" + "=" * 60)
        footer.append("报告结束")
        footer.append("=" * 60)
        
        full_report = header + self.report_lines + footer
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(full_report))
        
        print(f"\n分析报告已保存至: {report_path}")
        
        return report_path
    
    def run(self):
        """运行完整分析流程"""
        print("\n" + "=" * 60)
        print("共享单车数据分析流水线启动")
        print("=" * 60 + "\n")
        
        self.step1_generate_data()
        self.step2_preprocess_data()
        self.step3_time_series_analysis()
        self.step4_demand_prediction()
        self.step5_visualization()
        
        report_path = self.generate_report()
        
        print("\n" + "=" * 60)
        print("分析完成！")
        print("=" * 60)
        print(f"数据文件: {self.data_dir}")
        print(f"图表文件: {self.figures_dir}")
        print(f"分析报告: {report_path}")
        
        return report_path


def main():
    """主函数"""
    pipeline = BikeAnalysisPipeline()
    pipeline.run()


if __name__ == '__main__':
    main()
