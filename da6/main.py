import os
import sys
from datetime import datetime

# 添加modules目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))

from modules.data_generator import ECommerceDataGenerator
from modules.data_preprocessor import DataPreprocessor
from modules.modeling import RepurchasePredictor
from modules.visualization import ECommerceVisualizer


class ECommerceAnalysisPipeline:
    """电商用户行为分析主流程"""

    def __init__(self, n_users=5000, n_records=50000, random_seed=42):
        self.n_users = n_users
        self.n_records = n_records
        self.random_seed = random_seed

        # 路径配置
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.base_dir, 'data')
        self.output_dir = os.path.join(self.base_dir, 'output')
        self.reports_dir = os.path.join(self.base_dir, 'reports')

        # 创建目录
        for dir_path in [self.data_dir, self.output_dir, self.reports_dir]:
            os.makedirs(dir_path, exist_ok=True)

        # 数据存储
        self.raw_df = None
        self.processed_df = None
        self.rfm_df = None
        self.model_results = None
        self.feature_importance = None

        # 报告内容
        self.report_content = []

    def add_report_section(self, title, content):
        """添加报告章节"""
        self.report_content.append(f"\n{'='*60}")
        self.report_content.append(f"【{title}】")
        self.report_content.append('='*60)
        if isinstance(content, list):
            self.report_content.extend(content)
        else:
            self.report_content.append(str(content))

    def step1_generate_data(self):
        """步骤1：生成模拟数据"""
        print("\n" + "="*60)
        print("步骤1: 生成电商用户行为数据集")
        print("="*60)

        generator = ECommerceDataGenerator(
            n_users=self.n_users,
            n_records=self.n_records,
            random_seed=self.random_seed
        )

        data_path = os.path.join(self.data_dir, 'ecommerce_behavior.csv')
        self.raw_df = generator.generate(save_path=data_path)

        # 数据统计
        categories = self.raw_df['category'].dropna().unique()
        stats = [
            f"数据集规模: {self.raw_df.shape[0]} 条记录, {self.raw_df.shape[1]} 个字段",
            f"用户数量: {self.raw_df['user_id'].nunique()}",
            f"商品类别: {', '.join(str(c) for c in categories)}",
            f"时间范围: {self.raw_df['timestamp'].min()} 至 {self.raw_df['timestamp'].max()}",
            ""
        ]

        behavior_stats = self.raw_df['behavior_type'].value_counts()
        stats.append("行为分布:")
        for behavior, count in behavior_stats.items():
            stats.append(f"  - {behavior}: {count} ({count/len(self.raw_df)*100:.1f}%)")

        self.add_report_section("数据集概况", stats)

        return self.raw_df

    def step2_preprocess(self):
        """步骤2：数据预处理"""
        print("\n" + "="*60)
        print("步骤2: 数据预处理")
        print("="*60)

        preprocessor = DataPreprocessor(self.raw_df)
        self.processed_df, self.rfm_df = preprocessor.run_preprocessing()

        # 预处理报告
        missing_count = self.raw_df.isnull().sum().sum()
        rfm_stats = [
            f"原始数据缺失值: {missing_count} 个",
            f"处理后数据量: {len(self.processed_df)} 条",
            f"RFM特征用户数: {len(self.rfm_df)} 人",
            "",
            "RFM特征统计:",
            f"  - Recency (最近购买天数): {self.rfm_df['recency'].mean():.1f} ± {self.rfm_df['recency'].std():.1f}",
            f"  - Frequency (购买频次): {self.rfm_df['frequency'].mean():.1f} ± {self.rfm_df['frequency'].std():.1f}",
            f"  - Monetary (消费金额): ¥{self.rfm_df['monetary'].mean():.2f} ± {self.rfm_df['monetary'].std():.2f}",
            "",
            "用户分群统计:"
        ]

        segment_stats = self.rfm_df['segment'].value_counts()
        for segment, count in segment_stats.items():
            rfm_stats.append(f"  - {segment}: {count} 人 ({count/len(self.rfm_df)*100:.1f}%)")

        self.add_report_section("RFM特征与用户分群", rfm_stats)

        return self.processed_df, self.rfm_df

    def step3_modeling(self):
        """步骤3：建模分析"""
        print("\n" + "="*60)
        print("步骤3: 复购预测建模")
        print("="*60)

        # 准备建模数据
        preprocessor = DataPreprocessor(self.raw_df)
        preprocessor.processed_df = self.processed_df
        preprocessor.rfm_df = self.rfm_df
        X, y, feature_names = preprocessor.prepare_modeling_data()

        # 训练模型
        predictor = RepurchasePredictor(X, y, feature_names)
        self.model_results = predictor.run_modeling()

        # 获取最佳模型和特征重要性
        best_model = predictor.get_best_model()
        self.feature_importance = predictor.get_feature_importance('random_forest')

        # 建模报告
        model_report = [
            "模型性能对比:",
            ""
        ]

        for model_name, metrics in self.model_results.items():
            model_report.append(f"【{model_name}】")
            model_report.append(f"  - 准确率 (Accuracy): {metrics['accuracy']:.4f}")
            model_report.append(f"  - 精确率 (Precision): {metrics['precision']:.4f}")
            model_report.append(f"  - 召回率 (Recall): {metrics['recall']:.4f}")
            model_report.append(f"  - F1分数: {metrics['f1']:.4f}")
            model_report.append(f"  - AUC: {metrics['auc']:.4f}")
            model_report.append("")

        model_report.extend([
            f"最佳模型: {best_model[0]}",
            "",
            "特征重要性排名 (Random Forest):"
        ])

        for idx, row in self.feature_importance.iterrows():
            model_report.append(f"  {idx+1}. {row['feature']}: {row['importance']:.4f}")

        self.add_report_section("复购预测模型分析", model_report)

        return self.model_results

    def step4_visualization(self):
        """步骤4：可视化"""
        print("\n" + "="*60)
        print("步骤4: 生成可视化图表")
        print("="*60)

        visualizer = ECommerceVisualizer(
            self.processed_df,
            self.rfm_df,
            output_dir=self.output_dir
        )

        visualizer.generate_all_plots(feature_importance=self.feature_importance)

        # 可视化报告
        viz_report = [
            "已生成以下高清PNG报表:",
            "",
            f"1. {self.output_dir}/01_conversion_funnel.png - 用户行为转化漏斗图",
            f"2. {self.output_dir}/02_rfm_clustering.png - RFM用户价值聚类散点图",
            f"3. {self.output_dir}/03_repurchase_trend.png - 复购率趋势图",
            f"4. {self.output_dir}/04_feature_importance.png - 特征重要性图",
            "",
            "图表说明:",
            "  - 转化漏斗图展示了从浏览到购买的各阶段转化情况",
            "  - RFM聚类图展示了用户价值分群和RFM特征分布",
            "  - 复购率趋势图展示了月度复购率变化趋势",
            "  - 特征重要性图展示了影响复购的关键特征"
        ]

        self.add_report_section("数据可视化", viz_report)

    def generate_report(self):
        """生成分析报告"""
        print("\n" + "="*60)
        print("生成分析报告")
        print("="*60)

        # 报告头部
        header = [
            "="*60,
            "        电商用户行为数据分析报告",
            "="*60,
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"分析样本: {self.n_users} 用户, {self.n_records} 行为记录",
            "="*60,
            ""
        ]

        # 报告尾部
        footer = [
            "",
            "="*60,
            "                        报告结束",
            "="*60,
            "",
            "分析总结:",
            "  1. 通过RFM模型对用户进行了有效分群，识别出重要价值客户",
            "  2. 构建了复购预测模型，可用于识别潜在复购用户",
            "  3. 可视化图表清晰展示了用户行为转化和复购趋势",
            "  4. 建议针对高价值用户制定精准营销策略，提升复购率",
            "="*60
        ]

        # 合并报告
        full_report = header + self.report_content + footer

        # 保存报告
        report_path = os.path.join(self.reports_dir, 'analysis_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(full_report))

        print(f"分析报告已保存: {report_path}")
        return report_path

    def run(self):
        """运行完整分析流程"""
        print("\n" + "="*60)
        print("  电商用户行为数据分析系统")
        print("="*60)
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            # 执行各步骤
            self.step1_generate_data()
            self.step2_preprocess()
            self.step3_modeling()
            self.step4_visualization()

            # 生成报告
            report_path = self.generate_report()

            print("\n" + "="*60)
            print("分析流程全部完成!")
            print("="*60)
            print(f"输出文件:")
            print(f"  - 原始数据: {self.data_dir}/ecommerce_behavior.csv")
            print(f"  - 可视化图表: {self.output_dir}/")
            print(f"  - 分析报告: {report_path}")
            print("="*60)

            return True

        except Exception as e:
            print(f"\n错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """主函数"""
    # 创建分析流程实例
    pipeline = ECommerceAnalysisPipeline(
        n_users=5000,
        n_records=50000,
        random_seed=42
    )

    # 运行分析
    success = pipeline.run()

    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
