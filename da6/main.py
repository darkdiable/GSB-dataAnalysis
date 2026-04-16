import os
import sys
from datetime import datetime

from data_generator import EcommerceDataGenerator
from preprocessing import DataPreprocessor, RFMFeatureBuilder, UserBehaviorFeatures
from modeling import RepurchasePredictor, ModelEvaluator
from visualization import (
    FunnelVisualizer, RFMClusterVisualizer, 
    RepurchaseTrendVisualizer, FeatureImportanceVisualizer
)


class EcommerceAnalysisPipeline:
    
    def __init__(self, output_dir='output', reports_dir='reports'):
        self.output_dir = output_dir
        self.reports_dir = reports_dir
        self.report_lines = []
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        
    def log(self, message):
        print(message)
        self.report_lines.append(message)
        
    def generate_report(self, filename='analysis_report.txt'):
        filepath = os.path.join(self.reports_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.report_lines))
        self.log(f"\n分析报告已保存至: {filepath}")
        return filepath
    
    def run(self):
        start_time = datetime.now()
        
        self.log("=" * 80)
        self.log("电商用户行为数据分析报告")
        self.log("=" * 80)
        self.log(f"分析时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.log("")
        
        self.log("=" * 80)
        self.log("第一阶段: 数据生成")
        self.log("=" * 80)
        
        generator = EcommerceDataGenerator(n_users=1000, n_records=50000, seed=42)
        raw_df = generator.generate(inject_issues=True)
        
        data_path = os.path.join('data', 'ecommerce_data.csv')
        os.makedirs('data', exist_ok=True)
        generator.save_data(raw_df, data_path)
        
        self.log(f"原始数据集形状: {raw_df.shape}")
        self.log(f"用户数量: {raw_df['user_id'].nunique()}")
        self.log(f"时间范围: {raw_df['timestamp'].min()} 至 {raw_df['timestamp'].max()}")
        self.log("")
        
        self.log("=" * 80)
        self.log("第二阶段: 数据预处理")
        self.log("=" * 80)
        
        preprocessor = DataPreprocessor(raw_df)
        clean_df = preprocessor.handle_missing_values().handle_outliers().get_processed_data()
        
        self.log(f"预处理后数据量: {len(clean_df)}")
        self.log(f"移除记录数: {len(raw_df) - len(clean_df)}")
        
        for log_entry in preprocessor.preprocessing_log:
            self.log(f"  - {log_entry['step']}: 移除 {log_entry['removed_rows']} 条记录")
        
        behavior_dist = clean_df['behavior_type'].value_counts()
        self.log("\n行为类型分布:")
        for behavior, count in behavior_dist.items():
            self.log(f"  - {behavior}: {count} ({count/len(clean_df)*100:.2f}%)")
        self.log("")
        
        self.log("=" * 80)
        self.log("第三阶段: RFM特征构建")
        self.log("=" * 80)
        
        rfm_builder = RFMFeatureBuilder(clean_df)
        rfm_df = rfm_builder.calculate_rfm_scores()
        
        segment_dist = rfm_df['customer_segment'].value_counts()
        self.log("\n客户价值分层分布:")
        for segment, count in segment_dist.items():
            self.log(f"  - {segment}: {count} ({count/len(rfm_df)*100:.2f}%)")
        
        self.log("\nRFM特征统计:")
        self.log(f"  - Recency: 均值={rfm_df['recency'].mean():.2f}, 中位数={rfm_df['recency'].median():.2f}")
        self.log(f"  - Frequency: 均值={rfm_df['frequency'].mean():.2f}, 中位数={rfm_df['frequency'].median():.2f}")
        self.log(f"  - Monetary: 均值={rfm_df['monetary'].mean():.2f}, 中位数={rfm_df['monetary'].median():.2f}")
        self.log("")
        
        self.log("=" * 80)
        self.log("第四阶段: 用户行为特征构建")
        self.log("=" * 80)
        
        feature_builder = UserBehaviorFeatures(clean_df)
        behavior_features = feature_builder.build_features()
        
        self.log(f"特征维度: {behavior_features.shape}")
        self.log(f"特征列表: {list(behavior_features.columns)}")
        self.log("")
        
        self.log("=" * 80)
        self.log("第五阶段: 复购预测模型训练")
        self.log("=" * 80)
        
        predictor = RepurchasePredictor(model_type='random_forest', random_state=42)
        target = predictor.create_target_variable(clean_df)
        
        features = predictor.prepare_features(behavior_features, rfm_df)
        
        data = features.merge(target, on='user_id', how='inner')
        X = data[predictor.feature_names]
        y = data['is_repurchase']
        
        self.log(f"训练样本数: {len(X)}")
        self.log(f"正样本比例: {y.mean()*100:.2f}%")
        
        metrics = predictor.train(X, y)
        
        self.log("\n模型性能指标:")
        self.log(f"  - 准确率 (Accuracy): {metrics['accuracy']:.4f}")
        self.log(f"  - 精确率 (Precision): {metrics['precision']:.4f}")
        self.log(f"  - 召回率 (Recall): {metrics['recall']:.4f}")
        self.log(f"  - F1分数 (F1-Score): {metrics['f1']:.4f}")
        self.log(f"  - AUC-ROC: {metrics['roc_auc']:.4f}")
        
        feature_importance = predictor.get_feature_importance()
        self.log("\n特征重要性排名 (Top 10):")
        for idx, row in feature_importance.head(10).iterrows():
            self.log(f"  {row['feature']}: {row['importance']:.4f}")
        self.log("")
        
        self.log("=" * 80)
        self.log("第六阶段: 可视化分析")
        self.log("=" * 80)
        
        self.log("\n生成转化漏斗图...")
        funnel_viz = FunnelVisualizer(clean_df)
        funnel_path = funnel_viz.plot_funnel(
            save_path=os.path.join(self.output_dir, 'funnel_chart.png')
        )
        self.log(f"  保存路径: {funnel_path}")
        
        funnel_data = funnel_viz.funnel_data
        self.log("\n转化漏斗数据:")
        for _, row in funnel_data.iterrows():
            self.log(f"  - {row['stage']}: {row['count']:,} (转化率: {row['conversion_rate']:.2f}%)")
        
        self.log("\n生成RFM聚类散点图...")
        rfm_viz = RFMClusterVisualizer(rfm_df)
        rfm_scatter_path = rfm_viz.plot_rfm_scatter(
            save_path=os.path.join(self.output_dir, 'rfm_scatter.png')
        )
        self.log(f"  保存路径: {rfm_scatter_path}")
        
        segment_dist_path = rfm_viz.plot_segment_distribution(
            save_path=os.path.join(self.output_dir, 'segment_distribution.png')
        )
        self.log(f"  客户分层分布图: {segment_dist_path}")
        
        self.log("\n生成复购率趋势图...")
        trend_viz = RepurchaseTrendVisualizer(clean_df)
        trend_path = trend_viz.plot_repurchase_trend(
            save_path=os.path.join(self.output_dir, 'repurchase_trend.png')
        )
        self.log(f"  保存路径: {trend_path}")
        
        self.log("\n生成特征重要性图...")
        fi_viz = FeatureImportanceVisualizer(feature_importance)
        fi_path = fi_viz.plot_feature_importance(
            save_path=os.path.join(self.output_dir, 'feature_importance.png')
        )
        self.log(f"  保存路径: {fi_path}")
        self.log("")
        
        self.log("=" * 80)
        self.log("第七阶段: 分析总结")
        self.log("=" * 80)
        
        self.log("\n关键发现:")
        self.log(f"  1. 用户行为转化率: 浏览→购买转化率为 {funnel_data.iloc[-1]['conversion_rate']:.2f}%")
        self.log(f"  2. 高价值客户占比: {segment_dist.get('高价值客户', 0)/len(rfm_df)*100:.2f}%")
        self.log(f"  3. 复购预测模型准确率: {metrics['accuracy']*100:.2f}%")
        self.log(f"  4. 最重要特征: {feature_importance.iloc[0]['feature']} (重要性: {feature_importance.iloc[0]['importance']:.4f})")
        
        self.log("\n业务建议:")
        self.log("  1. 针对低价值客户群体,可推送优惠券提升活跃度")
        self.log("  2. 优化加购到购买的转化路径,提升整体转化率")
        self.log("  3. 重点关注高RFM评分用户,提供VIP服务")
        self.log("  4. 根据特征重要性,优先优化关键行为指标")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        self.log("")
        self.log("=" * 80)
        self.log(f"分析完成! 总耗时: {duration:.2f} 秒")
        self.log("=" * 80)
        
        report_path = self.generate_report()
        
        return {
            'raw_data': raw_df,
            'clean_data': clean_df,
            'rfm_data': rfm_df,
            'behavior_features': behavior_features,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'report_path': report_path
        }


def main():
    pipeline = EcommerceAnalysisPipeline(output_dir='output', reports_dir='reports')
    results = pipeline.run()
    
    print("\n" + "=" * 80)
    print("分析完成! 生成的文件:")
    print("=" * 80)
    print(f"  - 数据文件: data/ecommerce_data.csv")
    print(f"  - 分析报告: {results['report_path']}")
    print(f"  - 漏斗图: output/funnel_chart.png")
    print(f"  - RFM散点图: output/rfm_scatter.png")
    print(f"  - 客户分层图: output/segment_distribution.png")
    print(f"  - 复购趋势图: output/repurchase_trend.png")
    print(f"  - 特征重要性图: output/feature_importance.png")
    print("=" * 80)


if __name__ == '__main__':
    main()
