import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_generator import ECommerceDataGenerator
from data_preprocessor import DataPreprocessor
from modeling import RepurchasePredictor
from visualization import Visualizer


def generate_report(df, rfm_df, metrics, feature_importance, report_path):
    print("\n=== 生成分析报告 ===")
    
    report_content = """
================================================================================
                        电商用户行为数据分析报告
================================================================================

生成时间: {}

================================================================================
                            一、数据概览
================================================================================

数据规模:
  - 总记录数: {:,} 条
  - 用户总数: {:,} 人
  - 时间跨度: {} 至 {}
  - 商品类别数: {} 类

行为类型分布:
{}

================================================================================
                            二、RFM用户分层分析
================================================================================

RFM用户分群统计:
{}

各分群用户占比:
{}

RFM特征描述性统计:
{}

================================================================================
                            三、复购预测模型
================================================================================

模型类型: Random Forest 随机森林

模型评估指标:
  - 准确率:   {:.4f}
  - 精确率:   {:.4f}
  - 召回率:   {:.4f}
  - F1分数:   {:.4f}
  - AUC:      {:.4f}

特征重要性排名:
{}

================================================================================
                            四、业务洞察与建议
================================================================================

1. 用户价值分层:
   - 重要价值用户: 需要重点维护，提供VIP专属服务
   - 新用户: 需要引导转化，优化首购体验
   - 重要召回用户: 需要制定召回策略，推送优惠券
   - 流失用户: 需要分析流失原因，尝试挽回
   - 一般用户: 需要精细化运营，提升活跃度

2. 转化优化建议:
   - 根据转化漏斗分析，重点优化从浏览到加购的转化路径
   - 加购后推送限时优惠，提升购买转化率

3. 复购提升策略:
   - 根据模型特征重要性，{}是影响复购的最关键因素
   - 建立用户成长体系，提升用户粘性

================================================================================
                                报告结束
================================================================================
""".format(
        df['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S'),
        len(df),
        df['user_id'].nunique(),
        df['timestamp'].min().strftime('%Y-%m-%d'),
        df['timestamp'].max().strftime('%Y-%m-%d'),
        df['category'].nunique(),
        df['behavior_type'].value_counts().to_string(),
        rfm_df['User_Segment'].value_counts().to_string(),
        (rfm_df['User_Segment'].value_counts(normalize=True) * 100).round(2).astype(str) + '%',
        rfm_df[['Recency', 'Frequency', 'Monetary']].describe().round(2).to_string(),
        metrics['accuracy'],
        metrics['precision'],
        metrics['recall'],
        metrics['f1'],
        metrics['auc'],
        feature_importance.to_string(index=False),
        feature_importance.iloc[0]['feature']
    )
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"分析报告已保存: {report_path}")
    print("\n报告预览:")
    print("=" * 80)
    print(report_content[:2000] + "...")


def main():
    print("=" * 80)
    print("          电商用户行为数据分析系统 - 模块化工程")
    print("=" * 80)
    
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, 'data', 'ecommerce_behavior.csv')
    report_path = os.path.join(base_dir, 'output', 'reports', 'analysis_report.txt')
    figures_dir = os.path.join(base_dir, 'output', 'figures')
    
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    print("\n【步骤1/5】生成模拟电商用户行为数据...")
    generator = ECommerceDataGenerator(n_users=5000, n_records=50000)
    df = generator.generate_data()
    generator.save_data(df, filepath=data_path)
    
    print("\n【步骤2/5】数据预处理与特征工程...")
    preprocessor = DataPreprocessor()
    df, rfm_df = preprocessor.preprocess(data_path)
    
    print("\n【步骤3/5】构建复购预测模型...")
    predictor = RepurchasePredictor(model_type='random_forest')
    model, feature_importance, metrics = predictor.run(rfm_df)
    
    print("\n【步骤4/5】生成可视化图表...")
    visualizer = Visualizer(output_dir=figures_dir)
    visualizer.generate_all_charts(df, rfm_df, feature_importance)
    
    print("\n【步骤5/5】生成分析报告...")
    generate_report(df, rfm_df, metrics, feature_importance, report_path)
    
    print("\n" + "=" * 80)
    print("                   数据分析完成！")
    print("=" * 80)
    print("\n输出文件目录:")
    print(f"  - 数据集: {data_path}")
    print(f"  - 报告: {report_path}")
    print(f"  - 图表: {figures_dir}/")
    print("\n请查看 output 目录获取完整结果。")


if __name__ == '__main__':
    main()
