import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False


class ECommerceVisualizer:
    """电商数据可视化类"""

    def __init__(self, raw_df, rfm_df, output_dir='../output'):
        self.raw_df = raw_df
        self.rfm_df = rfm_df
        self.output_dir = output_dir
        self.plots = {}

        # 设置样式
        sns.set_style("whitegrid")
        plt.style.use('seaborn-v0_8-whitegrid')

    def plot_conversion_funnel(self, save_path=None):
        """绘制用户行为转化漏斗图"""
        # 统计各行为数量
        behavior_counts = self.raw_df['behavior_type'].value_counts()
        behavior_order = ['浏览', '收藏', '加购', '购买']
        counts = [behavior_counts.get(b, 0) for b in behavior_order]

        # 计算转化率
        conversion_rates = [100]  # 浏览为100%
        for i in range(1, len(counts)):
            rate = (counts[i] / counts[0]) * 100
            conversion_rates.append(rate)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 左图：漏斗图
        colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
        bars = ax1.barh(behavior_order, counts, color=colors, edgecolor='white', linewidth=2)

        # 添加数值标签
        for i, (bar, count, rate) in enumerate(zip(bars, counts, conversion_rates)):
            width = bar.get_width()
            ax1.text(width + max(counts)*0.01, bar.get_y() + bar.get_height()/2,
                    f'{count:,} ({rate:.1f}%)',
                    ha='left', va='center', fontsize=11, fontweight='bold')

        ax1.set_xlabel('用户数量', fontsize=12)
        ax1.set_title('用户行为转化漏斗', fontsize=14, fontweight='bold', pad=20)
        ax1.set_xlim(0, max(counts) * 1.3)

        # 右图：阶段转化率
        stage_names = ['浏览→收藏', '收藏→加购', '加购→购买']
        stage_rates = []
        for i in range(len(counts)-1):
            if counts[i] > 0:
                stage_rates.append((counts[i+1] / counts[i]) * 100)
            else:
                stage_rates.append(0)

        bars2 = ax2.bar(stage_names, stage_rates, color=['#9b59b6', '#1abc9c', '#e67e22'],
                       edgecolor='white', linewidth=2)

        for bar, rate in zip(bars2, stage_rates):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(stage_rates)*0.01,
                    f'{rate:.2f}%',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax2.set_ylabel('转化率 (%)', fontsize=12)
        ax2.set_title('各阶段转化率', fontsize=14, fontweight='bold', pad=20)
        ax2.set_ylim(0, max(stage_rates) * 1.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"转化漏斗图已保存: {save_path}")

        self.plots['conversion_funnel'] = fig
        return fig

    def plot_rfm_scatter(self, save_path=None):
        """绘制RFM用户价值聚类散点图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        # 1. R vs F 散点图（按M着色）
        ax1 = axes[0, 0]
        scatter1 = ax1.scatter(self.rfm_df['recency'], self.rfm_df['frequency'],
                              c=self.rfm_df['monetary'], cmap='viridis',
                              alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
        ax1.set_xlabel('Recency (最近购买天数)', fontsize=11)
        ax1.set_ylabel('Frequency (购买频次)', fontsize=11)
        ax1.set_title('RFM分析: Recency vs Frequency (颜色=Monetary)', fontsize=12, fontweight='bold')
        cbar1 = plt.colorbar(scatter1, ax=ax1)
        cbar1.set_label('Monetary (消费金额)', fontsize=10)

        # 2. F vs M 散点图（按R着色）
        ax2 = axes[0, 1]
        scatter2 = ax2.scatter(self.rfm_df['frequency'], self.rfm_df['monetary'],
                              c=self.rfm_df['recency'], cmap='plasma',
                              alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
        ax2.set_xlabel('Frequency (购买频次)', fontsize=11)
        ax2.set_ylabel('Monetary (消费金额)', fontsize=11)
        ax2.set_title('RFM分析: Frequency vs Monetary (颜色=Recency)', fontsize=12, fontweight='bold')
        cbar2 = plt.colorbar(scatter2, ax=ax2)
        cbar2.set_label('Recency (最近购买天数)', fontsize=10)

        # 3. 用户分群分布
        ax3 = axes[1, 0]
        segment_counts = self.rfm_df['segment'].value_counts()
        colors_pie = plt.cm.Set3(np.linspace(0, 1, len(segment_counts)))
        wedges, texts, autotexts = ax3.pie(segment_counts.values, labels=segment_counts.index,
                                          autopct='%1.1f%%', colors=colors_pie,
                                          startangle=90, textprops={'fontsize': 10})
        ax3.set_title('用户价值分群分布', fontsize=12, fontweight='bold')

        # 4. RFM得分分布
        ax4 = axes[1, 1]
        rfm_scores = self.rfm_df['RFM_score'].value_counts().sort_index()
        bars = ax4.bar(rfm_scores.index, rfm_scores.values, color='steelblue', edgecolor='white')
        ax4.set_xlabel('RFM综合得分', fontsize=11)
        ax4.set_ylabel('用户数量', fontsize=11)
        ax4.set_title('RFM综合得分分布', fontsize=12, fontweight='bold')

        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"RFM聚类图已保存: {save_path}")

        self.plots['rfm_scatter'] = fig
        return fig

    def plot_repurchase_trend(self, save_path=None):
        """绘制复购率趋势图"""
        df = self.raw_df.copy()
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        df['month'] = pd.to_datetime(df['timestamp']).dt.to_period('M')

        # 按月统计
        monthly_stats = []
        for month in df['month'].unique():
            month_df = df[df['month'] == month]

            # 该月购买用户
            purchase_users = month_df[month_df['behavior_type'] == '购买']['user_id'].unique()

            # 统计这些用户的复购情况
            repurchase_count = 0
            for user in purchase_users:
                user_purchases = month_df[month_df['user_id'] == user]['behavior_type'].value_counts()
                if '购买' in user_purchases and user_purchases['购买'] >= 2:
                    repurchase_count += 1

            repurchase_rate = repurchase_count / len(purchase_users) if len(purchase_users) > 0 else 0

            monthly_stats.append({
                'month': str(month),
                'total_buyers': len(purchase_users),
                'repurchase_buyers': repurchase_count,
                'repurchase_rate': repurchase_rate * 100
            })

        monthly_df = pd.DataFrame(monthly_stats).sort_values('month')

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # 图1：复购率趋势
        ax1.plot(monthly_df['month'], monthly_df['repurchase_rate'],
                marker='o', linewidth=2.5, markersize=8, color='#e74c3c')
        ax1.fill_between(monthly_df['month'], monthly_df['repurchase_rate'],
                        alpha=0.3, color='#e74c3c')
        ax1.set_xlabel('月份', fontsize=11)
        ax1.set_ylabel('复购率 (%)', fontsize=11)
        ax1.set_title('月度复购率趋势', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)

        # 添加数值标签
        for i, (month, rate) in enumerate(zip(monthly_df['month'], monthly_df['repurchase_rate'])):
            ax1.annotate(f'{rate:.1f}%', (month, rate),
                        textcoords="offset points", xytext=(0,10),
                        ha='center', fontsize=9)

        # 图2：购买用户数与复购用户数
        x = np.arange(len(monthly_df))
        width = 0.35

        bars1 = ax2.bar(x - width/2, monthly_df['total_buyers'], width,
                       label='总购买用户', color='#3498db', edgecolor='white')
        bars2 = ax2.bar(x + width/2, monthly_df['repurchase_buyers'], width,
                       label='复购用户', color='#2ecc71', edgecolor='white')

        ax2.set_xlabel('月份', fontsize=11)
        ax2.set_ylabel('用户数量', fontsize=11)
        ax2.set_title('月度购买用户与复购用户数量', fontsize=13, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(monthly_df['month'], rotation=45)
        ax2.legend()

        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=8)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"复购率趋势图已保存: {save_path}")

        self.plots['repurchase_trend'] = fig
        return fig

    def plot_feature_importance(self, feature_importance, save_path=None):
        """绘制特征重要性图"""
        fig, ax = plt.subplots(figsize=(10, 6))

        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(feature_importance)))
        bars = ax.barh(feature_importance['feature'], feature_importance['importance'],
                      color=colors, edgecolor='white', linewidth=1)

        ax.set_xlabel('重要性', fontsize=11)
        ax.set_title('复购预测模型 - 特征重要性', fontsize=13, fontweight='bold')
        ax.invert_yaxis()

        # 添加数值标签
        for bar, importance in zip(bars, feature_importance['importance']):
            width = bar.get_width()
            ax.text(width + max(feature_importance['importance'])*0.01,
                   bar.get_y() + bar.get_height()/2,
                   f'{importance:.4f}',
                   ha='left', va='center', fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"特征重要性图已保存: {save_path}")

        self.plots['feature_importance'] = fig
        return fig

    def generate_all_plots(self, feature_importance=None):
        """生成所有可视化图表"""
        print("=" * 50)
        print("开始生成可视化图表...")
        print("=" * 50)

        # 1. 转化漏斗图
        self.plot_conversion_funnel(f'{self.output_dir}/01_conversion_funnel.png')

        # 2. RFM聚类图
        self.plot_rfm_scatter(f'{self.output_dir}/02_rfm_clustering.png')

        # 3. 复购率趋势图
        self.plot_repurchase_trend(f'{self.output_dir}/03_repurchase_trend.png')

        # 4. 特征重要性图
        if feature_importance is not None:
            self.plot_feature_importance(feature_importance,
                                        f'{self.output_dir}/04_feature_importance.png')

        print("\n所有图表生成完成!")
        return self.plots


if __name__ == '__main__':
    # 测试代码
    from data_generator import ECommerceDataGenerator
    from data_preprocessor import DataPreprocessor

    generator = ECommerceDataGenerator(n_users=1000, n_records=5000)
    df = generator.generate()

    preprocessor = DataPreprocessor(df)
    processed_df, rfm_df = preprocessor.run_preprocessing()

    visualizer = ECommerceVisualizer(processed_df, rfm_df, output_dir='../output')
    visualizer.generate_all_plots()
